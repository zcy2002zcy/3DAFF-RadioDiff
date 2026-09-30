"""Train the independently adapted PEFNet-inspired VIE-regularized baseline.

This is not an official PEFNet implementation.  It preserves the two-U-Net
and volume-integral-equation (VIE) training pattern described by Jiang et al.,
while making every dataset-specific assumption explicit in the run metadata.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.pefnet_inspired_dataset import PEFNetInspiredDataset
from models.pefnet_inspired_baseline import build_pefnet_inspired
from physics.vie_operator import VIEOperator
from utils.checkpoint import save_checkpoint
from utils.config import load_config
from utils.seed import seed_everything


DOI = "10.1109/TWC.2024.3429196"
SOURCE_PDF_SHA256 = "0DCE0DC9F353DA6657FE6C90F5B126B6A0A1D75422709DF96A5B22AE5598956D"
METHOD_LABEL = "PEFNet-inspired VIE-regularized U-Net (adapted)"
STABILITY_REVISION = "groupnorm_bf16_refinement_fp32_warmup_v2"
_ACTIVE_OUTPUT: Path | None = None
_ACTIVE_CONTEXT: dict[str, Any] = {}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/pefnet_inspired_adapted.yaml")
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--data-seed", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--precision", choices=("fp32", "bf16", "fp16"))
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def _get(mapping: dict[str, Any], section: str, key: str, default: Any) -> Any:
    value = mapping.get(section, {})
    if not isinstance(value, dict):
        raise TypeError(f"Config section '{section}' must be a mapping")
    return value.get(key, default)


def _learning_rate_for_epoch(
    *,
    epoch: int,
    epochs: int,
    base_learning_rate: float,
    minimum_learning_rate: float,
    warmup_epochs: int,
    warmup_start_factor: float,
) -> float:
    """Return the documented linear-warmup/cosine learning rate."""

    if not 1 <= epoch <= epochs:
        raise ValueError(f"epoch must be in [1, {epochs}], got {epoch}")
    if not 0.0 < minimum_learning_rate <= base_learning_rate:
        raise ValueError("minimum_learning_rate must be positive and <= base LR")
    if not 0.0 < warmup_start_factor <= 1.0:
        raise ValueError("warmup_start_factor must be in (0, 1]")
    warmup_epochs = max(0, min(int(warmup_epochs), epochs))
    if warmup_epochs > 0 and epoch <= warmup_epochs:
        if warmup_epochs == 1:
            return base_learning_rate
        fraction = (epoch - 1) / (warmup_epochs - 1)
        factor = warmup_start_factor + (1.0 - warmup_start_factor) * fraction
        return base_learning_rate * factor
    cosine_epochs = max(epochs - warmup_epochs, 1)
    progress = (epoch - warmup_epochs) / cosine_epochs
    return minimum_learning_rate + 0.5 * (
        base_learning_rate - minimum_learning_rate
    ) * (1.0 + math.cos(math.pi * progress))


def _precision_policy(
    precision: str, device: torch.device
) -> tuple[bool, torch.dtype, bool]:
    """Return autocast enabled, autocast dtype, and GradScaler enabled."""

    if precision == "fp32":
        return False, torch.float32, False
    if device.type != "cuda":
        raise RuntimeError(f"precision={precision} requires CUDA")
    if precision == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("precision=bf16 requested but this CUDA GPU lacks BF16 support")
        return True, torch.bfloat16, False
    if precision == "fp16":
        return True, torch.float16, True
    raise ValueError(f"Unsupported precision: {precision}")


def _nonfinite_names(values: dict[str, torch.Tensor]) -> list[str]:
    return [name for name, value in values.items() if not torch.isfinite(value).all()]


def _masked_statistics(
    prediction: torch.Tensor, target: torch.Tensor, support: torch.Tensor
) -> tuple[torch.Tensor, float, float, int]:
    error = prediction[support] - target[support]
    if error.numel() == 0:
        raise ValueError("A training/evaluation batch has no valid receiver pixels")
    # Resolution-invariant normalized L2.  The paper writes ||e||_2/N; RMS is
    # used here so a physics-grid resolution change does not rescale gradients.
    loss = torch.sqrt(error.float().square().mean() + 1e-12)
    return loss, error.float().square().sum().item(), target[support].float().square().sum().item(), error.numel()


def _dataset(
    *, split: str, config: dict[str, Any], data_seed: int, smoke_test: bool, training: bool
) -> PEFNetInspiredDataset:
    maximum = {"train": 4096, "val": 128, "test": 128}[split]
    maximum = int(_get(config, "data", f"{split}_maps", maximum))
    if smoke_test:
        maximum = 4 if split == "train" else 2
    return PEFNetInspiredDataset(
        PROJECT_ROOT / "data" / "processed" / "splits" / f"{split}.csv",
        max_records=maximum,
        selection_strategy="sequential" if smoke_test else str(
            _get(config, "data", "selection_strategy", "balanced_scene")
        ),
        seed=data_seed,
        training=training,
        dimensionless_wavenumber=float(
            _get(config, "physics", "dimensionless_wavenumber", 0.35)
        ),
        grid_spacing=float(_get(config, "physics", "grid_spacing", 1.0)),
        epsilon_air=float(_get(config, "physics", "epsilon_air", 1.0)),
        epsilon_building=float(_get(config, "physics", "epsilon_building", 4.0)),
        conductivity=float(_get(config, "physics", "conductivity_s_per_m", 0.0)),
    )


def _forward_losses(
    *,
    model,
    vie,
    batch,
    device,
    lambda_data: float,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    inputs = batch["input"].to(device, non_blocking=True)
    target = batch["target"].to(device, non_blocking=True)
    valid = batch["valid_mask"].to(device, non_blocking=True).bool()
    observed = batch["observation_mask"].to(device, non_blocking=True).bool()
    support = valid & ~observed
    incident = batch["incident_field"].to(device, non_blocking=True)
    contrast = batch["contrast"].to(device, non_blocking=True)
    with torch.autocast(
        device_type=device.type, enabled=amp_enabled, dtype=amp_dtype
    ):
        outputs = model(inputs)
    if not isinstance(outputs, dict):
        raise TypeError("PEFNet-inspired model must return a dictionary of intermediate fields")
    data_loss, _, _, _ = _masked_statistics(outputs["prediction"], target, support)
    # CUDA FFT support for half precision is size/hardware dependent; compute
    # the physics residual in FP32 while retaining gradients to the U-Net.
    physics_loss = vie.loss(
        outputs["total_field"].float(), incident.float(), contrast.float()
    )
    total_loss = physics_loss + lambda_data * data_loss
    return total_loss, data_loss, physics_loss, outputs


def _validate(
    model, loader, device, amp_enabled: bool, amp_dtype: torch.dtype
) -> dict[str, float]:
    model.eval()
    sse = target_sq = 0.0
    count = 0
    with torch.inference_mode():
        for batch in loader:
            inputs = batch["input"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            support = (
                batch["valid_mask"].to(device, non_blocking=True).bool()
                & ~batch["observation_mask"].to(device, non_blocking=True).bool()
            )
            with torch.autocast(
                device_type=device.type, enabled=amp_enabled, dtype=amp_dtype
            ):
                prediction = model(inputs)["prediction"]
            if not torch.isfinite(prediction).all():
                raise FloatingPointError("Non-finite prediction during validation")
            _, batch_sse, batch_target_sq, batch_count = _masked_statistics(
                prediction, target, support
            )
            sse += batch_sse
            target_sq += batch_target_sq
            count += batch_count
    if count == 0:
        raise ValueError("Validation split contains no valid receiver pixels")
    return {"rmse": math.sqrt(sse / count), "nmse": sse / max(target_sq, 1e-12)}


def main() -> None:
    global _ACTIVE_OUTPUT
    args = arguments()
    config, config_path = load_config(
        PROJECT_ROOT / args.config if not args.config.is_absolute() else args.config
    )
    seed = int(args.seed if args.seed is not None else _get(config, "experiment", "seed", 42))
    data_seed = int(
        args.data_seed if args.data_seed is not None else _get(config, "data", "data_seed", 42)
    )
    epochs = int(args.epochs if args.epochs is not None else _get(config, "training", "epochs", 100))
    batch_size = int(
        args.batch_size if args.batch_size is not None else _get(config, "training", "batch_size", 1)
    )
    learning_rate = float(
        args.learning_rate
        if args.learning_rate is not None
        else _get(config, "training", "learning_rate", 1e-4)
    )
    num_workers = int(
        args.num_workers if args.num_workers is not None else _get(config, "training", "num_workers", 0)
    )
    patience_limit = int(_get(config, "training", "early_stopping_patience", 15))
    lambda_data = float(_get(config, "training", "lambda_data", 1.0))
    max_grad_norm = float(_get(config, "training", "max_grad_norm", 1.0))
    minimum_learning_rate = float(
        _get(config, "training", "minimum_learning_rate", 1e-6)
    )
    warmup_epochs = int(_get(config, "training", "warmup_epochs", 5))
    warmup_start_factor = float(
        _get(config, "training", "warmup_start_factor", 0.1)
    )
    requested_precision = str(
        args.precision
        if args.precision is not None
        else _get(config, "training", "precision", "bf16")
    ).lower()
    max_steps = _get(
        config, "training", "max_steps_per_epoch",
        _get(config, "training", "steps_per_epoch", None),
    )
    max_steps = None if max_steps in (None, 0) else int(max_steps)
    if args.smoke_test:
        epochs, batch_size, num_workers, max_steps = 2, 1, 0, 1
        warmup_epochs = min(warmup_epochs, epochs)
    if not torch.cuda.is_available() and not args.smoke_test:
        raise RuntimeError("Formal PEFNet-inspired training requires a CUDA-enabled PyTorch environment")
    if batch_size < 1 or epochs < 1:
        raise ValueError("batch_size and epochs must be positive")
    seed_everything(seed, deterministic=True)

    experiment_name = str(
        _get(
            config,
            "experiment",
            "name",
            "pefnet_inspired_adapted_stable_v2_random0",
        )
    )
    name = f"{experiment_name}_seed{seed}"
    if args.smoke_test:
        name += "_smoke"
    output = PROJECT_ROOT / "outputs" / name
    _ACTIVE_OUTPUT = output
    _ACTIVE_CONTEXT.update(
        {"seed": seed, "data_seed": data_seed, "experiment": name}
    )
    checkpoint_dir = output / "checkpoints"
    log_dir = output / "logs"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    # Hash the effective merged configuration, including all CLI/runtime
    # overrides, rather than only the child YAML file.
    effective_config = json.loads(json.dumps(config))
    effective_config.setdefault("experiment", {})["seed"] = seed
    effective_config.setdefault("data", {})["data_seed"] = data_seed
    effective_training = effective_config.setdefault("training", {})
    effective_training.update(
        {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "minimum_learning_rate": minimum_learning_rate,
            "warmup_epochs": warmup_epochs,
            "warmup_start_factor": warmup_start_factor,
            "num_workers": num_workers,
            "max_steps_per_epoch": max_steps,
            "precision": "fp32" if args.smoke_test and not torch.cuda.is_available() else requested_precision,
            "deterministic_algorithms": True,
            "smoke_test": bool(args.smoke_test),
        }
    )
    if args.smoke_test:
        effective_config.setdefault("model", {})["base_channels"] = min(
            int(_get(config, "model", "base_channels", 128)), 8
        )
        effective_config.setdefault("physics", {})["physics_grid_size"] = min(
            int(_get(config, "physics", "physics_grid_size", 64)), 16
        )
    (output / "resolved_config.json").write_text(
        json.dumps(effective_config, indent=2), encoding="utf-8"
    )

    source_config_digest = hashlib.sha256(config_path.read_bytes()).hexdigest()
    config_digest = hashlib.sha256(
        json.dumps(
            effective_config, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    metadata = {
        "method_label": METHOD_LABEL,
        "implementation_status": "independent_inspired_adaptation",
        "faithful_reproduction": False,
        "official_code_available": False,
        "external_pretraining": False,
        "source": {
            "authors": "F. Jiang, T. Li, X. Lv, H. Rui, and D. Jin",
            "title": "Physics-Informed Neural Networks for Path Loss Estimation by Solving Electromagnetic Integral Equations",
            "venue": "IEEE Transactions on Wireless Communications 23(10), 15380-15393 (2024)",
            "doi": DOI,
            "local_pdf_sha256": SOURCE_PDF_SHA256,
        },
        "preserved_components": [
            "complex total-field U-Net",
            "VIE residual physics loss",
            "field-amplitude rough path-loss map",
            "second U-Net residual refinement",
            "joint physics-plus-data optimization",
        ],
        "adaptations_and_assumptions": [
            "UrbanRadio3D encoded target; no claim that raw units are dB",
            "scene-disjoint common project split and valid-Rx-only supervision",
            "2.5D receiver-height input added",
            "building occupancy is sliced at receiver height",
            "dimensionless wavenumber because complex material/phase metadata are unavailable",
            "relative permittivity air=1 and building=4; conductivity=0",
            "FFT implementation of the translation-invariant VIE operator",
            "physics residual evaluated on a downsampled grid",
            "learned affine log-amplitude to encoded-path-loss conversion because the paper omits constants",
            "RMS-normalized residuals for resolution-invariant optimization",
            "GroupNorm replaces BatchNorm for the formal batch-size-one protocol",
            "knowledge U-Net uses BF16 autocast while log conversion and refinement remain FP32",
            "five-epoch linear learning-rate warm-up precedes cosine decay",
        ],
        "stability_revision": STABILITY_REVISION,
        "normalization": "group_norm",
        "max_norm_groups": int(_get(config, "model", "max_norm_groups", 32)),
        "force_refinement_fp32": bool(
            _get(config, "model", "force_refinement_fp32", True)
        ),
        "requested_precision": requested_precision,
        "seed": seed,
        "data_seed": data_seed,
        "sample_rate": 0.0,
        "support": "valid_rx_only",
        "config": str(config_path),
        "config_sha256": config_digest,
        "config_hash_scope": "canonical_effective_merged_config_with_cli_overrides",
        "source_config_sha256": source_config_digest,
    }
    (output / "provenance.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    train_set = _dataset(split="train", config=config, data_seed=data_seed, smoke_test=args.smoke_test, training=True)
    val_set = _dataset(split="val", config=config, data_seed=data_seed, smoke_test=args.smoke_test, training=False)
    loader_args = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
    }
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, **loader_args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_channels = int(_get(config, "model", "base_channels", 128))
    if args.smoke_test:
        base_channels = min(base_channels, 8)
    max_norm_groups = int(_get(config, "model", "max_norm_groups", 32))
    force_refinement_fp32 = bool(
        _get(config, "model", "force_refinement_fp32", True)
    )
    model = build_pefnet_inspired(
        in_channels=5,
        base_channels=base_channels,
        max_norm_groups=max_norm_groups,
        force_refinement_fp32=force_refinement_fp32,
    ).to(device)
    physics_grid = int(_get(config, "physics", "physics_grid_size", 64))
    if args.smoke_test:
        physics_grid = min(physics_grid, 16)
    vie = VIEOperator(
        grid_size=physics_grid,
        dimensionless_wavenumber=float(
            _get(config, "physics", "dimensionless_wavenumber", 0.35)
        ),
        grid_spacing=float(_get(config, "physics", "grid_spacing", 1.0)),
        epsilon_air=float(_get(config, "physics", "epsilon_air", 1.0)),
        epsilon_building=float(_get(config, "physics", "epsilon_building", 4.0)),
        conductivity=float(_get(config, "physics", "conductivity_s_per_m", 0.0)),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=float(_get(config, "training", "weight_decay", 1e-6)),
    )
    runtime_precision = (
        "fp32" if args.smoke_test and device.type != "cuda" else requested_precision
    )
    amp_enabled, amp_dtype, scaler_enabled = _precision_policy(
        runtime_precision, device
    )
    scaler = torch.amp.GradScaler(
        device.type,
        enabled=scaler_enabled,
        init_scale=128.0,
        growth_interval=2000,
    )
    metadata["runtime_precision"] = runtime_precision
    metadata["autocast_scope"] = "knowledge_unet_only"
    (output / "provenance.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(
        f"baseline={name} device={device} parameters={sum(p.numel() for p in model.parameters())} "
        f"physics_grid={physics_grid} lambda_data={lambda_data} "
        f"precision={runtime_precision} refinement_fp32={force_refinement_fp32} "
        f"normalization=group_norm"
    )

    fields = (
        "epoch", "train_total_loss", "train_data_loss", "train_physics_loss",
        "val_rmse", "val_nmse", "learning_rate", "max_gradient_norm",
    )
    best = math.inf
    best_epoch = 0
    patience = 0
    stop_reason = "max_epochs"
    epoch = 0
    history_path = log_dir / "train_history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, epochs + 1):
            epoch_learning_rate = _learning_rate_for_epoch(
                epoch=epoch,
                epochs=epochs,
                base_learning_rate=learning_rate,
                minimum_learning_rate=minimum_learning_rate,
                warmup_epochs=warmup_epochs,
                warmup_start_factor=warmup_start_factor,
            )
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] = epoch_learning_rate
            train_set.set_epoch(epoch)
            model.train()
            total_sum = data_sum = physics_sum = 0.0
            max_epoch_gradient_norm = 0.0
            steps = 0
            for batch in train_loader:
                _ACTIVE_CONTEXT.update(
                    {
                        "epoch": epoch,
                        "step": steps + 1,
                        "learning_rate": epoch_learning_rate,
                        "runtime_precision": runtime_precision,
                    }
                )
                optimizer.zero_grad(set_to_none=True)
                total_loss, data_loss, physics_loss, outputs = _forward_losses(
                    model=model, vie=vie, batch=batch, device=device,
                    lambda_data=lambda_data, amp_enabled=amp_enabled,
                    amp_dtype=amp_dtype,
                )
                if not all(torch.isfinite(value) for value in (total_loss, data_loss, physics_loss)):
                    bad_outputs = _nonfinite_names(outputs)
                    raise FloatingPointError(
                        f"Non-finite loss at epoch={epoch}, step={steps + 1}: "
                        f"total={total_loss.item()} data={data_loss.item()} "
                        f"physics={physics_loss.item()} bad_outputs={bad_outputs} "
                        f"lr={epoch_learning_rate} precision={runtime_precision}"
                    )
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_grad_norm,
                    error_if_nonfinite=True,
                )
                max_epoch_gradient_norm = max(
                    max_epoch_gradient_norm, float(gradient_norm.detach().item())
                )
                scaler.step(optimizer)
                scaler.update()
                total_sum += total_loss.item()
                data_sum += data_loss.item()
                physics_sum += physics_loss.item()
                steps += 1
                if max_steps is not None and steps >= max_steps:
                    break
            bad_parameters = _nonfinite_names(dict(model.named_parameters()))
            bad_buffers = _nonfinite_names(dict(model.named_buffers()))
            if bad_parameters or bad_buffers:
                raise FloatingPointError(
                    f"Non-finite model state after epoch={epoch}: "
                    f"parameters={bad_parameters} buffers={bad_buffers}"
                )
            validation = _validate(
                model, val_loader, device, amp_enabled, amp_dtype
            )
            if not all(math.isfinite(value) for value in validation.values()):
                raise FloatingPointError(
                    f"Non-finite validation metrics after epoch={epoch}: {validation}"
                )
            row = {
                "epoch": epoch,
                "train_total_loss": total_sum / max(steps, 1),
                "train_data_loss": data_sum / max(steps, 1),
                "train_physics_loss": physics_sum / max(steps, 1),
                "val_rmse": validation["rmse"],
                "val_nmse": validation["nmse"],
                "learning_rate": epoch_learning_rate,
                "max_gradient_norm": max_epoch_gradient_norm,
            }
            writer.writerow(row)
            handle.flush()
            print(
                f"epoch={epoch:03d} total={row['train_total_loss']:.6f} "
                f"data={row['train_data_loss']:.6f} physics={row['train_physics_loss']:.6f} "
                f"val_rmse={validation['rmse']:.6f} val_nmse={validation['nmse']:.6f} "
                f"grad_norm_max={max_epoch_gradient_norm:.6f} lr={epoch_learning_rate:.8f}"
            )
            improved = validation["nmse"] < best
            if improved:
                best = validation["nmse"]
                best_epoch = epoch
                patience = 0
            else:
                patience += 1
            state = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "grad_scaler": scaler.state_dict(),
                "metadata": metadata,
                "model_config": {
                    "in_channels": 5,
                    "base_channels": base_channels,
                    "normalization": "group_norm",
                    "max_norm_groups": max_norm_groups,
                    "force_refinement_fp32": force_refinement_fp32,
                },
                "physics_config": {
                    "grid_size": physics_grid,
                    "dimensionless_wavenumber": float(
                        _get(config, "physics", "dimensionless_wavenumber", 0.35)
                    ),
                },
                "val_metrics": validation,
                "training_state": {
                    "best_val_nmse": best,
                    "best_epoch": best_epoch,
                    "patience": patience,
                    "learning_rate": epoch_learning_rate,
                    "precision": runtime_precision,
                    "warmup_epochs": warmup_epochs,
                    "warmup_start_factor": warmup_start_factor,
                    "minimum_learning_rate": minimum_learning_rate,
                    "config_sha256": config_digest,
                },
            }
            save_checkpoint(checkpoint_dir / "latest.pt", state)
            if improved:
                save_checkpoint(checkpoint_dir / "best.pt", state)
            if not args.smoke_test and patience >= patience_limit:
                stop_reason = "early_stopping"
                print(f"early_stopping epoch={epoch} best_epoch={best_epoch}")
                break

    completion = {
        "last_epoch": epoch,
        "best_epoch": best_epoch,
        "best_val_nmse": best,
        "stop_reason": stop_reason,
        "early_stopping_patience": patience_limit,
        "device": str(device),
        "runtime_precision": runtime_precision,
        "normalization": "group_norm",
        "force_refinement_fp32": force_refinement_fp32,
        "stability_revision": STABILITY_REVISION,
        "seed": seed,
        "data_seed": data_seed,
        "config_sha256": config_digest,
        "config_hash_scope": "canonical_effective_merged_config_with_cli_overrides",
        "source_config_sha256": source_config_digest,
    }
    (output / "run_complete.json").write_text(json.dumps(completion, indent=2), encoding="utf-8")
    failed_marker = output / "run_failed.json"
    if failed_marker.is_file():
        failed_marker.unlink()
    print(f"finished baseline={name} device={device} best={best:.6f} output={output}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        if _ACTIVE_OUTPUT is not None:
            _ACTIVE_OUTPUT.mkdir(parents=True, exist_ok=True)
            failure = {
                **_ACTIVE_CONTEXT,
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "traceback": traceback.format_exc(),
            }
            (_ACTIVE_OUTPUT / "run_failed.json").write_text(
                json.dumps(failure, indent=2), encoding="utf-8"
            )
        raise
