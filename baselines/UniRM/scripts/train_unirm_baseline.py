"""Train the pinned official UniRM architecture under the no-sampling protocol."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.dense_baseline_dataset import DenseBaselineDataset
from models.unirm_baseline import UPSTREAM_COMMIT, build_unirm_adapted
from utils.checkpoint import save_checkpoint
from utils.seed import seed_everything


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--stage0-epochs", type=int, default=100)
    parser.add_argument("--stage1-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--early-stopping-patience", type=int, default=15)
    parser.add_argument("--skip-completed-stage0", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def masked_statistics(prediction, target, mask):
    error = prediction[mask] - target[mask]
    loss = error.square().mean()
    return loss, error.square().sum().item(), target[mask].square().sum().item(), error.numel()


def run_epoch(model, loader, device, optimizer, gradient_clip, max_steps):
    training = optimizer is not None
    model.train(training)
    loss_sum = sse = target_sq = 0.0
    count = steps = 0
    for batch in loader:
        inputs = batch["input"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        support = batch["valid_mask"].to(device, non_blocking=True).bool()
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            prediction = model(inputs)
            loss, batch_sse, batch_target_sq, batch_count = masked_statistics(
                prediction, target, support
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite UniRM loss at step {steps + 1}")
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                optimizer.step()
        loss_sum += loss.item()
        sse += batch_sse
        target_sq += batch_target_sq
        count += batch_count
        steps += 1
        if max_steps is not None and steps >= max_steps:
            break
    return {
        "loss": loss_sum / max(steps, 1),
        "rmse": math.sqrt(sse / max(count, 1)),
        "nmse": sse / max(target_sq, 1e-12),
    }


def optimizer_for(model, stage, learning_rate):
    if stage == 0:
        return torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-6)
    prompt, relation, backbone = [], [], []
    for name, parameter in model.named_parameters():
        if "prompt" in name or any(key in name for key in ("fusion", "gate", "projects", "norm_")):
            prompt.append(parameter)
        elif "relation_layers" in name:
            relation.append(parameter)
        else:
            backbone.append(parameter)
    return torch.optim.AdamW(
        [
            {"params": backbone, "lr": learning_rate * 0.1},
            {"params": prompt, "lr": learning_rate},
            {"params": relation, "lr": learning_rate * 2.0},
        ],
        weight_decay=1e-6,
    )


def train_stage(
    *, model, stage, epochs, train_loader, val_loader, train_set, device,
    learning_rate, patience_limit, checkpoint_dir, log_dir, smoke_test,
):
    optimizer = optimizer_for(model, stage, learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1), eta_min=1e-6
    )
    best = math.inf
    patience = 0
    prefix = "stage0_" if stage == 0 else ""
    history_path = log_dir / f"{prefix}train_history.csv"
    fields = ("stage", "epoch", "train_loss", "train_rmse", "train_nmse", "val_loss", "val_rmse", "val_nmse", "learning_rate")
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, epochs + 1):
            train_set.set_epoch(epoch)
            train = run_epoch(model, train_loader, device, optimizer, 0.05, 1 if smoke_test else 1000)
            with torch.inference_mode():
                val = run_epoch(model, val_loader, device, None, 0.05, None)
            row = {
                "stage": stage, "epoch": epoch,
                **{f"train_{key}": value for key, value in train.items()},
                **{f"val_{key}": value for key, value in val.items()},
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
            writer.writerow(row)
            handle.flush()
            print(
                f"stage={stage} epoch={epoch:03d} train_rmse={train['rmse']:.6f} "
                f"val_rmse={val['rmse']:.6f} val_nmse={val['nmse']:.6f}"
            )
            state = {
                "stage": stage, "epoch": epoch, "model": model.state_dict(),
                "val_metrics": val, "upstream_commit": UPSTREAM_COMMIT,
            }
            save_checkpoint(checkpoint_dir / f"{prefix}latest.pt", state)
            if val["nmse"] < best:
                best = val["nmse"]
                patience = 0
                save_checkpoint(checkpoint_dir / f"{prefix}best.pt", state)
            else:
                patience += 1
            scheduler.step()
            if not smoke_test and patience >= patience_limit:
                print(f"early_stopping stage={stage} epoch={epoch}")
                break
    return best


def main() -> None:
    args = arguments()
    seed_everything(args.seed)
    if not torch.cuda.is_available() and not args.smoke_test:
        raise RuntimeError("Formal UniRM training requires CUDA")
    name = f"unirm_adapted_random0_seed{args.seed}"
    if args.smoke_test:
        name += "_smoke"
    output = PROJECT_ROOT / "outputs" / name
    checkpoint_dir = output / "checkpoints"
    log_dir = output / "logs"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "method_label": "UniRM (adapted, no-sampling)",
        "upstream": "https://github.com/Shirleyue/UniRM",
        "upstream_commit": UPSTREAM_COMMIT,
        "upstream_license": "not declared at pinned commit",
        "external_pretraining": False,
        "stages": ["same-data backbone pretraining", "same-data prompt tuning"],
        "input_adaptations": [
            "official 128x128 internal resolution",
            "sparse-RSS channel replaced by Tx indicator",
            "hard-coded sparse-RSS identity residual removed",
            "three building occupancy slices at 1/2/4 m",
            "single-band distance prior replaces sparse-RSS-derived AP prior",
        ],
        "seed": args.seed,
        "data_seed": args.data_seed,
        "support": "valid_rx_only",
    }
    (output / "provenance.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    split_root = PROJECT_ROOT / "data" / "processed" / "splits"
    train_set = DenseBaselineDataset(
        split_root / "train.csv", max_records=4 if args.smoke_test else 4096,
        selection_strategy="sequential" if args.smoke_test else "balanced_scene",
        seed=args.data_seed, sample_rate=0.0, training=True,
    )
    val_set = DenseBaselineDataset(
        split_root / "val.csv", max_records=2 if args.smoke_test else 128,
        selection_strategy="sequential" if args.smoke_test else "balanced_scene",
        seed=args.data_seed, sample_rate=0.0, training=False,
    )
    loader_args = dict(
        batch_size=1 if args.smoke_test else args.batch_size,
        num_workers=0 if args.smoke_test else args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, **loader_args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stage0_path = checkpoint_dir / "stage0_best.pt"
    stage0_complete = output / "stage0_complete.json"
    if not (
        args.skip_completed_stage0
        and stage0_path.is_file()
        and stage0_complete.is_file()
    ):
        stage0 = build_unirm_adapted(enable_prompt=False).to(device)
        print(f"stage=0 device={device} parameters={sum(p.numel() for p in stage0.parameters())}")
        stage0_best = train_stage(
            model=stage0, stage=0, epochs=1 if args.smoke_test else args.stage0_epochs,
            train_loader=train_loader, val_loader=val_loader, train_set=train_set,
            device=device, learning_rate=args.learning_rate,
            patience_limit=args.early_stopping_patience,
            checkpoint_dir=checkpoint_dir, log_dir=log_dir, smoke_test=args.smoke_test,
        )
        stage0_complete.write_text(
            json.dumps({"stage": 0, "best": stage0_best, "device": str(device)}, indent=2),
            encoding="utf-8",
        )
        del stage0
        if device.type == "cuda":
            torch.cuda.empty_cache()
    stage0_checkpoint = torch.load(stage0_path, map_location="cpu", weights_only=False)
    stage1 = build_unirm_adapted(enable_prompt=True)
    incompatible = stage1.load_state_dict(stage0_checkpoint["model"], strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError(f"Unexpected stage-0 keys: {incompatible.unexpected_keys}")
    stage1 = stage1.to(device)
    print(
        f"stage=1 initialized_from={stage0_path} device={device} "
        f"parameters={sum(p.numel() for p in stage1.parameters())} "
        f"new_prompt_keys={len(incompatible.missing_keys)}"
    )
    best = train_stage(
        model=stage1, stage=1, epochs=1 if args.smoke_test else args.stage1_epochs,
        train_loader=train_loader, val_loader=val_loader, train_set=train_set,
        device=device, learning_rate=args.learning_rate,
        patience_limit=args.early_stopping_patience,
        checkpoint_dir=checkpoint_dir, log_dir=log_dir, smoke_test=args.smoke_test,
    )
    (output / "run_complete.json").write_text(
        json.dumps({"stage": 1, "best": best, "device": str(device)}, indent=2),
        encoding="utf-8",
    )
    print(f"finished baseline={name} device={device} best={best:.6f} output={output}")


if __name__ == "__main__":
    main()
