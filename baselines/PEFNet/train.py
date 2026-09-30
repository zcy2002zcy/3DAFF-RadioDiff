"""Train the adapted two-stage PEFNet baseline on local path-loss maps."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import PathlossDataset
from model import AdaptedPEFNet


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=r"D:\datas\processed_pl_aff_dataset")
    parser.add_argument("--building-dir", default=r"D:\zdata\Building_Infomation\buildings")
    parser.add_argument("--output-dir", default="runs/pefnet")
    parser.add_argument("--stage", choices=("physics", "refine", "joint"), default="physics")
    parser.add_argument("--init-checkpoint", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--sampling-rate", type=float, default=0.10)
    parser.add_argument("--sampling-mode", choices=("random", "uniform"), default="random")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--max-height", type=float, default=20.0)
    parser.add_argument("--max-building-height", type=float, default=20.0)
    parser.add_argument("--frequency-mhz", type=float, default=3500.0)
    parser.add_argument("--pixel-size-m", type=float, default=1.0)
    parser.add_argument("--contrast-scale", type=float, default=1.0)
    parser.add_argument("--vie-coupling", type=float, default=0.25)
    parser.add_argument("--physics-weight", type=float, default=1e-3)
    parser.add_argument("--coarse-weight", type=float, default=0.25)
    parser.add_argument("--observation-weight", type=float, default=1.0)
    parser.add_argument("--freeze-physics", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--val-limit", type=int, default=None)
    parser.add_argument("--test-limit", type=int, default=None)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--pathloss-min-db", type=float, default=None)
    parser.add_argument("--pathloss-max-db", type=float, default=None)
    return parser.parse_args()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def loader(args: argparse.Namespace, split: str, limit: int | None, shuffle: bool) -> DataLoader:
    dataset = PathlossDataset(
        root=args.data_root,
        split=split,
        building_dir=args.building_dir,
        sampling_rate=args.sampling_rate,
        sampling_mode=args.sampling_mode,
        max_height=args.max_height,
        max_building_height=args.max_building_height,
        subset_seed=args.seed,
        limit=limit,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        persistent_workers=args.num_workers > 0,
    )


def to_device(batch: Dict, device: torch.device) -> Dict:
    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value for key, value in batch.items()}


def losses(model: AdaptedPEFNet, outputs: Dict, batch: Dict, args: argparse.Namespace) -> Dict[str, torch.Tensor]:
    target = batch["target"]
    coarse_loss = F.mse_loss(outputs["coarse"], target)
    final_loss = F.mse_loss(outputs["prediction"], target)
    observed_count = batch["mask"].sum().clamp_min(1.0)
    observation_loss = (((outputs["refined_raw"] - batch["sparse"]) * batch["mask"]).square().sum() / observed_count)
    physics_loss = model.vie.loss(outputs["field"], outputs["incident"], batch["building"])

    if args.stage == "physics":
        total = coarse_loss + args.physics_weight * physics_loss
    elif args.stage == "refine":
        total = final_loss + args.observation_weight * observation_loss
    else:
        total = (
            final_loss
            + args.coarse_weight * coarse_loss
            + args.physics_weight * physics_loss
            + args.observation_weight * observation_loss
        )
    return {
        "total": total,
        "final": final_loss,
        "coarse": coarse_loss,
        "physics": physics_loss,
        "observation": observation_loss,
    }


@torch.no_grad()
def evaluate(
    model: AdaptedPEFNet,
    data: DataLoader,
    device: torch.device,
    db_range: float | None,
    stage: str,
) -> Dict[str, float]:
    model.eval()
    sq = absolute = energy = ssim_sum = 0.0
    pixels = samples = 0
    for batch in tqdm(data, desc=f"eval:{data.dataset.split}", leave=False):
        batch = to_device(batch, device)
        outputs = model(batch)
        prediction_key = "coarse" if stage == "physics" else "prediction"
        prediction = outputs[prediction_key].clamp(0.0, 1.0)
        error = prediction - batch["target"]
        sq += float(error.square().sum())
        absolute += float(error.abs().sum())
        energy += float(batch["target"].square().sum())
        pixels += error.numel()
        for pred, target in zip(prediction.cpu().numpy(), batch["target"].cpu().numpy()):
            ssim_sum += structural_similarity(target[0], pred[0], data_range=1.0)
            samples += 1
    mse = sq / max(pixels, 1)
    result = {
        "mse": mse,
        "rmse": math.sqrt(mse),
        "mae": absolute / max(pixels, 1),
        "nmse": sq / max(energy, 1e-12),
        "psnr": -10.0 * math.log10(max(mse, 1e-12)),
        "ssim": ssim_sum / max(samples, 1),
    }
    if db_range is not None:
        result["rmse_db"] = result["rmse"] * db_range
        result["mae_db"] = result["mae"] * db_range
    return result


def load_weights(model: AdaptedPEFNet, path: str, device: torch.device, strict: bool = False) -> dict:
    payload = torch.load(path, map_location=device, weights_only=False)
    state = payload.get("model", payload)
    message = model.load_state_dict(state, strict=strict)
    print(f"Loaded {path}; missing={len(message.missing_keys)}, unexpected={len(message.unexpected_keys)}")
    return payload if isinstance(payload, dict) else {}


def save(path: Path, model: AdaptedPEFNet, optimizer: torch.optim.Optimizer, epoch: int, best: float, args: argparse.Namespace) -> None:
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, "best_rmse": best, "args": vars(args)}, path)


def main() -> None:
    args = arguments()
    seed_all(args.seed)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    model = AdaptedPEFNet(
        image_size=args.image_size,
        base_channels=args.base_channels,
        frequency_mhz=args.frequency_mhz,
        pixel_size_m=args.pixel_size_m,
        contrast_scale=args.contrast_scale,
        coupling=args.vie_coupling,
    ).to(device)
    if args.init_checkpoint:
        load_weights(model, args.init_checkpoint, device)
    if args.stage == "physics":
        model.freeze_refiner()
    elif args.stage == "refine" and args.freeze_physics:
        model.freeze_physics()
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    start_epoch = 0
    best_rmse = math.inf
    if args.resume:
        payload = load_weights(model, args.resume, device, strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        start_epoch = int(payload.get("epoch", -1)) + 1
        best_rmse = float(payload.get("best_rmse", math.inf))

    db_range = None
    if args.pathloss_min_db is not None and args.pathloss_max_db is not None:
        db_range = abs(args.pathloss_max_db - args.pathloss_min_db)
    val_data = loader(args, "val", args.val_limit, False)
    print(f"device={device}, stage={args.stage}, val_samples={len(val_data.dataset)}")
    if args.eval_only:
        print(json.dumps(evaluate(model, val_data, device, db_range, args.stage), indent=2))
        return

    train_data = loader(args, "train", args.train_limit, True)
    print(f"train_samples={len(train_data.dataset)}")
    amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        totals = {name: 0.0 for name in ("total", "final", "coarse", "physics", "observation")}
        seen = 0
        progress = tqdm(train_data, desc=f"{args.stage}:{epoch + 1}/{args.epochs}")
        for batch in progress:
            batch = to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp):
                outputs = model(batch)
                loss_dict = losses(model, outputs, batch, args)
            scaler.scale(loss_dict["total"]).backward()
            scaler.step(optimizer)
            scaler.update()
            batch_size = batch["target"].shape[0]
            seen += batch_size
            for name, value in loss_dict.items():
                totals[name] += float(value.detach()) * batch_size
            progress.set_postfix(loss=f"{totals['total'] / seen:.5f}")

        metrics = evaluate(model, val_data, device, db_range, args.stage)
        record = {"epoch": epoch + 1, **{f"train_{k}": v / seen for k, v in totals.items()}, **metrics}
        with (output / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps(record, indent=2))
        save(output / "last.pt", model, optimizer, epoch, best_rmse, args)
        if metrics["rmse"] < best_rmse:
            best_rmse = metrics["rmse"]
            save(output / "best.pt", model, optimizer, epoch, best_rmse, args)

    test_data = loader(args, "test", args.test_limit, False)
    payload = torch.load(output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(payload["model"])
    test_metrics = evaluate(model, test_data, device, db_range, args.stage)
    (output / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")
    print("test:", json.dumps(test_metrics, indent=2))


if __name__ == "__main__":
    main()
