import argparse
import csv
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.radiomap_dataset import build_dataloader_from_config
from models.resnet import ResNetBaseline


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_input(batch):
    height, width = batch["q"].shape[-2:]
    altitude = ((batch["z"].view(-1, 1, 1, 1) - 1.0) / 3.0).clamp(0.0, 1.0)
    altitude_map = altitude.expand(-1, -1, height, width)
    return torch.cat(
        [batch["q"], batch["building_mask"], batch["s"], altitude_map],
        dim=1,
    )


def move_batch(batch, device):
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


def compute_metrics(pred, target):
    diff = pred - target
    dims = (1, 2, 3)
    mse_per_sample = torch.mean(diff ** 2, dim=dims)
    mse = mse_per_sample.mean().item()
    rmse = torch.sqrt(mse_per_sample).mean().item()
    mae = torch.mean(torch.abs(diff), dim=dims).mean().item()
    nmse = (
        torch.sum(diff ** 2, dim=dims)
        / (torch.sum(target ** 2, dim=dims) + 1e-8)
    ).mean().item()
    return mse, rmse, mae, nmse


def run_epoch(model, loader, optimizer, device, cfg, train=True, epoch=0, iter_log_path=None):
    model.train(train)
    use_amp = bool(cfg["train"].get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)
    max_steps = int(cfg.get("logging", {}).get("max_train_steps_per_epoch" if train else "max_val_steps_per_epoch", 0) or 0)
    total = min(len(loader), max_steps) if max_steps > 0 else len(loader)
    loss_sum = 0.0
    mse_sum = rmse_sum = mae_sum = nmse_sum = 0.0
    count = 0
    context = torch.enable_grad() if train else torch.no_grad()
    desc = "Train" if train else "Val"
    iter_log_every = max(1, int(cfg.get("logging", {}).get("iter_log_every", 200)))
    start_time = time.time()

    with context:
        for step, batch in enumerate(loader, start=1):
            if max_steps > 0 and step > max_steps:
                break
            batch = move_batch(batch, device)
            x = build_input(batch)
            y = batch["psi"]
            if train:
                optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                pred = model(x)
                loss = torch.mean((pred - y) ** 2)
            if train:
                scaler.scale(loss).backward()
                grad_clip = float(cfg["train"].get("grad_clip", 0.0) or 0.0)
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()

            bs = y.shape[0]
            metric_pred = pred.detach().float().clamp(-1.0, 1.0)
            metric_pred = torch.where(batch["s_mask"] > 0.5, batch["s"], metric_pred)
            mse, rmse, mae, nmse = compute_metrics(metric_pred, y.detach().float())
            loss_sum += float(loss.detach().float().cpu()) * bs
            mse_sum += mse * bs
            rmse_sum += rmse * bs
            mae_sum += mae * bs
            nmse_sum += nmse * bs
            count += bs

            if train and step % iter_log_every == 0:
                print(
                    f"[Train] Epoch {epoch:03d} | Step {step:05d}/{total:05d} "
                    f"| Loss {loss_sum / max(count, 1):.6f} "
                    f"| NMSE {nmse_sum / max(count, 1):.6f} "
                    f"| Time {time.time() - start_time:.1f}s"
                )

            if train and iter_log_path is not None and step % iter_log_every == 0:
                exists = iter_log_path.exists()
                with iter_log_path.open("a", newline="", encoding="utf-8-sig") as f:
                    writer = csv.DictWriter(f, fieldnames=["epoch", "step", "loss"])
                    if not exists:
                        writer.writeheader()
                    writer.writerow({"epoch": epoch, "step": step, "loss": float(loss.detach().float().cpu())})

    return {
        "loss": loss_sum / max(count, 1),
        "mse": mse_sum / max(count, 1),
        "rmse": rmse_sum / max(count, 1),
        "mae": mae_sum / max(count, 1),
        "nmse": nmse_sum / max(count, 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg_path = resolve_path(args.config)
    cfg = load_yaml(cfg_path)
    set_seed(int(cfg.get("seed", 42)))

    device = torch.device("cuda" if torch.cuda.is_available() and str(cfg["train"].get("device", "cuda")).lower() == "cuda" else "cpu")
    print("Config:", cfg_path)
    print("Using device:", device)
    if device.type == "cuda":
        print("Visible CUDA devices:", torch.cuda.device_count())
        torch.backends.cudnn.benchmark = True

    model_cfg = cfg.get("model", {})
    model = ResNetBaseline(
        in_channels=int(model_cfg.get("in_channels", 4)),
        out_channels=int(model_cfg.get("out_channels", 1)),
        base_channels=int(model_cfg.get("base_channels", 22)),
        blocks_per_stage=int(model_cfg.get("blocks_per_stage", 2)),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["train"]["lr"]), weight_decay=float(cfg["train"].get("weight_decay", 0.0)))

    train_loader = build_dataloader_from_config(cfg["data"]["train_dir"], cfg, shuffle=True, drop_last=True, split="train")
    val_loader = build_dataloader_from_config(cfg["data"]["val_dir"], cfg, shuffle=False, drop_last=False, split="val")

    ckpt_dir = resolve_path(cfg["checkpoint"]["save_dir"])
    log_dir = resolve_path(cfg["logging"]["log_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    iter_log_path = log_dir / "train_iter_loss.csv"
    if iter_log_path.exists():
        iter_log_path.unlink()

    history = []
    best_val_nmse = float("inf")
    for epoch in range(1, int(cfg["train"]["epochs"]) + 1):
        print("\n" + "=" * 80)
        print(f"ResNet training epoch {epoch}/{cfg['train']['epochs']}")
        train_m = run_epoch(model, train_loader, optimizer, device, cfg, True, epoch, iter_log_path)
        val_m = run_epoch(model, val_loader, optimizer, device, cfg, False, epoch)
        print(f"[Train] loss={train_m['loss']:.6f} nmse={train_m['nmse']:.6f} | [Val] loss={val_m['loss']:.6f} nmse={val_m['nmse']:.6f}")

        ckpt = {"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "config": cfg, "best_val_nmse": best_val_nmse}
        torch.save(ckpt, ckpt_dir / "latest.pt")
        if val_m["nmse"] < best_val_nmse:
            best_val_nmse = val_m["nmse"]
            ckpt["best_val_nmse"] = best_val_nmse
            torch.save(ckpt, ckpt_dir / "best.pt")
            print(">>> Save BEST")

        row = {"epoch": epoch}
        for prefix, metrics in [("train", train_m), ("val", val_m)]:
            for key, value in metrics.items():
                row[f"{prefix}_{key}"] = value
        history.append(row)
        with (log_dir / "train_history.csv").open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)

    print("\nTraining finished.")
    print("Best val NMSE:", best_val_nmse)
    print("Checkpoints:", ckpt_dir)


if __name__ == "__main__":
    main()



