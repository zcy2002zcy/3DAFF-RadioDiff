import argparse
import csv
import random
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.radiomap_dataset import build_dataloader_from_config
from models.pix2pix import PatchDiscriminator, Pix2PixGenerator


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


def metric_prediction(pred, batch):
    pred = pred.float().clamp(-1.0, 1.0)
    return torch.where(batch["s_mask"] > 0.5, batch["s"], pred)


def move_batch(batch, device):
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


def batch_nmse(pred, target):
    dims = (1, 2, 3)
    values = torch.sum((pred - target) ** 2, dim=dims) / (
        torch.sum(target ** 2, dim=dims) + 1e-8
    )
    return values.mean().item()


@torch.no_grad()
def validate(gen, loader, device):
    gen.eval()
    loss_sum = nmse_sum = 0.0
    count = 0
    for batch in loader:
        batch = move_batch(batch, device)
        x = build_input(batch)
        y = batch["psi"]
        pred = gen(x)
        bs = y.shape[0]
        pred_eval = metric_prediction(pred, batch)
        loss_sum += torch.mean((pred_eval - y) ** 2).item() * bs
        nmse_sum += batch_nmse(pred_eval, y.float()) * bs
        count += bs
    gen.train()
    return {"loss": loss_sum / max(count, 1), "nmse": nmse_sum / max(count, 1)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_yaml(resolve_path(args.config))
    set_seed(int(cfg.get("seed", 42)))
    device = torch.device("cuda" if torch.cuda.is_available() and cfg["train"].get("device", "cuda") == "cuda" else "cpu")
    print("Config:", resolve_path(args.config))
    print("Using device:", device)
    if device.type == "cuda":
        print("Visible CUDA devices:", torch.cuda.device_count())
        torch.backends.cudnn.benchmark = True

    mcfg = cfg["model"]
    gen = Pix2PixGenerator(mcfg["in_channels"], mcfg["out_channels"], mcfg["base_channels"]).to(device)
    disc = PatchDiscriminator(mcfg["in_channels"] + mcfg["out_channels"], mcfg["base_channels"]).to(device)
    opt_g = torch.optim.Adam(gen.parameters(), lr=cfg["train"]["lr"], betas=(cfg["train"].get("beta1", 0.5), cfg["train"].get("beta2", 0.999)))
    opt_d = torch.optim.Adam(disc.parameters(), lr=cfg["train"]["lr"], betas=(cfg["train"].get("beta1", 0.5), cfg["train"].get("beta2", 0.999)))
    adv_loss = nn.BCEWithLogitsLoss()
    l1_loss = nn.L1Loss()
    lambda_l1 = float(mcfg.get("lambda_l1", 100.0))

    train_loader = build_dataloader_from_config(cfg["data"]["train_dir"], cfg, shuffle=True, drop_last=True, split="train")
    val_loader = build_dataloader_from_config(cfg["data"]["val_dir"], cfg, shuffle=False, drop_last=False, split="val")
    ckpt_dir = resolve_path(cfg["checkpoint"]["save_dir"])
    log_dir = resolve_path(cfg["logging"]["log_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_val_nmse = float("inf")
    max_steps = int(cfg["logging"].get("max_train_steps_per_epoch", 0) or 0)
    iter_log_every = max(1, int(cfg["logging"].get("iter_log_every", 200)))

    for epoch in range(1, int(cfg["train"]["epochs"]) + 1):
        print("\n" + "=" * 80)
        print(f"Pix2Pix training epoch {epoch}/{cfg['train']['epochs']}")
        total = min(len(train_loader), max_steps) if max_steps > 0 else len(train_loader)
        g_sum = d_sum = nmse_sum = 0.0
        count = 0
        start_time = time.time()
        for step, batch in enumerate(train_loader, start=1):
            if max_steps > 0 and step > max_steps:
                break
            batch = move_batch(batch, device)
            x = build_input(batch)
            y = batch["psi"]
            fake = gen(x)

            opt_d.zero_grad(set_to_none=True)
            real_logits = disc(x, y)
            fake_logits = disc(x, fake.detach())
            loss_d = 0.5 * (adv_loss(real_logits, torch.ones_like(real_logits)) + adv_loss(fake_logits, torch.zeros_like(fake_logits)))
            loss_d.backward()
            opt_d.step()

            opt_g.zero_grad(set_to_none=True)
            fake_logits_g = disc(x, fake)
            loss_g = adv_loss(fake_logits_g, torch.ones_like(fake_logits_g)) + lambda_l1 * l1_loss(fake, y)
            loss_g.backward()
            grad_clip = float(cfg["train"].get("grad_clip", 0.0) or 0.0)
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(gen.parameters(), grad_clip)
            opt_g.step()

            bs = y.shape[0]
            g_sum += loss_g.item() * bs
            d_sum += loss_d.item() * bs
            nmse_sum += batch_nmse(metric_prediction(fake.detach(), batch), y.float()) * bs
            count += bs
            if step % iter_log_every == 0:
                print(
                    f"[Train] Epoch {epoch:03d} | Step {step:05d}/{total:05d} "
                    f"| G-Loss {g_sum / max(count, 1):.6f} "
                    f"| D-Loss {d_sum / max(count, 1):.6f} "
                    f"| NMSE {nmse_sum / max(count, 1):.6f} "
                    f"| Time {time.time() - start_time:.1f}s"
                )

        train_m = {"g_loss": g_sum / max(count, 1), "d_loss": d_sum / max(count, 1), "nmse": nmse_sum / max(count, 1)}
        val_m = validate(gen, val_loader, device)
        print(f"[Train] g_loss={train_m['g_loss']:.6f} d_loss={train_m['d_loss']:.6f} nmse={train_m['nmse']:.6f} | [Val] loss={val_m['loss']:.6f} nmse={val_m['nmse']:.6f}")
        ckpt = {
            "epoch": epoch,
            "generator_state_dict": gen.state_dict(),
            "discriminator_state_dict": disc.state_dict(),
            "optimizer_g_state_dict": opt_g.state_dict(),
            "optimizer_d_state_dict": opt_d.state_dict(),
            "config": cfg,
            "best_val_nmse": best_val_nmse,
        }
        torch.save(ckpt, ckpt_dir / "latest.pt")
        if val_m["nmse"] < best_val_nmse:
            best_val_nmse = val_m["nmse"]
            ckpt["best_val_nmse"] = best_val_nmse
            torch.save(ckpt, ckpt_dir / "best.pt")
            print(">>> Save BEST")
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_m.items()}, **{f"val_{k}": v for k, v in val_m.items()}}
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
