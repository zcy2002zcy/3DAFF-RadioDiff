import argparse
import csv
import sys
import time
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets import RadioDiff3DVolumeDataset
from diffusion import GaussianDiffusion
from utils.common import build_condition, build_model, load_config, move_batch, project_path, seed_everything
from utils.ema import EMA
from utils.metrics import metrics_per_slice


def make_loader(cfg, split: str, shuffle: bool) -> DataLoader:
    data = cfg["data"]
    manifest = project_path(data["manifest_dir"]) / f"{split}.json"
    dataset = RadioDiff3DVolumeDataset(
        str(manifest), building_dir=data["building_dir"],
        image_size=data["image_size"], max_building_height=data["max_building_height"],
        use_sampling=data["use_sampling"], sample_rate=data["sample_rate"],
        deterministic_sampling=(split != "train" and data["deterministic_sampling_for_eval"]),
        seed=cfg["project"]["seed"]
    )
    generator = torch.Generator().manual_seed(cfg["project"]["seed"])
    return DataLoader(
        dataset, batch_size=cfg["train"]["batch_size"], shuffle=shuffle,
        num_workers=data["num_workers"], pin_memory=data["pin_memory"],
        persistent_workers=data["num_workers"] > 0, generator=generator, drop_last=shuffle
    )


def noise_prediction_loss(prediction, noise, cfg):
    name = str(cfg["train"].get("loss", "mse")).lower()
    if name == "mse":
        return F.mse_loss(prediction, noise)
    if name == "l1":
        return F.l1_loss(prediction, noise)
    raise ValueError(f"Unsupported training loss: {name}")


@torch.no_grad()
def validate(model, loader, diffusion, device, amp_enabled, max_steps: int, cfg) -> float:
    model.eval()
    total, count = 0.0, 0
    torch.manual_seed(123456)
    for step, batch in enumerate(tqdm(loader, desc="Val", leave=False)):
        if max_steps > 0 and step >= max_steps:
            break
        batch = move_batch(batch, device)
        target = batch["psi"]
        t = torch.randint(0, diffusion.timesteps, (target.shape[0],), device=device)
        noisy, noise = diffusion.q_sample(target, t)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            prediction = model(noisy, build_condition(batch, cfg), t)
            loss = noise_prediction_loss(prediction, noise, cfg)
        total += float(loss) * target.shape[0]
        count += target.shape[0]
    return total / max(count, 1)


@torch.no_grad()
def validate_reconstruction(model, loader, diffusion, device, cfg, max_volumes: int) -> float:
    """Select checkpoints on fixed validation volumes, never on the test set."""
    model.eval()
    total_nmse, slice_count, volume_count = 0.0, 0, 0
    dcfg = cfg["diffusion"]
    apply_projection = (
        bool(cfg["data"].get("use_sampling", False))
        and str(cfg["eval"].get("measurement_consistency", "none")) == "projection_final"
    )
    progress = tqdm(loader, desc="Val reconstruction", leave=False)
    for batch in progress:
        if max_volumes > 0 and volume_count >= max_volumes:
            break
        batch = move_batch(batch, device)
        original_batch_size = batch["psi"].shape[0]
        if max_volumes > 0:
            keep = min(original_batch_size, max_volumes - volume_count)
            batch = {
                key: value[:keep]
                if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == original_batch_size
                else value
                for key, value in batch.items()
            }
        target = batch["psi"]
        noises = []
        for offset in range(target.shape[0]):
            generator = torch.Generator(device=device).manual_seed(
                int(cfg["project"]["seed"]) + volume_count + offset
            )
            noises.append(
                torch.randn(target[offset:offset + 1].shape, device=device, generator=generator)
            )
        initial_noise = torch.cat(noises, dim=0)
        prediction = diffusion.ddim_sample(
            model=model,
            condition=build_condition(batch, cfg),
            shape=target.shape,
            steps=dcfg["ddim_steps"],
            eta=dcfg["ddim_eta"],
            clip_denoised=dcfg["clip_denoised"],
            initial_noise=initial_noise,
        )
        if apply_projection:
            prediction = torch.where(batch["s_mask"] > 0.5, batch["s"], prediction)
        values = metrics_per_slice(prediction, target)["nmse"]
        total_nmse += float(values.sum())
        slice_count += values.numel()
        volume_count += target.shape[0]
    return total_nmse / max(slice_count, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg["project"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = bool(cfg["train"]["amp"] and device.type == "cuda")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    print(f"Using device: {device}")

    train_loader = make_loader(cfg, "train", True)
    val_loader = make_loader(cfg, "val", False)
    model = build_model(cfg).to(device)
    dcfg = cfg["diffusion"]
    diffusion = GaussianDiffusion(
        timesteps=dcfg["timesteps"], schedule=dcfg["beta_schedule"],
        beta_start=dcfg["beta_start"], beta_end=dcfg["beta_end"],
        cosine_s=dcfg.get("cosine_s", 0.008),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["train"]["learning_rate"],
        weight_decay=cfg["train"]["weight_decay"]
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    ema = EMA(model, cfg["train"]["ema_decay"])
    output_dir = project_path(cfg["train"]["output_dir"])
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "training_log.csv"
    start_epoch, global_step, best_val = 1, 0, float("inf")
    latest = checkpoint_dir / "latest.pt"
    validation_pending = checkpoint_dir / "validation_pending.pt"
    resume_validation_epoch = None
    recovered_train_loss = None
    recovered_train_seconds = 0.0
    if args.resume:
        resume_path = validation_pending if validation_pending.exists() else latest
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        state = torch.load(resume_path, map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        ema.load_state_dict(state["ema"])
        global_step = state["global_step"]
        best_val = state["best_val"]
        if state.get("phase") == "validation_pending":
            start_epoch = state["epoch"]
            resume_validation_epoch = start_epoch
            recovered_train_loss = state["train_loss"]
            recovered_train_seconds = state.get("train_seconds", 0.0)
            print(f"Resuming epoch {start_epoch} from validation; training is not repeated.")
        else:
            start_epoch = state["epoch"] + 1

    if not log_path.exists() or not args.resume:
        output_dir.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(
                ["epoch", "train_loss", "val_noise_loss", "val_reconstruction_nmse", "selection_value", "seconds"]
            )

    update_every = int(cfg["train"]["ema_update_every"])
    for epoch in range(start_epoch, cfg["train"]["epochs"] + 1):
        if epoch == resume_validation_epoch:
            train_loss = recovered_train_loss
            train_seconds = recovered_train_seconds
        else:
            model.train()
            total_loss, count, started = 0.0, 0, time.time()
            progress = tqdm(train_loader, desc=f"Train {epoch}/{cfg['train']['epochs']}")
            for step, batch in enumerate(progress):
                if cfg["train"]["max_train_steps_per_epoch"] > 0 and step >= cfg["train"]["max_train_steps_per_epoch"]:
                    break
                batch = move_batch(batch, device)
                target = batch["psi"]
                t = torch.randint(0, diffusion.timesteps, (target.shape[0],), device=device)
                noisy, noise = diffusion.q_sample(target, t)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                    prediction = model(noisy, build_condition(batch, cfg), t)
                    loss = noise_prediction_loss(prediction, noise, cfg)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
                scaler.step(optimizer)
                scaler.update()
                global_step += 1
                if global_step % update_every == 0:
                    ema.update(model)
                total_loss += float(loss.detach()) * target.shape[0]
                count += target.shape[0]
                if global_step % cfg["train"]["log_every"] == 0:
                    progress.set_postfix(loss=f"{total_loss / count:.5f}", step=global_step)

            train_loss = total_loss / max(count, 1)
            train_seconds = time.time() - started
            recovery_state = {
                "phase": "validation_pending", "epoch": epoch,
                "global_step": global_step, "best_val": best_val,
                "train_loss": train_loss, "train_seconds": train_seconds,
                "model": model.state_dict(), "ema": ema.state_dict(),
                "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                "config": cfg,
            }
            torch.save(recovery_state, validation_pending)

        validation_started = time.time()
        current = {
            name: value.detach().clone() for name, value in model.state_dict().items()
        }
        model.load_state_dict(ema.state_dict())
        val_loss = validate(
            model, val_loader, diffusion, device, amp_enabled,
            cfg["train"]["max_val_steps_per_epoch"], cfg
        )
        reconstruction_volumes = int(cfg["train"].get("reconstruction_val_volumes", 0))
        reconstruction_nmse = None
        if reconstruction_volumes > 0:
            reconstruction_nmse = validate_reconstruction(
                model, val_loader, diffusion, device, cfg, reconstruction_volumes
            )
        selection_metric = str(cfg["train"].get("selection_metric", "noise_loss"))
        if selection_metric == "reconstruction_nmse":
            if reconstruction_nmse is None:
                raise ValueError("reconstruction_val_volumes must be positive for reconstruction_nmse selection")
            selection_value = reconstruction_nmse
        elif selection_metric == "noise_loss":
            selection_value = val_loss
        else:
            raise ValueError(f"Unsupported selection_metric: {selection_metric}")
        ema_weights = {name: value.detach().cpu() for name, value in model.state_dict().items()}
        model.load_state_dict(current)
        seconds = train_seconds + (time.time() - validation_started)
        state = {
            "phase": "epoch_complete",
            "epoch": epoch, "global_step": global_step,
            "best_val": min(best_val, selection_value),
            "selection_metric": selection_metric,
            "val_noise_loss": val_loss,
            "val_reconstruction_nmse": reconstruction_nmse,
            "model": model.state_dict(), "ema": ema.state_dict(),
            "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(), "config": cfg
        }
        torch.save(state, latest)
        if selection_value < best_val:
            best_val = selection_value
            state["best_val"] = best_val
            state["model_ema"] = ema_weights
            torch.save(state, checkpoint_dir / "best.pt")
            marker = " BEST"
        else:
            marker = ""
        with log_path.open("a", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(
                [epoch, train_loss, val_loss, reconstruction_nmse, selection_value, seconds]
            )
        validation_pending.unlink(missing_ok=True)
        reconstruction_text = "n/a" if reconstruction_nmse is None else f"{reconstruction_nmse:.6f}"
        print(
            f"Epoch {epoch}: train_loss={train_loss:.6f} val_noise={val_loss:.6f} "
            f"val_reconstruction_nmse={reconstruction_text} time={seconds:.1f}s{marker}"
        )


if __name__ == "__main__":
    main()
