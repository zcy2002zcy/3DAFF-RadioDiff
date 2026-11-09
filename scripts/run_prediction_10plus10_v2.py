"""Isolated 10+10 AFF prediction experiment; never modifies baseline recipes.

The first ten epochs retain the existing coarse predictor. The next ten train
the same residual corrector with geometry consistency and high-noise clean
residual supervision. Validation alone locks inference settings before test.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib
import json
import math
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE if (HERE / 'datasets' / 'pl3d_manifest_dataset.py').is_file() else Path(
    'D:/zdata/3DAFF-RadioDiff/3d-radiodiff')
if not (ROOT / 'scripts' / 'run_residual_prediction.py').is_file():
    raise RuntimeError('Place this script inside the project; legacy project root was not found')
sys.path.insert(0, str(ROOT))
from compat_runtime import autocast, load_checkpoint, make_grad_scaler
from scripts.run_residual_prediction import audit, dataset, loader, tensors, metrics, summarize
from models.pl3d_residual_prediction import Predictor, Corrector, schedule, masked_mse
import models

# A staged runner can be checked without changing the live project. Production
# installation puts this module next to the original model; no monkey-patching.
if HERE != ROOT:
    models.__path__.insert(0, str(HERE / 'models'))
improved = importlib.import_module('models.pl3d_residual_prediction_v2')
available_from_condition = improved.available_from_condition
draw_timesteps = improved.draw_timesteps
diffusion_losses = improved.diffusion_losses
sample_residual = improved.sample_residual

VERSION = 'q_building_prediction_10plus10_geometry_highnoise_v2'
CALIBRATION_STEPS = (1, 30)
CALIBRATION_STRENGTHS = (0., .1, .25, .5, .75, 1.)
METRIC_NAMES = ('nmse', 'rmse', 'psnr', 'ssim')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def content_hash(value):
    data = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(data.encode('utf-8')).hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def save(path, payload):
    temporary = path.with_name(path.name + '.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def load(path, device='cpu'):
    return load_checkpoint(path, map_location=device)


def read_json(path):
    # Windows PowerShell 5 writes a UTF-8 BOM in launcher status files.
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def validate_bootstrap_output(output):
    """Permit only the launcher's pre-training metadata in a fresh directory.

    This exception never permits model files, arbitrary subdirectories, links,
    or malformed metadata. An existing protocol still always requires resume.
    """
    if not output.exists():
        return
    allowed = {'main_run.lock', 'launcher_status.json', 'launch_claim.json',
               'training.stdout.log', 'training.stderr.log'}
    for entry in output.iterdir():
        if entry.name not in allowed or not entry.is_file() or entry.is_symlink():
            raise ValueError(f'Unexpected existing output: {entry.name}; choose a NEW output directory')
        if entry.resolve().parent != output.resolve():
            raise ValueError('Output bootstrap files must not point outside the output directory')
        if entry.suffix != '.json':
            continue
        record = read_json(entry)
        if not isinstance(record, dict) or not isinstance(record.get('output'), str):
            raise ValueError(f'Invalid launcher metadata: {entry.name}')
        if Path(record['output']).resolve() != output.resolve():
            raise ValueError(f'Launcher metadata names a different output: {entry.name}')
        if entry.name == 'launch_claim.json':
            if (record.get('state') not in ('launch_requested', 'launched')
                    or record.get('coarse_epochs') != 10 or record.get('diffusion_epochs') != 10):
                raise ValueError('Unsafe or incompatible pre-training launch claim')
        elif record.get('state') != 'running' or record.get('action') not in ('All', 'Train', 'Smoke'):
            raise ValueError('Launcher is not running an authorized fresh training action')


def source_hashes():
    files = {
        'runner': Path(__file__).resolve(),
        'revised_math': Path(improved.__file__).resolve(),
        'legacy_runner_metric_dataset_helpers': ROOT / 'scripts/run_residual_prediction.py',
        'legacy_predictor_corrector': ROOT / 'models/pl3d_residual_prediction.py',
        'aff_architecture': ROOT / 'models/radiodiff_aff_full3d.py',
        'dataset': ROOT / 'datasets/pl3d_manifest_dataset.py',
        'legacy_ssim_and_scene_selection': ROOT / 'scripts/evaluate_pl3d_aff_fullnoise.py',
    }
    return {name: sha256(path) for name, path in files.items()}


def build_protocol(args, report):
    return {
        'version': VERSION, 'smoke_only': args.smoke, 'audit': report,
        'source_sha256': source_hashes(), 'seed': args.seed,
        'base_channels': args.channels, 'batch_size': args.batch_size,
        'workers': args.workers, 'lr': args.lr, 'weight_decay': 1e-4,
        'coarse_epochs': args.coarse_epochs, 'diffusion_epochs': args.diffusion_epochs,
        'coarse_objective': 'unchanged valid normalized map MSE',
        'architecture': 'unchanged Predictor and Corrector from pl3d_residual_prediction',
        'ema_decay': .999, 'scale': args.scale, 'diffusion_steps': 200,
        'schedule': 'cosine', 'prediction_type': 'v', 'residual_clipping': False,
        'training_timestep_distribution': '0.75 uniform over [0,199] + 0.25 exact terminal 199',
        'diffusion_loss': {'valid_v_mse': 1., 'valid_map_l1': .5, 'valid_gradient_l1': .05,
                           'invalid_v_mse_zero_clean_residual': .1,
                           'high_noise_clean_residual_mse': 1., 'high_noise_alpha_bar_threshold': .2},
        'coarse_forward_for_diffusion': 'frozen EMA coarse best, FP32, outside autocast',
        'training_precision': 'float16_amp_with_scaler' if args.device.type == 'cuda' else 'float32',
        'condition_profile': 'aff_prediction', 'sample_rate': 0, 'include_idw_prior': False,
        'train_samples': args.train_samples, 'val_samples': args.val_samples,
        'checkpoint_selection': {'split': 'val', 'samples': args.val_samples,
                                 'objective': 'macro per sample-height MSE',
                                 'diffusion_steps': 30, 'strength': 1., 'geometry_projection': True,
                                 'noise_seed': 'seed+100000+validation_iteration_index'},
        'inference_calibration': {'split': 'val', 'samples': args.val_samples,
                                  'steps': list(CALIBRATION_STEPS),
                                  'strengths': list(CALIBRATION_STRENGTHS), 'repeats': 3,
                                  'geometry_projection': True, 'objective': 'macro sample-height-repeat MSE',
                                  'noise_seed': 'seed+2000000+repeat*100000+manifest_index',
                                  'tie_break': 'lowest steps, then lowest strength', 'ensemble': False},
        'test': {'samples': args.max_samples, 'repeats': args.repeats,
                 'noise_seed': 'seed+1000000+repeat*100000+manifest_index',
                 'metrics': 'valid_four_legacy_normalized_minus1_plus1',
                 'psnr_data_range': 2., 'aggregation': 'macro sample-height, metrics averaged over repeats',
                 'ensemble': False},
        'budget_disclosure': '10 coarse + 10 diffusion epochs = 20 epoch-passes; not equal compute to 10-epoch baselines',
        'no_guaranteed_improvement': True,
    }


def ensure_geometry(c, valid):
    available = available_from_condition(c)
    if available.shape != valid.shape or not torch.equal(available, valid.bool()):
        raise ValueError('Dataset valid mask differs from building-input-derived support; abort to avoid masking leakage')
    if not valid.any():
        raise ValueError('No valid training/evaluation voxels')


def slice_mses(pred, target, mask):
    values = []
    for b in range(pred.shape[0]):
        for h in range(pred.shape[2]):
            keep = mask[b, 0, h]
            if keep.any():
                values.append(float((pred[b, 0, h] - target[b, 0, h]).square()[keep].mean()))
    return values


@torch.no_grad()
def validate(args, coarse, corrector, data, alpha):
    coarse.eval()
    if corrector is not None:
        corrector.eval()
    values = []
    begun = time.perf_counter()
    for i, batch in enumerate(data):
        c, y, m = tensors(batch, args.device)
        ensure_geometry(c, m)
        mu = coarse(c.float()).float().clamp(-1, 1)
        pred = mu
        if corrector is not None:
            delta = sample_residual(corrector, c, mu, alpha, args.scale, steps=30,
                                    seed=args.seed + 100000 + i, project_geometry=True)
            pred = (mu + delta).clamp(-1, 1)
        values.extend(slice_mses(pred, y, m))
        if (i + 1) % 4 == 0 or i + 1 == len(data):
            seconds = time.perf_counter() - begun
            label = 'coarse' if corrector is None else 'fixed DDIM30/strength1'
            print(f'[validation {label}] {i+1}/{len(data)} '
                  f'MSE={np.mean(values):.6f} elapsed={seconds/60:.1f}min', flush=True)
    if not values or not np.isfinite(values).all():
        raise FloatingPointError('Empty/nonfinite validation metrics')
    return float(np.mean(values))


def restore_rng(ck, device):
    random.setstate(ck['python_rng'])
    np.random.set_state(ck['numpy_rng'])
    torch.set_rng_state(ck['torch_rng'].cpu())
    if device.type == 'cuda':
        torch.cuda.set_rng_state_all([s.cpu() for s in ck['cuda_rng']])


def rng_state(device):
    return {'python_rng': random.getstate(), 'numpy_rng': np.random.get_state(),
            'torch_rng': torch.get_rng_state(),
            'cuda_rng': torch.cuda.get_rng_state_all() if device.type == 'cuda' else []}


def checkpoint_hashes(args):
    return {stage: sha256(args.output / stage / 'best.pt') for stage in ('coarse', 'diffusion')}


def check_ck(ck, config, stage, coarse_hash=None):
    if ck.get('version') != VERSION or ck.get('stage') != stage or ck.get('protocol') != config:
        raise ValueError(f'{stage} checkpoint protocol/source mismatch; do not reuse old checkpoints')
    if ck.get('coarse_sha256') != coarse_hash:
        raise ValueError('Frozen coarse checkpoint provenance mismatch')


def train_stage(args, stage, coarse, data, val, alpha, config):
    out = args.output / stage
    out.mkdir(parents=True, exist_ok=True)
    last = out / 'last.pt'
    coarse_hash = sha256(args.output / 'coarse/best.pt') if stage == 'diffusion' else None
    net = (Predictor(args.channels) if stage == 'coarse' else Corrector(args.channels)).to(args.device)
    ema = copy.deepcopy(net).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = make_grad_scaler(enabled=args.device.type == 'cuda')
    epochs = args.coarse_epochs if stage == 'coarse' else args.diffusion_epochs
    start, best = 0, float('inf')
    if last.exists():
        if not args.resume:
            raise ValueError('Existing checkpoint requires --resume')
        ck = load(last, args.device)
        check_ck(ck, config, stage, coarse_hash)
        net.load_state_dict(ck['model'])
        ema.load_state_dict(ck['ema'])
        optimizer.load_state_dict(ck['optimizer'])
        scaler.load_state_dict(ck['scaler'])
        start, best = ck['epoch'], ck['best']
        if start > epochs:
            raise ValueError('Checkpoint epoch exceeds requested budget')
        restore_rng(ck, args.device)
        del ck
        print(f'[{stage}] RESUME epoch={start}/{epochs}, best={best:.6f}', flush=True)
    if stage == 'diffusion':
        coarse.eval().requires_grad_(False)
    for epoch in range(start, epochs):
        net.train()
        begun = time.perf_counter()
        totals, count, updates, skipped = {}, 0, 0, 0
        for i, batch in enumerate(data):
            c, y, m = tensors(batch, args.device)
            ensure_geometry(c, m)
            optimizer.zero_grad(set_to_none=True)
            # Keep the residual target identical to FP32 inference. Do not put
            # the frozen coarse predictor inside the corrector's AMP context.
            if stage == 'diffusion':
                with torch.no_grad():
                    mu = coarse(c.float()).float().clamp(-1, 1)
                clean = torch.where(m, (y - mu) / args.scale, torch.zeros_like(y))
                t = draw_timesteps(len(c), alpha, args.device, terminal_probability=.25)
                a = alpha[t].view(-1, 1, 1, 1, 1)
                eps = torch.randn_like(clean)
                xt = a.sqrt() * clean + (1 - a).sqrt() * eps
                v_target = a.sqrt() * eps - (1 - a).sqrt() * clean
            with autocast(device_type=args.device.type, enabled=args.device.type == 'cuda'):
                if stage == 'coarse':
                    pred = net(c)
                    loss = masked_mse(pred.float(), y, m)
                    components = {'map_mse': loss}
                else:
                    v = net(xt, c, t, mu).float()
                    loss, components = diffusion_losses(
                        v, xt, v_target, a, mu, y, m, args.scale,
                        invalid_weight=.1, high_noise_weight=1., high_noise_threshold=.2)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite {stage} training loss at epoch={epoch+1} batch={i+1}')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.)
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            did_update = scaler.get_scale() >= previous_scale
            if args.device.type != 'cuda' and not torch.isfinite(grad_norm):
                raise FloatingPointError('Nonfinite gradients with no AMP recovery')
            if did_update:
                updates += 1
                with torch.no_grad():
                    for ep, p in zip(ema.parameters(), net.parameters()):
                        ep.lerp_(p, .001)
            else:
                skipped += 1
            scalars = {name: float(value.detach()) for name, value in components.items()}
            scalars['loss'] = float(loss.detach())
            for name, value in scalars.items():
                if not math.isfinite(value):
                    raise FloatingPointError(f'Nonfinite component {name}')
                totals[name] = totals.get(name, 0.) + value
            count += 1
            if (i + 1) % args.log_every == 0 or i + 1 == len(data):
                elapsed = time.perf_counter() - begun
                details = ' '.join(f'{name}={value/count:.6f}' for name, value in totals.items())
                print(f'[{stage}] epoch={epoch+1}/{epochs} batch={i+1}/{len(data)} {details} '
                      f'updates={updates} amp_skipped={skipped} grad_norm={float(grad_norm):.4g} '
                      f'lr={optimizer.param_groups[0]["lr"]:.3g} elapsed={elapsed/60:.1f}min '
                      f'eta_epoch={elapsed/(i+1)*(len(data)-i-1)/60:.1f}min', flush=True)
        if not count or not updates:
            raise RuntimeError('Epoch contained zero successful optimizer updates; no checkpoint saved')
        score = validate(args, ema if stage == 'coarse' else coarse,
                         None if stage == 'coarse' else ema, val, alpha)
        improved_score = score < best
        best = min(best, score)
        ck = {'version': VERSION, 'stage': stage, 'epoch': epoch + 1,
              'model': net.state_dict(), 'ema': ema.state_dict(),
              'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict(),
              'best': best, 'val_sample_mse': score, 'protocol': config,
              'coarse_sha256': coarse_hash, 'updates': updates, 'amp_skipped': skipped,
              **rng_state(args.device)}
        # Both checkpoints use a temporary file and atomic replacement. Save
        # best before last, so last never advertises a best file not yet saved.
        if improved_score:
            save(out / 'best.pt', ck)
        save(last, ck)
        history = {'epoch': epoch + 1, 'train_loss': totals['loss'] / count,
                   'train_components': {k: v / count for k, v in totals.items()},
                   'val_sample_mse': score, 'best': best, 'updates': updates,
                   'amp_skipped': skipped, 'seconds': time.perf_counter() - begun}
        with (out / 'history.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(history, allow_nan=False) + '\n')
        dump(args.output / 'training_status.json', {'stage': stage, 'epoch': epoch + 1,
             'stage_epochs': epochs, 'state': 'running', 'updated_utc': datetime.now(timezone.utc).isoformat(),
             'last_validation_mse': score, 'best_validation_mse': best})
        print(f'[{stage}] epoch={epoch+1}/{epochs} validation_MSE={score:.6f} '
              f'best={best:.6f} best_updated={improved_score} checkpoint_saved', flush=True)
    best_ck = load(out / 'best.pt', args.device)
    check_ck(best_ck, config, stage, coarse_hash)
    ema.load_state_dict(best_ck['ema'])
    del best_ck, net, optimizer, scaler
    if args.device.type == 'cuda':
        torch.cuda.empty_cache()
    return ema


def load_finished_models(args, config):
    coarse_hash = sha256(args.output / 'coarse/best.pt')
    for stage, epochs in (('coarse', args.coarse_epochs), ('diffusion', args.diffusion_epochs)):
        last = load(args.output / stage / 'last.pt')
        check_ck(last, config, stage, coarse_hash if stage == 'diffusion' else None)
        if last['epoch'] != epochs:
            raise ValueError(f'{stage} training is incomplete; calibration/test forbidden')
        del last
    coarse_ck = load(args.output / 'coarse/best.pt', args.device)
    diffusion_ck = load(args.output / 'diffusion/best.pt', args.device)
    check_ck(coarse_ck, config, 'coarse')
    check_ck(diffusion_ck, config, 'diffusion', coarse_hash)
    coarse = Predictor(args.channels).to(args.device).eval().requires_grad_(False)
    corrector = Corrector(args.channels).to(args.device).eval().requires_grad_(False)
    coarse.load_state_dict(coarse_ck['ema'])
    corrector.load_state_dict(diffusion_ck['ema'])
    del coarse_ck, diffusion_ck
    return coarse, corrector


def validate_calibration(args, config, result):
    if result.get('protocol_sha256') != content_hash(config):
        raise ValueError('Frozen calibration protocol/source mismatch')
    if result.get('checkpoint_sha256') != checkpoint_hashes(args):
        raise ValueError('Frozen calibration checkpoints changed; test is forbidden')
    if result.get('split') != 'val' or result.get('selection_spec') != config['inference_calibration']:
        raise ValueError('Frozen calibration did not use the predeclared validation grid')
    selected = result.get('selected', {})
    if selected.get('steps') not in CALIBRATION_STEPS or selected.get('strength') not in CALIBRATION_STRENGTHS:
        raise ValueError('Invalid frozen inference selection')
    declared = result.get('content_sha256')
    if declared != content_hash({k: v for k, v in result.items() if k != 'content_sha256'}):
        raise ValueError('Calibration contents changed after freezing')


def write_rows(path, rows):
    if not rows:
        raise ValueError('Cannot write empty result set')
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


@torch.no_grad()
def calibrate(args, coarse, corrector, alpha, config):
    frozen = args.output / 'calibration_frozen.json'
    if frozen.exists():
        result = read_json(frozen)
        validate_calibration(args, config, result)
        print('[calibration] Existing locked validation selection verified; reused, not retuned', flush=True)
        return result
    ds = dataset(args, 'val', args.val_samples)
    coarse.eval()
    corrector.eval()
    rows, indices, ids = [], [int(x) for x in ds.indices], []
    started = time.perf_counter()
    for i, batch in enumerate(loader(args, ds)):
        c, y, m = tensors(batch, args.device)
        ensure_geometry(c, m)
        mu = coarse(c.float()).float().clamp(-1, 1)
        common = {'id': str(batch['id'][0]), 'scene_id': int(batch['scene_id'][0])}
        ids.append(common['id'])
        for repeat in range(3):
            seed = args.seed + 2000000 + repeat * 100000 + indices[i]
            for steps in CALIBRATION_STEPS:
                delta = sample_residual(corrector, c, mu, alpha, args.scale, steps=steps,
                                        seed=seed, project_geometry=True)
                for strength in CALIBRATION_STRENGTHS:
                    pred = (mu + strength * delta).clamp(-1, 1)
                    for h in range(pred.shape[2]):
                        keep = m[0, 0, h]
                        if not keep.any():
                            continue
                        mse = float((pred[0, 0, h] - y[0, 0, h]).square()[keep].mean())
                        rows.append({**common, 'repeat': repeat, 'noise_seed': seed,
                                     'steps': steps, 'strength': strength, 'height_m': h + 1, 'mse': mse})
        elapsed = time.perf_counter() - started
        print(f'[validation calibration] {i+1}/{len(ds)} grid=2steps*6strengths*3seeds '
              f'elapsed={elapsed/60:.1f}min eta={elapsed/(i+1)*(len(ds)-i-1)/60:.1f}min', flush=True)
    candidates = []
    for steps in CALIBRATION_STEPS:
        for strength in CALIBRATION_STRENGTHS:
            values = [r['mse'] for r in rows if r['steps'] == steps and r['strength'] == strength]
            if not values or not np.isfinite(values).all():
                raise FloatingPointError('Invalid calibration metric')
            candidates.append({'steps': steps, 'strength': strength, 'macro_mse': float(np.mean(values)),
                               'sample_height_repeat_count': len(values)})
    selected = min(candidates, key=lambda r: (r['macro_mse'], r['steps'], r['strength']))
    result = {'version': VERSION, 'split': 'val', 'smoke_only': args.smoke,
              'frozen_utc': datetime.now(timezone.utc).isoformat(),
              'protocol_sha256': content_hash(config), 'checkpoint_sha256': checkpoint_hashes(args),
              'selection_spec': config['inference_calibration'], 'sample_ids': ids,
              'manifest_indices': indices, 'candidates': candidates, 'selected': selected,
              'selected_uses_diffusion': selected['strength'] > 0,
              'interpretation': 'strength=0 means coarse-only selection, not evidence of a diffusion gain',
              'test_used_for_selection': False, 'ensemble': False}
    result['content_sha256'] = content_hash(result)
    write_rows(args.output / 'calibration_val_per_sample.csv', rows)
    dump(frozen, result)
    print(f'[calibration FROZEN BEFORE TEST] steps={selected["steps"]} strength={selected["strength"]} '
          f'validation_MSE={selected["macro_mse"]:.6f} uses_diffusion={selected["strength"]>0}', flush=True)
    return result


@torch.no_grad()
def evaluate(args, coarse, corrector, alpha, config, calibration):
    validate_calibration(args, config, calibration)
    # A smoke run is never permitted to load test labels, even with action=all.
    split = 'val' if args.smoke else 'test'
    ds = dataset(args, split, args.max_samples)
    out = args.output / f'eval_{split}_{len(ds)}_r{args.repeats}'
    summary_path = out / 'summary.json'
    if summary_path.exists():
        old = read_json(summary_path)
        if (old.get('protocol') != config or old.get('calibration_sha256') != calibration['content_sha256']
                or old.get('checkpoint_sha256') != checkpoint_hashes(args)):
            raise ValueError('Existing evaluation does not match the frozen protocol/checkpoints')
        if not args.resume:
            raise ValueError('Evaluation is already complete; use --resume to verify/skip it')
        print(f'[evaluation] Completed immutable result verified; skipped: {out}', flush=True)
        return
    out.mkdir(parents=True, exist_ok=True)
    coarse.eval()
    corrector.eval()
    selected = calibration['selected']
    records = []
    started = time.perf_counter()
    method_names = ('coarse', 'projected_ddim30_strength1', 'validation_selected')
    for i, batch in enumerate(loader(args, ds)):
        c, y, m = tensors(batch, args.device)
        ensure_geometry(c, m)
        mu = coarse(c.float()).float().clamp(-1, 1)
        common = {'id': str(batch['id'][0]), 'scene_id': int(batch['scene_id'][0])}
        for row in metrics(mu, y, m):
            records.append({**common, 'model': 'coarse', 'repeat': 0, 'noise_seed': -1, **row})
        for repeat in range(args.repeats):
            seed = args.seed + 1000000 + repeat * 100000 + int(ds.indices[i])
            delta30 = sample_residual(corrector, c, mu, alpha, args.scale, steps=30,
                                     seed=seed, project_geometry=True)
            fixed = (mu + delta30).clamp(-1, 1)
            if selected['strength'] == 0:
                chosen = mu
            else:
                delta = delta30 if selected['steps'] == 30 else sample_residual(
                    corrector, c, mu, alpha, args.scale, steps=selected['steps'],
                    seed=seed, project_geometry=True)
                chosen = (mu + selected['strength'] * delta).clamp(-1, 1)
            for name, pred in (('projected_ddim30_strength1', fixed), ('validation_selected', chosen)):
                for row in metrics(pred, y, m):
                    records.append({**common, 'model': name, 'repeat': repeat, 'noise_seed': seed, **row})
        if (i + 1) % 5 == 0 or i + 1 == len(ds):
            elapsed = time.perf_counter() - started
            print(f'[{split} fixed+locked-selected] {i+1}/{len(ds)} repeats={args.repeats} '
                  f'elapsed={elapsed/60:.1f}min eta={elapsed/(i+1)*(len(ds)-i-1)/60:.1f}min', flush=True)
    for row in records:
        if not all(math.isfinite(row[key]) for key in METRIC_NAMES):
            raise FloatingPointError('Nonfinite evaluation metrics')
    write_rows(out / 'per_sample.csv', records)
    summary = {'version': VERSION, 'protocol': config, 'split': split, 'smoke_only': args.smoke,
               'samples': len(ds), 'repeats': args.repeats,
               'checkpoint_sha256': checkpoint_hashes(args),
               'calibration_sha256': calibration['content_sha256'], 'selected': selected,
               'selected_uses_diffusion': selected['strength'] > 0,
               'sampling': 'Gaussian initial residual; DDIM eta=0; building-input-derived clean residual projection',
               'aggregation': 'macro per sample-height; metrics averaged over repeats; NO image ensemble',
               'nmse_domain': '[-1,1] legacy normalized values; not physical power NMSE',
               'ssim': 'legacy valid centers with target-filled excluded pixels',
               'seconds': time.perf_counter() - started,
               'parameters': sum(p.numel() for p in coarse.parameters()) + sum(p.numel() for p in corrector.parameters()),
               'corrector_forward_calls_per_prediction': {
                   'coarse': 0, 'projected_ddim30_strength1': 30,
                   'validation_selected': selected['steps'] if selected['strength'] > 0 else 0},
               'not_equal_compute_to_ten_epoch_baselines': True}
    for name in method_names:
        selected_rows = [r for r in records if r['model'] == name]
        summary[name] = summarize(selected_rows)
        summary[name]['macro_mse'] = float(np.mean([r['rmse'] ** 2 for r in selected_rows]))
        summary[name + '_by_height'] = {
            str(h): summarize([r for r in selected_rows if r['height_m'] == h]) for h in range(1, 5)}
        if name != 'coarse':
            summary[name + '_by_repeat'] = {
                str(k): summarize([r for r in selected_rows if r['repeat'] == k]) for k in range(args.repeats)}
    dump(summary_path, summary)
    print(json.dumps({name: summary[name] for name in method_names}, indent=2), flush=True)
    print(f'V2_EVALUATION_COMPLETE: {out}', flush=True)


def parse_args():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--action', choices=('check', 'train', 'calibrate', 'eval', 'all'), default='all')
    p.add_argument('--dataset-root', type=Path, required=True)
    p.add_argument('--output', type=Path, default=ROOT / 'outputs_prediction_main_10plus10_v2')
    p.add_argument('--coarse-epochs', type=int, default=10)
    p.add_argument('--diffusion-epochs', type=int, default=10)
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--channels', type=int, default=32)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--scale', type=float, default=.5)
    p.add_argument('--val-samples', type=int, default=32)
    p.add_argument('--max-samples', type=int, default=200)
    p.add_argument('--train-samples', type=int, default=0)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--log-every', type=int, default=20)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--device', choices=('auto', 'cuda', 'cpu'), default='auto')
    p.add_argument('--smoke', action='store_true', help='1+1 epochs, 2 samples, channels4, validation-only; never a formal result')
    args = p.parse_args()
    if min(args.batch_size, args.channels, args.log_every) < 1 or args.workers < 0:
        p.error('Batch, channels and log interval must be positive; workers >=0')
    if args.smoke:
        if args.output == ROOT / 'outputs_prediction_main_10plus10_v2':
            args.output = ROOT / 'outputs_prediction_main_10plus10_v2_SMOKE'
        args.coarse_epochs = args.diffusion_epochs = 1
        args.train_samples = args.val_samples = args.max_samples = 2
        args.channels = 4
        args.repeats = 1
    else:
        fixed = {'coarse_epochs': 10, 'diffusion_epochs': 10, 'batch_size': 2, 'channels': 32,
                 'lr': 1e-4, 'scale': .5, 'val_samples': 32, 'max_samples': 200,
                 'train_samples': 0, 'repeats': 3, 'seed': 42}
        for key, expected in fixed.items():
            if getattr(args, key) != expected:
                p.error(f'Formal v2 protocol requires --{key.replace("_", "-")} {expected}; use --smoke only for checks')
    selected_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.device != 'auto':
        selected_device = args.device
    if selected_device == 'cuda' and not torch.cuda.is_available():
        p.error('CUDA requested but unavailable')
    args.device = torch.device(selected_device)
    args.output = args.output.resolve()
    args.dataset_root = args.dataset_root.resolve()
    return args


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
    report = audit(args.dataset_root)
    config = build_protocol(args, report)
    alpha = schedule(args.device, steps=200)
    print(f'version={VERSION} device={args.device} batch={args.batch_size} '
          f'epochs={args.coarse_epochs}+{args.diffusion_epochs} channels={args.channels} '
          f'alpha_T={float(alpha[-1]):.6g} SMOKE_ONLY={args.smoke}', flush=True)
    print('Budget: 20 epoch-passes, NOT equal compute to 10-epoch baselines; improvement is not guaranteed.', flush=True)
    if args.action == 'check':
        # Check only train/validation data tensors; test CSV is hashed by audit,
        # but no test target is loaded for preflight or smoke.
        for split in ('train', 'val'):
            ds = dataset(args, split, 1)
            check_args = copy.copy(args)
            check_args.workers = 0
            batch = next(iter(loader(check_args, ds)))
            c, y, m = tensors(batch, args.device)
            ensure_geometry(c, m)
            if not torch.isfinite(c).all() or not torch.isfinite(y).all():
                raise ValueError('Nonfinite input/target data')
            print(f'[check] split={split} condition={tuple(c.shape)} target={tuple(y.shape)} '
                  f'valid={int(m.sum())} geometry_consistent=True', flush=True)
        print(json.dumps(config, indent=2), flush=True)
        print('V2_PREFLIGHT_PASSED', flush=True)
        return
    protocol_path = args.output / 'protocol.json'
    if args.action in ('train', 'all'):
        if protocol_path.exists():
            if not args.resume:
                raise ValueError('Output contains a protocol; use a NEW directory or explicitly --resume')
            if read_json(protocol_path) != config:
                raise ValueError('Resume source/protocol mismatch; no existing outputs were changed')
        else:
            validate_bootstrap_output(args.output)
        args.output.mkdir(parents=True, exist_ok=True)
        if not protocol_path.exists():
            dump(protocol_path, config)
        diffusion_last = args.output / 'diffusion/last.pt'
        if diffusion_last.exists():
            coarse_last = load(args.output / 'coarse/last.pt')
            if coarse_last['epoch'] != args.coarse_epochs:
                raise ValueError('Cannot continue coarse training after diffusion has begun')
            diff_last = load(diffusion_last)
            check_ck(diff_last, config, 'diffusion', sha256(args.output / 'coarse/best.pt'))
            del coarse_last, diff_last
        train = loader(args, dataset(args, 'train', args.train_samples), train=True)
        val = loader(args, dataset(args, 'val', args.val_samples))
        coarse = train_stage(args, 'coarse', None, train, val, alpha, config)
        corrector = train_stage(args, 'diffusion', coarse, train, val, alpha, config)
        del train, val
        dump(args.output / 'training_status.json', {'state': 'training_complete',
             'coarse_epochs': args.coarse_epochs, 'diffusion_epochs': args.diffusion_epochs,
             'updated_utc': datetime.now(timezone.utc).isoformat()})
        print('V2_TRAINING_COMPLETE', flush=True)
    else:
        if not protocol_path.exists() or read_json(protocol_path) != config:
            raise ValueError('Current settings/source differ from saved training protocol')
        coarse, corrector = load_finished_models(args, config)
    if args.action in ('calibrate', 'all'):
        calibration = calibrate(args, coarse, corrector, alpha, config)
    if args.action in ('eval', 'all'):
        if args.action == 'eval':
            frozen = args.output / 'calibration_frozen.json'
            if not frozen.is_file():
                raise ValueError('Run --action calibrate on validation first; no test-based parameter selection allowed')
            calibration = read_json(frozen)
        evaluate(args, coarse, corrector, alpha, config, calibration)


if __name__ == '__main__':
    main()
