"""Auditable coarse prediction + conditional residual diffusion experiment."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.pl3d_manifest_dataset import PL3DManifestDataset
from models.pl3d_residual_prediction import Predictor, Corrector, schedule, sample, x0_from_v, masked_mse, gradient_loss
from scripts.evaluate_pl3d_aff_fullnoise import balanced_scene_indices, gaussian_window, compute_ssim_map

VERSION = 'q_building_residual_prediction_v1'


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def save(path, payload):
    tmp = path.with_suffix('.tmp')
    torch.save(payload, tmp)
    tmp.replace(path)


def load(path, device):
    return torch.load(path, map_location=device, weights_only=False)


def audit(root):
    sets, report = {}, {}
    for split in ('train', 'val', 'test'):
        path = root / (split + '.csv')
        with path.open(newline='', encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
        ids = [r['sample_id'] for r in rows]
        if not rows or len(ids) != len(set(ids)):
            raise ValueError('Empty split or duplicate sample IDs: ' + split)
        sets[split] = {r['scene_id'] for r in rows}
        report[split] = {'samples': len(rows), 'scenes': len(sets[split]),
                         'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    for a, b in (('train', 'val'), ('train', 'test'), ('val', 'test')):
        if sets[a] & sets[b]:
            raise ValueError('Scene leakage: ' + a + '/' + b)
    report['metadata_sha256'] = hashlib.sha256((root / 'metadata.json').read_bytes()).hexdigest()
    report['status'] = 'scene_disjoint'
    return report


def dataset(args, split, maximum):
    ds = PL3DManifestDataset(args.dataset_root, split, sample_rate=0,
                            condition_profile='aff_prediction', include_idw_prior=False,
                            deterministic_sampling=True, seed=args.seed)
    indices = balanced_scene_indices(ds, maximum, args.seed)
    return Subset(ds, indices)


def loader(args, ds, train=False):
    kw = dict(batch_size=args.batch_size if train else 1, shuffle=train,
              num_workers=args.workers, pin_memory=args.device.type == 'cuda')
    if args.workers:
        kw.update(persistent_workers=True, prefetch_factor=2)
    return DataLoader(ds, **kw)


def tensors(batch, device):
    c = batch['condition'].to(device, non_blocking=True)
    return c, batch['target'].to(device, non_blocking=True), batch['valid_mask'].to(device, non_blocking=True).bool()


def metrics(pred, target, mask):
    # Exact historical valid_four convention: normalized [-1,1], range=2,
    # per-height macro averaging. Target-fill outside mask only for SSIM.
    p, y, m = pred[:, 0].float(), target[:, 0].float(), mask[:, 0].bool()
    win = gaussian_window(11, 1.5, 4, p.device)
    sm = compute_ssim_map(torch.where(m, p, y), y, win, data_range=2.)
    out = []
    for h in range(4):
        v = m[0, h]
        if not v.any():
            continue
        e = (p[0, h] - y[0, h]).square()[v]
        mse = e.mean()
        out.append(dict(height_m=h+1, nmse=float(e.sum()/y[0,h][v].square().sum().clamp_min(1e-12)),
                        rmse=float(mse.sqrt()), psnr=float(10*torch.log10(4/(mse+1e-8))),
                        ssim=float(sm[0,h][v].mean()), evaluated_voxels=int(v.sum())))
    return out


@torch.no_grad()
def validate(args, coarse, corrector, data, alpha):
    coarse.eval()
    if corrector is not None:
        corrector.eval()
    scores = []
    for i, batch in enumerate(data):
        c, y, m = tensors(batch, args.device)
        mu = coarse(c).clamp(-1, 1)
        pred = mu if corrector is None else sample(corrector, c, mu, alpha, args.scale, args.steps, args.seed + 100000 + i)
        for h in range(4):
            v = m[:, :, h]
            if v.any():
                scores.append(float((pred[:, :, h] - y[:, :, h]).square()[v].mean()))
        if (i+1) % 8 == 0 or i+1 == len(data):
            print(f'[validation sampling] {i+1}/{len(data)} slice_mse={np.mean(scores):.6f}', flush=True)
    if not scores:
        raise ValueError('No valid validation voxels')
    return float(np.mean(scores))


def protocol(args, report):
    return dict(version=VERSION, audit=report, seed=args.seed, base_channels=args.channels,
                batch_size=args.batch_size, lr=args.lr, scale=args.scale, steps=args.steps,
                diffusion_steps=200, schedule='cosine', prediction_type='v',
                val_samples=args.val_samples, train_samples=args.train_samples,
                ema_decay=.999, condition_profile='aff_prediction', sample_rate=0,
                residual_clipping=False, metric='valid_four_legacy_normalized_minus1_plus1')


def train_stage(args, stage, coarse, data, val, alpha, config):
    out = args.output / stage
    out.mkdir(parents=True, exist_ok=True)
    last = out / 'last.pt'
    coarse_hash = (hashlib.sha256((args.output/'coarse'/'best.pt').read_bytes()).hexdigest()
                   if stage == 'diffusion' else None)
    if last.exists() and not args.resume:
        raise ValueError(f'Existing training output: {last}; use --resume or a new output directory')
    net = Predictor(args.channels) if stage == 'coarse' else Corrector(args.channels)
    net = net.to(args.device)
    ema = copy.deepcopy(net).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=args.device.type == 'cuda')
    start, best = 0, float('inf')
    if args.resume and last.exists():
        ck = load(last, args.device)
        if ck['protocol'] != config:
            raise ValueError('Resume protocol mismatch; retain original settings/data')
        if ck.get('coarse_sha256') != coarse_hash:
            raise ValueError('Coarse checkpoint changed since diffusion training')
        net.load_state_dict(ck['model'])
        ema.load_state_dict(ck['ema'])
        optimizer.load_state_dict(ck['optimizer'])
        scaler.load_state_dict(ck['scaler'])
        start, best = ck['epoch'], ck['best']
        random.setstate(ck['python_rng'])
        np.random.set_state(ck['numpy_rng'])
        torch.set_rng_state(ck['torch_rng'].cpu())
        if args.device.type == 'cuda':
            torch.cuda.set_rng_state_all([s.cpu() for s in ck['cuda_rng']])
    epochs = args.coarse_epochs if stage == 'coarse' else args.diffusion_epochs
    if stage == 'diffusion':
        coarse.eval().requires_grad_(False)
    for epoch in range(start, epochs):
        net.train()
        begun, total, count = time.perf_counter(), 0., 0
        for i, batch in enumerate(data):
            c, y, m = tensors(batch, args.device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=args.device.type, enabled=args.device.type == 'cuda'):
                if stage == 'coarse':
                    pred = net(c)
                    loss = masked_mse(pred.float(), y, m)
                else:
                    with torch.no_grad():
                        mu = coarse(c).float().clamp(-1, 1)
                    r = (y - mu) / args.scale
                    # Invalid building interiors have a fixed residual of zero;
                    # both training noise and inference noise cover the whole volume.
                    r = torch.where(m, r, torch.zeros_like(r))
                    t = torch.randint(0, len(alpha), (len(c),), device=args.device)
                    a = alpha[t].view(-1, 1, 1, 1, 1)
                    eps = torch.randn_like(r)
                    xt = a.sqrt()*r + (1-a).sqrt()*eps
                    v_target = a.sqrt()*eps - (1-a).sqrt()*r
                    v = net(xt, c, t, mu).float()
                    r0 = x0_from_v(xt, v, a)
                    pred = mu + args.scale*r0
                    loss = (masked_mse(v, v_target, m) + .5*(pred-y).abs()[m].mean()
                            + .05*gradient_loss(pred, y, m))
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training loss')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.)
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= old_scale:
                with torch.no_grad():
                    for ep, p in zip(ema.parameters(), net.parameters()):
                        ep.lerp_(p, .001)
            total += float(loss.detach())
            count += 1
            if (i+1) % args.log_every == 0 or i+1 == len(data):
                elapsed = time.perf_counter()-begun
                eta = elapsed/(i+1)*(len(data)-i-1)/60
                print(f'[{stage}] epoch {epoch+1}/{epochs} batch {i+1}/{len(data)} '
                      f'loss={total/count:.6f} eta_epoch={eta:.1f}min', flush=True)
        score = validate(args, ema if stage=='coarse' else coarse,
                         None if stage=='coarse' else ema, val, alpha)
        improved = score < best
        best = min(best, score)
        ck = dict(version=VERSION, stage=stage, epoch=epoch+1, model=net.state_dict(), ema=ema.state_dict(),
                  optimizer=optimizer.state_dict(), scaler=scaler.state_dict(), best=best,
                  val_sample_mse=score, protocol=config, coarse_sha256=coarse_hash, python_rng=random.getstate(),
                  numpy_rng=np.random.get_state(), torch_rng=torch.get_rng_state(),
                  cuda_rng=torch.cuda.get_rng_state_all() if args.device.type=='cuda' else [])
        save(last, ck)
        if improved:
            save(out/'best.pt', ck)
        with (out/'history.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(dict(epoch=epoch+1, train_loss=total/count, val_sample_mse=score,
                                    seconds=time.perf_counter()-begun))+'\n')
        print(f'[{stage}] epoch={epoch+1} validation_MSE={score:.6f} best={best:.6f} saved', flush=True)
    best_ck = load(out/'best.pt', args.device)
    ema.load_state_dict(best_ck['ema'])
    return ema


def summarize(rows):
    return {k: {'mean': float(np.mean([r[k] for r in rows])),
                'std': float(np.std([r[k] for r in rows]))} for k in ('nmse','rmse','ssim','psnr')}


@torch.no_grad()
def evaluate(args, coarse, corrector, alpha, config):
    ds = dataset(args, 'test', args.max_samples)
    out = args.output / f'eval_ddim{args.steps}_valid_{len(ds)}_r{args.repeats}'
    out.mkdir(parents=True, exist_ok=True)
    coarse.eval()
    corrector.eval()
    records = []
    started = time.perf_counter()
    for i, batch in enumerate(loader(args, ds)):
        c, y, m = tensors(batch, args.device)
        mu = coarse(c).clamp(-1, 1)
        common = dict(id=batch['id'][0], scene_id=int(batch['scene_id'][0]))
        for r in metrics(mu, y, m):
            records.append(dict(**common, model='coarse', repeat=0, noise_seed=-1, **r))
        for repeat in range(args.repeats):
            seed = args.seed + 1000000 + repeat*100000 + ds.indices[i]
            pred = sample(corrector, c, mu, alpha, args.scale, args.steps, seed)
            for r in metrics(pred, y, m):
                records.append(dict(**common, model='residual_diffusion', repeat=repeat, noise_seed=seed, **r))
        if (i+1) % 5 == 0 or i+1 == len(ds):
            elapsed = time.perf_counter()-started
            print(f'[test] {i+1}/{len(ds)} repeats={args.repeats} elapsed={elapsed/60:.1f}min '
                  f'eta={elapsed/(i+1)*(len(ds)-i-1)/60:.1f}min', flush=True)
    with (out/'per_sample.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    summary = dict(protocol=config, split='test', samples=len(ds), repeats=args.repeats,
                   sampling='Gaussian residual initialization; DDIM eta=0',
                   aggregation='macro per sample-height; average metrics over repeats, NOT ensemble',
                   nmse_domain='[-1,1] legacy normalized values; not physical power NMSE',
                   ssim='legacy valid centers with target-filled excluded pixels',
                   seconds=time.perf_counter()-started,
                   parameters=sum(p.numel() for p in coarse.parameters())+sum(p.numel() for p in corrector.parameters()),
                   checkpoints={s:hashlib.sha256((args.output/s/'best.pt').read_bytes()).hexdigest() for s in ('coarse','diffusion')})
    for name in ('coarse','residual_diffusion'):
        selected = [r for r in records if r['model']==name]
        summary[name] = summarize(selected)
        summary[name+'_by_height'] = {str(h):summarize([r for r in selected if r['height_m']==h]) for h in range(1,5)}
    summary['diffusion_by_repeat'] = {str(k):summarize([r for r in records if r['model']=='residual_diffusion' and r['repeat']==k]) for k in range(args.repeats)}
    dump(out/'summary.json', summary)
    print(json.dumps({k:summary[k] for k in ('coarse','residual_diffusion')}, indent=2), flush=True)
    print(f'FINISHED: {out}', flush=True)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--action', choices=('check','train','eval','all'), default='all')
    p.add_argument('--dataset-root', type=Path, required=True)
    p.add_argument('--output', type=Path, default=ROOT/'outputs_prediction_residual_v1')
    p.add_argument('--coarse-epochs', type=int, default=10)
    p.add_argument('--diffusion-epochs', type=int, default=10)
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--channels', type=int, default=32)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--scale', type=float, default=.5)
    p.add_argument('--steps', type=int, default=30)
    p.add_argument('--val-samples', type=int, default=32)
    p.add_argument('--max-samples', type=int, default=200)
    p.add_argument('--train-samples', type=int, default=0)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--log-every', type=int, default=20)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    if min(args.coarse_epochs,args.diffusion_epochs,args.batch_size,args.repeats,args.log_every) < 1 or args.scale<=0 or not 1<=args.steps<=200:
        p.error('Invalid positive parameter / sampling steps')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.output.mkdir(parents=True, exist_ok=True)
    report = audit(args.dataset_root)
    config = protocol(args, report)
    alpha = schedule(args.device)
    print(f'version={VERSION} device={args.device} batch={args.batch_size} '
          f'epochs={args.coarse_epochs}+{args.diffusion_epochs} alpha_T={float(alpha[-1]):.3g}', flush=True)
    if args.action == 'check':
        print(json.dumps(report, indent=2))
        return
    if args.action in ('train','all'):
        for stage in ('coarse','diffusion'):
            last = args.output/stage/'last.pt'
            if last.exists():
                if not args.resume:
                    raise ValueError('Output already exists; use --resume or a new output directory')
                if load(last, 'cpu')['protocol'] != config:
                    raise ValueError('Resume protocol mismatch; no files changed')
        if (args.output/'diffusion'/'last.pt').exists() and args.resume:
            ck = load(args.output/'coarse'/'last.pt', 'cpu')
            if args.coarse_epochs != ck['epoch']:
                raise ValueError('Cannot change coarse training after diffusion has started')
        dump(args.output/'protocol.json', config)
        data = loader(args, dataset(args, 'train', args.train_samples), True)
        val = loader(args, dataset(args, 'val', args.val_samples))
        coarse = train_stage(args, 'coarse', None, data, val, alpha, config)
        corrector = train_stage(args, 'diffusion', coarse, data, val, alpha, config)
    else:
        ck = load(args.output/'diffusion'/'best.pt', args.device)
        config = ck['protocol']
        if config['audit'] != report:
            raise ValueError('Evaluation dataset manifests differ from training')
        for field, key in (('channels','base_channels'),('scale','scale'),('steps','steps'),('seed','seed')):
            setattr(args, field, config[key])
        coarse = Predictor(args.channels).to(args.device)
        coarse_ck = load(args.output/'coarse'/'best.pt', args.device)
        if coarse_ck['protocol'] != config:
            raise ValueError('Coarse and diffusion protocols differ')
        if ck.get('coarse_sha256') != hashlib.sha256((args.output/'coarse'/'best.pt').read_bytes()).hexdigest():
            raise ValueError('Evaluation coarse checkpoint differs from diffusion training')
        coarse.load_state_dict(coarse_ck['ema'])
        corrector = Corrector(args.channels).to(args.device)
        corrector.load_state_dict(ck['ema'])
    if args.action in ('eval','all'):
        evaluate(args, coarse, corrector, alpha, config)


if __name__ == '__main__':
    main()
