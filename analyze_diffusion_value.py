"""Diffusion-value analysis for 3DAFF-RadioDiff (v3, frozen checkpoints, read-only).

Computes, on N test samples:
  * coarse prediction and diffusion predictions for several DDIM step counts S;
  * K independent samples per S -> single-sample prediction, K-sample mean, per-voxel std;
  * error versus distance to building edges and to LoS/NLoS transitions;
  * radially averaged power spectra (detail preservation);
  * sparsification curves (does the diffusion std locate large errors?);
  * RMSE of the K-sample mean for K = 1, 2, 4, ..., K_max.
Nothing is trained or overwritten; results go to a new folder.

Run from the package root (same environment used for the DDIM-200 evaluation):
  D:\\Anaconda3\\envs\\rmdm5070\\python.exe analyze_diffusion_value.py --samples 40 --K 8 --steps 1 30 200
Then:
  D:\\Anaconda3\\envs\\rmdm5070\\python.exe plot_diffusion_value.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

PACKAGE = Path(__file__).resolve().parent
ORIGIN = PACKAGE / 'outputs_aff_transformer_v3_3090'
DEFAULT_DATA = Path(r'D:\datas\pl_3d_h1_h4_npz_v1')
DIST_BINS = [(0, 1), (1, 2), (2, 4), (4, 8), (8, 16), (16, 10 ** 9)]   # pixels, [lo, hi)
MAX_DIST = 16


def parse():
    p = argparse.ArgumentParser()
    p.add_argument('--samples', type=int, default=40)
    p.add_argument('--split', choices=['test', 'val'], default='test')
    p.add_argument('--K', type=int, default=8)
    p.add_argument('--steps', type=int, nargs='+', default=[1, 30, 200])
    p.add_argument('--chunk', type=int, default=4, help='samples drawn per corrector batch')
    p.add_argument('--data', type=Path, default=DEFAULT_DATA)
    p.add_argument('--output', type=Path, default=PACKAGE / 'outputs_diffusion_value_analysis')
    p.add_argument('--qualitative', type=int, default=4, help='how many samples to store as full arrays')
    p.add_argument('--voxels_per_sample', type=int, default=20000, help='voxels kept for sparsification')
    return p.parse_args()


def distance_map(torch, F, invalid_or_edge):
    """Chebyshev distance (pixels, capped at MAX_DIST) to the nearest True pixel, per 2D slice.
    invalid_or_edge: bool tensor [H, W]."""
    cur = invalid_or_edge.float()[None, None]
    dist = torch.full(invalid_or_edge.shape, float(MAX_DIST), device=cur.device)
    dist[invalid_or_edge] = 0.
    reached = invalid_or_edge.clone()
    for d in range(1, MAX_DIST):
        cur = F.max_pool2d(cur, 3, stride=1, padding=1)
        new = (cur[0, 0] > 0) & ~reached
        dist[new] = float(d)
        reached |= new
    return dist


def radial_psd(x):
    """Radially averaged power spectrum of a 2D numpy array (mean removed)."""
    x = x - x.mean()
    f = np.fft.fftshift(np.fft.fft2(x))
    p = np.abs(f) ** 2
    h, w = x.shape
    yy, xx = np.indices((h, w))
    r = np.sqrt((yy - h / 2) ** 2 + (xx - w / 2) ** 2).astype(int)
    rmax = min(h, w) // 2
    s = np.bincount(r.ravel(), p.ravel(), minlength=rmax + 1)[:rmax]
    n = np.bincount(r.ravel(), minlength=rmax + 1)[:rmax]
    return s / np.maximum(n, 1)


def main():
    a = parse()
    import torch
    import torch.nn.functional as F
    sys.path.insert(0, str(PACKAGE))
    from scripts.run_prediction_transformer_v3 import make_engine
    engine = make_engine()
    config = json.loads((ORIGIN / 'protocol.json').read_text(encoding='utf-8'))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    args = SimpleNamespace(output=ORIGIN, dataset_root=a.data, device=device, coarse_epochs=10,
                           diffusion_epochs=10, channels=32, seed=42, workers=0, batch_size=1)
    coarse, corrector = engine.load_finished_models(args, config)
    coarse.eval(); corrector.eval()
    alpha = engine.schedule(device, steps=200)
    out = a.output
    out.mkdir(parents=True, exist_ok=False)
    (out / 'arrays').mkdir()
    ds = engine.dataset(args, a.split, a.samples)
    methods = ['coarse'] + [f'S{s}_single' for s in a.steps] + [f'S{s}_mean{a.K}' for s in a.steps]
    # accumulators
    bin_sse = {m: {k: np.zeros(len(DIST_BINS)) for k in ('bld', 'los')} for m in methods}
    bin_n = {k: np.zeros(len(DIST_BINS)) for k in ('bld', 'los')}
    psd = {m: None for m in methods + ['gt']}
    psd_count = 0
    per_slice = []                       # rows of per-sample-height RMSE
    kcurve = {s: {} for s in a.steps}    # S -> K' -> list of slice MSE
    spars = {s: {'std': [], 'err': []} for s in a.steps}
    rng = np.random.default_rng(0)
    begun = time.time()
    with torch.inference_mode():
        for i, batch in enumerate(engine.loader(args, ds)):
            c, y, m = engine.tensors(batch, device)
            engine.ensure_geometry(c, m)
            mu = coarse(c.float()).float().clamp(-1, 1)          # [1,1,4,H,W]
            valid = m.bool()
            preds = {'coarse': mu}
            samples = {}
            for s in a.steps:
                draws = []
                for j in range(0, a.K, a.chunk):
                    n = min(a.chunk, a.K - j)
                    cc = c.expand(n, *c.shape[1:]).contiguous()
                    mm = mu.expand(n, *mu.shape[1:]).contiguous()
                    seed = 42 + 3_000_000 + int(ds.indices[i]) * 1000 + s * 10 + j
                    delta = engine.sample_residual(corrector, cc, mm, alpha, .5, steps=s, seed=seed,
                                                   project_geometry=True)
                    draws.append((mm + delta).clamp(-1, 1))
                stack = torch.cat(draws, 0)                          # [K,1,4,H,W]
                samples[s] = stack
                preds[f'S{s}_single'] = stack[:1]
                preds[f'S{s}_mean{a.K}'] = stack.mean(0, keepdim=True)
            sid = str(batch['id'][0])
            for h in range(y.shape[2]):
                vh = valid[0, 0, h]
                gt = y[0, 0, h]
                d_bld = distance_map(torch, F, ~vh)
                los = c[0, 5, h] > 0.5
                edge = los ^ F.max_pool2d(los.float()[None, None], 3, 1, 1)[0, 0].bool()
                d_los = distance_map(torch, F, edge)
                bins = {}
                for key, dm in (('bld', d_bld), ('los', d_los)):
                    bins[key] = [vh & (dm >= lo) & (dm < hi) for lo, hi in DIST_BINS]
                    for b, sel in enumerate(bins[key]):
                        bin_n[key][b] += float(sel.sum())
                row = dict(id=sid, height=h + 1)
                for name, p in preds.items():
                    e2 = (p[0, 0, h] - gt).square()
                    row[name] = float(e2[vh].mean().sqrt())
                    for key in ('bld', 'los'):
                        for b, sel in enumerate(bins[key]):
                            bin_sse[name][key][b] += float(e2[sel].sum())
                per_slice.append(row)
                # power spectra (invalid pixels filled with ground truth for every map)
                g_np = gt.cpu().numpy()
                v_np = vh.cpu().numpy()
                spectra = {'gt': radial_psd(g_np)}
                for name, p in preds.items():
                    x = np.where(v_np, p[0, 0, h].cpu().numpy(), g_np)
                    spectra[name] = radial_psd(x)
                for name, sp in spectra.items():
                    psd[name] = sp if psd[name] is None else psd[name] + sp
                psd_count += 1
                # K-curve and sparsification
                for s in a.steps:
                    st = samples[s][:, 0, h]                           # [K,H,W]
                    k = 1
                    while k <= a.K:
                        mse = float((st[:k].mean(0) - gt).square()[vh].mean())
                        kcurve[s].setdefault(k, []).append(mse)
                        k *= 2
                    std = st.std(0, unbiased=True)[vh].cpu().numpy()
                    err = (st.mean(0) - gt).abs()[vh].cpu().numpy()
                    take = rng.choice(std.size, size=min(std.size, a.voxels_per_sample // 4), replace=False)
                    spars[s]['std'].append(std[take]); spars[s]['err'].append(err[take])
            if i < a.qualitative:
                save = dict(gt=y[0, 0].cpu().numpy(), valid=valid[0, 0].cpu().numpy(),
                            los=c[0, 5].cpu().numpy(), coarse=mu[0, 0].cpu().numpy())
                for s in a.steps:
                    save[f'S{s}_single'] = samples[s][0, 0].cpu().numpy()
                    save[f'S{s}_mean'] = samples[s].mean(0)[0].cpu().numpy()
                    save[f'S{s}_std'] = samples[s].std(0, unbiased=True)[0].cpu().numpy()
                np.savez_compressed(out / 'arrays' / f'{i:03d}_{sid}.npz', **save)
            print(json.dumps(dict(sample=i + 1, of=a.samples, id=sid, elapsed_min=round((time.time() - begun) / 60, 1),
                                  coarse=round(float(np.mean([r['coarse'] for r in per_slice[-4:]])), 5))), flush=True)
    # summaries
    summary = dict(split=a.split, samples=a.samples, K=a.K, steps=a.steps, dist_bins_px=DIST_BINS,
                   overall_rmse={mname: float(np.mean([r[mname] for r in per_slice])) for mname in methods},
                   boundary_rmse={mname: {k: (np.sqrt(bin_sse[mname][k] / np.maximum(bin_n[k], 1))).tolist()
                                          for k in ('bld', 'los')} for mname in methods},
                   boundary_voxels={k: bin_n[k].tolist() for k in ('bld', 'los')},
                   k_curve={str(s): {str(k): float(np.mean(np.sqrt(v))) for k, v in kcurve[s].items()}
                            for s in a.steps})
    sp_out = {}
    for s in a.steps:
        std = np.concatenate(spars[s]['std']); err = np.concatenate(spars[s]['err'])
        fr = np.linspace(0, 0.95, 20)
        def curve(order):
            e = err[order]
            return [float(np.sqrt(np.mean(e[int(f * e.size):] ** 2))) for f in fr]
        by_unc = curve(np.argsort(-std)); oracle = curve(np.argsort(-err))
        corr = float(np.corrcoef(std, err)[0, 1])
        sp_out[str(s)] = dict(fraction_removed=fr.tolist(), by_uncertainty=by_unc, oracle=oracle,
                              ause=float(np.trapz(np.array(by_unc) - np.array(oracle), fr)), pearson_std_abs_err=corr)
    summary['sparsification'] = sp_out
    (out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    np.savez_compressed(out / 'psd.npz', **{k: v / psd_count for k, v in psd.items()})
    with (out / 'per_slice.csv').open('w', encoding='utf-8') as f:
        keys = list(per_slice[0].keys())
        f.write(','.join(keys) + '\n')
        for r in per_slice:
            f.write(','.join(str(r[k]) for k in keys) + '\n')
    print('done ->', out)


if __name__ == '__main__':
    main()
