"""Draw the diffusion-value figures from outputs_diffusion_value_analysis/.
Usage:  python plot_diffusion_value.py [--input outputs_diffusion_value_analysis] [--S 30]
Figures (PDF + PNG) are written to <input>/figures/."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'],
                     'font.size': 9, 'axes.labelsize': 9, 'legend.fontsize': 8,
                     'xtick.labelsize': 8, 'ytick.labelsize': 8, 'mathtext.fontset': 'stix'})
COL = {'coarse': '#5b6b8c', 'single': '#e0a030', 'mean': '#2b4cb0', 'gt': '#222222'}


def save(fig, out, name):
    for ext in ('pdf', 'png'):
        fig.savefig(out / f'{name}.{ext}', dpi=300, bbox_inches='tight')
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=Path, default=Path(__file__).resolve().parent / 'outputs_diffusion_value_analysis')
    p.add_argument('--S', type=int, default=None, help='step count shown in single-S figures (default: largest)')
    a = p.parse_args()
    src = a.input
    out = src / 'figures'; out.mkdir(exist_ok=True)
    s = json.loads((src / 'summary.json').read_text(encoding='utf-8'))
    steps, K = s['steps'], s['K']
    S = a.S or max(steps)
    single, mean = f'S{S}_single', f'S{S}_mean{K}'

    # ---- Fig A: error vs distance to building edge / LoS transition
    labels = ['0', '1', '2–3', '4–7', '8–15', '≥16']
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4), sharey=False)
    for ax, key, title in zip(axes, ('bld', 'los'), ('Distance to building edge (px)', 'Distance to LoS/NLoS transition (px)')):
        x = np.arange(len(labels))
        for name, lab, col, mk in (('coarse', 'Coarse', COL['coarse'], 's'),
                                   (single, f'Diffusion, S={S}, single', COL['single'], '^'),
                                   (mean, f'Diffusion, S={S}, mean of {K}', COL['mean'], 'o')):
            ax.plot(x, s['boundary_rmse'][name][key], marker=mk, color=col, label=lab, lw=1.4, ms=4)
        ax.set_xticks(x, labels); ax.set_xlabel(title); ax.set_ylabel('RMSE'); ax.grid(alpha=.3)
    axes[0].legend(frameon=False)
    save(fig, out, 'fig_boundary_error')

    # ---- Fig B: power spectra
    psd = np.load(src / 'psd.npz')
    fr = np.arange(1, len(psd['gt']))
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    ax.loglog(fr, psd['gt'][1:], color=COL['gt'], lw=1.6, label='Ground truth')
    ax.loglog(fr, psd['coarse'][1:], color=COL['coarse'], lw=1.2, label='Coarse')
    ax.loglog(fr, psd[single][1:], color=COL['single'], lw=1.2, label=f'Diffusion, S={S}, single')
    ax.loglog(fr, psd[mean][1:], color=COL['mean'], lw=1.2, ls='--', label=f'Diffusion, S={S}, mean of {K}')
    ax.set_xlabel('Spatial frequency (cycles / map)'); ax.set_ylabel('Power'); ax.grid(alpha=.3, which='both')
    ax.legend(frameon=False)
    save(fig, out, 'fig_power_spectrum')

    # ---- Fig C: sparsification
    fig, axes = plt.subplots(1, len(steps), figsize=(2.4 * len(steps), 2.3), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, st in zip(axes, steps):
        d = s['sparsification'][str(st)]
        ax.plot(d['fraction_removed'], d['by_uncertainty'], color=COL['mean'], lw=1.4, label='Ranked by diffusion std')
        ax.plot(d['fraction_removed'], d['oracle'], color=COL['gt'], lw=1.0, ls='--', label='Oracle (ranked by error)')
        ax.set_title(f'S={st}  (ρ={d["pearson_std_abs_err"]:.2f})', fontsize=8)
        ax.set_xlabel('Fraction of voxels removed'); ax.grid(alpha=.3)
    axes[0].set_ylabel('RMSE of remaining voxels'); axes[0].legend(frameon=False)
    save(fig, out, 'fig_sparsification')

    # ---- Fig D: RMSE vs steps and vs ensemble size
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4))
    ax = axes[0]
    ax.axhline(s['overall_rmse']['coarse'], color=COL['coarse'], ls=':', label='Coarse')
    ax.plot(steps, [s['overall_rmse'][f'S{t}_single'] for t in steps], marker='^', color=COL['single'], label='Single sample')
    ax.plot(steps, [s['overall_rmse'][f'S{t}_mean{K}'] for t in steps], marker='o', color=COL['mean'], label=f'Mean of {K}')
    ax.set_xscale('log'); ax.set_xlabel('DDIM steps S'); ax.set_ylabel('RMSE'); ax.grid(alpha=.3, which='both'); ax.legend(frameon=False)
    ax = axes[1]
    ax.axhline(s['overall_rmse']['coarse'], color=COL['coarse'], ls=':', label='Coarse')
    for t, mk in zip(steps, 'o^sdv'):
        kc = s['k_curve'][str(t)]
        ks = sorted(int(k) for k in kc)
        ax.plot(ks, [kc[str(k)] for k in ks], marker=mk, label=f'S={t}')
    ax.set_xscale('log', base=2); ax.set_xlabel('Number of averaged samples K'); ax.set_ylabel('RMSE')
    ax.grid(alpha=.3, which='both'); ax.legend(frameon=False)
    save(fig, out, 'fig_steps_and_ensemble')

    # ---- Fig E: qualitative panels with uncertainty
    files = sorted((src / 'arrays').glob('*.npz'))
    if files:
        rows = min(3, len(files)); h = 3   # 4 m layer
        fig, axes = plt.subplots(rows, 5, figsize=(7.2, 1.5 * rows + 0.3))
        axes = np.atleast_2d(axes)
        for r, f in enumerate(files[:rows]):
            d = np.load(f)
            v = d['valid'][h]
            def mask(x):
                return np.ma.array(x, mask=~v)
            err = np.abs(d[f'S{S}_mean'][h] - d['gt'][h])
            panels = [(mask(d['gt'][h]), 'Ground truth', 'viridis', (-1, 1)),
                      (mask(d['coarse'][h]), 'Coarse', 'viridis', (-1, 1)),
                      (mask(d[f'S{S}_mean'][h]), f'Diffusion (mean of {K})', 'viridis', (-1, 1)),
                      (mask(d[f'S{S}_std'][h]), 'Diffusion std', 'magma', (0, None)),
                      (mask(err), '|Error|', 'magma', (0, None))]
            for cidx, (img, title, cmap, (lo, hi)) in enumerate(panels):
                ax = axes[r, cidx]
                cm = plt.get_cmap(cmap).copy(); cm.set_bad('#d9d9d9')
                im = ax.imshow(img, cmap=cm, vmin=lo, vmax=hi if hi is not None else float(np.nanpercentile(img.filled(np.nan), 99)))
                ax.set_xticks([]); ax.set_yticks([])
                if r == 0: ax.set_title(title, fontsize=8)
        save(fig, out, 'fig_qualitative_uncertainty')
    print('figures ->', out)


if __name__ == '__main__':
    main()
