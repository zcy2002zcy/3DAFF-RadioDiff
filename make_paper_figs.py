"""Publication figures for the diffusion-value discussion (IEEE two-column sizing).
Input : an output folder of analyze_diffusion_value.py (summary.json, per_slice.csv, arrays/*.npz).
Output: <input>/paper_figures/*.pdf|png
Usage : python make_paper_figs.py --input outputs_diffusion_value_analysis [--S 200]
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['STIXGeneral', 'Times New Roman', 'Nimbus Roman'],
    'mathtext.fontset': 'stix', 'font.size': 7, 'axes.labelsize': 7, 'axes.titlesize': 7,
    'legend.fontsize': 6, 'xtick.labelsize': 6, 'ytick.labelsize': 6, 'axes.linewidth': .5,
    'lines.linewidth': 1.2, 'lines.markersize': 3.5, 'legend.frameon': False, 'savefig.dpi': 400})
C_COARSE, C_S1, C_S30, C_S200, C_GT = '#7f7f7f', '#1f77b4', '#ff7f0e', '#2ca02c', '#222222'
COL_W, PAGE_W = 3.5, 7.16


def save(fig, out, name):
    for ext in ('pdf', 'png'):
        fig.savefig(out / f'{name}.{ext}', bbox_inches='tight', pad_inches=.02)
    plt.close(fig)


def panel_label(ax, s):
    # IEEE style: sub-figure label centred below the x-axis label
    xl = ax.get_xlabel()
    ax.set_xlabel(f'{s} {xl}' if xl else s)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--S', type=int, default=None)
    p.add_argument('--hide', type=int, nargs='*', default=[], help='step counts to omit from all figures')
    a = p.parse_args()
    src = a.input
    out = src / 'paper_figures'; out.mkdir(exist_ok=True)
    s = json.loads((src / 'summary.json').read_text(encoding='utf-8'))
    K, steps = s['K'], [t for t in s['steps'] if t not in a.hide]
    show_s1 = 1 in steps
    S = a.S or max(steps)
    n = s['samples']
    rows = list(csv.DictReader((src / 'per_slice.csv').open(encoding='utf-8')))
    mean_key = f'S{S}_mean{K}'

    # ------------------------------------------------------------------ Fig. 1
    fig, ax = plt.subplots(1, 3, figsize=(PAGE_W, 2.05), gridspec_kw=dict(wspace=.42))
    # (a) RMSE vs number of averaged samples
    cols = {1: C_S1, 30: C_S30, 200: C_S200}
    ax[0].axhline(s['overall_rmse']['coarse'], color=C_COARSE, ls=':', lw=1, label='Coarse predictor')
    for t in steps:
        kc = s['k_curve'][str(t)]
        ks = sorted(int(k) for k in kc)
        ax[0].plot(ks, [kc[str(k)] for k in ks], marker='o', color=cols.get(t, None), label=f'Diffusion, $S={t}$')
    ax[0].set_xscale('log', base=2); ax[0].set_xticks([1, 2, 4, 8], ['1', '2', '4', '8'])
    ax[0].set_xlabel('Number of averaged samples $K$'); ax[0].set_ylabel('RMSE')
    ax[0].grid(alpha=.25); ax[0].legend(loc='upper right', fontsize=5.8, framealpha=1, frameon=True, edgecolor='none')
    panel_label(ax[0], '(a)')
    # (b) per-slice scatter
    xc = np.array([float(r['coarse']) for r in rows]); yd = np.array([float(r[mean_key]) for r in rows])
    lim = (0, max(xc.max(), yd.max()) * 1.05)
    ax[1].plot(lim, lim, color=C_GT, lw=.7, ls='--')
    ax[1].scatter(xc, yd, s=6, color=C_S200, alpha=.75, lw=0, label=f'$S={S}$, mean of {K}')
    win = int((yd < xc).sum())
    ax[1].set_xlim(lim); ax[1].set_ylim(lim); ax[1].set_aspect('equal')
    ax[1].set_xlabel('RMSE of coarse predictor'); ax[1].set_ylabel('RMSE after residual diffusion')
    ax[1].text(.04, .93, f'{win}/{len(xc)} slices improved', transform=ax[1].transAxes, fontsize=7)
    ax[1].grid(alpha=.25)
    panel_label(ax[1], '(b)')
    # (c) per-height bars
    hs = sorted({int(r['height']) for r in rows})
    methods = [('coarse', 'Coarse', C_COARSE)]
    if show_s1:
        methods.append(('S1_single', '$S=1$', C_S1))
    methods.append((mean_key, f'$S={S}$, mean of {K}', C_S200))
    w = .8 / len(methods)
    for i, (k, lab, col) in enumerate(methods):
        v = [np.mean([float(r[k]) for r in rows if int(r['height']) == h]) for h in hs]
        ax[2].bar(np.arange(len(hs)) + (i - (len(methods) - 1) / 2) * w, v, w, color=col, label=lab)
    ax[2].set_xticks(range(len(hs)), [f'{h} m' for h in hs]); ax[2].set_ylabel('RMSE')
    ax[2].set_xlabel('Receiver height'); ax[2].grid(alpha=.25, axis='y'); ax[2].legend(loc='upper left')
    lo = min(np.mean([float(r[k]) for r in rows if int(r['height']) == h]) for k, _, _ in methods for h in hs)
    ax[2].set_ylim(lo * 0.85, None)
    panel_label(ax[2], '(c)')
    save(fig, out, 'fig_accuracy_ensemble')

    # ------------------------------------------------------------------ Fig. 2
    files = sorted((src / 'arrays').glob('*.npz'))
    fig, ax = plt.subplots(1, 2, figsize=(PAGE_W, 2.0), gridspec_kw=dict(wspace=.28))
    for t, col in (((1, C_S1),) if show_s1 else ()) + ((S, C_S200),):
        if str(t) not in s['sparsification']:
            continue
        d = s['sparsification'][str(t)]
        f = np.array(d['fraction_removed']) * 100
        by = np.array(d['by_uncertainty']); orc = np.array(d['oracle'])
        ax[0].plot(f, by / by[0], color=col, label=f'By sampling SD ($\\rho$={d["pearson_std_abs_err"]:.2f})')
        if t == S:
            ax[0].plot(f, orc / orc[0], color=C_GT, ls='--', lw=.9, label='Oracle (by true error)')
    ax[0].axhline(1, color=C_COARSE, ls=':', lw=1, label='Random removal')
    ax[0].set_xlabel('Fraction of voxels removed (%)'); ax[0].set_ylabel('Normalized RMSE')
    ax[0].set_ylim(0, 1.08); ax[0].grid(alpha=.25); ax[0].legend(loc='upper right', bbox_to_anchor=(1.0, 0.93), fontsize=5.8)
    panel_label(ax[0], '(a)')
    # (b) reliability: SD deciles vs RMSE within decile (voxel level, stored scenes)
    stds, errs = [], []
    for fpath in files:
        d = np.load(fpath)
        v = d['valid']
        stds.append(d[f'S{S}_std'][v]); errs.append((d[f'S{S}_mean'] - d['gt'])[v])
    if stds:
        sd = np.concatenate(stds); er = np.concatenate(errs)
        q = np.quantile(sd, np.linspace(0, 1, 11))
        idx = np.clip(np.searchsorted(q, sd, side='right') - 1, 0, 9)
        msd = [sd[idx == b].mean() for b in range(10)]
        rm = [np.sqrt((er[idx == b] ** 2).mean()) for b in range(10)]
        ax[1].plot(msd, rm, marker='o', color=C_S200, label=f'Sampling SD ($K={K}$)')
        m = max(max(msd), max(rm)) * 1.1
        ax[1].plot([0, m], [0, m], color=C_GT, ls='--', lw=.8, label='Ideal calibration')
        ax[1].set_xlim(0, m); ax[1].set_ylim(0, m)
        ax[1].set_xlabel('Mean sampling SD within decile'); ax[1].set_ylabel('RMSE within decile')
        ax[1].grid(alpha=.25); ax[1].legend(loc='upper left')
        ax[1].text(.97, .04, f'{len(files)} test scenes, all valid voxels', transform=ax[1].transAxes,
                   ha='right', fontsize=5.8, color='#555555')
    panel_label(ax[1], '(b)')
    save(fig, out, 'fig_uncertainty_quality')

    # ------------------------------------------------------------------ Fig. 3 qualitative
    if files:
        pick = files[:2]
        h = 3
        fig, ax = plt.subplots(len(pick), 6, figsize=(PAGE_W, 1.32 * len(pick) + 0.45),
                               gridspec_kw=dict(wspace=.04, hspace=.08))
        ax = np.atleast_2d(ax)
        emax = 0
        for fpath in pick:
            d = np.load(fpath); v = d['valid'][h]
            emax = max(emax, np.percentile(np.abs(d['coarse'][h] - d['gt'][h])[v], 99))
        smax = max(np.percentile(np.load(fp)[f'S{S}_std'][h][np.load(fp)['valid'][h]], 99) for fp in pick)
        titles = ['Ground truth', 'Coarse', f'Diffusion (mean of {K})', '|Error|, coarse', '|Error|, diffusion', 'Sampling SD']
        ims = {}
        for r, fpath in enumerate(pick):
            d = np.load(fpath); v = d['valid'][h]
            def m(x):
                return np.ma.array(x, mask=~v)
            data = [(m(d['gt'][h]), 'viridis', -1, 1), (m(d['coarse'][h]), 'viridis', -1, 1),
                    (m(d[f'S{S}_mean'][h]), 'viridis', -1, 1),
                    (m(np.abs(d['coarse'][h] - d['gt'][h])), 'inferno', 0, emax),
                    (m(np.abs(d[f'S{S}_mean'][h] - d['gt'][h])), 'inferno', 0, emax),
                    (m(d[f'S{S}_std'][h]), 'inferno', 0, smax)]
            for c, (img, cmap, lo, hi) in enumerate(data):
                cm = plt.get_cmap(cmap).copy(); cm.set_bad('#d0d0d0')
                ims[c] = ax[r, c].imshow(img, cmap=cm, vmin=lo, vmax=hi, interpolation='nearest')
                ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
                for sp in ax[r, c].spines.values():
                    sp.set_linewidth(.4)
                if r == 0:
                    ax[r, c].set_title(titles[c], fontsize=6.5)
            ax[r, 0].set_ylabel(f'Scene {r + 1}', fontsize=7)
        ticks = {0: [-1, -0.5, 0, 0.5, 1], 3: np.round(np.linspace(0, emax, 4), 2), 5: np.round(np.linspace(0, smax, 3), 2)}
        for c0, c1, lab in ((0, 3, 'Normalized path loss'), (3, 5, 'Absolute error'), (5, 6, 'Sampling SD')):
            x0 = ax[-1, c0].get_position().x0 + .02; x1 = ax[-1, c1 - 1].get_position().x1 - .02
            cax = fig.add_axes([x0, ax[-1, 0].get_position().y0 - .07, x1 - x0, .025])
            cb = fig.colorbar(ims[c0], cax=cax, orientation='horizontal', ticks=ticks[c0]); cb.set_label(lab, fontsize=7)
            cb.ax.tick_params(labelsize=6, width=.4, length=2); cb.outline.set_linewidth(.4)
        save(fig, out, 'fig_qualitative_uncertainty')

    # ------------------------------------------------------------------ Fig. 4 error vs distance
    labels = ['1', '2–3', '4–7', '8–15', '≥16']
    fig, ax = plt.subplots(1, 2, figsize=(PAGE_W, 2.0), gridspec_kw=dict(wspace=.32))
    curves = [('coarse', 'Coarse', C_COARSE, 's')]
    if show_s1:
        curves.append(('S1_single', '$S=1$', C_S1, '^'))
    curves.append((mean_key, f'$S={S}$, mean of {K}', C_S200, 'o'))
    for k, lab, col, mk in curves:
        ax[0].plot(range(len(labels)), s['boundary_rmse'][k]['bld'][1:], marker=mk, color=col, label=lab)
    ax[0].set_xticks(range(len(labels)), labels); ax[0].set_xlabel('Distance to nearest building (pixels)')
    ax[0].set_ylabel('RMSE'); ax[0].grid(alpha=.25); ax[0].legend()
    panel_label(ax[0], '(a)')
    base = np.array(s['boundary_rmse']['coarse']['bld'][1:])
    for k, lab, col, mk in curves[1:]:
        rel = 100 * (1 - np.array(s['boundary_rmse'][k]['bld'][1:]) / base)
        ax[1].plot(range(len(labels)), rel, marker=mk, color=col, label=lab)
    ax[1].set_xticks(range(len(labels)), labels); ax[1].set_xlabel('Distance to nearest building (pixels)')
    ax[1].set_ylabel('RMSE reduction (%)'); ax[1].grid(alpha=.25); ax[1].legend()
    ax[1].axhline(0, color=C_GT, lw=.6)
    panel_label(ax[1], '(b)')
    save(fig, out, 'fig_error_vs_distance')
    print('paper figures ->', out, '| test samples =', n)


if __name__ == '__main__':
    main()
