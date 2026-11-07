"""One composite figure of the ibm_fez runs (reads results.csv only; runs nothing).

Usage:
    python hw_summary.py RUN_DIR [RUN_DIR ...] --out FOLDER

Each RUN_DIR is one seed written by `python -m thesis hardware`. Panels:
(a) QVMC width of the Omega_m posterior relative to the reference, before
training (warm start, sampling noise only) and after SPSA; (b) QMCMC shift of
the posterior mean in reference sigmas, with the 2 x MCSE band; (c) final
exact KL of QVMC; (d) QGA best chi2 minus the grid floor. Bars = mean over
seeds, error bars = sample sd (n >= 3) or range (n = 2), dots = seeds.
"""
import argparse
import csv
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

C_CLASSICAL, C_GENETIC = '#1f77b4', '#2ca02c'
LOCS = ['classical', 'ideal', 'noisy', 'device']
LABEL = {'classical': 'classical', 'ideal': 'ideal\nsimulator', 'noisy': 'noisy\ntwin',
         'device': 'device\n(ibm_fez)'}
FIG_DPI = 200


def warm(loc):
    return plt.cm.autumn_r({'ideal': 0.30, 'noisy': 0.62, 'device': 0.95}[loc])


def col(loc, alg='x'):
    if loc == 'classical':
        return C_GENETIC if alg == 'genetic' else C_CLASSICAL
    return warm(loc)


def f(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def err(v):
    v = [x for x in v if x is not None]
    if len(v) < 2:
        return 0.0
    return np.std(v, ddof=1) if len(v) >= 3 else (max(v) - min(v)) / 2


def bar(ax, x, vals, color, width=0.6, hatch=None, fill=True):
    v = [y for y in vals if y is not None]
    if not v:
        return
    m = np.mean(v)
    ax.bar(x, m, width, color=color if fill else 'white', edgecolor=color if not fill else 'k',
           lw=1.2 if not fill else 0.6, hatch=hatch)
    ax.errorbar(x, m, yerr=err(v), color='k', capsize=4, lw=1.2)
    ax.scatter(np.full(len(v), x) + np.linspace(-0.08, 0.08, len(v)), v, s=22, facecolor='white',
               edgecolor='k', zorder=4)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='+')
    ap.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    runs = {}
    for d in a.runs:
        rows = list(csv.DictReader(open(os.path.join(d, 'results.csv'))))
        runs[rows[0]['seed']] = {(r['algorithm'], r['location']): r for r in rows}
    seeds = sorted(runs, key=int)
    get = lambda alg, loc, key: [f(runs[s].get((alg, loc), {}).get(key)) for s in seeds]
    rule = 'sd' if len(seeds) >= 3 else 'range'

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.6))
    (ax, bx), (cx, dx) = axes
    # (a) QVMC width
    for i, loc in enumerate(LOCS[1:]):
        bar(ax, i - 0.18, get('vi', loc, 'start_width_ratio_Om'), col(loc), width=0.34, fill=False)
        bar(ax, i + 0.18, get('vi', loc, 'width_ratio_Om'), col(loc), width=0.34)
    ax.axhline(1, color='k', ls='--', lw=0.9)
    ax.set_xticks(range(3), [LABEL[l] for l in LOCS[1:]])
    ax.set_ylabel(r'$\sigma(\Omega_m)\,/\,\sigma_{\rm ref}$')
    ax.set_title(r'(a) QVMC: width of the $\Omega_m$ posterior', fontsize=11)
    ax.legend(handles=[Patch(facecolor='white', edgecolor='0.3', label='before training (sampling only)'),
                       Patch(facecolor='0.6', edgecolor='k', label='after SPSA')],
              fontsize=8, loc='upper left')
    # (b) QMCMC shift
    mcse = [x for loc in LOCS for x in get('mcmc', loc, 'shift_mcse_sigma') if x is not None]
    band = 2 * float(np.mean(mcse)) if mcse else None
    if band:
        bx.axhspan(-band, band, color='0.88', lw=0)
    for i, loc in enumerate(LOCS):
        for dxx, p, mk in ((-0.15, 'Om', 'o'), (0.15, 'H0', 's')):
            v = [x for x in get('mcmc', loc, f'shift_{p}_sigma') if x is not None]
            if not v:
                continue
            bx.errorbar(i + dxx, np.mean(v), yerr=err(v), fmt=mk, color=col(loc), ms=7, capsize=4,
                        mec='k')
            bx.scatter(np.full(len(v), i + dxx + 0.06), v, s=12, color='0.5', zorder=3)
    bx.axhline(0, color='k', lw=0.8)
    bx.set_xticks(range(4), [LABEL[l] for l in LOCS])
    bx.set_ylabel(r'shift of the mean $/\ \sigma_{\rm ref}$')
    bx.set_title('(b) QMCMC: bias of the posterior mean', fontsize=11)
    bx.legend(handles=[Line2D([], [], marker='o', ls='', color='0.4', mec='k', label=r'$\Omega_m$'),
                       Line2D([], [], marker='s', ls='', color='0.4', mec='k', label=r'$H_0$')]
              + ([Patch(color='0.88', label=f'±2 MCSE (≈ ±{band:.2f}σ): not distinguishable from 0')]
                 if band else []), fontsize=8, loc='upper left')
    lim = max(0.15, 1.3 * max(abs(x) for loc in LOCS for p in ('Om', 'H0')
                              for x in get('mcmc', loc, f'shift_{p}_sigma') if x is not None))
    bx.set_ylim(-lim, lim)
    # (c) QVMC KL
    for i, loc in enumerate(LOCS[1:]):
        bar(cx, i, get('vi', loc, 'kl_final_exact'), col(loc))
    k0 = [x for x in get('vi', 'ideal', 'kl_start_exact') if x is not None]
    if k0:
        cx.axhline(np.mean(k0), color='k', ls='--', lw=1, label=f'warm start (mean {np.mean(k0):.3f})')
        cx.legend(fontsize=8, loc='upper left')
    cx.set_xticks(range(3), [LABEL[l] for l in LOCS[1:]])
    cx.set_ylabel(r'final KL$(Q_\varphi\,\|\,P_{\rm target})$')
    cx.set_title('(c) QVMC: final KL', fontsize=11)
    # (d) QGA distance to the floor
    ticks = []
    for i, loc in enumerate(LOCS):
        best = get('genetic', loc, 'chi2_estimate')
        floor = get('genetic', loc, 'chi2_grid_floor')
        v = [b - fl if None not in (b, fl) else None for b, fl in zip(best, floor)]
        bar(dx, i, v, col(loc, 'genetic'))
        reached = sum(1 for x in v if x is not None and x <= 1e-9)
        ticks.append(f'{LABEL[loc]}\n[optimum {reached}/{len(seeds)}]')
    dx.axhline(0, color='k', ls='--', lw=0.9)
    dx.set_xticks(range(4), ticks, fontsize=9)
    dx.set_ylabel(r'$\chi^2_{\rm best} - \chi^2_{\rm floor}$')
    dx.set_title('(d) QGA: distance to the grid optimum', fontsize=11)
    for x in axes.ravel():
        x.grid(alpha=0.3, axis='y')
    fig.legend(handles=[Line2D([], [], marker='o', ls='', mfc='white', mec='k',
                               label='individual seeds (' + ', '.join(seeds) + ')'),
                        Line2D([], [], color='k', label=f'error bar: {rule} across seeds')],
               loc='lower center', ncol=2, fontsize=9, frameon=False)
    fig.suptitle(r'$\Lambda$CDM on ibm_fez', fontsize=13, fontweight='bold')
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    os.makedirs(a.out, exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(a.out, f'resumen_hardware.{ext}'), dpi=FIG_DPI, bbox_inches='tight')
    print('written', os.path.join(a.out, 'resumen_hardware.png'))


if __name__ == '__main__':
    main()
