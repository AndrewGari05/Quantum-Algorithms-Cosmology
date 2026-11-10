#!/usr/bin/env python3
"""Multi-seed figures of the noise axis (reads CSVs and npz only; runs nothing).

The noisy run of every cell is compared with its ``none-counts`` control, the
same cell read by counts without noise, never with ``none``: that would mix the
noise with the change of readout route. ``none`` is used only for the route
check (figure 4) and must equal the ideal multi-seed campaign row by row.

Figures (PNG at 200 dpi + PDF) written to ``--out``:

* ``ruido_desplazamiento``  shift of the posterior means against the noise level,
                            in control sigmas, per rung; one column per cell
                            (model, nqpp), one row per ladder; 1-sigma line; the
                            seeds above 1 sigma are labelled.
* ``ruido_sigma``           sigma(Om) of the noisy rung over sigma(Om) of the
                            control; last row: sigma(Om) of the variational rungs
                            over classical MCMC, with the ideal ratio of the
                            multi-seed campaign as reference.
* ``ruido_diagnosticos``    acceptance of the QMCMC ladder and final KL of the
                            QVMC ladder against the level, one panel per cell
                            (KL is never compared across nqpp).
* ``ruido_ruta``            effect of the readout route alone: none-counts
                            against none, per rung and cell.
* ``ruido_costo``           simulator wall time per level, rung and cell.
* ``ruido_corner_<model>_nqpp<N>_<qmcmc|qvmc>``  posteriors of QMCMC 100% and
                            QVMC 100% (+ quantum training) for none-counts,
                            readout and full, pooling the seeds (needs the npz).

plus ``noise_summary_seeds.csv`` and ``noise_report.txt`` (faithful-cell counts,
cells above 1 sigma, every number in the figures).

Usage:
    python seed_figures_noise.py RESULTS --out FIGS [--seeds 43-50] [--ideal RESULTS]

RESULTS holds ``noise_A_s<NN>`` / ``noise_B_s<NN>`` (one folder per seed);
``--ideal`` is the folder with ``seeds_s<NN>`` of the ideal multi-seed campaign
(same seeds), used for the reference ratio and the none = ideal check.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

import campaign_io  # noqa: E402
import cosmo_core as core  # noqa: E402
import seed_figures as sf  # noqa: E402

CONTROL = 'none-counts'
NOISY = ('readout', 'full')
LEVELS = (CONTROL,) + NOISY
LADDERS = (('QMCMC', ('Classical MCMC', 'QMCMC 50%', 'QMCMC 100%')),
           ('QVMC', ('Classical VI', 'QVMC 33%', 'QVMC 67%')))
QUANTUM = ('QMCMC 50%', 'QMCMC 100%', 'QVMC 33%', 'QVMC 67%')
#: colour, marker, filled: blue = classical, orange = half quantum, red = fully quantum.
STYLE = {'Classical MCMC': ('#1f77b4', 'o', False), 'QMCMC 50%': ('#ff7f0e', 's', True),
         'QMCMC 100%': ('#d62728', 'o', True), 'Classical VI': ('#1f77b4', 'D', False),
         'QVMC 33%': ('#ff7f0e', 's', True), 'QVMC 67%': ('#d62728', 'D', True)}
FIG_DPI = 200
SIGMA2D = (0.393, 0.865)
LEVEL_COLORS = {'none-counts': '0.35', 'readout': '#ff7f0e', 'full': '#d62728'}
COMPARE = ('_mean', '_std')


# ── reading ──────────────────────────────────────────────────────────────────
def load_noise(root, seeds=None, pattern='noise_*_s*'):
    """``data[(model, nqpp)][level][method][seed] = row`` and the task folders.

    Rows are read with `campaign_io.read_campaign` (level from the folder name;
    nqpp from the folder or, when the folder has no tag, from the rows).
    """
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    taskdirs = defaultdict(dict)
    for folder in sorted(glob.glob(os.path.join(root, pattern))):
        if not os.path.isdir(folder):
            continue
        for r in campaign_io.read_campaign(folder, warn=False):
            if r['_fam'] != 'samplers' or r['_g'] is None:
                continue
            seed = (r.get('seed') or '').strip()
            if seeds is not None and seed not in seeds:
                continue
            cell = (r['_mod'], r['_g'])
            data[cell][r['_noi']][r['Method']][seed] = r
            taskdirs[cell + (r['_noi'],)][seed] = os.path.dirname(os.path.dirname(r['_path']))
    return data, taskdirs


def cells_of(data):
    order = {m: i for i, m in enumerate(sf.MODEL_ORDER)}
    return sorted(data, key=lambda c: (order.get(c[0], 99), c[1]))


def seeds_of(data, cell):
    return sorted({s for lvl in data[cell].values() for by in lvl.values() for s in by}, key=int)


def params_of(row):
    return [c[:-5] for c in row if c.endswith('_mean')]


def shift(row, ref):
    """max over parameters of |mean(row) - mean(ref)| / sigma(ref)."""
    out = None
    for p in params_of(ref):
        a, b, s = sf.num(row, p + '_mean'), sf.num(ref, p + '_mean'), sf.num(ref, p + '_std')
        if None in (a, b, s) or not s:
            continue
        d = abs(a - b) / s
        out = d if out is None else max(out, d)
    return out


def per_seed(data, cell, level, method, fn, ref_level=CONTROL):
    """``fn(row, control_row)`` for every seed of the cell (None where missing)."""
    out = []
    for s in seeds_of(data, cell):
        r = data[cell].get(level, {}).get(method, {}).get(s)
        c = data[cell].get(ref_level, {}).get(method, {}).get(s)
        out.append(fn(r, c) if r is not None and c is not None else None)
    return out


def same_row(a, b, extra=('acceptance', 'final_KL', 'ESS')):
    cols = [c for c in a if c.endswith(COMPARE)] + list(extra)
    return all((a.get(c) or '') == (b.get(c) or '') for c in cols)


def label(method):
    return campaign_io.display_method(method)


def cell_title(data, cell):
    return f'{sf.short(cell[0])}, nqpp {cell[1]} (n = {len(seeds_of(data, cell))})'


def save(fig, out, name):
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, name)
    fig.savefig(path + '.png', dpi=FIG_DPI, bbox_inches='tight')
    fig.savefig(path + '.pdf', bbox_inches='tight')
    plt.close(fig)
    return path + '.png'


# ── drawing helpers ──────────────────────────────────────────────────────────
def _series(ax, xs, vals_by_x, method, off=0.0, label_seeds=None, seeds=None, log=False):
    """Seed dots (grey), mean ± sd across seeds, means joined; optional seed labels.

    On a log axis the lower bar is clipped at 90 % of the mean, so a spread larger
    than the mean does not run off the axis (the seed dots show the real range).
    """
    col, mk, fill = STYLE[method]
    pts = []
    for x, vals in zip(xs, vals_by_x):
        sf.draw_seeds(ax, x + off, vals, width=0.03)
        m, sd, _se, n = sf.stats(vals)
        if n:
            yerr = [[min(sd, 0.9 * m)], [sd]] if log and m > 0 else [sd]
            ax.errorbar([x + off], [m], yerr=yerr, fmt=mk, color=col, ms=6, capsize=3, lw=1.4,
                        mfc=col if fill else 'white', zorder=3)
            pts.append((x + off, m))
        if label_seeds is not None and seeds is not None:
            for s, v in zip(seeds, vals):
                if v is not None and v > label_seeds:
                    ax.annotate(f's{s}', (x + off, v), xytext=(4, 2), textcoords='offset points',
                                fontsize=7, color=col)
    if pts:
        ax.plot(*zip(*pts), color=col, lw=1.3, ls=':' if not fill else '-', zorder=2,
                marker=mk, mfc=col if fill else 'white', ms=6, label=label(method))


def _grid(nrows, ncols, w=3.3, h=3.4):
    fig, axes = plt.subplots(nrows, ncols, figsize=(w * ncols, h * nrows), squeeze=False)
    return fig, axes


def _ladder_legend(fig, axes_row, y):
    h, l = axes_row[0].get_legend_handles_labels()
    return h, l


def _footer(fig, text, y=0.0):
    fig.text(0.5, y, text, ha='center', va='top', fontsize=8, color='0.35')


# ── 1. shift ─────────────────────────────────────────────────────────────────
def fig_shift(data, out, report):
    cells = cells_of(data)
    fig, axes = _grid(2, len(cells))
    xs = list(range(len(LEVELS)))
    over = []
    for j, cell in enumerate(cells):
        seeds = seeds_of(data, cell)
        for i, (fam, ladder) in enumerate(LADDERS):
            ax = axes[i][j]
            rungs = [m for m in ladder if m not in ('Classical MCMC',)]
            for k, meth in enumerate(rungs):
                vals = [per_seed(data, cell, lvl, meth, shift) for lvl in LEVELS]
                _series(ax, xs, vals, meth, off=(k - (len(rungs) - 1) / 2) * 0.12,
                        label_seeds=1.0, seeds=seeds)
                for lvl, v in zip(LEVELS, vals):
                    for s, x in zip(seeds, v):
                        if x is not None and x > 1.0 and lvl in NOISY:
                            over.append((cell, lvl, meth, s, x))
            ax.axhline(1.0, color='k', ls='--', lw=0.9)
            ax.set_yscale('symlog', linthresh=0.01, linscale=0.4)
            ax.set_ylim(-0.002, max(6.0, ax.get_ylim()[1]))
            ax.set_xticks(xs)
            ax.set_xticklabels(LEVELS, fontsize=8)
            ax.grid(alpha=0.3)
            if i == 0:
                ax.set_title(cell_title(data, cell), fontsize=10)
            if j == 0:
                ax.set_ylabel(f'{fam}: max |Δp| / σ(control)')
    for i in range(2):
        h, l = axes[i][0].get_legend_handles_labels()
        axes[i][-1].legend(h, l, fontsize=7, loc='upper left', bbox_to_anchor=(1.02, 1.0),
                           frameon=False)
    _footer(fig, 'Shift of the posterior means of the noisy run relative to the none-counts control '
            '(same seed), maximum over parameters, in control standard deviations; symmetric-log axis, '
            'linear below 0.01. Classical MCMC is identical at every level and not shown; classical VI '
            'changes by design (its KL is evaluated on the noisy simulator). Seeds above 1σ are labelled.',
            y=-0.01)
    fig.suptitle('Noise axis: shift of the posterior mean against the noise level',
                 fontsize=13, fontweight='bold')
    fig.tight_layout(rect=(0, 0.02, 0.93, 0.95))
    report.append('\n== cells above 1 sigma (noisy rung vs none-counts control) ==')
    for cell, lvl, meth, s, x in sorted(over, key=lambda t: -t[4]):
        report.append(f'  {cell[0]:5s} nqpp{cell[1]} {lvl:8s} {label(meth):32s} seed {s}  {x:.3f}')
    return save(fig, out, 'ruido_desplazamiento'), over


# ── 2. width ─────────────────────────────────────────────────────────────────
def _ratio_ctrl(r, c):
    a, b = sf.num(r, 'Om_std'), sf.num(c, 'Om_std')
    return a / b if a is not None and b else None


def fig_sigma(data, ideal, out, report):
    cells = cells_of(data)
    fig, axes = _grid(3, len(cells))
    xs = list(range(len(LEVELS)))
    for j, cell in enumerate(cells):
        for i, (fam, ladder) in enumerate(LADDERS):
            ax = axes[i][j]
            rungs = [m for m in ladder if m != 'Classical MCMC']
            for k, meth in enumerate(rungs):
                vals = [per_seed(data, cell, lvl, meth, _ratio_ctrl) for lvl in LEVELS]
                _series(ax, xs, vals, meth, off=(k - (len(rungs) - 1) / 2) * 0.12)
            ax.axhline(1.0, color='k', ls='--', lw=0.9)
            ax.set_xticks(xs)
            ax.set_xticklabels(LEVELS, fontsize=8)
            ax.grid(alpha=0.3)
            if i == 0:
                ax.set_title(cell_title(data, cell), fontsize=10)
            if j == 0:
                ax.set_ylabel(f'{fam}: σ(Ωm) / σ(Ωm, control)')
        # last row: variational rungs relative to classical MCMC, with the ideal reference
        ax = axes[2][j]
        mc = lambda r, c: None  # noqa: E731
        for k, meth in enumerate(('Classical VI', 'QVMC 67%')):
            vals = []
            for lvl in LEVELS:
                v = []
                for s in seeds_of(data, cell):
                    r = data[cell][lvl].get(meth, {}).get(s)
                    m0 = data[cell][lvl].get('Classical MCMC', {}).get(s)
                    v.append(_ratio_ctrl(r, m0) if r is not None and m0 is not None else None)
                vals.append(v)
            _series(ax, xs, vals, meth, off=(k - 0.5) * 0.12)
        if ideal is not None and cell in ideal:
            ref = sf.sigma_ratio(ideal[cell], 'QVMC 67%', sorted(ideal[cell].get('QVMC 67%', {})))
            m, sd, _se, n = sf.stats(ref)
            if n:
                ax.axhspan(m - sd, m + sd, color=STYLE['QVMC 67%'][0], alpha=0.12, lw=0)
                ax.axhline(m, color=STYLE['QVMC 67%'][0], ls='-.', lw=1.0,
                           label=f'ideal campaign, QVMC 100% (n = {n})')
                report.append(f'  ideal reference {cell[0]} nqpp{cell[1]}: QVMC 100%/MCMC '
                              f'{sf.fmt(m, sd, _se, prec=3)}')
        ax.axhline(1.0, color='k', ls='--', lw=0.9)
        ax.set_xticks(xs)
        ax.set_xticklabels(LEVELS, fontsize=8)
        ax.grid(alpha=0.3)
        if j == 0:
            ax.set_ylabel('σ(Ωm) / σ(Ωm, classical MCMC)')
    for i in range(3):
        h, l = axes[i][0].get_legend_handles_labels()
        axes[i][-1].legend(h, l, fontsize=7, loc='upper left', bbox_to_anchor=(1.02, 1.0),
                           frameon=False)
    _footer(fig, 'Rows 1–2: width of the Ωm posterior of each rung relative to its none-counts control '
            '(same seed). Row 3: width relative to classical MCMC at the same level; dash-dotted line and '
            'band: ideal QVMC 100% / MCMC ratio of the multi-seed ideal campaign (mean ± sd). Noise turns '
            'the ideal under-dispersion into over-dispersion.', y=-0.01)
    fig.suptitle('Noise axis: width of the Ωm posterior', fontsize=13, fontweight='bold')
    fig.tight_layout(rect=(0, 0.02, 0.93, 0.95))
    return save(fig, out, 'ruido_sigma')


# ── 3. diagnostics ───────────────────────────────────────────────────────────
def _value(key):
    return lambda r, c: sf.num(r, key)


def fig_diagnostics(data, out):
    cells = cells_of(data)
    fig, axes = _grid(2, len(cells))
    xs = list(range(len(LEVELS)))
    for j, cell in enumerate(cells):
        for i, (key, ladder, ylab) in enumerate(
                (('acceptance', LADDERS[0][1], 'QMCMC: acceptance fraction'),
                 ('final_KL', LADDERS[1][1], 'QVMC: final KL'))):
            ax = axes[i][j]
            for k, meth in enumerate(ladder):
                vals = [per_seed(data, cell, lvl, meth, _value(key)) for lvl in LEVELS]
                _series(ax, xs, vals, meth, off=(k - 1) * 0.12)
            ax.set_xticks(xs)
            ax.set_xticklabels(LEVELS, fontsize=8)
            ax.grid(alpha=0.3)
            if i == 0:
                ax.set_title(cell_title(data, cell), fontsize=10)
            if j == 0:
                ax.set_ylabel(ylab)
    for i in range(2):
        h, l = axes[i][0].get_legend_handles_labels()
        axes[i][-1].legend(h, l, fontsize=7, loc='upper left', bbox_to_anchor=(1.02, 1.0),
                           frameon=False)
    _footer(fig, 'Top: MCMC acceptance (QMCMC 50% and 100% differ under noise: the noisy acceptance is a '
            'different sampler). Bottom: final KL of the variational rungs; classical VI changes by '
            'design because its KL is evaluated on the noisy simulator. KL values are compared only within '
            'a cell, never across nqpp.', y=-0.01)
    fig.suptitle('Noise axis: sampler diagnostics against the noise level', fontsize=13,
                 fontweight='bold')
    fig.tight_layout(rect=(0, 0.02, 0.93, 0.95))
    return save(fig, out, 'ruido_diagnosticos')


# ── 4. readout route ─────────────────────────────────────────────────────────
def fig_route(data, out, report):
    cells = cells_of(data)
    rungs = ('QMCMC 50%', 'QMCMC 100%', 'Classical VI', 'QVMC 33%', 'QVMC 67%')
    fig, ax = plt.subplots(figsize=(11, 4.6))
    report.append('\n== readout route: none-counts vs none, max |dp| / sigma(none) ==')
    for k, meth in enumerate(rungs):
        off = (k - (len(rungs) - 1) / 2) * 0.13
        vals = [per_seed(data, cell, CONTROL, meth, shift, ref_level='none') for cell in cells]
        _series(ax, list(range(len(cells))), vals, meth, off=off)
        for cell, v in zip(cells, vals):
            m, sd, _se, n = sf.stats(v)
            report.append(f'  {label(meth):32s} {cell[0]:5s} nqpp{cell[1]} {sf.fmt(m, sd, _se, prec=4)}'
                          f'  max {max(x for x in v if x is not None):.4f}')
    ax.axhline(0.05, color='k', ls='--', lw=0.9, label='0.05σ')
    ax.set_yscale('symlog', linthresh=1e-3, linscale=0.5)
    ax.set_ylim(-1e-4, 0.2)
    ax.set_xticks(range(len(cells)))
    ax.set_xticklabels([cell_title(data, c).replace(', ', '\n', 1) for c in cells], fontsize=8)
    ax.set_ylabel('max |Δp| / σ(none)')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1.01, 1.0), frameon=False)
    ax.set_title('Readout route alone (no noise): counts against amplitudes', fontsize=12,
                 fontweight='bold')
    _footer(fig, 'Shift of the none-counts run relative to the none run of the same seed. The QVMC rungs '
            'and classical VI give exactly 0 (shown on the linear part of the symmetric-log axis, below '
            '10⁻³); the QMCMC shift is the effect of reading the proposal by counts.', y=-0.02)
    fig.tight_layout()
    return save(fig, out, 'ruido_ruta')


# ── 5. cost ──────────────────────────────────────────────────────────────────
def fig_cost(data, out):
    cells = cells_of(data)
    fig, axes = _grid(1, len(cells), h=3.8)
    levels = ('none',) + LEVELS
    xs = list(range(len(levels)))
    for j, cell in enumerate(cells):
        ax = axes[0][j]
        for k, meth in enumerate(m for _f, lad in LADDERS for m in lad):
            vals = [per_seed(data, cell, lvl, meth, _value('Time_s')) for lvl in levels]
            _series(ax, xs, vals, meth, off=(k - 2.5) * 0.08, log=True)
        ax.set_yscale('log')
        ax.set_xticks(xs)
        ax.set_xticklabels(levels, fontsize=8, rotation=20)
        ax.grid(alpha=0.3, which='both')
        ax.set_title(cell_title(data, cell), fontsize=10)
        if j == 0:
            ax.set_ylabel('wall time per rung [s]')
    h, l = axes[0][0].get_legend_handles_labels()
    axes[0][-1].legend(h, l, fontsize=7, loc='upper left', bbox_to_anchor=(1.02, 1.0), frameon=False)
    fig.suptitle('Noise axis: simulator cost (density-matrix simulation on CPU, not hardware time)',
                 fontsize=12, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 0.93, 0.92))
    return save(fig, out, 'ruido_costo')


# ── 6. corners ───────────────────────────────────────────────────────────────
def _pooled(taskdirs, cell, level, seeds, key, wkey=None, jitter=None, rng=None):
    S, W = [], []
    for s in seeds:
        d = taskdirs.get(cell + (level,), {}).get(s)
        z = sf._npz(d, cell[0]) if d else None
        if z is None or key not in z:
            continue
        x = np.asarray(z[key], float)
        if jitter is not None:
            x = x + rng.uniform(-0.5, 0.5, x.shape) * jitter
        w = np.asarray(z[wkey], float) if wkey else np.ones(len(x))
        S.append(x)
        W.append(w / w.sum())
    if not S:
        return None, None
    return np.vstack(S), np.concatenate(W) / len(W)


def fig_corners(data, taskdirs, out, report):
    import corner
    made = []
    rng = np.random.default_rng(2026)
    for cell in cells_of(data):
        model = core.MODELS[cell[0]]
        names = list(model.param_names)
        seeds = seeds_of(data, cell)
        width = np.array([(hi - lo) / (2 ** cell[1] - 1) for lo, hi in model.sample_box])
        for tag, key, wkey, meth in (('qmcmc', 'qmcmc_100_flat', None, 'QMCMC 100%'),
                                     ('qvmc', 'qvmc_67_S', 'qvmc_67_W', 'QVMC 67%')):
            sets = []
            for lvl in LEVELS:
                S, W = _pooled(taskdirs, cell, lvl, seeds, key, wkey,
                               width if wkey else None, rng)
                if S is not None:
                    sets.append((lvl, S[:, :len(names)], W))
            if len(sets) < 2:
                continue
            allx = np.vstack([s[1] for s in sets])
            rngs = [tuple(np.percentile(allx[:, i], [0.5, 99.5])) for i in range(len(names))]
            fig = None
            for lvl, S, W in sets:
                fig = corner.corner(S, weights=W, fig=fig, color=LEVEL_COLORS[lvl], bins=35,
                                    range=rngs, smooth=1.0, levels=SIGMA2D, plot_datapoints=False,
                                    plot_density=False, fill_contours=False,
                                    labels=[sf.tex(model, p) for p in names],
                                    hist_kwargs=dict(density=True, lw=1.5),
                                    contour_kwargs=dict(linewidths=1.5))
            fig.set_size_inches(2.6 * len(names) + 1.2, 2.6 * len(names) + 0.6)
            fig.legend(handles=[Line2D([0], [0], color=LEVEL_COLORS[l], lw=2,
                                       label=l + (' (control)' if l == CONTROL else ''))
                                for l, _s, _w in sets],
                       loc='upper right', bbox_to_anchor=(0.97, 0.80), fontsize=9,
                       frameon=False, title='noise level')
            fig.suptitle(f'{label(meth)}\n{cell_title(data, cell)}, seeds pooled',
                         fontsize=11, y=1.04)
            made.append(save(fig, out, f'ruido_corner_{cell[0]}_nqpp{cell[1]}_{tag}'))
    report.append(f'\n{len(made)} corner figures')
    return made


# ── 7. summary table and checks ──────────────────────────────────────────────
def checks(data, ideal, report):
    """Faithful-cell counts and the none = ideal campaign check."""
    n_mc = ok_mc = n_vi = ch_vi = 0
    same50 = defaultdict(lambda: [0, 0])
    kl33 = defaultdict(lambda: [0, 0])
    n_id = ok_id = 0
    for cell in cells_of(data):
        for s in seeds_of(data, cell):
            base = {m: data[cell]['none'].get(m, {}).get(s) for m in STYLE}
            for lvl in LEVELS:
                rows = {m: data[cell][lvl].get(m, {}).get(s) for m in STYLE}
                if rows['Classical MCMC'] and base['Classical MCMC']:
                    n_mc += 1
                    ok_mc += same_row(rows['Classical MCMC'], base['Classical MCMC'])
                if lvl in NOISY and rows['Classical VI'] and base['Classical VI']:
                    n_vi += 1
                    ch_vi += not same_row(rows['Classical VI'], base['Classical VI'])
            for lvl in ('none',) + LEVELS:
                rows = data[cell][lvl]
                a, b = rows.get('QMCMC 50%', {}).get(s), rows.get('QMCMC 100%', {}).get(s)
                if a is not None and b is not None:
                    same50[lvl][1] += 1
                    same50[lvl][0] += same_row(a, b, extra=('acceptance',))
                a, b = rows.get('QVMC 33%', {}).get(s), rows.get('Classical VI', {}).get(s)
                if a is not None and b is not None:
                    kl33[lvl][1] += 1
                    kl33[lvl][0] += (a.get('final_KL') or '') == (b.get('final_KL') or '')
            if ideal is not None and cell in ideal:
                for m, by in data[cell]['none'].items():
                    r, i = by.get(s), ideal[cell].get(m, {}).get(s)
                    if r is not None and i is not None:
                        n_id += 1
                        ok_id += same_row(r, i)
    report.append('== faithful cells ==')
    report.append(f'  classical MCMC identical to none at every level: {ok_mc}/{n_mc}')
    report.append(f'  classical VI changes with readout / full noise: {ch_vi}/{n_vi} '
                  '(by design: its KL is evaluated on the noisy simulator)')
    for lvl in ('none',) + LEVELS:
        a, n = same50[lvl]
        report.append(f'  QMCMC 50% == QMCMC 100% at {lvl:11s}: {a}/{n}')
    for lvl in ('none',) + LEVELS:
        a, n = kl33[lvl]
        report.append(f'  final KL QVMC 50% == classical VI at {lvl:11s}: {a}/{n}')
    if ideal is not None:
        report.append(f'  none rows identical to the ideal multi-seed campaign (all but time): '
                      f'{ok_id}/{n_id}')
    return dict(mcmc=(ok_mc, n_mc), vi=(ch_vi, n_vi), same50=dict(same50), ideal=(ok_id, n_id))


def summary_rows(data):
    rows = []
    for cell in cells_of(data):
        for lvl in NOISY:
            for meth in STYLE:
                sh = per_seed(data, cell, lvl, meth, shift)
                wr = per_seed(data, cell, lvl, meth, _ratio_ctrl)
                acc = per_seed(data, cell, lvl, meth,
                               lambda r, c: (sf.num(r, 'acceptance') - sf.num(c, 'acceptance'))
                               if sf.num(r, 'acceptance') is not None
                               and sf.num(c, 'acceptance') is not None else None)
                kl = per_seed(data, cell, lvl, meth,
                              lambda r, c: (sf.num(r, 'final_KL') - sf.num(c, 'final_KL'))
                              if sf.num(r, 'final_KL') is not None
                              and sf.num(c, 'final_KL') is not None else None)
                t = per_seed(data, cell, lvl, meth, _value('Time_s'))
                row = dict(model=cell[0], nqpp=cell[1], level=lvl, method=meth,
                           display=label(meth), n_seeds=sum(v is not None for v in sh))
                for name, v in (('shift', sh), ('sigma_ratio', wr), ('acceptance_diff', acc),
                                ('final_KL_diff', kl), ('time_s', t)):
                    m, sd, _se, n = sf.stats(v)
                    row[name + '_mean'] = '' if m is None else f'{m:.6g}'
                    row[name + '_sd'] = '' if sd is None or n < 2 else f'{sd:.6g}'
                    vv = [x for x in v if x is not None]
                    row[name + '_max'] = f'{max(vv):.6g}' if vv else ''
                rows.append(row)
    return rows


def write_csv(rows, path):
    with open(path, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def table_report(rows, report):
    report.append('\n== vs none-counts control: mean ± sd over seeds ==')
    for r in rows:
        if r['method'] == 'Classical MCMC':
            continue
        txt = (f'  {r["model"]:5s} nqpp{r["nqpp"]} {r["level"]:8s} {r["display"]:32s} n={r["n_seeds"]} '
               f'shift {r["shift_mean"]}±{r["shift_sd"]} (max {r["shift_max"]}) '
               f'σ ratio {r["sigma_ratio_mean"]}±{r["sigma_ratio_sd"]}')
        if r['acceptance_diff_mean']:
            txt += f' Δacc {r["acceptance_diff_mean"]}'
        if r['final_KL_diff_mean']:
            txt += f' ΔKL {r["final_KL_diff_mean"]}±{r["final_KL_diff_sd"]}'
        report.append(txt)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('root', help='folder with noise_A_s<NN> / noise_B_s<NN>')
    ap.add_argument('--out', required=True)
    ap.add_argument('--seeds', default=None)
    ap.add_argument('--ideal', default=None,
                    help='folder with seeds_s<NN> of the ideal multi-seed campaign')
    ap.add_argument('--no-corners', action='store_true')
    a = ap.parse_args(argv)
    seeds = sf.parse_seeds(a.seeds)
    data, taskdirs = load_noise(a.root, seeds)
    if not data:
        print('no noise rows found', file=sys.stderr)
        return 1
    ideal = sf.load_campaign(a.ideal, seeds)[0] if a.ideal else None
    report = [f'seed_figures_noise: {len(cells_of(data))} cells: ' +
              ', '.join(cell_title(data, c) for c in cells_of(data))]
    checks(data, ideal, report)
    files = []
    f, _over = fig_shift(data, a.out, report)
    files.append(f)
    report.append('\n== width: ideal reference ==')
    files.append(fig_sigma(data, ideal, a.out, report))
    files.append(fig_diagnostics(data, a.out))
    files.append(fig_route(data, a.out, report))
    files.append(fig_cost(data, a.out))
    if not a.no_corners:
        files += fig_corners(data, taskdirs, a.out, report)
    rows = summary_rows(data)
    write_csv(rows, os.path.join(a.out, 'noise_summary_seeds.csv'))
    table_report(rows, report)
    text = '\n'.join(report) + '\n'
    with open(os.path.join(a.out, 'noise_report.txt'), 'w', encoding='utf-8') as fh:
        fh.write(text)
    print(text)
    print(f'{len(files)} figures + noise_summary_seeds.csv + noise_report.txt in {a.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
