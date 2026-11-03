#!/usr/bin/env python3
"""Multi-seed figures: the ladder figures aggregated over seeds.

Reads a multi-seed campaign (one folder per seed, ``seeds_s<NN>``, each with
the usual ``samplers_<model>_nqpp<N>`` tasks) and draws the ladder figures
with the seed scatter on them:

* point = mean over seeds, bar = standard deviation across seeds; the
  standard error (sd / sqrt(n)) is in the printed report and in
  ``ladder_summary.csv``;
* every seed is also drawn as a faint grey point, so outliers stay visible;
* traces (KL, R-hat) come from the task logs (`histories_from_log`) and are
  shown as the median over seeds with a 16-84 percentile band.

Only analysis and drawing: no number is recomputed by a sampler. The rows
are read with `compare_seeds.load_rows` (per-model CSV, or the cumulative
file with a warning); labels come from `campaign_io.display_method` and
`cosmo_modular_quantum.qvmc_rung_label`; the normalization rung
(`QVMC 100%`, equal to `QVMC 67%` by construction) is never drawn.

Output layout (PNG + PDF)::

    <out>/sigma_vs_grid, qmcmc_efficiency, om_bias, paired_kl,
          modelos_aic_bic, ladder_summary.csv, seed_figures_report.txt
    <out>/<model>/ladder_trends_<model>_nqpp<N>, ladder_summary_<model>,
                  ladder_kl_qvmc_<model>_nqpp<N>, ladder_rhat_qmcmc_<model>,
                  corner_ladder_qmcmc_<model>, corner_ladder_qvmc_<model>_nqpp<N>

Usage:
    python seed_figures.py RESULTS_DIR --out FIGURES_DIR [--seeds 43-50] [--no-corner]

``RESULTS_DIR`` holds the ``seeds_s<NN>`` folders. Do not mix campaigns run
with different code (for example seed 42, pre-errata, with 43-50): pass
``--seeds`` to choose.
"""
from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import re
import sys
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

import campaign_io  # noqa: E402
import compare_seeds  # noqa: E402
import cosmo_core as core  # noqa: E402
import cosmo_modular_quantum as cmq  # noqa: E402

#: Rungs in ladder order: (CSV Method, family, stored per-method %).
RUNGS = [('Classical MCMC', 'qmcmc', 0), ('QMCMC 50%', 'qmcmc', 50),
         ('QMCMC 100%', 'qmcmc', 100), ('Classical VI', 'qvmc', 0),
         ('QVMC 33%', 'qvmc', 33), ('QVMC 67%', 'qvmc', 67),
         ('QVMC 100%', 'qvmc', 100)]
FAMILY = {m: f for m, f, _ in RUNGS}
STORED_PCT = {m: p for m, _, p in RUNGS}
QMCMC_RUNGS = [m for m, f, _ in RUNGS if f == 'qmcmc']
QVMC_RUNGS = [m for m, f, _ in RUNGS if f == 'qvmc' and not campaign_io.is_hidden_method(m)]
SHOWN = QMCMC_RUNGS + QVMC_RUNGS
#: Quantum VI rung compared with Classical VI (the training rung).
QVMC_REF = 'QVMC 67%'
MODEL_ORDER = ['lcdm', 'pede', 'wcdm', 'gede', 'cpl']
MODEL_COLORS = {'lcdm': '#1f77b4', 'pede': '#2ca02c', 'wcdm': '#9467bd',
                'gede': '#8c564b', 'cpl': '#d62728'}
SEED_GREY = '0.55'


# ── reading ──────────────────────────────────────────────────────────────────
def parse_seeds(spec):
    """'43-50' or '43,45,47' -> set of seed strings (None -> all).

    Examples:
        >>> sorted(parse_seeds('43-45'))
        ['43', '44', '45']
        >>> sorted(parse_seeds('43,50'))
        ['43', '50']
        >>> parse_seeds(None) is None
        True
    """
    if not spec:
        return None
    out = set()
    for part in spec.split(','):
        if '-' in part:
            a, b = part.split('-')
            out |= {str(s) for s in range(int(a), int(b) + 1)}
        elif part.strip():
            out.add(part.strip())
    return out


def display_pct(method):
    """Position of a rung on its per-method axis (QVMC on the display scale).

    Examples:
        >>> [display_pct(m) for m in ('Classical MCMC', 'QMCMC 50%', 'QVMC 33%', 'QVMC 67%')]
        [0.0, 50.0, 50.0, 100.0]
    """
    p = float(STORED_PCT[method])
    return cmq.qvmc_display_pct(p) if FAMILY[method] == 'qvmc' else p


def num(row, key):
    """Float value of a CSV cell, or None."""
    return compare_seeds._num(row.get(key)) if row else None


def load_campaign(root, seeds=None, pattern='seeds_s*', grid='nqpp'):
    """Rows of a multi-seed campaign, indexed for the figures.

    Args:
        root: folder holding one sub-folder per seed.
        seeds: set of seed strings to keep (None = all).
        pattern: glob of the per-seed folders (``genetic_s*`` for the
            genetic campaign).
        grid: grid tag of the task folders, ``nqpp`` (samplers) or ``nb``
            (genetic n_bits).

    Returns:
        ``(data, taskdirs)``. ``data[(model, grid)][method][seed] = row``;
        ``taskdirs[(model, grid)][seed]`` is the task folder (for logs, npz).
    """
    data = defaultdict(lambda: defaultdict(dict))
    taskdirs = defaultdict(dict)
    for folder in sorted(glob.glob(os.path.join(root, pattern))):
        if not os.path.isdir(folder):
            continue
        cells, folder_seeds, _ = compare_seeds.load_rows([folder])
        if len(folder_seeds) != 1:
            print(f'WARNING: {folder} holds rows of {len(folder_seeds)} seeds '
                  f'({sorted(folder_seeds)}); skipped.', file=sys.stderr)
            continue
        seed = next(iter(folder_seeds))
        if seeds is not None and seed not in seeds:
            continue
        for (task, model, _ds, noise, method), by_seed in cells.items():
            if noise not in ('none', ''):
                continue
            g = compare_seeds.grid_of(task)
            if not re.fullmatch(grid + r'\d+', g):
                continue
            n = int(g[len(grid):])
            data[(model, n)][method][seed] = by_seed[seed]
            taskdirs[(model, n)][seed] = os.path.join(folder, task.split('/')[0])
    return data, taskdirs


def stats(values):
    """``(mean, sd, se, n)`` of the usable values (sd across seeds)."""
    m, sd, n = compare_seeds.dispersion([v for v in values if v is not None])
    if n == 0:
        return None, None, None, 0
    return m, sd, sd / math.sqrt(n), n


def fmt(m, sd, se=None, prec=4):
    """'m ± sd' (and '(se x)') with a fixed precision."""
    if m is None:
        return '—'
    s = f'{m:.{prec}f} ± {sd:.{prec}f}'
    return s + (f'  (se {se:.{prec}f})' if se is not None else '')


def seeds_of(data):
    """Sorted list of the seeds present."""
    return sorted({s for cell in data.values() for rows in cell.values() for s in rows},
                  key=int)


def models_and_grids(data):
    """``{model: [nqpp, ...]}`` in thesis order."""
    out = defaultdict(list)
    for (m, n) in data:
        out[m].append(n)
    return {m: sorted(out[m]) for m in MODEL_ORDER if m in out} | \
           {m: sorted(v) for m, v in out.items() if m not in MODEL_ORDER}


def per_seed(cell, method, getter, seeds):
    """``[getter(row) or None for seed in seeds]`` for one rung of one cell."""
    rows = cell.get(method, {})
    return [getter(rows[s]) if s in rows else None for s in seeds]


def sigma_ratio(cell, method, seeds, p='Om'):
    """Per-seed sigma(method)/sigma(Classical MCMC) of parameter ``p``."""
    out = []
    for s in seeds:
        a = num(cell.get(method, {}).get(s), f'{p}_std')
        b = num(cell.get('Classical MCMC', {}).get(s), f'{p}_std')
        out.append(a / b if a is not None and b else None)
    return out


def om_shift(cell, method, seeds, p='Om'):
    """Per-seed (mean(method) - mean(MCMC)) / sigma(MCMC) of ``p``."""
    out = []
    for s in seeds:
        r, c = cell.get(method, {}).get(s), cell.get('Classical MCMC', {}).get(s)
        a, b, sc = num(r, f'{p}_mean'), num(c, f'{p}_mean'), num(c, f'{p}_std')
        out.append((a - b) / sc if None not in (a, b, sc) and sc else None)
    return out


def rhat_flags(data, model, seeds):
    """``{method: [(seed, rhat), ...]}`` of MCMC-family rows with R-hat above
    the threshold, over all grids of ``model`` (rows repeat across grids)."""
    thr = core.RHAT_THRESHOLD
    bad = defaultdict(dict)
    for (m, n), cell in data.items():
        if m != model:
            continue
        for meth in QMCMC_RUNGS:
            for s, r in cell.get(meth, {}).items():
                v = num(r, 'rhat')
                if v is not None and v >= thr:
                    bad[meth][s] = max(v, bad[meth].get(s, 0))
    return {k: sorted(v.items(), key=lambda t: int(t[0])) for k, v in bad.items()}


def rhat_note(data, model, seeds):
    """One-line convergence warning for figures of ``model`` ('' if none)."""
    bad = rhat_flags(data, model, seeds)
    if not bad:
        return ''
    parts = []
    for meth in QMCMC_RUNGS:
        if meth in bad:
            mx = max(v for _, v in bad[meth])
            parts.append(f'{meth} {len(bad[meth])}/{len(seeds)} seeds (max {mx:.3f})')
    return (rf'NOT CONVERGED: $\hat R \geq {core.RHAT_THRESHOLD:g}$ in ' + '; '.join(parts)
            + ' — MCMC-family numbers of this model carry that caveat')


def mcmc_rows_identical(data, model, seeds, cols=('Om_mean', 'Om_std', 'ESS', 'acceptance')):
    """True if the MCMC-family rows of ``model`` are equal on every grid."""
    grids = sorted(n for (m, n) in data if m == model)
    for meth in QMCMC_RUNGS:
        for s in seeds:
            vals = {tuple(data[(model, n)].get(meth, {}).get(s, {}).get(c) for c in cols)
                    for n in grids}
            if len(vals) > 1:
                return False
    return True


# ── drawing helpers ──────────────────────────────────────────────────────────
def save(fig, path):
    """PNG + PDF via the repo helper (creates the folder)."""
    cmq._save_fig(fig, path + '.png')
    return path + '.png'


def draw_seeds(ax, x, vals, width=0.0, rng=None, zorder=1):
    """Faint grey points, one per seed (jittered horizontally by ``width``)."""
    v = [y for y in vals if y is not None]
    if not v:
        return
    if rng is None:
        rng = np.random.default_rng(0)
    xs = x + (rng.uniform(-width, width, len(v)) if width else np.zeros(len(v)))
    ax.scatter(xs, v, s=11, color=SEED_GREY, alpha=0.55, lw=0, zorder=zorder)


def mean_bar(ax, x, vals, color, marker='o', label=None, ls='none', ms=6, mfc=None):
    """Mean ± sd across seeds at ``x``; returns the mean."""
    m, sd, _se, n = stats(vals)
    if n == 0:
        return None
    ax.errorbar([x], [m], yerr=[sd], fmt=marker, color=color, ms=ms, capsize=4,
                lw=1.6, ls=ls, label=label, mfc=mfc if mfc else color, zorder=3)
    return m


def series(ax, xs, val_lists, color, marker, label, ls='-', mfc=None, jitter=1.2):
    """A ladder series: seed points, mean ± sd, means joined by a line."""
    means = []
    for x, vals in zip(xs, val_lists):
        draw_seeds(ax, x, vals, width=jitter)
        m, sd, _se, n = stats(vals)
        means.append(m)
        if n:
            ax.errorbar([x], [m], yerr=[sd], fmt=marker, color=color, ms=6,
                        capsize=4, lw=1.6, mfc=mfc or color, zorder=3)
    pts = [(x, m) for x, m in zip(xs, means) if m is not None]
    if pts:
        ax.plot(*zip(*pts), ls=ls, color=color, marker=marker, lw=1.8,
                mfc=mfc or color, label=label, zorder=2)


def seed_caption(seeds):
    """Caption fragment with the seed set."""
    return (f'{len(seeds)} seeds ({seeds[0]}–{seeds[-1]}): point = mean, '
            'bar = sd across seeds, grey = individual seeds')


# ── figures ──────────────────────────────────────────────────────────────────
def fig_trends(data, model_name, n, seeds, outdir):
    """ladder_trends_<model>_nqpp<N>: every scalar along both ladders."""
    model = core.MODELS[model_name]
    cell = data[(model_name, n)]
    pn = model.param_names
    panels = [(p, f'{p}_mean', f'{p} (posterior mean)', model.fiducial[i])
              for i, p in enumerate(pn)]
    panels += [('sigma', f'{pn[0]}_std', f'σ({pn[0]}) (posterior sd)', None),
               ('time', 'Time_s', 'runtime [s]', None),
               ('chi2_red', 'chi2_red', r'$\chi^2_\nu$', None),
               ('AIC', 'AIC', 'AIC', None), ('BIC', 'BIC', 'BIC', None),
               ('ESS', 'ESS', 'ESS', None),
               ('acc', 'acceptance', 'acceptance (QMCMC)', None),
               ('kl', 'final_KL', 'final KL (QVMC)', None)]
    ncols = 4
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.9 * nrows))
    axes = axes.flatten()
    mx = [display_pct(m) - 3 for m in QMCMC_RUNGS]
    vx = [display_pct(m) + 3 for m in QVMC_RUNGS]
    for ax, (tag, col, ylab, fid) in zip(axes, panels):
        getter = (lambda r, c=col: num(r, c))
        if tag != 'kl':
            series(ax, mx, [per_seed(cell, m, getter, seeds) for m in QMCMC_RUNGS],
                   cmq.C_QUANTUM, 'o', 'QMCMC')
        if tag != 'acc':
            series(ax, vx, [per_seed(cell, m, getter, seeds) for m in QVMC_RUNGS],
                   cmq.C_QUANTUM2, 's', 'QVMC')
        if fid is not None:
            ax.axhline(fid, color='k', ls='--', lw=0.9, label='fiducial')
        if tag in pn:
            # keep the seed scatter readable: references are drawn, but the
            # y range stays on the data (a wide Planck band would flatten it)
            lim = ax.get_ylim()
            cmq._draw_ref_lines(ax, tag, axis='y', band=True)
            ax.set_ylim(*lim)
        ax.set_xticks([0, 50, 100])
        ax.set_xlabel('quantumness %  (per method)')
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    for ax in axes[len(panels):]:
        ax.axis('off')
    note = rhat_note(data, model_name, seeds)
    fig.suptitle(f'{model.label} — ladder trends over seeds, nqpp = {n}\n'
                 f'{seed_caption(seeds)}.  QMCMC red (0 % = classical MCMC), '
                 'QVMC orange (0 % = classical VI, 50 % = quantum sampling, '
                 '100 % = + quantum training)'
                 + (f'\n{note}' if note else ''),
                 fontsize=12, fontweight='bold', y=1.0,
                 color='firebrick' if note else 'k')
    fig.tight_layout()
    return save(fig, os.path.join(outdir, model_name, f'ladder_trends_{model_name}_nqpp{n}'))



def summary_rows(data, seeds):
    """Long table: one line per (model, nqpp, rung, metric) with mean/sd/se/n."""
    out = []
    for model_name, grids in models_and_grids(data).items():
        model = core.MODELS[model_name]
        metrics = ([f'{p}_mean' for p in model.param_names]
                   + [f'{p}_std' for p in model.param_names]
                   + ['chi2', 'AIC', 'BIC', 'ESS', 'acceptance', 'final_KL', 'Time_s', 'rhat'])
        for n in grids:
            cell = data[(model_name, n)]
            for meth in SHOWN:
                if meth not in cell:
                    continue
                for met in metrics:
                    m, sd, se, k = stats(per_seed(cell, meth, lambda r, c=met: num(r, c), seeds))
                    if k:
                        out.append(dict(model=model_name, nqpp=n, method=meth,
                                        display=campaign_io.display_method(meth),
                                        metric=met, mean=m, sd=sd, se=se, n=k))
        # sigma ratios are part of the summary too
        for n in grids:
            cell = data[(model_name, n)]
            for meth in SHOWN[1:]:
                m, sd, se, k = stats(sigma_ratio(cell, meth, seeds))
                if k:
                    out.append(dict(model=model_name, nqpp=n, method=meth,
                                    display=campaign_io.display_method(meth),
                                    metric='Om_std_ratio_vs_MCMC', mean=m, sd=sd, se=se, n=k))
    return out


def write_summary_csv(rows, path):
    """ladder_summary.csv (long format)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, ['model', 'nqpp', 'method', 'display', 'metric',
                                'mean', 'sd', 'se', 'n'])
        w.writeheader()
        for r in rows:
            w.writerow({**r, **{k: f'{r[k]:.6g}' for k in ('mean', 'sd', 'se')}})
    return path


def fig_summary(data, model_name, seeds, outdir):
    """ladder_summary_<model>: table of mean ± sd per rung and grid."""
    model = core.MODELS[model_name]
    pn = model.param_names
    grids = models_and_grids(data)[model_name]

    def cellstr(cell, meth, col, prec):
        m, sd, _se, k = stats(per_seed(cell, meth, lambda r: num(r, col), seeds))
        return '—' if not k else f'{m:.{prec}f} ± {sd:.{prec}f}'

    rows = []
    for n in grids:
        cell = data[(model_name, n)]
        for meth in SHOWN:
            if meth not in cell:
                continue
            ratio = stats(sigma_ratio(cell, meth, seeds)) if meth != 'Classical MCMC' else (None,)
            rows.append([f'{n}', campaign_io.display_method(meth)]
                        + [cellstr(cell, meth, f'{p}_mean', 2 if p == 'H0' else 4) for p in pn]
                        + [cellstr(cell, meth, f'{pn[0]}_std', 4),
                           '—' if ratio[0] is None else f'{ratio[0]:.3f} ± {ratio[1]:.3f}',
                           cellstr(cell, meth, 'acceptance', 4),
                           cellstr(cell, meth, 'ESS', 0),
                           cellstr(cell, meth, 'final_KL', 4)])
    cols = ['nqpp', 'method'] + pn + [f'σ({pn[0]})', 'σ/σ_MCMC', 'acc', 'ESS', 'final KL']
    fig = plt.figure(figsize=(16 + 2.2 * max(0, len(pn) - 2), 0.3 * len(rows) + 0.6))
    ax = fig.add_subplot(111)
    ax.axis('off')
    tbl = ax.table(cellText=rows, colLabels=cols, cellLoc='center', bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    w_n, w_m = 0.04, 0.2
    rest = (1 - w_n - w_m) / (len(cols) - 2)
    for (i, j), c in tbl.get_celld().items():
        c.set_width(w_n if j == 0 else w_m if j == 1 else rest)
        if i > 0 and rows[i - 1][0] != (rows[i - 2][0] if i > 1 else None):
            c.visible_edges = 'closed'
        if i > 0 and int(rows[i - 1][0]) % 2 == 0:
            c.set_facecolor('#f3f3f3')
    note = rhat_note(data, model_name, seeds)
    fig.suptitle(f'{model.label} — ladder summary over {len(seeds)} seeds '
                 f'({seeds[0]}–{seeds[-1]}): mean ± sd across seeds '
                 '(standard errors in ladder_summary.csv)'
                 + (f'\n{note}' if note else ''), fontsize=11, fontweight='bold',
                 color='firebrick' if note else 'k')
    return save(fig, os.path.join(outdir, model_name, f'ladder_summary_{model_name}'))


def fig_sigma_vs_grid(data, seeds, outdir, report):
    """sigma_vs_grid: sigma(QVMC 67%)/sigma(MCMC) in Om vs nqpp, with the
    classical-VI control."""
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    mg = models_and_grids(data)
    off = {m: (i - (len(mg) - 1) / 2) * 0.06 for i, m in enumerate(mg)}
    report.append('\n== sigma(Om) ratio vs Classical MCMC (mean ± sd (se), n) ==')
    for model_name, grids in mg.items():
        col = MODEL_COLORS.get(model_name, 'k')
        for meth, ls, mfc, dx in ((QVMC_REF, '-', None, 0.0), ('Classical VI', ':', 'white', 0.02)):
            xs = [n + off[model_name] + dx for n in grids]
            vals = [sigma_ratio(data[(model_name, n)], meth, seeds) for n in grids]
            for n, v in zip(grids, vals):
                report.append(f'  {model_name} nqpp{n} {meth:14s} {fmt(*stats(v)[:3], prec=3)}  n={stats(v)[3]}')
            lab = (f'{core.MODELS[model_name].label.split("  ")[0]}'
                   if meth == QVMC_REF else None)
            series(ax, xs, vals, col, 'o' if meth == QVMC_REF else 'D', lab,
                   ls=ls, mfc=mfc, jitter=0.012)
    ax.axhline(1.0, color='k', lw=0.9, ls='--')
    ax.set_xticks(sorted({n for g in mg.values() for n in g}))
    ax.set_xlabel('nqpp (qubits per parameter)')
    ax.set_ylabel(r'$\sigma(\Omega_m)\,/\,\sigma_{\rm MCMC}(\Omega_m)$')
    from matplotlib.lines import Line2D
    h, _ = ax.get_legend_handles_labels()
    h += [Line2D([0], [0], color='k', ls='-', marker='o',
                 label=campaign_io.display_method(QVMC_REF)),
          Line2D([0], [0], color='k', ls=':', marker='D', mfc='white',
                 label='Classical VI (control, same grid and budget)')]
    ax.legend(handles=h, fontsize=8, loc='upper left', bbox_to_anchor=(1.01, 1.0))
    ax.grid(alpha=0.3)
    ax.set_title('Width of the Ωm posterior relative to classical MCMC\n'
                 + seed_caption(seeds), fontsize=10)
    return save(fig, os.path.join(outdir, 'sigma_vs_grid'))


def _kl_traces(taskdirs, model_name, n, seeds):
    """``{stored pct: {seed: (its, kls)}}`` from the task logs."""
    out = defaultdict(dict)
    for s in seeds:
        d = taskdirs.get((model_name, n), {}).get(s)
        logs = sorted(glob.glob(os.path.join(d, 'sweep_all_*.log'))) if d else []
        if not logs:
            continue
        _rh, kl = cmq.histories_from_log(logs[-1])
        for pct, h in kl.items():
            if h:
                out[pct][s] = (np.array([x['it'] for x in h], float),
                               np.array([max(x['kl'], 1e-12) for x in h]))
    return out


def _band(traces, grid):
    """Median and 16/84 percentiles of log-interpolated traces on ``grid``
    (a seed contributes only up to its own last iteration)."""
    Y = np.full((len(traces), len(grid)), np.nan)
    for k, (xs, ys) in enumerate(traces):
        ok = grid <= xs[-1]
        Y[k, ok] = np.exp(np.interp(grid[ok], xs, np.log(ys)))
    with np.errstate(all='ignore'):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            return (np.nanmedian(Y, 0), np.nanpercentile(Y, 16, 0),
                    np.nanpercentile(Y, 84, 0))


def fig_kl(data, taskdirs, model_name, n, seeds, outdir, report):
    """ladder_kl_qvmc_<model>_nqpp<N>: median KL trace over seeds + band."""
    tr = _kl_traces(taskdirs, model_name, n, seeds)
    rungs = [(m, float(STORED_PCT[m])) for m in QVMC_RUNGS]
    rungs = [(m, p) for m, p in rungs if p in tr]
    if not rungs:
        return None
    # identical traces (classical VI vs quantum sampling) are flagged, not assumed
    same = None
    if 0.0 in tr and 33.0 in tr:
        same = all(s in tr[33.0] and np.array_equal(tr[0.0][s][1], tr[33.0][s][1])
                   for s in tr[0.0])
    disp = cmq.qvmc_display_runs([{'pct': p} for _, p in rungs])
    cols = cmq._q_colors([r['pct'] for r in disp])
    # [B-KLXLIM] window = shortest rung (its shortest seed), when rungs differ a lot
    ends = [min(xs[-1] for xs, _ in tr[p].values()) for _, p in rungs]
    x_hi = 1.08 * min(ends) if len(ends) >= 2 and max(ends) > 1.5 * min(ends) else None
    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    for (meth, p), r, col in zip(rungs, disp, cols):
        traces = [tr[p][s] for s in sorted(tr[p], key=int)]
        top = max(xs[-1] for xs, _ in traces)
        grid = np.unique(np.concatenate([xs for xs, _ in traces]))
        grid = grid[grid <= (x_hi if x_hi else top) * 1.0001]
        med, lo, hi = _band(traces, grid)
        ls = '--' if (p == 33.0 and same) else '-'
        lab = (f'{cmq.qvmc_rung_label(r["pct"], short=True)}'
               + (' (classical VI)' if r['pct'] == 0 else '') + f'  (n={len(traces)})')
        if p == 33.0 and same:
            lab += ' — identical to classical VI in every seed'
        fin = np.array([ys[-1] for _, ys in traces])
        endit = np.array([xs[-1] for xs, _ in traces])
        if x_hi is not None and np.median(endit) > x_hi:
            lab += (f'\n    continues to median {np.median(endit):,.0f} its; '
                    f'final KL median {np.median(fin):.4f}')
            ax.annotate('', xy=(x_hi, np.median(fin)), xytext=(0.985 * x_hi, np.median(fin)),
                        arrowprops=dict(arrowstyle='->', color=col, lw=1.4),
                        annotation_clip=False)
        ax.fill_between(grid, lo, hi, color=col, alpha=0.18, lw=0)
        ax.semilogy(grid, med, color=col, lw=2.6 if r['pct'] == 0 else 1.9, ls=ls, label=lab)
        report.append(f'  KL {model_name} nqpp{n} {meth:13s} final median {np.median(fin):.4f} '
                      f'[16-84: {np.percentile(fin, 16):.4f}-{np.percentile(fin, 84):.4f}] '
                      f'end its median {np.median(endit):.0f}')
    if x_hi is not None:
        ax.set_xlim(0, x_hi)
    ax.set_xlabel('training iteration')
    ax.set_ylabel(r'KL$(Q_\varphi\,\|\,P_{\rm target})$')
    ax.set_title(f'{core.MODELS[model_name].label} — QVMC training over seeds, nqpp = {n}\n'
                 f'median over {len(seeds)} seeds, band = 16–84 percentiles (from the task logs)',
                 fontsize=10)
    ax.legend(title='QVMC %', fontsize=7.5)
    ax.grid(True, alpha=0.3)
    return save(fig, os.path.join(outdir, model_name, f'ladder_kl_qvmc_{model_name}_nqpp{n}'))


def fig_rhat(data, taskdirs, model_name, seeds, outdir, report):
    """ladder_rhat_qmcmc_<model>: median R-hat - 1 over seeds + band.

    QMCMC does not depend on nqpp, so the largest grid's logs are used; the
    other grids are checked to give the same traces.
    """
    grids = models_and_grids(data)[model_name]
    per = {}
    for n in grids:
        tr = defaultdict(dict)
        for s in seeds:
            d = taskdirs.get((model_name, n), {}).get(s)
            logs = sorted(glob.glob(os.path.join(d, 'sweep_all_*.log'))) if d else []
            if logs:
                rh, _ = cmq.histories_from_log(logs[-1])
                for p, h in rh.items():
                    tr[p][s] = h
        per[n] = tr
    n0 = grids[-1]
    tr = per[n0]
    same = all(per[n] == tr for n in grids)
    if not same:
        report.append(f'  WARNING: R-hat traces of {model_name} differ across nqpp; '
                      f'figure uses nqpp{n0}')
    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    pcts = sorted(tr)
    cols = cmq._q_colors(pcts)
    for p, col in zip(pcts, cols):
        seeds_p = sorted(tr[p], key=int)
        steps = sorted({st for s in seeds_p for st, _ in tr[p][s]})
        Y = np.full((len(seeds_p), len(steps)), np.nan)
        for k, s in enumerate(seeds_p):
            d = dict(tr[p][s])
            Y[k] = [d.get(st, np.nan) - 1 for st in steps]
        med = np.nanmedian(Y, 0)
        lo, hi = np.nanpercentile(Y, 16, 0), np.nanpercentile(Y, 84, 0)
        name = 'Classical MCMC' if p == 0 else f'QMCMC {p:.0f}%'
        ls = '--' if p == 100 and 50.0 in tr and tr[50.0] == tr[100.0] else '-'
        lab = f'{p:.0f}%' + (' (classical)' if p == 0 else '')
        if ls == '--':
            lab += ' — identical to 50 % in every seed'
        finals = Y[:, -1] + 1
        nbad = int(np.sum(finals >= core.RHAT_THRESHOLD))
        if nbad:
            lab += f'  [final R̂ ≥ {core.RHAT_THRESHOLD:g} in {nbad}/{len(seeds_p)} seeds, max {np.nanmax(finals):.3f}]'
        ax.fill_between(steps, np.clip(lo, 1e-5, None), np.clip(hi, 1e-5, None),
                        color=col, alpha=0.18, lw=0)
        ax.semilogy(steps, np.clip(med, 1e-5, None), color=col, ls=ls,
                    lw=2.6 if p == 0 else 1.9, marker='o', ms=3, label=lab)
        report.append(f'  Rhat {model_name} {name:14s} final median {np.nanmedian(finals):.4f} '
                      f'max {np.nanmax(finals):.4f}  seeds>=thr {nbad}/{len(seeds_p)}')
    ax.axhline(core.RHAT_THRESHOLD - 1, color='k', ls='--', lw=1.1,
               label=rf'$\hat R-1={core.RHAT_THRESHOLD - 1:g}$')
    ax.set_xlabel('sampling steps')
    ax.set_ylabel(r'$\hat R_{\max}-1$')
    note = rhat_note(data, model_name, seeds)
    ax.set_title(f'{core.MODELS[model_name].label} — QMCMC convergence over seeds\n'
                 f'median over {len(seeds)} seeds, band = 16–84 percentiles '
                 f'(logs of nqpp{n0}; {"identical on every grid" if same else "GRIDS DIFFER"})',
                 fontsize=10, color='firebrick' if note else 'k')
    ax.legend(title='QMCMC %', fontsize=7.5)
    ax.grid(True, alpha=0.3)
    return save(fig, os.path.join(outdir, model_name, f'ladder_rhat_qmcmc_{model_name}'))


def _largest_cell(data, model_name):
    return data[(model_name, models_and_grids(data)[model_name][-1])]


def fig_efficiency(data, seeds, outdir, report):
    """qmcmc_efficiency: ESS ratio QMCMC/MCMC and acceptance difference."""
    mg = models_and_grids(data)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    report.append('\n== QMCMC efficiency (identical on every grid; largest grid used) ==')
    for k, model_name in enumerate(mg):
        cell = _largest_cell(data, model_name)
        if not mcmc_rows_identical(data, model_name, seeds):
            report.append(f'  WARNING: MCMC-family rows of {model_name} differ across grids')
        bad = bool(rhat_flags(data, model_name, seeds))
        for ax, which in zip(axes, ('ess', 'acc')):
            for j, meth in enumerate(('QMCMC 50%', 'QMCMC 100%')):
                vals = []
                for s in seeds:
                    q, c = cell.get(meth, {}).get(s), cell.get('Classical MCMC', {}).get(s)
                    if which == 'ess':
                        a, b = num(q, 'ESS'), num(c, 'ESS')
                        vals.append(a / b if a is not None and b else None)
                    else:
                        a, b = num(q, 'acceptance'), num(c, 'acceptance')
                        vals.append(a - b if None not in (a, b) else None)
                x = k + (j - 0.5) * 0.28
                draw_seeds(ax, x, vals, width=0.05)
                mean_bar(ax, x, vals, cmq.C_QUANTUM, marker='o' if j == 0 else 's',
                         mfc=None if j == 0 else 'white')
                m, sd, se, nn = stats(vals)
                d, se_p, t = compare_seeds.in_sigmas(
                    per_seed(cell, meth, lambda r: num(r, 'acceptance'), seeds),
                    per_seed(cell, 'Classical MCMC', lambda r: num(r, 'acceptance'), seeds))
                extra = f'  paired t = {t:.2f}' if which == 'acc' and t is not None else ''
                report.append(f'  {model_name:5s} {meth:10s} {"ESS ratio " if which == "ess" else "acc diff  "}'
                              f'{fmt(m, sd, se, prec=4)}  n={nn}{extra}'
                              + ('  [R-hat caveat]' if bad else ''))
            if bad:
                ax.axvspan(k - 0.45, k + 0.45, color='firebrick', alpha=0.07, lw=0)
    labels = [m.upper() + ('\n(R̂ ≥ 1.01 in\nsome seeds)' if rhat_flags(data, m, seeds) else '')
              for m in mg]
    for ax, ttl, ref in zip(axes, ('ESS(QMCMC) / ESS(MCMC)', 'acceptance(QMCMC) − acceptance(MCMC)'),
                            (1.0, 0.0)):
        ax.axhline(ref, color='k', ls='--', lw=0.9)
        ax.set_xticks(range(len(mg)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(ttl, fontsize=10)
        ax.grid(alpha=0.3)
    from matplotlib.lines import Line2D
    axes[0].legend(handles=[Line2D([0], [0], color=cmq.C_QUANTUM, marker='o', ls='none', label='QMCMC 50%'),
                            Line2D([0], [0], color=cmq.C_QUANTUM, marker='s', mfc='white', ls='none',
                                   label='QMCMC 100%')], fontsize=8)
    fig.suptitle('QMCMC vs classical MCMC, same seed (does not depend on nqpp)\n'
                 + seed_caption(seeds) + '; shaded = not converged in some seeds',
                 fontsize=10)
    fig.tight_layout()
    return save(fig, os.path.join(outdir, 'qmcmc_efficiency'))


def fig_bias(data, seeds, outdir, report):
    """om_bias: (Om(rung) - Om(MCMC)) / sigma_MCMC per rung and grid."""
    mg = models_and_grids(data)
    rungs = SHOWN[1:]
    fig, axes = plt.subplots(1, len(mg), figsize=(3.6 * len(mg), 4.6), sharey=True,
                             squeeze=False)
    axes = axes[0]
    cols = {'QMCMC 50%': cmq.C_QUANTUM, 'QMCMC 100%': '#8b0000',
            'Classical VI': cmq.C_CLASSICAL2, 'QVMC 33%': '#ffbb55', 'QVMC 67%': cmq.C_QUANTUM2}
    marks = {'QMCMC 50%': 'o', 'QMCMC 100%': 's', 'Classical VI': 'D', 'QVMC 33%': 'v', 'QVMC 67%': '^'}
    report.append('\n== Om shift vs MCMC, in sigma_MCMC (mean ± sd (se)) ==')
    for ax, (model_name, grids) in zip(axes, mg.items()):
        for j, meth in enumerate(rungs):
            for n in grids:
                vals = om_shift(data[(model_name, n)], meth, seeds)
                x = n + (j - (len(rungs) - 1) / 2) * 0.15
                draw_seeds(ax, x, vals, width=0.02)
                mean_bar(ax, x, vals, cols[meth], marker=marks[meth],
                         label=campaign_io.display_method(meth) if n == grids[0] else None, ms=5)
                report.append(f'  {model_name} nqpp{n} {meth:14s} {fmt(*stats(vals)[:3], prec=3)}')
        ax.axhline(0, color='k', lw=0.9, ls='--')
        ax.set_xticks(grids)
        ax.set_xlabel('nqpp')
        note = ' (R̂ caveat)' if rhat_flags(data, model_name, seeds) else ''
        ax.set_title(core.MODELS[model_name].label.split('  ')[0] + note, fontsize=10,
                     color='firebrick' if note else 'k')
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(r'$(\bar\Omega_m - \bar\Omega_m^{\rm MCMC})\,/\,\sigma_{\rm MCMC}$')
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=len(rungs), fontsize=8, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle('Shift of the Ωm posterior mean relative to classical MCMC (same seed)\n'
                 + seed_caption(seeds) + '. QMCMC rungs repeat on every grid (they do not use it).',
                 fontsize=10)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return save(fig, os.path.join(outdir, 'om_bias'))


def fig_paired_kl(data, seeds, outdir, report):
    """paired_kl: final_KL(QVMC 67%) - final_KL(Classical VI) within each cell."""
    mg = models_and_grids(data)
    fig, axes = plt.subplots(1, len(mg), figsize=(3.4 * len(mg), 4.4), squeeze=False)
    axes = axes[0]
    report.append(f'\n== paired final_KL: {QVMC_REF} − Classical VI, within each cell ==')
    for ax, (model_name, grids) in zip(axes, mg.items()):
        for n in grids:
            cell = data[(model_name, n)]
            a = per_seed(cell, QVMC_REF, lambda r: num(r, 'final_KL'), seeds)
            b = per_seed(cell, 'Classical VI', lambda r: num(r, 'final_KL'), seeds)
            diffs = [x - y if None not in (x, y) else None for x, y in zip(a, b)]
            d, se, t = compare_seeds.in_sigmas(a, b)
            sig = t is not None and abs(t) > 3
            col = MODEL_COLORS.get(model_name, 'k')
            draw_seeds(ax, n, diffs, width=0.06)
            mean_bar(ax, n, diffs, col, marker='*' if sig else 'o', ms=11 if sig else 6,
                         mfc=col if sig else 'white')
            mm, sd, _, _ = stats(diffs)
            if t is not None:
                ax.annotate(f't={t:.1f}', (n, mm + sd), xytext=(0, 4),
                            textcoords='offset points', ha='center', fontsize=7.5,
                            fontweight='bold' if sig else 'normal')
            report.append(f'  {model_name} nqpp{n}  diff {fmt(mm, sd, se, prec=4)}  '
                          f't = {t:.2f}' if t is not None else f'  {model_name} nqpp{n} t = n/a')
        ax.axhline(0, color='k', lw=0.9, ls='--')
        ax.set_xticks(grids)
        ax.set_xlim(grids[0] - 0.6, grids[-1] + 0.6)
        ax.set_xlabel('nqpp')
        ax.set_title(core.MODELS[model_name].label.split('  ')[0], fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(r'final KL$_{\rm QVMC\,100\%}$ − final KL$_{\rm classical\ VI}$')
    fig.suptitle(f'Paired difference of final KL, {campaign_io.display_method(QVMC_REF)} minus '
                 'classical VI, same seed and grid\n'
                 '★ = paired t > 3; each point is a within-cell difference — '
                 'final KL is not comparable across nqpp (different target grids). '
                 + seed_caption(seeds), fontsize=9.5)
    fig.tight_layout()
    return save(fig, os.path.join(outdir, 'paired_kl'))


def fig_aic_bic(data, seeds, outdir, report):
    """modelos_aic_bic: Delta AIC / Delta BIC vs LCDM, with seed bars."""
    mg = models_and_grids(data)
    report.append('\n== model selection (all rows of a seed/model must agree) ==')
    per = {}
    for model_name in mg:
        vals = defaultdict(dict)
        for (m, n), cell in data.items():
            if m != model_name:
                continue
            for meth, rows in cell.items():
                if campaign_io.is_hidden_method(meth):
                    continue
                for s, r in rows.items():
                    for c in ('AIC', 'BIC', 'chi2'):
                        v = num(r, c)
                        if v is not None:
                            vals[c].setdefault(s, set()).add(round(v, 4))
        for c in vals:
            for s, v in vals[c].items():
                if len(v) > 1:
                    report.append(f'  WARNING: {model_name} seed {s} has {len(v)} different {c} '
                                  f'values across rungs/grids: {sorted(v)}')
        per[model_name] = {c: {s: min(v) for s, v in vals[c].items()} for c in vals}
    ref = 'lcdm' if 'lcdm' in per else next(iter(per))
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    others = [m for m in mg if m != ref]
    for j, (c, col) in enumerate((('AIC', '#4c72b0'), ('BIC', '#dd8452'))):
        for k, m in enumerate(others):
            d = [per[m][c].get(s) - per[ref][c].get(s)
                 if per[m][c].get(s) is not None and per[ref][c].get(s) is not None else None
                 for s in seeds]
            mm, sd, se, nn = stats(d)
            x = k + (j - 0.5) * 0.36
            ax.bar(x, mm, 0.34, color=col, yerr=sd, capsize=4,
                   label=f'Δ{c}' if k == 0 else None)
            draw_seeds(ax, x, d, width=0.08, zorder=4)
            ax.annotate(f'{mm:+.3f}\nsd {sd:.1e}', (x, mm), xytext=(0, 3 if mm >= 0 else -4),
                        textcoords='offset points', ha='center', fontsize=7,
                        va='bottom' if mm >= 0 else 'top')
            report.append(f'  Δ{c} {m} − {ref}: {fmt(mm, sd, se, prec=4)}  n={nn}'
                          + ('' if sd == 0 else '   WARNING: sd != 0 across seeds'))
    ax.axhline(0, color='k', lw=0.8)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo - 0.12 * (hi - lo), hi + 0.1 * (hi - lo))
    ax.set_xticks(range(len(others)))
    ax.set_xticklabels([core.MODELS[m].label.split('  ')[0] for m in others])
    ax.set_ylabel(f'Δ relative to {core.MODELS[ref].label}')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    ax.set_title('Model selection over seeds (positive = disfavoured vs ΛCDM)\n'
                 + f'{len(seeds)} seeds ({seeds[0]}–{seeds[-1]}): bar = mean, error bar = sd across '
                 'seeds (expected 0), grey = individual seeds', fontsize=9.5)
    return save(fig, os.path.join(outdir, 'modelos_aic_bic'))


# ── corners from the stored chains ───────────────────────────────────────────
def _npz(taskdir, model_name):
    p = os.path.join(taskdir, f'model_{model_name}', f'samples_ladder_{model_name}.npz')
    return np.load(p) if os.path.exists(p) else None


def fig_corners(data, taskdirs, model_name, seeds, outdir, report):
    """corner_ladder_qmcmc_<model> and corner_ladder_qvmc_<model>_nqpp<N>,
    pooling the chains of every seed (each seed weighs the same)."""
    model = core.MODELS[model_name]
    grids = models_and_grids(data)[model_name]
    rng = np.random.default_rng(12345)          # display jitter only
    out = []
    # QMCMC family: chains do not depend on nqpp -> largest grid
    n0 = grids[-1]
    pools = defaultdict(list)
    for s in seeds:
        d = taskdirs[(model_name, n0)].get(s)
        z = _npz(d, model_name) if d else None
        if z is None:
            continue
        for p in (0, 50, 100):
            k = f'qmcmc_{p}_flat'
            if k in z:
                pools[p].append(z[k])
    if pools:
        pcts = sorted(pools)
        nseeds = len(pools[pcts[0]])
        labels = [('Classical MCMC' if p == 0 else f'QMCMC {p}%') for p in pcts]
        f = cmq.plot_corner_multi([np.vstack(pools[p]) for p in pcts], cmq._q_colors(pcts),
                                  labels, model, os.path.join(outdir, model_name),
                                  f'ladder_qmcmc_{model_name}',
                                  title=f'{model.label} — QMCMC ladder, {nseeds} seeds pooled'
                                        + ('\n' + rhat_note(data, model_name, seeds)
                                           if rhat_flags(data, model_name, seeds) else ''))
        out.append(f)
        report.append(f'  corner qmcmc {model_name}: {nseeds} seeds pooled (nqpp{n0} chains)')
    # QVMC family per grid (+ classical MCMC as the reference)
    for n in grids:
        cell_cell = np.array([(hi - lo) / (2 ** n - 1) for lo, hi in model.sample_box])
        S_pool, W_pool, mc = defaultdict(list), defaultdict(list), []
        for s in seeds:
            d = taskdirs[(model_name, n)].get(s)
            z = _npz(d, model_name) if d else None
            if z is None:
                continue
            mc.append(z['qmcmc_0_flat'])
            for meth in QVMC_RUNGS:
                p = STORED_PCT[meth]
                if f'qvmc_{p}_S' not in z:
                    continue
                S = z[f'qvmc_{p}_S']
                W = np.asarray(z[f'qvmc_{p}_W'], float)
                S_pool[p].append(S + rng.uniform(-0.5, 0.5, S.shape) * cell_cell)
                W_pool[p].append(W / W.sum())
        if not S_pool:
            continue
        pcts = [STORED_PCT[m] for m in QVMC_RUNGS if STORED_PCT[m] in S_pool]
        disp = [cmq.qvmc_display_pct(p) for p in pcts]
        nseeds = len(mc)
        mc_all = np.vstack(mc)
        datasets = [mc_all] + [np.vstack(S_pool[p]) for p in pcts]
        weights = [np.full(len(mc_all), 1.0 / len(mc_all))] + \
                  [np.concatenate(W_pool[p]) / len(W_pool[p]) for p in pcts]
        colors = ['0.25'] + cmq._q_colors(disp)
        labels = ['Classical MCMC (reference)'] + \
                 [cmq.qvmc_rung_label(dp) + (' (classical VI)' if dp == 0 else '') for dp in disp]
        f = cmq.plot_corner_multi(datasets, colors, labels, model, os.path.join(outdir, model_name),
                                  f'ladder_qvmc_{model_name}_nqpp{n}',
                                  title=f'{model.label} — QVMC ladder, nqpp = {n}, {nseeds} seeds '
                                        'pooled (cell-jittered for display)',
                                  weights_list=weights)
        out.append(f)
        report.append(f'  corner qvmc {model_name} nqpp{n}: {nseeds} seeds pooled')
    return out


# ── driver ───────────────────────────────────────────────────────────────────
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('root', help='folder holding seeds_s<NN>/')
    ap.add_argument('--out', required=True, help='output folder for the figures')
    ap.add_argument('--seeds', default=None, help="e.g. '43-50' (default: all)")
    ap.add_argument('--no-corner', action='store_true', help='skip the corner plots')
    a = ap.parse_args(argv)
    cmq.set_headless_backend()
    data, taskdirs = load_campaign(a.root, parse_seeds(a.seeds))
    if not data:
        print('no rows found', file=sys.stderr)
        return 1
    seeds = seeds_of(data)
    mg = models_and_grids(data)
    report = [f'seed_figures: {len(seeds)} seeds {seeds}; cells: '
              + ', '.join(f'{m} nqpp{",".join(map(str, g))}' for m, g in mg.items())]
    for (m, n), cell in sorted(data.items()):
        miss = [meth for meth in SHOWN if len(cell.get(meth, {})) != len(seeds)]
        if miss:
            report.append(f'  WARNING: {m} nqpp{n} missing seeds for {miss}')
    for m in mg:
        note = rhat_note(data, m, seeds)
        if note:
            report.append(f'  {m}: {note}')
            for meth, lst in rhat_flags(data, m, seeds).items():
                report.append(f'     {meth}: ' + ', '.join(f's{s} {v:.4f}' for s, v in lst))
    os.makedirs(a.out, exist_ok=True)
    files = []
    rows = summary_rows(data, seeds)
    files.append(write_summary_csv(rows, os.path.join(a.out, 'ladder_summary.csv')))
    files.append(fig_sigma_vs_grid(data, seeds, a.out, report))
    files.append(fig_efficiency(data, seeds, a.out, report))
    files.append(fig_bias(data, seeds, a.out, report))
    files.append(fig_paired_kl(data, seeds, a.out, report))
    files.append(fig_aic_bic(data, seeds, a.out, report))
    report.append('\n== traces from logs ==')
    for m, grids in mg.items():
        files.append(fig_summary(data, m, seeds, a.out))
        files.append(fig_rhat(data, taskdirs, m, seeds, a.out, report))
        for n in grids:
            files.append(fig_trends(data, m, n, seeds, a.out))
            files.append(fig_kl(data, taskdirs, m, n, seeds, a.out, report))
    if not a.no_corner:
        report.append('\n== corners ==')
        for m in mg:
            files += fig_corners(data, taskdirs, m, seeds, a.out, report)
    text = '\n'.join(report) + '\n'
    with open(os.path.join(a.out, 'seed_figures_report.txt'), 'w') as fh:
        fh.write(text)
    print(text)
    print(f'{len([f for f in files if f])} figures/tables written to {a.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
