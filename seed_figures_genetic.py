#!/usr/bin/env python3
"""Multi-seed figures of the genetic campaign (CGA and the QGA ladder).

The genetic counterpart of `seed_figures.py`, with the same conventions:
point = mean over seeds, bar = sd across seeds, every seed as a faint grey
point; when the sd across seeds is exactly zero the figure says so instead
of drawing an invisible bar. Traces (best chi2 and best parameters per
generation) come from the task logs and are drawn as the median over seeds
with a 16-84 percentile band.

What is compared [B-REFINE]: ``chi2_grid``, the chi2 of the best grid point
before the local refinement. The refined chi2 is the same for every method
and tells nothing apart. The ``<p>_std`` columns of the genetic rows are the
spread of the final population (fitness-weighted), NOT a parameter
uncertainty, and are never compared with the sigmas of the samplers.

The genetic grid is n_bits per parameter. It is stored in the ``nqpp``
column of the CSV but it is NOT the samplers' nqpp, hence the ``_genetico``
suffix of every file.

Output (PNG + PDF), under ``--out``::

    genetico_chi2grid_contra_nbits, genetic_summary.csv, genetic_report.txt,
    gif_index.html (when GIFs are found under the campaign folder)
    <model>/convergence_<model>_genetico, cost_<model>_genetico,
            fitness_<model>_nb<N>_genetico, genetic_convergence_<model>_nb<N>_genetico

Usage:
    python seed_figures_genetic.py RESULTS_DIR --out FIGURES_DIR [--seeds 43-50]

``RESULTS_DIR`` holds the ``genetic_s<NN>`` folders.
"""
from __future__ import annotations

import argparse
import csv
import glob
import html
import os
import re
import sys
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

import compare_seeds  # noqa: E402
import cosmo_core as core  # noqa: E402
import seed_figures as sf  # noqa: E402

#: Methods in ladder order (CSV ``Method`` values).
METHODS = ['CGA', 'QGA (q=0%)', 'QGA (q=33%)', 'QGA (q=67%)', 'QGA (q=100%)']
QGA_LEVELS = METHODS[1:]
COLORS = {'CGA': '#1f77b4', 'QGA (q=0%)': '#17becf', 'QGA (q=33%)': '#ff7f0e',
          'QGA (q=67%)': '#d62728', 'QGA (q=100%)': '#7f0000'}
MARKERS = {'CGA': 'o', 'QGA (q=0%)': 's', 'QGA (q=33%)': 'D', 'QGA (q=67%)': '^',
           'QGA (q=100%)': 'v'}
STYLES = {'CGA': '-', 'QGA (q=0%)': '--', 'QGA (q=33%)': '-', 'QGA (q=67%)': '-',
          'QGA (q=100%)': '-.'}
SUFFIX = '_genetico'
TOL = 1e-9

# ── log parsing ──────────────────────────────────────────────────────────────
_RE_START = re.compile(r'\[(CGA|QGA)\] start \|.*quantumness=([\d.]+)%')
_RE_GEN = re.compile(r'\[(CGA|QGA)\] gen\s+(\d+)/\d+ \| best χ²=([\d.eE+-]+) \| '
                     r'mean χ²=([\d.eE+-]+|inf|nan) \|(.*)$')
_RE_GRID = re.compile(r'\[(CGA|QGA)\]\s+best on the grid \(NOT refined\):(.*)\| χ²=([\d.eE+-]+)')
_RE_KV = re.compile(r'([A-Za-z_][\w]*)=([-+]?[\d.]+(?:[eE][-+]?\d+)?)')


def method_name(tag, q):
    """CSV ``Method`` of a log tag and quantumness.

    Examples:
        >>> method_name('CGA', '0'), method_name('QGA', '67')
        ('CGA', 'QGA (q=67%)')
    """
    return 'CGA' if tag == 'CGA' else f'QGA (q={int(round(float(q)))}%)'


def parse_genetic_log(path):
    """Per-generation traces and grid optimum of every method in one log.

    The genetic sweep log prints, for each method, a ``start`` line with its
    quantumness, a ``gen`` line every 10 generations (plus the last one)
    with the best chi2 (on the grid, not refined), the mean chi2 and the
    best individual's parameters, and a ``best on the grid`` line at the end.

    Returns:
        ``{method: {'gen': [...], 'best': [...], 'mean': [...],
        'params': {p: [...]}, 'grid_chi2': float, 'grid_params': {p: v}}}``

    Examples:
        >>> import tempfile, os
        >>> t = tempfile.NamedTemporaryFile('w', suffix='.log', delete=False, encoding='utf-8')
        >>> _ = t.write('[QGA] start | pop=5 gens=2 | model=X | quantumness=67%\\n'
        ...             '[QGA] gen    0/2 | best χ²=10.5 | mean χ²=12.0 | Om=0.30  H0=70.0\\n'
        ...             '[QGA]   best on the grid (NOT refined): Om=0.30  H0=70.0 | χ²=10.500 (x)\\n')
        >>> t.close()
        >>> r = parse_genetic_log(t.name)['QGA (q=67%)']
        >>> r['gen'], r['best'], r['params']['H0'], r['grid_chi2']
        ([0], [10.5], [70.0], 10.5)
        >>> os.unlink(t.name)
    """
    out = {}
    cur = None
    try:
        lines = open(path, encoding='utf-8', errors='ignore').read().splitlines()
    except OSError:
        return out
    for ln in lines:
        m = _RE_START.search(ln)
        if m:
            cur = method_name(*m.groups())
            out[cur] = {'gen': [], 'best': [], 'mean': [], 'params': defaultdict(list),
                        'grid_chi2': None, 'grid_params': {}}
            continue
        m = _RE_GEN.search(ln)
        if m and cur is not None:
            r = out[cur]
            r['gen'].append(int(m.group(2)))
            r['best'].append(float(m.group(3)))
            r['mean'].append(float(m.group(4)))
            for k, v in _RE_KV.findall(m.group(5)):
                r['params'][k].append(float(v))
            continue
        m = _RE_GRID.search(ln)
        if m and cur is not None:
            out[cur]['grid_chi2'] = float(m.group(3))
            out[cur]['grid_params'] = {k: float(v) for k, v in _RE_KV.findall(m.group(2))}
    for r in out.values():
        r['params'] = dict(r['params'])
    return out


def load_traces(taskdirs, seeds):
    """``traces[(model, nb)][seed] = parse_genetic_log(...)`` from each task."""
    out = defaultdict(dict)
    for cell, by_seed in taskdirs.items():
        for s in seeds:
            d = by_seed.get(s)
            logs = sorted(glob.glob(os.path.join(d, 'genetic_sweep_all_*.log'))) if d else []
            if logs:
                out[cell][s] = parse_genetic_log(logs[-1])
    return out


# ── numbers ──────────────────────────────────────────────────────────────────
def chi2_grid(cell, method, seeds):
    """Per-seed ``chi2_grid`` of one method of one cell."""
    return sf.per_seed(cell, method, lambda r: sf.num(r, 'chi2_grid'), seeds)


def delta_vs_cga(cell, method, seeds):
    """Per-seed chi2_grid(method) - chi2_grid(CGA) (paired, same seed)."""
    a, b = chi2_grid(cell, method, seeds), chi2_grid(cell, 'CGA', seeds)
    return [x - y if None not in (x, y) else None for x, y in zip(a, b)]


def paired(cell, method, seeds):
    """Paired comparison with CGA.

    Returns:
        dict with mean, sd, se, n, t (None if every difference is equal) and
        ``worse``, the number of seeds where the method is worse than CGA.
    """
    d = delta_vs_cga(cell, method, seeds)
    m, sd, se, n = sf.stats(d)
    _d, _se, t = compare_seeds.in_sigmas(chi2_grid(cell, method, seeds),
                                         chi2_grid(cell, 'CGA', seeds))
    worse = sum(1 for x in d if x is not None and x > TOL)
    return dict(mean=m, sd=sd, se=se, n=n, t=t, worse=worse)


def identical_rows(cell, a, b, seeds,
                   cols=('Om_mean', 'Om_std', 'H0_mean', 'H0_std', 'chi2', 'chi2_grid')):
    """True if methods ``a`` and ``b`` wrote the same values in every seed."""
    for s in seeds:
        ra, rb = cell.get(a, {}).get(s), cell.get(b, {}).get(s)
        if ra is None or rb is None:
            return False
        if any((ra.get(c) or '') != (rb.get(c) or '') for c in cols):
            return False
    return True


def zero_sd(vals):
    """True when the values are identical in every seed (sd exactly 0)."""
    v = [x for x in vals if x is not None]
    return len(v) > 1 and max(v) - min(v) <= TOL


# ── figures ──────────────────────────────────────────────────────────────────
def fig_delta_vs_nbits(data, seeds, out, report):
    """genetico_chi2grid_contra_nbits: Delta chi2_grid (QGA q - CGA) vs n_bits.

    The y axis is symmetric-logarithmic (linear below 1e-3): the differences
    run from ~1.7 at n_bits 4 to ~1e-3 at n_bits 8, so a linear axis puts
    every n_bits >= 6 on top of zero, and a log axis cannot show the exact
    zeros of q = 0 % and q = 33 %.
    """
    mg = sf.models_and_grids(data)
    fig, axes = plt.subplots(1, len(mg), figsize=(3.7 * len(mg), 4.6), sharey=True,
                             squeeze=False)
    axes = axes[0]
    for ax, (model, grids) in zip(axes, mg.items()):
        nonzero = []
        for j, meth in enumerate(QGA_LEVELS):
            xs, means = [], []
            for nb in grids:
                d = delta_vs_cga(data[(model, nb)], meth, seeds)
                x = nb + (j - 1.5) * 0.09
                sf.draw_seeds(ax, x, d, width=0.015)
                m, sd, _se, n = sf.stats(d)
                if n:
                    ax.errorbar([x], [m], yerr=[sd], fmt=MARKERS[meth], color=COLORS[meth],
                                ms=5, capsize=3, lw=1.3, zorder=3)
                    xs.append(x)
                    means.append(m)
                    if not zero_sd(d):
                        nonzero.append((meth.split('=')[1][:-1], nb))
            ax.plot(xs, means, STYLES[meth], color=COLORS[meth], lw=1.5, marker=MARKERS[meth],
                    ms=5, label=meth, zorder=2)
        ax.axhline(0, color='k', lw=0.8)
        ax.set_yscale('symlog', linthresh=1e-3, linscale=0.6)
        ax.set_xticks(grids)
        ax.set_xlabel('n_bits per parameter (genetic grid)')
        ax.grid(alpha=0.3)
        by_level = defaultdict(list)
        for q, nb in nonzero:
            by_level[q].append(nb)
        txt = ('sd = 0 across seeds\nin every point' if not nonzero else
               'sd > 0 across seeds only for\n' + '\n'.join(
                   f'q={q}: nb ' + ','.join(map(str, v)) for q, v in by_level.items()))
        ax.set_title(core.MODELS[model].label.split('  ')[0] + '\n' + txt, fontsize=8)
        report.append(f'  {model}: non-zero seed sd in '
                      + (', '.join(f'q{q} nb{nb}' for q, nb in nonzero) or 'none'))
    axes[0].set_ylabel(r'$\Delta\chi^2_{\rm grid}$ = QGA − CGA  (same seed)')
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.05))
    fig.suptitle('Best grid point found by each QGA rung relative to the classical GA '
                 '(chi2 before refinement)\n'
                 + sf.seed_caption(seeds) + '. y axis symmetric-log, linear below 1e-3; '
                 'positive = QGA worse.', fontsize=10)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return sf.save(fig, os.path.join(out, 'genetico_chi2grid_contra_nbits'))


def _plain_chi2_axis(ax, n_data=None):
    """chi2 axis without offset notation ([B-OFFSET]) and with n_data."""
    ax.ticklabel_format(axis='y', style='plain', useOffset=False)
    lab = r'$\chi^2_{\rm grid}$  (raw, best grid point, not refined)'
    if n_data:
        lab += f'\n{n_data} data points'
    ax.set_ylabel(lab, fontsize=9)


def fig_convergence(data, traces, model, seeds, out):
    """convergence_<model>_genetico: chi2_grid and grid-best parameters vs n_bits."""
    grids = sf.models_and_grids(data)[model]
    pn = core.MODELS[model].param_names
    fig, axes = plt.subplots(1, 1 + len(pn), figsize=(4.6 * (1 + len(pn)), 4.4), squeeze=False)
    axes = axes[0]
    n_data = None
    for j, meth in enumerate(METHODS):
        dx = (j - 2) * 0.08
        xs = [nb + dx for nb in grids]
        vals = [chi2_grid(data[(model, nb)], meth, seeds) for nb in grids]
        sf.series(axes[0], xs, vals, COLORS[meth], MARKERS[meth], meth, ls=STYLES[meth],
                  jitter=0.012)
        for k, p in enumerate(pn):
            pv = [[(traces[(model, nb)].get(s, {}).get(meth, {}).get('grid_params', {}).get(p))
                   for s in seeds] for nb in grids]
            sf.series(axes[1 + k], xs, pv, COLORS[meth], MARKERS[meth], meth, ls=STYLES[meth],
                      jitter=0.012)
        for nb in grids:
            r = next(iter(data[(model, nb)].get(meth, {}).values()), None)
            n_data = n_data or (r and r.get('n_data'))
    _plain_chi2_axis(axes[0], n_data)
    zeros = [f'nb{nb}' for nb in grids
             if all(zero_sd(chi2_grid(data[(model, nb)], m, seeds)) for m in METHODS)]
    axes[0].set_title(r'$\chi^2_{\rm grid}$ vs n_bits' + ('\nsd = 0 across seeds for every method at '
                      + ', '.join(zeros) if zeros else ''), fontsize=9)
    for k, p in enumerate(pn):
        axes[1 + k].set_ylabel(f'{p} of the best grid point')
        axes[1 + k].set_title(f'{p} (best individual, not refined)', fontsize=9)
        lim = axes[1 + k].get_ylim()            # keep the range on the data
        axes[1 + k].axhline(core.MODELS[model].fiducial[k], color='k', ls='--', lw=0.8)
        axes[1 + k].set_ylim(*lim)
    for ax in axes:
        ax.set_xticks(grids)
        ax.set_xlabel('n_bits per parameter')
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.suptitle(f'{core.MODELS[model].label} — genetic estimates vs grid resolution\n'
                 + sf.seed_caption(seeds) + '. CGA and QGA 0 % coincide (dashed)', fontsize=10)
    fig.tight_layout()
    return sf.save(fig, os.path.join(out, model, f'convergence_{model}{SUFFIX}'))


def fig_cost(data, model, seeds, out):
    """cost_<model>_genetico: Time_s vs n_bits per method."""
    grids = sf.models_and_grids(data)[model]
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    for j, meth in enumerate(METHODS):
        xs = [nb + (j - 2) * 0.08 for nb in grids]
        vals = [sf.per_seed(data[(model, nb)], meth, lambda r: sf.num(r, 'Time_s'), seeds)
                for nb in grids]
        sf.series(ax, xs, vals, COLORS[meth], MARKERS[meth], meth, ls=STYLES[meth], jitter=0.012)
    ax.set_yscale('log')
    ax.set_xticks(grids)
    ax.set_xlabel('n_bits per parameter')
    ax.set_ylabel('wall time per method [s]')
    ax.grid(alpha=0.3, which='both')
    ax.legend(fontsize=8)
    ax.set_title(f'{core.MODELS[model].label} — cost of the genetic ladder ON THE SIMULATOR '
                 '(statevector, CPU)\nnot a hardware cost; ' + sf.seed_caption(seeds), fontsize=9)
    return sf.save(fig, os.path.join(out, model, f'cost_{model}{SUFFIX}'))


def _trace_band(per_seed, key, param=None):
    """Median and 16/84 percentiles over seeds on the shared generation grid."""
    series = []
    for r in per_seed:
        y = r['params'].get(param) if param else r[key]
        if y and len(y) == len(r['gen']):
            series.append((np.array(r['gen']), np.array(y, float)))
    if not series:
        return None
    gens = sorted(set.intersection(*[set(g.tolist()) for g, _ in series]))
    Y = np.array([[dict(zip(g.tolist(), y))[k] for k in gens] for g, y in series])
    return (np.array(gens), np.median(Y, 0), np.percentile(Y, 16, 0),
            np.percentile(Y, 84, 0), len(series))


def floor_found(data, model, nb, seeds):
    """Lowest chi2_grid reached by any method in any seed of the cell.

    It is the best grid point FOUND, not a proven grid optimum (the grid is
    not enumerated).
    """
    v = [x for m in METHODS for x in chi2_grid(data[(model, nb)], m, seeds) if x is not None]
    return min(v) if v else None


def fig_fitness(data, traces, model, nb, seeds, out, report):
    """fitness_<model>_nb<N>_genetico: best chi2 per generation over seeds.

    Plotted as best chi2 minus the lowest grid chi2 found in the cell, on a
    symmetric-log axis: the rungs differ by 1e-3..1 on top of ~1060.
    """
    tr = traces.get((model, nb), {})
    fl = floor_found(data, model, nb, seeds)
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    flat = []
    top = 1e-3
    for meth in METHODS:
        runs = [tr[s][meth] for s in seeds if s in tr and meth in tr[s]]
        b = _trace_band(runs, 'best')
        if b is None:
            continue
        g, med, lo, hi, n = b
        top = max(top, float(np.max(hi - fl)))
        ax.fill_between(g, lo - fl, hi - fl, color=COLORS[meth], alpha=0.15, lw=0)
        ax.plot(g, med - fl, STYLES[meth], color=COLORS[meth], lw=2, marker=MARKERS[meth], ms=4,
                label=f'{meth}  (n={n})')
        # does the best individual ever improve after generation 0?
        stuck = sum(1 for r in runs if abs(r['best'][-1] - r['best'][0]) <= 1e-9)
        if stuck:
            flat.append(f'{meth}: {stuck}/{len(runs)}')
    ax.axhline(0, color='k', ls='--', lw=1.0,
               label=f'lowest grid χ² found in this cell = {fl:.4f} (not a proven optimum)')
    ax.set_yscale('symlog', linthresh=1e-3, linscale=0.6)
    ax.set_ylim(0, 4 * top)
    ax.set_xlabel('generation (logged every 10)')
    ax.set_ylabel(r'best $\chi^2_{\rm grid}$ − lowest found')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5, loc='center right')
    note = ('best never improves after generation 0 in: ' + '; '.join(flat)) if flat else ''
    ax.set_title(f'{core.MODELS[model].label} — genetic fitness, n_bits = {nb}\n'
                 f'median over {len(seeds)} seeds, band = 16–84 percentiles (task logs)'
                 + (f'\n{note}' if note else ''), fontsize=9)
    if flat:
        report.append(f'  {model} nb{nb}: {note}')
    return sf.save(fig, os.path.join(out, model, f'fitness_{model}_nb{nb}{SUFFIX}'))


def fig_genetic_convergence(data, traces, model, nb, seeds, out):
    """genetic_convergence_<model>_nb<N>_genetico.

    Top: best individual's parameters per generation (median + band).
    Bottom: per method, the best grid point at the end (mean ± sd across
    seeds) and, to its right, the final population's fitness-weighted mean
    with the population spread as a bar (a spread, not an uncertainty).
    """
    model_o = core.MODELS[model]
    pn = model_o.param_names
    tr = traces.get((model, nb), {})
    cell = data[(model, nb)]
    fig, axes = plt.subplots(2, len(pn), figsize=(4.4 * len(pn), 8.0), squeeze=False)
    for k, p in enumerate(pn):
        ax = axes[0, k]
        for meth in METHODS:
            runs = [tr[s][meth] for s in seeds if s in tr and meth in tr[s]]
            b = _trace_band(runs, None, param=p)
            if b is None:
                continue
            g, med, lo, hi, n = b
            ax.fill_between(g, lo, hi, color=COLORS[meth], alpha=0.15, lw=0)
            ax.plot(g, med, STYLES[meth], color=COLORS[meth], lw=1.8, label=meth)
        ax.set_xlabel('generation')
        ax.set_ylabel(f'{p} of the best individual')
        ax.grid(alpha=0.3)
        ax.ticklabel_format(axis='y', useOffset=False)
        ax = axes[1, k]
        for j, meth in enumerate(METHODS):
            gp = [tr.get(s, {}).get(meth, {}).get('grid_params', {}).get(p) for s in seeds]
            sf.draw_seeds(ax, j - 0.12, gp, width=0.04)
            sf.mean_bar(ax, j - 0.12, gp, COLORS[meth], marker=MARKERS[meth])
            pm = sf.per_seed(cell, meth, lambda r: sf.num(r, f'{p}_mean'), seeds)
            ps = sf.per_seed(cell, meth, lambda r: sf.num(r, f'{p}_std'), seeds)
            m_pm, _sd, _se, n = sf.stats(pm)
            if n:
                ax.errorbar([j + 0.16], [m_pm], yerr=[float(np.median([x for x in ps if x is not None]))],
                            fmt='_', color=COLORS[meth], alpha=0.55, ms=12, capsize=5, lw=3)
        ax.set_xticks(range(len(METHODS)))
        ax.set_xticklabels([m.replace('QGA ', 'QGA\n') for m in METHODS], fontsize=7)
        ax.set_ylabel(p)
        ax.grid(alpha=0.3)
        ax.ticklabel_format(axis='y', useOffset=False)
    axes[0, 0].legend(fontsize=7)
    axes[1, 0].legend(handles=[
        Line2D([0], [0], marker='o', color='k', ls='none',
               label='best grid point at the end: mean ± sd across seeds'),
        Line2D([0], [0], marker='_', color='k', alpha=0.55, ls='none', ms=12, mew=3,
               label='final population: weighted mean ± population spread\n'
                     '(median over seeds; a spread, NOT an uncertainty)')], fontsize=7)
    fig.suptitle(f'{model_o.label} — genetic convergence, n_bits = {nb}\n'
                 f'top: median over {len(seeds)} seeds, band 16–84 % (task logs); '
                 'bottom: final estimates, grey = individual seeds', fontsize=10)
    fig.tight_layout()
    return sf.save(fig, os.path.join(out, model, f'genetic_convergence_{model}_nb{nb}{SUFFIX}'))


# ── tables ───────────────────────────────────────────────────────────────────
def summary(data, seeds, report):
    """genetic_summary rows + the report text (chi2_grid per cell and method)."""
    rows = []
    n_same = n_cells = 0
    report.append('\n== chi2_grid per cell (mean ± sd across seeds); '
                  'paired Δ vs CGA: mean (se), t, [seeds where QGA is worse] ==')
    for model, grids in sf.models_and_grids(data).items():
        for nb in grids:
            cell = data[(model, nb)]
            n_cells += 1
            same = identical_rows(cell, 'CGA', 'QGA (q=0%)', seeds)
            n_same += same
            parts = []
            for meth in METHODS:
                m, sd, se, n = sf.stats(chi2_grid(cell, meth, seeds))
                pr = paired(cell, meth, seeds) if meth != 'CGA' else None
                rows.append(dict(model=model, n_bits=nb, method=meth, chi2_grid_mean=m,
                                 chi2_grid_sd=sd, chi2_grid_se=se, n=n,
                                 delta_vs_cga_mean=pr and pr['mean'],
                                 delta_vs_cga_sd=pr and pr['sd'],
                                 delta_vs_cga_se=pr and pr['se'],
                                 paired_t=pr and pr['t'],
                                 seeds_worse=pr and pr['worse']))
                if meth == 'CGA':
                    parts.append(f'CGA {m:.3f} ± {sd:.3f}')
                else:
                    t = pr['t']
                    ts = 'all equal' if t is None else f't={t:+.1f}'
                    parts.append(f'{meth.split("=")[1][:-1]:>4}: {pr["mean"]:+.3f} '
                                 f'({pr["se"]:.3f}, {ts}) [{pr["worse"]}/{pr["n"]}]')
            report.append(f'  {model:5s} nb{nb}  ' + ' | '.join(parts))
    report.insert(1, f'  CGA == QGA (q=0%) (every value, every seed): {n_same}/{n_cells} cells '
                     f'({n_same * len(seeds)}/{n_cells * len(seeds)} runs)')
    return rows


def write_csv(rows, path):
    cols = list(rows[0].keys())
    with open(path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: ('' if v is None else f'{v:.6g}' if isinstance(v, float) else v)
                        for k, v in r.items()})
    return path


def refined_and_time(data, seeds, report):
    """Report lines: refined chi2 range per model and max time per task."""
    report.append('\n== refined chi2 (should be the same for every method) ==')
    for model, grids in sf.models_and_grids(data).items():
        v = sorted({round(x, 4) for nb in grids for m in METHODS
                    for x in sf.per_seed(data[(model, nb)], m, lambda r: sf.num(r, 'chi2'), seeds)
                    if x is not None})
        report.append(f'  {model}: {v}')
    report.append('\n== simulator time per task (sum over methods, max over seeds) [h] ==')
    for model, grids in sf.models_and_grids(data).items():
        parts = []
        for nb in grids:
            tot = [sum(x for x in (sf.num(data[(model, nb)].get(m, {}).get(s), 'Time_s')
                                   for m in METHODS) if x is not None) for s in seeds]
            parts.append(f'nb{nb} {max(tot) / 3600:.3f}')
        report.append(f'  {model}: ' + ', '.join(parts))


def gif_index(root, out, seeds):
    """gif_index.html: every evolution GIF by model, n_bits and seed.

    Links are relative to ``out`` so the page works from the figures folder.
    Returns the path, or None when no GIF is found.
    """
    found = {}
    for p in glob.glob(os.path.join(root, 'genetic_s*', 'genetic_*_nb*', 'model_*', '*.gif')):
        m = re.search(r'genetic_s(\d+)[/\\]genetic_([a-z0-9]+)_nb(\d+)', p)
        if m:
            found[(m.group(2), int(m.group(3)), m.group(1))] = p
    if not found:
        return None
    cells = sorted({(m, nb) for m, nb, _ in found},
                   key=lambda c: (sf.MODEL_ORDER.index(c[0]) if c[0] in sf.MODEL_ORDER else 99, c[1]))
    rows = []
    for m, nb in cells:
        tds = []
        for s in seeds:
            p = found.get((m, nb, s))
            if p:
                rel = os.path.relpath(p, out).replace('\\', '/')
                tds.append(f'<td><a href="{html.escape(rel)}">s{s}</a></td>')
            else:
                tds.append('<td class="miss">—</td>')
        rows.append(f'<tr><th>{m}</th><td>{nb}</td>{"".join(tds)}</tr>')
    page = ('<!doctype html><meta charset="utf-8"><title>Genetic GIFs</title>'
            '<style>body{font-family:sans-serif;margin:16px}table{border-collapse:collapse}'
            'td,th{border:1px solid #bbb;padding:4px 10px;text-align:center}'
            '.miss{color:#b00}</style>'
            f'<h2>Genetic evolution GIFs — {len(found)} files</h2>'
            '<p>One GIF per task (all methods of the cell). Click to open.</p>'
            '<table><tr><th>model</th><th>n_bits</th>'
            + ''.join(f'<th>seed {s}</th>' for s in seeds) + '</tr>' + ''.join(rows) + '</table>')
    path = os.path.join(out, 'gif_index.html')
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write(page)
    return path


# ── driver ───────────────────────────────────────────────────────────────────
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('root', help='folder holding genetic_s<NN>/')
    ap.add_argument('--out', required=True)
    ap.add_argument('--seeds', default=None, help="e.g. '43-50' (default: all)")
    ap.add_argument('--gif-index-only', action='store_true',
                    help='only write gif_index.html')
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    data, taskdirs = sf.load_campaign(a.root, sf.parse_seeds(a.seeds),
                                      pattern='genetic_s*', grid='nb')
    if not data:
        print('no genetic rows found', file=sys.stderr)
        return 1
    seeds = sf.seeds_of(data)
    if a.gif_index_only:
        print(gif_index(a.root, a.out, seeds))
        return 0
    traces = load_traces(taskdirs, seeds)
    mg = sf.models_and_grids(data)
    report = [f'seed_figures_genetic: {len(seeds)} seeds {seeds}; cells: '
              + ', '.join(f'{m} nb{",".join(map(str, g))}' for m, g in mg.items())]
    # the log's grid optimum must be the CSV chi2_grid (same run)
    mism = 0
    for cell, by_seed in traces.items():
        for s, runs in by_seed.items():
            for meth, r in runs.items():
                c = sf.num(data[cell].get(meth, {}).get(s), 'chi2_grid')
                if c is not None and r['grid_chi2'] is not None and abs(c - r['grid_chi2']) > 5e-3:
                    mism += 1
    report.append(f'  log grid chi2 vs CSV chi2_grid: {mism} mismatches (> 0.005)')
    rows = summary(data, seeds, report)
    files = [write_csv(rows, os.path.join(a.out, 'genetic_summary.csv'))]
    report.append('\n== Δchi2_grid figure: seed spread ==')
    files.append(fig_delta_vs_nbits(data, seeds, a.out, report))
    report.append('\n== fitness traces ==')
    for m, grids in mg.items():
        files.append(fig_convergence(data, traces, m, seeds, a.out))
        files.append(fig_cost(data, m, seeds, a.out))
        for nb in grids:
            files.append(fig_fitness(data, traces, m, nb, seeds, a.out, report))
            files.append(fig_genetic_convergence(data, traces, m, nb, seeds, a.out))
    refined_and_time(data, seeds, report)
    g = gif_index(a.root, a.out, seeds)
    report.append(f'\nGIF index: {g or "no GIFs found under the campaign folder"}')
    text = '\n'.join(report) + '\n'
    with open(os.path.join(a.out, 'genetic_report.txt'), 'w', encoding='utf-8') as fh:
        fh.write(text)
    print(text)
    print(f'{len(files)} figures/tables written to {a.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
