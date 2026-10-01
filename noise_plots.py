#!/usr/bin/env python3
"""noise_plots.py — the noise axis, read from results that already exist.

This script does NOT run any algorithm nor touch any file of the campaign.
It reads the CSVs the campaign already wrote and draws three things:

  (1) metric  vs  nqpp        — one curve per noise rung  (samplers)
  (2) metric  vs  n_bits      — one curve per noise rung  (genetic)
  (3) at FIXED nqpp / n_bits  — the noise rungs side by side, ladder rung by ladder rung

The question all three answer is the same: how much does noise move the
result, and does that shift grow with the register size or not?

────────────────────────────────────────────────────────────────────────────
WHERE THE NOISE RUNG COMES FROM
────────────────────────────────────────────────────────────────────────────
From the FOLDER NAME (`..._noise-<rung>`), not from the `noise` column of
the CSV, and this is deliberate:

  * `none-counts` is a rung of the project's axis but it is NOT a value of
    the column: the CSV says `noise='none'` + `proposal_route='counts'`,
    because that is literally what ran. The folder name is the only place
    where the control appears as such.
  * [B-PROV-GEN] until this correction, EVERY genetic row wrote
    `noise='none'` no matter what, because the CSV writer read the global
    of another module. Campaigns already run have that column wrong.

So the folder rules and the column is used only to WARN when the two
disagree. That keeps old campaigns perfectly usable: the scientific data
was never wrong, only the label, and the label is recovered from the folder
name without re-running anything.

────────────────────────────────────────────────────────────────────────────
WHAT IS MEASURED
────────────────────────────────────────────────────────────────────────────
  * `shift`           — |p(noisy) − p(ideal)| / sigma(ideal), per parameter,
    against the SAME cell (same model, same ladder rung, same nqpp) run
    without noise. It is the only metric that says whether noise moves the
    SCIENCE.
  * `acceptance`      — Metropolis acceptance fraction (QMCMC only).
  * `final_KL`        — final KL of the variational method (QVMC/VI only).
  * `chi2_grid`       — [B-REFINE] chi2 of the best point on the GRID,
    before refinement. It is the only genetic number that distinguishes one
    ladder rung from another: the local refiner takes all four to the same
    continuous minimum.
  * `Time_s`          — cost. The noise axis is very expensive and it pays
    to see it.

Usage:
    python noise_plots.py CAMPAIGN_FOLDER [...] [--out FIGS]
    python noise_plots.py results/hpc_20260907_130425 --out noise_figures
"""
from __future__ import annotations

import argparse
import csv
import glob

import campaign_io
import math
import os
import re
import sys
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt          # noqa: E402

# ── axis vocabulary ──────────────────────────────────────────────────────
# The order is that of the axis, from ideal to noisiest, and it is the one
# used by the legend and by the x axis of the fixed-grid figures.
NOISE_ORDER = ['none', 'none-counts', 'readout', 'full']

# Colour per rung. `none` and `none-counts` share a hue on purpose: they are
# the SAME noise level read through two different routes (amplitudes vs
# counts), not two levels. They are told apart by the line style, not by
# the colour, so the figure does not suggest a gradation that does not exist.
NOISE_COLORS = {
    'none':         '#1f77b4',
    'none-counts':  '#1f77b4',
    'readout':      '#ff7f0e',
    'full':         '#d62728',
}
NOISE_LINESTYLES = {
    'none':         'solid',
    'none-counts':  'dashed',
    'readout':      'solid',
    'full':         'solid',
}
# Real backends (FakeBrisbane, …) get these fallback colours: they are not
# a synthetic rung and have no fixed place on the scale.
BACKEND_COLORS = ['#2ca02c', '#9467bd', '#8c564b', '#e377c2']

# Columns that are NOT cosmological parameters even if they end in _mean/_std.
_NOT_PARAMS = {'Om_mean', 'Om_std'} - {'Om_mean', 'Om_std'}   # (none)


# ═════════════════════════════════════════════════════════════════════════
# 1.  READING
# ═════════════════════════════════════════════════════════════════════════

def _save_figure(fig, path: str) -> str:
    """[B-PDF] Save the figure as PNG and also as PDF.

    The PNG is for looking at on screen; the PDF is vector and is what the
    paper needs (a 150 dpi PNG looks pixelated in print, and several
    journals reject it). Both come from the SAME `fig` object, so they
    cannot disagree: there is no way for the PDF to show an older version
    of what the PNG shows.

    Args:
        fig: the matplotlib figure, already finished.
        path: path of the PNG. The PDF is written next to it, same stem.

    Returns:
        The path of the PNG (what the rest of the code already expected).
    """
    fig.savefig(path, dpi=150)
    fig.savefig(os.path.splitext(path)[0] + '.pdf')
    return path


def _num(txt):
    """float or None. The CSV uses '' and '—' for 'not applicable here'."""
    if txt is None:
        return None
    t = str(txt).strip()
    if t in ('', '—', '-', 'nan', 'None'):
        return None
    try:
        v = float(t)
    except ValueError:
        return None
    return v if math.isfinite(v) else None


def param_columns(fields) -> list:
    """Names of the cosmological parameters present in the header.

    The per-run schema names the columns (`Om_mean`, `H0_mean`,
    `w0_mean`…); the cumulative one calls them `p1_mean`…`p4_mean`. Both
    are accepted, because different folders of the same campaign use one
    or the other.
    """
    return [c[:-5] for c in fields
            if c.endswith('_mean') and f'{c[:-5]}_std' in fields]


def load_campaign(root: str) -> list:
    """All rows of one campaign, each with its provenance resolved.

    [E-READ] Delegates to `campaign_io.read_campaign` (optional grid/noise
    tags, no double counting, current and legacy CSV names) and maps its
    keys to the names used here: `_family`, `_model`, `_grid`, `_noise`
    (from the folder), `_noise_csv` (the CSV column, kept to cross-check),
    `_source` (the CSV path) and `_camp`.
    """
    rows = []
    for r in campaign_io.read_campaign(root):
        r['_family'] = r['_fam']
        r['_model'] = r['_mod']
        r['_grid'] = r['_g']
        r['_noise'] = r['_noi']
        r['_noise_csv'] = (r.get('noise') or '').strip()
        r['_source'] = r['_path']
        rows.append(r)
    return rows


def warn_inconsistent_labels(rows) -> None:
    """Report the rows whose `noise` column does not say what their folder says.

    It is not cosmetic: if someone groups by the column instead of by the
    folder, all the genetic rungs fall into the same group and the
    differences due to noise vanish without anything failing.
    """
    bad = defaultdict(int)
    for r in rows:
        expected = 'none' if r['_noise'] == 'none-counts' else r['_noise']
        if r['_noise_csv'] and r['_noise_csv'] != expected:
            bad[(r['_family'], r['_noise'], r['_noise_csv'])] += 1
    if not bad:
        return
    print("\n  WARNING [B-PROV-GEN] — the `noise` column does not match the "
          "folder:")
    for (fam, folder, col), n in sorted(bad.items()):
        print(f"    {fam:8s}  folder says '{folder}'  column says "
              f"'{col}'   ({n} rows)")
    print("    The FOLDER is used. The scientific data is valid; only the "
          "label was wrong,\n    and it is recovered without re-running anything.")


# ═════════════════════════════════════════════════════════════════════════
# 2.  METRICS
# ═════════════════════════════════════════════════════════════════════════

def shift_in_sigmas(row, ideal) -> float | None:
    """How far noise moved the parameter, in sigmas of the ideal run.

    The MAXIMUM over the model's parameters is taken: if noise biases even
    one of them, the result is already biased. The divisor is the IDEAL
    sigma, not the noisy one: noise usually widens the posterior, and
    dividing by the noisy one would hide the bias behind its own spread.

    Args:
        row: noisy row.
        ideal: row of the same cell run without noise.

    Returns:
        float, or None if there are no comparable parameters.
    """
    worst = None
    for p in param_columns(row.keys()):
        a, b = _num(row.get(f'{p}_mean')), _num(ideal.get(f'{p}_mean'))
        s = _num(ideal.get(f'{p}_std'))
        if a is None or b is None or not s:
            continue
        d = abs(a - b) / s
        worst = d if worst is None else max(worst, d)
    return worst


METRICS = {
    'shift':          ('maximum shift  |Δp| / σ(ideal)', None),
    'acceptance':     ('acceptance fraction', 'acceptance'),
    'final_KL':       ('final KL', 'final_KL'),
    'chi2_grid':      ('χ² of the best grid point (unrefined)',
                       'chi2_grid'),
    'Time_s':         ('wall time (s)', 'Time_s'),
}


def metric_value(row, metric, ideal_index):
    """Value of a metric in a row, or None if it does not apply there."""
    if metric == 'shift':
        # [E-QPU9] For the genetic family `*_std` is the spread of the final
        # population, not an uncertainty, so a shift "in sigmas" is
        # meaningless there (it produced 223-sigma artefacts).
        if row['_family'] == 'genetic':
            return None
        ideal = ideal_index.get(
            campaign_io.cell_key_without_noise(row) + (row['Method'],))
        return None if ideal is None else shift_in_sigmas(row, ideal)
    return _num(row.get(METRICS[metric][1]))


def ideal_cell_index(rows) -> dict:
    """The noise-FREE run of each cell, to measure against it.

    The reference is `none` — the amplitude route —, not `none-counts`. That
    way the shift of `none-counts` measures exactly the effect of changing
    the readout route, which is what that control exists for.
    """
    # [E-QPU8] Keyed by campaign, dataset, prior and seed as well, so a noisy
    # row is never measured against another campaign's ideal row.
    return {campaign_io.cell_key_without_noise(r) + (r['Method'],): r
            for r in rows if r['_noise'] == 'none'}


# ═════════════════════════════════════════════════════════════════════════
# 3.  FIGURES
# ═════════════════════════════════════════════════════════════════════════

def _style(rung, spare):
    if rung in NOISE_COLORS:
        return NOISE_COLORS[rung], NOISE_LINESTYLES[rung]
    return BACKEND_COLORS[spare % len(BACKEND_COLORS)], 'dashdot'


def _merged_legend(axes):
    """Handles and labels of ALL panels, without repeats and in axis order.

    Reading the legend of a single panel loses the rungs that do not appear
    in that panel — and those are exactly the interesting ones, because they
    are the ones that only got run on some ladder rungs.
    """
    seen = {}
    for axes_row in axes:
        for ax in axes_row:
            for h, e in zip(*ax.get_legend_handles_labels()):
                seen.setdefault(e, h)
    order = _order_rungs(set(seen))
    return [seen[e] for e in order], order


def _order_rungs(rungs):
    """Synthetic rungs in axis order; backends afterwards, alphabetically."""
    known = [p for p in NOISE_ORDER if p in rungs]
    others = sorted(p for p in rungs if p not in NOISE_ORDER)
    return known + others


def figure_vs_grid(rows, family, model, metric, ideal_index,
                   out_dir) -> str | None:
    """(1) and (2): the metric against nqpp / n_bits, one curve per rung.

    One panel per method (ladder rung), because mixing the classical method
    with the 100% quantum one on a single axis hides exactly what we want
    to see.
    """
    sub = [r for r in rows
           if r['_family'] == family and r['_model'] == model]
    if not sub:
        return None

    methods = sorted({r['Method'] for r in sub})
    data = defaultdict(lambda: defaultdict(list))     # method → rung → pts
    for r in sub:
        v = metric_value(r, metric, ideal_index)
        if v is not None and r['_grid'] is not None:
            data[r['Method']][r['_noise']].append((r['_grid'], v))
    methods = [m for m in methods if data[m]]
    if not methods:
        return None

    x_label = 'nqpp  (qubits per parameter)' if family == 'samplers' \
        else 'n_bits  (bits per parameter)'
    ncol = min(len(methods), 3)
    nrow = math.ceil(len(methods) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.6 * nrow),
                             squeeze=False)
    rungs = _order_rungs({r['_noise'] for r in sub})

    for k, met in enumerate(methods):
        ax = axes[k // ncol][k % ncol]
        spare = 0
        for p in rungs:
            pts = sorted(data[met].get(p, []))
            if not pts:
                continue
            c, ls = _style(p, spare)
            if p not in NOISE_COLORS:
                spare += 1
            xs, ys = zip(*pts)
            ax.plot(xs, ys, marker='o', ms=4, lw=1.6, color=c, ls=ls, label=p)
        ax.set_title(met, fontsize=10)
        ax.set_xlabel(x_label, fontsize=8)
        ax.set_ylabel(METRICS[metric][0], fontsize=8)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=.25, lw=.5)
        if metric in ('Time_s', 'final_KL'):
            ax.set_yscale('log')
        if metric == 'shift':
            # 1σ: below this line noise does not move the science.
            ax.axhline(1.0, color='0.4', lw=.9, ls=':')
    for k in range(len(methods), nrow * ncol):
        axes[k // ncol][k % ncol].axis('off')

    # The legend is built from the UNION of the panels: a rung that only
    # appears in one method (e.g. the real backend, which was only run on
    # some ladder rungs) disappeared from the legend if only the first axis
    # was read.
    handles, labels = _merged_legend(axes)
    fig.legend(handles, labels, loc='lower center', ncol=len(labels),
               fontsize=8, frameon=False, title='noise rung',
               title_fontsize=8)
    fig.suptitle(f'{model.upper()} — {METRICS[metric][0]} against the '
                 f'register size', fontsize=11)
    fig.tight_layout(rect=(0, .07, 1, .96))
    name = f'noise_vs_grid_{family}_{model}_{metric}.png'
    path = os.path.join(out_dir, name)
    _save_figure(fig, path)
    plt.close(fig)
    return path


def figure_at_fixed_grid(rows, family, model, grid, metrics,
                         ideal_index, out_dir) -> str | None:
    """(3): at FIXED nqpp / n_bits, the noise rungs side by side, per ladder rung.

    Grouped bars: one group per ladder method, one bar per noise rung, one
    panel per metric. It is the figure that answers 'which of my methods
    withstands noise?' without the register size getting in the way — and it
    is ONE figure per cell, not one per metric, because the metrics are only
    read well when compared with each other.
    """
    sub = [r for r in rows
           if r['_family'] == family and r['_model'] == model
           and r['_grid'] == grid]
    if not sub:
        return None
    rungs = _order_rungs({r['_noise'] for r in sub})

    panels = []
    for metric in metrics:
        table = {}
        for r in sub:
            v = metric_value(r, metric, ideal_index)
            if v is not None:
                table[(r['Method'], r['_noise'])] = v
        if table:
            methods = [m for m in sorted({r['Method'] for r in sub})
                       if any((m, p) in table for p in rungs)]
            panels.append((metric, table, methods))
    if not panels:
        return None

    bar_width = 0.8 / max(len(rungs), 1)
    fig_width = max(1.5 * max(len(m) for _, _, m in panels) + 3.0, 6.0)
    fig, axes = plt.subplots(len(panels), 1, squeeze=False,
                             figsize=(fig_width, 3.4 * len(panels)))
    for k, (metric, table, methods) in enumerate(panels):
        ax = axes[k][0]
        spare = 0
        for j, p in enumerate(rungs):
            c, _ = _style(p, spare)
            if p not in NOISE_COLORS:
                spare += 1
            xs, ys = [], []
            for i, m in enumerate(methods):
                if (m, p) in table:
                    xs.append(i + j * bar_width - 0.4 + bar_width / 2)
                    ys.append(table[(m, p)])
            # `none-counts` is hatched: it shares the noise level with `none`
            # and what changes is the readout route, not the rung.
            ax.bar(xs, ys, width=bar_width * .92, color=c,
                   label=p if xs else None,
                   hatch='//' if p == 'none-counts' else None,
                   edgecolor='white' if p == 'none-counts' else 'none',
                   alpha=.9)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=15, ha='right', fontsize=8)
        ax.set_ylabel(METRICS[metric][0], fontsize=8)
        ax.tick_params(labelsize=8)
        ax.grid(axis='y', alpha=.25, lw=.5)
        if metric in ('Time_s', 'final_KL'):
            ax.set_yscale('log')
        if metric == 'shift':
            ax.axhline(1.0, color='0.4', lw=.9, ls=':')

    label = 'nqpp' if family == 'samplers' else 'n_bits'
    handles, labels = _merged_legend(axes)
    fig.legend(handles, labels, loc='lower center', ncol=len(labels),
               fontsize=8, frameon=False, title='noise rung',
               title_fontsize=8)
    fig.suptitle(f'{model.upper()} — {label}={grid}: effect of noise '
                 f'rung by rung', fontsize=11)
    fig.tight_layout(rect=(0, .06, 1, .96))
    name = f'noise_fixed_{family}_{model}_{label}{grid}.png'
    path = os.path.join(out_dir, name)
    _save_figure(fig, path)
    plt.close(fig)
    return path


# ═════════════════════════════════════════════════════════════════════════
# 4.  ORCHESTRATION
# ═════════════════════════════════════════════════════════════════════════

def useful_metrics(rows, family) -> list:
    """Metrics this family actually fills; empty ones are not drawn."""
    sub = [r for r in rows if r['_family'] == family]
    out = [] if family == 'genetic' else ['shift']   # [E-QPU9]
    for m, (_, col) in METRICS.items():
        if col and any(_num(r.get(col)) is not None for r in sub):
            out.append(m)
    return out


def build(folders, out_dir) -> list:
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for c in folders:
        if not os.path.isdir(c):
            print(f"  does not exist: {c}")
            continue
        f = load_campaign(c)
        print(f"  {os.path.basename(c)}: {len(f)} rows in "
              f"{len({r['_noise'] for r in f})} rungs")
        rows += f
    if not rows:
        print("  no row with a recognisable task name was read.")
        return []

    warn_inconsistent_labels(rows)
    ideal = ideal_cell_index(rows)
    made = []

    for family in ('samplers', 'genetic'):
        sub = [r for r in rows if r['_family'] == family]
        if not sub:
            continue
        models = sorted({r['_model'] for r in sub})
        metrics = useful_metrics(rows, family)
        print(f"\n  [{family}] models={','.join(models)} | "
              f"metrics={','.join(metrics)}")
        for model in models:
            # Only meaningful if that model has MORE than one rung.
            rungs = {r['_noise'] for r in sub if r['_model'] == model}
            if len(rungs) < 2:
                print(f"    {model}: a single rung ({rungs}), skipped")
                continue
            for metric in metrics:
                r1 = figure_vs_grid(rows, family, model, metric,
                                    ideal, out_dir)
                if r1:
                    made.append(r1)
            # Fixed grid: on the grids where there really is a comparison.
            for grid in sorted({r['_grid'] for r in sub
                                if r['_model'] == model
                                and r['_grid'] is not None}):
                present = {r['_noise'] for r in sub
                           if r['_model'] == model and r['_grid'] == grid}
                if len(present) < 2:
                    continue
                r2 = figure_at_fixed_grid(rows, family, model, grid,
                                          metrics, ideal, out_dir)
                if r2:
                    made.append(r2)
    return made


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description='Noise-axis figures from results that already exist. '
                    'Runs nothing and does not modify the campaign.')
    p.add_argument('folders', nargs='+',
                   help='campaign master folders (hpc_YYYYMMDD_HHMMSS)')
    p.add_argument('--out', default='noise_figures',
                   help='folder where the figures are written')
    a = p.parse_args(argv)

    print("\n" + "=" * 72)
    print("  NOISE AXIS — figures from existing results")
    print("=" * 72)
    made = build(a.folders, a.out)
    print("\n" + "-" * 72)
    print(f"  {len(made)} figures in {os.path.abspath(a.out)}/")
    print("-" * 72 + "\n")
    return 0 if made else 1


if __name__ == '__main__':
    sys.exit(main())
