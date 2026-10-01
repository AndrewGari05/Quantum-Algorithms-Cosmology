#!/usr/bin/env python3
"""compare_algorithms.py — which number to compare one method against another with.

Reads results that already exist; it does NOT run anything. It answers the
question "does the quantum method win or lose?" with the estimator that fits
each family, and NOT with the same one for all, because they do not measure
the same thing:

  * SAMPLERS (MCMC / QMCMC) -> ESS, time, and ESS per second.
    The ESS (effective sample size) says how many INDEPENDENT samples the
    chain is worth. It is the standard currency for comparing chains: a
    chain of 10000 highly correlated steps is worth less than one of 2000
    independent steps.

  * VARIATIONAL (VI / QVMC) -> `final_KL` at a matched circuit budget.
    [B-ESSCOMP] The ESS of this family is the number of shots (constant,
    12288 in the campaign) and does NOT measure quality: using it as if it
    did is a misreading.

  * GENETIC (CGA / QGA) -> `chi2_grid`, the chi2 of the best point on the
    GRID, before refinement. [B-REFINE] The refined chi2 is identical on all
    four rungs and distinguishes nothing.

────────────────────────────────────────────────────────────────────────────
THE SECONDS TRAP
────────────────────────────────────────────────────────────────────────────
The ESS/s ratio penalises the quantum method for the cost of the SIMULATOR,
which has nothing to do with what it would cost on hardware. That is why the
script reports the two things separately:

  * ESS ratio   -> hardware independent (at matched steps, how many
    effective samples each one produced). THIS is the number for the paper.
  * ESS/s ratio -> useful for planning campaigns, not for claiming anything
    about quantum computing.

Usage:
    python compare_algorithms.py CAMPAIGN_FOLDER [...] [--out FIGS]
"""
from __future__ import annotations

import argparse
import csv
import glob

import campaign_io
import math
import os
import re
import statistics
import sys
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt          # noqa: E402

# Semantics inherited from the circuits document: indigo = quantum,
# ochre = classical. Here they encode POLARITY around 1 (who wins), with
# grey for a tie — it is a diverging scale, not a categorical one.
C_QUANTUM_WINS = '#5347C4'
C_CLASSICAL_WINS = '#A8651F'
C_TIE = '#8A8A94'
C_REF = '#3A3A44'

SAMPLER_PAIRS = [
    ('Classical MCMC', 'QMCMC 100%'),
    ('Classical MCMC', 'QMCMC 50%'),
]


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


def _num(x):
    try:
        v = float(str(x).strip())
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def read_rows(roots) -> list:
    """All rows of every campaign, with model/grid/noise level resolved.

    [E-READ] Delegates to `campaign_io.read_campaign`, which accepts task
    folders with or without the grid and noise tags (HPC-8/QPU-10) and never
    reads the cumulative CSV twice (QPU-6).
    """
    return campaign_io.read_campaigns(roots)


def sampler_ratios(rows, rung='none') -> dict:
    """Quantum/classical ratios per MODEL, paired cell by cell.

    Grouped by model and not by cell because the quantum proposal uses
    d qubits (d = model parameters) and does not depend on `nqpp`: cells
    that differ only in nqpp repeat exactly the same comparison. Counting
    them as independent observations would inflate the sample size from 5
    to 87.
    """
    cells = defaultdict(dict)
    for r in rows:
        if r['_fam'] != 'samplers' or r['_noi'] != rung:
            continue
        cells[campaign_io.cell_key(r)][r['Method']] = r

    by_model = defaultdict(lambda: defaultdict(list))
    for key, v in cells.items():
        mod, g = key[5], key[6]
        for classical, quantum in SAMPLER_PAIRS:
            c, q = v.get(classical), v.get(quantum)
            if not (c and q):
                continue
            ec, eq = _num(c['ESS']), _num(q['ESS'])
            tc, tq = _num(c['Time_s']), _num(q['Time_s'])
            if None in (ec, eq, tc, tq) or 0 in (ec, tc, tq):
                continue
            d = by_model[(classical, quantum)][mod]
            d.append({'ess': eq / ec, 'time': tq / tc,
                      'eff': (eq / tq) / (ec / tc),
                      'acc_c': _num(c['acceptance']),
                      'acc_q': _num(q['acceptance'])})
    return by_model


def _med(vals):
    return statistics.median(vals) if vals else float('nan')


def method_table(by_model) -> None:
    for (classical, quantum), per_mod in by_model.items():
        print(f"\n{'='*72}\n  {quantum}  vs  {classical}"
              f"   (ideal rung)\n{'='*72}")
        print(f"  {'model':8s} {'cells':>7s} {'ESS q/c':>9s} "
              f"{'time q/c':>11s} {'ESS/s q/c':>10s} "
              f"{'accept c':>8s} {'accept q':>8s}")
        all_ess, all_eff = [], []
        for mod in sorted(per_mod):
            v = per_mod[mod]
            e = _med([x['ess'] for x in v])
            t = _med([x['time'] for x in v])
            f = _med([x['eff'] for x in v])
            ac = _med([x['acc_c'] for x in v if x['acc_c'] is not None])
            aq = _med([x['acc_q'] for x in v if x['acc_q'] is not None])
            all_ess.append(e)
            all_eff.append(f)
            print(f"  {mod:8s} {len(v):7d} {e:9.4f} {t:11.2f} {f:10.4f} "
                  f"{ac:8.4f} {aq:8.4f}")
        if all_ess:
            wins = sum(1 for x in all_ess if x > 1)
            print(f"\n  ESS:  median over models = {_med(all_ess):.4f}   "
                  f"({wins}/{len(all_ess)} models in favour of the quantum method)")
            print(f"  ESS/s: median = {_med(all_eff):.4f}  -> the classical method "
                  f"is {1/_med(all_eff):.1f}x more efficient on the simulator")
            print("\n  Reading: the ESS ratio is the honest comparison "
                  "(hardware\n  independent). The ESS/s ratio measures the cost "
                  "of Aer, not of quantum\n  computing, and supports "
                  "no claim about real hardware.")


def method_figure(by_model, out_dir) -> list:
    """One figure per pair: ESS ratio and efficiency ratio."""
    os.makedirs(out_dir, exist_ok=True)
    made = []
    for (classical, quantum), per_mod in by_model.items():
        mods = sorted(per_mod)
        if not mods:
            continue
        ess = [_med([x['ess'] for x in per_mod[m]]) for m in mods]
        eff = [_med([x['eff'] for x in per_mod[m]]) for m in mods]

        fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.6))
        y = range(len(mods))

        # (a) quality per step — linear scale, the range is narrow
        ax = axes[0]
        for i, v in zip(y, ess):
            c = C_QUANTUM_WINS if v > 1.01 else (C_CLASSICAL_WINS if v < 0.99 else C_TIE)
            ax.barh(i, v - 1.0, left=1.0, height=.55, color=c)
        ax.axvline(1.0, color=C_REF, lw=1.1)
        ax.set_yticks(list(y))
        ax.set_yticklabels([m.upper() for m in mods], fontsize=9)
        ax.set_xlabel('quantum ESS / classical ESS', fontsize=9)
        ax.set_title('(a) sampling quality, at matched steps',
                     fontsize=10, loc='left')
        lo, hi = min(ess), max(ess)
        pad = max(0.03, (hi - lo) * 0.45)
        ax.set_xlim(min(lo, 1.0) - pad, max(hi, 1.0) + pad)
        ax.grid(axis='x', alpha=.25, lw=.5)
        ax.tick_params(labelsize=8)
        for i, v in zip(y, ess):
            ax.text(v + (0.012 if v >= 1 else -0.012), i, f'{v:.3f}',
                    va='center', ha='left' if v >= 1 else 'right',
                    fontsize=8, color=C_REF)

        # (b) wall-clock efficiency — log, the range spans an order of magnitude
        ax = axes[1]
        for i, v in zip(y, eff):
            c = C_QUANTUM_WINS if v > 1 else C_CLASSICAL_WINS
            ax.barh(i, v, height=.55, color=c)
        ax.axvline(1.0, color=C_REF, lw=1.1)
        ax.set_xscale('log')
        ax.set_yticks(list(y))
        ax.set_yticklabels([])
        ax.set_xlabel('quantum (ESS/s) / classical (ESS/s)   [log scale]',
                      fontsize=9)
        ax.set_title('(b) efficiency on the SIMULATOR', fontsize=10, loc='left')
        ax.grid(axis='x', alpha=.25, lw=.5)
        ax.tick_params(labelsize=8)
        for i, v in zip(y, eff):
            ax.text(v * 1.08, i, f'{v:.3f}', va='center', ha='left',
                    fontsize=8, color=C_REF)

        fig.suptitle(f'{quantum} vs {classical} — ideal rung',
                     fontsize=11)
        # Legend by polarity: identity is not carried by colour alone.
        from matplotlib.patches import Patch
        fig.legend(handles=[Patch(color=C_QUANTUM_WINS, label='quantum wins'),
                            Patch(color=C_CLASSICAL_WINS, label='classical wins'),
                            Patch(color=C_TIE, label='tie (±1%)')],
                   loc='lower center', ncol=3, fontsize=8, frameon=False)
        fig.tight_layout(rect=(0, .09, 1, .95))
        name = ('efficiency_' +
                quantum.replace(' ', '').replace('%', '') + '.png')
        path = os.path.join(out_dir, name)
        _save_figure(fig, path)
        plt.close(fig)
        made.append(path)
    return made


# ═════════════════════════════════════════════════════════════════════
# REPORTED UNCERTAINTY
# ═════════════════════════════════════════════════════════════════════
#
# "Which one has the smaller uncertainty?" is a trick question, and it is
# worth being clear about it before reading the table: in Bayesian
# inference a smaller sigma is NOT better. The correct sigma is the one of
# the posterior. A method that reports less uncertainty than there is is
# BIASED, not more precise.
#
# The gold reference is the classical MCMC: with enough steps and a
# converged R-hat, its sigmas are those of the posterior. Everything else
# is measured AGAINST it, and what we look for is a ratio close to 1 — not
# below it.
#
# The variational methods are expected to come out below: minimising
# KL(Q || P) is mode-seeking (zero-forcing), so Q comes out narrower than
# P. It is a textbook result of variational inference, and here it is
# quantified on the project's data.

REFERENCE = 'Classical MCMC'


def sigma_ratios(rows, rung='none') -> dict:
    """sigma of each method / sigma of the classical MCMC, per parameter.

    Returns:
        dict[method][parameter] -> list of ratios (one per cell)
    """
    cells = defaultdict(dict)
    for r in rows:
        if r['_fam'] == 'samplers' and r['_noi'] == rung:
            cells[campaign_io.cell_key(r)][r['Method']] = r

    rel = defaultdict(lambda: defaultdict(list))
    for v in cells.values():
        ref = v.get(REFERENCE)
        if not ref:
            continue
        pars = [c[:-4] for c in ref if c.endswith('_std')]
        for meth, r in v.items():
            if meth == REFERENCE:
                continue
            for p in pars:
                a, b = _num(r.get(p + '_std')), _num(ref.get(p + '_std'))
                if a is not None and b:
                    rel[meth][p].append(a / b)
    return rel


METHOD_ORDER = ['QMCMC 50%', 'QMCMC 100%', 'Classical VI',
                'QVMC 33%', 'QVMC 67%', 'QVMC 100%']
PARAM_ORDER = ['Om', 'H0', 'w', 'w0', 'wa', 'Delta']


def sigma_table(rel) -> None:
    print(f"\n{'='*72}\n  REPORTED UNCERTAINTY / that of {REFERENCE}"
          f"\n{'='*72}")
    print("  Close to 1 = agrees with the reference.")
    print("  BELOW 1 = reports less uncertainty than there is.")
    print("  That is NOT better: it is bias.\n")
    pars = [p for p in PARAM_ORDER if any(p in rel[m] for m in rel)]
    print(f"  {'Method':16s}" + ''.join(f"{p:>9s}" for p in pars))
    for meth in METHOD_ORDER:
        if meth not in rel:
            continue
        line = f"  {meth:16s}"
        for p in pars:
            v = rel[meth].get(p)
            line += f"{_med(v):9.3f}" if v else f"{'-':>9s}"
        print(line)


def sigma_figure(rel, out_dir) -> str | None:
    """Sigma ratio: one group per method, one point per parameter."""
    methods = [m for m in METHOD_ORDER if m in rel]
    if not methods:
        return None
    os.makedirs(out_dir, exist_ok=True)
    pars = [p for p in PARAM_ORDER if any(p in rel[m] for m in methods)]

    fig, ax = plt.subplots(figsize=(8.4, 4.0))
    for i, meth in enumerate(methods):
        vals = [_med(rel[meth][p]) for p in pars if p in rel[meth]]
        labels = [p for p in pars if p in rel[meth]]
        # the big marker is the median over parameters; the small ones, each one
        # Labels alternate above/below: several parameters land at almost
        # the same value and overlapping they cannot be read.
        for j, (v, p) in enumerate(zip(vals, labels)):
            ax.scatter(v, i, s=26, color=C_CLASSICAL_WINS if v < 0.95 else C_TIE,
                       zorder=3, alpha=.85)
            if v < 0.95:
                dy = -12 if j % 2 == 0 else 8
                ax.annotate(p, (v, i), textcoords='offset points',
                            xytext=(0, dy), ha='center', fontsize=7,
                            color=C_REF)
        if vals:
            ax.scatter(statistics.median(vals), i, s=120, marker='|',
                       color=C_REF, zorder=4, lw=1.6)
    ax.axvline(1.0, color=C_REF, lw=1.2)
    ax.axvspan(0.0, 0.95, color=C_CLASSICAL_WINS, alpha=.06)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods, fontsize=9)
    ax.set_xlabel(f'sigma of the method / sigma of {REFERENCE}', fontsize=9)
    ax.set_xlim(0.45, 1.12)
    ax.set_ylim(-0.6, len(methods) - 0.4)
    ax.grid(axis='x', alpha=.25, lw=.5)
    ax.tick_params(labelsize=8)
    ax.set_title('Reported uncertainty: below 1 is under-dispersion,'
                 ' not precision', fontsize=10, loc='left')
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([], [], marker='o', ls='', color=C_TIE,
               label='agrees with the reference'),
        Line2D([], [], marker='o', ls='', color=C_CLASSICAL_WINS,
               label='under-dispersion (<0.95)'),
        Line2D([], [], marker='|', ls='', color=C_REF, markersize=10,
               label='median over parameters')],
        fontsize=8, frameon=False, loc='upper center',
        bbox_to_anchor=(0.5, -0.16), ncol=3)
    fig.tight_layout(rect=(0, .06, 1, 1))
    path = os.path.join(out_dir, 'reported_uncertainty.png')
    _save_figure(fig, path)
    plt.close(fig)
    return path


# ═════════════════════════════════════════════════════════════════════
# THE GENETIC FAMILY
# ═════════════════════════════════════════════════════════════════════
#
# The genetic family is NOT compared with the same rules as the samplers,
# for two reasons that must be clear before reading anything:
#
#   1. Its figure of merit is `chi2_grid` — the chi2 of the best point on
#      the GRID, before refinement. [B-REFINE] The reported `chi2` is the
#      one from the continuous local refiner, which takes all four rungs to
#      the SAME minimum and therefore distinguishes nothing.
#
#   2. **Its `*_std` column is NOT an uncertainty.** A genetic algorithm
#      converges to a POINT: that deviation is the spread of the final
#      population, and it shrinks as the population converges. Measured
#      against the classical MCMC it gives between 0.07 and 0.18 — not
#      because the genetic algorithm is 6-14 times more precise, but because
#      it is not measuring the same thing. Putting it on the same axis as the
#      samplers' sigmas would be a serious misreading, so here it is reported
#      SEPARATELY and with a warning.
#
# Since chi2_grid depends on the grid resolution, the comparison only makes
# sense at FIXED n_bits: it is paired cell by cell against the CGA.
# And the DIFFERENCE is reported, not the ratio, because it is a chi2: what
# means something is how many units of chi2 are lost, not the percentage.

GENETIC_BASELINE = 'CGA'
QGA_ORDER = ['QGA (q=0%)', 'QGA (q=33%)', 'QGA (q=67%)', 'QGA (q=100%)']

# Single-hue ordinal ramp for the ladder (validated with the dataviz skill
# script: monotone L, steps >= 0.06, light end visible on the background).
# The ladder 0->33->67->100 is ORDINAL, not categorical.
LADDER_RAMP = ['#8fb4dd', '#5b8ecb', '#2f66ac', '#17406f']
# Categorical colours for the facets (validated with --pairs all).
C_CAT = ['#2a78d6', '#d95926', '#199e70']


def genetic_ratios(rows, rung='none') -> dict:
    """Delta chi2_grid of each rung against the CGA, per (model, n_bits).

    Returns:
        dict[method][(model, n_bits)] -> Delta chi2_grid (positive = worse)
    """
    cells = defaultdict(dict)
    for r in rows:
        if r['_fam'] == 'genetic' and r['_noi'] == rung:
            cells[campaign_io.cell_key(r)][r['Method']] = r

    out = defaultdict(dict)
    for key, v in cells.items():
        mod, g = key[5], key[6]
        base = v.get(GENETIC_BASELINE)
        if not base:
            continue
        c0 = _num(base.get('chi2_grid'))
        if c0 is None:
            continue
        for meth in QGA_ORDER:
            x = _num(v.get(meth, {}).get('chi2_grid'))
            if x is not None:
                out[meth][(mod, g)] = x - c0
    return out


def genetic_table(gen) -> None:
    if not gen:
        return
    print(f"\n{'='*72}\n  GENETIC — Delta chi2_grid against {GENETIC_BASELINE}"
          f"\n{'='*72}")
    print("  Negative = the quantum method found a BETTER grid point.")
    print("  Reference for statistical relevance: Delta chi2 = 1.\n")
    cells = sorted({k for m in gen for k in gen[m]})
    print(f"  {'model':7s} {'nb':>3s} " +
          ''.join(f"{m.replace('QGA (q=', '').replace(')', ''):>13s}"
                  for m in QGA_ORDER))
    for mod, g in cells:
        line = f"  {mod:7s} {g:3d} "
        for meth in QGA_ORDER:
            d = gen[meth].get((mod, g))
            line += f"{'-':>13s}" if d is None else f"{d:+13.4f}"
        print(line)

    print()
    for meth in QGA_ORDER:
        v = list(gen[meth].values())
        if not v:
            continue
        worse = sum(1 for x in v if x > 1e-6)
        better = sum(1 for x in v if x < -1e-6)
        rel = sum(1 for x in v if abs(x) >= 1.0)
        print(f"  {meth:14s} median {_med(v):+8.4f} | better than CGA "
              f"{better:3d}/{len(v)} | worse {worse:3d}/{len(v)} | "
              f"|Delta|>=1 in {rel}/{len(v)}")
    print("\n  Reading: a consistent sign across ALL cells is a real\n"
          "  systematic effect, even if the magnitude is small. They are two\n"
          "  different things and both must be reported.")


def genetic_sigma_warning(rows, rung='none') -> None:
    """The genetic sigma against the MCMC one, with the warning it needs."""
    refs, gen = {}, defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r['_noi'] != rung:
            continue
        if r['_fam'] == 'samplers' and r['Method'] == REFERENCE:
            refs[(r['_camp'], r['_mod'])] = r
    for r in rows:
        if r['_fam'] != 'genetic' or r['_noi'] != rung:
            continue
        R = refs.get((r['_camp'], r['_mod']))
        if not R:
            continue
        for c in R:
            if c.endswith('_std'):
                a, b = _num(r.get(c)), _num(R.get(c))
                if a is not None and b:
                    gen[r['Method']][c[:-4]].append(a / b)
    if not gen:
        return
    print(f"\n{'='*72}\n  GENETIC — spread of the final population"
          f"\n{'='*72}")
    print("  WARNING: this is NOT an uncertainty and is not comparable with")
    print("  the samplers' sigmas. A genetic algorithm converges to a POINT;")
    print("  the *_std column measures how much the population shrank, not")
    print("  how much uncertainty the parameter has.\n")
    pars = [p for p in PARAM_ORDER if any(p in gen[m] for m in gen)]
    print(f"  {'Method':16s}" + ''.join(f"{p:>9s}" for p in pars))
    for meth in [GENETIC_BASELINE] + QGA_ORDER:
        if meth not in gen:
            continue
        line = f"  {meth:16s}"
        for p in pars:
            v = gen[meth].get(p)
            line += f"{_med(v):9.3f}" if v else f"{'-':>9s}"
        print(line)


def genetic_figure(gen, out_dir) -> str | None:
    """Delta chi2_grid against n_bits, one facet per model.

    Faceted by model instead of putting all five on one axis because five
    categorical series do not pass the colour-blind separation check; with
    facets each panel carries three series, which does pass.
    """
    active = [m for m in QGA_ORDER[1:] if gen.get(m)]
    if not active:
        return None
    os.makedirs(out_dir, exist_ok=True)
    models = sorted({mod for m in active for (mod, _) in gen[m]})
    ncol = min(len(models), 5)
    fig, axes = plt.subplots(1, ncol, figsize=(2.5 * ncol + 1.2, 3.4),
                             squeeze=False, sharey=True)
    for j, mod in enumerate(models):
        ax = axes[0][j]
        for k, meth in enumerate(active):
            pts = sorted((g, d) for (m2, g), d in gen[meth].items()
                         if m2 == mod)
            if not pts:
                continue
            xs, ys = zip(*pts)
            # 67% and 100% coincide in almost every cell (the quantum
            # crossover rarely changes the best point found). The 67% is
            # drawn thicker and dashed so that it shows UNDER the 100%
            # instead of disappearing beneath it.
            width, dash = (3.0, (0, (4, 2))) if '67' in meth else (1.6, '-')
            ax.plot(xs, ys, marker='o', ms=4, lw=width, ls=dash,
                    color=C_CAT[k % len(C_CAT)], alpha=.95,
                    label=meth.replace('QGA (q=', '').replace(')', ''))
        ax.axhline(1.0, color=C_REF, lw=1.0, ls=':')
        ax.set_yscale('symlog', linthresh=1e-3)
        ax.set_title(mod.upper(), fontsize=10)
        ax.set_xlabel('n_bits', fontsize=8)
        ax.xaxis.set_major_locator(
            matplotlib.ticker.MaxNLocator(integer=True))
        ax.grid(alpha=.22, lw=.5)
        ax.tick_params(labelsize=8)
        if j == 0:
            ax.set_ylabel(r'grid $\Delta\chi^2$ against CGA', fontsize=9)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(labels),
               fontsize=8, frameon=False, title='quantumness',
               title_fontsize=8)
    fig.suptitle('Genetic: the cost of the quantum rung fades as the '
                 'grid is refined   (dotted: Δχ² = 1; 67% and 100% '
                 'coincide except for CPL)', fontsize=10)
    fig.tight_layout(rect=(0, .11, 1, .93))
    path = os.path.join(out_dir, 'genetic_chi2grid.png')
    _save_figure(fig, path)
    plt.close(fig)
    return path


# ═════════════════════════════════════════════════════════════════════
# COSMOLOGICAL MODEL COMPARISON  (another question, other tools)
# ═════════════════════════════════════════════════════════════════════
#
# Everything above compares ALGORITHMS. This compares MODELS, which is a
# different and orthogonal question: do the data prefer LCDM or CPL?
#
# It is not answered with chi2_red. The five models give chi2_red ~ 0.97,
# i.e. all of them fit acceptably; that does not discriminate. The chi2
# ALWAYS drops when parameters are added, so complexity must be penalised:
#
#     AIC = chi2 + 2k
#     BIC = chi2 + k ln(n)
#
# with k = free parameters and n = number of data points. Both columns are
# already in the CSV. What is compared are DIFFERENCES against the best.
#
# With n = 1099, ln(n) = 7.0: the BIC charges 7 units of chi2 per extra
# parameter and the AIC only 2. That is why they can disagree, and when
# they disagree the disagreement IS information: it means the extra
# parameter buys an intermediate improvement — enough for the AIC, not for
# the BIC.
#
# LIMITATION THAT MUST BE STATED: AIC and BIC are approximations to the
# Bayesian evidence. The properly Bayesian answer is the Bayes factor, which
# requires nested sampling (MultiNest, PolyChord) and this framework does
# not have it. Moreover they are evaluated at the MAP, so they are valid
# only if the MAP was really found.

# Jeffreys scale on Delta BIC, with its label. The colour is a single-hue
# ordinal ramp (validated: monotone L, light end visible on the
# background); the label always travels with the colour, never alone.
JEFFREYS = [(2.0,  'indistinguishable from the best', LADDER_RAMP[0]),
            (6.0,  'positive evidence against',       LADDER_RAMP[1]),
            (10.0, 'strong evidence against',         LADDER_RAMP[2]),
            (1e18, 'very strong evidence against',    LADDER_RAMP[3])]


def _jeffreys(d):
    for upper, label, color in JEFFREYS:
        if d < upper:
            return label, color
    return JEFFREYS[-1][1], JEFFREYS[-1][2]


def compare_models(rows, rung='none', dataset=None, prior=None) -> list:
    """The best fit of each model, with its AIC and BIC.

    The MINIMUM chi2 over all methods and grids of that model is taken: it
    is the best available estimate of the MAP, and that is why the
    cross-check between families (the genetic algorithm and the samplers
    landing on the same chi2) is what justifies using it.
    """
    # [E-QPU8] Model selection is only meaningful on one dataset and prior:
    # rows with different n_data must never share a table.
    groups = {((r.get('dataset') or '').strip(), (r.get('prior') or '').strip())
              for r in rows if r['_noi'] == rung}
    if dataset is None and len(groups) > 1:
        raise ValueError(f"compare_models: several (dataset, prior) groups "
                         f"{sorted(groups)}; pass dataset=/prior= explicitly")
    best = {}
    for r in rows:
        if r['_noi'] != rung:
            continue
        if dataset is not None and ((r.get('dataset') or '').strip() != dataset
                                    or (r.get('prior') or '').strip() != prior):
            continue
        c = _num(r.get('chi2'))
        if c is None:
            continue
        if r['_mod'] not in best or c < _num(best[r['_mod']]['chi2']):
            best[r['_mod']] = r

    out = []
    for mod, r in best.items():
        c, a, b = (_num(r['chi2']), _num(r['AIC']), _num(r['BIC']))
        n = int(_num(r['n_data']) or 0)
        if None in (c, a, b) or not n:
            continue
        out.append({'model': mod, 'k': round((a - c) / 2), 'chi2': c,
                    'chi2_red': _num(r['chi2_red']), 'aic': a, 'bic': b,
                    'n': n})
    return sorted(out, key=lambda x: x['bic'])


def models_table(mods) -> None:
    if not mods:
        return
    n = mods[0]['n']
    print(f"\n{'='*72}\n  COSMOLOGICAL MODELS — which one the data prefer"
          f"\n{'='*72}")
    print(f"  n = {n} data points,  ln(n) = {math.log(n):.3f}")
    print(f"  AIC = chi2 + 2k     BIC = chi2 + k*ln(n)\n")
    print(f"  {'model':7s} {'k':>2s} {'chi2':>11s} {'chi2_red':>9s} "
          f"{'dAIC':>8s} {'dBIC':>8s}   verdict (dBIC)")
    amin = min(m['aic'] for m in mods)
    bmin = min(m['bic'] for m in mods)
    for m in mods:
        lab, _ = _jeffreys(m['bic'] - bmin)
        print(f"  {m['model']:7s} {m['k']:2d} {m['chi2']:11.4f} "
              f"{m['chi2_red']:9.4f} {m['aic']-amin:8.3f} "
              f"{m['bic']-bmin:8.3f}   {lab}")
    ga = min(mods, key=lambda x: x['aic'])['model']
    gb = mods[0]['model']
    print(f"\n  Best by AIC: {ga}      Best by BIC: {gb}")
    if ga != gb:
        print("  Their disagreement is not a problem: the BIC charges "
              f"{math.log(n):.1f} units of chi2\n  per extra parameter and "
              "the AIC only 2. The extra parameter buys an\n  intermediate "
              "improvement — enough for the AIC and not for the BIC.")
    print("\n  chi2_red close to 1 for all: the five models fit\n"
          "  acceptably. That is why chi2_red CANNOT choose between them.")


def models_figure(mods, out_dir) -> str | None:
    if len(mods) < 2:
        return None
    os.makedirs(out_dir, exist_ok=True)
    amin = min(m['aic'] for m in mods)
    bmin = min(m['bic'] for m in mods)

    # ONE single order for both panels. With `sharey` the second panel
    # overwrites the labels of the first, so sorting them separately
    # paired each name with the value of the other criterion.
    # Also, this way each row is the SAME model in both panels, which is
    # what makes the AIC/BIC disagreement visible.
    order = sorted(mods, key=lambda x: -x['bic'])
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.6), sharey=True)
    for ax, key, mini, title, pen in (
            (axes[0], 'aic', amin, r'(a) $\Delta$AIC', '2 per parameter'),
            (axes[1], 'bic', bmin, r'(b) $\Delta$BIC',
             f"{math.log(mods[0]['n']):.1f} per parameter")):
        for i, m in enumerate(order):
            d = m[key] - mini
            _, color = _jeffreys(d)
            ax.barh(i, max(d, 0.0), height=.58, color=color)
            ax.text(d + 0.35, i, f'{d:.2f}', va='center', fontsize=8,
                    color=C_REF)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([f"{m['model'].upper()}  (k={m['k']})"
                            for m in order], fontsize=9)
        for u in (2, 6, 10):
            ax.axvline(u, color=C_REF, lw=.8, ls=':')
        ax.set_xlabel(f'{title.split()[-1]}  —  penalises {pen}', fontsize=9)
        ax.set_title(title, fontsize=10, loc='left')
        ax.grid(axis='x', alpha=.22, lw=.5)
        ax.tick_params(labelsize=8)
        ax.set_xlim(0, max(4.0, max(m[key] - mini for m in mods) * 1.22))

    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=c, label=e) for _, e, c in JEFFREYS],
               loc='lower center', ncol=2, fontsize=8, frameon=False)
    fig.suptitle('Which model the data prefer   '
                 '(0 = the best; the dotted lines are 2, 6 and 10)', fontsize=10)
    fig.tight_layout(rect=(0, .16, 1, .93))
    path = os.path.join(out_dir, 'models_aic_bic.png')
    _save_figure(fig, path)
    plt.close(fig)
    return path


# =============================================================================
# [B-SIGMANQPP] Does the variational under-dispersion go away as the grid is refined?
# =============================================================================
#
# It is the obvious objection to the `reported_uncertainty` result: "you
# report a narrower sigma because your grid is coarse; with more qubits it
# gets fixed". This section answers it with data, following the ratio
# sigma(QVMC)/sigma(MCMC) along the WHOLE nqpp sweep.
#
# Beware of what CANNOT be compared here: `final_KL` changes meaning with
# nqpp, because the target distribution is defined on the grid and the
# grid changes. Two KLs computed on different grids are not the same
# quantity. The sigma ratio is comparable: sigma is the width of the same
# parameter in the same units, and the classical reference does not depend
# on nqpp.

def sigma_vs_grid(rows, method='QVMC 100%', param='Om',
                  rung='none') -> dict:
    """Ratio sigma(method)/sigma(MCMC) as a function of nqpp, per model.

    Args:
        rows: rows already read by `read_rows`.
        method: the quantum rung to follow.
        param: cosmological parameter (every model has Om).
        rung: noise level; the ideal one by default.

    Returns:
        dict[model] -> list of `(nqpp, ratio)` sorted by nqpp.
    """
    cells = defaultdict(dict)
    for r in rows:
        if r['_fam'] == 'samplers' and r['_noi'] == rung:
            cells[campaign_io.cell_key(r)][r['Method']] = r

    # [E-QPU8] One series per (model, campaign, dataset, prior, seed); the
    # campaign is added to the label only when several are present.
    series = {k[:4] for k in cells}
    out = defaultdict(list)
    for key, v in cells.items():
        mod, g = key[5], key[6]
        if len(series) > 1:
            mod = f"{mod} [{key[0]}|{key[1]}|seed {key[3] or '?'}]"
        ref, q = v.get(REFERENCE), v.get(method)
        if not ref or not q:
            continue
        a, b = _num(q.get(param + '_std')), _num(ref.get(param + '_std'))
        if a is None or not b or g is None:
            continue
        out[mod].append((g, a / b))
    return {m: sorted(v) for m, v in out.items() if len(v) >= 2}


def sigma_vs_grid_table(series, method='QVMC 100%') -> None:
    """Print the sigma ratio along the nqpp sweep."""
    if not series:
        return
    print(f"\n{'='*72}\n  DOES THE UNDER-DISPERSION GO AWAY WITH MORE QUBITS?"
          f"\n{'='*72}")
    print(f"  ratio sigma({method}) / sigma({REFERENCE}) in Om,")
    print("  along the nqpp sweep.\n")
    for mod in sorted(series):
        pts = series[mod]
        body = '  '.join(f"{g}:{r:.3f}" for g, r in pts)
        print(f"  {mod:6s}  {body}")
        if len(pts) >= 4:
            tail = [r for _, r in pts[-3:]]
            print(f"  {'':6s}  -> last three: "
                  f"{min(tail):.3f}–{max(tail):.3f}")
    print("\n  If the ratio rose towards 1 as nqpp grows, the under-dispersion")
    print("  would be a coarse-grid artefact. If it flattens out below 1,")
    print("  it is a structural bias of the variational objective and more")
    print("  qubits do not fix it.")


def sigma_vs_grid_figure(series, out_dir, method='QVMC 100%') -> str | None:
    """One curve per model: sigma ratio against nqpp.

    There are FIVE models and the validated palette has three colours, so
    colour alone is not enough (two pairs would share a hue). Each series
    is distinguished by the colour+marker+linestyle triple, and is also
    labelled at the end of the curve, which is what is actually read when
    the curves bunch together on the plateau.
    """
    if not series:
        return None
    os.makedirs(out_dir, exist_ok=True)
    markers = ['o', 's', '^', 'D', 'v']
    linestyles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]

    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    for i, mod in enumerate(sorted(series)):
        pts = series[mod]
        x = [g for g, _ in pts]
        y = [r for _, r in pts]
        col = C_CAT[i % len(C_CAT)]
        ax.plot(x, y, color=col, lw=1.7, ms=5,
                marker=markers[i % len(markers)],
                linestyle=linestyles[i % len(linestyles)], label=mod.upper())
        ax.annotate(f' {mod.upper()}', (x[-1], y[-1]), fontsize=8,
                    color=col, va='center', ha='left')

    ax.axhline(1.0, color=C_REF, lw=1.2, ls='--')
    ax.annotate('agrees with the MCMC', (min(x), 1.0), fontsize=8,
                color=C_REF, va='bottom', ha='left')
    ax.set_xlabel('nqpp  (qubits per grid parameter)', fontsize=9)
    ax.set_ylabel(f'sigma({method}) / sigma({REFERENCE})', fontsize=9)
    ax.set_title('The variational under-dispersion is NOT fixed by more '
                 'qubits', fontsize=10, loc='left')
    ax.set_xticks(sorted({g for v in series.values() for g, _ in v}))
    ax.set_ylim(top=max(1.06, ax.get_ylim()[1]))
    ax.margins(x=.10)
    ax.grid(alpha=.22, lw=.5)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    path = os.path.join(out_dir, 'sigma_vs_grid.png')
    _save_figure(fig, path)
    plt.close(fig)
    return path


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description='Compares methods with the estimator that fits each '
                    'family. Runs nothing.')
    p.add_argument('folders', nargs='+')
    p.add_argument('--out', default='efficiency_figures')
    p.add_argument('--rung', default='none',
                   help="noise rung to compare (default 'none')")
    a = p.parse_args(argv)

    rows = read_rows([c for c in a.folders if os.path.isdir(c)])
    if not rows:
        print("  no row was read.")
        return 1
    print(f"\n  {len(rows)} rows read from {len(a.folders)} campaign(s)")
    by = sampler_ratios(rows, rung=a.rung)
    method_table(by)
    made = method_figure(by, a.out)

    rel = sigma_ratios(rows, rung=a.rung)
    if rel:
        sigma_table(rel)
        r = sigma_figure(rel, a.out)
        if r:
            made.append(r)

    # [B-SIGMANQPP] answers the "coarse grid" objection.
    series = sigma_vs_grid(rows, rung=a.rung)
    if series:
        sigma_vs_grid_table(series)
        r = sigma_vs_grid_figure(series, a.out)
        if r:
            made.append(r)

    groups = sorted({((r.get('dataset') or '').strip(),
                      (r.get('prior') or '').strip())
                     for r in rows if r['_noi'] == a.rung})
    for ds, pr in groups:
        mods = compare_models(rows, rung=a.rung, dataset=ds, prior=pr)
        if mods:
            print(f"\n  dataset = {ds or '?'}   prior = {pr or '?'}")
            models_table(mods)
            if len(groups) == 1:
                r = models_figure(mods, a.out)
                if r:
                    made.append(r)
    if len(groups) > 1:
        print("  (model-selection figure skipped: several datasets present)")

    gen = genetic_ratios(rows, rung=a.rung)
    if gen:
        genetic_table(gen)
        genetic_sigma_warning(rows, rung=a.rung)
        r = genetic_figure(gen, a.out)
        if r:
            made.append(r)

    print(f"\n  {len(made)} figures in {os.path.abspath(a.out)}/\n")
    return 0


if __name__ == '__main__':
    sys.exit(main())
