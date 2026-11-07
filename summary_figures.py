#!/usr/bin/env python3
"""Summary (composite) figures for the thesis body.

The per-cell figures of `seed_figures.py` and `seed_figures_genetic.py`
are many (about 120); these composites carry the same evidence in a few
multi-panel figures, meant for the body of the thesis. The per-cell figures
stay as the appendix. Only reading and drawing: nothing is recomputed by a
sampler, and the helpers of `seed_figures.py` are reused.

Figures (PNG + PDF) written to ``--out``:

* ``resumen_qmcmc``             QMCMC fidelity: Ωm marginals of MCMC and QMCMC for
                                every model, and the width ratio, mean shift and
                                ESS ratio with seed bars.
* ``resumen_qvmc``              QVMC: width of the Ωm posterior vs nqpp (multi-seed,
                                plus the single-seed nqpp >= 6 cells of an older
                                campaign when ``--single-seed`` is given), and the
                                paired final-KL difference per cell.
* ``resumen_posteriores_qvmc``  Ωm–H0 posteriors of MCMC, classical VI and QVMC
                                100 % at the largest grid of every model.
* ``resumen_kl_qvmc``           KL training traces at the largest grid of every model.
* ``resumen_rhat_qmcmc``        R-hat traces of every model.
* ``resumen_fitness_genetico``  genetic fitness traces in three representative cells
                                (needs ``--genetic``).
* ``resumen_comparacion``       every family against its classical counterpart
                                (shift, width, Δχ²_grid) and the wall time of
                                every family (needs ``--genetic``).
* ``resumen_ruido``             shift caused by each noise level in every noisy cell
                                of a single-seed campaign (needs ``--single-seed``).

Usage:
    python summary_figures.py SEEDS_RESULTS --out FIGS [--seeds 43-50]
        [--genetic GENETIC_RESULTS] [--single-seed CAMPAIGN_DIR ...]

``--single-seed`` takes the folders of a campaign run with one seed (for
example ``hpc_20260907_130425``); its rows are read with
`campaign_io.read_campaign` and are drawn with their own marker, never
averaged with the multi-seed rows.
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

import compare_algorithms  # noqa: E402
import compare_seeds  # noqa: E402
import cosmo_core as core  # noqa: E402
import cosmo_modular_quantum as cmq  # noqa: E402
import noise_plots  # noqa: E402
import seed_figures as sf  # noqa: E402

QVMC_FULL = 'QVMC 67%'        # stored name of "QVMC 100% (+ quantum training)"
QMCMC_FULL = 'QMCMC 100%'
SIGMA2D = (0.393, 0.865)      # 1 and 2 sigma in two dimensions
NOISE_COLORS = {'none-counts': '#7f7f7f', 'readout': '#ff7f0e', 'full': '#d62728',
                'fakebrisbane': '#2ca02c'}


def largest(data, model):
    """Largest grid of ``model`` in the multi-seed campaign."""
    return sf.models_and_grids(data)[model][-1]


def pooled(taskdirs, model, n, seeds, key, weights_key=None, jitter=None, rng=None):
    """Samples of one npz key pooled over seeds, each seed with equal weight."""
    S, W = [], []
    for s in seeds:
        d = taskdirs.get((model, n), {}).get(s)
        z = sf._npz(d, model) if d else None
        if z is None or key not in z:
            continue
        x = np.asarray(z[key], float)
        if jitter is not None:
            x = x + rng.uniform(-0.5, 0.5, x.shape) * jitter
        w = (np.asarray(z[weights_key], float) if weights_key else np.ones(len(x)))
        S.append(x)
        W.append(w / w.sum())
    if not S:
        return None, None
    return np.vstack(S), np.concatenate(W) / len(W)


def model_axis(ax, models, labels=None):
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(labels or [sf.short(m) for m in models])
    ax.set_xlim(-0.6, len(models) - 0.4)
    ax.grid(alpha=0.3)


# ── A. QMCMC fidelity ─────────────────────────────────────────────────────────
def fig_qmcmc(data, taskdirs, seeds, out, report):
    mg = sf.models_and_grids(data)
    models = list(mg)
    fig = plt.figure(figsize=(3.2 * len(models), 7.4))
    top = fig.add_gridspec(1, len(models), top=0.86, bottom=0.56, left=0.05, right=0.98)
    bot = fig.add_gridspec(1, 3, top=0.44, bottom=0.12, left=0.05, right=0.98, wspace=0.18)
    for k, m in enumerate(models):
        ax = fig.add_subplot(top[0, k])
        n = largest(data, m)
        for key, col, ls, lab in (('qmcmc_0_flat', cmq.C_CLASSICAL, '-', 'Classical MCMC'),
                                  ('qmcmc_100_flat', cmq.C_QUANTUM, '--', 'QMCMC 100%')):
            S, _ = pooled(taskdirs, m, n, seeds, key)
            if S is not None:
                ax.hist(S[:, 0], bins=45, density=True, histtype='step', color=col, ls=ls,
                        lw=1.8, label=lab)
        ax.set_title(sf.short(m), fontsize=11)
        ax.set_xlabel(r'$\Omega_m$')
        ax.set_yticks([])
        ax.grid(alpha=0.3)
    panels = [(r'$\sigma(\Omega_m)$ ratio QMCMC / MCMC', 1.0,
               lambda c: sf.sigma_ratio(c, QMCMC_FULL, seeds)),
              (r'shift of $\bar\Omega_m$ [$\sigma_{\rm MCMC}$]', 0.0,
               lambda c: sf.om_shift(c, QMCMC_FULL, seeds)),
              ('ESS ratio QMCMC / MCMC', 1.0,
               lambda c: [(a / b) if a is not None and b else None for a, b in zip(
                   sf.per_seed(c, QMCMC_FULL, lambda r: sf.num(r, 'ESS'), seeds),
                   sf.per_seed(c, 'Classical MCMC', lambda r: sf.num(r, 'ESS'), seeds))])]
    for k3, ((ttl, ref, get), tag) in enumerate(zip(panels, ('sigma ratio', 'Om shift', 'ESS ratio'))):
        ax = fig.add_subplot(bot[0, k3])
        for k, m in enumerate(models):
            vals = get(data[(m, largest(data, m))])
            sf.draw_seeds(ax, k, vals, width=0.08)
            sf.mean_bar(ax, k, vals, cmq.C_QUANTUM)
            if sf.rhat_flags(data, m, seeds):
                ax.axvspan(k - 0.45, k + 0.45, color='firebrick', alpha=0.07, lw=0)
            mm, sd, se, nn = sf.stats(vals)
            report.append(f'  qmcmc {tag:11s} {m:5s} {sf.fmt(mm, sd, se, prec=3)}')
        ax.axhline(ref, color='k', ls='--', lw=0.9)
        model_axis(ax, models)
        ax.set_title(ttl, fontsize=10)
    fig.legend(handles=[Line2D([0], [0], color=cmq.C_CLASSICAL, lw=1.8, label='Classical MCMC'),
                        Line2D([0], [0], color=cmq.C_QUANTUM, lw=1.8, ls='--', label='QMCMC 100%'),
                        Line2D([0], [0], color=sf.SEED_GREY, marker='o', ls='none', ms=4,
                               label='individual seeds')],
               loc='lower center', ncol=3, fontsize=9, frameon=False)
    fig.suptitle('Fidelity of QMCMC to classical MCMC', fontsize=13, fontweight='bold')
    return sf.save(fig, os.path.join(out, 'resumen_qmcmc'))


# ── B. QVMC width and KL ──────────────────────────────────────────────────────
def single_seed_sigma(rows):
    """``{model: [(nqpp, ratio), ...]}`` of QVMC 100 % vs MCMC, ideal rows."""
    return compare_algorithms.sigma_vs_grid(rows, method=QVMC_FULL)


def fig_qvmc(data, seeds, out, report, single=None):
    mg = sf.models_and_grids(data)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(15, 5.6), gridspec_kw=dict(width_ratios=[1.15, 1]))
    off = {m: (i - (len(mg) - 1) / 2) * 0.06 for i, m in enumerate(mg)}
    top_grid = max(n for g in mg.values() for n in g)
    for m, grids in mg.items():
        col = sf.MODEL_COLORS.get(m, 'k')
        for meth, ls, mk, mfc in ((QVMC_FULL, '-', 'o', None), ('Classical VI', ':', 'D', 'white')):
            xs = [n + off[m] for n in grids]
            vals = [sf.sigma_ratio(data[(m, n)], meth, seeds) for n in grids]
            sf.series(ax, xs, vals, col, mk, sf.short(m) if meth == QVMC_FULL else None,
                      ls=ls, mfc=mfc, jitter=0.012)
        if single:
            pts = [(g, r) for g, r in single.get(m, []) if g > max(grids)]
            if pts:
                ax.plot([g + off[m] for g, _ in pts], [r for _, r in pts], '*-', color=col,
                        ms=9, lw=1.0, alpha=0.8, mfc='white')
                top_grid = max(top_grid, max(g for g, _ in pts))
                report.append(f'  single-seed sigma {m}: ' +
                              ', '.join(f'nqpp{g} {r:.3f}' for g, r in pts))
    ax.axhline(1, color='k', ls='--', lw=0.9)
    ax.set_xticks(range(min(n for g in mg.values() for n in g), top_grid + 1))
    ax.set_xlabel('nqpp (qubits per parameter)')
    ax.set_ylabel(r'$\sigma(\Omega_m)\,/\,\sigma_{\rm MCMC}(\Omega_m)$')
    ax.grid(alpha=0.3)
    ax.set_title(r'(a) Width of the $\Omega_m$ posterior', fontsize=11)
    h = [Line2D([0], [0], color=sf.MODEL_COLORS[m], lw=2, label=sf.short(m)) for m in mg]
    h += [Line2D([0], [0], color='k', marker='o', label='QVMC 100%, 8 seeds'),
          Line2D([0], [0], color='k', ls=':', marker='D', mfc='white', label='classical VI, 8 seeds')]
    if single:
        h.append(Line2D([0], [0], color='k', marker='*', mfc='white', ms=9, lw=1,
                        label='QVMC 100%, single seed'))
    ax.legend(handles=h, fontsize=8, ncol=2, loc='upper right', framealpha=0.9)
    ax.set_ylim(0, 2.45)
    # (b) paired KL, every cell, grouped by model
    x, ticks, labels, sep = 0, [], [], []
    for m, grids in mg.items():
        col = sf.MODEL_COLORS.get(m, 'k')
        for n in grids:
            cell = data[(m, n)]
            a = sf.per_seed(cell, QVMC_FULL, lambda r: sf.num(r, 'final_KL'), seeds)
            b = sf.per_seed(cell, 'Classical VI', lambda r: sf.num(r, 'final_KL'), seeds)
            d = [p - q if None not in (p, q) else None for p, q in zip(a, b)]
            _d, _se, t = compare_seeds.in_sigmas(a, b)
            sig = t is not None and t > 3
            sf.draw_seeds(bx, x, d, width=0.12)
            sf.mean_bar(bx, x, d, col, marker='*' if sig else 'o', ms=11 if sig else 6,
                        mfc=col if sig else 'white')
            ticks.append(x)
            labels.append(str(n))
            x += 1
        sep.append(x - 0.5)
        x += 0.6
    for s_ in sep[:-1]:
        bx.axvline(s_ + 0.3, color='0.8', lw=0.8)
    pos = 0
    for m, grids in mg.items():
        mid = np.mean(ticks[pos:pos + len(grids)])
        bx.text(mid, 1.02, sf.short(m), transform=bx.get_xaxis_transform(), ha='center',
                fontsize=10)
        pos += len(grids)
    bx.axhline(0, color='k', ls='--', lw=0.9)
    bx.set_xticks(ticks)
    bx.set_xticklabels(labels)
    bx.set_xlabel('nqpp (within each model)')
    bx.set_ylabel('final KL: QVMC 100% − classical VI')
    bx.grid(alpha=0.3, axis='y')
    bx.set_title('(b) Paired final-KL difference', fontsize=11, pad=22)
    bx.legend(handles=[Line2D([0], [0], color='0.3', marker='*', ms=11, ls='none', label='paired t > 3'),
                       Line2D([0], [0], color='0.3', marker='o', mfc='white', ls='none',
                              label='paired t ≤ 3')], fontsize=8, loc='upper right')
    fig.suptitle('QVMC: posterior width and training quality', fontsize=13, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return sf.save(fig, os.path.join(out, 'resumen_qvmc'))


# ── C. QVMC posteriors ────────────────────────────────────────────────────────
def fig_posteriors(data, taskdirs, seeds, out):
    import corner
    mg = sf.models_and_grids(data)
    rng = np.random.default_rng(12345)
    fig, axes = plt.subplots(1, len(mg), figsize=(3.6 * len(mg), 4.2), squeeze=False)
    axes = axes[0]
    for ax, m in zip(axes, mg):
        model = core.MODELS[m]
        n = largest(data, m)
        cell = np.array([(hi - lo) / (2 ** n - 1) for lo, hi in model.sample_box])
        sets = [pooled(taskdirs, m, n, seeds, 'qmcmc_0_flat') + ('0.35', True, 'Classical MCMC'),
                pooled(taskdirs, m, n, seeds, 'qvmc_0_S', 'qvmc_0_W', cell, rng)
                + (cmq.C_CLASSICAL, False, 'Classical VI'),
                pooled(taskdirs, m, n, seeds, 'qvmc_67_S', 'qvmc_67_W', cell, rng)
                + (cmq.C_QUANTUM, False, 'QVMC 100% (+ quantum training)')]
        allx = np.vstack([s[0][:, :2] for s in sets if s[0] is not None])
        rngs = [tuple(np.percentile(allx[:, j], [0.5, 99.5])) for j in (0, 1)]
        for S, W, col, fill, lab in sets:
            if S is None:
                continue
            corner.hist2d(S[:, 0], S[:, 1], weights=W, ax=ax, levels=SIGMA2D, bins=35,
                          range=rngs, smooth=1.0, color=col, plot_datapoints=False,
                          plot_density=False, fill_contours=fill, no_fill_contours=not fill,
                          contour_kwargs=dict(linewidths=1.6, colors=[col]))
        ax.set_xlabel(r'$\Omega_m$')
        ax.set_title(f'{sf.short(m)}, nqpp = {n}', fontsize=11)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel(r'$H_0$ [km/s/Mpc]')
    fig.legend(handles=[Line2D([0], [0], color='0.35', lw=6, alpha=0.5, label='Classical MCMC'),
                        Line2D([0], [0], color=cmq.C_CLASSICAL, lw=1.8, label='Classical VI'),
                        Line2D([0], [0], color=cmq.C_QUANTUM, lw=1.8,
                               label='QVMC 100% (+ quantum training)')],
               loc='lower center', ncol=3, fontsize=9, frameon=False)
    fig.suptitle(r'$\Omega_m$–$H_0$ posteriors at the largest grid', fontsize=13, fontweight='bold')
    fig.tight_layout(rect=(0, 0.08, 1, 0.93))
    return sf.save(fig, os.path.join(out, 'resumen_posteriores_qvmc'))


# ── D/E. traces ───────────────────────────────────────────────────────────────
def fig_kl(data, taskdirs, seeds, out):
    mg = sf.models_and_grids(data)
    fig, axes = plt.subplots(1, len(mg), figsize=(3.6 * len(mg), 4.0), sharey=False, squeeze=False)
    axes = axes[0]
    rungs = [('Classical VI', 0.0, cmq.C_CLASSICAL, '-'), ('QVMC 33%', 33.0, '#ffbb55', '--'),
             (QVMC_FULL, 67.0, cmq.C_QUANTUM, '-')]
    for ax, m in zip(axes, mg):
        n = largest(data, m)
        tr = sf._kl_traces(taskdirs, m, n, seeds)
        x_hi = 1.08 * min(xs[-1] for xs, _ in tr[67.0].values()) if 67.0 in tr else None
        for name, p, col, ls in rungs:
            if p not in tr:
                continue
            traces = [tr[p][s] for s in sorted(tr[p], key=int)]
            grid = np.unique(np.concatenate([xs for xs, _ in traces]))
            if x_hi:
                grid = grid[grid <= x_hi]
            med, lo, hi = sf._band(traces, grid)
            ax.fill_between(grid, lo, hi, color=col, alpha=0.15, lw=0)
            ax.semilogy(grid, med, color=col, ls=ls, lw=2.0)
        if x_hi:
            ax.set_xlim(0, x_hi)
        ax.set_title(f'{sf.short(m)}, nqpp = {n}', fontsize=11)
        ax.set_xlabel('training iteration')
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(r'KL$(Q_\varphi\,\|\,P_{\rm target})$')
    fig.legend(handles=[Line2D([0], [0], color=c, ls=ls, lw=2, label=lab) for lab, c, ls in (
        ('Classical VI', cmq.C_CLASSICAL, '-'),
        ('QVMC 50% (quantum sampling; identical to classical VI)', '#ffbb55', '--'),
        ('QVMC 100% (+ quantum training)', cmq.C_QUANTUM, '-'))],
        loc='lower center', ncol=3, fontsize=9, frameon=False)
    fig.suptitle('QVMC training at the largest grid', fontsize=13, fontweight='bold')
    fig.tight_layout(rect=(0, 0.08, 1, 0.93))
    return sf.save(fig, os.path.join(out, 'resumen_kl_qvmc'))


def fig_rhat(data, taskdirs, seeds, out):
    mg = sf.models_and_grids(data)
    fig, axes = plt.subplots(1, len(mg), figsize=(3.6 * len(mg), 4.0), sharey=True, squeeze=False)
    axes = axes[0]
    for ax, m in zip(axes, mg):
        n = largest(data, m)
        tr = defaultdict(dict)
        for s in seeds:
            d = taskdirs[(m, n)].get(s)
            logs = sorted(__import__('glob').glob(os.path.join(d, 'sweep_all_*.log'))) if d else []
            if logs:
                rh, _ = cmq.histories_from_log(logs[-1])
                for p, h in rh.items():
                    tr[p][s] = dict(h)
        for p, col, ls in ((0.0, cmq.C_CLASSICAL, '-'), (100.0, cmq.C_QUANTUM, '--')):
            if p not in tr:
                continue
            steps = sorted({st for h in tr[p].values() for st in h})
            Y = np.array([[h.get(st, np.nan) - 1 for st in steps] for h in tr[p].values()])
            ax.fill_between(steps, np.nanpercentile(Y, 16, 0), np.nanpercentile(Y, 84, 0),
                            color=col, alpha=0.15, lw=0)
            ax.semilogy(steps, np.nanmedian(Y, 0), color=col, ls=ls, lw=2)
        ax.axhline(core.RHAT_THRESHOLD - 1, color='k', ls=':', lw=1)
        ax.set_title(sf.short(m), fontsize=11)
        ax.set_xlabel('sampling steps')
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(r'$\hat R_{\max}-1$')
    fig.legend(handles=[Line2D([0], [0], color=cmq.C_CLASSICAL, lw=2, label='Classical MCMC'),
                        Line2D([0], [0], color=cmq.C_QUANTUM, lw=2, ls='--',
                               label='QMCMC 100% (50% identical)'),
                        Line2D([0], [0], color='k', ls=':', label=r'$\hat R = 1.01$')],
               loc='lower center', ncol=3, fontsize=9, frameon=False)
    fig.suptitle('QMCMC convergence', fontsize=13, fontweight='bold')
    fig.tight_layout(rect=(0, 0.08, 1, 0.93))
    return sf.save(fig, os.path.join(out, 'resumen_rhat_qmcmc'))


# ── F. genetic ────────────────────────────────────────────────────────────────
def fig_fitness(gdata, gtraces, seeds, out, cells=(('lcdm', 4), ('lcdm', 8), ('cpl', 4))):
    import seed_figures_genetic as sfg
    cells = [c for c in cells if c in gdata]
    fig, axes = plt.subplots(1, len(cells), figsize=(4.6 * len(cells), 4.2), squeeze=False)
    axes = axes[0]
    for ax, (m, nb) in zip(axes, cells):
        tr = gtraces.get((m, nb), {})
        fl = sfg.floor_found(gdata, m, nb, seeds)
        top = 1e-3
        for meth in sfg.METHODS:
            runs = [tr[s][meth] for s in seeds if s in tr and meth in tr[s]]
            b = sfg._trace_band(runs, 'best')
            if b is None:
                continue
            g, med, lo, hi, _n = b
            top = max(top, float(np.max(hi - fl)))
            ax.fill_between(g, lo - fl, hi - fl, color=sfg.COLORS[meth], alpha=0.15, lw=0)
            ax.plot(g, med - fl, sfg.STYLES[meth], color=sfg.COLORS[meth], lw=2,
                    marker=sfg.MARKERS[meth], ms=3.5, label=meth)
        ax.set_yscale('symlog', linthresh=1e-3, linscale=0.6)
        ax.set_ylim(0, 4 * top)
        ax.set_title(f'{sf.short(m)}, n_bits = {nb}', fontsize=11)
        ax.set_xlabel('generation')
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(r'best $\chi^2_{\rm grid}$ − lowest found')
    fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', ncol=5, fontsize=9,
               frameon=False)
    fig.suptitle('Genetic fitness in representative cells', fontsize=13, fontweight='bold')
    fig.tight_layout(rect=(0, 0.08, 1, 0.93))
    return sf.save(fig, os.path.join(out, 'resumen_fitness_genetico'))


# ── G. noise (single-seed campaign) ──────────────────────────────────────────
def noise_shifts(rows):
    """``[(rung, method, model, grid, shift)]`` of every noisy sampler cell."""
    ideal = noise_plots.ideal_cell_index(rows)
    out = []
    for r in rows:
        if r['_family'] != 'samplers' or r['_noise'] == 'none':
            continue
        if r.get('Method') == 'QVMC 100%':      # normalization rung, equal to 67 % by construction
            continue
        v = noise_plots.metric_value(r, 'shift', ideal)
        if v is not None:
            out.append((r['_noise'], r['Method'], r['_model'], r['_grid'], v))
    return out


def fig_noise(shifts, out, report):
    rungs = noise_plots._order_rungs(sorted({s[0] for s in shifts}))
    fams = [('QMCMC (MCMC, 50%, 100%)', lambda m: 'MCMC' in m, 'o'),
            ('VI (classical, 50%, 100%)', lambda m: 'VI' in m or m.startswith('QVMC'), 's')]
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    rng = np.random.default_rng(1)
    for i, rung in enumerate(rungs):
        for j, (lab, test, mk) in enumerate(fams):
            v = [s[4] for s in shifts if s[0] == rung and test(s[1])]
            if not v:
                continue
            x = i + (j - 0.5) * 0.3
            ax.scatter(x + rng.uniform(-0.08, 0.08, len(v)), v, s=14, marker=mk,
                       color=NOISE_COLORS.get(rung, 'k'), alpha=0.55, lw=0)
            ax.plot([x - 0.12, x + 0.12], [np.median(v)] * 2, color='k', lw=2)
            report.append(f'  noise {rung:12s} {lab.split()[0]:6s} n={len(v):3d} median {np.median(v):.3f} '
                          f'p90 {np.percentile(v, 90):.3f} max {max(v):.3f} >1σ {sum(x_ > 1 for x_ in v)}')
    ax.axhline(1, color='firebrick', ls='--', lw=1.1)
    ax.set_yscale('symlog', linthresh=0.01, linscale=0.5)
    ax.set_ylim(0, None)
    ax.set_xticks(range(len(rungs)))
    ax.set_xticklabels(rungs)
    ax.set_ylabel(r'max$_p\,|\Delta p|\,/\,\sigma_{\rm ideal}$')
    ax.grid(alpha=0.3)
    ax.legend(handles=[Line2D([0], [0], color='0.4', marker=mk, ls='none', label=lab)
                       for lab, _, mk in fams]
              + [Line2D([0], [0], color='k', lw=2, label='median'),
                 Line2D([0], [0], color='firebrick', ls='--', label='1σ')],
              fontsize=8, loc='upper left', bbox_to_anchor=(1.01, 1.0), frameon=False)
    ax.set_title('Shift caused by noise, every noisy sampler cell', fontsize=12, fontweight='bold')
    fig.tight_layout()
    return sf.save(fig, os.path.join(out, 'resumen_ruido'))



# ── F. All methods side by side ───────────────────────────────────────────────
#: (CSV Method, label, colour, marker, filled). Colour: classical blue, quantum
#: red (the repo convention); marker: one per family.
COMPARE_POSTERIOR = [('QMCMC 100%', 'QMCMC 100%', cmq.C_QUANTUM, 'o', True),
                     ('Classical VI', 'classical VI', cmq.C_CLASSICAL, 'D', False),
                     (QVMC_FULL, 'QVMC 100% (+ quantum training)', cmq.C_QUANTUM, 'D', True)]
COMPARE_COST = [('Classical MCMC', 'classical MCMC', cmq.C_CLASSICAL, 'o', False, 'samplers'),
                ('QMCMC 100%', 'QMCMC 100%', cmq.C_QUANTUM, 'o', True, 'samplers'),
                ('Classical VI', 'classical VI', cmq.C_CLASSICAL, 'D', False, 'samplers'),
                (QVMC_FULL, 'QVMC 100% (+ quantum training)', cmq.C_QUANTUM, 'D', True, 'samplers'),
                ('CGA', 'CGA', cmq.C_CLASSICAL, '^', False, 'genetic'),
                ('QGA (q=100%)', 'QGA (q=100%)', '#7f0000', 'v', True, 'genetic')]
COMPARE_DPI = 200


def _grouped(ax, k, j, n, vals, color, marker, filled, label=None, span=0.62):
    """Seed dots and mean ± sd of one method at slot ``j`` of ``n`` around model ``k``."""
    x = k + (j - (n - 1) / 2) * span / max(n - 1, 1)
    sf.draw_seeds(ax, x, vals, width=0.02)
    return sf.mean_bar(ax, x, vals, color, marker=marker, label=label, ms=7,
                       mfc=color if filled else 'white')


def fig_comparison(data, gdata, seeds, out, report):
    """resumen_comparacion: every family against its classical counterpart, plus cost.

    (a) shift of the Ωm posterior mean and (b) width of the Ωm posterior of the
    posterior methods relative to classical MCMC (same seed); (c) Δχ²_grid of
    every QGA rung relative to CGA (same seed); (d) wall time of the classical
    and fully quantum rung of each family. Each family is read at the largest
    grid of each model (nqpp for the samplers, n_bits for the genetic ladder).
    """
    import seed_figures_genetic as sfg
    models = [m for m in sf.models_and_grids(data) if m in sf.models_and_grids(gdata)]
    labels = [sf.short(m) for m in models]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.4))
    (ax, bx), (cx, dx) = axes
    for k, m in enumerate(models):
        cell = data[(m, largest(data, m))]
        for j, (meth, lab, col, mk, fill) in enumerate(COMPARE_POSTERIOR):
            sh = sf.om_shift(cell, meth, seeds)
            wr = sf.sigma_ratio(cell, meth, seeds)
            _grouped(ax, k, j, 3, sh, col, mk, fill, label=lab if k == 0 else None)
            _grouped(bx, k, j, 3, wr, col, mk, fill)
            for tag, v in (('shift', sh), ('width', wr)):
                mm, sd, se, nn = sf.stats(v)
                report.append(f'  compare {tag:5s} {m:5s} {meth:14s} {sf.fmt(mm, sd, se, prec=3)}')
        gcell = gdata[(m, largest(gdata, m))]
        for j, meth in enumerate(sfg.QGA_LEVELS[1:]):
            d = sfg.delta_vs_cga(gcell, meth, seeds)
            _grouped(cx, k, j, 3, d, sfg.COLORS[meth], sfg.MARKERS[meth], True,
                     label=meth if k == 0 else None, span=0.5)
            mm, sd, se, nn = sf.stats(d)
            report.append(f'  compare dchi2 {m:5s} {meth:14s} {sf.fmt(mm, sd, se, prec=4)}')
        for j, (meth, lab, col, mk, fill, fam) in enumerate(COMPARE_COST):
            c = cell if fam == 'samplers' else gcell
            t = sf.per_seed(c, meth, lambda r: sf.num(r, 'Time_s'), seeds)
            _grouped(dx, k, j, len(COMPARE_COST), t, col, mk, fill,
                     label=lab if k == 0 else None, span=0.72)
    for a_, ref in ((ax, 0.0), (bx, 1.0), (cx, 0.0)):
        a_.axhline(ref, color='k', ls='--', lw=0.9)
    for a_ in axes.ravel():
        model_axis(a_, models, labels)
    ax.set_ylabel(r'shift of $\bar\Omega_m$ [$\sigma_{\rm MCMC}$]')
    ax.set_title('(a) Posterior mean relative to classical MCMC', fontsize=11)
    bx.set_ylabel(r'$\sigma(\Omega_m)\,/\,\sigma_{\rm MCMC}(\Omega_m)$')
    bx.set_title('(b) Posterior width relative to classical MCMC', fontsize=11)
    cx.set_yscale('symlog', linthresh=1e-3, linscale=0.6)
    cx.set_ylabel(r'$\Delta\chi^2_{\rm grid}$ = QGA − CGA  (same seed)')
    cx.set_title('(c) Best grid point relative to CGA', fontsize=11)
    dx.set_yscale('log')
    dx.set_ylabel('wall time per method [s]')
    dx.set_title('(d) Simulator cost', fontsize=11)
    dx.grid(alpha=0.3, which='both')
    ax.legend(fontsize=8, loc='best', framealpha=0.9)
    cx.legend(fontsize=8, loc='best', framealpha=0.9)
    dx.legend(fontsize=8, ncol=2, loc='upper center', bbox_to_anchor=(0.5, -0.1), frameon=False)
    fig.text(0.5, 0.005, 'Grey dots: individual seeds; markers and bars: mean ± sd across '
             f'{len(seeds)} seeds; largest grid of each model.', ha='center', fontsize=8,
             color='0.35')
    fig.suptitle('Every method against its classical counterpart', fontsize=13,
                 fontweight='bold')
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    path = os.path.join(out, 'resumen_comparacion')
    os.makedirs(out, exist_ok=True)
    fig.savefig(path + '.png', dpi=COMPARE_DPI, bbox_inches='tight')
    fig.savefig(path + '.pdf', bbox_inches='tight')
    plt.close(fig)
    return path + '.png'


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('root', help='multi-seed samplers campaign (holds seeds_s<NN>/)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--seeds', default=None)
    ap.add_argument('--genetic', default=None, help='multi-seed genetic campaign (genetic_s<NN>/)')
    ap.add_argument('--single-seed', nargs='*', default=[],
                    help='folders of a single-seed campaign (nqpp >= 6 and the noise axis)')
    a = ap.parse_args(argv)
    cmq.set_headless_backend()
    data, taskdirs = sf.load_campaign(a.root, sf.parse_seeds(a.seeds))
    if not data:
        print('no rows found', file=sys.stderr)
        return 1
    seeds = sf.seeds_of(data)
    os.makedirs(a.out, exist_ok=True)
    report = [f'summary_figures: {len(seeds)} seeds {seeds}']
    files = [fig_qmcmc(data, taskdirs, seeds, a.out, report)]
    single_rows = []
    for d in a.single_seed:
        single_rows += noise_plots.load_campaign(d)
    if a.single_seed:
        report.append(f'  single-seed campaign: {len(single_rows)} rows from {a.single_seed}')
    sig = single_seed_sigma(single_rows) if single_rows else None
    files.append(fig_qvmc(data, seeds, a.out, report, single=sig))
    files.append(fig_posteriors(data, taskdirs, seeds, a.out))
    files.append(fig_kl(data, taskdirs, seeds, a.out))
    files.append(fig_rhat(data, taskdirs, seeds, a.out))
    if a.genetic:
        import seed_figures_genetic as sfg
        gdata, gtask = sf.load_campaign(a.genetic, sf.parse_seeds(a.seeds), pattern='genetic_s*',
                                        grid='nb')
        files.append(fig_fitness(gdata, sfg.load_traces(gtask, seeds), seeds, a.out))
        files.append(fig_comparison(data, gdata, seeds, a.out, report))
    if single_rows:
        sh = noise_shifts(single_rows)
        if sh:
            files.append(fig_noise(sh, a.out, report))
    text = '\n'.join(report) + '\n'
    with open(os.path.join(a.out, 'summary_report.txt'), 'w', encoding='utf-8') as fh:
        fh.write(text)
    print(text)
    print(f'{len(files)} summary figures written to {a.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
