"""Figures for `python -m thesis hardware` outputs (reads files only; runs nothing).

Usage:
    python scripts/hw_plots.py results/hw_fez_sn_s42 [--out <folder>]

Reads results.csv, jobs.csv, vi_trace.csv, circuits.json and samples/*.npz
written by the hardware protocol and writes PNG + PDF figures (default:
<run>/figures). A figure whose inputs are missing (a location that did not
run, or a run made before samples/ existed) is skipped with a message.

Visual conventions are those of the simulation figures (cosmo_modular_quantum,
cosmo_genetic_optimizers, noise_plots on thesis-errata), so hardware and
simulation figures can sit side by side:
  * classical baselines in blue (#1f77b4; the genetic one in green #2ca02c),
    quantum runs in warm colors (the autumn_r ramp of the quantumness ladders),
    here ordered ideal -> noisy twin -> device;
  * corner.py with bins=35, smooth=1.0, 1/2/3-sigma 2D levels (0.393, 0.865,
    0.989) with filled bands, line styles cycling solid/dashed/dashdot/...,
    fiducial (Planck prior) black dashed, Planck 2018 green dashed and SH0ES
    purple dotted reference lines; QVMC draws cell-jittered for display only;
  * R-hat as R_max - 1 on a log axis with the 1.01 threshold; KL on a log axis;
  * 8 x 5.2 single panels, grid alpha 0.3, bold suptitles, PNG at 150 dpi + PDF.
"""
from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np

# ── shared visual vocabulary (copied from the simulation code) ────────────────
C_CLASSICAL = '#1f77b4'     # blue   — classical baseline (MCMC / VI)
C_CLASSICAL2 = '#17becf'    # teal   — second classical set in the same panel
C_GENETIC = '#2ca02c'       # green  — classical genetic (CGA)
C_PLANCK = '#2ca02c'        # green  — Planck 2018 reference
C_SHOES = '#9467bd'         # purple — SH0ES reference
LS_CYCLE = ['solid', 'dashed', 'dashdot', 'dotted', 'solid']   # named styles: corner contours reject tuples
SIGMA_LEVELS = (0.393, 0.865, 0.989)          # 1, 2, 3 sigma in 2D
RHAT_THRESHOLD = 1.01
REF_PLANCK = {'label': 'Planck 2018', 'H0': (67.66, 0.42), 'Om': (0.3111, 0.0056)}
REF_SHOES = {'label': 'SH0ES (Riess 2022)', 'H0': (73.04, 1.04), 'Om': None}
FIDUCIAL = {'Om': 0.3111, 'H0': 67.66, 'w': -1.0, 'w0': -1.0, 'wa': 0.0}
LATEX = {'Om': r'$\Omega_m$', 'H0': r'$H_0$', 'w': r'$w$', 'w0': r'$w_0$',
         'wa': r'$w_a$', 'Delta': r'$\Delta$'}
MODEL_LABEL = {'lcdm': r'$\Lambda$CDM', 'wcdm': r'$w$CDM', 'cpl': 'CPL',
               'pede': 'PEDE', 'gede': 'GEDE'}

QLOCS = ('ideal', 'noisy', 'device')
ALG_NAME = {'mcmc': 'QMCMC', 'vi': 'QVMC', 'genetic': 'QGA'}
CLASSICAL_NAME = {'mcmc': 'Classical MCMC', 'vi': 'Classical VI', 'genetic': 'CGA'}


def _warm(i, n=3):
    """Warm ramp of the quantumness ladders (autumn_r), one shade per location."""
    import matplotlib.pyplot as plt
    return plt.cm.autumn_r((0.30, 0.62, 0.95)[i] if n == 3 else 0.25 + 0.75 * (i + 1) / n)


def loc_color(loc):
    return C_CLASSICAL if loc == 'classical' else _warm(QLOCS.index(loc))


def loc_label(alg, loc, device):
    if loc == 'classical':
        return f"{CLASSICAL_NAME[alg]} (same recipe)"
    where = {'ideal': 'ideal (Aer)', 'noisy': f'noisy twin (Aer + {device} calibration)',
             'device': f'{device} (device)'}[loc]
    return f"{ALG_NAME[alg]} — {where}"


# ── I/O ───────────────────────────────────────────────────────────────────────
def _f(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def _csv(path):
    if not os.path.exists(path):
        return []
    with open(path) as fh:
        return list(csv.DictReader(fh))


def _npz(sdir, name):
    p = os.path.join(sdir, name + '.npz')
    return np.load(p, allow_pickle=False) if os.path.exists(p) else None


class Figs:
    def __init__(self, out):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        self.plt, self.out, self.made = plt, out, []
        os.makedirs(out, exist_ok=True)

    def save(self, fig, name):
        """PNG at 150 dpi + vector PDF from the same figure (as _save_fig)."""
        fig.savefig(os.path.join(self.out, name + '.png'), dpi=150, bbox_inches='tight')
        fig.savefig(os.path.join(self.out, name + '.pdf'), bbox_inches='tight')
        self.plt.close(fig)
        self.made.append(name)

    @staticmethod
    def skip(name, why):
        print(f'  skip {name}: {why}')


# ── corner (same arguments as plot_corner_multi) ──────────────────────────────
def _sigma_fill_colors(color, n_levels=3):
    from matplotlib.colors import to_rgba
    r, g, b, _ = to_rgba(color)
    return [(r, g, b, a) for a in [0.0] + [0.10 + 0.11 * i for i in range(n_levels)]]


def corner_multi(F, datasets, colors, labels, names, title, fname):
    import corner
    from matplotlib.lines import Line2D
    allcat = np.vstack(datasets)
    rng_ = []
    for p in range(len(names)):
        lo, hi = np.percentile(allcat[:, p], [0.5, 99.5])
        pad = 0.08 * (hi - lo + 1e-12)
        rng_.append((lo - pad, hi + pad))
    latex = [LATEX.get(n, n) for n in names]

    def _kw(ls, col):
        return dict(labels=latex, bins=35, range=rng_,
                    plot_datapoints=False, plot_density=False, smooth=1.0,
                    levels=SIGMA_LEVELS, fill_contours=True, no_fill_contours=False,
                    contourf_kwargs=dict(colors=_sigma_fill_colors(col)),
                    contour_kwargs=dict(linestyles=ls, linewidths=1.7, alpha=0.95),
                    hist_kwargs=dict(density=True, lw=1.8, ls=ls))

    nd = len(names)
    side = max(3.6, 2.6 + 1.0 * nd)                 # room for the legend in the empty corner
    fig, _ = F.plt.subplots(nd, nd, figsize=(side * nd * 0.75 + 2.0, side * nd * 0.75 + 1.0))
    for k, (data, col) in enumerate(zip(datasets, colors)):
        fig = corner.corner(data, color=col, fig=fig, **_kw(LS_CYCLE[k % len(LS_CYCLE)], col))
    fid = [FIDUCIAL.get(n) for n in names]
    if any(v is not None for v in fid):
        corner.overplot_lines(fig, fid, color='k', ls='--', lw=1.2)
    handles = [Line2D([0], [0], color=c, lw=2.4, ls=LS_CYCLE[k % len(LS_CYCLE)], label=l)
               for k, (c, l) in enumerate(zip(colors, labels))]
    handles.append(Line2D([0], [0], color='k', ls='--', lw=1.2, label='Fiducial (Planck prior)'))
    for ref, col, ls in ((REF_PLANCK, C_PLANCK, 'dashed'), (REF_SHOES, C_SHOES, 'dotted')):
        vec = [(ref.get(n) or (None,))[0] if n in ('Om', 'H0') else None for n in names]
        if any(v is not None for v in vec):
            corner.overplot_lines(fig, vec, color=col, ls=ls, lw=1.8)
            handles.append(Line2D([0], [0], color=col, ls=ls, lw=1.8, label=ref['label']))
    fig.legend(handles=handles, loc='upper right', fontsize=9, bbox_to_anchor=(0.99, 0.94))
    fig.suptitle(title, fontsize=13, fontweight='bold', y=1.02)
    F.save(fig, fname)


def _grid_n(z, ndim):
    return int(round(len(z['target']) ** (1.0 / ndim)))


def jitter_vi(S, z, ndim, seed=42):
    w = np.asarray(z['window'], float)
    n = _grid_n(z, ndim)
    cell = (w[:, 1] - w[:, 0]) / (n - 1)
    return S + np.random.default_rng(seed).uniform(-0.5, 0.5, size=S.shape) * cell


def _style_ax(ax):
    ax.grid(True, alpha=0.3)


# ── main ──────────────────────────────────────────────────────────────────────
def main(run, out):
    F = Figs(out)
    plt = F.plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    res = _csv(os.path.join(run, 'results.csv'))
    jobs = _csv(os.path.join(run, 'jobs.csv'))
    trace = _csv(os.path.join(run, 'vi_trace.csv'))
    meta = {}
    if os.path.exists(os.path.join(run, 'circuits.json')):
        with open(os.path.join(run, 'circuits.json')) as fh:
            meta = json.load(fh)
    sdir = os.path.join(run, 'samples')
    if not res:
        print('  no results.csv')
        return 1
    model, dataset = res[0].get('model', '?'), res[0].get('dataset', '?')
    device, seed = res[0].get('device', '?'), res[0].get('seed', '?')
    cfg = meta.get('config', {})
    mlabel = MODEL_LABEL.get(model, model)
    head = f'{mlabel} | {dataset} | {device} | seed {seed}'
    row = {(r['algorithm'], r['location']): r for r in res}
    refz = _npz(sdir, 'reference_mcmc')
    steps = cfg.get('mcmc_steps', '?')
    iters = cfg.get('vi_iters', '?')
    nqpp = cfg.get('vi_grid', '?')

    # 1. corners: QMCMC and QVMC, every location; 1-to-1 per location --------
    for alg, key in (('mcmc', 'chain'), ('vi', 'samples')):
        sets = []                                   # (data, color, label, loc)
        names = None
        if refz is not None:
            names = [str(n) for n in refz['param_names']]
            sets.append((refz['chain'].reshape(-1, refz['chain'].shape[-1]), C_CLASSICAL,
                         'Classical MCMC (reference, '
                         f"{cfg.get('reference_steps', '?')}×{cfg.get('reference_chains', '?')})",
                         'reference'))
        for loc in ('classical',) + QLOCS:
            z = _npz(sdir, f'{loc}_{alg}')
            if z is None:
                continue
            names = [str(n) for n in z['param_names']]
            S = z[key].reshape(-1, z[key].shape[-1])
            if alg == 'vi':
                S = jitter_vi(S, z, len(names))
            col = C_CLASSICAL2 if loc == 'classical' else loc_color(loc)
            sets.append((S, col, loc_label(alg, loc, device), loc))
        if len(sets) < 2:
            F.skip(f'corner_{alg}', 'no samples/ for this algorithm')
            continue
        extra = (f'[steps={steps}]' if alg == 'mcmc'
                 else f'[iters={iters}, nqpp={nqpp}]  (cell-jittered for display)')
        corner_multi(F, [s[0] for s in sets], [s[1] for s in sets], [s[2] for s in sets], names,
                     f'{head} — {ALG_NAME[alg]} by location  {extra}', f'corner_{alg}')
        for s in sets:
            if s[3] in QLOCS and sets[0][3] == 'reference':
                corner_multi(F, [sets[0][0], s[0]], [sets[0][1], s[1]], [sets[0][2], s[2]], names,
                             f'{head} — {ALG_NAME[alg]} {s[3]} vs reference  {extra}',
                             f'corner_1to1_{alg}_{s[3]}')

    # 2. R-hat (R_max - 1 vs steps), ESS/acceptance/autocorrelation, traces ---
    from qablate import diagnostics
    chains = {loc: _npz(sdir, f'{loc}_mcmc') for loc in ('classical',) + QLOCS}
    chains = {k: v for k, v in chains.items() if v is not None}
    if chains:
        names = [str(n) for n in next(iter(chains.values()))['param_names']]
        nd = len(names)
        fig, ax = plt.subplots(figsize=(8, 5.2))
        for loc, z in chains.items():
            ch = np.swapaxes(z['chain'], 0, 1)                       # (chains, steps, ndim)
            ns = np.unique(np.linspace(50, ch.shape[1], 25).astype(int))
            r = np.array([np.max(diagnostics.rhat(ch[:, :n])) for n in ns])
            ax.semilogy(ns, np.maximum(r - 1, 1e-6), 'o-', color=loc_color(loc),
                        lw=2.8 if loc == 'classical' else 1.9, ms=4,
                        label=loc_label('mcmc', loc, device))
        ax.axhline(RHAT_THRESHOLD - 1.0, color='k', ls='--', lw=1.2,
                   label=rf'$\hat R-1={RHAT_THRESHOLD - 1.0:g}$')
        ax.set_xlabel('Sampling steps (after burn-in)')
        ax.set_ylabel(r'$\hat{R}_{\max}-1$')
        ax.set_title(f'QMCMC convergence by location — {head}\n(total steps = {steps})')
        ax.legend(fontsize=9)
        _style_ax(ax)
        F.save(fig, 'rhat_qmcmc')

        locs = list(chains)
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
        e = np.array([diagnostics.ess(np.swapaxes(chains[l]['chain'], 0, 1)) for l in locs])
        x = np.arange(len(locs))
        for j in range(nd):
            axes[0].bar(x + (j - (nd - 1) / 2) * 0.8 / nd, e[:, j], 0.8 / nd,
                        color=[loc_color(l) for l in locs], edgecolor='k', lw=0.6,
                        hatch=['', '//', '..', 'xx'][j % 4])
        axes[0].set_ylabel('bulk ESS')
        axes[0].legend(handles=[Patch(facecolor='white', edgecolor='k', hatch=['', '//', '..', 'xx'][j % 4],
                                      label=LATEX.get(names[j], names[j])) for j in range(nd)],
                       fontsize=9)
        acc = [float(np.mean(chains[l]['acceptance'])) for l in locs]
        axes[1].bar(x, acc, color=[loc_color(l) for l in locs], edgecolor='k', lw=0.6)
        axes[1].set_ylabel('acceptance fraction')
        for loc in locs:
            ch = np.swapaxes(chains[loc]['chain'], 0, 1)
            acf = np.mean([diagnostics._autocorr_1d(ch[k, :, 0]) for k in range(ch.shape[0])], axis=0)
            axes[2].plot(acf[:200], color=loc_color(loc), lw=1.9, label=loc)
        axes[2].axhline(0, color='0.4', lw=0.9, ls=':')
        axes[2].set_xlabel('lag')
        axes[2].set_ylabel(f'autocorrelation of {LATEX.get(names[0], names[0])}')
        axes[2].legend(fontsize=9)
        for ax in axes[:2]:
            ax.set_xticks(x, locs)
        for ax in axes:
            _style_ax(ax)
        fig.suptitle(f'QMCMC efficiency by location — {head}  [steps={steps}]',
                     fontsize=12, fontweight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        F.save(fig, 'efficiency_qmcmc')

        fig, axes = plt.subplots(nd, len(locs), figsize=(3.6 * len(locs) + 1.0, 2.7 * nd),
                                 squeeze=False, sharey='row')
        for c, loc in enumerate(locs):
            ch = chains[loc]['chain']                                 # (steps, chains, ndim)
            for j in range(nd):
                ax = axes[j, c]
                for k in range(ch.shape[1]):
                    ax.plot(ch[:, k, j], lw=0.6, alpha=0.55, color=loc_color(loc))
                if names[j] in FIDUCIAL:
                    ax.axhline(FIDUCIAL[names[j]], color='k', ls='--', lw=1)
                if c == 0:
                    ax.set_ylabel(LATEX.get(names[j], names[j]))
                if j == nd - 1:
                    ax.set_xlabel('Step')
                ax.grid(True, alpha=0.25)
            axes[0, c].set_title(loc)
        fig.suptitle(f'Trace plots — QMCMC by location — {head}\nMCMC steps = {steps}',
                     fontsize=12, fontweight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.92])
        F.save(fig, 'traces_qmcmc')
    else:
        F.skip('rhat_qmcmc / efficiency_qmcmc / traces_qmcmc', 'no samples/*_mcmc.npz')

    # 3. QVMC KL vs iteration, and vs billed QPU seconds -----------------------
    if trace:
        fig, ax = plt.subplots(figsize=(8, 5.2))
        for loc in QLOCS:
            t = [r for r in trace if r['location'] == loc]
            if not t:
                continue
            it = [int(r['iteration']) for r in t]
            ax.semilogy(it, [max(float(r['kl_exact']), 1e-12) for r in t], color=loc_color(loc),
                        lw=1.9, label=loc_label('vi', loc, device) + ' — exact KL')
            ax.semilogy(it, [max(float(r['kl_shots_perturbed']), 1e-12) for r in t],
                        color=loc_color(loc), lw=1.0, alpha=0.45, ls='dashed')
        k0 = _f(next(iter(row.get(('vi', l), {}).get('kl_start_exact') for l in QLOCS
                          if ('vi', l) in row), None))
        if k0 is not None:
            ax.axhline(k0, color='k', ls='--', lw=1.2, label=f'warm start (exact KL = {k0:.3g})')
        ax.plot([], [], color='0.4', lw=1.0, alpha=0.6, ls='dashed', label='shot KL seen by SPSA')
        ax.set_xlabel('Training iteration')
        ax.set_ylabel(r'KL$(Q_\varphi\,\|\,P_{\rm target})$')
        ax.set_title(f'QVMC training by location — {head}\n(iterations = {iters}, nqpp = {nqpp})')
        ax.legend(fontsize=9)
        _style_ax(ax)
        F.save(fig, 'kl_qvmc')

        t = [r for r in trace if r['location'] == 'device' and _f(r['quantum_s'])]
        if t:
            fig, ax = plt.subplots(figsize=(8, 5.2))
            ax.semilogy([float(r['quantum_s']) for r in t], [float(r['kl_exact']) for r in t],
                        'o-', color=loc_color('device'), lw=1.9, ms=4,
                        label=loc_label('vi', 'device', device))
            ax.set_xlabel('Cumulative billed QPU seconds')
            ax.set_ylabel(r'KL$(Q_\varphi\,\|\,P_{\rm target})$')
            ax.set_title(f'QVMC: quality against billed QPU time — {head}\n'
                         f'(iterations = {iters}, nqpp = {nqpp})')
            ax.legend(fontsize=9)
            _style_ax(ax)
            F.save(fig, 'kl_vs_qpu_seconds_qvmc')
        else:
            F.skip('kl_vs_qpu_seconds_qvmc', 'no billed device iterations in vi_trace.csv')
    else:
        F.skip('kl_qvmc', 'no vi_trace.csv')

    # 4. fidelity: shift (with 2×MCSE) and width ratio per algorithm/location --
    pnames = [p for p in ('Om', 'H0', 'w', 'w0', 'wa', 'Delta')
              if any(f'shift_{p}_sigma' in r for r in res)]
    if pnames:
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
        ticks, i = [], 0
        markers = ['o', 's', '^', 'D']
        for alg in ('mcmc', 'vi'):
            for loc in ('classical',) + QLOCS:
                r = row.get((alg, loc))
                if r is None or r.get('error'):
                    continue
                for j, p in enumerate(pnames):
                    off = (j - (len(pnames) - 1) / 2) * 0.18
                    s, w = _f(r.get(f'shift_{p}_sigma')), _f(r.get(f'width_ratio_{p}'))
                    err = _f(r.get('shift_mcse_sigma'))
                    if s is not None:
                        axes[0].errorbar(i + off, s, yerr=2 * err if err else None,
                                         fmt=markers[j % 4], ms=6, color=loc_color(loc), capsize=3)
                    if w is not None:
                        axes[1].plot(i + off, w, markers[j % 4], ms=6, color=loc_color(loc))
                ticks.append(CLASSICAL_NAME[alg].replace(' ', '\n') if loc == 'classical'
                             else f"{ALG_NAME[alg]}\n{loc}")
                i += 1
        for ax in axes:
            ax.set_xticks(range(len(ticks)), ticks, fontsize=8)
            _style_ax(ax)
        axes[0].axhline(0, color='0.4', lw=0.9, ls=':')
        axes[0].set_ylabel(r'shift $/\ \sigma_{\rm ref}$   (bars: $2\times$MCSE)')
        axes[1].axhline(1.0, color='0.4', lw=0.9, ls=':')
        axes[1].set_ylabel(r'$\sigma\ /\ \sigma_{\rm ref}$')
        axes[0].legend(handles=[Line2D([], [], marker=markers[j % 4], ls='', color='0.3',
                                       label=LATEX.get(p, p)) for j, p in enumerate(pnames)],
                       fontsize=9)
        fig.suptitle(f'Posterior fidelity against the reference — {head}',
                     fontsize=12, fontweight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        F.save(fig, 'fidelity')

    # 5. QVMC: noise on sampling alone (warm start measured) vs after SPSA ------
    pts = [(loc, row[('vi', loc)]) for loc in QLOCS
           if ('vi', loc) in row and not row[('vi', loc)].get('error')]
    if pts and pnames:
        fig, ax = plt.subplots(figsize=(8, 5.2))
        markers = ['o', 's', '^', 'D']
        for k, (loc, r) in enumerate(pts):
            for j, p in enumerate(pnames):
                a, b = _f(r.get(f'start_width_ratio_{p}')), _f(r.get(f'width_ratio_{p}'))
                if a is None or b is None:
                    continue
                xx = k + (j - (len(pnames) - 1) / 2) * 0.2
                ax.annotate('', xy=(xx, b), xytext=(xx, a),
                            arrowprops=dict(arrowstyle='->', color=loc_color(loc), lw=1.6))
                ax.plot(xx, a, markers[j % 4], mfc='white', ms=7, color=loc_color(loc))
                ax.plot(xx, b, markers[j % 4], ms=7, color=loc_color(loc))
        ax.axhline(1.0, color='0.4', lw=0.9, ls=':')
        ax.set_xticks(range(len(pts)), [loc_label('vi', p[0], device).split(' — ')[1] for p in pts],
                      fontsize=9)
        ax.set_ylabel(r'$\sigma\ /\ \sigma_{\rm ref}$')
        ax.legend(handles=[Line2D([], [], marker='o', ls='', mfc='white', color='0.3',
                                  label='warm start measured (sampling noise only)'),
                           Line2D([], [], marker='o', ls='', color='0.3', label='after SPSA')]
                  + [Line2D([], [], marker=markers[j % 4], ls='', color='0.3', label=LATEX.get(p, p))
                     for j, p in enumerate(pnames)], fontsize=9)
        ax.set_title(f'QVMC: noise on sampling vs reaction of training — {head}\n'
                     f'(iterations = {iters}, nqpp = {nqpp})')
        _style_ax(ax)
        F.save(fig, 'sampling_vs_training_qvmc')

    # 6. genetic: fitness curve and convergence (as plot_fitness_curve /
    #    plot_genetic_convergence) ----------------------------------------------
    gas = {loc: _npz(sdir, f'{loc}_genetic') for loc in ('classical',) + QLOCS}
    gas = {k: v for k, v in gas.items() if v is not None}
    if gas:
        floor = None
        for loc in ('ideal', 'classical', 'noisy', 'device'):
            floor = floor or _f(row.get(('genetic', loc), {}).get('chi2_grid_floor'))
        gcol = {loc: (C_GENETIC if loc == 'classical' else loc_color(loc)) for loc in gas}
        glab = {loc: (f"{CLASSICAL_NAME['genetic']} (classical operators)" if loc == 'classical'
                      else loc_label('genetic', loc, device)) for loc in gas}

        fig, ax = plt.subplots(figsize=(9, 5))
        ends, lo_all, hi_all = [], [], []
        for i, (loc, z) in enumerate(gas.items()):
            g = np.arange(1, len(z['best_log_prob']) + 1)
            best, mean = -2 * z['best_log_prob'], -2 * z['mean_log_prob']
            ax.plot(g, best, color=gcol[loc], lw=max(1.3, 3.4 - 0.55 * i),
                    ls=LS_CYCLE[i % len(LS_CYCLE)], label=f'{glab[loc]} — best')
            ax.plot(g, mean, color=gcol[loc], lw=1.0, alpha=0.45, ls=LS_CYCLE[i % len(LS_CYCLE)])
            ax.annotate(f'{best[-1]:.3f}', xy=(g[-1], best[-1]), xytext=(5, 0),
                        textcoords='offset points', fontsize=8, color=gcol[loc],
                        fontweight='bold', va='center', annotation_clip=False)
            lo_all.append(best.min()); hi_all.append(best.max())
            ends.append(len(g))
        if floor is not None:
            ax.axhline(floor, color='k', ls='--', lw=1.2, label=f'grid floor ({floor:.3f})')
            lo_all.append(floor)
        lo, hi = min(lo_all), max(hi_all)
        pad = max(0.15 * (hi - lo), 0.05)
        ax.set_ylim(lo - pad, hi + 3 * pad)          # zoom on the best curves
        ytop = hi + 3 * pad
        mean_min = min(float(np.nanmin(-2 * z['mean_log_prob'])) for z in gas.values())
        if mean_min > ytop:
            ax.text(0.01, 0.98, f'population means (faint) lie above the range (min {mean_min:.1f})',
                    transform=ax.transAxes, fontsize=8, va='top', color='0.35')
        ax.set_xlim(0, max(ends) * 1.12)
        ax.ticklabel_format(axis='y', style='plain', useOffset=False)
        ax.set_xlabel('Generation', fontsize=12)
        ax.set_ylabel(r'$\chi^2$  (bold: best; faint: population mean)', fontsize=12)
        ax.set_title(f'{head} — genetic fitness convergence by location', fontsize=12,
                     fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        fig.tight_layout()
        F.save(fig, 'fitness_genetic')

        names = [str(n) for n in next(iter(gas.values()))['param_names']]
        d = len(names)
        fig, axes = plt.subplots(2, d, figsize=(4.6 * d, 8.2), squeeze=False)
        mk = ['o', 's', '^', 'D']
        for i, (loc, z) in enumerate(gas.items()):
            pops, lps = z['populations'], z['log_probs']                     # (gen, pop, d), (gen, pop)
            best_traj = np.array([pops[t][np.nanargmax(np.where(np.isfinite(lps[t]), lps[t], -np.inf))]
                                  for t in range(len(pops))])
            gens = np.arange(1, len(pops) + 1)
            for j in range(d):
                axes[0][j].plot(gens, best_traj[:, j], '-', color=gcol[loc], lw=max(1.4, 4.0 - 0.7 * i),
                                marker=mk[i % 4], markevery=max(1, len(gens) // 8), ms=5, alpha=0.95,
                                label=glab[loc] if j == 0 else None, zorder=2 + i)
                axes[0][j].ticklabel_format(axis='y', style='plain', useOffset=False)
                fin = pops[-1][:, j]
                axes[1][j].errorbar(z['theta_best'][j], i, xerr=np.std(fin), fmt=mk[i % 4], ms=7,
                                    color=gcol[loc], capsize=4, lw=1.8)
        for j in range(d):
            for row_ in (0, 1):
                ax = axes[row_][j]
                ax.grid(True, alpha=0.25, lw=0.6)
                for s in ('top', 'right'):
                    ax.spines[s].set_visible(False)
            fid = FIDUCIAL.get(names[j])
            if fid is not None:
                axes[1][j].axvline(fid, ls=':', color='0.45', lw=1.2)
            axes[0][j].set_xlabel('Generation', fontsize=11)
            axes[0][j].set_ylabel(LATEX.get(names[j], names[j]), fontsize=12)
            axes[1][j].set_xlabel(LATEX.get(names[j], names[j]) + '  (MAP ± final population std)',
                                  fontsize=11)
            axes[1][j].set_yticks(range(len(gas)), ['CGA' if l == 'classical' else l for l in gas],
                                  fontsize=10)
        fig.legend(*axes[0][0].get_legend_handles_labels(), loc='upper center', ncol=2, fontsize=9,
                   bbox_to_anchor=(0.5, 0.955))
        fig.suptitle(f'{head} — genetic convergence and final estimate by location',
                     fontsize=13, fontweight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.88])
        F.save(fig, 'genetic_convergence')
    else:
        F.skip('fitness_genetic / genetic_convergence', 'no samples/*_genetic.npz')

    # 7. total time: classical on the PC vs simulation vs QPU -------------------
    algs = [a for a in ('vi', 'genetic', 'mcmc') if any((a, l) in row for l in ('classical',) + QLOCS)]
    if algs:
        fig, axes = plt.subplots(1, len(algs), figsize=(4.8 * len(algs), 5.2), squeeze=False)
        seg = {'PC (classical algorithm)': C_CLASSICAL, 'PC part of the run': '#9ecae1',
               'Aer simulation': _warm(0), 'IBM queue': '#fdd0a2',
               'billed QPU time': _warm(2), 'network / metadata / wait': '#bdbdbd'}
        for k, alg in enumerate(algs):
            ax, xs, names_ = axes[0, k], [], []
            for loc in ('classical',) + QLOCS:
                r = row.get((alg, loc))
                if r is None:
                    continue
                x = len(xs)
                xs.append(x)
                names_.append(loc)
                if r.get('error'):
                    ax.text(x, 1, 'failed', ha='center', va='bottom', fontsize=8, rotation=90)
                    continue
                if loc == 'classical':
                    parts = [(_f(r.get('wall_s')) or 0.0, 'PC (classical algorithm)')]
                else:
                    pc = _f(r.get('classical_s')) or 0.0
                    backend = _f(r.get('backend_wall_s')) or 0.0
                    if loc == 'device':
                        q, b = _f(r.get('queue_s')) or 0.0, _f(r.get('quantum_s')) or 0.0
                        parts = [(pc, 'PC part of the run'), (q, 'IBM queue'), (b, 'billed QPU time'),
                                 (max(backend - q - b, 0.0), 'network / metadata / wait')]
                    else:
                        parts = [(pc, 'PC part of the run'), (backend, 'Aer simulation')]
                bottom = 0.0
                for v, lab in parts:
                    ax.bar(x, v, bottom=bottom, color=seg[lab], edgecolor='k', lw=0.5)
                    bottom += v
                ax.text(x, bottom, f'{bottom:.0f} s' if bottom >= 10 else f'{bottom:.1f} s',
                        ha='center', va='bottom', fontsize=8)
            ax.set_xticks(xs, names_)
            ax.set_yscale('symlog', linthresh=1)
            ax.set_ylabel('wall-clock time [s]')
            ax.set_title(ALG_NAME[alg], fontsize=12, fontweight='bold')
            ws = _f(row.get((alg, 'ideal'), {}).get('warm_start_s')) if alg == 'vi' else None
            if ws:
                ax.text(0.02, 0.98, f'+ warm start on the PC (once, shared): {ws:.0f} s',
                        transform=ax.transAxes, fontsize=8, va='top')
            _style_ax(ax)
        fig.legend(handles=[Patch(facecolor=c, edgecolor='k', lw=0.5, label=l) for l, c in seg.items()],
                   loc='lower center', ncol=3, fontsize=9, bbox_to_anchor=(0.5, -0.02))
        fig.suptitle(f'Total time: classical on the PC vs simulation vs QPU — {head}',
                     fontsize=12, fontweight='bold')
        fig.tight_layout(rect=[0, 0.1, 1, 0.94])
        F.save(fig, 'time_total')

    # 8. cost per device job -----------------------------------------------------
    dj = [j for j in jobs if j.get('location') == 'device']
    if dj:
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
        labs = sorted({j['label'] for j in dj})
        pal = [_warm(2), _warm(1), _warm(0), C_SHOES, '#8c564b']
        for i, lab in enumerate(labs):
            jj = [j for j in dj if j['label'] == lab]
            axes[0].scatter([int(j['n_circuits']) * int(j['shots']) for j in jj],
                            [float(j['quantum_s']) for j in jj], s=22, color=pal[i % len(pal)],
                            marker=['o', 's', '^', 'D', 'v'][i % 5], label=lab)
            axes[1].scatter([float(j['queue_s']) for j in jj], [float(j['quantum_s']) for j in jj],
                            s=22, color=pal[i % len(pal)], marker=['o', 's', '^', 'D', 'v'][i % 5],
                            label=lab)
        axes[0].set_xscale('log')
        axes[0].set_xlabel('executions per job (circuits × shots)')
        axes[1].set_xlabel('queue time [s]')
        for ax in axes:
            ax.set_ylabel('billed QPU seconds')
            ax.legend(fontsize=9)
            _style_ax(ax)
        est = sum(j.get('quantum_estimated') == 'True' for j in dj)
        tot = sum(float(j['quantum_s']) for j in dj)
        fig.suptitle(f'{device}: billed time per job — {len(dj)} jobs, {tot:.0f} s billed '
                     f'({est} estimated) — {head}', fontsize=12, fontweight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        F.save(fig, 'device_cost_per_job')
    else:
        F.skip('device_cost_per_job', 'no device jobs in jobs.csv')

    print(f'  {len(F.made)} figures in {out}: ' + ', '.join(F.made))
    return 0


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('run', help='folder written by python -m thesis hardware --out')
    ap.add_argument('--out', default=None, help='figure folder (default: <run>/figures)')
    a = ap.parse_args()
    raise SystemExit(main(a.run, a.out or os.path.join(a.run, 'figures')))
