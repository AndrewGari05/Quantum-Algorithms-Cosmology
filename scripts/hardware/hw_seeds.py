"""Seed-to-seed spread of `python -m thesis hardware` runs (reads files only; runs nothing).

Usage:
    python scripts/hw_seeds.py results/hw_fez_sn_s42 results/hw_fez_sn_s43 [...] [--out results/hw_seeds]

Each argument is one run folder (one seed). Every figure shows, per location,
the mean over seeds as a bar or marker, every seed as a dot, and an error bar:

  * n = 2 seeds: the error bar is the RANGE (min to max). Two values do not give
    a usable standard deviation, so nothing here is a 1-sigma interval.
  * n >= 3 seeds: the error bar is the sample standard deviation (ddof = 1).

The convention is printed in every title and in seeds_summary.csv.

Figures (PNG + PDF in --out):
  seeds_qvmc_width    sigma/sigma_ref of the QVMC posterior, before training
                      (warm start measured: noise on sampling only) and after SPSA
  seeds_qvmc_kl       exact KL of the final QVMC state
  seeds_qmcmc         QMCMC mean shift / sigma_ref (with 2x MCSE band), R-hat, ESS
  seeds_qga           best chi2 minus the grid floor (CGA and QGA)
  seeds_cost          billed QPU seconds per algorithm and seed (device only)
plus seeds_summary.csv (one row per metric x location: n, mean, std, min, max).

Colors follow scripts/hw_plots.py (classical blue / green, ideal -> noisy -> device
on the autumn_r ramp).
"""
from __future__ import annotations

import argparse
import csv
import os

import numpy as np

C_CLASSICAL = '#1f77b4'
C_GENETIC = '#2ca02c'
QLOCS = ('ideal', 'noisy', 'device')
LOC_LABEL = {'classical': 'clásico', 'ideal': 'ideal', 'noisy': 'simulador\ncon ruido',
             'device': 'dispositivo'}


def _warm(i):
    import matplotlib.pyplot as plt
    return plt.cm.autumn_r((0.30, 0.62, 0.95)[i])


def loc_color(loc, alg=None):
    if loc == 'classical':
        return C_GENETIC if alg == 'genetic' else C_CLASSICAL
    return _warm(QLOCS.index(loc))


def _f(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def load(run):
    with open(os.path.join(run, 'results.csv'), newline='') as fh:
        rows = list(csv.DictReader(fh))
    jobs = []
    p = os.path.join(run, 'jobs.csv')
    if os.path.exists(p):
        with open(p, newline='') as fh:
            jobs = list(csv.DictReader(fh))
    seed = rows[0]['seed'] if rows else os.path.basename(run)
    device = rows[0].get('device', '?') if rows else '?'
    return seed, device, {(r['algorithm'], r['location']): r for r in rows if not r.get('error')}, jobs


def spread(vals):
    """(center, lower error, upper error, rule) for a list of seed values."""
    v = np.asarray([x for x in vals if x is not None], dtype=float)
    if len(v) == 0:
        return None
    m = float(v.mean())
    if len(v) >= 3:
        s = float(v.std(ddof=1))
        return m, s, s, 'std'
    return m, m - float(v.min()), float(v.max()) - m, 'range'


def rule_text(n):
    if n >= 3:
        return f'{n} semillas: barra = media, error = desviación estándar'
    if n == 2:
        return '2 semillas: barra = media, error = rango (mín–máx), no es 1σ'
    return '1 semilla: sin barra de error'


class Figs:
    def __init__(self, out):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        self.plt, self.out, self.made = plt, out, []
        os.makedirs(out, exist_ok=True)

    def save(self, fig, name):
        fig.savefig(os.path.join(self.out, name + '.png'), dpi=150, bbox_inches='tight')
        fig.savefig(os.path.join(self.out, name + '.pdf'), bbox_inches='tight')
        self.plt.close(fig)
        self.made.append(name)


def _style(ax):
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)


def _bars(ax, x, vals_by_seed, color, width, label=None, hatch=None, fill=True):
    """Bar at the mean, error bar per the spread rule, one dot per seed."""
    sp = spread(vals_by_seed)
    if sp is None:
        return
    m, lo, hi, _ = sp
    ax.bar(x, m, width, color=color if fill else 'white', edgecolor=color if not fill else 'black',
           linewidth=1.2 if not fill else 0.6, hatch=hatch, label=label, zorder=2)
    if len([v for v in vals_by_seed if v is not None]) >= 2:
        ax.errorbar(x, m, yerr=[[lo], [hi]], fmt='none', ecolor='black', elinewidth=1.3,
                    capsize=5, zorder=4)
    for v in vals_by_seed:
        if v is not None:
            ax.plot(x, v, 'o', ms=5, mfc='white', mec='black', mew=1.0, zorder=5)


def main(runs, out):
    data = [load(r) for r in runs]
    seeds = [d[0] for d in data]
    device = data[0][1]
    n = len(data)
    F = Figs(out)
    plt = F.plt
    head = f'ΛCDM | CC+BAO+Pantheon | {device} | semillas {", ".join(seeds)}'
    summary = []

    def col(alg, loc, key):
        return [_f(d[2].get((alg, loc), {}).get(key)) for d in data]

    def record(metric, loc, vals):
        v = [x for x in vals if x is not None]
        if not v:
            return
        a = np.asarray(v, dtype=float)
        summary.append({'metric': metric, 'location': loc, 'n': len(a), 'mean': a.mean(),
                        'std': a.std(ddof=1) if len(a) >= 2 else '', 'min': a.min(), 'max': a.max(),
                        'error_bar_rule': 'std (ddof=1)' if len(a) >= 3 else ('range' if len(a) == 2 else 'none'),
                        'seeds': ' '.join(str(x) for x in vals)})

    # 1. QVMC widths: before (warm start measured) and after SPSA ------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    for ax, p, pl in zip(axes, ('Om', 'H0'), (r'$\Omega_m$', r'$H_0$')):
        for i, loc in enumerate(QLOCS):
            start, final = col('vi', loc, f'start_width_ratio_{p}'), col('vi', loc, f'width_ratio_{p}')
            record(f'qvmc_width_start_{p}', loc, start)
            record(f'qvmc_width_final_{p}', loc, final)
            _bars(ax, i - 0.18, start, loc_color(loc), 0.34, fill=False,
                  label='antes de entrenar (solo muestreo)' if i == 0 else None)
            _bars(ax, i + 0.18, final, loc_color(loc), 0.34,
                  label='después de 60 SPSA' if i == 0 else None)
        ax.axhline(1.0, color='0.3', ls='--', lw=1.0, label='MCMC de referencia')
        ax.set_xticks(range(3), [LOC_LABEL[l] for l in QLOCS])
        ax.set_title(pl, fontsize=13)
        _style(ax)
    axes[0].set_ylabel(r'$\sigma_{\rm QVMC}\,/\,\sigma_{\rm ref}$')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=10, frameon=False,
               bbox_to_anchor=(0.5, 0.0))           # below the panels: never covers a bar or a seed
    fig.suptitle(f'QVMC: ancho del posterior por ubicación — {head}\n({rule_text(n)}; puntos = cada semilla)',
                 fontsize=12, fontweight='bold')
    fig.tight_layout(rect=[0, 0.07, 1, 0.92])
    F.save(fig, 'seeds_qvmc_width')

    # 2. QVMC final exact KL -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5.2))
    for i, loc in enumerate(QLOCS):
        v = col('vi', loc, 'kl_final_exact')
        record('qvmc_kl_final', loc, v)
        _bars(ax, i, v, loc_color(loc), 0.55)
    ks = [x for x in col('vi', 'ideal', 'kl_start_exact') if x is not None]
    if ks:
        ax.axhline(float(np.mean(ks)), color='black', ls='--', lw=1.0,
                   label=(f'arranque en caliente, media de las semillas (KL = {np.mean(ks):.3f}; '
                          f'por semilla: {", ".join(f"{k:.3f}" for k in ks)})'))
        ax.legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.16), frameon=False)  # below the axes
    ax.set_xticks(range(3), [LOC_LABEL[l] for l in QLOCS])
    ax.set_ylabel(r'KL$(Q_\varphi\,\|\,P)$ final (exacta)')
    ax.set_title(f'QVMC: KL final — {head}\n({rule_text(n)})', fontsize=11)
    _style(ax)
    F.save(fig, 'seeds_qvmc_kl')

    # 3. QMCMC: shifts with MCSE, R-hat, ESS ---------------------------------------------
    locs = ('classical',) + QLOCS
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.0))
    mcse = [x for loc in locs for x in col('mcmc', loc, 'shift_mcse_sigma') if x is not None]
    band = 2 * float(np.mean(mcse)) if mcse else None
    for j, (p, mk) in enumerate((('Om', 'o'), ('H0', 's'))):
        for i, loc in enumerate(locs):
            v = col('mcmc', loc, f'shift_{p}_sigma')
            record(f'qmcmc_shift_{p}', loc, v)
            sp = spread(v)
            if sp is None:
                continue
            x = i + (j - 0.5) * 0.25
            m, lo, hi, _ = sp
            axes[0].errorbar(x, m, yerr=[[lo], [hi]] if n >= 2 else None, fmt=mk, ms=8,
                             color=loc_color(loc), mec='black', capsize=4, ecolor='black')
            for s in v:
                if s is not None:
                    axes[0].plot(x, s, '.', color='0.25', ms=5)
    if band:
        axes[0].axhspan(-band, band, color='0.85', zorder=0,
                        label=f'±2×MCSE (≈±{band:.2f}σ): no distinguible de cero')
    axes[0].axhline(0, color='0.3', lw=0.9)
    axes[0].plot([], [], 'o', color='0.4', label=r'$\Omega_m$')
    axes[0].plot([], [], 's', color='0.4', label=r'$H_0$')
    h0, l0 = axes[0].get_legend_handles_labels()
    axes[0].set_ylabel(r'desplazamiento de la media $/\ \sigma_{\rm ref}$')
    axes[0].set_title('Sesgo')
    for k, (key, ttl, ref) in enumerate((('rhat_max', r'$\hat R_{\max}$', 1.01),
                                         ('ess_min', 'ESS mínimo', None)), start=1):
        for i, loc in enumerate(locs):
            v = col('mcmc', loc, key)
            record(f'qmcmc_{key}', loc, v)
            _bars(axes[k], i, v, loc_color(loc), 0.6)
        if ref:
            axes[k].axhline(ref, color='black', ls='--', lw=1.0, label=r'$\hat R = 1.01$')
            axes[k].set_ylim(1.0, None)
            hk, lk = axes[k].get_legend_handles_labels()
            h0, l0 = h0 + hk, l0 + lk
        axes[k].set_title(ttl)
    for ax in axes:
        ax.set_xticks(range(len(locs)), [LOC_LABEL[l] for l in locs], fontsize=9)
        _style(ax)
    fig.legend(h0, l0, loc='lower center', ncol=len(l0), fontsize=10, frameon=False,
               bbox_to_anchor=(0.5, 0.0))           # below the panels: never covers a marker
    fig.suptitle(f'QMCMC por ubicación — {head}\n({rule_text(n)}; puntos = cada semilla)',
                 fontsize=12, fontweight='bold')
    fig.tight_layout(rect=[0, 0.08, 1, 0.90])
    F.save(fig, 'seeds_qmcmc')

    # 4. QGA: best chi2 minus grid floor -------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5.2))
    for i, loc in enumerate(locs):
        best, floor = col('genetic', loc, 'chi2_estimate'), col('genetic', loc, 'chi2_grid_floor')
        gap = [b - f if b is not None and f is not None else None for b, f in zip(best, floor)]
        record('qga_chi2_minus_floor', loc, gap)
        _bars(ax, i, gap, loc_color(loc, 'genetic'), 0.55)
    ax.axhline(0, color='black', ls='--', lw=1.0, label='piso de la malla (óptimo alcanzable)')
    ax.set_xticks(range(len(locs)), [LOC_LABEL[l] for l in locs])
    ax.set_ylabel(r'$\chi^2_{\rm mejor} - \chi^2_{\rm piso}$')
    ax.set_title(f'QGA: distancia al óptimo de la malla — {head}\n({rule_text(n)})', fontsize=11)
    ax.legend(fontsize=9, loc='upper left')
    _style(ax)
    F.save(fig, 'seeds_qga')

    # 5. Billed QPU seconds per algorithm and seed ---------------------------------------
    algs = ('vi', 'genetic', 'mcmc')
    names = {'vi': 'QVMC', 'genetic': 'QGA', 'mcmc': 'QMCMC'}
    fig, ax = plt.subplots(figsize=(8, 5.2))
    w = 0.8 / n
    for k, d in enumerate(data):
        vals = [_f(d[2].get((a, 'device'), {}).get('quantum_s')) or 0.0 for a in algs]
        xs = np.arange(len(algs)) + (k - (n - 1) / 2) * w
        ax.bar(xs, vals, w, color=plt.cm.Reds(0.45 + 0.4 * k / max(1, n - 1)), edgecolor='black',
               label=f'semilla {d[0]} (total {sum(vals):.0f} s)')
        for x, v in zip(xs, vals):
            ax.text(x, v + 2, f'{v:.0f}', ha='center', fontsize=8)
    for a in algs:
        record(f'quantum_s_{a}', 'device', [_f(d[2].get((a, 'device'), {}).get('quantum_s')) for d in data])
    ax.set_xticks(range(len(algs)), [names[a] for a in algs])
    ax.set_ylabel('QPU facturada [s]')
    ax.set_title(f'Costo en {device} por semilla — {head}', fontsize=11)
    ax.legend(fontsize=9)
    _style(ax)
    F.save(fig, 'seeds_cost')

    with open(os.path.join(out, 'seeds_summary.csv'), 'w', newline='') as fh:
        wr = csv.DictWriter(fh, fieldnames=['metric', 'location', 'n', 'mean', 'std', 'min', 'max',
                                            'error_bar_rule', 'seeds'])
        wr.writeheader()
        for r in summary:
            wr.writerow({k: (f'{v:.6g}' if isinstance(v, float) else v) for k, v in r.items()})
    print(f'  {len(F.made)} figures + seeds_summary.csv in {out}: {", ".join(F.made)}')
    print(f'  {rule_text(n)}')
    return 0


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('runs', nargs='+', help='run folders, one per seed')
    ap.add_argument('--out', default=None, help='output folder (default: <parent of first run>/hw_seeds)')
    a = ap.parse_args()
    out = a.out or os.path.join(os.path.dirname(os.path.abspath(a.runs[0])), 'hw_seeds')
    raise SystemExit(main(a.runs, out))
