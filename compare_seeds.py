#!/usr/bin/env python3
"""
compare_seeds.py — What survives a change of seed?

A campaign with a single seed cannot tell whether a difference is real or
luck of the random number generator. This script takes several identical
runs that differ only in `--seed` and answers three things:

    1. How much does each number move when the seed changes? (the spread)
    2. Are the differences I want to report LARGER than that spread?
    3. Do the faithful cells still coincide on EVERY seed?

The third one is the most important and the one nobody usually checks: a
faithful cell must give identical results within one seed. If it matches on
seed 1 but not on seed 3, that is not a fluctuation — it is a bug that seed
42 was hiding.

The unit of measure of the whole report is the seed-to-seed spread itself.
Saying "the classical KL is 1.6250 and the quantum one 1.6286" means nothing
on its own; saying "they differ by 0.4 seed standard deviations" does.

Usage:

    python compare_seeds.py results/seed_*
    python compare_seeds.py run_a run_b run_c

Each argument is the master folder of ONE seed. The seed is read from the
`seed` column of the CSV, not from the folder name.

Standard library only: it runs on the HPC node without activating the
environment.
"""

from __future__ import annotations

import csv
import glob

import campaign_io
import math
import os
import re
import sys
from collections import defaultdict

#: Pairs that MUST coincide exactly within one seed, with the reason why it
#: is required. They are the same as in `triage_campaign.py`; if you change
#: one there, change it here.
FAITHFUL_PAIRS = [
    ('QMCMC 50%', 'QMCMC 100%', 'the quantum acceptance reproduces Metropolis'),
    ('QVMC 67%', 'QVMC 100%', 'the normalization returns the exact sum'),
    ('CGA', 'QGA (q=0%)', 'quantumness 0% means the classical operators'),
]

#: Comparisons whose whole point is that the difference is NOT zero, and
#: that can only be defended if they exceed the seed-to-seed spread.
COMPARISONS = [
    ('Classical VI', 'QVMC 100%', 'final_KL',
     'classical vs quantum at matched budget'),
    ('Classical MCMC', 'QMCMC 100%', 'acceptance',
     'classical vs quantum acceptance'),
]


def param_columns(cols):
    """Cosmological parameter columns present in the CSV.

    The names depend on the schema: `Om_mean` for samplers, `p1_mean` for
    the genetic sweep. They are discovered by suffix instead of hard-coded.

    Args:
        cols: set of column names.

    Returns:
        Sorted list of the columns ending in `_mean`.

    Examples:
        >>> param_columns({'Method', 'Om_mean', 'Om_std', 'chi2'})
        ['Om_mean']
        >>> param_columns({'Method', 'chi2'})
        []
    """
    return sorted(c for c in cols if c.endswith('_mean'))


def noise_level_of(row, path):
    """Noise level of a row: the column if present, otherwise the folder.

    Campaigns before 2026-09-04 have no `noise` column, but the runner does
    put the level in the folder name.

    Examples:
        >>> noise_level_of({'noise': 'readout'}, 'x/noise-full/r.csv')
        'readout'
        >>> noise_level_of({}, 'x/samplers_lcdm_noise-full/r.csv')
        'full'
        >>> noise_level_of({'noise': ''}, 'x/genetic_lcdm_nb6/r.csv')
        'none'
    """
    if row.get('noise'):
        return row['noise']
    m = re.search(r'noise-([A-Za-z0-9_.-]+)', path.replace('\\', '/'))
    return m.group(1) if m else 'none'


def model_from_path(path, known=('lcdm', 'pede', 'wcdm', 'gede', 'cpl')):
    """Infer the model from the path when the CSV has no `model` column.

    Examples:
        >>> model_from_path('run/model_wcdm/results_config.csv')
        'wcdm'
        >>> model_from_path('run/model_wcdm/resultados_config.csv')
        'wcdm'
        >>> model_from_path('run/results_config.csv')
        '?'
    """
    parts = path.replace('\\', '/').split('/')
    for p in parts:
        m = re.match(r'model_(\w+)$', p)
        if m and m.group(1) in known:
            return m.group(1)
    for p in parts:
        m = re.search(r'_(' + '|'.join(known) + r')_', p)
        if m:
            return m.group(1)
    return '?'


def _num(x):
    """Convert to float, or None if it cannot be converted."""
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def load_rows(folders):
    """Read every run and index it by cell and seed.

    Args:
        folders: paths of the master folders, one per seed.

    Returns:
        `(cells, seeds, columns)`.

        `cells` maps `(task, model, dataset, noise, method)` to a dict
        `seed -> row`. `seeds` is the set of seeds seen and `columns` the
        union of column names.

    The identity of a cell is the path of the CSV **relative to the folder of
    its seed**, not the `nqpp`. Two reasons:

      * the classical rungs write `nqpp = '—'` while their quantum partners
        write the number, so grouping by nqpp split every pair and the
        classical-vs-quantum comparisons were NEVER made — they did not
        fail, they simply did not show up;
      * the relative path is identical across seeds
        (`model_lcdm/results_config.csv`) and different across tasks, which
        is exactly what is needed. A legacy file name
        (`resultados_config.csv`) is mapped to the current one, so a legacy
        seed folder still pairs with a new one.
    """
    cells = defaultdict(dict)
    seeds, columns = set(), set()
    for folder in folders:
        root = os.path.abspath(folder)
        # [E-QPU6] Only the per-model CSV: the per-task cumulative file
        # repeats the same rows and used to double every cell.
        for p in campaign_io.result_csvs_any_layout(folder):
            try:
                rows = list(csv.DictReader(open(p, newline='')))
            except OSError:
                continue
            if not rows:
                continue
            columns |= set(rows[0].keys())
            path_model = model_from_path(p)
            task = campaign_io.canonical_csv_name(
                os.path.relpath(os.path.abspath(p), root).replace('\\', '/'))
            for r in rows:
                s = r.get('seed')
                if s in (None, ''):
                    continue
                seeds.add(s)
                key = (task, r.get('model') or path_model,
                       r.get('dataset', ''), noise_level_of(r, p),
                       (r.get('Method') or '').strip())
                cells[key][s] = r
    return cells, seeds, columns


def dispersion(values):
    """Mean and sample standard deviation of a list of numbers.

    Returns:
        `(mean, deviation, n)`. The deviation is 0.0 with a single value.

    Examples:
        >>> m, s, n = dispersion([1.0, 2.0, 3.0])
        >>> round(m, 3), round(s, 3), n
        (2.0, 1.0, 3)
        >>> dispersion([5.0])
        (5.0, 0.0, 1)
        >>> dispersion([])
        (None, None, 0)
    """
    v = [x for x in values if x is not None]
    if not v:
        return None, None, 0
    m = sum(v) / len(v)
    if len(v) == 1:
        return m, 0.0, 1
    var = sum((x - m) ** 2 for x in v) / (len(v) - 1)
    return m, math.sqrt(var), len(v)


def in_sigmas(a_vals, b_vals):
    """Paired comparison of two methods run with the same seeds.

    [E-QPU7] Both methods share every seed, so the right test is a paired
    t-test on the per-seed differences d_s = a_s - b_s. The previous version
    divided the difference of means by sqrt(sa^2 + sb^2), the per-seed
    spreads, which ignores the pairing and is sqrt(n) too conservative (a
    systematic offset at t = 20.8 was reported as 0.54 "sigmas").

    Args:
        a_vals, b_vals: values of each method, one per seed, same order.

    Returns:
        `(mean_diff, standard_error, t_statistic)`, or `(None, None, None)`
        with fewer than two usable pairs. `t_statistic` is None when every
        difference is identical (zero standard error).

    Examples:
        >>> d, se, t = in_sigmas([1.0, 1.1, 0.9], [2.0, 2.1, 1.9])
        >>> round(d, 3), t is None
        (-1.0, True)
        >>> d, se, t = in_sigmas([1.00, 1.12, 0.93], [0.99, 1.10, 0.92])
        >>> round(t, 2)
        4.0
        >>> in_sigmas([1.0], [2.0])[2] is None
        True
    """
    pairs = [(a, b) for a, b in zip(a_vals, b_vals)
             if a is not None and b is not None]
    if len(pairs) < 2:
        return None, None, None
    d = [a - b for a, b in pairs]
    m, sd, n = dispersion(d)
    se = sd / math.sqrt(n)
    return m, se, (m / se if se > 1e-15 * max(1.0, abs(m)) else None)


def p_value_two_sided(t, n):
    """Two-sided p-value of a paired t statistic with n pairs (n-1 dof)."""
    from scipy import stats
    return float(2 * stats.t.sf(abs(t), n - 1))


def review(folders):
    """Print the full seed-comparison report."""
    cells, seeds, columns = load_rows(folders)
    print('=' * 74)
    print('SEED COMPARISON')
    print('=' * 74)
    if not cells:
        print('\nFound no CSV with a `seed` column in those folders.')
        print('Campaigns before 2026-09-04 do not have it: the seed study')
        print('needs runs of the corrected code.')
        return 2

    order = sorted(seeds, key=lambda s: (_num(s) is None, _num(s), s))
    print(f'\nSeeds found: {len(order)}  ->  {", ".join(order)}')
    if len(order) < 2:
        print('\nWith ONE seed only there is no spread to measure. This')
        print('report needs at least two runs that differ only in')
        print('--seed; with three or more the error bars are credible.')

    fields = param_columns(columns)
    metrics = [c for c in ('final_KL', 'acceptance', 'ESS', 'chi2_grid')
               if c in columns] + fields

    # ── 1. faithful cells, seed by seed ──────────────────────────────────
    print('\n' + '-' * 74)
    print('1. FAITHFUL CELLS — must coincide WITHIN each seed')
    print('-' * 74)
    print('   A faithful cell that coincides on one seed and not on another')
    print('   is not chance: it is a bug that the first seed was hiding.')
    any_pair = False
    for a, b, reason in FAITHFUL_PAIRS:
        ok = failed = 0
        culprits = []
        for key, per_seed in cells.items():
            if key[4] != a:
                continue
            key_b = key[:4] + (b,)
            if key_b not in cells:
                continue
            # Equality is only meaningful on the ideal axis.
            if key[3] not in ('none', 'none-counts', ''):
                continue
            for s, row_a in per_seed.items():
                row_b = cells[key_b].get(s)
                if row_b is None:
                    continue
                common = [c for c in fields if c in row_a and c in row_b]
                if not common:
                    continue
                any_pair = True
                if all(row_a[c] == row_b[c] for c in common):
                    ok += 1
                else:
                    failed += 1
                    culprits.append(f'{key[1]}/{key[0]}/seed={s}')
        if ok or failed:
            mark = 'OK' if failed == 0 else '*** CHECK ***'
            print(f'\n   {a} == {b}   {ok}/{ok+failed}   {mark}')
            print(f'      ({reason})')
            for c in culprits[:5]:
                print(f'      differs in: {c}')
    if not any_pair:
        print('\n   (no comparable faithful pairs in these data)')

    # ── 2. how much each number moves ────────────────────────────────────
    print('\n' + '-' * 74)
    print('2. SEED-TO-SEED SPREAD — how much each number moves')
    print('-' * 74)
    spread_rows = []
    for key, per_seed in sorted(cells.items()):
        if len(per_seed) < 2:
            continue
        for met in metrics:
            vals = [_num(f.get(met)) for f in per_seed.values()]
            m, s, n = dispersion(vals)
            if m is None or n < 2:
                continue
            rel = abs(s / m) * 100 if m else float('nan')
            spread_rows.append((rel, key, met, m, s, n))
    if not spread_rows:
        print('\n   (no cell has two or more seeds)')
    else:
        spread_rows.sort(reverse=True)
        print(f'\n   The 12 most unstable (largest relative spread):\n')
        print(f'   {"cell":34s}{"metric":12s}{"mean":>12s}'
              f'{"sigma":>11s}{"%":>7s}')
        for rel, key, met, m, s, n in spread_rows[:12]:
            lab = f'{key[1]}/{key[3]}/{key[4]}'[:33]
            print(f'   {lab:34s}{met:12s}{m:12.5g}{s:11.4g}{rel:7.2f}')
        stable = sum(1 for f in spread_rows if f[0] < 1.0)
        print(f'\n   {stable} of {len(spread_rows)} combinations move '
              f'less than 1% between seeds.')

    # ── 3. what you want to report, measured in sigmas ───────────────────
    print('\n' + '-' * 74)
    print('3. DO THE DIFFERENCES YOU WANT TO REPORT SURVIVE?')
    print('-' * 74)
    print('   Paired t-test over seeds (both methods share every seed).')
    print('   A difference with p >= 0.05 cannot be defended.')
    anything = False
    for a, b, met, reason in COMPARISONS:
        if met not in columns:
            continue
        for key, per_seed in sorted(cells.items()):
            if key[4] != a:
                continue
            key_b = key[:4] + (b,)
            if key_b not in cells:
                continue
            sa = set(per_seed) & set(cells[key_b])
            if len(sa) < 2:
                continue
            va = [_num(per_seed[s].get(met)) for s in sorted(sa)]
            vb = [_num(cells[key_b][s].get(met)) for s in sorted(sa)]
            diff, se, tstat = in_sigmas(va, vb)
            if diff is None:
                continue
            anything = True
            lab = f'{key[1]}/{key[3]}  [{key[0]}]'
            # [E-QPU7] Verdict from the paired t-test p-value.
            if tstat is None:
                verdict = ('identical difference on every seed: systematic'
                           if diff else 'no difference on any seed')
            else:
                pv = p_value_two_sided(tstat, len(sa))
                if pv >= 0.05:
                    verdict = f'p = {pv:.3g}: within seed-to-seed noise, not reportable'
                elif pv >= 0.001:
                    verdict = f'p = {pv:.3g}: marginal, report with its error bar'
                else:
                    verdict = f'p = {pv:.3g}: solid'
            print(f'\n   {a} vs {b}  [{met}]  — {reason}')
            print(f'      {lab}   ({len(sa)} seeds, paired)')
            print(f'      mean difference = {diff:+.6g}   SE = {se:.3g}'
                  + (f'   ->  t = {tstat:+.2f}' if tstat is not None else ''))
            print(f'      {verdict}')
    if not anything:
        print('\n   (no comparable pairs with two or more seeds)')

    print('\n' + '=' * 74)
    return 0


def main(argv=None):
    """Entry point.

    Args:
        argv: master folders, one per seed; None uses `sys.argv`.

    Returns:
        0 if the report was produced, 2 if there was no usable data.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        print(__doc__)
        return 2
    folders = []
    for pattern in args:
        folders += sorted(glob.glob(pattern)) or [pattern]
    folders = [c for c in folders if os.path.isdir(c)]
    if not folders:
        print('None of those paths is a folder.')
        return 2
    return review(folders)


if __name__ == '__main__':
    sys.exit(main())
