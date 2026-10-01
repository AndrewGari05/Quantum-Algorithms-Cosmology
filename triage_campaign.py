#!/usr/bin/env python3
"""
triage_campaign.py — Quick diagnosis of a results folder.

Answers at a glance the three questions one asks when coming back to a long
campaign:

    1. Which tasks died, which are still running and which finished well?
    2. Are the fits good?
    3. Which columns of this CSV are citable and which are not?

The third one matters most and is the one you cannot see by looking at
files: a campaign run with code older than 2026-09-04 carries known bugs
that invalidate specific columns, and this script detects it by the ABSENCE
of the provenance columns (`noise`, `budget_mode`, `chi2_grid`).

Usage:

    python triage_campaign.py results/hpc_20260903_161310
    python triage_campaign.py results/hpc_*            # several at once

It needs neither numpy nor qiskit: only the standard library, so it can run
anywhere, even on the HPC node without the environment activated.
"""

from __future__ import annotations

import csv
import glob

import campaign_io
import os
import re
import sys
from collections import defaultdict

#: Last line a samplers task writes before starting the QVMC.
#: If the log ends RIGHT THERE and there is no CSV, it is the OOM killer's signature.
OOM_SIGNATURE = 'Adaptive QVMC grid window'

#: Columns that only exist since 2026-09-04. Their absence marks a
#: campaign run with code that carries known bugs.
NEW_COLUMNS = ('noise', 'proposal_route', 'seed', 'budget_mode',
               'circuits_train', 'chi2_grid')


def _read_csv(path):
    """Rows of a results CSV, or an empty list if it cannot be read."""
    try:
        with open(path, newline='') as fh:
            return list(csv.DictReader(fh))
    except Exception:
        return []


def _last_line(path):
    """Last non-empty line of a text file."""
    try:
        with open(path, errors='ignore') as fh:
            lines = [l.rstrip('\n') for l in fh if l.strip()]
        return lines[-1] if lines else ''
    except Exception:
        return ''


#: Number of free parameters per model. An nqpp of N qubits per parameter
#: gives N * MODEL_DIM qubits in total, which is what decides whether the task
#: fits in RAM: the dense state weighs 16 * 4**n_qubits bytes.
MODEL_DIM = {'lcdm': 2, 'pede': 2, 'wcdm': 3, 'gede': 3, 'cpl': 4}


def _qubit_suffix(name):
    """Annotate the total number of qubits of a task from its name.

    Args:
        name: folder name, e.g. `samplers_cpl_nqpp5_CC+BAO`.

    Returns:
        The string `'  (20 qubits)'`, or `''` if the name does not allow it.

    Examples:
        >>> _qubit_suffix('samplers_cpl_nqpp5_CC+BAO')
        '  (20 qubits)'
        >>> _qubit_suffix('genetic_lcdm_nb6_noise-none')
        ''
    """
    q = re.search(r'nqpp(\d+)', name)
    m = re.search(r'samplers_(\w+?)_', name)
    if q and m and m.group(1) in MODEL_DIM:
        return f"  ({int(q.group(1)) * MODEL_DIM[m.group(1)]} qubits)"
    return ''


def param_columns(cols):
    """Cosmological parameter columns present in the CSV.

    The names depend on the model (`Om_mean`, `H0_std`, `w0_mean`...), so
    they cannot be fixed in advance: they are discovered by suffix.

    Args:
        cols: set of column names of the CSV.

    Returns:
        Sorted list of the columns ending in `_mean` or `_std`.

    Examples:
        >>> param_columns({'Method', 'Om_mean', 'Om_std', 'chi2'})
        ['Om_mean', 'Om_std']
        >>> param_columns({'Method', 'chi2'})
        []
    """
    return sorted(c for c in cols
                  if c.endswith('_mean') or c.endswith('_std'))


def _rows_equal(a, b, fields):
    """Do two rows coincide on all the given fields?

    Compares only the fields that EXIST in both rows and requires at least
    one. Without that condition, comparing two rows by non-existent column
    names gives `None == None` for everything and the faithful cell comes
    out "OK" without having checked anything — which is exactly the opposite
    of what this section is for.

    Args:
        a, b: the two rows to compare.
        fields: column names to check.

    Returns:
        True if they coincide on all common fields; False if they differ or
        if they share none.

    Examples:
        >>> _rows_equal({'x': '1'}, {'x': '1'}, ['x'])
        True
        >>> _rows_equal({'x': '1'}, {'x': '2'}, ['x'])
        False
        >>> _rows_equal({'x': '1'}, {'x': '1'}, ['does_not_exist'])
        False
    """
    common = [c for c in fields if c in a and c in b]
    if not common:
        return False
    return all(a[c] == b[c] for c in common)


def model_from_path(path):
    """Infer the cosmological model from the path of a CSV.

    The CSV has no `model` column: the model is in the folder
    (`model_lcdm/`) or in the file name (`results_lcdm.csv`, legacy
    `resultados_lcdm.csv`).

    Args:
        path: path of the CSV.

    Returns:
        Model key, or `'?'` if the path does not say.

    Examples:
        >>> model_from_path('run/model_wcdm/results_config.csv')
        'wcdm'
        >>> model_from_path('run/results_cpl.csv')
        'cpl'
        >>> model_from_path('run/resultados_cpl.csv')
        'cpl'
        >>> model_from_path('run/results_config.csv')
        '?'
    """
    parts = path.replace('\\', '/').split('/')
    for p in parts:
        m = re.match(r'model_(\w+)$', p)
        if m and m.group(1) in MODEL_DIM:
            return m.group(1)
    m = re.search(r'(?:results|resultados)_(\w+)\.csv$', parts[-1])
    if m and m.group(1) in MODEL_DIM:
        return m.group(1)
    for p in parts:
        m = re.search(r'_(' + '|'.join(MODEL_DIM) + r')_', p)
        if m:
            return m.group(1)
    return '?'


def noise_level_of(row, path):
    """Noise level of a row: the column if present, otherwise the path.

    Campaigns before 2026-09-04 have no `noise` column — it is one of those
    introduced by [B-PROV] —, but the runner DOES put the level in the
    folder name (`samplers_lcdm_nqpp5_noise-full`). Without this fallback to
    the path, every old campaign looks ideal and the noisy cells are flagged
    as errors when they are doing exactly what they should.

    Args:
        row: the CSV row.
        path: path of the file it came from.

    Returns:
        The level as a string; `'none'` if it cannot be determined.

    Examples:
        >>> noise_level_of({'noise': 'readout'}, 'x/samplers_lcdm_noise-full/r.csv')
        'readout'
        >>> noise_level_of({}, 'x/samplers_lcdm_nqpp5_noise-full/r.csv')
        'full'
        >>> noise_level_of({'noise': ''}, 'x/genetic_lcdm_nb6/r.csv')
        'none'
    """
    if row.get('noise'):
        return row['noise']
    m = re.search(r'noise-([A-Za-z0-9_.-]+)', path.replace('\\', '/'))
    return m.group(1) if m else 'none'


def task_status(folder):
    """Classify a task folder.

    Args:
        folder: path of a task subfolder.

    Returns:
        `(status, detail)`, with status in {'ok', 'oom', 'failed',
        'running', 'no_log'}.
    """
    has_csv = any(glob.glob(os.path.join(folder, prefix + '*.csv'))
                  for prefix in campaign_io.RESULT_PREFIXES)
    # [E-QPU11] Newest log by modification time, not alphabetical order
    # (`sweep_all_*.log` sorts after `stdout.log`).
    logs = [l for l in glob.glob(os.path.join(folder, '*.log'))
            if os.path.getsize(l) > 0]
    logs.sort(key=os.path.getmtime)
    if logs and _log_has_failure(logs):
        return 'failed', _log_has_failure(logs)
    if has_csv:
        return 'ok', ''
    if not logs:
        return 'no_log', 'the task never wrote anything'
    last = _last_line(logs[-1])
    if OOM_SIGNATURE in last:
        return 'oom', 'the log stops when the QVMC starts (SIGKILL from the OOM killer)'
    return 'running', last[-70:]


def _log_has_failure(logs):
    """First failure line found in the task logs, or '' if none.

    [E-QPU11] A task that crashed (traceback) or had a model fail inside the
    sweep used to be reported as still running.
    """
    for path in logs:
        try:
            with open(path, encoding='utf-8', errors='replace') as fh:
                for line in fh:
                    if ': FAILED — ' in line or line.startswith('Traceback (most recent call last)'):
                        return line.strip()[-90:]
        except OSError:
            continue
    return ''


def review(master_dir):
    """Print the diagnosis of a campaign folder."""
    print(f"\n{'='*74}\n{master_dir}\n{'='*74}")
    subs = sorted(d for d in glob.glob(os.path.join(master_dir, '*'))
                  if os.path.isdir(d))

    # ── 1. status of each task ───────────────────────────────────────────
    # A campaign may have no subfolders (partial download, or a single-task
    # run that writes into the root). In that case there is nothing to
    # classify, but sections 2-4 are still useful: the CSVs are searched
    # recursively from the root.
    if not subs:
        print("\n1. TASKS: there are no task subfolders in this folder")
        print("   (partial download, or a single-task run)")
    else:
        by_status = defaultdict(list)
        for d in subs:
            st, det = task_status(d)
            by_status[st].append((os.path.basename(d), det))

        print(f"\n1. TASKS  ({len(subs)} folders)")
        for st, label in (('ok', 'finished'),
                          ('oom', 'DEAD (OOMKill)'),
                          ('failed', 'FAILED (traceback / model failure)'),
                          ('running', 'still running'),
                          ('no_log', 'no log')):
            items = by_status.get(st, [])
            if not items:
                continue
            print(f"   {label}: {len(items)}")
            if st in ('oom', 'no_log', 'failed'):
                for name, det in items:
                    print(f"      - {name}{_qubit_suffix(name)}")
                    print(f"        {det}")
            elif st == 'running':
                for name, det in items[:6]:
                    print(f"      - {name}: ...{det}")
                if len(items) > 6:
                    print(f"      ... and {len(items)-6} more")

    # ── 2. fit ───────────────────────────────────────────────────────────
    # The samplers and genetic CSVs do NOT share a schema: `noise` and
    # `budget_mode` only exist in the former, `chi2_grid` only in the
    # latter. That is why section 3 looks at the UNION of the columns of all
    # files, not at those of an arbitrary row: looking at a single one would
    # always say "missing columns" even if the campaign used the corrected code.
    rows, cols = [], set()
    # [E-QPU6] Only the per-model CSV; the per-task cumulative file repeats
    # the same rows and used to double every count.
    for p in campaign_io.result_csvs_any_layout(master_dir):
        new_rows = _read_csv(p)
        # The model is not a column: it is noted from the path so rows can
        # be grouped by model in sections 2 and 4.
        mod = model_from_path(p)
        for r in new_rows:
            # The genetic sweep CSV DOES have a `model` column; the samplers
            # one does not. The file's value is preferred, falling back to the path.
            r['_model'] = r.get('model') or mod
            r['_file'] = p
            r['_noise'] = noise_level_of(r, p)
        rows += new_rows
        for r in new_rows[:1]:
            cols |= set(r.keys())
    if not rows:
        print("\n2. FIT: there is no CSV yet")
        return

    print("\n2. FIT  (chi2 of the best fit, identical on every rung)")
    seen = {}
    for r in rows:
        mod = r.get('_model') or '?'
        if mod not in seen and r.get('chi2'):
            seen[mod] = (r.get('chi2'), r.get('n_data'), r.get('chi2_red'),
                         r.get('AIC'), r.get('BIC'))
    print(f"   {'model':8s}{'chi2':>12s}{'n':>7s}{'chi2_red':>10s}"
          f"{'AIC':>11s}{'BIC':>11s}")
    for mod, v in sorted(seen.items(), key=lambda x: float(x[1][3] or 0)):
        print(f"   {mod:8s}{v[0]:>12s}{v[1]:>7s}{v[2]:>10s}{v[3]:>11s}"
              f"{v[4]:>11s}")
    reds = [float(v[2]) for v in seen.values() if v[2]]
    if reds:
        lo, hi = min(reds), max(reds)
        if 0.8 <= lo and hi <= 1.2:
            print(f"   -> chi2_red between {lo:.3f} and {hi:.3f}: good fit.")
        elif hi < 0.8:
            print(f"   -> chi2_red {lo:.3f}-{hi:.3f}, BELOW 1: error bars "
                  f"probably inflated (typical of CC alone).")
        else:
            print(f"   -> chi2_red {lo:.3f}-{hi:.3f}: review the fit.")

    # ── 3. what can be cited ─────────────────────────────────────────────
    missing = [c for c in NEW_COLUMNS if c not in cols]
    print("\n3. WHAT IS CITABLE IN THIS CAMPAIGN")
    if not missing:
        print("   It has the provenance columns: campaign run with the")
        print("   corrected code. Everything citable, with the usual reading")
        print("   caveats (do not compare against `none` without")
        print("   `none-counts`; the ESS is not a quality metric on the noise")
        print("   axis).")
    else:
        print(f"   Missing columns: {', '.join(missing)}")
        print("   -> Campaign run with code from BEFORE 2026-09-04.")
        print()
        print("   NOT citable:")
        print("     * KL and ESS of the QVMC  — [B-BUDGET] the quantum branch got")
        print("       between 57x and 225x more circuit evaluations than the")
        print("       classical one with the same --qvmc-iter, and [B-ESSCOMP] the")
        print("       ESS was compressed.")
        print("     * genetic chi2/AIC/BIC as a comparison between rungs —")
        print("       [B-REFINE] the refiner takes all four to the same")
        print("       minimum; without the chi2_grid column there is nothing")
        print("       to tell them apart.")
        print()
        print("   Citable:")
        print("     * All of the QMCMC (means, sigmas, acceptance, R-hat). None")
        print("       of those bugs touch it.")
        print("     * chi2/AIC/BIC as the goodness of fit of each MODEL")
        print("       (which is what they are for).")

    # ── 4. faithful cells ────────────────────────────────────────────────
    # The grouping key deliberately does NOT include nqpp. The classical
    # rungs ('Classical VI', 'CGA') write nqpp='—' while their quantum
    # partners write the number, so grouping by nqpp split every pair and the
    # CGA == QGA (q=0%) pair was never checked. The source file is used
    # instead: within a task CSV the nqpp is unique.
    groups = defaultdict(dict)
    for r in rows:
        key = (r.get('_file'), r.get('_model'), r.get('dataset'),
               r.get('_noise'))
        groups[key][r.get('Method')] = r

    fields = param_columns(cols)
    if not fields:
        print("\n4. FAITHFUL CELLS: the CSV has no *_mean / *_std column,")
        print("   so NOTHING can be verified. Inspect the CSV.")
        return

    # Only REALLY faithful pairs. 'Classical VI' vs 'QVMC 33%' is NOT one and
    # was wrongly listed here: the 33% rung changes the sampling (shots
    # instead of amplitudes), so it differs in the fourth digit by
    # construction. Requiring equality would have produced a false failure
    # in every cell.
    # The third field says whether the pair MUST separate when noise is on.
    # That is only true when the component that distinguishes them is a
    # quantum circuit that noise touches:
    #
    #   QMCMC 50 -> 100 : adds the quantum acceptance, a real circuit.
    #                     Noise degrades it. It MUST separate.
    #   QVMC  67 -> 100 : adds the normalization, which runs the circuit
    #                     but DISCARDS its result and uses the exact sum
    #                     (see quantum_amplitude_normalization). It is immune
    #                     to noise by construction: staying identical is
    #                     correct, not a failure.
    #   CGA  -> QGA 0%  : quantumness 0%, there is no quantum component
    #                     noise could touch.
    #
    # Without this distinction the triage reported 'noise did not reach' in
    # 70 of 92 healthy cells.
    pairs = [('QMCMC 50%', 'QMCMC 100%', True),
             ('QVMC 67%', 'QVMC 100%', False),
             ('CGA', 'QGA (q=0%)', False)]
    # Faithful equality is ONLY required on the ideal axis. With noise, the
    # two rungs have to separate: that is the result of the NISQ axis, not
    # an error. Mixing both axes in a single counter gave '40/62 CHECK' on a
    # perfectly healthy campaign — and an alarm that always goes off is an
    # alarm nobody reads any more. The converse is informative too: a noisy
    # rung that matches the ideal one BIT FOR BIT means the noise was not
    # applied, and that IS an error.
    summary = defaultdict(lambda: [0, 0])       # ideal: how many coincide
    noisy = defaultdict(lambda: [0, 0])         # noise: how many separate
    immune = defaultdict(lambda: [0, 0])        # noise: how many stay equal
    for (_file, _mod, _ds, noise), g in groups.items():
        ideal = (noise or 'none') in ('none', 'none-counts', '')
        for a, b, sensitive in pairs:
            if a not in g or b not in g:
                continue
            equal = _rows_equal(g[a], g[b], fields)
            if ideal:
                summary[(a, b)][1] += 1
                summary[(a, b)][0] += int(equal)
            elif sensitive:
                noisy[(a, b)][1] += 1
                noisy[(a, b)][0] += int(not equal)
            else:
                immune[(a, b)][1] += 1
                immune[(a, b)][0] += int(equal)

    if summary or noisy or immune:
        print("\n4. FAITHFUL CELLS")
        print(f"   comparing: {', '.join(fields)}")
    if summary:
        print("\n   Without noise — MUST coincide exactly:")
        for (a, b), (ok, tot) in sorted(summary.items()):
            mark = 'OK' if ok == tot else '*** CHECK ***'
            print(f"     {a:14s} == {b:14s}  {ok}/{tot}  {mark}")
    if noisy:
        print("\n   With noise — MUST separate (differing is the result):")
        for (a, b), (sep, tot) in sorted(noisy.items()):
            mark = 'OK' if sep == tot else '*** CHECK: noise did not reach ***'
            print(f"     {a:14s} != {b:14s}  {sep}/{tot}  {mark}")
    if immune:
        print("\n   With noise, pairs IMMUNE by construction "
              "(must stay equal):")
        for (a, b), (ok, tot) in sorted(immune.items()):
            mark = 'OK' if ok == tot else '*** CHECK ***'
            print(f"     {a:14s} == {b:14s}  {ok}/{tot}  {mark}")


def main(argv=None):
    """Entry point.

    Args:
        argv: campaign folder paths; None uses `sys.argv`.

    Returns:
        Exit code (always 0, unless no path is given).
    """
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        print(__doc__)
        return 2
    for pattern in args:
        for d in sorted(glob.glob(pattern)) or [pattern]:
            if os.path.isdir(d):
                review(d)
            else:
                print(f"not a folder: {d}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
