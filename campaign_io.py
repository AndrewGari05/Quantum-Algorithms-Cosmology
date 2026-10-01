"""Shared reader for HPC campaign folders (thesis-errata).

Every analysis script used to parse campaign folders on its own, with four
different regular expressions and two different CSV globs. That caused three
confirmed errors (see ERRATA.md):

* QPU-6 / HPC-11c — the glob ``**/resultados_*.csv`` matched both the per-model
  CSV and the per-task cumulative file (legacy name
  ``resultados_TODOS_los_modelos.csv``), so every task was counted twice;
* HPC-8 / QPU-10 — task folders only carry the grid tag (``_nqpp5``/``_nb6``)
  and the noise tag (``_noise-readout``) when the campaign sweeps them, but the
  regexes required both, so whole campaigns were skipped without a warning;
* QPU-8 — rows were paired on (model, grid, method) only, so rows from
  different campaigns, datasets, priors or seeds were silently mixed.

This module is the single place that knows the on-disk layout::

    <campaign>/<family>_<model>[_nqpp<g>|_nb<g>][_noise-<level>]/model_<model>/results_config.csv

It reads both the original CSV schema and the provenance-extended one, and
both file-name generations: the current English names (``results_config.csv``,
cumulative ``results_all_models.csv``, ``results_<model>.csv``) and the legacy
Spanish names that older campaigns on disk still carry
(``resultados_config.csv``, cumulative ``resultados_TODOS_los_modelos.csv``,
``resultados_<model>.csv``). A cumulative file is never read, whatever its name.
"""
from __future__ import annotations

import csv
import glob
import os
import re
import sys
from typing import Dict, Iterable, List, Optional, Tuple

#: Per-model result file written by every task. The per-task cumulative file
#: (``results_all_models.csv``) duplicates these rows and is ignored.
RESULT_CSV = "results_config.csv"
#: Legacy (pre-translation) name of the per-model result file.
LEGACY_RESULT_CSV = "resultados_config.csv"
#: Every accepted name of the per-model result file, current name first.
RESULT_CSV_NAMES = (RESULT_CSV, LEGACY_RESULT_CSV)

#: Per-task cumulative file; it repeats the rows of the per-model files.
CUMULATIVE_CSV = "results_all_models.csv"
#: Legacy (pre-translation) name of the per-task cumulative file.
LEGACY_CUMULATIVE_CSV = "resultados_TODOS_los_modelos.csv"
#: Every name of the cumulative file; files with these names are never read.
CUMULATIVE_CSV_NAMES = (CUMULATIVE_CSV, LEGACY_CUMULATIVE_CSV)

#: File-name prefixes of result CSVs (current, legacy).
RESULT_PREFIXES = ("results_", "resultados_")

TASK_RE = re.compile(
    r"^(?P<fam>samplers|genetic)_(?P<mod>[a-z0-9]+)"
    r"(?:_(?:nqpp|nb)(?P<g>\d+))?"
    r"(?:_noise-(?P<noi>.+))?$")


def parse_task_name(name: str) -> Optional[Dict[str, Optional[str]]]:
    """Split a task folder name into family, model, grid and noise tags.

    Examples:
        >>> parse_task_name('samplers_lcdm_nqpp5_noise-readout')['g']
        '5'
        >>> parse_task_name('samplers_lcdm')['noi'] is None
        True
        >>> parse_task_name('genetic_cpl_noise-fake_brisbane')['noi']
        'fake_brisbane'
        >>> parse_task_name('analysis') is None
        True
    """
    m = TASK_RE.match(name)
    return m.groupdict() if m else None


def is_cumulative_csv(path: str) -> bool:
    """True for the per-task cumulative file, under its current or legacy name.

    Examples:
        >>> is_cumulative_csv('t/results_all_models.csv')
        True
        >>> is_cumulative_csv('t/resultados_TODOS_los_modelos.csv')
        True
        >>> is_cumulative_csv('t/model_lcdm/results_config.csv')
        False
    """
    return os.path.basename(path) in CUMULATIVE_CSV_NAMES


def canonical_csv_name(path: str) -> str:
    """``path`` with a legacy result-file prefix rewritten to the current one.

    Used where the file path is part of a cell identity (seed comparison), so
    a legacy campaign and a new one still pair up cell by cell.

    Examples:
        >>> canonical_csv_name('model_lcdm/resultados_config.csv')
        'model_lcdm/results_config.csv'
        >>> canonical_csv_name('model_lcdm/results_config.csv')
        'model_lcdm/results_config.csv'
    """
    head, base = os.path.split(path)
    if base.startswith("resultados_"):
        base = "results_" + base[len("resultados_"):]
    return os.path.join(head, base) if head else base


def result_csvs(task_dir: str) -> List[str]:
    """Per-model result CSVs of one task (never the cumulative file)."""
    return sorted(p for name in RESULT_CSV_NAMES
                  for p in glob.glob(os.path.join(task_dir, "model_*", name)))


def all_result_csvs(root: str) -> List[str]:
    """Per-model result CSVs anywhere under ``root`` (recursive)."""
    return sorted(p for name in RESULT_CSV_NAMES
                  for p in glob.glob(os.path.join(root, "**", "model_*", name),
                                     recursive=True))


def result_csvs_any_layout(root: str) -> List[str]:
    """Every ``results_*.csv`` / ``resultados_*.csv`` under ``root`` except
    the cumulative copy (current or legacy name).

    Older campaigns also used ``resultados_<model>.csv`` and flat layouts, so
    tools that must read them (triage, seed comparison) glob broadly but drop
    the per-task cumulative file, which is what caused the double counting.
    """
    return sorted(p for prefix in RESULT_PREFIXES
                  for p in glob.glob(os.path.join(root, "**", prefix + "*.csv"),
                                     recursive=True)
                  if not is_cumulative_csv(p))


def moved_aside_csvs(task_dir: str) -> List[str]:
    """Per-model CSVs moved aside after a header change (QV-5), either name.

    They look like ``results_config.old-schema-<stamp>.csv`` (or the legacy
    ``resultados_config.old-schema-<stamp>.csv``) and are never read.
    """
    return sorted(p for name in RESULT_CSV_NAMES
                  for p in glob.glob(os.path.join(
                      task_dir, "model_*",
                      os.path.splitext(name)[0] + ".old-schema-*.csv")))


def _int_or_none(x) -> Optional[int]:
    try:
        return int(float(str(x).strip()))
    except (TypeError, ValueError):
        return None


def read_campaign(root: str, warn: bool = True) -> List[dict]:
    """All result rows of one campaign folder, with provenance resolved.

    Adds these keys to every CSV row:

    ``_camp``  campaign folder name; ``_fam`` 'samplers' | 'genetic';
    ``_mod``   model; ``_g`` grid size (nqpp or n_bits, int or None);
    ``_noi``   noise level; ``_path`` source CSV.

    The grid falls back to the numeric ``nqpp`` column of the task's rows and
    the noise level to the ``noise`` column (then to 'none') when the folder
    name carries no tag. Folders that hold result CSVs but do not look like
    tasks are reported on stderr instead of being dropped silently.
    """
    camp = os.path.basename(os.path.normpath(root))
    rows: List[dict] = []
    skipped = []
    for name in sorted(os.listdir(root)):
        task_dir = os.path.join(root, name)
        if not os.path.isdir(task_dir):
            continue
        tags = parse_task_name(name)
        paths = result_csvs(task_dir)
        moved = moved_aside_csvs(task_dir)
        if warn and moved:
            print(f"  [campaign_io] {camp}/{name}: {len(moved)} moved-aside CSV(s) with an "
                  f"older header are not read: {[os.path.basename(m) for m in moved]}",
                  file=sys.stderr)
        if tags is None:
            if paths:
                skipped.append(name)
            continue
        task_rows = []
        for p in paths:
            with open(p, newline="") as fh:
                for r in csv.DictReader(fh):
                    r["_path"] = p
                    task_rows.append(r)
        g = _int_or_none(tags["g"])
        if g is None:
            grids = [v for v in (_int_or_none(r.get("nqpp")) for r in task_rows)
                     if v is not None]
            g = max(grids) if grids else None
        for r in task_rows:
            noi = tags["noi"] or (r.get("noise") or "").strip() or "none"
            r.update({"_camp": camp, "_fam": tags["fam"], "_mod": tags["mod"],
                      "_g": g, "_noi": noi})
            rows.append(r)
    if warn and skipped:
        print(f"  [campaign_io] {camp}: {len(skipped)} folder(s) with results "
              f"but no task-name pattern were skipped: {skipped[:5]}",
              file=sys.stderr)
    return rows


def read_campaigns(roots: Iterable[str], warn: bool = True) -> List[dict]:
    """``read_campaign`` over several campaign folders."""
    out: List[dict] = []
    for root in roots:
        out += read_campaign(root, warn=warn)
    return out


def cell_key(row: dict) -> Tuple:
    """Identity of the experimental cell a row belongs to (method excluded).

    Two rows may be compared only if they share this key: same campaign,
    dataset, prior, seed, family, model, grid and noise level.
    """
    return (row.get("_camp"), (row.get("dataset") or "").strip(),
            (row.get("prior") or "").strip(), (row.get("seed") or "").strip(),
            row.get("_fam"), row.get("_mod"), row.get("_g"), row.get("_noi"))


def cell_key_without_noise(row: dict) -> Tuple:
    """``cell_key`` with the noise level removed (for noisy-vs-ideal pairs)."""
    return cell_key(row)[:-1]
