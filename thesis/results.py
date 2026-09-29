"""Tidy results table: one row per (task, rung), explicit key columns.

Every row is self-describing: it carries the campaign, model, dataset,
prior, seed, grid, noise level and its parameters, the code version and the
convergence diagnostics, so no metadata is ever parsed from folder names.
"""
from __future__ import annotations

import csv
import os
import subprocess
import time
from collections.abc import Iterable

PARAMS = ("Om", "H0", "w", "w0", "wa", "Delta")

KEY_COLUMNS = ["campaign", "task", "family", "rung", "components", "model", "dataset", "prior",
               "seed", "grid", "noise", "noise_params", "route", "backend"]
FIELDS = (KEY_COLUMNS
          + ["n_data", "n_steps", "n_chains", "burn", "max_iter", "shots", "pop_size",
             "n_generations"]
          + [f"mean_{p}" for p in PARAMS] + [f"std_{p}" for p in PARAMS]
          + ["corr_01", "rhat_max", "ess_min", "converged", "acceptance", "detailed_balance",
             "kl", "grid_mass_outside", "circuit_evaluations", "chi2_estimate", "chi2_map",
             "chi2_red", "AIC", "BIC", "chi2_grid_floor", "time_s", "code_version",
             "qablate_version", "timestamp"])


def code_version() -> str:
    """``git describe --tags --always --dirty`` of this checkout, or 'unknown'."""
    try:
        return subprocess.run(["git", "describe", "--tags", "--always", "--dirty"],
                              cwd=os.path.dirname(os.path.abspath(__file__)),
                              capture_output=True, text=True, timeout=10).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _fmt(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return repr(v)
    return str(v)


def append(path: str, rows: Iterable[dict[str, object]]) -> None:
    """Append rows to ``path``; refuses to write under a different header."""
    import qablate
    rows = list(rows)
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    version = code_version()
    new = not os.path.exists(path) or os.path.getsize(path) == 0
    if not new:
        with open(path, newline="") as fh:
            header = next(csv.reader(fh), [])
        if header != FIELDS:
            raise ValueError(f"{path} has a different header; write to a new file")
    unknown = {k for r in rows for k in r} - set(FIELDS)
    if unknown:
        raise KeyError(f"unknown result fields {sorted(unknown)}")
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for r in rows:
            full = {"code_version": version, "qablate_version": qablate.__version__,
                    "timestamp": stamp, **r}
            w.writerow({k: _fmt(full.get(k)) for k in FIELDS})


def read(paths: Iterable[str]) -> list[dict[str, str]]:
    """Rows of several results files."""
    out: list[dict[str, str]] = []
    for p in paths:
        with open(p, newline="") as fh:
            out += list(csv.DictReader(fh))
    return out


def find(root: str) -> list[str]:
    """Every ``results.csv`` under ``root``."""
    hits = []
    for d, _, files in os.walk(root):
        if "results.csv" in files:
            hits.append(os.path.join(d, "results.csv"))
    return sorted(hits)
