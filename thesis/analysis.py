"""Tables and figures from ``results.csv`` files.

Rows are only ever compared within the same experimental cell: campaign,
dataset, prior, seed, model, grid and noise level (all explicit columns).
Differences are reported with their Monte Carlo uncertainty where it exists.
Figures are written as PNG and PDF.
"""
from __future__ import annotations

import math
import os
from collections import defaultdict

from . import results

PARAMS = results.PARAMS
KASS_RAFTERY = [(2.0, "not worth more than a bare mention"), (6.0, "positive"),
                (10.0, "strong"), (math.inf, "very strong")]


def _f(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _cell(r, with_noise=True):
    k = (r["campaign"], r["dataset"], r["prior"], r["seed"], r["family"], r["model"], r["grid"])
    return k + ((r["noise"],) if with_noise else ())


def mcmc_fidelity(rows: list[dict]) -> list[dict]:
    """Shift of each circuit rung from the classical MCMC, in units of sigma.

    ``mc_se`` is the Monte Carlo standard error of that shift,
    ``sqrt(1/ESS_q + 1/ESS_c)``; |shift| > 3 mc_se is flagged.
    """
    cells = defaultdict(dict)
    for r in rows:
        if r["family"] == "mcmc":
            cells[_cell(r)][r["rung"]] = r
    out = []
    for key, v in sorted(cells.items()):
        ref = v.get("mcmc")
        if not ref:
            continue
        for rung, q in v.items():
            if rung == "mcmc":
                continue
            ess = [_f(ref["ess_min"]), _f(q["ess_min"])]
            se = math.sqrt(sum(1 / e for e in ess)) if all(ess) else None
            for p in PARAMS:
                m0, m1, s0, s1 = (_f(ref[f"mean_{p}"]), _f(q[f"mean_{p}"]),
                                  _f(ref[f"std_{p}"]), _f(q[f"std_{p}"]))
                if None in (m0, m1, s0, s1):
                    continue
                shift = (m1 - m0) / s0
                out.append({"model": key[5], "dataset": key[1], "noise": key[7], "rung": rung,
                            "param": p, "shift_sigma": shift, "std_ratio": s1 / s0,
                            "mc_se": se, "flag": bool(se and abs(shift) > 3 * se),
                            "both_converged": ref["converged"] == q["converged"] == "True"})
    return out


def vi_widths(rows: list[dict]) -> list[dict]:
    """sigma(VI rung) / sigma(MCMC) per parameter, with the trained correlation."""
    mcmc = {(r["campaign"], r["dataset"], r["prior"], r["seed"], r["model"], r["noise"]): r
            for r in rows if r["family"] == "mcmc" and r["rung"] == "mcmc"}
    out = []
    for r in rows:
        if r["family"] != "vi":
            continue
        ref = mcmc.get((r["campaign"], r["dataset"], r["prior"], r["seed"], r["model"], "none"))
        if not ref:
            continue
        for p in PARAMS:
            a, b = _f(r[f"std_{p}"]), _f(ref[f"std_{p}"])
            if a and b:
                out.append({"model": r["model"], "noise": r["noise"], "grid": int(r["grid"]),
                            "rung": r["rung"], "param": p, "std_ratio": a / b,
                            "corr_vi": _f(r["corr_01"]), "corr_mcmc": _f(ref["corr_01"]),
                            "kl": _f(r["kl"])})
    return out


def genetic_gaps(rows: list[dict]) -> list[dict]:
    """chi2 of every genetic rung relative to the grid-genome classical rung and floor."""
    cells = defaultdict(dict)
    for r in rows:
        if r["family"] == "genetic":
            cells[_cell(r)[:-2] + (r["noise"],)][(r["rung"], r["grid"])] = r
    out = []
    for key, v in sorted(cells.items()):
        for (rung, g), r in v.items():
            base = v.get(("ga-grid", g))
            out.append({"model": key[5], "noise": key[6], "grid": g, "rung": rung,
                        "chi2_best": _f(r["chi2_estimate"]),
                        "minus_classical_grid": (_f(r["chi2_estimate"]) - _f(base["chi2_estimate"])
                                                 if base else None),
                        "minus_floor": (_f(r["chi2_estimate"]) - _f(r["chi2_grid_floor"])
                                        if _f(r["chi2_grid_floor"]) is not None else None)})
    return out


def model_selection(rows: list[dict]) -> dict[tuple, list[dict]]:
    """Best fit per model, per (dataset, prior); Delta AIC / BIC with Kass & Raftery labels."""
    best = defaultdict(dict)
    for r in rows:
        c = _f(r["chi2_map"])
        if c is None or r["noise"] != "none":
            continue
        g = (r["dataset"], r["prior"])
        if r["model"] not in best[g] or c < _f(best[g][r["model"]]["chi2_map"]):
            best[g][r["model"]] = r
    out = {}
    for g, v in best.items():
        amin = min(_f(r["AIC"]) for r in v.values())
        bmin = min(_f(r["BIC"]) for r in v.values())
        tab = []
        for m, r in sorted(v.items(), key=lambda kv: _f(kv[1]["BIC"])):
            db = _f(r["BIC"]) - bmin
            tab.append({"model": m, "chi2": _f(r["chi2_map"]), "chi2_red": _f(r["chi2_red"]),
                        "dAIC": _f(r["AIC"]) - amin, "dBIC": db,
                        "evidence_against": next(label for t, label in KASS_RAFTERY if db < t)})
        out[g] = tab
    return out


def noise_shifts(rows: list[dict]) -> list[dict]:
    """Shift of every noisy row from the same cell run without noise (MCMC and VI)."""
    ideal = {_cell(r, False) + (r["rung"],): r for r in rows if r["noise"] == "none"}
    out = []
    for r in rows:
        if r["noise"] == "none" or r["family"] == "genetic":
            continue
        ref = ideal.get(_cell(r, False) + (r["rung"],))
        if not ref:
            continue
        worst = max((abs(_f(r[f"mean_{p}"]) - _f(ref[f"mean_{p}"])) / _f(ref[f"std_{p}"])
                     for p in PARAMS if _f(r[f"mean_{p}"]) is not None and _f(ref[f"std_{p}"])),
                    default=None)
        out.append({"model": r["model"], "family": r["family"], "rung": r["rung"],
                    "grid": r["grid"], "noise": r["noise"], "max_shift_sigma": worst})
    return out


def _write(path, rows):
    import csv
    if not rows:
        return
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def _save(fig, path):
    fig.savefig(path + ".png", dpi=150, bbox_inches="tight")
    fig.savefig(path + ".pdf", bbox_inches="tight")


def figures(vi: list[dict], noise: list[dict], out: str) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    made = []
    series = defaultdict(list)
    for r in vi:
        if r["param"] == "Om" and r["noise"] == "none":
            series[(r["model"], r["rung"])].append((r["grid"], r["std_ratio"]))
    if series:
        fig, ax = plt.subplots(figsize=(6, 4))
        for (m, rung), pts in sorted(series.items()):
            pts.sort()
            ax.plot(*zip(*pts, strict=True), marker="o", label=f"{m} {rung}")
        ax.axhline(1.0, color="k", lw=0.8)
        ax.set_xlabel("qubits per parameter")
        ax.set_ylabel(r"$\sigma_{VI}(\Omega_m) / \sigma_{MCMC}(\Omega_m)$")
        ax.legend(fontsize=7)
        _save(fig, os.path.join(out, "vi_width_vs_grid"))
        made.append("vi_width_vs_grid")
        plt.close(fig)
    if noise:
        fig, ax = plt.subplots(figsize=(6, 4))
        levels = sorted({r["noise"] for r in noise})
        for i, lev in enumerate(levels):
            v = [r["max_shift_sigma"] for r in noise if r["noise"] == lev and r["max_shift_sigma"] is not None]
            ax.scatter([i] * len(v), v, alpha=0.6)
        ax.set_xticks(range(len(levels)), levels)
        ax.set_ylabel(r"max $|\Delta\theta| / \sigma_{ideal}$")
        _save(fig, os.path.join(out, "noise_shifts"))
        made.append("noise_shifts")
        plt.close(fig)
    return made


def main(roots: list[str], out: str) -> int:
    paths = [p for r in roots for p in results.find(r)]
    rows = results.read(paths)
    if not rows:
        print("  no results.csv found")
        return 1
    os.makedirs(out, exist_ok=True)
    fid, vi, gen, noi = mcmc_fidelity(rows), vi_widths(rows), genetic_gaps(rows), noise_shifts(rows)
    _write(os.path.join(out, "mcmc_fidelity.csv"), fid)
    _write(os.path.join(out, "vi_widths.csv"), vi)
    _write(os.path.join(out, "genetic_gaps.csv"), gen)
    _write(os.path.join(out, "noise_shifts.csv"), noi)
    for (ds, pr), tab in model_selection(rows).items():
        print(f"\n  model selection | {ds} | prior {pr}")
        for t in tab:
            print(f"    {t['model']:6s} chi2={t['chi2']:.4f}  dAIC={t['dAIC']:6.2f}  "
                  f"dBIC={t['dBIC']:6.2f}  {t['evidence_against']}")
    flagged = [f for f in fid if f["flag"]]
    print(f"\n  MCMC fidelity: {len(flagged)}/{len(fid)} parameter shifts beyond 3 Monte Carlo SE")
    made = figures(vi, noi, out)
    print(f"  {len(paths)} files, {len(rows)} rows -> {out}/ ({len(made)} figures)")
    return 0
