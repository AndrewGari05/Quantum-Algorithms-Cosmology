"""Errata tools: re-state campaign results without re-running the samplers.

Subcommands
-----------
refit CAMPAIGN [CAMPAIGN ...]
    [E-CO3/4] Recompute chi2, chi2_red, AIC and BIC with the corrected
    best-fit search, starting from the means stored in each row. No MCMC is
    re-run. Prints old vs new per (dataset, model) and writes
    ``refit_<campaign>.csv`` with one line per row.

grid-floor --model M --dataset D --n-bits 4 5 6
    [E-GA1] Lowest chi2 reachable on the QGA grid (cell centres of the
    2^n_bits x ... x 2^n_bits grid over the prior box). QGA 67 % and 100 %
    return cell centres, so their ``chi2_grid`` cannot go below this floor;
    Delta chi2 against the continuous CGA includes the floor.

mc-error CAMPAIGN [CAMPAIGN ...]
    [E-QM3] For every QMCMC row paired with its Classical MCMC row, report
    whether the sigma ratio differs from 1 by more than two combined Monte
    Carlo standard errors, SE(sigma)/sigma ~ 1/sqrt(2 ESS).
"""
from __future__ import annotations

import argparse
import csv
import itertools
import math
import os
import sys
from collections import defaultdict

import numpy as np

import campaign_io


def _num(x):
    try:
        v = float(str(x).strip())
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _means(row, model):
    vals = [_num(row.get(f"{p}_mean")) for p in model.param_names]
    return None if any(v is None for v in vals) else np.array(vals)


# --------------------------------------------------------------------------- #
def cmd_refit(roots, quiet=False):
    import contextlib
    import io
    import cosmo_core as cc
    posts, out = {}, []
    for root in roots:
        rows = campaign_io.read_campaign(root)
        for r in rows:
            model = cc.MODELS.get(r["_mod"])
            ds = (r.get("dataset") or "").strip()
            prior = (r.get("prior") or "flat").strip() or "flat"
            theta = _means(r, model) if model else None
            if theta is None or not ds:
                continue
            key = (r["_mod"], ds, prior)
            if key not in posts:
                with contextlib.redirect_stdout(io.StringIO()):
                    posts[key] = cc.Posterior(model, dataset=ds, prior_type=prior)
            st = cc.fit_statistics(posts[key], theta)
            out.append({"campaign": r["_camp"], "task_dir": os.path.basename(
                os.path.dirname(os.path.dirname(r["_path"]))),
                "Method": r["Method"], "model": r["_mod"], "dataset": ds,
                "prior": prior, "noise": r["_noi"], "grid": r["_g"],
                "chi2_old": _num(r.get("chi2")), "chi2_new": st["chi2"],
                "AIC_old": _num(r.get("AIC")), "AIC_new": st["AIC"],
                "BIC_old": _num(r.get("BIC")), "BIC_new": st["BIC"],
                "chi2_red_new": st["chi2_red"]})
        path = f"refit_{os.path.basename(os.path.normpath(root))}.csv"
        mine = [o for o in out if o["campaign"] == os.path.basename(os.path.normpath(root))]
        if mine:
            with open(path, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(mine[0]))
                w.writeheader()
                w.writerows(mine)
            if not quiet:
                print(f"  wrote {path} ({len(mine)} rows)")
    if not quiet:
        best = defaultdict(lambda: [np.inf, np.inf])
        for o in out:
            b = best[(o["dataset"], o["model"])]
            if o["chi2_old"] is not None:
                b[0] = min(b[0], o["chi2_old"])
            b[1] = min(b[1], o["chi2_new"])
        print(f"\n  {'dataset':18s} {'model':6s} {'best chi2 old':>14s} {'best chi2 new':>14s} {'delta':>9s}")
        for (ds, m), (a, b) in sorted(best.items()):
            print(f"  {ds:18s} {m:6s} {a:14.4f} {b:14.4f} {b - a:+9.4f}")
    return out


# --------------------------------------------------------------------------- #
def grid_floor(model_name, dataset, n_bits, prior="flat"):
    """Minimum chi2 over the QGA cell centres (exhaustive, vectorized)."""
    import contextlib
    import io
    import cosmo_core as cc
    model = cc.MODELS[model_name]
    with contextlib.redirect_stdout(io.StringIO()):
        post = cc.Posterior(model, dataset=dataset, prior_type=prior)
    lo = np.array([b[0] for b in model.bounds], float)
    hi = np.array([b[1] for b in model.bounds], float)
    levels = 2 ** n_bits
    axes = [lo[i] + (np.arange(levels) + 0.5) / levels * (hi[i] - lo[i])
            for i in range(model.n_params)]
    best, arg = np.inf, None
    grid = np.array(list(itertools.product(*axes)))
    for chunk in np.array_split(grid, max(1, len(grid) // 20000)):
        # flat prior and cell centres strictly inside the box: -2 log p = chi2
        chi = -2.0 * post.log_prob_batch(chunk)
        i = int(np.argmin(chi))
        if chi[i] < best:
            best, arg = float(chi[i]), chunk[i]
    cont = cc.fit_statistics(post, arg)["chi2"]
    return {"model": model_name, "dataset": dataset, "n_bits": n_bits,
            "chi2_grid_floor": best, "chi2_continuous": float(cont),
            "floor_minus_continuous": best - float(cont), "theta_floor": arg}


def cmd_grid_floor(model, dataset, bits):
    print(f"\n  {model} / {dataset}")
    print(f"  {'n_bits':>6s} {'grid floor':>12s} {'continuous':>12s} {'floor - cont':>13s}")
    res = []
    for nb in bits:
        g = grid_floor(model, dataset, nb)
        res.append(g)
        print(f"  {nb:6d} {g['chi2_grid_floor']:12.4f} {g['chi2_continuous']:12.4f} "
              f"{g['floor_minus_continuous']:13.4f}")
    return res


# --------------------------------------------------------------------------- #
def cmd_mc_error(roots):
    rows = campaign_io.read_campaigns(roots)
    cells = defaultdict(dict)
    for r in rows:
        if r["_fam"] == "samplers":
            cells[campaign_io.cell_key(r)][r["Method"]] = r
    n_tot = n_sig = 0
    print(f"\n  {'model':6s} {'noise':14s} {'grid':>4s} {'method':11s} {'param':6s} "
          f"{'ratio':>7s} {'2*SE':>7s}  verdict")
    for key, v in sorted(cells.items(), key=lambda kv: str(kv[0])):
        ref = v.get("Classical MCMC")
        if not ref:
            continue
        ess_c = _num(ref.get("ESS"))
        for meth in ("QMCMC 50%", "QMCMC 100%"):
            q = v.get(meth)
            if not q or not ess_c:
                continue
            ess_q = _num(q.get("ESS"))
            if not ess_q:
                continue
            se = math.sqrt(1 / (2 * ess_c) + 1 / (2 * ess_q))
            for c in q:
                if not c.endswith("_std"):
                    continue
                a, b = _num(q.get(c)), _num(ref.get(c))
                if not a or not b:
                    continue
                ratio = a / b
                sig = abs(ratio - 1) > 2 * se
                n_tot += 1
                n_sig += sig
                print(f"  {key[5]:6s} {key[7]:14s} {str(key[6]):>4s} {meth:11s} {c[:-4]:6s} "
                      f"{ratio:7.3f} {2 * se:7.3f}  "
                      f"{'beyond MC error' if sig else 'within MC error'}")
    print(f"\n  {n_sig}/{n_tot} sigma ratios differ from 1 by more than 2 Monte Carlo SE")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("refit")
    a.add_argument("campaigns", nargs="+")
    b = sub.add_parser("grid-floor")
    b.add_argument("--model", required=True)
    b.add_argument("--dataset", default="CC+BAO")
    b.add_argument("--n-bits", type=int, nargs="+", default=[4, 5, 6])
    c = sub.add_parser("mc-error")
    c.add_argument("campaigns", nargs="+")
    args = ap.parse_args(argv)
    if args.cmd == "refit":
        cmd_refit(args.campaigns)
    elif args.cmd == "grid-floor":
        cmd_grid_floor(args.model, args.dataset, args.n_bits)
    else:
        cmd_mc_error(args.campaigns)
    return 0


if __name__ == "__main__":
    sys.exit(main())
