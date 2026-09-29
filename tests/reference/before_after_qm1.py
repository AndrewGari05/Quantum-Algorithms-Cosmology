"""Before/after table for QM-1: QMCMC 50 % vs Classical MCMC, same seed.

Run once per code version (the script imports the modules from --root):

    python tests/reference/before_after_qm1.py --root <checkout> --out result.json \
        --model cpl --route counts --dataset CC+BAO+Pantheon --steps 10000 --chains 8

Prints mean and sigma of every parameter, split-R-hat and ESS for both rungs,
and the QMCMC - MCMC shift in units of the MCMC sigma.
"""
import argparse
import contextlib
import io
import json
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="lcdm")
    ap.add_argument("--route", default="amplitude", choices=["amplitude", "counts"])
    ap.add_argument("--dataset", default="CC+BAO+Pantheon")
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--chains", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    sys.path.insert(0, os.path.abspath(a.root))
    os.chdir(a.root)
    import cosmo_core as cc
    import cosmo_modular_quantum as cmq
    import cosmo_noise as cn
    cmq.set_noise(cn.NoiseSpec.from_level("none"), proposal_route=a.route)
    with contextlib.redirect_stdout(io.StringIO()):
        post = cc.Posterior(cc.MODELS[a.model], dataset=a.dataset)
    burn = max(50, int(0.1 * a.steps))
    out = {"model": a.model, "route": a.route, "dataset": a.dataset,
           "steps": a.steps, "chains": a.chains, "seed": a.seed,
           "params": list(cc.MODELS[a.model].param_names)}
    for tag, cfg in (("C-MCMC", {}), ("QMCMC50", {"proposal": True})):
        cmq._reseed(a.seed)
        s = cmq.QMCMCModular(post, cfg, n_chains=a.chains, n_burn=burn,
                             stop_on_convergence=False)
        r = s.run(n_steps=a.steps, progress=False)
        f = r["flat"]
        out[tag] = {"mean": f.mean(0).tolist(), "std": f.std(0).tolist(),
                    "ess": float(r["ess"]),
                    "rhat": float(cc.split_rhat(r["chains"])),
                    "acceptance": float(r["acceptance"])}
    m, q = out["C-MCMC"], out["QMCMC50"]
    out["shift_in_sigma"] = [(qm - mm) / ms for qm, mm, ms in
                             zip(q["mean"], m["mean"], m["std"])]
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out["shift_in_sigma"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
