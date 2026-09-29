"""Deterministic reference set for regression testing (Phase 0 of the plan).

Runs a fixed set of short, seed-42, CPU-only cells through the legacy API and
stores every numeric output in ``reference.npz`` plus a SHA-256 manifest in
``reference_manifest.json``.

Rules enforced by ``tests/test_reference.py``:

* commits that only move/translate code or fix plumbing must reproduce every
  array bit-for-bit;
* a commit that intentionally changes results must declare, in its message,
  ``Changes-results: <ID>, <keys>`` and regenerate this set in the same commit.

Usage::

    python tests/reference/make_reference.py            # regenerate
    python tests/reference/make_reference.py --check    # compare, exit 1 on diff
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
warnings.filterwarnings("ignore")

import cosmo_core as cc                      # noqa: E402
import cosmo_modular_quantum as cmq          # noqa: E402
import cosmo_genetic_optimizers as cgo       # noqa: E402
import cosmo_noise as cnoise                 # noqa: E402

SEED = 42
NPZ = os.path.join(HERE, "reference.npz")
MANIFEST = os.path.join(HERE, "reference_manifest.json")


def _set_noise(level: str, route: str = "auto") -> None:
    spec = cnoise.NoiseSpec.from_level(level)
    cmq.set_noise(spec, proposal_route=route)
    cgo.set_noise(spec)


def _qmcmc(post, config, n_steps=300, n_chains=4):
    cmq._reseed(SEED)
    run = cmq.QMCMCModular(post, config, n_chains=n_chains, n_burn=50,
                           stop_on_convergence=False)
    out = run.run(n_steps=n_steps, logger=None, progress=False, tag="ref")
    return {"chains": np.asarray(out["chains"], dtype=float),
            "acceptance": np.atleast_1d(np.asarray(out["acceptance"], dtype=float))}


def _qvmc(post, config, nqpp=3, max_iter=15):
    cmq._reseed(SEED + 1)
    run = cmq.QVMCModular(post, config, n_qubits_per_param=nqpp,
                          n_shots=512, budget_mode="circuits")
    out = run.run(max_iter=max_iter, n_chains=2, logger=None, progress=False,
                  tag="ref")
    return {"mu": np.asarray(out["mu"], float), "sd": np.asarray(out["sd"], float),
            "kl_final": np.atleast_1d(float(out["kl_final"]))}


def _ga(post, method, qcfg, n_bits=4):
    ga = cgo.GAConfig(pop_size=40, n_generations=8, seed=SEED)
    rng = np.random.default_rng(SEED)
    opt = cgo._build_optimizer(method, post, ga, qcfg, n_bits, rng, shots=1)
    res = opt.evolve(live=False, logger=None, record_population=False,
                     log_every=10**9)
    return {"final_pop": np.asarray(res.final_pop, float),
            "theta_grid": np.asarray(res.theta_grid, float),
            "theta_map": np.asarray(res.theta_map, float),
            "chi2_grid": np.atleast_1d(float(res.chi2_grid)),
            "chi2_map": np.atleast_1d(float(res.chi2_map))}


def build() -> dict:
    out: dict = {}

    def put(prefix, d):
        for k, v in d.items():
            out[f"{prefix}/{k}"] = np.asarray(v)

    post = cc.Posterior(cc.MODELS["lcdm"], dataset="CC+BAO")

    # --- deterministic physics / statistics ---------------------------------
    for name, model in cc.MODELS.items():
        p = cc.Posterior(model, dataset="CC+BAO+Pantheon")
        th = np.array([np.mean(b) for b in model.sample_box])
        out[f"logprob/{name}"] = np.atleast_1d(p.log_prob(th))
        fs = cc.fit_statistics(p, th)
        out[f"fitstats/{name}"] = np.array(
            [fs["chi2"], fs["chi2_red"], fs["AIC"], fs["BIC"]], float)

    # --- QMCMC ladder, ideal, both proposal routes --------------------------
    for route in ("amplitude", "counts"):
        _set_noise("none", route)
        for i, cfg in enumerate(cmq.qmcmc_ladder()):
            if i == 0 and route == "counts":
                continue                              # classical rung: route-free
            put(f"qmcmc/{route}/rung{i}", _qmcmc(post, cfg))

    # --- QVMC ladder, ideal --------------------------------------------------
    _set_noise("none", "auto")
    for i, cfg in enumerate(cmq.qvmc_ladder()):
        put(f"qvmc/none/rung{i}", _qvmc(post, cfg))

    # --- noisy cells ---------------------------------------------------------
    for level in ("readout", "fake_brisbane"):
        _set_noise(level, "auto")
        put(f"qmcmc/{level}/rung2", _qmcmc(post, cmq.qmcmc_ladder()[2], n_steps=150))
        put(f"qvmc/{level}/rung2", _qvmc(post, cmq.qvmc_ladder()[2], nqpp=2, max_iter=6))

    # --- genetic ladder, ideal ----------------------------------------------
    _set_noise("none", "auto")
    put("ga/cga", _ga(post, "cga", dict(cgo.QGA_PRESETS[0])))
    for pct in sorted(cgo.QGA_PRESETS):
        put(f"ga/qga{pct}", _ga(post, "qga", dict(cgo.QGA_PRESETS[pct])))
    return out


def digest(arr: np.ndarray) -> str:
    a = np.ascontiguousarray(arr)
    return hashlib.sha256(a.dtype.str.encode() + str(a.shape).encode()
                          + a.tobytes()).hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true",
                    help="compare against the stored set instead of writing it")
    args = ap.parse_args(argv)
    data = build()
    manifest = {k: digest(v) for k, v in sorted(data.items())}
    if args.check:
        with open(MANIFEST) as fh:
            ref = json.load(fh)
        bad = sorted(k for k in set(ref) | set(manifest) if ref.get(k) != manifest.get(k))
        for k in bad:
            print(f"DIFF {k}")
        print(f"{len(manifest) - len(bad)}/{len(manifest)} arrays identical")
        return 1 if bad else 0
    np.savez_compressed(NPZ, **data)
    with open(MANIFEST, "w") as fh:
        json.dump(manifest, fh, indent=1, sort_keys=True)
    print(f"wrote {len(data)} arrays -> {NPZ}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
