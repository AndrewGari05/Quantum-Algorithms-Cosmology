"""Summaries of the post-errata thesis code used by the library's tier (c) tests.

Writes ``errata_statistics.npz`` (path given as the first argument):
chain summaries of Classical MCMC and QMCMC 50 % (counts route), the
Classical VI result on a fixed grid window, and the CGA best fit, all for
LCDM on CC+BAO with seed 42.
"""
import contextlib
import io
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import arviz as az  # noqa: E402

with contextlib.redirect_stdout(io.StringIO()):
    import cosmo_core as cc  # noqa: E402
    import cosmo_genetic_optimizers as cgo  # noqa: E402
    import cosmo_modular_quantum as cmq  # noqa: E402
    import cosmo_noise as cn  # noqa: E402

SEED, STEPS, CHAINS = 42, 16000, 6


def chain_summary(chains):
    """(chains, steps, d) -> mean, sd, bulk ESS per parameter."""
    flat = chains.reshape(-1, chains.shape[2])
    ess = [float(az.ess(chains[:, :, j], method="bulk")) for j in range(chains.shape[2])]
    return np.array([flat.mean(0), flat.std(0), ess])


def main(out):
    with contextlib.redirect_stdout(io.StringIO()):
        post = cc.Posterior(cc.MODELS["lcdm"], dataset="CC+BAO")
    res = {}
    cmq.set_noise(cn.NoiseSpec.from_level("none"), proposal_route="counts")
    for tag, cfg in (("mcmc", {}), ("qmcmc", {"proposal": True})):
        cmq._reseed(SEED)
        s = cmq.QMCMCModular(post, cfg, n_chains=CHAINS, n_burn=400, stop_on_convergence=False)
        r = s.run(n_steps=STEPS, progress=False)
        res[f"{tag}/summary"] = chain_summary(r["chains"])
    window = [(0.20, 0.32), (66.0, 76.0)]
    cmq._reseed(SEED + 1)
    q = cmq.QVMCModular(post, {}, n_qubits_per_param=3, n_shots=4000, grid_window=window)
    r = q.run(max_iter=20, n_chains=1, progress=False)
    res["vi/window"] = np.array(window)
    res["vi/summary"] = np.array([r["mu"], r["sd"]])
    res["vi/kl"] = np.array([r["kl_final"]])
    res["vi/n_samples"] = np.array([len(r["S"])])
    ga = cgo.GAConfig(pop_size=60, n_generations=30, seed=SEED)
    with contextlib.redirect_stdout(io.StringIO()):
        g = cgo.CGA(post, ga, rng=np.random.default_rng(SEED)).evolve(
            live=False, record_population=False, log_every=10 ** 9)
    res["ga/chi2_best"] = np.array([g.chi2_grid])
    res["ga/chi2_map"] = np.array([g.chi2_map])
    np.savez_compressed(out, **res)
    for k, v in res.items():
        print(k, np.round(v, 5).tolist())


if __name__ == "__main__":
    main(sys.argv[1])
