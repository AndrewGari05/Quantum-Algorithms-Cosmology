"""Agreement with the thesis-errata code (tag ``v0.8.2-thesis-defense`` lineage).

Tier (a): deterministic quantities must match the post-errata reference.
Files in ``tests/golden`` were produced by the errata branch at commit
``eed1cc4`` (``make_reference.py`` and the grid dump in the v1 notes).
"""
import os

import numpy as np
import pytest

from qablate import Grid
from qablate.cosmology import MODELS, Posterior, fit_statistics

GOLDEN = os.path.join(os.path.dirname(__file__), "golden")


@pytest.fixture(scope="module")
def ref():
    return np.load(os.path.join(GOLDEN, "errata_reference.npz"))


@pytest.mark.parametrize("name", sorted(MODELS))
def test_log_prob_matches_errata_code(ref, name):
    post = Posterior(name, "CC+BAO+Pantheon")
    th = np.array([np.mean(b) for b in MODELS[name].sample_box])
    # rel 1e-10: vectorized exp/log differ by a few ulps between CPU instruction
    # sets, and the Pantheon likelihood sums ~1000 of them (seen: 3e-12 relative)
    assert post.log_prob(th)[0] == pytest.approx(ref[f"logprob/{name}"][0], rel=1e-10, abs=0)


@pytest.mark.parametrize("name", sorted(MODELS))
def test_fit_statistics_match_errata_code(ref, name):
    post = Posterior(name, "CC+BAO+Pantheon")
    th = np.array([np.mean(b) for b in MODELS[name].sample_box])
    st = fit_statistics(post, th)
    chi2, chi2_red, aic, bic = ref[f"fitstats/{name}"]
    # optimizer tolerance, not bit identity: two float paths to the same minimum
    assert st["chi2"] == pytest.approx(chi2, abs=1e-6)
    assert st["BIC"] == pytest.approx(bic, abs=1e-6)


@pytest.mark.parametrize("name,dataset", [("lcdm", "CC+BAO"), ("cpl", "CC+BAO+Pantheon"),
                                          ("gede", "CC+BAO")])
def test_grid_encoding_and_target_match_legacy(name, dataset):
    g = np.load(os.path.join(GOLDEN, "legacy_grid.npz"))
    grid = Grid(g[f"{name}/window"], int(g[f"{name}/meta"][0]))
    assert np.array_equal(grid.points, g[f"{name}/points"])
    target = grid.target(Posterior(name, dataset).log_prob)
    # The legacy grid target used a second, batched interpolation path whose
    # log-probabilities differ from the scalar path by ~1e-9 on supernova
    # data; qablate uses one path everywhere.
    assert np.allclose(target, g[f"{name}/target"], rtol=1e-7, atol=1e-300)


@pytest.mark.parametrize("name,dataset", [("cpl", "CC+BAO"), ("gede", "CC+BAO+Pantheon")])
def test_best_fit_not_worse_than_multistart(name, dataset):
    from scipy.optimize import minimize
    post = Posterior(name, dataset)
    st = fit_statistics(post, MODELS[name].fiducial)
    lo, hi = post._lo, post._hi
    rng = np.random.default_rng(0)
    best = np.inf
    for _ in range(8):
        r = minimize(lambda u: post.chi2(np.clip(lo + u * (hi - lo), lo, hi)),
                     rng.uniform(0.05, 0.95, len(lo)), method="Nelder-Mead",
                     bounds=[(0, 1)] * len(lo), options={"maxiter": 6000, "xatol": 1e-10,
                                                         "fatol": 1e-10})
        best = min(best, r.fun)
    assert st["chi2"] <= best + 1e-6


# --------------------------------------------------------------------------- #
# Tier (c): statistical agreement with the post-errata thesis code
# (tests/golden/errata_statistics.npz, made by tests/reference/make_golden_v1.py
# on branch thesis-errata): LCDM on CC+BAO, seed 42.
# --------------------------------------------------------------------------- #
from qablate import (  # noqa: E402
    BornMachineVI,
    GaussianProposal,
    MetropolisHastings,
    RandomCircuitProposal,
    diagnostics,
)


@pytest.fixture(scope="module")
def stats_ref():
    return np.load(os.path.join(GOLDEN, "errata_statistics.npz"))


@pytest.mark.parametrize("rung", ["mcmc", "qmcmc"])
def test_chains_agree_with_errata_code_within_mc_error(stats_ref, rung):
    post = Posterior("lcdm", "CC+BAO")
    box = np.array(post.model.sample_box)
    scale = 0.06 * (box[:, 1] - box[:, 0])
    prop = GaussianProposal(scale) if rung == "mcmc" else RandomCircuitProposal(scale)
    rng = np.random.default_rng(42)
    mh = MetropolisHastings(post.log_prob, 2, 6, proposal=prop, rng=rng)
    mh.run_mcmc(box[:, 0] + rng.random((6, 2)) * (box[:, 1] - box[:, 0]), 16400)
    ch = np.swapaxes(mh.get_chain(discard=400), 0, 1)
    mean, sd = ch.reshape(-1, 2).mean(0), ch.reshape(-1, 2).std(0)
    ess = diagnostics.ess(ch)
    ref_mean, ref_sd, ref_ess = stats_ref[f"{rung}/summary"]
    se_mean = np.sqrt(sd ** 2 / ess + ref_sd ** 2 / ref_ess)
    se_sd = np.sqrt(sd ** 2 / (2 * ess) + ref_sd ** 2 / (2 * ref_ess))
    assert np.all(np.abs(mean - ref_mean) < 4 * se_mean), (mean, ref_mean, se_mean)
    assert np.all(np.abs(sd - ref_sd) < 4 * se_sd), (sd, ref_sd, se_sd)


def test_classical_vi_agrees_with_errata_code(stats_ref):
    # COBYLA on 42 angles is chaotic: the last bits of floating-point arithmetic
    # (SciPy version, CPU vector instructions) send a fixed seed to different
    # local minima (KL from 0.18 to 2.1 over 12 seeds on one machine). The
    # errata run (one seed, KL 0.48) is therefore compared with the best of four
    # seeds, which reaches a comparable minimum on every platform tested, and the
    # moments are compared at the scale of the posterior, not of one run (minima
    # with the same KL differ by up to ~40 % in width).
    post = Posterior("lcdm", "CC+BAO")
    grid = Grid(stats_ref["vi/window"], 3)
    target = grid.target(post.log_prob)
    best = min((BornMachineVI(grid, optimizer="cobyla").fit(target, 20, np.random.default_rng(s))
                for s in range(4)), key=lambda r: r.kl)
    ref_mean, ref_sd = stats_ref["vi/summary"]
    assert best.kl <= float(stats_ref["vi/kl"][0]) + 0.15
    assert np.all(np.abs(best.mean - ref_mean) < 0.5 * ref_sd)
    assert np.all(np.abs(best.std / ref_sd - 1.0) < 0.6)     # widths vary between minima


def test_continuous_ga_reaches_the_errata_optimum(stats_ref):
    from qablate.experimental import GAConfig, GeneticAlgorithm
    post = Posterior("lcdm", "CC+BAO")
    r = GeneticAlgorithm(post.log_prob, post.model.bounds, GAConfig(pop_size=60, n_generations=30),
                         init_box=post.model.sample_box, rng=42).run()
    assert post.chi2(r.theta_best) < float(stats_ref["ga/chi2_best"][0]) + 0.05
    assert fit_statistics(post, r.theta_best)["chi2"] == pytest.approx(
        float(stats_ref["ga/chi2_map"][0]), abs=1e-5)
