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
    assert post.log_prob(th)[0] == pytest.approx(ref[f"logprob/{name}"][0], rel=1e-13, abs=0)


@pytest.mark.parametrize("name", sorted(MODELS))
def test_fit_statistics_match_errata_code(ref, name):
    post = Posterior(name, "CC+BAO+Pantheon")
    th = np.array([np.mean(b) for b in MODELS[name].sample_box])
    st = fit_statistics(post, th)
    chi2, chi2_red, aic, bic = ref[f"fitstats/{name}"]
    # optimizer tolerance, not bit identity: two float paths to the same minimum
    assert st["chi2"] == pytest.approx(chi2, abs=1e-8)
    assert st["BIC"] == pytest.approx(bic, abs=1e-8)


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
    assert np.allclose(target, g[f"{name}/target"], rtol=1e-8, atol=1e-300)


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
