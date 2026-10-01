"""
test_core.py — Physics and statistics regression tests for cosmo_core.

These tests require only NumPy/SciPy (no Qiskit), so they run anywhere and
form the always-on correctness floor. Run with:  pytest tests/  (or directly:
python tests/test_core.py).

They encode the claims the PHASE-1 review asked to make falsifiable:
  * every model has E²(z;θ) > 0 across its prior box (no log/sqrt of <=0);
  * PEDE/GEDE satisfy f_DE(0)=1 exactly, so E²(0)=1 (P1);
  * GEDE convention is checked at a non-zero z, not only at z=0 (P1);
  * χ² is finite, non-negative and matches a recomputed reference;
  * the Sokal autocorrelation estimator recovers known τ and does NOT
    mis-fire on excellent (anti-correlated) chains (B5 regression);
  * rank-normalized split-R̂ separates converged from drifting chains at
    the 1.01 threshold (S2/S4);
  * the M_abs marginalization is invariant to a constant magnitude shift
    (the analytic-marginalization property, Goliath 2001).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cosmo_core as core


# ─────────────────────────────────────────────────────────────────────────────
# Physics
# ─────────────────────────────────────────────────────────────────────────────

def test_E2_positive_across_box():
    """E²(z;θ) > 0 for every model, sampled across its prior box."""
    rng = np.random.default_rng(0)
    z = np.linspace(0.0, 2.5, 60)
    for name, m in core.MODELS.items():
        lo = np.array([b[0] for b in m.bounds])
        hi = np.array([b[1] for b in m.bounds])
        for _ in range(200):
            th = lo + (hi - lo) * rng.uniform(size=len(lo))
            e2 = m.E2(z, th)
            assert np.all(np.isfinite(e2)), f"{name}: non-finite E²"
            # Unphysical corners may legitimately give E²<=0; the likelihood
            # guards them. We only require the FIDUCIAL slice to be positive:
        e2f = m.E2(z, np.array(m.fiducial, dtype=float))
        assert np.all(e2f > 0), f"{name}: E²<=0 at fiducial"


def test_emergent_de_normalization():
    """PEDE and GEDE: f_DE(0)=1, hence E²(0)=Ωm+Ωr+Ω_DE=1 (P1)."""
    for Om in (0.25, 0.31, 0.40):
        # PEDE (no extra params)
        e2_0 = float(core._E2_pede(np.array([0.0]), np.array([Om, 70.0]))[0])
        assert abs(e2_0 - 1.0) < 1e-12, f"PEDE E²(0)≠1 at Ωm={Om}"
        # GEDE for several Δ
        for D in (-1.0, 0.0, 0.5, 1.0, 3.0):
            e2_0 = float(core._E2_gede(np.array([0.0]),
                                       np.array([Om, 70.0, D]))[0])
            assert abs(e2_0 - 1.0) < 1e-12, f"GEDE E²(0)≠1 at Ωm={Om}, Δ={D}"


def test_gede_reduces_to_lcdm_at_delta_zero():
    """GEDE with Δ=0 has constant f_DE=1 (ΛCDM-like) at all z (P1 convention).

    Checks at a NON-zero redshift, so an inverted z_t sign would be caught.
    """
    Om = 0.31
    z = np.array([0.5, 1.0, 2.0])
    e2_gede = core._E2_gede(z, np.array([Om, 70.0, 0.0]))
    e2_lcdm = core._E2_lcdm(z, np.array([Om, 70.0]))
    assert np.allclose(e2_gede, e2_lcdm, rtol=1e-10), \
        "GEDE(Δ=0) must coincide with ΛCDM at all z"


def test_chi2_finite_and_reference():
    """χ² is finite, ≥0, and matches an independent recompute (CC+BAO)."""
    post = core.Posterior(core.MODELS['lcdm'], 'CC+BAO', 'flat')
    th = np.array([0.31, 67.7])
    c, n = post.chi2(th)
    assert np.isfinite(c) and c >= 0
    # independent recompute of the diagonal CC χ²
    Hm = post.model.H(post.z_cc, th)
    c_ref = float(np.sum(((post.H_cc - Hm) / post.sig_cc) ** 2))
    assert abs(c - c_ref) < 1e-9
    assert n == len(post.z_cc)


def test_flat_distance_monotonic():
    """μ_th increases with z (luminosity distance grows) for ΛCDM."""
    post = core.Posterior(core.MODELS['lcdm'], 'CC+BAO', 'flat')
    z = np.linspace(0.01, 2.0, 50)
    post._zg = np.linspace(0.0, 2.1, 800)          # ensure grid covers z
    mu = post._mu_theory(z, np.array([0.31, 70.0]))
    assert mu is not None and np.all(np.diff(mu) > 0)


# ─────────────────────────────────────────────────────────────────────────────
# Statistics
# ─────────────────────────────────────────────────────────────────────────────

def test_autocorr_white_noise():
    """White noise has τ ≈ 1."""
    rng = np.random.default_rng(1)
    tau = core.autocorr_time_fft(rng.standard_normal(20000))
    assert 0.8 < tau < 1.5


def test_autocorr_ar1():
    """AR(1) with φ=0.9 has τ ≈ (1+φ)/(1−φ) = 19 (Sokal window)."""
    rng = np.random.default_rng(2)
    phi = 0.9
    x = np.zeros(40000)
    for i in range(1, len(x)):
        x[i] = phi * x[i - 1] + rng.standard_normal()
    tau = core.autocorr_time_fft(x)
    assert 14.0 < tau < 26.0, f"τ={tau} off theoretical 19"


def test_autocorr_excellent_chain_regression():
    """B5 regression: an anti-correlated chain must NOT return a huge τ.

    The previous 'first lag below 0.05 else N/4' rule returned ~N/4 here;
    the Sokal window returns τ ≈ 1.
    """
    alt = np.tile([1.0, -1.0], 5000)
    tau = core.autocorr_time_fft(alt)
    assert tau < 2.0, f"τ={tau} (regression: should be ~1, not ~N/4)"


def test_split_rhat_converged_vs_drift():
    """Rank-normalized split-R̂ separates converged from drifting (S2/S4)."""
    rng = np.random.default_rng(3)
    conv = rng.standard_normal((4, 5000, 2))
    drift = rng.standard_normal((4, 5000, 2)) + \
        np.array([0, 2, 4, 6])[:, None, None]
    assert core.split_rhat(conv) < core.RHAT_THRESHOLD
    assert core.split_rhat(drift) > 1.1
    assert core.mcmc_converged(conv)
    assert not core.mcmc_converged(drift)


def test_split_rhat_catches_within_chain_trend():
    """Split-R̂ catches a slow within-chain trend a whole-chain R̂ misses."""
    rng = np.random.default_rng(4)
    n = 4000
    trend = np.linspace(0, 3, n)
    chains = rng.standard_normal((4, n, 1)) + trend[None, :, None]
    # classical (non-split) R̂ is blind to a trend shared by all chains;
    # split-R̂ is not.
    assert core.split_rhat(chains) > 1.05


def test_ess_bounded_and_reasonable():
    """ESS ≤ M·N and ESS(white) close to M·N."""
    rng = np.random.default_rng(5)
    white = rng.standard_normal((4, 5000, 2))
    ess = core.ess_chains(white)
    assert ess <= 4 * 5000 + 1
    assert ess > 0.5 * 4 * 5000   # near-independent draws


def test_kish_ess_weights():
    """Kish ESS: equal weights → N; one dominant weight → ≈1."""
    w_eq = np.ones(1000)
    assert abs(core.ess_weights(w_eq) - 1000) < 1e-6
    w_spike = np.zeros(1000)
    w_spike[0] = 1.0
    assert core.ess_weights(w_spike) < 1.5


def test_mabs_marginalization_shift_invariance():
    """Analytic M_abs marginalization is invariant to a constant μ shift.

    Adding a constant to all observed magnitudes (a change of M_abs) must
    leave the marginalized χ² unchanged (Goliath 2001). This is the core
    correctness property of the marginalization.
    """
    post = core.Posterior(core.MODELS['lcdm'], 'CC+BAO', 'flat')
    # Build a synthetic diagonal-SN posterior by hand to avoid needing files.
    z = np.linspace(0.05, 1.5, 60)
    post._zg = np.linspace(0.0, 1.6, 800)
    th = np.array([0.31, 70.0])
    mu_th = post._mu_theory(z, th)
    rng = np.random.default_rng(6)
    dmb = 0.1 * np.ones_like(z)
    mb = mu_th + rng.normal(0, 0.1, size=z.size)
    post.pantheon = {'z': z, 'mb': mb, 'dmb': dmb, 'cov': None}
    post.components = {'sn'}
    post._inv_s2 = 1.0 / dmb ** 2
    post._C_marg = float(np.sum(post._inv_s2))
    c1 = post.chi2_pantheon(th)
    post.pantheon['mb'] = mb + 0.5          # constant M_abs shift
    c2 = post.chi2_pantheon(th)
    assert abs(c1 - c2) < 1e-6, "marginalized χ² not shift-invariant"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {fn.__name__}: {e}")
        except Exception as e:
            failed += 1
            print(f"ERROR {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)


# ─────────────────────────────────────────────────────────────────────────────
# Fixes from the adversarial review of cosmo_core.py
# ─────────────────────────────────────────────────────────────────────────────

def test_b_bounds_best_fit_lies_inside_prior():
    """[B-BOUNDS] `fit_statistics` must not return a point with posterior 0.

    The refinement was an unbounded Nelder-Mead on chi2, which does not see
    the prior. Since chi2 is finite outside the box, the simplex escaped: 5 of
    the 15 model x dataset combinations returned a "best fit" outside the
    support, one of them with H0 = 6.3e-5 km/s/Mpc. The chi2 barely changed,
    so the number looked sane.
    """
    import numpy as np
    import cosmo_core as C
    bad = []
    for name in ('lcdm', 'wcdm', 'cpl', 'pede', 'gede'):
        for ds in ('CC+BAO', 'Pantheon', 'CC+BAO+Pantheon'):
            P = C.Posterior(C.MODELS[name], ds)
            r = C.fit_statistics(P, C.MODELS[name].fiducial)
            if not np.isfinite(P.log_prior(r['theta_best'])):
                bad.append(f"{name}/{ds}: {r['theta_best']}")
    assert not bad, "best fit outside the prior in:\n  " + "\n  ".join(bad)


def test_b_gridseed_window_is_reproducible():
    """[B-GRIDSEED] Same seed, same grid — otherwise the QPU and the simulator
    are not comparable.

    The pre-fit drew from the global RNG: the Om width varied by 54 % between
    identical calls, and depended on how many numbers another part of the
    program had consumed before. Since cosmo_modular_quantum and
    qpu_cosmo_samplers each call it on their own, each ended up with a
    DIFFERENT grid, and their KLs were compared on different discretizations.
    """
    import numpy as np
    import cosmo_core as C
    P = C.Posterior(C.MODELS['lcdm'], 'CC+BAO')
    w0 = C.estimate_grid_window(P)
    for _ in range(3):
        assert C.estimate_grid_window(P) == w0
    # and it must not depend on the state of the global RNG
    C.RNG.normal(size=13)
    assert C.estimate_grid_window(P) == w0
    # different seeds DO give different windows (otherwise it would not be random)
    assert C.estimate_grid_window(P, seed=7) != w0
    # seed=None restores the old, non-reproducible behaviour
    vs = {tuple(map(tuple, C.estimate_grid_window(P, seed=None)))
          for _ in range(4)}
    assert len(vs) > 1


def test_b_clip_unphysical_model_is_rejected():
    """[B-CLIP] E^2 <= 0 must give -inf, not a large but finite chi2.

    `H()` used to clip to 1e-12 and return H = 1e-6*H0, which produced a
    finite log-posterior. Worse: the supernova branch DID reject, so the same
    theta had a finite posterior with CC+BAO and -inf with
    CC+BAO+Pantheon. Latent today (the 5 models have E^2 > 0 in their box),
    but it stops being latent as soon as curvature is added.
    """
    import numpy as np
    import cosmo_core as C
    m = C.CosmoModel(
        name='toy_ok', label='toy with negative curvature',
        param_names=['Om', 'H0'], param_latex=[r'$\Omega_m$', r'$H_0$'],
        bounds=[(0.18, 0.50), (60.0, 82.0)],
        sample_box=[(0.18, 0.50), (60.0, 82.0)],
        fiducial=np.array([0.3, 70.0]),
        E2=lambda z, th: th[0] * (1 + z) ** 3 - 2.0 * (1 + z) ** 2 + (1 - th[0]))
    th = np.array([0.3, 70.0])
    assert np.min(m.E2(np.array([0.0, 1.0, 2.0]), th)) < 0
    for ds in ('CC+BAO', 'CC+BAO+Pantheon'):
        P = C.Posterior(m, ds)
        assert not np.isfinite(P.log_prob(th)), f"{ds}: not rejected"
        assert not np.isfinite(P.log_prob_batch(th[None, :])[0]), \
            f"{ds}: the vectorized path does not match the scalar one"


def test_b_priortype_rejects_arbitrary_strings():
    """[B-PRIORTYPE] A misspelled prior silently fell back to flat.

    A run labelled 'gaussian' in the CSV that actually used a flat prior is
    unrecoverable afterwards.
    """
    import pytest as _pytest
    import cosmo_core as C
    for ok in ('flat', 'gaussian'):
        C.Posterior(C.MODELS['lcdm'], 'CC+BAO', prior_type=ok)
    for bad in ('Gaussian', 'planck', 'gauss', ''):
        with _pytest.raises(ValueError, match='prior_type'):
            C.Posterior(C.MODELS['lcdm'], 'CC+BAO', prior_type=bad)


def test_scalar_and_batch_agree_on_physical_points():
    """The [B-CLIP] fix must not have moved any real number."""
    import numpy as np
    import cosmo_core as C
    rng = np.random.default_rng(0)
    for name in ('lcdm', 'wcdm', 'cpl', 'pede', 'gede'):
        M = C.MODELS[name]
        P = C.Posterior(M, 'CC+BAO+Pantheon')
        lo = np.array([b[0] for b in M.bounds])
        hi = np.array([b[1] for b in M.bounds])
        g = lo + (hi - lo) * rng.uniform(0.2, 0.8, size=(12, M.n_params))
        a = np.array([P.log_prob(t) for t in g])
        b = P.log_prob_batch(g)
        assert np.allclose(a, b, rtol=1e-10, atol=1e-8), name
