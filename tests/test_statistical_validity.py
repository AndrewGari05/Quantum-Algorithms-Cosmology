"""Every proposal x acceptance combination must sample the target.

These tests would have caught finding QM-1 (a drifting circuit proposal):
the chain mean is compared with the exact mean in units of its Monte Carlo
standard error, and the marginal with a Kolmogorov-Smirnov test.
"""
import numpy as np
import pytest
from scipy import stats

from qablate import (
    AerBackend,
    AmplitudeEncodedMetropolis,
    GaussianProposal,
    MetropolisAcceptance,
    MetropolisHastings,
    RandomCircuitProposal,
    diagnostics,
)


def gauss(x):
    return -0.5 * np.sum(x ** 2, axis=1)


def _check(mh, nsteps, burn=300, z_max=4.0):
    mh.run_mcmc(np.zeros((mh.nchains, mh.ndim)), nsteps)
    ch = np.swapaxes(mh.get_chain(discard=burn), 0, 1)          # (chains, steps, dim)
    ess = diagnostics.ess(ch)
    mean = ch.reshape(-1, mh.ndim).mean(axis=0)
    z = mean * np.sqrt(ess)
    assert np.all(np.abs(z) < z_max), (mean, ess, z)
    thin = max(1, int(np.ceil(ch.shape[1] * ch.shape[0] / ess.min())))
    p = stats.kstest(ch[:, ::thin, 0].ravel(), "norm").pvalue
    assert p > 1e-3, p
    return mean, ess


CASES = {
    "gaussian+metropolis": lambda: (GaussianProposal(0.9), MetropolisAcceptance()),
    "circuit-counts+metropolis": lambda: (RandomCircuitProposal(0.9), MetropolisAcceptance()),
    "circuit-statevector+metropolis": lambda: (RandomCircuitProposal(0.9, route="statevector"),
                                               MetropolisAcceptance()),
    "circuit-counts+amplitude": lambda: (RandomCircuitProposal(0.9), AmplitudeEncodedMetropolis()),
    "gaussian+amplitude-shots": lambda: (GaussianProposal(0.9), AmplitudeEncodedMetropolis(shots=16)),
}


@pytest.mark.parametrize("case", sorted(CASES))
@pytest.mark.parametrize("seed", [7, 42])
def test_ideal_components_sample_the_target(case, seed):
    prop, acc = CASES[case]()
    mh = MetropolisHastings(gauss, 2, 6, proposal=prop, acceptance=acc, rng=seed)
    _check(mh, 2500)


@pytest.mark.slow
@pytest.mark.parametrize("level", ["readout", "full"])
def test_noisy_circuit_proposal_still_samples_the_target(level):
    """Noise changes the increment distribution but not its symmetry."""
    prop = RandomCircuitProposal(0.9, backend=AerBackend(level), shots=64)
    mh = MetropolisHastings(gauss, 2, 6, proposal=prop, rng=11)
    _check(mh, 2500)


def _balance_ratio(acc, a, rng, n=4000):
    """Estimate log[h(a) / h(-a)] for an acceptance rule."""
    ha = acc.accept(np.full(n, a), rng).mean()
    hb = acc.accept(np.full(n, -a), rng).mean()
    return np.log(ha / hb)


def test_detailed_balance_ratio_ideal_vs_noisy():
    rng = np.random.default_rng(0)
    ideal = AmplitudeEncodedMetropolis()
    assert ideal.preserves_detailed_balance
    assert abs(_balance_ratio(ideal, 2.0, rng) - 2.0) < 0.15
    noisy = AmplitudeEncodedMetropolis(backend=AerBackend("readout"))
    assert not noisy.preserves_detailed_balance
    assert _balance_ratio(noisy, 6.0, rng) < 4.5                 # e^6 not reached
    mitigated = AmplitudeEncodedMetropolis(backend=AerBackend("readout"), readout_mitigation=True)
    assert mitigated.preserves_detailed_balance
    assert abs(_balance_ratio(mitigated, 2.0, rng) - 2.0) < 0.15


def test_sampler_warns_when_detailed_balance_is_broken():
    with pytest.warns(UserWarning, match="detailed balance"):
        MetropolisHastings(gauss, 1, 2, acceptance=AmplitudeEncodedMetropolis(
            backend=AerBackend("full")))


def test_circuit_increments_are_symmetric_for_any_seed():
    """QM-1 regression: the increment mean is zero at the 4-sigma level."""
    for seed in (7, 42, 123):
        prop = RandomCircuitProposal(1.0)
        rng = np.random.default_rng(seed)
        prop._setup(2, rng)
        inc = prop.increments(8192, rng)
        z = inc.mean(axis=0) / (inc.std(axis=0) / np.sqrt(len(inc)))
        assert np.all(np.abs(z) < 4.0), (seed, z)
        assert np.allclose((inc ** 2).mean(axis=0), 1.0, atol=0.08)
