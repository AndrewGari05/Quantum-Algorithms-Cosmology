"""Metropolis-Hastings kernel: API, reproducibility, edge cases."""
import numpy as np
import pytest

from qablate import GaussianProposal, MetropolisHastings, Proposal


def gauss(x):
    return -0.5 * np.sum(x ** 2, axis=1)


def test_shapes_and_emcee_names():
    mh = MetropolisHastings(gauss, ndim=3, nchains=5, proposal=GaussianProposal(0.5), rng=0)
    state = mh.run_mcmc(np.zeros((5, 3)), 100)
    assert mh.get_chain().shape == (100, 5, 3)
    assert mh.get_chain(discard=10, thin=3, flat=True).shape == (30 * 5, 3)
    assert mh.get_log_prob().shape == (100, 5)
    assert mh.acceptance_fraction.shape == (5,)
    assert state.coords.shape == (5, 3)
    mh.run_mcmc(None, 20)                        # continue from the last state
    assert mh.iteration == 120


def test_same_seed_same_chain_different_seed_different_chain():
    def run(seed):
        mh = MetropolisHastings(gauss, 2, 3, proposal=GaussianProposal(1.0), rng=seed)
        mh.run_mcmc(np.zeros((3, 2)), 200)
        return mh.get_chain()
    assert np.array_equal(run(4), run(4))
    assert not np.array_equal(run(4), run(5))


def test_non_vectorized_log_prob_gives_same_chain():
    a = MetropolisHastings(gauss, 2, 3, rng=1)
    b = MetropolisHastings(lambda x: -0.5 * float(x @ x), 2, 3, rng=1, vectorized=False)
    a.run_mcmc(np.zeros((3, 2)), 100)
    b.run_mcmc(np.zeros((3, 2)), 100)
    assert np.array_equal(a.get_chain(), b.get_chain())


def test_nan_is_rejected_and_bad_start_raises():
    def lp(x):
        return np.where(x[:, 0] > 1.0, np.nan, gauss(x))

    mh = MetropolisHastings(lp, 1, 4, proposal=GaussianProposal(2.0), rng=0)
    mh.run_mcmc(np.zeros((4, 1)), 500)
    assert np.all(mh.get_chain() <= 1.0)
    with pytest.raises(ValueError):
        MetropolisHastings(lp, 1, 1).run_mcmc(np.full((1, 1), 5.0), 10)


@pytest.mark.parametrize("bad", [dict(ndim=0), dict(nchains=0)])
def test_invalid_sizes(bad):
    kw = dict(ndim=2, nchains=2) | bad
    with pytest.raises(ValueError):
        MetropolisHastings(gauss, **kw)


def test_wrong_initial_shape_and_nsteps():
    mh = MetropolisHastings(gauss, 2, 3)
    with pytest.raises(ValueError):
        mh.run_mcmc(np.zeros((2, 2)), 10)
    with pytest.raises(ValueError):
        mh.run_mcmc(np.zeros((3, 2)), 0)


def test_hastings_term_is_used():
    """An asymmetric proposal with its Hastings term samples the target."""
    class Drift(Proposal):
        symmetric = False

        def propose(self, x, rng):
            step = rng.standard_normal(x.shape) + 0.5          # q(y|x) = N(y; x + 0.5, 1)
            y = x + step
            log_q_ratio = (-0.5 * np.sum((x - y - 0.5) ** 2, axis=1)
                           + 0.5 * np.sum((y - x - 0.5) ** 2, axis=1))
            return y, log_q_ratio

    mh = MetropolisHastings(gauss, 1, 8, proposal=Drift(), rng=3)
    mh.run_mcmc(np.zeros((8, 1)), 6000)
    x = mh.get_chain(discard=500, flat=True)
    assert abs(x.mean()) < 0.1 and abs(x.std() - 1.0) < 0.06


def test_reusing_a_proposal_instance_is_reproducible():
    from qablate import RandomCircuitProposal
    prop = RandomCircuitProposal(0.8)

    def run():
        mh = MetropolisHastings(gauss, 2, 3, proposal=prop, rng=9)
        mh.run_mcmc(np.zeros((3, 2)), 60)
        return mh.get_chain()
    assert np.array_equal(run(), run())
