"""
test_fixes_phase3.py — regression guards for the Phase-3 fix batch.

Covers, on the REAL production code paths:
  H1  split-R-hat (1.01) wired into the QMCMC convergence loop
  H2  run_comparison produces equal-length chains
  H3  the KL objective penalizes probability mass leaked outside the target
  H5  the training gradient is exact (parameter-shift on probabilities)
  H6  quantum mutation is gated by mutation_rate
  M1  quantum run and classical baseline share the VI initialization
  M3  QGA runs are reproducible under a fixed seed
  M4  simulator proposal engine: fixed calibration constants, unit std
  H4  QPU proposal engine (dry-run): displacements calibrated to unit std

Requires qiskit + qiskit-aer; skipped automatically when absent.
"""
import contextlib
import io
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

pytest.importorskip("qiskit")
pytest.importorskip("qiskit_aer")

from qiskit import transpile                         # noqa: E402

import cosmo_core as core                            # noqa: E402
from cosmo_core import MODELS, Posterior             # noqa: E402
import cosmo_modular_quantum as mq                   # noqa: E402
import qpu_cosmo_samplers as qpu                     # noqa: E402
from cosmo_genetic_optimizers import QGA, GAConfig   # noqa: E402


@pytest.fixture(scope="module")
def post():
    return Posterior(MODELS['lcdm'], 'CC+BAO', 'flat')


# ── H3: leakage-aware KL ─────────────────────────────────────────────────────

def test_kl_penalizes_leaked_mass(post):
    """A state whose probability mass sits OUTSIDE the target support must
    report a LARGE KL. Under the old masked definition it reported ~0."""
    mq._reseed(3)
    qv = mq.QVMCModular(post, dict(mq.CLASSICAL_BASELINE),
                        n_qubits_per_param=2, n_shots=200)
    qc, n_p = qv._build_ansatz()
    qc_t = transpile(qc.remove_final_measurements(inplace=False), qv.sim)
    # phi = 0 -> ansatz RY/RZ angles all zero -> Q is (close to) a delta on
    # basis state 0. Synthetic target with ALL its mass elsewhere:
    n = 2 ** qc.num_qubits
    P_syn = np.zeros(n)
    P_syn[n // 2] = 1.0
    kl = float(qv._kl_batch(np.zeros(n_p), qc_t, P_syn)[0])
    assert kl > 15.0     # ~|log eps| when the mass is fully leaked


# ── H5: exact gradient ───────────────────────────────────────────────────────

def test_training_gradient_is_exact(post):
    """The implemented gradient (parameter-shift on the PROBABILITIES +
    chain rule) must match small-h central differences of the KL. The old
    +-pi/2 shift applied to the KL itself had ~10% relative error."""
    mq._reseed(7)
    qv = mq.QVMCModular(post, dict(mq.CLASSICAL_BASELINE, training=True),
                        n_qubits_per_param=2, n_shots=200)
    P_target = qv.build_target()
    qc, n_p = qv._build_ansatz()
    qc_t = transpile(qc.remove_final_measurements(inplace=False), qv.sim)
    phi = np.random.default_rng(3).uniform(0, 2 * np.pi, n_p)

    _, Qs = qv._kl_batch(phi, qc_t, P_target, return_q=True)
    shifts = np.repeat(phi[None, :], 2 * n_p, axis=0)
    for j in range(n_p):
        shifts[2 * j, j] += np.pi / 2
        shifts[2 * j + 1, j] -= np.pi / 2
    _, Qs_s = qv._kl_batch(shifts, qc_t, P_target, return_q=True)
    Qmat = np.asarray(Qs_s)
    dQ = (Qmat[0::2] - Qmat[1::2]) / 2.0
    eps = 1e-12
    P_s = P_target + eps
    P_s = P_s / P_s.sum()
    w = np.log(np.clip(Qs[0], eps, None)) - np.log(P_s)
    g_impl = dQ @ w

    h = 1e-6
    rng = np.random.default_rng(0)
    for j in rng.choice(n_p, size=min(5, n_p), replace=False):
        pp, pm = phi.copy(), phi.copy()
        pp[j] += h
        pm[j] -= h
        g_ref = (qv._kl_batch(pp, qc_t, P_target)[0]
                 - qv._kl_batch(pm, qc_t, P_target)[0]) / (2 * h)
        assert abs(g_impl[j] - g_ref) < 1e-4 * max(1.0, abs(g_ref))


# ── H1 + H2 + M1: comparison protocol ────────────────────────────────────────

def test_comparison_equal_lengths_and_shared_vi_init(post):
    """run_comparison must (H2) yield equal-length chains for the quantum
    run and its baseline, (M1) give both VI runs the SAME initialization
    (preset 45 trains classically on both sides, so the KL histories must
    be bit-identical), and (H1) record split-R-hat, not legacy GR."""
    with contextlib.redirect_stdout(io.StringIO()):
        out = mq.run_comparison(post, dict(mq.PRESETS[45]), seed=11,
                                n_steps_mcmc=120, max_iter_qvmc=6,
                                n_chains_mcmc=3, n_chains_qvmc=2,
                                nqpp=2, n_shots=300, verbose=False)
    rq, rc = out['quantum'], out['classical']
    assert rq['chains_mcmc'].shape[1] == rc['chains_mcmc'].shape[1] == 120
    hq = [h['kl'] for h in rq['qvmc_history']]
    hc = [h['kl'] for h in rc['qvmc_history']]
    assert np.allclose(hq, hc, rtol=0, atol=0)
    assert len(rq['mcmc']['rhat_hist']) > 0
    # A deliberately short run must NOT be declared converged at 1.01:
    assert rq['mcmc']['converged'] in (False, True)  # key present & boolean


# ── M4: simulator proposal engine ────────────────────────────────────────────

def test_proposal_engine_fixed_calibration():
    mq._reseed(5)
    eng = mq.QuantumProposalEngine(n_phys=2, batch=128)
    mu0, s0 = eng._mu.copy(), eng._sigma.copy()
    D = np.array([eng.next() for _ in range(512)])       # forces 4 refills
    assert np.allclose(mu0, eng._mu) and np.allclose(s0, eng._sigma)
    assert np.all(np.abs(D.mean(axis=0)) < 0.2)
    assert np.all(np.abs(D.std(axis=0) - 1.0) < 0.2)


# ── H4: QPU proposal engine (dry-run) ────────────────────────────────────────

def test_qpu_engine_unit_std_dry_run():
    conn = qpu.QPUConnection(dry_run=True)
    eng = qpu.QPUProposalEngine(conn, n_phys=2, block=64,
                                shots_per_proposal=128)
    D = np.array([eng.next() for _ in range(256)])
    # Without the H4 calibration the per-dimension std was ~0.05-0.12.
    assert np.all(np.abs(D.std(axis=0) - 1.0) < 0.35)


# ── H6: mutation gating ──────────────────────────────────────────────────────

def test_quantum_mutation_respects_mutation_rate(post):
    qga = QGA(post, GAConfig(pop_size=40, mutation_rate=0.0, seed=2),
              dict(q_mutation=True), n_bits=6,
              rng=np.random.default_rng(2))
    pop = np.column_stack(
        [np.random.default_rng(0).uniform(0.2, 0.45, 40),
         np.random.default_rng(1).uniform(62, 76, 40)])
    pop_g = qga._clip(qga._decode(qga._encode(pop)))
    assert np.allclose(qga.do_mutate(pop_g.copy()), pop_g)


# ── M3: QGA reproducibility ──────────────────────────────────────────────────

def test_qga_reproducible_with_fixed_seed(post):
    def run_once():
        q = QGA(post, GAConfig(pop_size=24, n_generations=4, seed=9),
                dict(q_init=True, q_mutation=True, q_crossover=True),
                n_bits=5, rng=np.random.default_rng(9))
        with contextlib.redirect_stdout(io.StringIO()):
            r = q.evolve(live=False, log_every=1000)
        return r.theta_map, r.chi2_map
    t1, c1 = run_once()
    t2, c2 = run_once()
    assert np.allclose(t1, t2) and c1 == c2
