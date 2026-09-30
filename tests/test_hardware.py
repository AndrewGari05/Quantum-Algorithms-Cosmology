"""Hardware protocol: one compiled circuit at every location, SPSA, timing, budget."""
import os
import sys

import numpy as np
import pytest

pytest.importorskip("qiskit_ibm_runtime")
from qiskit_ibm_runtime.fake_provider import FakeFez  # noqa: E402

from qablate import AerBackend, BornMachineVI, Grid  # noqa: E402
from qablate.circuits import hardware_efficient_ansatz, random_proposal_circuit  # noqa: E402
from qablate.hardware import (  # noqa: E402
    DeviceBackend,
    DeviceCompiler,
    QuantumBudgetExceeded,
    _fingerprint,
)
from qablate.variational import reverse_kl  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


@pytest.fixture(scope="module")
def compiler():
    return DeviceCompiler(FakeFez(), seed_transpiler=1)


def test_identical_recipes_share_one_isa_circuit(compiler):
    a, b = hardware_efficient_ansatz(4, 3), hardware_efficient_ansatz(4, 3)
    assert _fingerprint(a) == _fingerprint(b) != _fingerprint(hardware_efficient_ansatz(4, 2))
    isa_a, _, _ = compiler.compile(a)
    isa_b, _, dt = compiler.compile(b)
    assert isa_a is isa_b and dt == 0.0


def test_ideal_location_matches_exact_probabilities(compiler):
    qc = hardware_efficient_ansatz(4, 3)
    rng = np.random.default_rng(0)
    phi = rng.uniform(0, 2 * np.pi, (3, qc.num_parameters))
    exact = AerBackend().probabilities(qc, phi, rng)
    f = DeviceBackend(compiler, "ideal").probabilities(qc, phi, np.random.default_rng(1),
                                                      shots=40000)
    assert np.max(0.5 * np.abs(f - exact).sum(axis=1)) < 0.03    # routing kept the logic


def test_backends_never_draw_from_the_callers_stream(compiler):
    qc = random_proposal_circuit(2, 3)
    pv = np.random.default_rng(0).uniform(0, 6, (4, qc.num_parameters))
    for loc in ("ideal", "noisy"):
        rng = np.random.default_rng(5)
        before = rng.bit_generator.state
        DeviceBackend(compiler, loc, seed=1).probabilities(qc, pv, rng, shots=16)
        assert rng.bit_generator.state == before


def test_device_path_with_a_fake_device_is_reproducible(compiler):
    qc = random_proposal_circuit(2, 3)
    pv = np.random.default_rng(0).uniform(0, 6, (3, qc.num_parameters))
    runs = []
    for _ in range(2):
        b = DeviceBackend(compiler, "device", seed=7)
        runs.append(b.probabilities(qc, pv, np.random.default_rng(0), shots=64))
        s = b.sample(qc, pv[:1], np.random.default_rng(0), shots=5)
        assert s.shape == (1, 5) and b.log[-1].location == "device"
    assert runs[0].shape == (3, 4) and np.allclose(runs[0].sum(axis=1), 1.0)
    assert np.array_equal(runs[0], runs[1])


def test_billed_time_fails_closed_when_ibm_has_not_finalized_it():
    from qablate.hardware import _ibm_times

    class Job:
        def metrics(self):
            return {"timestamps": {"created": "2026-01-01T00:00:00Z",
                                   "running": "2026-01-01T00:01:30Z"},
                    "usage": {"status": "pending", "qpu_charge_time_seconds": None}}

        def usage_estimation(self):
            return {"quantum_seconds": 4.2}

    queue, _, quantum, estimated = _ibm_times(Job(), result=None, polls=1)
    assert queue == 90.0 and quantum == 4.2 and estimated

    class Final(Job):
        def metrics(self):
            return {"usage": {"status": "completed", "qpu_charge_time_seconds": 3}}

    assert _ibm_times(Final(), result=None, polls=1)[2:] == (3.0, False)


def test_parameterless_circuit_rejects_several_rows(compiler):
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(1)
    qc.h(0)
    with pytest.raises(ValueError):
        DeviceBackend(compiler, "ideal").probabilities(qc, np.zeros((2, 0)),
                                                       np.random.default_rng(0), shots=4)


def test_device_budget_guard_refuses_before_submitting(compiler):
    b = DeviceBackend(compiler, "device", max_quantum_seconds=0.0)
    with pytest.raises(QuantumBudgetExceeded):
        b.probabilities(random_proposal_circuit(2, 3), np.zeros((1, 17)),
                        np.random.default_rng(0), shots=8)
    assert b.log == []


def test_device_backend_requires_shots(compiler):
    with pytest.raises(ValueError):
        DeviceBackend(compiler, "ideal").probabilities(hardware_efficient_ansatz(2, 1),
                                                       np.zeros((1, 6)), np.random.default_rng(0))


# -- SPSA -------------------------------------------------------------------- #
def _toy_grid():
    grid = Grid([(-1.0, 1.0), (-1.0, 1.0)], 2)
    p = grid.target(lambda x: -0.5 * np.sum((x / 0.5) ** 2, axis=1))
    return grid, p


def test_spsa_reduces_kl_and_records_iterates():
    grid, p = _toy_grid()
    vi = BornMachineVI(grid, n_layers=2, optimizer="spsa", spsa_gains=(0.3, 0.1))
    r = vi.fit(p, 150, np.random.default_rng(0))
    q0 = AerBackend().probabilities(vi.circuit, r.phi_history[:1], np.random.default_rng(0))
    assert r.phi_history.shape == (150, vi.n_params)
    assert r.kl < reverse_kl(q0, p)[0]
    assert r.circuit_evaluations == 2 * 150 + 1


def test_shots_only_with_spsa_and_initial_angles():
    grid, p = _toy_grid()
    with pytest.raises(ValueError):
        BornMachineVI(grid, optimizer="cobyla", shots=100)
    vi = BornMachineVI(grid, n_layers=1, optimizer="spsa", shots=256)
    phi0 = np.full(vi.n_params, 0.3)
    r = vi.fit(p, 1, np.random.default_rng(0), initial=phi0)
    assert np.all(np.abs(r.phi_history[0] - phi0) < 1.0)


# -- protocol ---------------------------------------------------------------- #
def test_protocol_plan_matches_the_jobs_submitted(tmp_path):
    from thesis import hardware as hw
    cfg = hw.HardwareConfig(vi_iters=3, vi_warm_evals=100, vi_warm_restarts=1, mcmc_steps=60,
                            mcmc_chains=2, mcmc_block=64, mcmc_calibration=32,
                            ga_generations=2, ga_population=12, reference_steps=500,
                            reference_chains=4)
    rows = hw.run(cfg, FakeFez(), locations=["ideal"], algorithms=["vi", "mcmc", "genetic"],
                  out=str(tmp_path), log=lambda *a: None)
    from qablate.cosmology import Posterior
    plan = hw.plan(cfg, Posterior("lcdm", "CC+BAO"))
    got = {r["algorithm"]: r for r in rows if r["location"] == "ideal"}
    for alg in ("vi", "mcmc", "genetic"):
        assert got[alg]["jobs"] == plan[alg]["jobs"]
    for alg in ("vi", "mcmc"):
        assert got[alg]["shots_total"] == plan[alg]["shots"]
    assert got["genetic"]["shots_total"] <= plan["genetic"]["shots"]
    for f in ("results.csv", "jobs.csv", "vi_trace.csv", "circuits.json", "summary.md"):
        assert (tmp_path / f).exists()


def test_old_usage_schema_and_conservative_floor():
    from qablate.hardware import _ibm_times, conservative_quantum_seconds

    class Old:
        def metrics(self):
            return {"usage": {"quantum_seconds": 5}}

    assert _ibm_times(Old(), result=None, polls=1)[2:] == (5.0, False)

    class Unknown:
        def metrics(self):
            return {"usage": {"status": "pending"}}

        def usage_estimation(self):
            return {"quantum_seconds": None}

    q, est = _ibm_times(Unknown(), result=None, polls=1, n_executions=10_000)[2:]
    assert est and q == conservative_quantum_seconds(10_000) == 2.0 + 2.5


def test_reseed_makes_an_algorithm_independent_of_earlier_jobs(compiler):
    qc = random_proposal_circuit(2, 3)
    pv = np.random.default_rng(0).uniform(0, 6, (2, qc.num_parameters))
    a, b = DeviceBackend(compiler, "noisy", seed=0), DeviceBackend(compiler, "noisy", seed=0)
    a.probabilities(qc, pv, np.random.default_rng(0), shots=32)       # an earlier "algorithm"
    for be in (a, b):
        be.reseed(123)
    fa = a.probabilities(qc, pv, np.random.default_rng(0), shots=32)
    fb = b.probabilities(qc, pv, np.random.default_rng(0), shots=32)
    assert np.array_equal(fa, fb)


def test_miller_madow_removes_the_plugin_bias():
    grid, p = _toy_grid()
    vi = BornMachineVI(grid, n_layers=1, optimizer="spsa", shots=200)
    q = np.zeros((1, 16))
    q[0, :5] = 0.2
    plug = BornMachineVI(grid, n_layers=1, optimizer="spsa", shots=200, kl_estimator="plugin")
    assert np.isclose(plug.kl(q, p)[0] - vi.kl(q, p)[0], 4 / 400)


def test_a_zero_charge_is_never_final():
    from qablate.hardware import _ibm_times

    class Zero:
        def metrics(self):
            return {"usage": {"quantum_seconds": 0}}

        def usage_estimation(self):
            return {}

    q, est = _ibm_times(Zero(), result=None, polls=1, n_executions=0)[2:]
    assert est and q == 2.0
