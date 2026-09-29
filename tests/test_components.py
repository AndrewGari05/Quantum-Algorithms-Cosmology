"""Backends, noise, variational inference, ablation studies and diagnostics."""
import subprocess
import sys

import numpy as np
import pytest

from qablate import (
    AerBackend,
    AmplitudeEncodedMetropolis,
    BornMachineVI,
    GaussianProposal,
    Grid,
    MetropolisAcceptance,
    MetropolisHastings,
    NoiseSpec,
    RandomCircuitProposal,
    Study,
    diagnostics,
)
from qablate.cosmology import Posterior


# -- import contract ------------------------------------------------------- #
def test_core_import_is_light():
    code = ("import sys, qablate; bad = [m for m in ('qiskit', 'qiskit_aer', 'arviz', "
            "'matplotlib', 'qablate.cosmology', 'qablate.experimental') if m in sys.modules];"
            "print(bad); sys.exit(1 if bad else 0)")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_core_modules_do_not_import_cosmology():
    import pathlib
    root = pathlib.Path(__import__("qablate").__file__).parent
    for f in root.glob("*.py"):
        assert "cosmology" not in "".join(line for line in f.read_text().splitlines()
                                          if line.startswith(("import", "from"))), f


# -- backends and noise ---------------------------------------------------- #
def _bell():
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    return qc


def test_exact_probabilities_little_endian_and_no_rng_use():
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(2)
    qc.x(0)                                   # qubit 0 = 1 -> index 1
    rng = np.random.default_rng(0)
    state = rng.bit_generator.state
    p = AerBackend().probabilities(qc, np.zeros((1, 0)), rng)
    assert np.array_equal(p, [[0, 1, 0, 0]])
    assert rng.bit_generator.state == state     # exact runs draw nothing


def test_sampling_is_reproducible_from_rng():
    b = AerBackend()
    a1 = b.sample(_bell(), np.zeros((1, 0)), np.random.default_rng(3), shots=50)
    a2 = b.sample(_bell(), np.zeros((1, 0)), np.random.default_rng(3), shots=50)
    assert np.array_equal(a1, a2) and set(np.unique(a1)) <= {0, 3}


def test_readout_closed_form_matches_aer_measurement():
    spec = NoiseSpec.from_level("readout", readout_p=0.2)
    b = AerBackend(spec)
    exact = b.probabilities(_bell(), np.zeros((1, 0)), np.random.default_rng(0))[0]
    shots = b.probabilities(_bell(), np.zeros((1, 0)), np.random.default_rng(1), shots=40000)[0]
    assert np.allclose(exact, shots, atol=0.01)
    assert exact[1] == pytest.approx(0.5 * (0.2 * 0.8) * 2, abs=1e-12)


def test_device_level_places_two_qubit_gates_on_device_pairs():
    spec = NoiseSpec.from_level("FakeBrisbane")
    from qiskit_aer import AerSimulator
    t = spec.transpile(_bell(), AerSimulator(method="density_matrix"))
    edges = set(map(tuple, spec.device.coupling_map.get_edges()))
    pairs = [tuple(t.find_bit(q).index for q in ci.qubits) for ci in t.data
             if ci.operation.num_qubits == 2]
    assert pairs and all(p in edges for p in pairs)
    p = AerBackend(spec).probabilities(_bell(), np.zeros((1, 0)), np.random.default_rng(0))[0]
    assert 0.0 < p[1] + p[2] < 0.2                 # noisy but close to the Bell state


def test_ibm_runtime_backend_with_fake_device():
    pytest.importorskip("qiskit_ibm_runtime")
    from qiskit_ibm_runtime.fake_provider import FakeManilaV2

    from qablate import IBMRuntimeBackend
    b = IBMRuntimeBackend(FakeManilaV2())
    f = b.probabilities(_bell(), np.zeros((1, 0)), np.random.default_rng(0), shots=2000)[0]
    assert f[0] + f[3] > 0.8
    with pytest.raises(ValueError):
        b.probabilities(_bell(), np.zeros((1, 0)), np.random.default_rng(0))


# -- variational ----------------------------------------------------------- #
def test_parameter_shift_gradient_matches_finite_differences():
    grid = Grid([(0.0, 1.0), (0.0, 1.0)], 2)
    vi = BornMachineVI(grid, n_layers=2, optimizer="parameter-shift")
    rng = np.random.default_rng(0)
    p = grid.target(lambda x: -8 * np.sum((x - 0.4) ** 2, axis=1))
    from qablate.variational import reverse_kl
    phi = rng.normal(0, 0.5, vi.n_params)
    q = vi.distributions(phi, rng)[0]
    ps = (p + 1e-12) / np.sum(p + 1e-12)
    shifts = np.repeat(phi[None], 2 * vi.n_params, 0)
    j = np.arange(vi.n_params)
    shifts[2 * j, j] += np.pi / 2
    shifts[2 * j + 1, j] -= np.pi / 2
    qs = vi.distributions(shifts, rng)
    grad = ((qs[0::2] - qs[1::2]) / 2) @ (np.log(np.clip(q, 1e-12, None)) - np.log(ps))
    h = 1e-6
    fd = np.array([(reverse_kl(vi.distributions(phi + h * e, rng)[0], p)[0]
                    - reverse_kl(vi.distributions(phi - h * e, rng)[0], p)[0]) / (2 * h)
                   for e in np.eye(vi.n_params)])
    assert np.allclose(grad, fd, atol=1e-6)


def test_vi_fit_counts_evaluations_and_reports_correlation():
    post = Posterior("lcdm", "CC+BAO")
    grid = Grid([(0.2, 0.32), (66.0, 75.0)], 2)
    p = grid.target(post.log_prob)
    for opt in ("cobyla", "parameter-shift"):
        vi = BornMachineVI(grid, n_layers=1, optimizer=opt)
        r = vi.fit(p, max_iter=3, rng=np.random.default_rng(1))
        expected = (len(r.history) + 1 if opt == "cobyla"
                    else 3 * (1 + 2 * vi.n_params) + 1)          # QV-3: work actually done
        assert r.circuit_evaluations == expected
        assert r.correlation.shape == (2, 2) and 0 <= r.kl
    exact_mu, exact_sd, _ = grid.moments(p)
    assert np.all(exact_sd > 0)


def test_grid_rejects_empty_support():
    grid = Grid([(10.0, 11.0)], 2)
    with pytest.raises(ValueError):
        grid.target(lambda x: np.full(len(x), -np.inf))


# -- ablation -------------------------------------------------------------- #
def test_study_equivalence_checks():
    def lp(x):
        return -0.5 * np.sum(x ** 2, axis=1)

    def run(parts, rng):
        mh = MetropolisHastings(lp, 2, 3, rng=rng, **parts)
        mh.run_mcmc(np.zeros((3, 2)), 150)
        return {"chain": mh.get_chain()}

    study = Study({"proposal": (lambda: GaussianProposal(0.8), lambda: RandomCircuitProposal(0.8)),
                   "acceptance": (MetropolisAcceptance, AmplitudeEncodedMetropolis)},
                  run, equivalent=[({"proposal": True}, {"proposal": True, "acceptance": True})])
    runs = study.run(study.ladder(), seed=5)
    assert [r.label for r in runs] == ["proposal=C,acceptance=C", "proposal=Q,acceptance=C",
                                       "proposal=Q,acceptance=Q"]
    assert study.check_equivalences(runs, ["chain"]) == [
        ("proposal=Q,acceptance=C", "proposal=Q,acceptance=Q", True)]
    assert not np.array_equal(runs[0].outputs["chain"], runs[1].outputs["chain"])
    with pytest.raises(ValueError):
        study.ladder(["nonexistent"])


# -- diagnostics ----------------------------------------------------------- #
def test_diagnostics_catch_what_the_legacy_code_missed():
    rng = np.random.default_rng(0)
    same_mean_diff_spread = rng.standard_normal((4, 2000))
    same_mean_diff_spread[0] *= 3.0
    assert diagnostics.rhat(same_mean_diff_spread)[0] > 1.05        # CO-5
    frozen = rng.standard_normal((4, 500, 2))
    frozen[:, :, 1] = 0.3
    assert np.isinf(diagnostics.rhat(frozen)[1]) and diagnostics.ess(frozen)[1] == 0   # CO-6
    offset = rng.standard_normal((4, 2000))
    offset[0] += 3.0
    assert diagnostics.ess(offset)[0] < 100                          # CO-7


def test_autocorr_matches_emcee():
    emcee = pytest.importorskip("emcee")
    rng = np.random.default_rng(1)
    x = np.zeros((4, 5000))
    for t in range(1, 5000):
        x[:, t] = 0.9 * x[:, t - 1] + rng.standard_normal(4)
    ours = diagnostics.autocorr_time(x)[0]
    theirs = emcee.autocorr.integrated_time(np.swapaxes(x, 0, 1)[:, :, None], quiet=True)[0]
    assert ours == pytest.approx(theirs, rel=1e-6)
