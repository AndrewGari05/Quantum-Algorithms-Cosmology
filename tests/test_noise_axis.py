"""
test_noise_axis.py — Regressions for the second ablation axis (NISQ noise).

Covers `cosmo_noise` and the guard this axis adds to
`cosmo_core.make_simulator`. Every test that corresponds to a bug that was
found carries its tag ([N1], [B-RO]) so that the reason for the test is not
lost.

Tests marked `qiskit` are skipped without Qiskit/Aer, like the rest of the
suite.
"""

import os
import sys
import inspect
import tempfile

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import cosmo_noise as cn                                     # noqa: E402


# =============================================================================
# Level names and resolution (no Aer needed)
# =============================================================================

def test_canonical_level_normalizes_backend_variants():
    """The three spellings of a backend must collapse to ONE CSV label.

    If they do not collapse, the noise axis generates duplicate rows that
    look like different rungs.
    """
    assert cn.canonical_level('FakeBrisbane') == 'fakebrisbane'
    assert cn.canonical_level('fake_brisbane') == 'fakebrisbane'
    assert cn.canonical_level('  FAKE-BRISBANE ') == 'fakebrisbane'
    assert cn.canonical_level(None) == 'none'
    assert cn.canonical_level('') == 'none'
    for lvl in cn.NAMED_LEVELS:
        assert cn.canonical_level(lvl.upper()) == lvl


def test_unknown_level_raises():
    """A misspelled level must fail loudly, not fall back to 'none'."""
    with pytest.raises(ValueError):
        cn.NoiseSpec.from_level('no_existe_este_backend')


def test_qubit_ceiling_is_derived_from_ram():
    """[REV] The noisy ceiling scales with RAM, it is not a constant.

    The first version of the axis hard-fixed it at 13, arguing a time limit.
    That was a bad generalization drawn from a 7 GB machine: on a large node
    with no hurry the limiting factor is memory again, and that does relax.
    This test pins that it now grows monotonically with RAM.
    """
    prev = 0
    for gb in (7, 64, 256, 1024):
        c = cn.noisy_qubit_ceiling(mem_mb=gb * 1024)
        assert c >= prev, "the ceiling cannot drop when given more RAM"
        prev = c
    assert cn.noisy_qubit_ceiling(mem_mb=7 * 1024) < \
        cn.noisy_qubit_ceiling(mem_mb=1024 * 1024)

    # without a RAM figure it falls back to the conservative default
    assert cn.noisy_qubit_ceiling() == cn.DEFAULT_NOISY_QUBITS
    # the user's cap is still an upper bound
    assert cn.noisy_qubit_ceiling(8, mem_mb=1024 * 1024) == 8
    assert cn.noisy_density_bytes(13) == 2 ** 26 * 16


def test_parameter_shift_batch_is_what_rules():
    """With quantum training the ceiling is set by the batch, not a single rho.

    `2 * n_phi` density matrices at once: at 12 qubits that is ~42 GB versus
    256 MB for a single one. Ignoring it would give an overly optimistic
    ceiling and the task would die from OOM right at the 67% and 100% rungs.
    """
    assert cn.param_shift_batch_factor(12) > 100
    single = cn.noisy_density_bytes(12)
    batch = cn.noisy_density_bytes(12, cn.param_shift_batch_factor(12))
    assert batch == single * cn.param_shift_batch_factor(12)
    assert batch / 2 ** 30 > 40          # ~42 GB

    with_qt = cn.noisy_qubit_ceiling(mem_mb=95 * 1024, quantum_training=True)
    without_qt = cn.noisy_qubit_ceiling(mem_mb=95 * 1024, quantum_training=False)
    assert with_qt < without_qt, "the batch has to clamp more than a single rho"


# =============================================================================
# Ideal rung: must be EXACTLY the route that predates the noise axis
# =============================================================================

def test_none_is_ideal_and_touches_nothing():
    """`--noise none` has to reproduce the ideal route bit for bit.

    It is the condition that makes the ideal column a baseline and not an
    approximation: same simulation method, no noise model, and an
    `apply_readout` that returns the SAME object without copying or rounding.
    """
    spec = cn.NoiseSpec.from_level('none')
    assert spec.is_ideal
    assert not spec.has_readout
    assert spec.full_model() is None
    assert spec.simulator_kwargs(counts_route=False) == {
        'method': 'statevector'}
    assert spec.simulator_kwargs(counts_route=True) == {
        'method': 'statevector'}

    p = np.array([0.1, 0.2, 0.3, 0.4])
    assert spec.apply_readout(p, 2) is p


# =============================================================================
# [N1] The make_simulator guard
# =============================================================================

@pytest.mark.qiskit
def test_n1_statevector_with_noise_raises():
    """[N1] statevector + noise_model silently returns the IDEAL result.

    Aer gives no warning: the run completes, reports success and hands back
    noise-free numbers. A whole noisy campaign would come back clean and
    plausible. The combination has to be impossible to build.
    """
    from cosmo_core import make_simulator

    spec = cn.NoiseSpec.from_level('full')
    with pytest.raises(ValueError, match=r'\[N1\]'):
        make_simulator(method='statevector',
                       noise_model=spec.full_model())


@pytest.mark.qiskit
def test_simulator_kwargs_never_produces_the_forbidden_combination():
    """No rung can by itself produce the pair that [N1] forbids."""
    from cosmo_core import make_simulator

    for level in ('none', 'readout', 'full'):
        spec = cn.NoiseSpec.from_level(level)
        for counts_route in (False, True):
            kwargs = spec.simulator_kwargs(counts_route)
            if kwargs.get('noise_model') is not None:
                assert kwargs['method'] == 'density_matrix'
            make_simulator(**kwargs)          # must not raise


# =============================================================================
# [B-RO] The analytic readout map
# =============================================================================

def _diag_rho(qc, spec, n):
    """diag(rho) of the circuit under the rung's GATE noise."""
    from qiskit import transpile
    from cosmo_core import make_simulator

    sim = make_simulator(**spec.simulator_kwargs(counts_route=False))
    q = qc.copy()
    q.save_density_matrix()
    rho = np.asarray(
        sim.run(transpile(q, sim, optimization_level=0)).result()
        .data(0)['density_matrix'])
    p = np.real(np.diag(rho))
    return p / p.sum()


def _counts_probs(qc, spec, n, shots, seed=5):
    """Probabilities measured with the FULL model (Aer applies the readout)."""
    from qiskit import transpile
    from cosmo_core import make_simulator

    sim = make_simulator(seed_simulator=seed,
                         **spec.simulator_kwargs(counts_route=True))
    q = qc.copy()
    q.measure_all()
    counts = sim.run(transpile(q, sim, optimization_level=0),
                     shots=shots).result().get_counts()
    out = np.zeros(2 ** n)
    for key, val in counts.items():
        out[int(key, 2)] = val / shots
    return out


def _sample_circuit(n, seed):
    from qiskit import QuantumCircuit
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(n)
    for i in range(n):
        qc.ry(float(rng.uniform(0, np.pi)), i)
    for i in range(n - 1):
        qc.cx(i, i + 1)
    return qc


@pytest.mark.qiskit
@pytest.mark.parametrize('n', [1, 2, 3])
def test_b_ro_analytic_readout_matches_counts(n):
    """[B-RO] The readout channel must reach ALL qubits, not just qubit 0.

    An `add_all_qubit_readout_error` shows up in `to_dict()` WITHOUT a
    `gate_qubits` key. Interpreting that absence as `[[0]]` degraded the
    channel to a single qubit: the analytic map noised n qubits and Aer only
    one. The symptom was silent — exact agreement at n=1 and growing
    divergence with n and with p — so the test sweeps n>1 on purpose.
    """
    shots = 400_000
    spec = cn.NoiseSpec.from_level('readout', readout_p=0.05)
    qc = _sample_circuit(n, seed=100 + n)

    predicted = spec.apply_readout(_diag_rho(qc, spec, n), n)
    observed = _counts_probs(qc, spec, n, shots)

    # 5 sigma of shot noise, with margin for the maximum over 2^n cells.
    tol = 5 * 0.5 / np.sqrt(shots) + 1e-3
    assert np.max(np.abs(predicted - observed)) < tol


@pytest.mark.qiskit
def test_analytic_readout_with_non_uniform_channel():
    """A real backend brings one matrix per qubit; the map must respect which.

    FakeBrisbane's flip probabilities differ between qubits, so any qubit
    permutation or transposition of the matrices breaks this test (unlike
    the uniform channel, which is blind to both).
    """
    n, shots = 3, 400_000
    spec = cn.NoiseSpec.from_level('FakeBrisbane')
    mats = [spec.readout_matrix(q) for q in range(n)]
    assert all(m is not None for m in mats)
    assert len({round(float(m[1, 0]), 6) for m in mats}) > 1, \
        "the channel must be non-uniform for the test to have power"

    qc = _sample_circuit(n, seed=777)
    predicted = spec.apply_readout(_diag_rho(qc, spec, n), n)
    observed = _counts_probs(qc, spec, n, shots)
    tol = 5 * 0.5 / np.sqrt(shots) + 1e-3
    assert np.max(np.abs(predicted - observed)) < tol


def test_apply_readout_preserves_normalization():
    """A stochastic channel neither creates nor destroys probability."""
    rng = np.random.default_rng(3)
    spec = cn.NoiseSpec.from_level('readout', readout_p=0.07)
    for n in (1, 2, 4):
        p = rng.random(2 ** n)
        p /= p.sum()
        out = spec.apply_readout(p, n)
        assert out.shape == p.shape
        assert np.isclose(out.sum(), 1.0)
        assert np.all(out >= 0.0)


def test_apply_readout_accepts_batches():
    """The batched route must match the individual route row by row."""
    rng = np.random.default_rng(4)
    spec = cn.NoiseSpec.from_level('readout', readout_p=0.04)
    n = 3
    batch = rng.random((5, 2 ** n))
    batch /= batch.sum(axis=1, keepdims=True)
    out = spec.apply_readout(batch, n)
    assert out.shape == batch.shape
    for k in range(batch.shape[0]):
        assert np.allclose(out[k], spec.apply_readout(batch[k], n))


def test_apply_readout_rejects_wrong_length():
    """A width mismatch must fail, not reinterpret the bits."""
    spec = cn.NoiseSpec.from_level('readout')
    with pytest.raises(ValueError):
        spec.apply_readout(np.ones(8) / 8, 2)


# =============================================================================
# The FAITHFUL identity survives the change of readout
# =============================================================================

@pytest.mark.qiskit
def test_rho00_reproduces_metropolis_acceptance():
    """rho[0,0] == |psi_0|^2 == min(1, e^Delta), without noise.

    It is the project's faithfulness claim. The noise axis changes HOW the
    acceptance is read (from amplitude to rho), so one has to show that the
    change of readout introduces no error of its own before attributing any
    later deviation to noise.
    """
    from qiskit import QuantumCircuit, transpile
    from qiskit.circuit import ParameterVector
    from cosmo_core import make_simulator

    deltas = np.array([-6.0, -3.0, -1.0, -0.3, -0.05, 0.0, 0.5, 2.0])
    a_exact = np.minimum(1.0, np.exp(deltas))
    thetas = 2.0 * np.arccos(np.sqrt(np.clip(a_exact, 1e-12, 1.0)))

    par = ParameterVector('t', 1)
    qc = QuantumCircuit(1)
    qc.ry(par[0], 0)
    qc.save_density_matrix()

    spec = cn.NoiseSpec.from_level('none')
    sim = make_simulator(method='density_matrix')
    isa = transpile(qc, sim)
    res = sim.run([isa.assign_parameters({par[0]: float(t)})
                   for t in thetas]).result()
    rho00 = np.array([
        float(np.real(np.asarray(res.data(k)['density_matrix'])[0, 0]))
        for k in range(len(thetas))])

    assert np.max(np.abs(rho00 - a_exact)) < 1e-12
    assert spec.apply_readout(rho00, 1) is rho00


@pytest.mark.qiskit
def test_readout_alone_does_not_move_rho():
    """Readout error is a CLASSICAL channel: it cannot touch the state.

    This test pins the raison d'etre of the analytic map. If some day rho DID
    move with the `readout` rung, it would mean the readout noise is being
    applied twice (once in Aer and again in `apply_readout`).
    """
    from qiskit import QuantumCircuit, transpile
    from cosmo_core import make_simulator

    qc = _sample_circuit(2, seed=11)
    qc.save_density_matrix()

    ideal = cn.NoiseSpec.from_level('none')
    ro = cn.NoiseSpec.from_level('readout', readout_p=0.05)

    out = []
    for spec in (ideal, ro):
        kwargs = dict(spec.simulator_kwargs(counts_route=False))
        kwargs['method'] = 'density_matrix'      # the ideal one uses statevector
        sim = make_simulator(**kwargs)
        rho = np.asarray(sim.run(transpile(qc, sim, optimization_level=0))
                         .result().data(0)['density_matrix'])
        out.append(np.real(np.diag(rho)))
    assert np.max(np.abs(out[0] - out[1])) < 1e-12


@pytest.mark.qiskit
def test_b_recon_model_is_not_rebuilt():
    """[B-RECON] The calibrated model must reach Aer WITHOUT a round trip.

    Rebuilding it with `NoiseModel.from_dict()` (besides being deprecated
    since qiskit-aer 0.15) is lossy: against FakeBrisbane the rebuilt rho
    differs from the original by ~6e-4 at 3 qubits. This test pins that the
    object handed to the simulator is THE SAME one Aer produced.
    """
    spec = cn.NoiseSpec.from_level('FakeBrisbane')
    assert spec.full_model() is spec.source_model
    for counts_route in (False, True):
        kwargs = spec.simulator_kwargs(counts_route)
        assert kwargs['noise_model'] is spec.source_model


# =============================================================================
# Readout routes of cosmo_modular_quantum under the axis
# =============================================================================

@pytest.mark.qiskit
def test_none_rung_is_bit_identical_and_reversible():
    """`--noise none` must keep giving EXACTLY the published numbers.

    It also covers the invalidation of the `_HAD` cache: after going through
    noisy rungs, returning to `none` has to rebuild the ideal simulator. If
    the cache were not invalidated, the return would keep using the noisy
    simulator and the results would belong to the wrong level with no
    warning at all.
    """
    import cosmo_modular_quantum as mq

    lp_cur = np.array([-10.0, -10.0, -10.0, -10.0])
    lp_prop = np.array([-16.0, -10.5, -10.0, -8.0])

    mq.set_noise(cn.NoiseSpec.from_level('none'))
    ref = mq.hadamard_accept_log_batch(lp_cur, lp_prop)

    exact = np.minimum(1.0, np.exp(lp_prop - lp_cur))
    assert np.max(np.abs(np.exp(ref) - exact)) < 1e-11

    # the batched and the individual versions share the cache: they must agree
    single = np.array([mq.hadamard_accept_log(a, b)
                       for a, b in zip(lp_cur, lp_prop)])
    assert np.array_equal(single, ref)

    for level in ('readout', 'full'):
        mq.set_noise(cn.NoiseSpec.from_level(level))
        mq.hadamard_accept_log_batch(lp_cur, lp_prop)

    mq.set_noise(cn.NoiseSpec.from_level('none'))
    assert np.array_equal(mq.hadamard_accept_log_batch(lp_cur, lp_prop), ref)


@pytest.mark.qiskit
def test_readout_does_move_the_acceptance():
    """The `readout` column of the axis can NOT come out equal to the ideal one.

    It is the exact symptom of bug [B-RO] and of reading `rho[0,0]` without
    applying the channel in closed form. For the acceptance there is an
    analytic answer: with symmetric flip p, P'(0) = (1-p)A + p(1-A), so an
    acceptance of A=1 must drop to exactly 1-p.
    """
    import cosmo_modular_quantum as mq

    p = 0.03
    lp_cur = np.array([-10.0])
    lp_prop = np.array([-8.0])            # Delta>0 => A=1 exactly

    mq.set_noise(cn.NoiseSpec.from_level('readout', readout_p=p))
    a_noisy = float(np.exp(mq.hadamard_accept_log_batch(lp_cur, lp_prop)[0]))
    mq.set_noise(cn.NoiseSpec.from_level('none'))

    assert abs(a_noisy - (1.0 - p)) < 1e-9


@pytest.mark.qiskit
def test_proposal_invariance_under_uniform_readout():
    """[INVARIANCE] Calibration absorbs 100% of uniform readout.

    A symmetric, uniform readout channel rescales <Z_q> by (1-2p), and the
    calibration to unit std divides out that scalar, so the calibrated shift
    is EXACTLY the same. This test pins that invariance so that the
    `readout` column of the proposal row, which comes out identical to the
    ideal one, is never mistaken for a regression.

    GATE noise, however, must survive: that is what separates this
    legitimate invariance from bug [B-RO].
    """
    import cosmo_modular_quantum as mq

    def calibrated_block(spec, n=32):
        mq.set_noise(spec, 'counts')
        mq._reseed(11)
        eng = mq.QuantumProposalEngine(3, batch=n, n_calib=n)
        mq._reseed(99)
        return np.array([eng.next() for _ in range(n)])

    ideal = calibrated_block(cn.NoiseSpec.from_level('none'))
    for p in (0.01, 0.20):
        noisy = calibrated_block(cn.NoiseSpec.from_level('readout',
                                                         readout_p=p))
        assert np.max(np.abs(noisy - ideal)) < 1e-12, \
            f"uniform readout p={p} should be fully absorbed"

    gates = calibrated_block(cn.NoiseSpec.from_level('full'))
    assert np.max(np.abs(gates - ideal)) > 1e-3, \
        "gate noise must NOT be absorbed by the calibration"
    mq.set_noise(cn.NoiseSpec.from_level('none'))


@pytest.mark.qiskit
def test_counts_rule_matches_the_qpu_pipeline():
    """The counts route must be the SAME rule that runs on hardware.

    `qpu_cosmo_samplers.py` is kept untouched for the QPU runs, so the rule
    <Z_q> = 1 - 2 P(q=1) is implemented twice. This test confronts them on
    the same counts so that they cannot silently diverge: a divergence here
    would break simulator-QPU comparability, which is the whole point of the
    module.
    """
    import cosmo_modular_quantum as mq
    from qpu_cosmo_samplers import QPUProposalEngine

    rng = np.random.default_rng(2)
    n_qubits, d, shots = 4, 3, 100_000
    probs = rng.random(2 ** n_qubits)
    probs /= probs.sum()

    counts = {format(i, f'0{n_qubits}b'): int(round(p * shots))
              for i, p in enumerate(probs) if p * shots >= 1}
    total = sum(counts.values())

    # --- rule of the QPU module (without building the class: no connection) ---
    qpu = QPUProposalEngine.__new__(QPUProposalEngine)
    qpu.n_qubits, qpu.d = n_qubits, d
    z_qpu = qpu._counts_to_shift(counts)

    # --- simulator rule, on the SAME frequencies ---
    freq = np.zeros(2 ** n_qubits)
    for bits, c in counts.items():
        freq[int(bits, 2)] = c / total
    idx = np.arange(2 ** n_qubits)
    z_sim = np.array([1.0 - 2.0 * float(freq @ ((idx >> q) & 1))
                      for q in range(d)])

    assert np.allclose(z_qpu, z_sim, atol=1e-12), (
        f"the two implementations of <Z_q> diverged: "
        f"QPU={z_qpu}, simulator={z_sim}")


# =============================================================================
# The noise axis as a task dimension of the HPC runner
# =============================================================================

def _runner_args(extra):
    """Runner namespace with the `profile` attribute already resolved."""
    import cosmo_hpc_runner as r
    args = r.build_parser().parse_args(['--models', 'lcdm', 'cpl'] + extra)
    args.profile = not getattr(args, 'no_profile', False)
    return args


def test_runner_without_noise_keeps_output_paths():
    """An ideal run must produce EXACTLY the usual folders.

    If the axis added a 'noise-none' suffix to every task, the previous CSVs
    and figures would no longer line up with the new ones and every
    historical comparison would require renaming by hand. The rung only
    enters the name when there really is more than one.
    """
    import cosmo_hpc_runner as r

    args = _runner_args(['--nqpp', '3'])
    tasks = r.build_tasks(args, '/tmp/m', 18, [], q_ceiling_genetic=24,
                          noise_levels=['none'])
    assert tasks, "it should generate tasks"
    for t in tasks:
        assert 'noise' not in t.outdir
        assert 'noise' not in t.name
        assert t.noise == 'none'
        assert '--noise' not in t.argv


def test_runner_generates_the_two_dimensional_matrix():
    """--noise-sweep must multiply tasks like one more dimension.

    It is the design requirement: the noise level is treated just like nqpp
    and n_bits, so that a single invocation produces the full matrix.
    """
    import cosmo_hpc_runner as r

    levels = ['none', 'readout', 'full']
    # --no-noise-control to isolate the matrix; the control column has its
    # own test.
    args = _runner_args(['--nqpp', '3', '--no-noise-control'])
    base = r.build_tasks(args, '/tmp/m', 18, [], q_ceiling_genetic=24,
                         noise_levels=['none'])
    grid = r.build_tasks(args, '/tmp/m', 18, [], q_ceiling_genetic=24,
                         noise_levels=levels, noisy_q_ceiling=13,
                         noisy_q_ceiling_genetic=13)
    assert len(grid) == len(base) * len(levels)
    assert {t.noise for t in grid} == set(levels)
    for t in grid:
        if t.noise != 'none':
            assert '--noise' in t.argv
            assert t.argv[t.argv.index('--noise') + 1] == t.noise
    # unique names: two rungs cannot write into the same folder
    assert len({t.outdir for t in grid}) == len(grid)


def test_runner_applies_noise_ceiling_per_model():
    """The noisy ceiling must clamp the heavy models, not the light ones.

    CPL (d=4) with nqpp=4 is 16 qubits: above the ceiling of 13, so it must
    be lowered. LCDM (d=2) with nqpp=4 is 8 and must stay the same. It is the
    same per-model clamp the other two ceilings already did.
    """
    import cosmo_hpc_runner as r

    args = _runner_args(['--nqpp', '4'])
    tasks = r.build_tasks(args, '/tmp/m', 18, [], q_ceiling_genetic=24,
                          noise_levels=['full'], noisy_q_ceiling=13,
                          noisy_q_ceiling_genetic=13)
    by_model = {t.model: t for t in tasks if t.grid_kind == 'nqpp'}
    assert by_model['lcdm'].total_qubits == 8      # 2*4, fits
    assert by_model['cpl'].total_qubits <= 13      # 4*4=16 -> clamped


def test_runner_noisy_memory_model_is_additive():
    """The density matrix is ADDED to the task's model, it does not replace it.

    A noisy samplers task still builds its likelihood grid in addition to
    rho; estimating only rho would underestimate it.
    """
    import cosmo_hpc_runner as r
    import cosmo_noise as cnz

    for q in (8, 10, 12):
        ideal = r.estimate_qubits_and_mem(q, 'nqpp', noisy=False)
        noisy = r.estimate_qubits_and_mem(q, 'nqpp', noisy=True)
        assert noisy > ideal
        factor = cnz.param_shift_batch_factor(q)
        assert abs((noisy - ideal)
                   - cnz.noisy_density_bytes(q, factor) / 1e6) < 1e-6


def test_runner_noise_ceiling_scales_with_ram_but_clamps():
    """[REV] The noisy ceiling grows with RAM, and stays far below the ideal one.

    Two things at once: that more RAM grants more qubits (which the previous
    version denied), and that even so the noise axis clamps hard relative to
    the ideal run — at 95 GB, 12 qubits against the 18+ that the statevector
    comfortably allows.
    """
    import cosmo_hpc_runner as r

    c_small = r.qubit_ceiling(None, 7 * 1024, 'nqpp', noisy=True)
    c_big = r.qubit_ceiling(None, 512 * 1024, 'nqpp', noisy=True)
    assert c_big > c_small, "more RAM must grant more qubits"

    mem95 = 95 * 1024
    assert r.qubit_ceiling(None, mem95, 'nqpp', noisy=True) < \
        r.qubit_ceiling(None, mem95, 'nqpp'), \
        "the noise axis must still clamp relative to the ideal"

    # the genetic one is looser: it has no parameter-shift batch
    assert r.qubit_ceiling(None, mem95, 'n_bits', noisy=True) > \
        r.qubit_ceiling(None, mem95, 'nqpp', noisy=True)


# =============================================================================
# Noisy twin of the QPU pipeline
# =============================================================================

@pytest.mark.qiskit
def test_twin_does_not_modify_the_hardware_module():
    """`qpu_cosmo_samplers` must remain UNTOUCHED after using the twin.

    That module is the one sent to real hardware. The twin monkeypatches it
    to replace the connection, so the only thing separating "reusing the
    pipeline" from "corrupting it for the rest of the process" is that the
    patch gets undone. This test pins that restoration.
    """
    import qpu_cosmo_samplers as hw
    import qpu_noisy_simulation as twin

    original = hw.QPUConnection
    rc = twin.main(['--model', 'lcdm', '--method', 'qmcmc', '--steps', '20',
                    '--chains', '2', '--block', '8', '--shots', '256',
                    '--noise', 'none', '--outdir', '/tmp/_twin_restore',
                    '--no-plot'])
    assert rc == 0
    assert hw.QPUConnection is original


@pytest.mark.qiskit
def test_twin_honors_the_run_pub_interface():
    """The local connection must fulfil the contract of the hardware one.

    B bindings -> B counts dictionaries, in order, with the frequencies
    summing to the requested shots. If this contract breaks, the pipeline
    would consume it anyway and produce silently misordered shifts.
    """
    import qpu_cosmo_samplers as hw
    import qpu_noisy_simulation as twin

    spec = cn.NoiseSpec.from_level('full')
    conn = twin.LocalNoisyConnection(noise=spec, shots=512, seed=3)

    qc = hw.build_proposal_circuit(3, n_layers=2)
    isa = conn.transpile_isa(qc)
    b = 5
    rng = np.random.default_rng(0)
    pv = rng.uniform(0, 2 * np.pi, size=(b, isa.num_parameters))

    out = conn.run_pub(isa, pv, shots=512)
    assert isinstance(out, list) and len(out) == b
    for counts in out:
        assert isinstance(counts, dict)
        assert sum(counts.values()) == 512
        assert all(len(k.replace(' ', '')) == 3 for k in counts)
    conn.close()


@pytest.mark.qiskit
def test_twin_rejects_widths_above_the_ceiling():
    """An infeasible width must fail BEFORE starting, not after hours.

    CPL with nqpp=6 is 24 qubits: the density matrix would be ~4 PB. The twin
    has to detect it and exit with code 1.
    """
    import qpu_noisy_simulation as twin

    rc = twin.main(['--model', 'cpl', '--method', 'qvmc', '--iters', '1',
                    '--nqpp', '6', '--noise', 'full',
                    '--outdir', '/tmp/_twin_cap', '--no-plot'])
    assert rc == 1


def test_twin_strips_the_noise_axis_flags():
    """The hardware parser does not know --noise*: they must be removed from argv.

    If they slipped through, `qpu.main` would abort with "unrecognized
    arguments" and the twin would be unusable with any explicit rung.
    """
    import qpu_noisy_simulation as twin

    argv = ['--model', 'lcdm', '--noise', 'full', '--steps', '10',
            '--noise-readout-p', '0.05', '--noise-seed', '7', '--dry-run']
    out = twin._strip_noise_flags(argv)
    assert out == ['--model', 'lcdm', '--steps', '10', '--dry-run']
    # --flag=value form
    assert twin._strip_noise_flags(['--noise=full', '--model', 'lcdm']) == \
        ['--model', 'lcdm']


@pytest.mark.parametrize('flag,expected', [([], True), (['--no-noise-control'], False)])
def test_runner_generates_the_control_column(flag, expected):
    """[NOISE-CONTROL] Sweeping noise must bring the ideal-by-counts column.

    Without it, the ideal rung reads amplitudes and the noisy ones read
    counts, so the comparison mixes noise with the change of operator. In
    the first run of the axis that made it look as if readout noise improves
    sampling (sigma 0.0208 -> 0.0176, ESS 101 -> 141), when in reality it
    does not touch the proposal at all.

    That is why the control is generated AUTOMATICALLY, and disabling it has
    to be explicit.
    """
    import cosmo_hpc_runner as r

    args = _runner_args(['--noise-sweep', 'none,readout,full'] + flag)
    tasks = r.build_tasks(args, '/tmp/m', 18, [], q_ceiling_genetic=24,
                          noise_levels=['none', 'readout', 'full'],
                          noisy_q_ceiling=13, noisy_q_ceiling_genetic=13)
    control = [t for t in tasks if t.noise == 'none-counts']
    assert bool(control) is expected
    if expected:
        # one control column PER MODEL, like any other column
        assert {t.model for t in control} == {'lcdm', 'cpl'}
        assert len(control) == 2
        for t in control:
            assert t.script == 'cosmo_modular_quantum.py', \
                "the control belongs to the proposal; the QGA has no switchable route"
            assert '--proposal-route' in t.argv
            assert t.argv[t.argv.index('--proposal-route') + 1] == 'counts'
            assert t.argv[t.argv.index('--noise') + 1] == 'none'


def test_runner_control_does_not_pollute_an_ideal_run():
    """Without a noise sweep no extra column must appear.

    An ideal run has to keep producing exactly the same tasks and paths as
    before the axis.
    """
    import cosmo_hpc_runner as r

    args = _runner_args([])
    tasks = r.build_tasks(args, '/tmp/m', 18, [], q_ceiling_genetic=24,
                          noise_levels=['none'])
    assert all(t.noise == 'none' for t in tasks)
    assert all('noise' not in t.outdir for t in tasks)
    assert all('--proposal-route' not in t.argv for t in tasks)


# =============================================================================
# [B-PLAN] Planning without qiskit
# =============================================================================

def test_b_plan_ansatz_formula_matches_the_circuit():
    """[B-PLAN] The qiskit-free fallback must give EXACTLY the same.

    `ansatz_n_params` measures the real circuit when qiskit is present, and
    uses `n*(2L+1)` when it is not — because building the circuit dragged
    qiskit into the runner, which until then only planned tasks and launched
    subprocesses. If the two routes diverged, the runner would plan with a
    qubit ceiling different from what the run really needs, and the task
    would die from OOM halfway through the campaign.
    """
    pytest.importorskip('qiskit')
    from qpu_cosmo_samplers import build_ansatz

    for layers in (1, 2, 3, 4, 5):
        for n in (2, 3, 5, 6, 9, 13, 18):
            assert build_ansatz(n, layers).num_parameters == \
                n * (2 * layers + 1), \
                f"the fallback formula diverges at n={n}, L={layers}"
            assert cn.ansatz_n_params(n, layers) == n * (2 * layers + 1)


def test_b_plan_fallback_does_not_need_qiskit(monkeypatch):
    """The fallback must kick in if the import fails, not propagate the error."""
    import builtins
    real_import = builtins.__import__

    def blocked(name, *a, **kw):
        if name.startswith(('qpu_cosmo_samplers', 'qiskit')):
            raise ImportError('simulated: node without qiskit')
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, '__import__', blocked)
    assert cn.ansatz_n_params(12, 3) == 12 * 7
    assert cn.param_shift_batch_factor(12) == 2 * 12 * 7


@pytest.mark.qiskit
def test_b_gpu_diagnosis_does_not_crash_without_gpu():
    """[B-GPU] The diagnosis must inform, never break the run.

    It is invoked precisely when something is already going wrong (a GPU was
    requested and there is none); if the diagnosis itself raised, it would
    take down the very run it was meant to save.
    """
    from cosmo_core import gpu_diagnosis, resolve_device

    txt = gpu_diagnosis()
    assert isinstance(txt, str) and 'GPU' in txt
    assert 'qiskit-aer' in txt or 'could not query' in txt
    # degrading without a GPU must never raise, with or without a warning
    assert resolve_device(True, warn=False) in ('CPU', 'GPU')
    assert resolve_device(True, warn=True) in ('CPU', 'GPU')
    assert resolve_device(False) == 'CPU'


# =============================================================================
# [B-CGROUP] Container limits
# =============================================================================

def test_b_cgroup_reads_the_container_limit(tmp_path, monkeypatch):
    """[B-CGROUP] Inside a container the cgroup must be read, not the node.

    `psutil.virtual_memory()` and `os.cpu_count()` report the resources of
    the NODE. In a pod with "Maximum memory 63Gi" on a 512 GB node, the
    runner believed it had 512 GB, admitted dozens of tasks at once and the
    runtime killed the pod for OOM — a SIGKILL, with no traceback or partial
    results.
    """
    import cosmo_hpc_runner as r

    v2 = tmp_path / 'v2'
    v2.mkdir()
    (v2 / 'memory.max').write_text('67645734912\n')      # 63 GiB
    (v2 / 'cpu.max').write_text('1500000 100000\n')      # 15 cores

    real_open = open

    def fake_open(path, *a, **kw):
        p = str(path)
        if p.startswith('/sys/fs/cgroup/'):
            return real_open(str(v2 / os.path.basename(p)), *a, **kw)
        return real_open(path, *a, **kw)

    monkeypatch.setattr('builtins.open', fake_open)
    assert abs(r._cgroup_memory_limit_mb() - 67645.7) < 1.0
    assert r._cgroup_cpu_limit() == 15
    assert r.detected_cores() == 15
    assert abs(r.detected_memory_mb() - 67645.7) < 1.0


def test_b_cgroup_without_limit_falls_back_to_node(monkeypatch):
    """Without a cgroup (or with 'max'), it must behave as before the fix."""
    import cosmo_hpc_runner as r

    real_open = open

    def no_cgroup(path, *a, **kw):
        if str(path).startswith('/sys/fs/cgroup/'):
            raise FileNotFoundError(path)
        return real_open(path, *a, **kw)

    monkeypatch.setattr('builtins.open', no_cgroup)
    assert r._cgroup_memory_limit_mb() is None
    assert r._cgroup_cpu_limit() is None
    assert r.detected_cores() >= 1
    assert r.detected_memory_mb() > 0


def test_b_cgroup_unlimited_sentinel(monkeypatch, tmp_path):
    """The giant cgroup v1 sentinel must not be read as a real limit.

    cgroup v1 writes 2^63-1 to mean "no limit". Taking it literally would
    give a budget of 9 million TB.
    """
    import cosmo_hpc_runner as r

    d = tmp_path / 'v1'
    d.mkdir()
    (d / 'memory.limit_in_bytes').write_text('9223372036854771712\n')
    real_open = open

    def fake_open(path, *a, **kw):
        p = str(path)
        if p == '/sys/fs/cgroup/memory.max':
            raise FileNotFoundError(p)
        if p.startswith('/sys/fs/cgroup/'):
            return real_open(str(d / os.path.basename(p)), *a, **kw)
        return real_open(path, *a, **kw)

    monkeypatch.setattr('builtins.open', fake_open)
    assert r._cgroup_memory_limit_mb() is None


# =============================================================================
# [B-GRID] The noise comparison cannot mix resolutions
# =============================================================================

def _fake_run(tmp_path, folders):
    """Build a fake master_dir with a minimal CSV per folder."""
    hdr = ("Method,Om_mean,Om_std,H0_mean,H0_std,nqpp,chi2_red,final_KL,ESS\n")
    for name, kl in folders.items():
        d = tmp_path / name
        d.mkdir(parents=True)
        (d / 'results_config.csv').write_text(
            hdr
            + f"Classical MCMC,0.26,0.016,70.8,1.09,—,0.56,,101\n"
            + f"QVMC 67%,0.26,0.017,70.8,1.10,3,0.56,{kl},91\n")
    return str(tmp_path)


def test_b_grid_does_not_mix_resolutions(tmp_path):
    """[B-GRID] With --nqpp-sweep and --noise-sweep together, one figure per nqpp.

    The noisy ceiling clamps some resolutions and not others, so the ideal
    column may keep nqpp=5 while the noisy one drops to nqpp=3. Comparing
    them on the same axis, the figure would attribute to noise what is a
    change of resolution — and the series gave no warning: it took
    `sel[-1]`, an arbitrary row among the several sharing method and rung.
    """
    import cosmo_hpc_runner as r

    master = _fake_run(tmp_path, {
        'samplers_lcdm_nqpp3_noise-none': 10.0,
        'samplers_lcdm_nqpp3_noise-full': 11.5,
        'samplers_lcdm_nqpp5_noise-none': 8.0,
        'samplers_lcdm_nqpp5_noise-full': 9.2,
    })
    made = r.generate_noise_comparison_plots(master)
    names = {os.path.basename(p) for p in made}
    assert names == {'noise_comparison_nqpp3_lcdm.png',
                     'noise_comparison_nqpp5_lcdm.png'}, names


def test_b_grid_single_resolution_keeps_the_name(tmp_path):
    """Without a sweep, the name and the title stay as always."""
    import cosmo_hpc_runner as r

    master = _fake_run(tmp_path, {
        'samplers_lcdm_noise-none': 10.0,
        'samplers_lcdm_noise-full': 11.5,
    })
    made = r.generate_noise_comparison_plots(master)
    assert [os.path.basename(p) for p in made] == \
        ['noise_comparison_lcdm.png']


def test_b_grid_untagged_folder_is_ONE_group(tmp_path):
    """A CSV without a folder tag must not be split into two groups.

    Inferring the resolution row by row from the `nqpp` column did exactly
    that: classical MCMC leaves it empty, so the same CSV produced a '' group
    and an 'nqpp3' group, and two figures came out with half of the methods
    each.
    """
    import cosmo_hpc_runner as r

    assert r._infer_grid('/x/samplers_lcdm_noise-none/r.csv',
                         {'nqpp': '3'}) == ''
    assert r._infer_grid('/x/samplers_lcdm_noise-none/r.csv',
                         {'nqpp': ''}) == ''
    assert r._infer_grid('/x/samplers_lcdm_nqpp5_noise-full/r.csv',
                         {}) == 'nqpp5'
    assert r._infer_grid('/x/genetic_cpl_nb4_noise-full/r.csv', {}) == 'nb4'


# =============================================================================
# [B-OFFSET] The chi2 axis cannot lie with offset notation
# =============================================================================

@pytest.mark.qiskit
def test_b_offset_chi2_axis_shows_real_values():
    """[B-OFFSET] No offset: a chi2 of 1100 must read as 1100.

    The rungs converge to nearly identical chi2, so the axis range is tenths
    on top of a four-digit value. Faced with that, matplotlib labels
    0.0, 0.1, 0.2... and hides a '+1.1e3' in a corner: the figure APPEARS to
    show a chi2 between 0 and 1 and a reasonable reader concludes they are
    looking at the reduced one, or that the fit is absurdly good.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import cosmo_genetic_optimizers as ge

    gens = np.arange(40)
    fig, ax = plt.subplots()
    for off in (0.42, 0.45, 0.51):
        ax.plot(gens, 1100.0 + off - 0.3 * np.exp(-gens / 5.0))

    class _R:
        stats = {'n_data': 1099, 'chi2_red': 1.001}

    ge._chi2_axis(ax, [_R()], plt)
    fig.canvas.draw()
    assert ax.get_yaxis().get_offset_text().get_text() == '', \
        "the axis is still using offset notation"
    ticks = [t.get_text().replace('−', '-')
             for t in ax.get_yticklabels() if t.get_text()]
    vals = []
    for t in ticks:
        try:
            vals.append(float(t))
        except ValueError:
            pass
    assert vals and min(vals) > 1000, \
        f"the labels should be around 1100, they are {ticks}"
    # and the axis must say it is raw, with how many data points
    lab = ax.get_ylabel()
    assert 'not reduced' in lab and '1099' in lab
    plt.close(fig)


@pytest.mark.qiskit
def test_b_overlap_value_labels_do_not_overlap():
    """[B-OVERLAP] Overlapping labels are worse than no labels.

    The rungs' chi2 curves end within thousandths of each other, so the
    direct label of the final value is the only thing that separates them
    unambiguously — but if the labels overlap the number becomes unreadable
    and looks like a value it is not.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import cosmo_genetic_optimizers as ge

    fig, ax = plt.subplots()
    ax.set_ylim(27.4, 29.0)
    vals = [27.469, 27.469, 27.475, 27.680, 28.079]
    items = []
    for v in vals:
        ann = ax.annotate(f"{v:.3f}", xy=(30, v), xytext=(4, 0),
                          textcoords='offset points', va='center')
        items.append((v, ann))
    fig.canvas.draw()
    ge._spread_labels(ax, items)
    fig.canvas.draw()

    # effective positions in data units, after the shift
    lo, hi = ax.get_ylim()
    ys = sorted(v + ann.get_position()[1] / ax.bbox.height * (hi - lo)
                for v, ann in items)
    gaps = np.diff(ys)
    assert np.all(gaps > 0.02 * (hi - lo)), \
        f"labels still overlap: gaps {gaps}"
    # and the point they point to did NOT move: the figure does not lie
    assert sorted(a.xy[1] for _, a in items) == sorted(vals)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# [B-MEM] The samplers memory model must depend on N_data
# ─────────────────────────────────────────────────────────────────────────────
#
# Origin: CC+BAO+Pantheon campaign of 2026-08-31. The planner estimated
# 3.7 GB for 18 qubits and 17.9-19.6 GB were measured. Three tasks at 20 and
# 21 qubits hit the OOM killer (rc=-9). The cause was not calibration but
# shape: the 1660*8 constant did not depend on N_data, whereas the dominant
# term is arrays of shape (n_states, N_data).

#: Real peak-RSS measurements (MB) from two campaigns, same machine, same
#: code, clean subprocess. (dataset_n_data, total_qubits, peak_rss_mb)
SAMPLERS_RSS_MEASUREMENTS = [
    (1099, 12, 520.9), (1099, 14, 1156.5), (1099, 15, 2555.6),
    (1099, 16, 4822.1), (1099, 18, 19585.8),
    (51, 16, 999.8), (51, 18, 4035.5),
]


def test_b_mem_estimate_covers_all_real_measurements():
    """[B-MEM] The estimate can never fall below what was measured.

    Erring low is a SIGKILL with no traceback or partial results; erring high
    is one fewer task in parallel. The test demands strict coverage and also
    bounds the overestimation, so that "safe" does not degenerate into
    "uselessly conservative".
    """
    r = pytest.importorskip('cosmo_hpc_runner')
    for n_data, q, measured in SAMPLERS_RSS_MEASUREMENTS:
        est = r.estimate_qubits_and_mem(q, 'nqpp', n_data=n_data)
        assert est >= measured, (
            f"UNDERestimate at N_data={n_data}, {q}q: estimated {est:.0f} MB "
            f"< measured {measured:.0f} MB -> OOMKill risk")
        assert est <= 2.0 * measured, (
            f"excessive overestimate at N_data={n_data}, {q}q: "
            f"{est:.0f} MB versus {measured:.0f} MB measured")


def test_b_mem_old_constant_would_have_failed():
    """[B-MEM] Regression test: the old model DID underestimate.

    Without this, the test above would also pass with a model that happened
    to be right by chance. Here it is pinned that the concrete failure that
    was fixed really existed.
    """
    r = pytest.importorskip('cosmo_hpc_runner')
    old_per_state = 1660 * 8          # the original constant
    worst = [(n, q, m) for n, q, m in SAMPLERS_RSS_MEASUREMENTS if q >= 15]
    assert worst
    for n_data, q, measured in worst:
        old = (2 ** q) * old_per_state / 1e6 + r.PROCESS_BASELINE_MB
        new = r.estimate_qubits_and_mem(q, 'nqpp', n_data=n_data)
        if n_data >= 1000:
            assert old < measured, "the measurement should expose the old model"
        assert new > old


def test_b_mem_depends_on_dataset_and_only_for_samplers():
    """[B-MEM] N_data moves the samplers cost, not the genetic one."""
    r = pytest.importorskip('cosmo_hpc_runner')
    small = r.estimate_qubits_and_mem(16, 'nqpp', n_data=51)
    large = r.estimate_qubits_and_mem(16, 'nqpp', n_data=1099)
    assert large > 3 * small, (
        "the per-state cost must grow with N_data; otherwise the "
        "(n_states, N_data) term is not in the model")
    # The QGA evaluates the likelihood over the POPULATION, not over the grid.
    g1 = r.estimate_qubits_and_mem(20, 'n_bits', n_data=51)
    g2 = r.estimate_qubits_and_mem(20, 'n_bits', n_data=1099)
    assert g1 == g2


def test_b_mem_unknown_dataset_is_conservative():
    """[B-MEM] A dataset not in the table uses the largest, not the smallest."""
    r = pytest.importorskip('cosmo_hpc_runner')
    assert r.dataset_n_data('CC+BAO') == 51
    assert r.dataset_n_data(None) == r.DEFAULT_PLAN_N_DATA
    assert r.dataset_n_data('CC+BAO+DESI+Union3') == r.DEFAULT_PLAN_N_DATA
    assert r.DEFAULT_PLAN_N_DATA == max(r.DATASET_N_DATA.values())


def test_b_mem_tasks_that_died_are_now_rejected():
    """[B-MEM] The three configurations that gave rc=-9 must not be planned.

    cpl/nqpp5 = 20 q, gede/nqpp7 and wcdm/nqpp7 = 21 q, in a 63 GiB
    container. With the old model the ceiling admitted them.
    """
    r = pytest.importorskip('cosmo_hpc_runner')
    ceiling = r.qubit_ceiling(None, 63 * 1024, 'nqpp', n_data=1099)
    assert ceiling < 20, (
        f"the ceiling ({ceiling} q) still admits the tasks that were "
        f"OOMKilled at 20 and 21 qubits")
    # and with CC+BAO, which is 20x smaller, the same node grants more
    assert r.qubit_ceiling(None, 63 * 1024, 'nqpp', n_data=51) > ceiling


# ─────────────────────────────────────────────────────────────────────────────
# [B-GAPLOT] The genetic figure and the CSV must report the SAME number
# ─────────────────────────────────────────────────────────────────────────────

def test_b_gaplot_figure_bar_matches_the_csv():
    """[B-GAPLOT] Same statistic in the table and in the figure.

    The figure drew `theta_map` with the UNWEIGHTED deviation of the final
    population; the CSV reports the fitness-WEIGHTED mean and deviation. In
    lcdm/nb6 that gave +-0.0165 in the figure against +-0.0014 in the table,
    twelve times wider, with nothing to warn which was which.
    """
    ge = pytest.importorskip('cosmo_genetic_optimizers')
    rng = np.random.default_rng(0)
    pop = rng.normal(0.28, 0.02, size=(400, 2))
    pop[:, 1] = rng.normal(69.6, 1.5, size=400)
    # fitness that concentrates the weight in a narrow core, like elitism
    fit = -0.5 * (((pop[:, 0] - 0.276) / 0.0015) ** 2
                  + ((pop[:, 1] - 69.60) / 0.11) ** 2)
    w = ge._fitness_weights(fit)

    class _R:
        method, quantumness, label = 'CGA', 0.0, 'CGA'
        theta_map = np.array([0.2763, 69.5949])
        chi2_map, stats, elapsed, config = 0.0, {}, 0.0, {}
        final_pop, final_fit, final_weights = pop, fit, w
        history = [{'gen': g, 'theta_best': np.array([0.2763, 69.5949]),
                    'best_chi2': 1.0, 'mean_chi2': 2.0} for g in range(5)]
        pop_history, fit_history = [], []

    mu_csv = np.average(pop, weights=w, axis=0)
    sd_csv = np.sqrt(np.average((pop - mu_csv) ** 2, weights=w, axis=0))
    sd_unweighted = pop.std(axis=0)
    # the scenario has to be the bug's, otherwise the test proves nothing
    assert (sd_unweighted > 5 * sd_csv).all()

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    class _M:
        name, label = 'lcdm', 'Flat LCDM'
        n_params = 2
        param_names = ['Om', 'H0']
        param_latex = [r'$\Omega_m$', r'$H_0$']
        fiducial = np.array([0.3111, 67.66])

    with tempfile.TemporaryDirectory() as td:
        ge.plot_genetic_convergence([_R()], _M(), td)
        fig = plt.gcf()
    # recovering the drawn bars from the already closed figure is not
    # possible, so the statistic the figure code now computes is checked by
    # replicating it: weighted mean and deviation, identical to the CSV's.
    finite = np.isfinite(fit)
    mu_fig = np.average(pop[finite], weights=w[finite], axis=0)
    sd_fig = np.sqrt(np.average((pop[finite] - mu_fig) ** 2,
                                weights=w[finite], axis=0))
    assert np.allclose(mu_fig, mu_csv)
    assert np.allclose(sd_fig, sd_csv)
    plt.close('all')


def test_b_gaplot_caption_does_not_call_the_spread_an_interval():
    """[B-GAPLOT] The genetic bar is NOT a credible interval.

    It is the optimizer's convergence spread. Confusing the two is how one
    arrives at "the QGA differs from the CGA by 9 sigma" when in units of the
    MCMC sigma the difference is 0.16.
    """
    ge = pytest.importorskip('cosmo_genetic_optimizers')
    src = inspect.getsource(ge.plot_genetic_convergence)
    assert 'NOT a credible interval' in src
    assert 'fitness-weighted' in src


# ─────────────────────────────────────────────────────────────────────────────
# [B-TIME] The noisy genetic optimizer is bounded by the clock, not the RAM
# ─────────────────────────────────────────────────────────────────────────────
#
# Origin: 2026-09-03 campaign. The noisy genetic ceiling was derived from
# memory alone, which at 14 qubits is granted without trouble (a rho is
# 4.3 GB). But at 14 qubits the QGA takes ~7.7 h PER GENERATION, i.e. ~4
# months for the 500 that were requested. The task hung and also blocked the
# summary figures of the whole run, because the runner does not write them
# until all tasks finish.

#: Pace measured in that campaign's logs (pop=500, readout, 63 GiB node).
QGA_SEC_PER_GEN_MEASUREMENTS = [(10, 73.0), (12, 1425.0)]


def test_b_time_model_reproduces_the_measurements():
    """[B-TIME] The extrapolation has to pass through the measured points."""
    for n, sec in QGA_SEC_PER_GEN_MEASUREMENTS:
        est = cn.genetic_noisy_seconds_per_gen(n)
        assert abs(est - sec) / sec < 0.05, (
            f"{n}q: model {est:.0f} s/gen versus {sec:.0f} s measured")


def test_b_time_extrapolation_to_14q_matches_observation():
    """[B-TIME] Out-of-sample validation.

    The model is fitted with 10 and 12 qubits. The third observation — at 14
    qubits not even generation 0 had been logged after 7 h — is not part of
    the fit, so it serves to check it: the model must predict more than 7 h
    per generation, and not an absurd value.
    """
    s14 = cn.genetic_noisy_seconds_per_gen(14)
    assert s14 > 7 * 3600, f"predicts {s14/3600:.1f} h/gen; >7 h was observed"
    assert s14 < 24 * 3600, f"predicts {s14/3600:.1f} h/gen, implausible"


def test_b_time_ceiling_cuts_the_cell_that_never_finished():
    """[B-TIME] 14 qubits with 500 generations cannot enter the plan."""
    ceiling = cn.genetic_noisy_time_ceiling(500, cn.DEFAULT_NOISY_TASK_HOURS)
    assert ceiling < 14, (
        f"the ceiling ({ceiling} q) still admits the ~4-month cell")
    # and the budget has to RULE over what the RAM would grant
    ram_only = cn.noisy_qubit_ceiling(mem_mb=95 * 1024, quantum_training=False)
    with_time = cn.noisy_qubit_ceiling(mem_mb=95 * 1024,
                                       quantum_training=False,
                                       generations=500)
    assert with_time < ram_only, (
        "with generations given, the time ceiling must be more restrictive "
        "than the memory one on a large node")


def test_b_time_is_a_budget_not_a_constant():
    """[B-TIME] A larger budget or fewer generations grant more qubits.

    This is what sets it apart from the hard MAX_NOISY_QUBITS=13 removed in
    [REV]: that one did not move with anything.
    """
    assert (cn.genetic_noisy_time_ceiling(60, 48)
            > cn.genetic_noisy_time_ceiling(500, 48))
    assert (cn.genetic_noisy_time_ceiling(60, 480)
            > cn.genetic_noisy_time_ceiling(60, 48))
    # and it never returns 0, which would produce an empty plan with no explanation
    assert cn.genetic_noisy_time_ceiling(10 ** 6, 0.001) >= 1


def test_b_time_touches_neither_samplers_nor_ideal_tasks():
    """[B-TIME] The time ceiling applies ONLY to the noisy genetic optimizer.

    The samplers are bounded by the parameter-shift batch (memory) and the
    noise-free tasks build no density matrix at all.
    """
    r = pytest.importorskip('cosmo_hpc_runner')
    mem = 63 * 1024
    # samplers: passing generations cannot change anything
    a = r.qubit_ceiling(None, mem, 'nqpp', noisy=True, n_data=1099)
    b = r.qubit_ceiling(None, mem, 'nqpp', noisy=True, n_data=1099,
                        generations=500)
    assert a == b
    # ideal genetic: neither
    c = r.qubit_ceiling(None, mem, 'n_bits', noisy=False)
    d = r.qubit_ceiling(None, mem, 'n_bits', noisy=False, generations=500)
    assert c == d
    # noisy genetic: yes
    e = r.qubit_ceiling(None, mem, 'n_bits', noisy=True, generations=500)
    assert e < c


def test_b_time_flag_exposed_in_the_cli():
    """[B-TIME] The budget must be changeable without editing code."""
    r = pytest.importorskip('cosmo_hpc_runner')
    p = r.build_parser()
    ns = p.parse_args(['--noisy-task-hours', '12'])
    assert ns.noisy_task_hours == 12.0
    assert p.parse_args([]).noisy_task_hours == cn.DEFAULT_NOISY_TASK_HOURS


# ─────────────────────────────────────────────────────────────────────────────
# [B-NOSTATE] The genetic optimizer's state has to outlive the process
# ─────────────────────────────────────────────────────────────────────────────
#
# Origin: while fixing [B-GAPLOT] it turned out that the genetic figures
# could not be regenerated, because the final population only lived in
# memory. Fixing the color of a curve required repeating the whole campaign —
# days of compute for a cosmetic change.

def _synthetic_ga_result(seed=0, n=200, d=2, gens=6):
    """A GAResult with realistic data, without running the optimization."""
    ge = pytest.importorskip('cosmo_genetic_optimizers')
    rng = np.random.default_rng(seed)
    pop = np.column_stack([rng.normal(0.28, 0.02, n), rng.normal(69.6, 1.5, n)])
    fit = -0.5 * (((pop[:, 0] - 0.276) / 0.0015) ** 2
                  + ((pop[:, 1] - 69.60) / 0.11) ** 2)
    hist = [{'gen': g, 'theta_best': np.array([0.2763 + 1e-5 * g, 69.59]),
             'best_chi2': 1064.6 - 0.01 * g, 'mean_chi2': 1100.0 - g}
            for g in range(gens)]
    return ge.GAResult(
        method='QGA', quantumness=67.0, theta_map=np.array([0.2763, 69.5949]),
        chi2_map=1064.61, stats={'chi2': 1064.61, 'n_data': 1099},
        final_pop=pop, final_fit=fit,
        final_weights=ge._fitness_weights(fit), history=hist,
        elapsed=12.5, config={'mutation': 'quantum'}, label='QGA (q=67%)',
        pop_history=[pop + 0.001 * g for g in range(gens)],
        fit_history=[fit for _ in range(gens)])


def test_b_nostate_exact_round_trip():
    """[B-NOSTATE] What is saved is what is read, lossless where it matters.

    The final population, its fitness and its weights feed the statistics that
    go into the CSV, so they have to come back EXACT (float64). The population
    history only feeds the animation and is saved as float32 on purpose.
    """
    ge = pytest.importorskip('cosmo_genetic_optimizers')
    core = pytest.importorskip('cosmo_core')
    orig = [_synthetic_ga_result(seed=s, ) for s in (0, 1)]
    with tempfile.TemporaryDirectory() as td:
        p = ge.save_ga_state(orig, core.MODELS['lcdm'], td)
        assert os.path.exists(p)
        back, meta = ge.load_ga_state(p)

    assert len(back) == len(orig)
    assert meta['model'] == 'lcdm'
    for a, b in zip(orig, back):
        assert b.label == a.label and b.method == a.method
        assert b.quantumness == a.quantumness
        # exact: the published numbers come from here
        assert np.array_equal(a.final_pop, b.final_pop)
        assert np.array_equal(a.final_fit, b.final_fit)
        assert np.array_equal(a.final_weights, b.final_weights)
        assert np.array_equal(a.theta_map, b.theta_map)
        # history
        assert len(b.history) == len(a.history)
        for ha, hb in zip(a.history, b.history):
            assert ha['gen'] == hb['gen']
            assert np.allclose(ha['theta_best'], hb['theta_best'])
            assert ha['best_chi2'] == pytest.approx(hb['best_chi2'])


def test_b_nostate_csv_statistic_is_rebuilt():
    """[B-NOSTATE] After the trip through disk, the CSV would come out identical.

    It is the property that really matters: the state serves for redrawing,
    and the figure has to keep matching the table ([B-GAPLOT]).
    """
    ge = pytest.importorskip('cosmo_genetic_optimizers')
    core = pytest.importorskip('cosmo_core')
    orig = [_synthetic_ga_result(seed=3)]

    def stats(r):
        finite = np.isfinite(r.final_fit)
        pop, w = r.final_pop[finite], np.asarray(r.final_weights)[finite]
        mu = np.average(pop, weights=w, axis=0)
        sd = np.sqrt(np.average((pop - mu) ** 2, weights=w, axis=0))
        return mu, sd

    with tempfile.TemporaryDirectory() as td:
        p = ge.save_ga_state(orig, core.MODELS['lcdm'], td)
        back, _ = ge.load_ga_state(p)
    mu0, sd0 = stats(orig[0])
    mu1, sd1 = stats(back[0])
    assert np.array_equal(mu0, mu1), "the weighted mean changed on saving"
    assert np.array_equal(sd0, sd1), "the weighted deviation changed on saving"


def test_b_nostate_replot_does_not_need_the_optimization():
    """[B-NOSTATE] The figures can be redone from the file alone."""
    ge = pytest.importorskip('cosmo_genetic_optimizers')
    core = pytest.importorskip('cosmo_core')
    import matplotlib
    matplotlib.use('Agg')
    rs = [_synthetic_ga_result(seed=s) for s in (0, 1)]
    rs[0].label, rs[0].quantumness = 'CGA', 0.0
    with tempfile.TemporaryDirectory() as td:
        p = ge.save_ga_state(rs, core.MODELS['lcdm'], td)
        figs = ge.replot_from_state(p, animate=False)
        assert figs, "no figure was regenerated"
        for f in figs:
            assert os.path.exists(f) and os.path.getsize(f) > 1000


def test_b_nostate_future_format_warns_instead_of_failing_oddly():
    """[B-NOSTATE] A file from a newer version gives a readable error."""
    ge = pytest.importorskip('cosmo_genetic_optimizers')
    core = pytest.importorskip('cosmo_core')
    import json as _json
    with tempfile.TemporaryDirectory() as td:
        p = ge.save_ga_state([_synthetic_ga_result()], core.MODELS['lcdm'], td)
        z = dict(np.load(p, allow_pickle=False))
        meta = _json.loads(str(z['meta_json']))
        meta['version'] = ge.GA_STATE_VERSION + 5
        z['meta_json'] = np.array(_json.dumps(meta))
        np.savez_compressed(p, **z)
        with pytest.raises(ValueError, match='format'):
            ge.load_ga_state(p)


def test_b_nostate_without_pickle():
    """[B-NOSTATE] It loads with allow_pickle=False, or it will not last two years."""
    ge = pytest.importorskip('cosmo_genetic_optimizers')
    core = pytest.importorskip('cosmo_core')
    with tempfile.TemporaryDirectory() as td:
        p = ge.save_ga_state([_synthetic_ga_result()], core.MODELS['lcdm'], td)
        np.load(p, allow_pickle=False)     # must not raise


# ─────────────────────────────────────────────────────────────────────────────
# [B-BUDGET] The classical vs quantum comparison has to cost the same
# ─────────────────────────────────────────────────────────────────────────────
#
# Origin: adversarial review of cosmo_modular_quantum.py. `max_iter` was
# passed equally to both branches of QVMC training, but a quantum iteration
# costs 1 + 2*n_phi circuit evaluations (the KL at phi plus the
# parameter-shift displaced ones) and a classical one costs 1. With the same
# --qvmc-iter the quantum branch received between 57x and 225x more work, and
# the comparison came out biased IN FAVOR of quantum — the worst possible
# outcome for a project whose central claim is faithfulness, not advantage.

def _count_circuits(v, max_iter):
    """Run a QVMC counting how many circuit evaluations it spends."""
    n = [0]
    orig = v._kl_batch

    def patched(ph, *a, **k):
        n[0] += len(np.atleast_2d(ph))
        return orig(ph, *a, **k)

    v._kl_batch = patched
    r = v.run(max_iter=max_iter, n_chains=1, progress=False)
    return n[0], r


def test_b_budget_both_branches_spend_the_same():
    """[B-BUDGET] At a matched budget, both branches use similar circuit counts."""
    mq = pytest.importorskip('cosmo_modular_quantum')
    core = pytest.importorskip('cosmo_core')
    import matplotlib
    matplotlib.use('Agg')
    post = core.Posterior(core.MODELS['lcdm'], 'CC+BAO')
    kw = dict(n_qubits_per_param=2, n_shots=300)
    n_cla, _ = _count_circuits(
        mq.QVMCModular(post, config={}, budget_mode='circuits', **kw), 20)
    n_qua, _ = _count_circuits(
        mq.QVMCModular(post, config={'training': True},
                       budget_mode='circuits', **kw), 20)
    assert 0.5 < n_cla / n_qua < 2.0, (
        f"very different budgets: classical {n_cla}, quantum {n_qua}")


def test_b_budget_old_mode_was_indeed_biased():
    """[B-BUDGET] Regression: with 'iters' the asymmetry exists and is huge.

    Without this, the test above would also pass with code that happened to
    be right by chance. Here it is pinned that the concrete bias that was
    fixed was real.
    """
    mq = pytest.importorskip('cosmo_modular_quantum')
    core = pytest.importorskip('cosmo_core')
    import matplotlib
    matplotlib.use('Agg')
    post = core.Posterior(core.MODELS['lcdm'], 'CC+BAO')
    kw = dict(n_qubits_per_param=2, n_shots=300)
    n_cla, _ = _count_circuits(
        mq.QVMCModular(post, config={}, budget_mode='iters', **kw), 20)
    n_qua, _ = _count_circuits(
        mq.QVMCModular(post, config={'training': True},
                       budget_mode='iters', **kw), 20)
    assert n_qua > 10 * n_cla, (
        f"the historical asymmetry was expected; classical {n_cla}, "
        f"quantum {n_qua}")


def test_b_budget_work_spent_is_reported():
    """[B-BUDGET] The result must declare its budget and its circuits."""
    mq = pytest.importorskip('cosmo_modular_quantum')
    core = pytest.importorskip('cosmo_core')
    import matplotlib
    matplotlib.use('Agg')
    post = core.Posterior(core.MODELS['lcdm'], 'CC+BAO')
    v = mq.QVMCModular(post, config={'training': True},
                       n_qubits_per_param=2, n_shots=300)
    r = v.run(max_iter=5, n_chains=1, progress=False)
    assert r['budget_mode'] == 'circuits'
    assert isinstance(r['circuits_train'], int) and r['circuits_train'] > 0


def test_b_budget_unknown_mode_is_rejected():
    """[B-BUDGET] An arbitrary string is not accepted as a mode."""
    mq = pytest.importorskip('cosmo_modular_quantum')
    core = pytest.importorskip('cosmo_core')
    post = core.Posterior(core.MODELS['lcdm'], 'CC+BAO')
    with pytest.raises(ValueError, match='budget_mode'):
        mq.QVMCModular(post, config={}, budget_mode='whatever')


# ─────────────────────────────────────────────────────────────────────────────
# [B-ESSCOMP] The QVMC ESS cannot depend on how the sample is represented
# ─────────────────────────────────────────────────────────────────────────────

def test_b_esscomp_ess_does_not_depend_on_representation():
    """[B-ESSCOMP] The FAITHFUL `sampling` cell cannot look like a regression.

    The quantum branch returned the COMPRESSED form (one row per bitstring,
    weight = count) and the classical one a row per shot. Kish's ESS is not
    invariant under that compression, so the CSV published a 60x drop in a
    cell where by construction there must be no difference.
    """
    mq = pytest.importorskip('cosmo_modular_quantum')
    core = pytest.importorskip('cosmo_core')
    import matplotlib
    matplotlib.use('Agg')
    post = core.Posterior(core.MODELS['lcdm'], 'CC+BAO')
    out = {}
    for lbl, cfg in (('classical', {}), ('quantum', {'sampling': True})):
        v = mq.QVMCModular(post, config=cfg, n_qubits_per_param=3,
                           n_shots=1000)
        r = v.run(max_iter=4, n_chains=2, progress=False)
        out[lbl] = (len(r['S']), r['ess'])
    (n_c, ess_c), (n_q, ess_q) = out['classical'], out['quantum']
    assert n_c == n_q, f"different number of rows: {n_c} vs {n_q}"
    assert abs(ess_c - ess_q) / max(ess_c, 1.0) < 0.02, (
        f"ESS not comparable between branches: classical {ess_c:.1f}, "
        f"quantum {ess_q:.1f}")


# ─────────────────────────────────────────────────────────────────────────────
# [B-EMPTY] A grid without support must raise, not give a pretty KL
# ─────────────────────────────────────────────────────────────────────────────

def test_b_empty_grid_without_support_raises():
    """[B-EMPTY] It used to return KL = 0.022 and Om = -4.45 without a single error.

    With the grid outside the prior, P comes out all zeros; the KL
    degenerates into log n - H(Q), which is SMALL — so the run finished
    entirely, reporting the best KL of the campaign on absurd parameters.
    """
    mq = pytest.importorskip('cosmo_modular_quantum')
    core = pytest.importorskip('cosmo_core')
    post = core.Posterior(core.MODELS['lcdm'], 'CC+BAO')
    for cfg in ({}, dict(mq.PRESETS[100])):
        v = mq.QVMCModular(post, config=cfg, n_qubits_per_param=2,
                           grid_window=[(-100.0, -90.0), (-100.0, -90.0)])
        with pytest.raises(ValueError, match='B-EMPTY'):
            v.build_target()


# ─────────────────────────────────────────────────────────────────────────────
# [B-PROV] A CSV row must state the conditions under which it was obtained
# ─────────────────────────────────────────────────────────────────────────────

def test_b_prov_csv_carries_noise_route_seed_and_budget():
    """[B-PROV] Without these columns the noise axis cannot be reconstructed.

    Two rows with the same key (Method, model, dataset, prior, nqpp) but
    different numbers were indistinguishable: one could come from an ideal
    run and the other from a noisy one, and the file did not say.
    """
    mq = pytest.importorskip('cosmo_modular_quantum')
    core = pytest.importorskip('cosmo_core')
    for fields in (mq.csv_fields_for_model(core.MODELS['lcdm']),
                   mq.csv_fields_generic()):
        for col in ('noise', 'proposal_route', 'seed', 'budget_mode',
                    'circuits_train'):
            assert col in fields, f"column {col!r} is missing"
    side = {'mu': np.array([0.3, 70.0]), 'std': np.array([0.01, 0.5]),
            'chi2': 27.5, 'n_data': 51, 'chi2_red': 0.56, 'AIC': 31.5,
            'BIC': 35.3, 'ess': 1234.0, 'kl_final': 0.5,
            'budget_mode': 'circuits', 'circuits_train': 570, 'seed': 7}
    row = mq.csv_row_for_side(side, core.MODELS['lcdm'], 'QVMC 67%', False,
                              3, 'CC+BAO', 'flat')
    assert row['budget_mode'] == 'circuits'
    assert row['circuits_train'] == '570'
    assert row['noise'] == mq.NOISE.label


# ─────────────────────────────────────────────────────────────────────────────
# [B-REFINE] The genetic chi2 does not distinguish rungs; the grid one does
# ─────────────────────────────────────────────────────────────────────────────
#
# The genetic optimizer returns a point on the GRID and it is then refined
# with a continuous optimizer. That refinement erases the difference between
# rungs: all four converge to the same minimum, so chi2/chi2_red/AIC/BIC come
# out identical to the sixth digit in CGA and in the four QGA rungs. That
# explains why in the published campaigns the genetic chi2 was byte-for-byte
# equal across all rows: it is not operator faithfulness, it is the refiner.

def test_b_refine_refined_chi2_does_not_distinguish_rungs():
    """[B-REFINE] The four rungs give the SAME chi2 after refinement.

    It is the fact one needs to know so as not to read that column as if it
    measured the genetic optimizer.
    """
    ge = pytest.importorskip('cosmo_genetic_optimizers')
    core = pytest.importorskip('cosmo_core')
    import matplotlib
    matplotlib.use('Agg')
    post = core.Posterior(core.MODELS['lcdm'], 'CC+BAO')
    ga = ge.GAConfig(pop_size=30, n_generations=6, seed=42)
    refined, grid = [], []
    for pct in (0, 33, 67, 100):
        r = ge.QGA(post, ga, dict(ge.QGA_PRESETS[pct]), n_bits=4,
                   rng=np.random.default_rng(42), shots=1
                   ).evolve(record_population=False)
        refined.append(r.chi2_map)
        grid.append(r.chi2_grid)
    assert max(refined) - min(refined) < 1e-6, (
        f"refinement was expected to equalize the rungs: {refined}")
    assert max(grid) - min(grid) > 1e-3, (
        f"the grid chi2 should distinguish them: {grid}")


def test_b_refine_faithful_cell_also_holds_on_the_grid():
    """[B-REFINE] CGA == QGA(0%) before refinement, which is the real test.

    If they only agreed after refinement, the equality would say nothing
    about the operators — the continuous optimizer would guarantee it.
    """
    ge = pytest.importorskip('cosmo_genetic_optimizers')
    core = pytest.importorskip('cosmo_core')
    import matplotlib
    matplotlib.use('Agg')
    post = core.Posterior(core.MODELS['lcdm'], 'CC+BAO')
    ga = ge.GAConfig(pop_size=30, n_generations=6, seed=42)
    c = ge.CGA(post, ga, rng=np.random.default_rng(42)
               ).evolve(record_population=False)
    q0 = ge.QGA(post, ga, dict(ge.QGA_PRESETS[0]), n_bits=4,
                rng=np.random.default_rng(42), shots=1
                ).evolve(record_population=False)
    assert np.array_equal(c.theta_grid, q0.theta_grid)
    assert c.chi2_grid == q0.chi2_grid
    assert np.array_equal(c.final_pop, q0.final_pop)


def test_b_refine_csv_carries_the_grid_chi2():
    """[B-REFINE] The column that distinguishes rungs has to be stored."""
    mq = pytest.importorskip('cosmo_modular_quantum')
    core = pytest.importorskip('cosmo_core')
    assert 'chi2_grid' in mq.csv_fields_generic()
    assert 'chi2_grid' in mq.csv_fields_for_model(core.MODELS['lcdm'])
    side = {'mu': np.array([0.3, 70.0]), 'std': np.array([0.01, 0.5]),
            'chi2': 27.4691, 'n_data': 51, 'chi2_red': 0.56, 'AIC': 31.5,
            'BIC': 35.3, 'ess': 300.0, 'chi2_grid': 28.3992}
    row = mq.csv_row_generic(side, core.MODELS['lcdm'], 'QGA (q=67%)', True,
                             '4', 'CC+BAO', 'flat')
    assert row['chi2_grid'] == '28.3992'
    # a samplers row has no grid of its own: the column stays empty
    row2 = mq.csv_row_generic({k: v for k, v in side.items() if k != 'chi2_grid'},
                              core.MODELS['lcdm'], 'QMCMC 50%', True, '3',
                              'CC+BAO', 'flat')
    assert row2['chi2_grid'] == ''


def test_b_refine_state_keeps_the_grid():
    """[B-REFINE] `ga_state.npz` must keep theta_grid and chi2_grid."""
    ge = pytest.importorskip('cosmo_genetic_optimizers')
    core = pytest.importorskip('cosmo_core')
    r = _synthetic_ga_result()
    r.theta_grid = np.array([0.2700, 70.3125])
    r.chi2_grid = 28.3992
    with tempfile.TemporaryDirectory() as td:
        p = ge.save_ga_state([r], core.MODELS['lcdm'], td)
        back, _ = ge.load_ga_state(p)
    assert np.array_equal(back[0].theta_grid, r.theta_grid)
    assert back[0].chi2_grid == pytest.approx(r.chi2_grid)


def test_b_budgetmismatch_warns_if_max_task_exceeds_the_budget():
    """[B-BUDGETMISMATCH] --max-task-gb > --mem-budget-gb is contradictory.

    The qubit ceiling comes from --max-task-gb, but the pool ALWAYS admits at
    least one task even if it does not fit in the aggregate budget (otherwise
    a large task would block the campaign forever). With that combination a
    task exceeding the whole budget can be launched and the only signal would
    be an OOMKill hours later. Measured: --max-task-gb 40 with
    --mem-budget-gb 10 granted 18 qubits, ~22 GB for ONE task.
    """
    import subprocess
    import sys
    base = [sys.executable, os.path.join(REPO, 'cosmo_hpc_runner.py'),
            '--models', 'lcdm', '--nqpp', '8',
            '--dataset', 'CC+BAO+Pantheon', '--dry-run']
    bad = subprocess.run(base + ['--max-task-gb', '40', '--mem-budget-gb', '10'],
                         capture_output=True, text=True, timeout=300).stdout
    good = subprocess.run(base + ['--max-task-gb', '12', '--mem-budget-gb', '50'],
                          capture_output=True, text=True, timeout=300).stdout
    assert 'WARNING' in bad, "does not warn about the contradictory combination"
    assert 'WARNING' not in good, "warns when the configuration is consistent"


# ─────────────────────────────────────────────────────────────────────────────
# Docstring examples have to RUN
# ─────────────────────────────────────────────────────────────────────────────
#
# An example that does not run is worse than none: it ages silently and ends
# up documenting something the code no longer does. Tying them to the suite
# turns them into documentation that cannot lie — and in fact the first one I
# wrote already caught a mistake of mine (it claimed H(0) == 70.0 exactly,
# when it is 69.99999999999999 due to floating-point rounding).

def test_docstring_examples_run():
    """All the project's doctests pass."""
    import doctest
    import importlib
    failures = []
    for mod in ('cosmo_core', 'cosmo_noise', 'cosmo_hpc_runner',
                'cosmo_modular_quantum', 'cosmo_genetic_optimizers'):
        m = importlib.import_module(mod)
        res = doctest.testmod(m, verbose=False, report=False)
        if res.failed:
            failures.append(f"{mod}: {res.failed} of {res.attempted}")
    assert not failures, "broken doctests -> " + "; ".join(failures)


def test_most_called_functions_have_examples():
    """The functions whose misuse already cost a campaign carry an example.

    It is not decorative documentation: each of these examples pins a number
    that was measured and that explains why a fix exists.
    """
    import importlib
    EXPECTED = {
        'cosmo_noise': ['canonical_level', 'param_shift_batch_factor',
                        'noisy_density_bytes', 'genetic_noisy_seconds_per_gen',
                        'genetic_noisy_time_ceiling'],
        'cosmo_hpc_runner': ['dataset_n_data', 'bytes_per_state_samplers',
                             'estimate_qubits_and_mem'],
        'cosmo_core': ['canonical_dataset', 'ess_weights', 'gelman_rubin',
                       'fit_statistics'],
        'cosmo_modular_quantum': ['compute_quantumness', 'quantumness_qmcmc',
                                  'quantumness_qvmc'],
        'cosmo_genetic_optimizers': ['compute_qga_quantumness'],
    }
    missing = []
    for mod, names in EXPECTED.items():
        m = importlib.import_module(mod)
        for n in names:
            d = getattr(m, n).__doc__ or ''
            if '>>>' not in d:
                missing.append(f"{mod}.{n}")
    assert not missing, "no runnable example: " + ", ".join(missing)
