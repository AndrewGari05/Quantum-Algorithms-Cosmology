"""test_genetic_noise.py — the noise axis seen from the genetic side.

Three real bugs, found by reading the Nicte-Ha and Saptiva campaigns, and
the net that keeps them from coming back:

  [B-PROV-GEN]  every genetic row wrote noise='none', whatever rung it ran
                with, because the CSV writer read the global of ANOTHER
                module. Measured: 116 mislabelled rows in
                `hpc_20260907_130425` (40 fakebrisbane, 40 readout, 36 full).
  [B-RZZ]       the quantum crossover died when transpiling against the
                `full` rung, which declares `rzz`/`rxx`/`ryy` as basis.
                Measured: the task `genetic_lcdm_nb4_noise-full` died in
                __init__ and left the CSV with the CGA row and nothing else.
  folder>column  the axis figures must take the rung from the FOLDER NAME,
                because `none-counts` does not exist as a column value and
                because the old campaigns have it wrong.
"""
import csv
import importlib.util
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import cosmo_noise as cnoise                      # noqa: E402
import cosmo_modular_quantum as cmq               # noqa: E402

_spec = importlib.util.spec_from_file_location(
    'noise_plots', os.path.join(ROOT, 'noise_plots.py'))
gr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gr)


# ═══ [B-PROV-GEN] the row states its own rung ═══════════════════════════

def test_b_prov_gen_genetic_row_carries_its_rung():
    """`csv_provenance` must prefer the rung carried by the row.

    Without this the genetic algorithm inherited the NOISE of
    cosmo_modular_quantum, which its CLI never sets: every noisy run was
    filed as ideal.
    """
    row = gr and {'noise': 'readout', 'seed': '7'}
    prov = cmq.csv_provenance(row)
    assert prov['noise'] == 'readout'
    assert prov['seed'] == '7'


def test_b_prov_gen_samplers_still_read_the_global():
    """The samplers path does NOT change: without the key, the global rules."""
    previous = cmq.NOISE
    try:
        cmq.NOISE = cnoise.NoiseSpec.from_level('full')
        assert cmq.csv_provenance({})['noise'] == 'full'
    finally:
        cmq.NOISE = previous


def test_b_prov_gen_genetic_side_builds_the_key():
    """`_ga_side` must copy the genetic module's rung and the seed."""
    import numpy as np
    import cosmo_genetic_optimizers as G

    class _R:
        final_fit = np.array([-1.0, -2.0])
        final_pop = np.array([[0.3, 70.0], [0.28, 69.0]])
        final_weights = np.array([0.6, 0.4])
        elapsed = 1.0
        chi2_grid = 1064.61
        stats = {'chi2': 1064.61, 'n_data': 1099, 'chi2_red': 0.97,
                 'AIC': 1068.6, 'BIC': 1078.6}

    previous = G.NOISE
    try:
        G.NOISE = cnoise.NoiseSpec.from_level('readout')
        side = G._ga_side(_R(), post=None, seed=123)
        assert side['noise'] == 'readout'
        assert side['seed'] == '123'
        # and that row, passed to the shared writer, keeps the rung
        assert cmq.csv_provenance(side)['noise'] == 'readout'
    finally:
        G.NOISE = previous


# ═══ [B-RZZ] the quantum crossover must not die with `full` ═════════════

def test_b_rzz_full_rung_declares_parametric_gates():
    """The root cause, pinned as a fact: `full` puts rzz/rxx/ryy in the basis.

    If this stops being true, the [B-RZZ] net is redundant and must be
    reviewed; while it is true, the transpiler can trip over them.
    """
    nm = cnoise.NoiseSpec.from_level('full').simulator_kwargs(
        counts_route=True).get('noise_model')
    assert nm is not None
    assert {'rzz', 'rxx', 'ryy'} <= set(nm.basis_gates)


def test_b_rzz_operator_on_unbound_rzz_is_the_exact_error():
    """The error from the Nicte-Ha log, reproduced in one line."""
    from qiskit.circuit import Parameter
    from qiskit.circuit.library import RZZGate
    from qiskit.quantum_info import Operator
    with pytest.raises(TypeError) as exc:
        Operator(RZZGate(Parameter('ϴ')))
    assert 'unbound parameter' in str(exc.value).lower()


def test_b_rzz_retries_without_parametric_gates_and_survives():
    """If the main path blows up with THAT error, there is a second path."""
    import cosmo_genetic_optimizers as G
    qga = G.QGA.__new__(G.QGA)
    qga.noise_label = 'full'
    kw = cnoise.NoiseSpec.from_level('full').simulator_kwargs(
        counts_route=True)
    from qiskit_aer import AerSimulator
    qga.sim = AerSimulator(**kw)

    calls = {'n': 0}
    real = G.transpile

    def _fail_the_first_time(qc, backend, **kwargs):
        calls['n'] += 1
        if 'basis_gates' not in kwargs:
            raise TypeError("ParameterExpression with unbound parameters "
                            "(dict_keys([Parameter(ϴ)])) cannot be cast to "
                            "a float")
        return real(qc, backend, **kwargs)

    G.transpile = _fail_the_first_time
    try:
        from qiskit import QuantumCircuit
        from qiskit.circuit import ParameterVector
        pv = ParameterVector('pa', 2)
        qc = QuantumCircuit(2)
        qc.ry(pv[0], 0)
        qc.ry(pv[1], 1)
        qc.cx(0, 1)
        qc.measure_all()
        with pytest.warns(RuntimeWarning, match='B-RZZ'):
            out = qga._transpile_cx(qc)
        assert out.num_qubits == 2
        assert calls['n'] == 2            # failed once, retried once
    finally:
        G.transpile = real


def test_b_rzz_a_different_error_propagates():
    """The net catches ONE specific error; any other must still kill the run.

    Swallowing generic exceptions here would turn a badly built circuit into
    a silently wrong run.
    """
    import cosmo_genetic_optimizers as G
    qga = G.QGA.__new__(G.QGA)
    qga.noise_label = 'full'
    from qiskit_aer import AerSimulator
    qga.sim = AerSimulator()
    real = G.transpile
    G.transpile = lambda *a, **k: (_ for _ in ()).throw(
        TypeError('something completely different'))
    try:
        with pytest.raises(TypeError, match='something completely'):
            qga._transpile_cx(None)
    finally:
        G.transpile = real


# ═══ the rung is read from the folder, not from the column ══════════════

COLS = ['Method', 'Om_mean', 'Om_std', 'H0_mean', 'H0_std', 'Time_s', 'nqpp',
        'chi2', 'n_data', 'chi2_red', 'AIC', 'BIC', 'acceptance', 'final_KL',
        'ESS', 'dataset', 'prior', 'noise', 'proposal_route', 'seed',
        'budget_mode', 'circuits_train', 'chi2_grid']


def _task(root, name, model, rows):
    d = os.path.join(root, name, f'model_{model}')
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, 'results_config.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, COLS)
        w.writeheader()
        for f in rows:
            r = dict.fromkeys(COLS, '')
            r.update(f)
            w.writerow(r)


def _base(method, om='0.2764', noise='none', **extra):
    r = {'Method': method, 'Om_mean': om, 'Om_std': '0.0100',
         'H0_mean': '69.60', 'H0_std': '0.80', 'Time_s': '10.0',
         'chi2': '1064.6', 'n_data': '1099', 'chi2_red': '0.97',
         'AIC': '1068.6', 'BIC': '1078.6', 'noise': noise,
         'dataset': 'CC+BAO+Pantheon', 'prior': 'flat', 'seed': '42'}
    r.update(extra)
    return r


def test_folder_overrides_column(tmp_path):
    """An old genetic campaign has the column wrong; the folder does not."""
    _task(str(tmp_path), 'genetic_lcdm_nb4_noise-readout', 'lcdm',
          [_base('CGA', noise='none')])           # <- the column lies
    rows = gr.load_campaign(str(tmp_path))
    assert len(rows) == 1
    assert rows[0]['_noise'] == 'readout'
    assert rows[0]['_noise_csv'] == 'none'
    assert rows[0]['_grid'] == 4
    assert rows[0]['_family'] == 'genetic'


def test_none_counts_is_not_confused_with_none(tmp_path):
    """The route control is a cell of its own even if the CSV says 'none'.

    Merging it with `none` would erase the only control that separates the
    change of readout route from the effect of the noise.
    """
    _task(str(tmp_path), 'samplers_lcdm_nqpp3_noise-none', 'lcdm',
          [_base('QMCMC 100%')])
    _task(str(tmp_path), 'samplers_lcdm_nqpp3_noise-none-counts', 'lcdm',
          [_base('QMCMC 100%', proposal_route='counts')])
    rows = gr.load_campaign(str(tmp_path))
    assert {f['_noise'] for f in rows} == {'none', 'none-counts'}
    # and `none-counts` is NOT flagged as inconsistent: its column says
    # 'none' because the noise level IS 'none'.
    import io
    from contextlib import redirect_stdout
    buf = io.StringIO()
    with redirect_stdout(buf):
        gr.warn_inconsistent_labels(rows)
    assert buf.getvalue() == ''


def test_campaign_with_wrong_labels_is_flagged(tmp_path):
    _task(str(tmp_path), 'genetic_lcdm_nb4_noise-full', 'lcdm',
          [_base('CGA', noise='none')])
    import io
    from contextlib import redirect_stdout
    buf = io.StringIO()
    with redirect_stdout(buf):
        gr.warn_inconsistent_labels(gr.load_campaign(str(tmp_path)))
    out = buf.getvalue()
    assert 'B-PROV-GEN' in out and 'full' in out


def test_shift_is_measured_in_sigmas_of_the_ideal_run(tmp_path):
    """Dividing by the NOISY sigma would hide the bias behind its spread."""
    ideal = _base('QMCMC 100%', om='0.2700', Om_std='0.0100')
    noisy = _base('QMCMC 100%', om='0.2750', Om_std='1.0000')
    d = gr.shift_in_sigmas(noisy, ideal)
    assert d == pytest.approx(0.5, abs=1e-9)


def test_shift_takes_the_worst_parameter(tmp_path):
    """If the noise biases even one parameter, the result is already biased."""
    ideal = _base('QMCMC 100%', om='0.2700', Om_std='0.0100')
    ideal['H0_mean'], ideal['H0_std'] = '69.00', '1.00'
    noisy = _base('QMCMC 100%', om='0.2700')
    noisy['H0_mean'] = '72.00'
    assert gr.shift_in_sigmas(noisy, ideal) == pytest.approx(3.0)


def test_cell_without_ideal_does_not_invent_a_shift(tmp_path):
    """Without an ideal run to compare against, the metric is None, not 0."""
    _task(str(tmp_path), 'samplers_lcdm_nqpp3_noise-readout', 'lcdm',
          [_base('QMCMC 100%', noise='readout')])
    rows = gr.load_campaign(str(tmp_path))
    idx = gr.ideal_cell_index(rows)
    assert gr.metric_value(rows[0], 'shift', idx) is None


def test_figures_are_made_and_do_not_mix_families(tmp_path):
    """nqpp and n_bits do not share an axis: each family gets its own figure."""
    for nb in (4, 5):
        for noi in ('none', 'readout'):
            _task(str(tmp_path), f'genetic_lcdm_nb{nb}_noise-{noi}', 'lcdm',
                  [_base('CGA', om=f'0.27{nb}0', chi2_grid='1064.6'),
                   _base('QGA (q=100%)', om=f'0.27{nb}1',
                         chi2_grid='1064.9')])
    for nq in (3, 4):
        for noi in ('none', 'readout'):
            _task(str(tmp_path), f'samplers_lcdm_nqpp{nq}_noise-{noi}',
                  'lcdm', [_base('QMCMC 100%', om=f'0.27{nq}0',
                                 acceptance='0.48')])
    out_dir = tmp_path / 'figs'
    made = gr.build([str(tmp_path)], str(out_dir))
    assert made
    names = {os.path.basename(h) for h in made}
    assert any('genetic' in n and 'grid' in n for n in names)
    assert any('samplers' in n and 'grid' in n for n in names)
    for h in made:
        assert os.path.getsize(h) > 5000


# ═══ [B-EPS15] the QVMC faithful cell must be truly faithful ════════════

def test_b_eps15_quantum_normalization_matches_bit_for_bit(tmp_path):
    """`QVMC 67%` and `QVMC 100%` differ only in this component.

    The classical branch does `P/total`; the quantum one did
    `P/(total+1e-15)`. With `max(P)=1` that is between 0 and 2 ULPs —
    invisible unless variational training amplifies it, which is exactly
    what it did in `samplers_pede_nqpp*_noise-fakebrisbane`.
    """
    import numpy as np
    rng = np.random.default_rng(0)
    for n in (64, 256, 4096):
        P = np.exp(-rng.random(n) * 20.0)
        P[0] = 1.0                      # as build_target leaves it: max(P) = 1
        classical = P / P.sum()
        quantum = cmq.quantum_amplitude_normalization(P)
        assert np.array_equal(classical, quantum), (
            f"n={n}: the quantum normalization is not bit for bit the "
            f"classical one; max difference "
            f"{np.abs(classical - quantum).max():.3e}")


def test_b_eps15_still_normalizes():
    """Removing the epsilon must not break the obvious: the output sums to 1."""
    import numpy as np
    P = np.array([1.0, 0.5, 0.25, 0.125])
    out = cmq.quantum_amplitude_normalization(P)
    assert abs(float(out.sum()) - 1.0) < 1e-15
