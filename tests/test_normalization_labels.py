"""The QVMC normalization rung is labelled for what it is, and nothing else moves.

The 100 % QVMC rung only adds the normalization switch, whose circuit output
is discarded (`quantum_amplitude_normalization`), so it equals the 67 % rung
by construction (ERRATA section 5). These tests check that:

* figures and printed tables say so, while CSV values stay `QVMC 100%`;
* the sigma-vs-grid comparison follows the 67 % rung by default and falls
  back to the identical 100 % rows when only those exist;
* `--save-samples` is opt-in, and with it `--replot-ladder` redraws the
  corner and 1-to-1 figures that a CSV alone cannot.
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'tests'))

import campaign_io  # noqa: E402


def test_display_method_only_relabels_the_normalization_rung():
    assert campaign_io.display_method('QVMC 100%') == \
        'QVMC 100% (classical norm., = 67%)'
    for name in ('QVMC 67%', 'QVMC 33%', 'QMCMC 100%', 'Classical VI', 'CGA'):
        assert campaign_io.display_method(name) == name


def test_qvmc_rung_label():
    pytest.importorskip('qiskit')
    import cosmo_modular_quantum as cmq
    assert cmq.qvmc_rung_label(67) == 'QVMC 67%'
    assert cmq.qvmc_rung_label(100) == 'QVMC 100% (classical norm., = 67%)'
    assert cmq.qvmc_rung_label(100, short=True) == '100% (classical norm., = 67%)'
    assert 'Fully Quantum' not in cmq.quantumness_label(100)
    assert 'Fully Quantum' not in cmq.PRESETS[100]['label']


def _rows(methods):
    rows = []
    for g in (3, 4):
        for met, sd in methods:
            rows.append({'_fam': 'samplers', '_noi': 'none', 'Method': met,
                         'Om_std': str(sd), '_grid': g, 'nqpp': str(g)})
    return rows


def test_sigma_vs_grid_follows_67_and_falls_back_to_100(monkeypatch):
    import compare_algorithms as ca
    monkeypatch.setattr(ca.campaign_io, 'cell_key',
                        lambda r: ('c', 'd', 'p', '42', 'none', 'lcdm', r['_grid']))
    both = _rows([(ca.REFERENCE, 0.010), ('QVMC 67%', 0.006),
                  ('QVMC 100%', 0.006)])
    only100 = _rows([(ca.REFERENCE, 0.010), ('QVMC 100%', 0.006)])
    expected = {'lcdm': [(3, pytest.approx(0.6)), (4, pytest.approx(0.6))]}
    assert ca.sigma_vs_grid(both) == expected
    assert ca.sigma_vs_grid(only100) == expected


def test_save_samples_is_opt_in():
    pytest.importorskip('qiskit')
    import cosmo_hpc_runner as hpc
    import cosmo_modular_quantum as cmq
    assert cmq.build_parser().parse_args([]).save_samples is False
    assert hpc.build_parser().parse_args([]).save_samples is False


def test_replot_draws_corners_only_with_saved_samples(tmp_path):
    pytest.importorskip('qiskit')
    pytest.importorskip('qiskit_aer')
    pytest.importorskip('corner')
    import matplotlib
    matplotlib.use('Agg')
    import cosmo_modular_quantum as cmq
    from test_fixes_phase3 import _task_csv

    csv_path = str(tmp_path / 'model_lcdm' / 'results_config.csv')
    _task_csv(csv_path)
    out = os.path.dirname(csv_path)

    cmq.replot_ladder_from_csv(csv_path)
    assert not os.path.exists(os.path.join(out, 'corner_ladder_qvmc_lcdm.png'))

    rng = np.random.default_rng(0)
    model = cmq.MODELS['lcdm']
    flat = rng.normal([0.276, 69.6], [0.01, 0.8], size=(2000, 2))
    grid = rng.normal([0.276, 69.6], [0.01, 0.8], size=(64, 2))
    qmcmc = [{'pct': p, 'flat': flat} for p in (0.0, 50.0, 100.0)]
    qvmc = [{'pct': p, 'S': grid, 'W': np.full(64, 1 / 64)}
            for p in (0.0, 33.0, 67.0, 100.0)]
    path = cmq.save_ladder_samples(qmcmc, qvmc, model, out)
    assert os.path.basename(path) == 'samples_ladder_lcdm.npz'
    with np.load(path, allow_pickle=False) as z:
        assert set(z.files) == {'qmcmc_0_flat', 'qmcmc_50_flat', 'qmcmc_100_flat',
                                'qvmc_0_S', 'qvmc_0_W', 'qvmc_33_S', 'qvmc_33_W',
                                'qvmc_67_S', 'qvmc_67_W', 'qvmc_100_S', 'qvmc_100_W'}

    cmq.replot_ladder_from_csv(csv_path)
    for f in ('corner_ladder_qvmc_lcdm.png', 'corner_ladder_qmcmc_lcdm.png',
              'corner_ladder_1to1_qvmc_lcdm_q100.png', 'ladder_trends_lcdm.png'):
        assert os.path.exists(os.path.join(out, f)), f
