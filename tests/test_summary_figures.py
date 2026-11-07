"""summary_figures.py on small synthetic campaigns (multi-seed + single-seed)."""
import contextlib
import csv
import io
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import summary_figures  # noqa: E402
from test_seed_figures import _with_logs_and_npz  # noqa: E402
from test_seed_tools import COLS, _rows  # noqa: E402


def _single_seed(root):
    """One seed, ideal grids 3..7 plus a readout task at nqpp 4."""
    for g, noise in [(3, None), (4, None), (6, None), (7, None), (4, 'readout')]:
        name = f'samplers_lcdm_nqpp{g}' + (f'_noise-{noise}' if noise else '')
        d = os.path.join(root, name, 'model_lcdm')
        os.makedirs(d, exist_ok=True)
        rows = _rows(42, g)
        for r in rows:
            r['noise'] = noise or 'none'
            if noise and r['Method'] == 'QVMC 67%':
                r['Om_mean'] = f"{float(r['Om_mean']) + 0.0104:.6f}"       # a 1-sigma shift
        with open(os.path.join(d, 'results_config.csv'), 'w', newline='') as fh:
            w = csv.DictWriter(fh, COLS)
            w.writeheader()
            w.writerows(rows)
    return root


def test_noise_shifts_measure_against_the_ideal_cell(tmp_path):
    import noise_plots
    rows = noise_plots.load_campaign(_single_seed(str(tmp_path / 'one')))
    sh = summary_figures.noise_shifts(rows)
    assert {s[0] for s in sh} == {'readout'}
    q = [s[4] for s in sh if s[1] == 'QVMC 67%']
    assert len(q) == 1 and abs(q[0] - 1.0) < 1e-6
    assert all(s[4] < 1e-9 for s in sh if s[1] != 'QVMC 67%')


def test_main_writes_the_summary_figures(tmp_path):
    _with_logs_and_npz(str(tmp_path / 'multi'))
    one = _single_seed(str(tmp_path / 'one'))
    out = tmp_path / 'f'
    with contextlib.redirect_stdout(io.StringIO()):
        assert summary_figures.main([str(tmp_path / 'multi'), '--out', str(out),
                                     '--single-seed', one]) == 0
    for name in ('resumen_qmcmc', 'resumen_qvmc', 'resumen_posteriores_qvmc',
                 'resumen_kl_qvmc', 'resumen_rhat_qmcmc', 'resumen_ruido'):
        for ext in ('.png', '.pdf'):
            assert (out / (name + ext)).exists(), name + ext
    rep = (out / 'summary_report.txt').read_text(encoding='utf-8')
    assert 'single-seed sigma lcdm: nqpp6' in rep and 'nqpp7' in rep
    assert 'noise readout' in rep


def test_comparison_figure_puts_every_family_on_one_page(tmp_path):
    from test_seed_figures_genetic import _campaign
    _with_logs_and_npz(str(tmp_path / 'multi'))
    _campaign(str(tmp_path / 'gen'))
    out = tmp_path / 'f'
    with contextlib.redirect_stdout(io.StringIO()):
        assert summary_figures.main([str(tmp_path / 'multi'), '--out', str(out),
                                     '--genetic', str(tmp_path / 'gen')]) == 0
    for ext in ('.png', '.pdf'):
        assert (out / ('resumen_comparacion' + ext)).exists()
    rep = (out / 'summary_report.txt').read_text(encoding='utf-8')
    assert 'compare shift lcdm  QMCMC 100%' in rep
    assert 'compare dchi2 lcdm  QGA (q=67%)' in rep
