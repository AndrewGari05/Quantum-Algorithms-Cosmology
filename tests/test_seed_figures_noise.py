"""seed_figures_noise.py on a small synthetic noise campaign (2 seeds, one cell)."""
import contextlib
import csv
import io
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import seed_figures_noise as sfn  # noqa: E402

COLS = ['Method', 'Om_mean', 'Om_std', 'H0_mean', 'H0_std', 'Time_s', 'nqpp', 'acceptance',
        'final_KL', 'ESS', 'noise', 'proposal_route', 'seed']
METHODS = ['Classical MCMC', 'QMCMC 50%', 'QMCMC 100%', 'Classical VI', 'QVMC 33%', 'QVMC 67%']


def _rows(seed, level):
    big = level == 'full' and seed == 44
    out = []
    for m in METHODS:
        quantum_mc = m == 'QMCMC 100%' and level in ('readout', 'full')
        vi = m in ('Classical VI', 'QVMC 33%', 'QVMC 67%')
        noisy = level in ('readout', 'full')
        om = 0.276 + (0.002 if quantum_mc else 0) + (0.03 if (m == 'QVMC 67%' and big) else 0)
        std = 0.010 * (1.08 if quantum_mc else 1.0) * (1.5 if (vi and noisy) else 1.0)
        out.append(dict(Method=m, Om_mean=f'{om:.6f}', Om_std=f'{std:.6f}', H0_mean='69.6',
                        H0_std='0.8', Time_s='10', nqpp='—' if m.startswith('Classical') else '3',
                        acceptance='' if vi else ('0.48' if quantum_mc else '0.47'),
                        final_KL=('0.5' if not noisy else '1.5') if vi else '', ESS='3000',
                        noise='none' if level == 'none-counts' else level,
                        proposal_route='counts' if level == 'none-counts' else 'auto', seed=str(seed)))
    return out


def _campaign(root, seeds=(43, 44)):
    for s in seeds:
        for level in ('none', 'none-counts', 'readout', 'full'):
            d = os.path.join(root, f'noise_A_s{s}', f'samplers_lcdm_nqpp3_noise-{level}', 'model_lcdm')
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, 'results_config.csv'), 'w', newline='') as fh:
                w = csv.DictWriter(fh, COLS)
                w.writeheader()
                w.writerows(_rows(s, level))


def test_loader_and_shift(tmp_path):
    _campaign(str(tmp_path))
    data, _ = sfn.load_noise(str(tmp_path))
    assert list(data) == [('lcdm', 3)]
    assert sfn.seeds_of(data, ('lcdm', 3)) == ['43', '44']
    sh = sfn.per_seed(data, ('lcdm', 3), 'full', 'QVMC 67%', sfn.shift)
    assert sh[0] < 1 and abs(sh[1] - 3.0) < 1e-6        # 0.03 / 0.010 control sigma


def test_main_writes_figures_report_and_table(tmp_path):
    _campaign(str(tmp_path / 'r'))
    out = tmp_path / 'f'
    with contextlib.redirect_stdout(io.StringIO()):
        assert sfn.main([str(tmp_path / 'r'), '--out', str(out), '--no-corners']) == 0
    for name in ('ruido_desplazamiento', 'ruido_sigma', 'ruido_diagnosticos', 'ruido_ruta',
                 'ruido_costo'):
        for ext in ('.png', '.pdf'):
            assert (out / (name + ext)).exists(), name + ext
    rep = (out / 'noise_report.txt').read_text(encoding='utf-8')
    assert 'classical MCMC identical to none at every level: 6/6' in rep
    assert 'QMCMC 50% == QMCMC 100% at readout    : 0/2' in rep
    assert 'QVMC 100% (+ quantum training)   seed 44  3.000' in rep
    rows = list(csv.DictReader(open(out / 'noise_summary_seeds.csv', encoding='utf-8')))
    assert any(r['method'] == 'QVMC 67%' and r['level'] == 'full' for r in rows)
