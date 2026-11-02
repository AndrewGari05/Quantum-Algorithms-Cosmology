"""seed_figures.py on a small synthetic multi-seed campaign (no QVMC 100%)."""
import contextlib
import io
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import seed_figures  # noqa: E402
from test_seed_tools import _campaign  # noqa: E402

LOG = ('[C-MCMC] step  500/1000 | acc=0.48 | R-hat-1=+0.0400 | mean: Om=0.27\n'
       '[C-MCMC] step 1000/1000 | acc=0.48 | R-hat-1=+0.0050 | mean: Om=0.27\n'
       '[QVMC33] iter  0/1000 | KL=5.0 | E[theta]\n'
       '[QVMC33] iter  500/1000 | KL=0.5 | E[theta]\n')


def _with_logs_and_npz(root):
    folders = _campaign(root, seeds=(1, 2, 3), grids=(3, 4))
    rng = np.random.default_rng(0)
    for sd in folders:
        for g in (3, 4):
            task = os.path.join(sd, f'samplers_lcdm_nqpp{g}')
            with open(os.path.join(task, 'sweep_all_x.log'), 'w') as fh:
                fh.write(LOG)
            flat = rng.normal([0.3, 70.0], [0.01, 0.8], (400, 2))
            S = rng.normal([0.3, 70.0], [0.01, 0.8], (300, 2))
            np.savez(os.path.join(task, 'model_lcdm', 'samples_ladder_lcdm.npz'),
                     qmcmc_0_flat=flat, qmcmc_50_flat=flat, qmcmc_100_flat=flat,
                     qvmc_0_S=S, qvmc_0_W=np.ones(300), qvmc_33_S=S,
                     qvmc_33_W=np.ones(300), qvmc_67_S=S, qvmc_67_W=np.ones(300))
    return folders


def test_campaign_is_read_per_cell_and_seed(tmp_path):
    _campaign(str(tmp_path))
    data, taskdirs = seed_figures.load_campaign(str(tmp_path))
    assert sorted(data) == [('lcdm', 3), ('lcdm', 4)]
    assert seed_figures.seeds_of(data) == ['1', '2', '3']
    assert 'QVMC 100%' not in data[('lcdm', 3)]
    assert set(taskdirs[('lcdm', 4)]) == {'1', '2', '3'}
    assert seed_figures.mcmc_rows_identical(data, 'lcdm', ['1', '2', '3'])


def test_seed_filter(tmp_path):
    _campaign(str(tmp_path))
    data, _ = seed_figures.load_campaign(str(tmp_path), seeds={'1', '3'})
    assert seed_figures.seeds_of(data) == ['1', '3']


def test_main_writes_every_figure(tmp_path):
    _with_logs_and_npz(str(tmp_path / 'c'))
    out = tmp_path / 'f'
    with contextlib.redirect_stdout(io.StringIO()):
        assert seed_figures.main([str(tmp_path / 'c'), '--out', str(out)]) == 0
    for name in ('sigma_vs_grid', 'qmcmc_efficiency', 'om_bias', 'paired_kl',
                 'lcdm/ladder_trends_lcdm_nqpp3', 'lcdm/ladder_summary_lcdm',
                 'lcdm/ladder_kl_qvmc_lcdm_nqpp4', 'lcdm/ladder_rhat_qmcmc_lcdm',
                 'lcdm/corner_ladder_qmcmc_lcdm', 'lcdm/corner_ladder_qvmc_lcdm_nqpp3'):
        for ext in ('.png', '.pdf'):
            assert (out / (name + ext)).exists(), name + ext
    report = (out / 'seed_figures_report.txt').read_text()
    assert 'lcdm nqpp3 QVMC 67%' in report
    assert (out / 'ladder_summary.csv').exists()
