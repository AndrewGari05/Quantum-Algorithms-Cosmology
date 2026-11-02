"""Seed tools on campaigns run without the QVMC 100% rung.

QVMC 100% only adds the classical normalization and equals QVMC 67% by
construction (ERRATA QV-6), so newer campaigns do not run it. These tests
check that the seed comparison and the triage keep working without it:

* the classical-vs-quantum VI comparison uses QVMC 67% (it used to vanish);
* the QVMC 67% == 100% faithful pair is reported as not evaluated;
* MCMC/QMCMC rows identical across grids are compared once, with the grids;
* a task without its per-model CSV is read from the cumulative file, with a
  warning, and gives the same report.
"""
import contextlib
import csv
import io
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import compare_seeds  # noqa: E402
import triage_campaign  # noqa: E402

COLS = ['Method', 'Om_mean', 'Om_std', 'H0_mean', 'H0_std', 'Time_s', 'nqpp',
        'chi2', 'n_data', 'chi2_red', 'AIC', 'BIC', 'acceptance', 'final_KL',
        'ESS', 'dataset', 'prior', 'noise', 'proposal_route', 'seed']
CUM_COLS = ['Method', 'model', 'params', 'p1_mean', 'p1_std', 'p2_mean',
            'p2_std', 'Time_s', 'nqpp', 'chi2', 'n_data', 'chi2_red', 'AIC',
            'BIC', 'acceptance', 'final_KL', 'ESS', 'dataset', 'prior', 'noise',
            'proposal_route', 'seed']


def _rows(seed, nqpp):
    """One task: identical MCMC/QMCMC rows on every grid, VI rows per grid."""
    j = 0.001 * seed
    base = dict(chi2='1064.61', n_data='1099', chi2_red='0.97', AIC='1068.6',
                BIC='1078.6', dataset='CC+BAO+Pantheon', prior='flat',
                noise='none', proposal_route='auto', seed=str(seed), Time_s='1')
    out = []
    for met, acc, kl, om in [('Classical MCMC', 0.48 + j, '', 0.276 + j),
                             ('QMCMC 50%', 0.46 + j, '', 0.277 + j),
                             ('QMCMC 100%', 0.46 + j, '', 0.277 + j),
                             ('Classical VI', '', 0.20 + j + 0.01 * nqpp, 0.275),
                             ('QVMC 33%', '', 0.20 + j + 0.01 * nqpp, 0.275),
                             ('QVMC 67%', '', 0.50 + 3 * j + 0.01 * nqpp, 0.274)]:
        out.append({**base, 'Method': met, 'Om_mean': f'{om:.6f}',
                    'Om_std': '0.0104', 'H0_mean': '69.6', 'H0_std': '0.80',
                    'acceptance': f'{acc}' if acc != '' else '',
                    'final_KL': f'{kl}' if kl != '' else '',
                    'ESS': '6800', 'nqpp': '—' if met.startswith('Classical') else str(nqpp)})
    return out


def _campaign(root, seeds=(1, 2, 3), grids=(3, 4), cumulative_only=False):
    folders = []
    for s in seeds:
        sd = os.path.join(root, f'seeds_s{s}')
        for g in grids:
            task = os.path.join(sd, f'samplers_lcdm_nqpp{g}')
            rows = _rows(s, g)
            if cumulative_only:
                os.makedirs(task, exist_ok=True)
                with open(os.path.join(task, 'results_all_models.csv'), 'w', newline='') as fh:
                    w = csv.DictWriter(fh, CUM_COLS)
                    w.writeheader()
                    for r in rows:
                        w.writerow({**{k: r.get(k, '') for k in CUM_COLS},
                                    'model': 'lcdm', 'params': 'Om|H0',
                                    'p1_mean': r['Om_mean'], 'p1_std': r['Om_std'],
                                    'p2_mean': r['H0_mean'], 'p2_std': r['H0_std']})
            else:
                os.makedirs(os.path.join(task, 'model_lcdm'), exist_ok=True)
                with open(os.path.join(task, 'model_lcdm', 'results_config.csv'), 'w', newline='') as fh:
                    w = csv.DictWriter(fh, COLS)
                    w.writeheader()
                    w.writerows(rows)
        folders.append(sd)
    return folders


def _report(folders):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        compare_seeds.review(folders)
    return out.getvalue(), err.getvalue()


def test_vi_comparison_uses_qvmc67_when_100_was_not_run(tmp_path):
    out, _ = _report(_campaign(str(tmp_path)))
    assert 'Classical VI vs QVMC 67%  [final_KL]' in out
    assert 'lcdm nqpp3 / none' in out and 'lcdm nqpp4 / none' in out
    assert 'QVMC 67% == QVMC 100%   not evaluated' in out


def test_identical_mcmc_rows_across_grids_are_compared_once(tmp_path):
    out, _ = _report(_campaign(str(tmp_path)))
    acc = [ln for ln in out.splitlines() if 'Classical MCMC vs QMCMC 100%' in ln]
    assert len(acc) == 1
    assert 'lcdm nqpp3, nqpp4 / none (identical rows on every grid listed)' in out


def test_cumulative_csv_fallback_gives_the_same_report(tmp_path):
    per_model, err0 = _report(_campaign(str(tmp_path / 'a')))
    cumulative, err1 = _report(_campaign(str(tmp_path / 'b'), cumulative_only=True))
    assert err0 == '' and 'WARNING' in err1 and 'results_all_models.csv' in err1
    strip = lambda t: t.replace(str(tmp_path / 'a'), '').replace(str(tmp_path / 'b'), '')
    assert strip(per_model) == strip(cumulative)


def test_per_model_csv_wins_over_cumulative(tmp_path):
    folders = _campaign(str(tmp_path))
    for sd in folders:                     # add a cumulative copy with junk values
        for g in (3, 4):
            p = os.path.join(sd, f'samplers_lcdm_nqpp{g}', 'results_all_models.csv')
            with open(p, 'w', newline='') as fh:
                w = csv.DictWriter(fh, CUM_COLS)
                w.writeheader()
                w.writerow({k: '' for k in CUM_COLS} | {'Method': 'Classical VI',
                           'model': 'lcdm', 'seed': '1', 'final_KL': '99'})
    out, err = _report(folders)
    assert err == '' and '99' not in out


def test_named_param_columns():
    r = compare_seeds.named_param_columns(
        {'params': 'Om|H0|w0|wa', 'p3_mean': '-1', 'p4_std': '0.2', 'Method': 'X'})
    assert r == {'params': 'Om|H0|w0|wa', 'w0_mean': '-1', 'wa_std': '0.2', 'Method': 'X'}


def test_triage_says_when_the_qvmc_pair_was_not_run(tmp_path):
    folders = _campaign(str(tmp_path), seeds=(1,))
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        triage_campaign.main([folders[0]])
    text = out.getvalue()
    assert 'QVMC 67% == QVMC 100%: not evaluated, QVMC 100% was not run' in text
    assert 'QMCMC 50%      == QMCMC 100%' in text
