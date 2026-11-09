"""scripts/errata_view.py: errata rows replace the invalid rows of a single-seed campaign."""
import csv
import importlib.util
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
_spec = importlib.util.spec_from_file_location('errata_view', os.path.join(ROOT, 'scripts', 'errata_view.py'))
ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ev)

COLS = ['Method', 'Om_mean', 'Om_std', 'H0_mean', 'H0_std', 'acceptance', 'ESS', 'noise',
        'proposal_route', 'seed']


def _task(root, name, mod, rows, fname='resultados_config.csv'):
    d = os.path.join(root, name, f'model_{mod}')
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, fname), 'w', newline='') as fh:
        w = csv.DictWriter(fh, COLS)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, '') for c in COLS})


def _row(m, om, noise='none', route='auto'):
    return dict(Method=m, Om_mean=om, Om_std='0.01', H0_mean='69.6', H0_std='0.8',
                acceptance='0.45', ESS='6000', noise=noise, proposal_route=route, seed='42')


def test_view_replaces_qmcmc_and_fakebrisbane(tmp_path):
    old, err, out = str(tmp_path / 'old'), str(tmp_path / 'err'), str(tmp_path / 'out')
    _task(old, 'samplers_lcdm_nqpp3_noise-none', 'lcdm',
          [_row('Classical MCMC', '0.276'), _row('QMCMC 100%', '0.270'), _row('Classical VI', '0.275')])
    _task(old, 'samplers_lcdm_nqpp3_noise-fakebrisbane', 'lcdm',
          [_row('Classical VI', '0.300', 'fakebrisbane')])
    _task(old, 'samplers_pede_nqpp5_noise-fakebrisbane', 'pede',
          [_row('Classical VI', '0.300', 'fakebrisbane')])
    _task(err + '/errata_qm1_ideal', 'samplers_lcdm_nqpp3', 'lcdm',
          [_row('Classical MCMC', '0.276'), _row('QMCMC 100%', '0.277')], 'results_config.csv')
    _task(err + '/errata_qm1_counts', 'samplers_lcdm_nqpp3', 'lcdm',
          [_row('Classical MCMC', '0.276', route='counts'), _row('QMCMC 100%', '0.278', route='counts')],
          'results_config.csv')
    _task(err + '/errata_fb_samplers_2p', 'samplers_lcdm_nqpp3', 'lcdm',
          [_row('Classical VI', '0.290', 'fakebrisbane')], 'results_config.csv')
    rep = ev.build(old, err, out)
    import campaign_io
    rows = campaign_io.read_campaign(out, warn=False)
    by = {(r['_noi'], r['Method']): r['Om_mean'] for r in rows}
    assert by[('none', 'QMCMC 100%')] == '0.277'
    assert by[('none', 'Classical VI')] == '0.275'
    assert by[('fakebrisbane', 'Classical VI')] == '0.290'
    assert by[('none-counts', 'QMCMC 100%')] == '0.278'
    assert not any(r['_mod'] == 'pede' for r in rows)
    assert any(x.startswith('dropped   samplers_pede_nqpp5_noise-fakebrisbane') for x in rep)
    assert any('Classical MCMC identical: True' in x for x in rep)
