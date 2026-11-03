"""seed_figures_genetic.py on a small synthetic multi-seed genetic campaign."""
import contextlib
import csv
import io
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import seed_figures_genetic as sfg  # noqa: E402

COLS = ['Method', 'Om_mean', 'Om_std', 'H0_mean', 'H0_std', 'Time_s', 'nqpp', 'chi2',
        'n_data', 'chi2_red', 'AIC', 'BIC', 'acceptance', 'final_KL', 'ESS', 'dataset',
        'prior', 'noise', 'proposal_route', 'seed', 'chi2_grid']
GRID = {'CGA': 1064.61, 'QGA (q=0%)': 1064.61, 'QGA (q=33%)': 1064.61,
        'QGA (q=67%)': 1064.97, 'QGA (q=100%)': 1064.97}


def _log(seed, nb):
    out = []
    for meth, g in GRID.items():
        tag, q = ('CGA', 0) if meth == 'CGA' else ('QGA', meth.split('=')[1][:-2])
        out.append(f'[{tag}] start | pop=5 gens=20 | model=Flat ΛCDM | quantumness={q}%')
        for gen, b in ((0, g + 0.5), (10, g + 0.01 * seed / 50), (19, g)):
            out.append(f'[{tag}] gen {gen:4d}/20 | best χ²={b:.3f} | mean χ²={b + 5:.3f} | '
                       f'Om=0.27{nb}  H0=69.5{seed % 10}')
        out.append(f'[{tag}]   best on the grid (NOT refined): Om=0.27{nb}  H0=69.5 | '
                   f'χ²={g:.3f} (refinement improved it by +0.000)')
    return '\n'.join(out) + '\n'


def _campaign(root, seeds=(43, 44, 45), grids=(4, 5)):
    for s in seeds:
        for nb in grids:
            task = os.path.join(root, f'genetic_s{s}', f'genetic_lcdm_nb{nb}')
            os.makedirs(os.path.join(task, 'model_lcdm'), exist_ok=True)
            with open(os.path.join(task, 'model_lcdm', 'results_config.csv'), 'w', newline='') as fh:
                w = csv.DictWriter(fh, COLS)
                w.writeheader()
                for meth, g in GRID.items():
                    w.writerow(dict(Method=meth, Om_mean='0.276', Om_std='0.0015', H0_mean='69.6',
                                    H0_std='0.11', Time_s='3', nqpp='—' if meth == 'CGA' else nb,
                                    chi2='1064.6099', n_data='1099', chi2_red='0.97', AIC='1068.6',
                                    BIC='1078.6', ESS='340', dataset='CC+BAO+Pantheon', prior='flat',
                                    noise='none', proposal_route='auto', seed=str(s),
                                    chi2_grid=f'{g:.4f}'))
            with open(os.path.join(task, 'genetic_sweep_all_x.log'), 'w', encoding='utf-8') as fh:
                fh.write(_log(s, nb))
            open(os.path.join(task, 'model_lcdm', 'genetic_evolution_lcdm.gif'), 'w').close()


def test_log_parser_reads_every_method(tmp_path):
    _campaign(str(tmp_path), seeds=(43,), grids=(4,))
    r = sfg.parse_genetic_log(str(tmp_path / 'genetic_s43' / 'genetic_lcdm_nb4' / 'genetic_sweep_all_x.log'))
    assert list(r) == sfg.METHODS
    assert r['QGA (q=67%)']['gen'] == [0, 10, 19]
    assert r['QGA (q=67%)']['grid_chi2'] == 1064.97
    assert r['CGA']['params']['Om'] == [0.274, 0.274, 0.274]


def test_paired_delta_and_zero_spread(tmp_path):
    _campaign(str(tmp_path))
    import seed_figures as sf
    data, _ = sf.load_campaign(str(tmp_path), pattern='genetic_s*', grid='nb')
    seeds = sf.seeds_of(data)
    cell = data[('lcdm', 4)]
    p = sfg.paired(cell, 'QGA (q=67%)', seeds)
    assert abs(p['mean'] - 0.36) < 1e-9 and p['t'] is None and p['worse'] == 3
    assert sfg.zero_sd(sfg.delta_vs_cga(cell, 'QGA (q=67%)', seeds))
    assert sfg.identical_rows(cell, 'CGA', 'QGA (q=0%)', seeds)


def test_main_writes_figures_tables_and_gif_index(tmp_path):
    _campaign(str(tmp_path / 'r'))
    out = tmp_path / 'f'
    with contextlib.redirect_stdout(io.StringIO()):
        assert sfg.main([str(tmp_path / 'r'), '--out', str(out)]) == 0
    for name in ('genetico_chi2grid_contra_nbits', 'lcdm/convergence_lcdm_genetico',
                 'lcdm/cost_lcdm_genetico', 'lcdm/fitness_lcdm_nb4_genetico',
                 'lcdm/genetic_convergence_lcdm_nb5_genetico'):
        for ext in ('.png', '.pdf'):
            assert (out / (name + ext)).exists(), name + ext
    rep = (out / 'genetic_report.txt').read_text(encoding='utf-8')
    assert 'CGA == QGA (q=0%) (every value, every seed): 2/2 cells' in rep
    assert '0 mismatches' in rep
    page = (out / 'gif_index.html').read_text(encoding='utf-8')
    assert page.count('href=') == 6
    assert (out / 'genetic_summary.csv').exists()
