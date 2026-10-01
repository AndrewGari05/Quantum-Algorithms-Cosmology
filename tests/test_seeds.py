"""
test_seeds.py — tests for `compare_seeds.py` and for the n_bits axis.

The seed comparator exists to answer a single question: is the difference I
want to report larger than how much the number moves just by changing the
seed? Everything locked down here are the ways that answer can come out
wrong WITHOUT failing:

  * pairing the methods wrongly (and then the comparison never shows up);
  * comparing by columns that do not exist (and then everything is "identical");
  * calling significant a difference that fits inside the spread.

Standard library only, like the script.
"""
import csv
import importlib.util
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_spec = importlib.util.spec_from_file_location(
    'compare_seeds', os.path.join(ROOT, 'compare_seeds.py'))
cs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cs)

COLS = ['Method', 'Om_mean', 'Om_std', 'H0_mean', 'H0_std', 'Time_s', 'nqpp',
        'chi2', 'n_data', 'chi2_red', 'AIC', 'BIC', 'acceptance', 'final_KL',
        'ESS', 'dataset', 'prior', 'noise', 'proposal_route', 'seed',
        'budget_mode', 'circuits_train', 'chi2_grid']


def _row(method, seed, om='0.2764', kl='', acc='', nqpp='3', **extra):
    r = dict.fromkeys(COLS, '')
    r.update({'Method': method, 'Om_mean': om, 'Om_std': '0.0104',
              'H0_mean': '69.60', 'H0_std': '0.80', 'nqpp': nqpp,
              'chi2': '1064.61', 'n_data': '1099', 'chi2_red': '0.970',
              'AIC': '1068.6', 'BIC': '1078.6', 'dataset': 'CC+BAO+Pantheon',
              'prior': 'flat', 'noise': 'none', 'seed': str(seed),
              'final_KL': kl, 'acceptance': acc, 'ESS': '6900',
              'budget_mode': 'circuits'})
    r.update(extra)
    return r


def _campaign(root, seed, rows, sub='model_lcdm', name='results_config.csv'):
    """Write a single-seed run with the real layout."""
    d = os.path.join(root, f'seed_{seed}', sub)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, name), 'w', newline='') as fh:
        w = csv.DictWriter(fh, COLS)
        w.writeheader()
        w.writerows(rows)


def _capture(folders):
    import io
    from contextlib import redirect_stdout
    buf = io.StringIO()
    with redirect_stdout(buf):
        cs.review(folders)
    return buf.getvalue()


# ── statistics ───────────────────────────────────────────────────────────

def test_the_spread_is_sample_not_population():
    """With n seeds the divisor is n-1: underestimating sigma inflates the
    significance of everything that is reported."""
    m, s, n = cs.dispersion([1.0, 2.0, 3.0])
    assert (round(m, 6), round(s, 6), n) == (2.0, 1.0, 3)


def test_a_single_seed_has_no_spread():
    assert cs.dispersion([5.0]) == (5.0, 0.0, 1)
    assert cs.in_sigmas([1.0], [2.0])[2] is None


def test_small_difference_relative_to_the_spread_is_not_significant():
    """Two overlapping methods: the separation must come out below 1."""
    a = [1.6250, 1.6300, 1.6200]
    b = [1.6286, 1.6240, 1.6330]
    _, _, ns = cs.in_sigmas(a, b)
    assert abs(ns) < 1.0


def test_large_difference_relative_to_the_spread_is_significant():
    # [E-QPU7] paired test: a constant offset has zero SE (t undefined) and is
    # reported as systematic; a noisy but consistent offset has a large t.
    d, _, ns = cs.in_sigmas([1.00, 1.01, 0.99], [2.00, 2.01, 1.99])
    assert ns is None and d == -1.0
    _, _, ns = cs.in_sigmas([1.00, 1.02, 0.99], [2.00, 2.01, 1.98])
    assert abs(ns) > 3.0


def test_paired_test_detects_offset_hidden_by_seed_spread():
    # QPU-7 reproducer: per-seed spread is large, the paired offset is not.
    a = [0.30, 0.25, 0.35, 0.28, 0.32]
    b = [x - 0.01 + e for x, e in zip(a, [0.0005, -0.0004, 0.0003, -0.0002, 0.0001])]
    _, _, t = cs.in_sigmas(a, b)
    assert abs(t) > 10


# ── cell pairing ─────────────────────────────────────────────────────────

def test_the_classical_rung_pairs_despite_a_different_nqpp(tmp_path):
    """`Classical VI` writes nqpp='—' and `QVMC 100%` writes the number.

    Grouping by nqpp each one fell into its own group and the
    classical-vs-quantum comparison was never made: the report did not say
    'not significant', it said nothing.
    """
    for s in (1, 2, 3):
        _campaign(str(tmp_path), s, [
            _row('Classical VI', s, kl=str(1.60 + 0.001 * s), nqpp='—'),
            _row('QVMC 100%', s, kl=str(2.60 + 0.001 * s), nqpp='3'),
        ])
    out = _capture([str(tmp_path / f'seed_{s}') for s in (1, 2, 3)])
    assert 'Classical VI vs QVMC 100%' in out
    assert 'paired' in out


def test_faithful_cell_failing_on_a_single_seed_is_reported(tmp_path):
    """Matching on two seeds and failing on the third is a bug, not chance."""
    for s in (1, 2, 3):
        om_b = '0.2764' if s != 3 else '0.9999'
        _campaign(str(tmp_path), s, [
            _row('QMCMC 50%', s), _row('QMCMC 100%', s, om=om_b)])
    out = _capture([str(tmp_path / f'seed_{s}') for s in (1, 2, 3)])
    assert 'CHECK' in out
    assert 'seed=3' in out


def test_intact_faithful_cells_pass(tmp_path):
    for s in (1, 2, 3):
        _campaign(str(tmp_path), s,
                  [_row('QMCMC 50%', s), _row('QMCMC 100%', s)])
    out = _capture([str(tmp_path / f'seed_{s}') for s in (1, 2, 3)])
    assert 'CHECK' not in out
    assert '3/3   OK' in out


def test_legacy_and_new_csv_names_pair_across_seeds(tmp_path):
    """A legacy seed folder (`resultados_config.csv`) pairs cell by cell with
    new ones (`results_config.csv`), and cumulative files are never read."""
    for s, name in ((1, 'resultados_config.csv'), (2, 'results_config.csv'),
                    (3, 'results_config.csv')):
        _campaign(str(tmp_path), s,
                  [_row('QMCMC 50%', s), _row('QMCMC 100%', s)], name=name)
    for s, name in ((1, 'resultados_TODOS_los_modelos.csv'),
                    (2, 'results_all_models.csv')):
        with open(tmp_path / f'seed_{s}' / name, 'w', newline='') as fh:
            w = csv.DictWriter(fh, COLS)
            w.writeheader()
            w.writerow(_row('QMCMC 100%', s, om='0.9999'))
    out = _capture([str(tmp_path / f'seed_{s}') for s in (1, 2, 3)])
    assert 'CHECK' not in out
    assert '3/3   OK' in out


def test_a_single_seed_warns_instead_of_inventing_error_bars(tmp_path):
    _campaign(str(tmp_path), 42, [_row('QMCMC 50%', 42)])
    out = _capture([str(tmp_path / 'seed_42')])
    assert 'ONE seed' in out


def test_campaign_without_seed_column_is_rejected(tmp_path):
    """Campaigns before [B-PROV] are no use for this study."""
    d = tmp_path / 'old' / 'model_lcdm'
    d.mkdir(parents=True)
    no_seed = [c for c in COLS if c != 'seed']
    with open(d / 'resultados_x.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, no_seed)
        w.writeheader()
        w.writerow({k: v for k, v in _row('QMCMC 50%', 1).items()
                    if k != 'seed'})
    out = _capture([str(tmp_path / 'old')])
    assert 'seed' in out and 'corrected' in out


# ── [B-NBITS] the genetic axis ───────────────────────────────────────────

def test_b_nbits_the_families_are_separated():
    """nqpp and n_bits are not the same quantity and do not share an axis."""
    import cosmo_hpc_runner as R
    assert R._method_family('CGA') == 'genetic'
    assert R._method_family('QGA (q=33%)') == 'genetic'
    assert R._method_family('QMCMC 50%') == 'samplers'
    assert R._method_family('QVMC 100%') == 'samplers'
    assert R._method_family('Classical VI') == 'samplers'
    assert R._method_family('Classical MCMC') == 'samplers'


@pytest.mark.parametrize('family,suffix,methods', [
    ('samplers', '', ['QMCMC 50%', 'QMCMC 100%']),
    ('genetic', '_genetic', ['CGA', 'QGA (q=33%)']),
])
def test_b_nbits_each_family_produces_its_figure(tmp_path, family, suffix,
                                                 methods):
    """And the x axis only carries the grid values of THAT family.

    Without the filter, the genetic figure inherited the samplers' nqpp
    values and came out half empty on a scale that belonged to neither.
    """
    import matplotlib
    matplotlib.use('Agg')
    import cosmo_hpc_runner as R

    prefix = 'samplers' if family == 'samplers' else 'genetic'
    for g in (4, 5):
        # The runner infers the model from the task folder name
        # (`samplers_<model>_...`), not from a CSV column.
        d = tmp_path / f'{prefix}_lcdm_g{g}' / 'model_lcdm'
        d.mkdir(parents=True)
        with open(d / 'results_config.csv', 'w', newline='') as fh:
            w = csv.DictWriter(fh, COLS)
            w.writeheader()
            for m in methods:
                w.writerow(_row(m, 42, om=f'0.2{70+g}', nqpp=str(g)))
    out_dir = tmp_path / 'fig'
    made = R.generate_convergence_plots(
        str(tmp_path), outdir=str(out_dir),
        xlabel='nqpp' if family == 'samplers' else 'n_bits',
        family=family, suffix=suffix)
    expected = os.path.join(str(out_dir), f'convergence_lcdm{suffix}.png')
    assert expected in made, f"{expected} was not generated: {made}"
    assert os.path.getsize(expected) > 5000
