"""
test_triage.py — tests for `triage_campaign.py`.

The triage computes no physics: it decides WHAT in a campaign is citable and
whether the faithful cells still coincide. So what has to be locked down is
not numbers but decisions, and every one of them here failed the first time
the script was run against real outputs.

The schemas below are the REAL ones, copied from the header the two modules
write. They do not match each other, and that is exactly the trap:

  * samplers  -> `Om_mean`, `H0_mean`, ... (per-parameter names), no
    `model` column.
  * genetic   -> `p1_mean`..`p4_std` (generic names), WITH a `model` and a
    `params` column.

A triage that hard-codes the column names gets one right and silently fails
on the other: `row.get('p1_mean')` gives None on both sides of a samplers
comparison, None == None is True, and the faithful cell is declared OK
without having checked anything. That is worse than not checking.

Standard library only, like the script.
"""
import csv
import importlib.util
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_spec = importlib.util.spec_from_file_location(
    'triage_campaign', os.path.join(ROOT, 'triage_campaign.py'))
triage = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(triage)


# Real header of cosmo_modular_quantum.py (samplers).
COLS_SAMPLERS = ['Method', 'Om_mean', 'Om_std', 'H0_mean', 'H0_std', 'Time_s',
                 'nqpp', 'chi2', 'n_data', 'chi2_red', 'AIC', 'BIC',
                 'acceptance', 'final_KL', 'ESS', 'dataset', 'prior', 'noise',
                 'proposal_route', 'seed', 'budget_mode', 'circuits_train',
                 'chi2_grid']

# Real header of cosmo_genetic_optimizers.py (--sweep-all).
COLS_GENETIC = ['Method', 'model', 'params', 'p1_mean', 'p1_std', 'p2_mean',
                'p2_std', 'p3_mean', 'p3_std', 'p4_mean', 'p4_std', 'Time_s',
                'nqpp', 'chi2', 'n_data', 'chi2_red', 'AIC', 'BIC',
                'acceptance', 'final_KL', 'ESS', 'dataset', 'prior', 'noise',
                'proposal_route', 'seed', 'budget_mode', 'circuits_train',
                'chi2_grid']

# The same headers before 2026-09-04: no provenance columns.
COLS_SAMPLERS_OLD = [c for c in COLS_SAMPLERS
                     if c not in triage.NEW_COLUMNS]

FIT = {'chi2': '27.4691', 'n_data': '51', 'chi2_red': '0.5606',
       'AIC': '31.4691', 'BIC': '35.3328', 'dataset': 'CC+BAO',
       'prior': 'flat', 'noise': 'none'}


def _samplers(method, om='0.269366', h0='69.988624', nqpp='2', **extra):
    """Samplers row with the real column names."""
    r = dict.fromkeys(COLS_SAMPLERS, '')
    r.update(FIT)
    r.update({'Method': method, 'Om_mean': om, 'Om_std': '0.010',
              'H0_mean': h0, 'H0_std': '1.0', 'nqpp': nqpp,
              'seed': '42', 'budget_mode': 'circuits',
              'proposal_route': 'auto', 'circuits_train': '2281'})
    r.update(extra)
    return r


def _genetic(method, p1='0.262709', p2='70.388775', nqpp='4', **extra):
    """Genetic sweep row with the real column names."""
    r = dict.fromkeys(COLS_GENETIC, '')
    r.update(FIT)
    r.update({'Method': method, 'model': 'lcdm', 'params': 'Om|H0',
              'p1_mean': p1, 'p1_std': '0.001124', 'p2_mean': p2,
              'p2_std': '0.175031', 'nqpp': nqpp, 'seed': '42',
              'budget_mode': 'circuits', 'proposal_route': 'auto',
              'circuits_train': '0', 'chi2_grid': '27.5367'})
    r.update(extra)
    return r


def _write(path, cols, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, cols, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


def _capture(master):
    """Run `review` on a folder and return what it printed."""
    import io
    from contextlib import redirect_stdout
    buf = io.StringIO()
    with redirect_stdout(buf):
        triage.review(master)
    return buf.getvalue()


# ── 1. task status ───────────────────────────────────────────────────────

def test_log_cut_at_the_signature_reads_as_oom(tmp_path):
    """A log that ends at the QVMC signature is a dead task, not a live one."""
    task = tmp_path / 'samplers_cpl_nqpp5_CC+BAO'
    task.mkdir()
    (task / 'task.log').write_text('earlier\n[i] ' + triage.OOM_SIGNATURE + '\n')
    assert triage.task_status(str(task))[0] == 'oom'


def test_log_with_progress_reads_as_running(tmp_path):
    task = tmp_path / 'samplers_lcdm_nqpp5_CC+BAO'
    task.mkdir()
    (task / 'task.log').write_text('step 300/1000 chain 2\n')
    assert triage.task_status(str(task))[0] == 'running'


def test_with_a_csv_the_task_is_finished_even_if_the_log_looks_bad(tmp_path):
    task = tmp_path / 'genetic_lcdm_nb6_noise-none'
    task.mkdir()
    (task / 'task.log').write_text('[i] ' + triage.OOM_SIGNATURE + '\n')
    _write(str(task / 'resultados_lcdm.csv'), COLS_SAMPLERS,
           [_samplers('QMCMC 100%')])
    assert triage.task_status(str(task))[0] == 'ok'


def test_with_a_new_name_csv_the_task_is_finished(tmp_path):
    task = tmp_path / 'genetic_lcdm_nb6_noise-none'
    task.mkdir()
    (task / 'task.log').write_text('[i] ' + triage.OOM_SIGNATURE + '\n')
    _write(str(task / 'results_lcdm.csv'), COLS_SAMPLERS,
           [_samplers('QMCMC 100%')])
    assert triage.task_status(str(task))[0] == 'ok'


def test_the_qubit_suffix_multiplies_by_the_model_dimension():
    # cpl has 4 parameters: 5 qubits per parameter are 20, which is exactly
    # where the August campaign ran out of RAM.
    assert triage._qubit_suffix('samplers_cpl_nqpp5_CC+BAO') == '  (20 qubits)'
    assert triage._qubit_suffix('samplers_lcdm_nqpp5_CC+BAO') == '  (10 qubits)'
    assert triage._qubit_suffix('genetic_lcdm_nb6_noise-none') == ''


# ── 2. discovery of columns and of the model ─────────────────────────────

def test_parameter_columns_are_discovered_not_hard_coded():
    """Both real schemas must resolve on their own."""
    assert triage.param_columns(set(COLS_SAMPLERS)) == [
        'H0_mean', 'H0_std', 'Om_mean', 'Om_std']
    assert triage.param_columns(set(COLS_GENETIC)) == [
        'p1_mean', 'p1_std', 'p2_mean', 'p2_std',
        'p3_mean', 'p3_std', 'p4_mean', 'p4_std']


def test_comparing_by_missing_columns_cannot_come_out_ok():
    """The false faithful cell bug.

    Two identical rows compared by a column neither has gave
    `None == None` -> True, and the triage reported OK without checking anything.
    """
    a = _samplers('QMCMC 50%')
    b = _samplers('QMCMC 100%')
    assert triage._rows_equal(a, b, ['Om_mean', 'H0_mean']) is True
    assert triage._rows_equal(a, b, ['p1_mean', 'p2_mean']) is False


def test_the_model_comes_from_the_path_when_the_csv_lacks_it():
    assert triage.model_from_path('run/model_wcdm/resultados_config.csv') == 'wcdm'
    assert triage.model_from_path('run/model_wcdm/results_config.csv') == 'wcdm'
    assert triage.model_from_path('run/resultados_cpl.csv') == 'cpl'
    assert triage.model_from_path('run/results_cpl.csv') == 'cpl'
    assert triage.model_from_path('run/resultados_config.csv') == '?'
    assert triage.model_from_path('run/results_config.csv') == '?'


# ── 3. citability verdict ────────────────────────────────────────────────

def test_old_campaign_is_marked_not_citable(tmp_path):
    _write(str(tmp_path / 't1' / 'resultados_lcdm.csv'),
           COLS_SAMPLERS_OLD, [_samplers('QMCMC 100%')])
    out = _capture(str(tmp_path))
    assert 'BEFORE 2026-09-04' in out
    assert 'NOT citable' in out


def test_different_schemas_give_no_false_positive(tmp_path):
    """The union of columns across the two real schemas."""
    _write(str(tmp_path / 'samplers_lcdm' / 'resultados_config.csv'),
           COLS_SAMPLERS, [_samplers('QMCMC 100%')])
    _write(str(tmp_path / 'genetic_lcdm' / 'resultados_config.csv'),
           COLS_GENETIC, [_genetic('CGA')])
    out = _capture(str(tmp_path))
    assert 'Everything citable' in out
    assert 'BEFORE' not in out


def test_folder_without_subfolders_still_reads_the_csvs(tmp_path):
    """A partial download must not abort the diagnosis."""
    _write(str(tmp_path / 'resultados_lcdm.csv'), COLS_SAMPLERS,
           [_samplers('QMCMC 100%')])
    out = _capture(str(tmp_path))
    assert 'no task subfolders' in out
    assert 'chi2_red' in out          # reached section 2
    assert 'CITABLE' in out           # and section 3


def test_new_and_legacy_names_are_read_and_cumulative_is_skipped(tmp_path):
    """Current and legacy file names are both read; a cumulative file
    (either name) is never counted."""
    _write(str(tmp_path / 'samplers_lcdm' / 'model_lcdm' / 'results_config.csv'),
           COLS_SAMPLERS, [_samplers('QMCMC 50%'), _samplers('QMCMC 100%')])
    _write(str(tmp_path / 'samplers_wcdm' / 'model_wcdm' / 'resultados_config.csv'),
           COLS_SAMPLERS, [_samplers('QMCMC 50%'), _samplers('QMCMC 100%')])
    for name in ('results_all_models.csv', 'resultados_TODOS_los_modelos.csv'):
        _write(str(tmp_path / 'samplers_lcdm' / name), COLS_SAMPLERS,
               [_samplers('QMCMC 50%'), _samplers('QMCMC 100%', om='0.9')])
    out = _capture(str(tmp_path))
    assert '2/2  OK' in out
    assert 'CHECK' not in out


# ── 4. faithful cells ────────────────────────────────────────────────────

def test_broken_faithful_pair_is_reported(tmp_path):
    """QVMC 67% and QVMC 100% must coincide: if not, there is a real error."""
    _write(str(tmp_path / 't1' / 'resultados_lcdm.csv'), COLS_SAMPLERS,
           [_samplers('QVMC 67%'),
            _samplers('QVMC 100%', om='0.999', h0='99.9')])
    assert 'CHECK' in _capture(str(tmp_path))


def test_intact_faithful_pair_passes(tmp_path):
    _write(str(tmp_path / 't1' / 'resultados_lcdm.csv'), COLS_SAMPLERS,
           [_samplers('QMCMC 50%'), _samplers('QMCMC 100%')])
    out = _capture(str(tmp_path))
    assert 'CHECK' not in out
    assert '1/1  OK' in out


def test_the_classical_rung_is_not_lost_for_having_a_different_nqpp(tmp_path):
    """CGA writes nqpp='—' and QGA (q=0%) writes the number.

    Grouping by nqpp each one fell into its own group and the pair was NEVER
    compared: the triage did not say 'wrong', it said nothing, which is the
    easiest way for an error to go unnoticed.
    """
    _write(str(tmp_path / 'g' / 'resultados_config.csv'), COLS_GENETIC,
           [_genetic('CGA', nqpp='—'), _genetic('QGA (q=0%)', nqpp='4')])
    out = _capture(str(tmp_path))
    assert 'CGA' in out and '1/1  OK' in out


def test_classical_vi_and_qvmc_33_are_not_required_to_match(tmp_path):
    """They are not a faithful cell: the 33% rung changes the sampling.

    Measured: Om = 0.258460 (VI) against 0.258262 (QVMC 33%). Requiring them
    to be equal flagged a false failure in every cell of every campaign.
    """
    _write(str(tmp_path / 't1' / 'resultados_lcdm.csv'), COLS_SAMPLERS,
           [_samplers('Classical VI', om='0.258460', nqpp='—'),
            _samplers('QVMC 33%', om='0.258262')])
    assert 'CHECK' not in _capture(str(tmp_path))


def test_without_parameter_columns_it_warns_instead_of_saying_ok(tmp_path):
    _write(str(tmp_path / 't1' / 'resultados_lcdm.csv'),
           ['Method', 'chi2', 'n_data', 'chi2_red', 'AIC', 'BIC'],
           [{'Method': 'QMCMC 50%', **{k: FIT[k] for k in
                                       ('chi2', 'n_data', 'chi2_red',
                                        'AIC', 'BIC')}}])
    out = _capture(str(tmp_path))
    assert 'NOTHING can be verified' in out
    assert 'OK' not in out.split('4. FAITHFUL')[-1]


# ── 5. the docstring examples ────────────────────────────────────────────

def test_the_docstring_examples_run():
    import doctest
    res = doctest.testmod(triage, verbose=False)
    assert res.failed == 0, f"{res.failed} doctests failed in triage_campaign"


# ── 6. the noise axis must not raise false alarms ────────────────────────

def test_with_noise_separation_is_expected_not_a_failure(tmp_path):
    """Without noise the two rungs coincide; with noise they MUST separate.

    Counting both axes together, a healthy campaign gave '40/62 CHECK'
    (measured on hpc_20260903_161310: 20/20 identical ideal cells,
    11/11 different noisy ones, zero exceptions). An alarm that always goes
    off is an alarm nobody reads any more.
    """
    _write(str(tmp_path / 'ideal' / 'resultados_lcdm.csv'), COLS_SAMPLERS,
           [_samplers('QMCMC 50%', noise='none'),
            _samplers('QMCMC 100%', noise='none')])
    _write(str(tmp_path / 'noisy' / 'resultados_lcdm.csv'), COLS_SAMPLERS,
           [_samplers('QMCMC 50%', noise='readout'),
            _samplers('QMCMC 100%', om='0.2731', noise='readout')])
    out = _capture(str(tmp_path))
    assert 'CHECK' not in out
    assert '1/1  OK' in out.split('Without noise')[1].split('With noise')[0]


def test_a_noisy_rung_identical_to_the_ideal_one_is_reported(tmp_path):
    """If the noise was not applied, the two rungs come out equal: that IS an error."""
    _write(str(tmp_path / 'noisy' / 'resultados_lcdm.csv'), COLS_SAMPLERS,
           [_samplers('QMCMC 50%', noise='full'),
            _samplers('QMCMC 100%', noise='full')])
    out = _capture(str(tmp_path))
    assert 'noise did not reach' in out


def test_none_counts_counts_as_ideal_axis(tmp_path):
    """The ideal-by-counts control column is still faithful."""
    _write(str(tmp_path / 'c' / 'resultados_lcdm.csv'), COLS_SAMPLERS,
           [_samplers('QMCMC 50%', noise='none-counts'),
            _samplers('QMCMC 100%', noise='none-counts')])
    out = _capture(str(tmp_path))
    assert 'Without noise' in out and 'CHECK' not in out


def test_the_noise_level_comes_from_the_folder_if_there_is_no_column(tmp_path):
    """Old campaign: there is no `noise` column, but the folder says it.

    Without this fallback to the path, hpc_20260903_161310 (which predates
    [B-PROV]) sent its 11 noisy cells to the ideal bucket and reported
    '40/62 CHECK' on a healthy campaign.
    """
    _write(str(tmp_path / 'samplers_lcdm_nqpp5_noise-full' /
               'resultados_lcdm.csv'), COLS_SAMPLERS_OLD,
           [_samplers('QMCMC 50%'),
            _samplers('QMCMC 100%', om='0.2731')])
    out = _capture(str(tmp_path))
    assert 'CHECK' not in out
    assert 'With noise' in out


def test_noise_immune_pairs_are_not_reported(tmp_path):
    """QVMC 67 vs 100 and CGA vs QGA(0%) must NOT separate with noise.

    The normalization runs its circuit but discards the result and uses the
    exact sum (see quantum_amplitude_normalization), and at q=0% there is no
    quantum component. Requiring them to separate flagged 70 of 92 healthy
    cells of hpc_20260903_161310.
    """
    _write(str(tmp_path / 'samplers_lcdm_nqpp5_noise-full' /
               'resultados_lcdm.csv'), COLS_SAMPLERS_OLD,
           [_samplers('QVMC 67%'), _samplers('QVMC 100%')])
    _write(str(tmp_path / 'genetic_lcdm_nb5_noise-full' /
               'resultados_g.csv'), COLS_GENETIC,
           [_genetic('CGA'), _genetic('QGA (q=0%)')])
    out = _capture(str(tmp_path))
    assert 'CHECK' not in out
    assert 'IMMUNE' in out


def test_an_immune_pair_that_does_separate_is_reported(tmp_path):
    """If the normalization starts depending on noise, we need to know."""
    _write(str(tmp_path / 'samplers_lcdm_nqpp5_noise-full' /
               'resultados_lcdm.csv'), COLS_SAMPLERS_OLD,
           [_samplers('QVMC 67%'), _samplers('QVMC 100%', om='0.9')])
    assert 'CHECK' in _capture(str(tmp_path))
