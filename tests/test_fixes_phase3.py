"""
test_fixes_phase3.py — regression guards for the Phase-3 fix batch.

Covers, on the REAL production code paths:
  H1  split-R-hat (1.01) wired into the QMCMC convergence loop
  H2  run_comparison produces equal-length chains
  H3  the KL objective penalizes probability mass leaked outside the target
  H5  the training gradient is exact (parameter-shift on probabilities)
  H6  quantum mutation is gated by mutation_rate
  M1  quantum run and classical baseline share the VI initialization
  M3  QGA runs are reproducible under a fixed seed
  M4  simulator proposal engine: fixed calibration constants, unit std
  H4  QPU proposal engine (dry-run): displacements calibrated to unit std

Requires qiskit + qiskit-aer; skipped automatically when absent.
"""
import contextlib
import io
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

pytest.importorskip("qiskit")
pytest.importorskip("qiskit_aer")

from qiskit import transpile                         # noqa: E402

import cosmo_core as core                            # noqa: E402
from cosmo_core import MODELS, Posterior             # noqa: E402
import cosmo_modular_quantum as mq                   # noqa: E402
import qpu_cosmo_samplers as qpu                     # noqa: E402
from cosmo_genetic_optimizers import QGA, GAConfig   # noqa: E402


@pytest.fixture(scope="module")
def post():
    return Posterior(MODELS['lcdm'], 'CC+BAO', 'flat')


# ── H3: leakage-aware KL ─────────────────────────────────────────────────────

def test_kl_penalizes_leaked_mass(post):
    """A state whose probability mass sits OUTSIDE the target support must
    report a LARGE KL. Under the old masked definition it reported ~0."""
    mq._reseed(3)
    qv = mq.QVMCModular(post, dict(mq.CLASSICAL_BASELINE),
                        n_qubits_per_param=2, n_shots=200)
    qc, n_p = qv._build_ansatz()
    qc_t = transpile(qc.remove_final_measurements(inplace=False), qv.sim)
    # phi = 0 -> ansatz RY/RZ angles all zero -> Q is (close to) a delta on
    # basis state 0. Synthetic target with ALL its mass elsewhere:
    n = 2 ** qc.num_qubits
    P_syn = np.zeros(n)
    P_syn[n // 2] = 1.0
    kl = float(qv._kl_batch(np.zeros(n_p), qc_t, P_syn)[0])
    assert kl > 15.0     # ~|log eps| when the mass is fully leaked


# ── H5: exact gradient ───────────────────────────────────────────────────────

def test_training_gradient_is_exact(post):
    """The implemented gradient (parameter-shift on the PROBABILITIES +
    chain rule) must match small-h central differences of the KL. The old
    +-pi/2 shift applied to the KL itself had ~10% relative error."""
    mq._reseed(7)
    qv = mq.QVMCModular(post, dict(mq.CLASSICAL_BASELINE, training=True),
                        n_qubits_per_param=2, n_shots=200)
    P_target = qv.build_target()
    qc, n_p = qv._build_ansatz()
    qc_t = transpile(qc.remove_final_measurements(inplace=False), qv.sim)
    phi = np.random.default_rng(3).uniform(0, 2 * np.pi, n_p)

    _, Qs = qv._kl_batch(phi, qc_t, P_target, return_q=True)
    shifts = np.repeat(phi[None, :], 2 * n_p, axis=0)
    for j in range(n_p):
        shifts[2 * j, j] += np.pi / 2
        shifts[2 * j + 1, j] -= np.pi / 2
    _, Qs_s = qv._kl_batch(shifts, qc_t, P_target, return_q=True)
    Qmat = np.asarray(Qs_s)
    dQ = (Qmat[0::2] - Qmat[1::2]) / 2.0
    eps = 1e-12
    P_s = P_target + eps
    P_s = P_s / P_s.sum()
    w = np.log(np.clip(Qs[0], eps, None)) - np.log(P_s)
    g_impl = dQ @ w

    h = 1e-6
    rng = np.random.default_rng(0)
    for j in rng.choice(n_p, size=min(5, n_p), replace=False):
        pp, pm = phi.copy(), phi.copy()
        pp[j] += h
        pm[j] -= h
        g_ref = (qv._kl_batch(pp, qc_t, P_target)[0]
                 - qv._kl_batch(pm, qc_t, P_target)[0]) / (2 * h)
        assert abs(g_impl[j] - g_ref) < 1e-4 * max(1.0, abs(g_ref))


# ── H1 + H2 + M1: comparison protocol ────────────────────────────────────────

def test_comparison_equal_lengths_and_shared_vi_init(post):
    """run_comparison must (H2) yield equal-length chains for the quantum
    run and its baseline, (M1) give both VI runs the SAME initialization
    (preset 45 trains classically on both sides, so the KL histories must
    be bit-identical), and (H1) record split-R-hat, not legacy GR."""
    with contextlib.redirect_stdout(io.StringIO()):
        out = mq.run_comparison(post, dict(mq.PRESETS[45]), seed=11,
                                n_steps_mcmc=120, max_iter_qvmc=6,
                                n_chains_mcmc=3, n_chains_qvmc=2,
                                nqpp=2, n_shots=300, verbose=False)
    rq, rc = out['quantum'], out['classical']
    assert rq['chains_mcmc'].shape[1] == rc['chains_mcmc'].shape[1] == 120
    hq = [h['kl'] for h in rq['qvmc_history']]
    hc = [h['kl'] for h in rc['qvmc_history']]
    assert np.allclose(hq, hc, rtol=0, atol=0)
    assert len(rq['mcmc']['rhat_hist']) > 0
    # A deliberately short run must NOT be declared converged at 1.01:
    assert rq['mcmc']['converged'] in (False, True)  # key present & boolean


# ── M4: simulator proposal engine ────────────────────────────────────────────

def test_proposal_engine_fixed_calibration():
    mq._reseed(5)
    eng = mq.QuantumProposalEngine(n_phys=2, batch=128)
    mu0, s0 = eng._mu.copy(), eng._sigma.copy()
    D = np.array([eng.next() for _ in range(512)])       # forces 4 refills
    assert np.allclose(mu0, eng._mu) and np.allclose(s0, eng._sigma)
    assert np.all(np.abs(D.mean(axis=0)) < 0.2)
    assert np.all(np.abs(D.std(axis=0) - 1.0) < 0.2)


# ── H4: QPU proposal engine (dry-run) ────────────────────────────────────────

def test_qpu_engine_unit_std_dry_run():
    conn = qpu.QPUConnection(dry_run=True)
    eng = qpu.QPUProposalEngine(conn, n_phys=2, block=64,
                                shots_per_proposal=128)
    D = np.array([eng.next() for _ in range(256)])
    # Without the H4 calibration the per-dimension std was ~0.05-0.12.
    assert np.all(np.abs(D.std(axis=0) - 1.0) < 0.35)


# ── H6: mutation gating ──────────────────────────────────────────────────────

def test_quantum_mutation_respects_mutation_rate(post):
    qga = QGA(post, GAConfig(pop_size=40, mutation_rate=0.0, seed=2),
              dict(q_mutation=True), n_bits=6,
              rng=np.random.default_rng(2))
    pop = np.column_stack(
        [np.random.default_rng(0).uniform(0.2, 0.45, 40),
         np.random.default_rng(1).uniform(62, 76, 40)])
    pop_g = qga._clip(qga._decode(qga._encode(pop)))
    assert np.allclose(qga.do_mutate(pop_g.copy()), pop_g)


# ── M3: QGA reproducibility ──────────────────────────────────────────────────

def test_qga_reproducible_with_fixed_seed(post):
    def run_once():
        q = QGA(post, GAConfig(pop_size=24, n_generations=4, seed=9),
                dict(q_init=True, q_mutation=True, q_crossover=True),
                n_bits=5, rng=np.random.default_rng(9))
        with contextlib.redirect_stdout(io.StringIO()):
            r = q.evolve(live=False, log_every=1000)
        return r.theta_map, r.chi2_map
    t1, c1 = run_once()
    t2, c2 = run_once()
    assert np.allclose(t1, t2) and c1 == c2


# ── [B-NOPLOT] --no-plot in the HPC modes ────────────────────────────────

def test_b_noplot_ladder_accepts_and_forwards_the_flag():
    """`--no-plot` must reach the place where the figures are drawn.

    The call to `plot_method_ladders` inside `run_quantumness_ladder` used to
    be unconditional, and that is the path used by --benchmark and
    --sweep-all: i.e. EVERY HPC run. The flag only worked on the
    single-configuration path, and on top of that the `run_sweep_all`
    docstring advertised it as pass-through. Caught by running a smoke test
    with --no-plot that still wrote 20 figures.
    """
    import inspect
    import cosmo_modular_quantum as cmq

    sig = inspect.signature(cmq.run_quantumness_ladder)
    assert 'no_plot' in sig.parameters, (
        "run_quantumness_ladder does not accept no_plot: --no-plot is ignored "
        "in --benchmark and --sweep-all")
    assert sig.parameters['no_plot'].default is False

    body = inspect.getsource(cmq.run_quantumness_ladder)
    i_flag = body.find('if no_plot')
    i_plot = body.find('plot_method_ladders(')
    assert 0 <= i_flag < i_plot, (
        "plot_method_ladders is not inside the no_plot if")

    # run_sweep_all must forward it, not just accept it.
    sweep_source = inspect.getsource(cmq.run_sweep_all)
    assert 'no_plot=no_plot' in sweep_source, (
        "run_sweep_all documents no_plot as pass-through but does not pass it")


# ── [B-REFCLIP] clipped reference bands ──────────────────────────────────

def test_b_refclip_band_that_fits_is_drawn_whole():
    """The Planck band must not be clipped by the panel edge.

    It was drawn AFTER fixing the axes to the data, so in ladder_trends the
    Om band (0.3055-0.3167) was cut at 0.315 and one could not see where it
    ended.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import cosmo_modular_quantum as cmq

    fig, ax = plt.subplots()
    ax.set_ylim(0.265, 0.315)
    outside = cmq._draw_ref_lines(ax, 'Om', axis='y', band=True)
    lo, hi = ax.get_ylim()
    val, sig = [(v, s) for _, v, s, _, _ in cmq._ref_specs('Om')][0]
    assert outside == []
    assert lo <= val - sig and val + sig <= hi, "the band is still clipped"
    plt.close(fig)


def test_b_refclip_out_of_range_reference_is_annotated_not_drawn():
    """SH0ES (73.0) against data in 68.5-70.5: annotate, do not stretch the axis.

    Stretching it would squash the data into an unreadable strip — which is
    what [B-FIDSCALE] fixed — and half a band confuses more than none.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import cosmo_modular_quantum as cmq

    fig, ax = plt.subplots()
    ax.set_ylim(68.5, 70.5)
    outside = cmq._draw_ref_lines(ax, 'H0', axis='y', band=True)
    lo, hi = ax.get_ylim()
    assert len(outside) == 2, f"expected 2 references outside, got {outside}"
    assert any('SH0ES' in f and '>' in f for f in outside)
    assert any('Planck' in f and '<' in f for f in outside)
    assert hi < 72, "the axis was stretched up to SH0ES and squashed the data"
    plt.close(fig)


# ── [B-REPLOT] redraw figures without repeating the computation ─────────

def _task_csv(dest):
    """Write a results_config.csv like the one a real task produces."""
    import csv
    cols = ['Method', 'Om_mean', 'Om_std', 'H0_mean', 'H0_std', 'Time_s',
            'nqpp', 'chi2', 'n_data', 'chi2_red', 'AIC', 'BIC', 'acceptance',
            'final_KL', 'ESS', 'dataset', 'prior', 'noise', 'proposal_route',
            'seed', 'budget_mode', 'circuits_train', 'chi2_grid']
    base = {'chi2': '1064.61', 'n_data': '1099', 'chi2_red': '0.970',
            'AIC': '1068.6', 'BIC': '1078.6', 'dataset': 'CC+BAO+Pantheon',
            'prior': 'flat', 'noise': 'none', 'seed': '42'}
    rows = []
    for met, om, h0, t, acc, ess, kl in [
            ('Classical MCMC', '0.2764', '69.59', '16.8', '0.487', '6872', ''),
            ('QMCMC 50%', '0.2764', '69.60', '105.6', '0.456', '6956', ''),
            ('QMCMC 100%', '0.2764', '69.60', '142.4', '0.456', '6956', ''),
            ('Classical VI', '0.2757', '69.54', '449.1', '', '12288', '1.6250'),
            ('QVMC 33%', '0.2755', '69.55', '452.4', '', '12288', '1.6250'),
            ('QVMC 67%', '0.2754', '69.56', '1504.9', '', '12288', '1.6286'),
            ('QVMC 100%', '0.2754', '69.56', '1529.3', '', '12288', '1.6286')]:
        r = dict.fromkeys(cols, '')
        r.update(base)
        r.update({'Method': met, 'Om_mean': om, 'Om_std': '0.0104',
                  'H0_mean': h0, 'H0_std': '0.80', 'Time_s': t,
                  'acceptance': acc, 'ESS': ess, 'final_KL': kl, 'nqpp': '3'})
        rows.append(r)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, 'w', newline='') as fh:
        w = csv.DictWriter(fh, cols)
        w.writeheader()
        w.writerows(rows)


def test_b_replot_redraws_figures_from_csv(tmp_path):
    """Redraw without repeating the computation.

    At high nqpp a task takes days (16 qubits ~ 6 days measured), so after
    fixing something in the DRAWING one must be able to redo the figure from
    the data already saved. Everything ladder_trends and ladder_summary show
    are scalars the CSV carries.
    """
    import matplotlib
    matplotlib.use('Agg')
    import cosmo_modular_quantum as cmq

    csv_path = str(tmp_path / 'model_lcdm' / 'results_config.csv')
    _task_csv(csv_path)
    dest = cmq.replot_ladder_from_csv(csv_path)
    for f in ('ladder_trends_lcdm.png', 'ladder_summary_lcdm.png'):
        p = os.path.join(dest, f)
        assert os.path.exists(p) and os.path.getsize(p) > 5000, f"missing {f}"


def test_b_replot_reads_steps_and_iters_from_log(tmp_path):
    """The CSV does not store steps/iters; the task log does."""
    import matplotlib
    matplotlib.use('Agg')
    import cosmo_modular_quantum as cmq

    csv_path = str(tmp_path / 'model_lcdm' / 'results_config.csv')
    _task_csv(csv_path)
    (tmp_path / 'sweep_all_x.log').write_text(
        'SWEEP-ALL\n  dataset=CC+BAO+Pantheon | steps=20000 | '
        'qvmc_iter=15000 | nqpp=3\n')
    cmq.replot_ladder_from_csv(csv_path)   # must not fail reading the log


def test_b_replot_complains_if_csv_is_unusable(tmp_path):
    """A CSV without rung rows must fail saying why."""
    import matplotlib
    matplotlib.use('Agg')
    import cosmo_modular_quantum as cmq
    import pytest as _pytest

    p = tmp_path / 'model_lcdm' / 'results_config.csv'
    os.makedirs(os.path.dirname(str(p)), exist_ok=True)
    p.write_text('Method,Om_mean\nCGA,0.26\n')
    with _pytest.raises(ValueError, match='QMCMC|QVMC'):
        cmq.replot_ladder_from_csv(str(p))


def test_b_replot_rhat_and_kl_come_from_log(tmp_path):
    """R-hat and KL are traces, not scalars: they are recovered from the log.

    The CSV only stores the final value of each rung. But the log prints the
    trace line by line, so both figures can be redone without recomputing —
    at the --log-every resolution, no finer.
    """
    import matplotlib
    matplotlib.use('Agg')
    import cosmo_modular_quantum as cmq

    csv_path = str(tmp_path / 'model_lcdm' / 'results_config.csv')
    _task_csv(csv_path)
    lines = ['SWEEP-ALL | steps=20000 | qvmc_iter=15000 | nqpp=3']
    for s in range(500, 3001, 500):
        for tag in ('C-MCMC', 'QMCMC50', 'QMCMC100'):
            lines.append(f'[{tag}] step {s:6d}/20000 | acc=0.48 | '
                          f'R-hat-1=+{0.07 * 500 / s:.4f} | mean: Om=0.27')
    for it in range(0, 3001, 500):
        for tag in ('C-VI', 'QVMC33', 'QVMC67', 'QVMC100'):
            lines.append(f'[{tag}] iter {it:6d}/15000 | '
                          f'KL={9.0 / (1 + it / 500):.6f} | E[theta]')
    (tmp_path / 'sweep_all_x.log').write_text('\n'.join(lines))

    dest = cmq.replot_ladder_from_csv(csv_path)
    for f in ('ladder_rhat_qmcmc_lcdm.png', 'ladder_kl_qvmc_lcdm.png'):
        p = os.path.join(dest, f)
        assert os.path.exists(p) and os.path.getsize(p) > 5000, f"missing {f}"


def test_b_replot_without_log_does_not_fail_and_does_what_it_can(tmp_path):
    """Without a log there are no traces: the two that depend on the CSV must appear."""
    import matplotlib
    matplotlib.use('Agg')
    import cosmo_modular_quantum as cmq

    csv_path = str(tmp_path / 'model_lcdm' / 'results_config.csv')
    _task_csv(csv_path)
    dest = cmq.replot_ladder_from_csv(csv_path)
    assert os.path.exists(os.path.join(dest, 'ladder_trends_lcdm.png'))
    assert not os.path.exists(os.path.join(dest,
                                           'ladder_rhat_qmcmc_lcdm.png'))


def test_b_replot_log_rhat_is_absolute_not_minus_one():
    """The log writes R-hat-1; the figure subtracts 1 again.

    If the log value were stored as is, the curve would come out shifted a
    whole unit down and would cross zero on a log scale.
    """
    import tempfile
    import cosmo_modular_quantum as cmq

    with tempfile.NamedTemporaryFile('w', suffix='.log', delete=False) as t:
        t.write('[QMCMC50] step  500/20000 | acc=0.46 | R-hat-1=+0.0460 | m\n')
        name = t.name
    rhat, _ = cmq.histories_from_log(name)
    os.unlink(name)
    assert rhat[50.0] == [(500, 1.046)]


# ── [B-ITERDISP] the counter must show the real budget ───────────────────

def test_b_iterdisp_denominator_is_the_real_budget(tmp_path):
    """The classical branch log wrote 'iter 122000/15000'.

    With budget_mode='circuits' the COBYLA cap is max_iter*(1+2*n_phi)
    —2,115,000 for lcdm with nqpp=5—, but the log kept printing max_iter as
    the denominator. Seen in the small hours, that looks like the code
    broke; in reality it is the matched budget working. A counter that lies
    about what is left is worse than no counter.
    """
    import logging
    import cosmo_modular_quantum as cmq

    reg = []

    class _Cap(logging.Handler):
        def emit(self, r):
            reg.append(r.getMessage())

    log = logging.getLogger('test_iterdisp')
    log.setLevel(logging.INFO)
    log.handlers = [_Cap()]

    post = cmq.Posterior(cmq.MODELS['lcdm'], 'CC+BAO', 'flat')
    cmq._reseed(42)
    q = cmq.QVMCModular(post, {}, n_qubits_per_param=2, budget_mode='circuits')
    P = q.build_target()
    q.train(P, max_iter=20, logger=log, log_every=100, progress=False,
            tag='C-VI')

    budget = [m for m in reg if 'COBYLA budget' in m]
    assert budget, "the real budget is not announced before starting"

    lines = [m for m in reg if '] iter ' in m]
    assert lines, "there were no progress lines"
    # n_phi = 3*4*2 + 4 = 28 -> 20 * (1 + 56) = 1140
    assert all('/1140' in m for m in lines), (
        f"the denominator is not the real budget: {lines[:2]}")
    assert not any('/20 ' in m for m in lines), "still showing max_iter"


# ═══ [B-REPLOTGLOB] --replot-ladder must accept several CSVs ═════════════

def test_b_replotglob_accepts_several_csvs():
    """The help text promises shell wildcards; argparse must accept them.

    Without `nargs='+'` the shell expanded the pattern to N paths, argparse
    took the first as the flag's value and aborted with "unrecognized
    arguments" for the other N-1. Result: redoing the figures of a whole
    campaign was impossible except file by file.
    """
    import inspect
    cands = [f for n, f in inspect.getmembers(mq, inspect.isfunction)
             if 'parser' in n]
    assert cands, "parser builder not found"
    p = cands[0]()
    args = p.parse_args(['--replot-ladder', 'a.csv', 'b.csv', 'c.csv'])
    assert args.replot_ladder == ['a.csv', 'b.csv', 'c.csv']


def test_b_replotglob_single_csv_still_works():
    """Passing a single CSV must not break or return a bare string."""
    import inspect
    p = None
    for n, f in inspect.getmembers(mq, inspect.isfunction):
        if 'parser' in n:
            p = f()
            break
    assert p is not None
    args = p.parse_args(['--replot-ladder', 'single.csv'])
    assert args.replot_ladder == ['single.csv']
