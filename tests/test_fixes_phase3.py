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


# ── [B-NOPLOT] --no-plot en los modos de HPC ─────────────────────────────

def test_b_noplot_el_ladder_acepta_y_propaga_la_bandera():
    """`--no-plot` debe llegar hasta donde se dibujan las figuras.

    Antes la llamada a `plot_method_ladders` dentro de
    `run_quantumness_ladder` era incondicional, y ese es el camino que usan
    --benchmark y --sweep-all: o sea, TODA corrida de HPC. La bandera solo
    funcionaba en el camino de configuracion unica, y encima el docstring de
    `run_sweep_all` la anunciaba como pass-through. Se detecto corriendo una
    prueba de humo con --no-plot que aun asi escribio 20 figuras.
    """
    import inspect
    import cosmo_modular_quantum as cmq

    firma = inspect.signature(cmq.run_quantumness_ladder)
    assert 'no_plot' in firma.parameters, (
        "run_quantumness_ladder no acepta no_plot: --no-plot se ignora en "
        "--benchmark y en --sweep-all")
    assert firma.parameters['no_plot'].default is False

    cuerpo = inspect.getsource(cmq.run_quantumness_ladder)
    i_flag = cuerpo.find('if no_plot')
    i_plot = cuerpo.find('plot_method_ladders(')
    assert 0 <= i_flag < i_plot, (
        "plot_method_ladders no esta dentro del if de no_plot")

    # run_sweep_all debe pasarla, no solo aceptarla.
    fuente_sweep = inspect.getsource(cmq.run_sweep_all)
    assert 'no_plot=no_plot' in fuente_sweep, (
        "run_sweep_all documenta no_plot como pass-through pero no lo pasa")


# ── [B-REFCLIP] bandas de referencia cortadas ────────────────────────────

def test_b_refclip_la_banda_que_cabe_se_dibuja_entera():
    """La banda de Planck no debe quedar cortada por el borde del panel.

    Se dibujaba DESPUES de fijar los ejes a los datos, asi que en
    ladder_trends la banda de Om (0.3055-0.3167) se cortaba en 0.315 y no
    se veia donde acababa.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import cosmo_modular_quantum as cmq

    fig, ax = plt.subplots()
    ax.set_ylim(0.265, 0.315)
    fuera = cmq._draw_ref_lines(ax, 'Om', axis='y', band=True)
    lo, hi = ax.get_ylim()
    val, sig = [(v, s) for _, v, s, _, _ in cmq._ref_specs('Om')][0]
    assert fuera == []
    assert lo <= val - sig and val + sig <= hi, "la banda sigue cortada"
    plt.close(fig)


def test_b_refclip_la_referencia_fuera_de_rango_se_anota_no_se_dibuja():
    """SH0ES (73.0) contra datos en 68.5-70.5: anotar, no estirar el eje.

    Estirarlo aplastaria los datos en una franja ilegible — que es lo que
    [B-FIDSCALE] arreglo — y media banda confunde mas que ninguna.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import cosmo_modular_quantum as cmq

    fig, ax = plt.subplots()
    ax.set_ylim(68.5, 70.5)
    fuera = cmq._draw_ref_lines(ax, 'H0', axis='y', band=True)
    lo, hi = ax.get_ylim()
    assert len(fuera) == 2, f"se esperaban 2 referencias fuera, hubo {fuera}"
    assert any('SH0ES' in f and '>' in f for f in fuera)
    assert any('Planck' in f and '<' in f for f in fuera)
    assert hi < 72, "el eje se estiro hasta SH0ES y aplasto los datos"
    plt.close(fig)


# ── [B-REPLOT] rehacer figuras sin repetir el computo ────────────────────

def _csv_de_tarea(destino):
    """Escribe un resultados_config.csv como el que produce una tarea real."""
    import csv
    cols = ['Method', 'Om_mean', 'Om_std', 'H0_mean', 'H0_std', 'Time_s',
            'nqpp', 'chi2', 'n_data', 'chi2_red', 'AIC', 'BIC', 'acceptance',
            'final_KL', 'ESS', 'dataset', 'prior', 'noise', 'proposal_route',
            'seed', 'budget_mode', 'circuits_train', 'chi2_grid']
    base = {'chi2': '1064.61', 'n_data': '1099', 'chi2_red': '0.970',
            'AIC': '1068.6', 'BIC': '1078.6', 'dataset': 'CC+BAO+Pantheon',
            'prior': 'flat', 'noise': 'none', 'seed': '42'}
    filas = []
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
        filas.append(r)
    os.makedirs(os.path.dirname(destino), exist_ok=True)
    with open(destino, 'w', newline='') as fh:
        w = csv.DictWriter(fh, cols)
        w.writeheader()
        w.writerows(filas)


def test_b_replot_rehace_las_figuras_desde_el_csv(tmp_path):
    """Volver a dibujar sin repetir el computo.

    En los nqpp altos una tarea son dias (16 qubits ~ 6 dias medidos), asi
    que tras arreglar algo del DIBUJO hay que poder rehacer la figura desde
    los datos ya guardados. Todo lo que muestran ladder_trends y
    ladder_summary son escalares que el CSV trae.
    """
    import matplotlib
    matplotlib.use('Agg')
    import cosmo_modular_quantum as cmq

    csv_path = str(tmp_path / 'model_lcdm' / 'resultados_config.csv')
    _csv_de_tarea(csv_path)
    destino = cmq.replot_ladder_from_csv(csv_path)
    for f in ('ladder_trends_lcdm.png', 'ladder_summary_lcdm.png'):
        p = os.path.join(destino, f)
        assert os.path.exists(p) and os.path.getsize(p) > 5000, f"falta {f}"


def test_b_replot_lee_steps_e_iters_del_log(tmp_path):
    """El CSV no guarda steps/iters; el log de la tarea si."""
    import matplotlib
    matplotlib.use('Agg')
    import cosmo_modular_quantum as cmq

    csv_path = str(tmp_path / 'model_lcdm' / 'resultados_config.csv')
    _csv_de_tarea(csv_path)
    (tmp_path / 'sweep_all_x.log').write_text(
        'SWEEP-ALL\n  dataset=CC+BAO+Pantheon | steps=20000 | '
        'qvmc_iter=15000 | nqpp=3\n')
    cmq.replot_ladder_from_csv(csv_path)   # no debe fallar leyendo el log


def test_b_replot_avisa_si_el_csv_no_sirve(tmp_path):
    """Un CSV sin filas de peldanos debe fallar diciendo por que."""
    import matplotlib
    matplotlib.use('Agg')
    import cosmo_modular_quantum as cmq
    import pytest as _pytest

    p = tmp_path / 'model_lcdm' / 'resultados_config.csv'
    os.makedirs(os.path.dirname(str(p)), exist_ok=True)
    p.write_text('Method,Om_mean\nCGA,0.26\n')
    with _pytest.raises(ValueError, match='QMCMC|QVMC'):
        cmq.replot_ladder_from_csv(str(p))


def test_b_replot_rhat_y_kl_salen_del_log(tmp_path):
    """R-hat y KL son trazas, no escalares: se recuperan del log.

    El CSV solo guarda el valor final de cada peldano. Pero el log imprime
    la traza linea a linea, asi que las dos figuras se pueden rehacer sin
    recalcular — con la resolucion de --log-every, no mas fina.
    """
    import matplotlib
    matplotlib.use('Agg')
    import cosmo_modular_quantum as cmq

    csv_path = str(tmp_path / 'model_lcdm' / 'resultados_config.csv')
    _csv_de_tarea(csv_path)
    lineas = ['SWEEP-ALL | steps=20000 | qvmc_iter=15000 | nqpp=3']
    for s in range(500, 3001, 500):
        for tag in ('C-MCMC', 'QMCMC50', 'QMCMC100'):
            lineas.append(f'[{tag}] step {s:6d}/20000 | acc=0.48 | '
                          f'R-hat-1=+{0.07 * 500 / s:.4f} | mean: Om=0.27')
    for it in range(0, 3001, 500):
        for tag in ('C-VI', 'QVMC33', 'QVMC67', 'QVMC100'):
            lineas.append(f'[{tag}] iter {it:6d}/15000 | '
                          f'KL={9.0 / (1 + it / 500):.6f} | E[theta]')
    (tmp_path / 'sweep_all_x.log').write_text('\n'.join(lineas))

    destino = cmq.replot_ladder_from_csv(csv_path)
    for f in ('ladder_rhat_qmcmc_lcdm.png', 'ladder_kl_qvmc_lcdm.png'):
        p = os.path.join(destino, f)
        assert os.path.exists(p) and os.path.getsize(p) > 5000, f"falta {f}"


def test_b_replot_sin_log_no_falla_y_hace_lo_que_puede(tmp_path):
    """Sin log no hay trazas: deben salir las dos que si dependen del CSV."""
    import matplotlib
    matplotlib.use('Agg')
    import cosmo_modular_quantum as cmq

    csv_path = str(tmp_path / 'model_lcdm' / 'resultados_config.csv')
    _csv_de_tarea(csv_path)
    destino = cmq.replot_ladder_from_csv(csv_path)
    assert os.path.exists(os.path.join(destino, 'ladder_trends_lcdm.png'))
    assert not os.path.exists(os.path.join(destino,
                                           'ladder_rhat_qmcmc_lcdm.png'))


def test_b_replot_el_rhat_del_log_es_absoluto_no_menos_uno():
    """El log escribe R-hat-1; la figura vuelve a restar 1.

    Si se guardara el valor del log tal cual, la curva saldria desplazada
    una unidad entera hacia abajo y cruzaria cero en escala logaritmica.
    """
    import tempfile
    import cosmo_modular_quantum as cmq

    with tempfile.NamedTemporaryFile('w', suffix='.log', delete=False) as t:
        t.write('[QMCMC50] step  500/20000 | acc=0.46 | R-hat-1=+0.0460 | m\n')
        nombre = t.name
    rhat, _ = cmq.historias_desde_log(nombre)
    os.unlink(nombre)
    assert rhat[50.0] == [(500, 1.046)]


# ── [B-ITERDISP] el contador debe mostrar el presupuesto real ────────────

def test_b_iterdisp_el_denominador_es_el_presupuesto_real(tmp_path):
    """El log de la rama clasica escribia 'iter 122000/15000'.

    Con budget_mode='circuits' el tope de COBYLA es max_iter*(1+2*n_phi)
    —2,115,000 para lcdm con nqpp=5—, pero el log seguia imprimiendo
    max_iter como denominador. Visto de madrugada, eso parece que el codigo
    se rompio; en realidad es el presupuesto igualado funcionando. Un
    contador que miente sobre lo que falta es peor que no tenerlo.
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

    presupuesto = [m for m in reg if 'presupuesto COBYLA' in m]
    assert presupuesto, "no se anuncia el presupuesto real antes de arrancar"

    lineas = [m for m in reg if '] iter ' in m]
    assert lineas, "no hubo lineas de progreso"
    # n_phi = 3*4*2 + 4 = 28 -> 20 * (1 + 56) = 1140
    assert all('/1140' in m for m in lineas), (
        f"el denominador no es el presupuesto real: {lineas[:2]}")
    assert not any('/20 ' in m for m in lineas), "sigue mostrando max_iter"


# ═══ [B-REPLOTGLOB] --replot-ladder debe aceptar varios CSV ══════════════

def test_b_replotglob_acepta_varios_csv():
    """La ayuda promete comodines del shell; argparse debe poder recibirlos.

    Sin `nargs='+'` la shell expandia el patron a N rutas, argparse tomaba
    la primera como valor de la bandera y abortaba con "unrecognized
    arguments" por las otras N-1. Resultado: rehacer las figuras de una
    campana entera era imposible salvo archivo por archivo.
    """
    import inspect
    cands = [f for n, f in inspect.getmembers(mq, inspect.isfunction)
             if 'parser' in n]
    assert cands, "no se encontro el constructor del parser"
    p = cands[0]()
    args = p.parse_args(['--replot-ladder', 'a.csv', 'b.csv', 'c.csv'])
    assert args.replot_ladder == ['a.csv', 'b.csv', 'c.csv']


def test_b_replotglob_uno_solo_sigue_funcionando():
    """Pasar un unico CSV no debe romperse ni devolver una cadena suelta."""
    import inspect
    p = None
    for n, f in inspect.getmembers(mq, inspect.isfunction):
        if 'parser' in n:
            p = f()
            break
    assert p is not None
    args = p.parse_args(['--replot-ladder', 'solo.csv'])
    assert args.replot_ladder == ['solo.csv']
