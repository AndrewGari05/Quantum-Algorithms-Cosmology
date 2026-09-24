"""test_ruido_genetico.py — el eje de ruido visto desde el genetico.

Tres bugs reales, encontrados leyendo las campanas de Nicte-Ha y Saptiva, y
la red que impide que vuelvan:

  [B-PROV-GEN]  toda fila genetica escribia noise='none', corriera con el
                peldano que corriera, porque el escritor de CSV leia el
                global de OTRO modulo. Medido: 116 filas mal etiquetadas en
                `hpc_20260907_130425` (40 fakebrisbane, 40 readout, 36 full).
  [B-RZZ]       el cruce cuantico moria al transpilar contra el peldano
                `full`, que declara `rzz`/`rxx`/`ryy` como base. Medido: la
                tarea `genetic_lcdm_nb4_noise-full` murio en __init__ y dejo
                el CSV con la fila del CGA y nada mas.
  carpeta>columna  las figuras del eje deben tomar el peldano del NOMBRE DE
                LA CARPETA, porque `none-counts` no existe como valor de la
                columna y porque las campanas viejas la tienen mal.
"""
import csv
import importlib.util
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import cosmo_noise as cnoise                      # noqa: E402
import cosmo_modular_quantum as cmq               # noqa: E402

_spec = importlib.util.spec_from_file_location(
    'graficas_ruido', os.path.join(ROOT, 'graficas_ruido.py'))
gr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gr)


# ═══ [B-PROV-GEN] la fila dice su propio peldano ════════════════════════

def test_b_prov_gen_la_fila_genetica_lleva_su_peldano():
    """`csv_provenance` debe preferir el peldano que trae la fila.

    Sin esto el genetico heredaba el NOISE de cosmo_modular_quantum, que su
    CLI nunca fija: toda corrida ruidosa se archivaba como ideal.
    """
    fila = gr and {'noise': 'readout', 'seed': '7'}
    prov = cmq.csv_provenance(fila)
    assert prov['noise'] == 'readout'
    assert prov['seed'] == '7'


def test_b_prov_gen_los_samplers_siguen_leyendo_el_global():
    """La ruta de los samplers NO cambia: sin clave, manda el global."""
    previo = cmq.NOISE
    try:
        cmq.NOISE = cnoise.NoiseSpec.from_level('full')
        assert cmq.csv_provenance({})['noise'] == 'full'
    finally:
        cmq.NOISE = previo


def test_b_prov_gen_el_lado_genetico_arma_la_clave():
    """`_ga_side` debe copiar el peldano del modulo genetico y la semilla."""
    import numpy as np
    import cosmo_genetic_optimizers as G

    class _R:
        final_fit = np.array([-1.0, -2.0])
        final_pop = np.array([[0.3, 70.0], [0.28, 69.0]])
        final_weights = np.array([0.6, 0.4])
        elapsed = 1.0
        chi2_grid = 1064.61
        stats = {'chi2': 1064.61, 'n_data': 1099, 'chi2_red': 0.97,
                 'AIC': 1068.6, 'BIC': 1078.6}

    previo = G.NOISE
    try:
        G.NOISE = cnoise.NoiseSpec.from_level('readout')
        side = G._ga_side(_R(), post=None, seed=123)
        assert side['noise'] == 'readout'
        assert side['seed'] == '123'
        # y esa fila, pasada al escritor compartido, conserva el peldano
        assert cmq.csv_provenance(side)['noise'] == 'readout'
    finally:
        G.NOISE = previo


# ═══ [B-RZZ] el cruce cuantico no debe morir con `full` ═════════════════

def test_b_rzz_el_peldano_full_declara_compuertas_parametricas():
    """La causa raiz, fijada como hecho: `full` mete rzz/rxx/ryy en la base.

    Si esto deja de ser cierto, la red de [B-RZZ] sobra y hay que revisarla;
    mientras sea cierto, el transpilador puede tropezar con ellas.
    """
    nm = cnoise.NoiseSpec.from_level('full').simulator_kwargs(
        counts_route=True).get('noise_model')
    assert nm is not None
    assert {'rzz', 'rxx', 'ryy'} <= set(nm.basis_gates)


def test_b_rzz_operator_sobre_rzz_sin_ligar_es_el_error_exacto():
    """El error del log de Nicte-Ha, reproducido en una linea."""
    from qiskit.circuit import Parameter
    from qiskit.circuit.library import RZZGate
    from qiskit.quantum_info import Operator
    with pytest.raises(TypeError) as exc:
        Operator(RZZGate(Parameter('ϴ')))
    assert 'unbound parameter' in str(exc.value).lower()


def test_b_rzz_reintenta_sin_las_parametricas_y_no_se_cae():
    """Si la via principal revienta con ESE error, hay segunda via."""
    import cosmo_genetic_optimizers as G
    qga = G.QGA.__new__(G.QGA)
    qga.noise_label = 'full'
    kw = cnoise.NoiseSpec.from_level('full').simulator_kwargs(
        counts_route=True)
    from qiskit_aer import AerSimulator
    qga.sim = AerSimulator(**kw)

    llamadas = {'n': 0}
    real = G.transpile

    def _falla_la_primera(qc, backend, **kwargs):
        llamadas['n'] += 1
        if 'basis_gates' not in kwargs:
            raise TypeError("ParameterExpression with unbound parameters "
                            "(dict_keys([Parameter(ϴ)])) cannot be cast to "
                            "a float")
        return real(qc, backend, **kwargs)

    G.transpile = _falla_la_primera
    try:
        from qiskit import QuantumCircuit
        from qiskit.circuit import ParameterVector
        pv = ParameterVector('pa', 2)
        qc = QuantumCircuit(2)
        qc.ry(pv[0], 0)
        qc.ry(pv[1], 1)
        qc.cx(0, 1)
        qc.measure_all()
        with pytest.warns(RuntimeWarning, match='B-RZZ'):
            salida = qga._transpile_cx(qc)
        assert salida.num_qubits == 2
        assert llamadas['n'] == 2         # fallo una vez, reintento una vez
    finally:
        G.transpile = real


def test_b_rzz_un_error_distinto_se_propaga():
    """La red atrapa UN error concreto; cualquier otro debe seguir matando.

    Tragarse excepciones genericas aqui convertiria un circuito mal
    construido en una corrida silenciosamente equivocada.
    """
    import cosmo_genetic_optimizers as G
    qga = G.QGA.__new__(G.QGA)
    qga.noise_label = 'full'
    from qiskit_aer import AerSimulator
    qga.sim = AerSimulator()
    real = G.transpile
    G.transpile = lambda *a, **k: (_ for _ in ()).throw(
        TypeError('otra cosa completamente distinta'))
    try:
        with pytest.raises(TypeError, match='otra cosa'):
            qga._transpile_cx(None)
    finally:
        G.transpile = real


# ═══ el peldano se lee de la carpeta, no de la columna ══════════════════

COLS = ['Method', 'Om_mean', 'Om_std', 'H0_mean', 'H0_std', 'Time_s', 'nqpp',
        'chi2', 'n_data', 'chi2_red', 'AIC', 'BIC', 'acceptance', 'final_KL',
        'ESS', 'dataset', 'prior', 'noise', 'proposal_route', 'seed',
        'budget_mode', 'circuits_train', 'chi2_grid']


def _tarea(raiz, nombre, modelo, filas):
    d = os.path.join(raiz, nombre, f'model_{modelo}')
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, 'resultados_config.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, COLS)
        w.writeheader()
        for f in filas:
            r = dict.fromkeys(COLS, '')
            r.update(f)
            w.writerow(r)


def _base(method, om='0.2764', noise='none', **extra):
    r = {'Method': method, 'Om_mean': om, 'Om_std': '0.0100',
         'H0_mean': '69.60', 'H0_std': '0.80', 'Time_s': '10.0',
         'chi2': '1064.6', 'n_data': '1099', 'chi2_red': '0.97',
         'AIC': '1068.6', 'BIC': '1078.6', 'noise': noise,
         'dataset': 'CC+BAO+Pantheon', 'prior': 'flat', 'seed': '42'}
    r.update(extra)
    return r


def test_la_carpeta_manda_sobre_la_columna(tmp_path):
    """Una campana genetica vieja tiene la columna mal; la carpeta no."""
    _tarea(str(tmp_path), 'genetic_lcdm_nb4_noise-readout', 'lcdm',
           [_base('CGA', noise='none')])          # <- la columna miente
    filas = gr.leer_campana(str(tmp_path))
    assert len(filas) == 1
    assert filas[0]['_ruido'] == 'readout'
    assert filas[0]['_ruido_csv'] == 'none'
    assert filas[0]['_rejilla'] == 4
    assert filas[0]['_familia'] == 'genetic'


def test_none_counts_no_se_confunde_con_none(tmp_path):
    """El control de ruta es una celda propia aunque el CSV diga 'none'.

    Fundirlo con `none` borraria el unico control que separa el cambio de
    ruta de lectura del efecto del ruido.
    """
    _tarea(str(tmp_path), 'samplers_lcdm_nqpp3_noise-none', 'lcdm',
           [_base('QMCMC 100%')])
    _tarea(str(tmp_path), 'samplers_lcdm_nqpp3_noise-none-counts', 'lcdm',
           [_base('QMCMC 100%', proposal_route='counts')])
    filas = gr.leer_campana(str(tmp_path))
    assert {f['_ruido'] for f in filas} == {'none', 'none-counts'}
    # y `none-counts` NO se denuncia como incoherente: su columna dice 'none'
    # porque el nivel de ruido ES 'none'.
    import io
    from contextlib import redirect_stdout
    buf = io.StringIO()
    with redirect_stdout(buf):
        gr.avisar_etiquetas_incoherentes(filas)
    assert buf.getvalue() == ''


def test_se_denuncia_la_campana_con_etiquetas_mal(tmp_path):
    _tarea(str(tmp_path), 'genetic_lcdm_nb4_noise-full', 'lcdm',
           [_base('CGA', noise='none')])
    import io
    from contextlib import redirect_stdout
    buf = io.StringIO()
    with redirect_stdout(buf):
        gr.avisar_etiquetas_incoherentes(gr.leer_campana(str(tmp_path)))
    salida = buf.getvalue()
    assert 'B-PROV-GEN' in salida and 'full' in salida


def test_el_desplazamiento_se_mide_en_sigmas_de_la_corrida_ideal(tmp_path):
    """Dividir entre la sigma RUIDOSA escondería el sesgo tras su dispersion."""
    ideal = _base('QMCMC 100%', om='0.2700', Om_std='0.0100')
    ruidosa = _base('QMCMC 100%', om='0.2750', Om_std='1.0000')
    d = gr.desplazamiento_en_sigmas(ruidosa, ideal)
    assert d == pytest.approx(0.5, abs=1e-9)


def test_el_desplazamiento_toma_el_peor_parametro(tmp_path):
    """Si el ruido sesga aunque sea un parametro, el resultado ya esta sesgado."""
    ideal = _base('QMCMC 100%', om='0.2700', Om_std='0.0100')
    ideal['H0_mean'], ideal['H0_std'] = '69.00', '1.00'
    ruidosa = _base('QMCMC 100%', om='0.2700')
    ruidosa['H0_mean'] = '72.00'
    assert gr.desplazamiento_en_sigmas(ruidosa, ideal) == pytest.approx(3.0)


def test_una_celda_sin_ideal_no_inventa_desplazamiento(tmp_path):
    """Sin corrida ideal con la que comparar, la metrica es None, no 0."""
    _tarea(str(tmp_path), 'samplers_lcdm_nqpp3_noise-readout', 'lcdm',
           [_base('QMCMC 100%', noise='readout')])
    filas = gr.leer_campana(str(tmp_path))
    idx = gr.indice_de_celdas_ideales(filas)
    assert gr.valor(filas[0], 'desplazamiento', idx) is None


def test_las_figuras_salen_y_no_mezclan_familias(tmp_path):
    """nqpp y n_bits no comparten eje: cada familia su figura."""
    for nb in (4, 5):
        for noi in ('none', 'readout'):
            _tarea(str(tmp_path), f'genetic_lcdm_nb{nb}_noise-{noi}', 'lcdm',
                   [_base('CGA', om=f'0.27{nb}0', chi2_grid='1064.6'),
                    _base('QGA (q=100%)', om=f'0.27{nb}1',
                          chi2_grid='1064.9')])
    for nq in (3, 4):
        for noi in ('none', 'readout'):
            _tarea(str(tmp_path), f'samplers_lcdm_nqpp{nq}_noise-{noi}',
                   'lcdm', [_base('QMCMC 100%', om=f'0.27{nq}0',
                                  acceptance='0.48')])
    salida = tmp_path / 'figs'
    hechas = gr.construir([str(tmp_path)], str(salida))
    assert hechas
    nombres = {os.path.basename(h) for h in hechas}
    assert any('genetic' in n and 'rejilla' in n for n in nombres)
    assert any('samplers' in n and 'rejilla' in n for n in nombres)
    for h in hechas:
        assert os.path.getsize(h) > 5000


# ═══ [B-EPS15] la celda fiel del QVMC debe ser fiel de verdad ═══════════

def test_b_eps15_la_normalizacion_cuantica_coincide_bit_a_bit(tmp_path):
    """`QVMC 67%` y `QVMC 100%` solo difieren en este componente.

    La rama clasica hace `P/total`; la cuantica hacia `P/(total+1e-15)`.
    Con `max(P)=1` eso es entre 0 y 2 ULPs — invisible salvo que el
    entrenamiento variacional lo amplifique, que es justo lo que hizo en
    `samplers_pede_nqpp*_noise-fakebrisbane`.
    """
    import numpy as np
    rng = np.random.default_rng(0)
    for n in (64, 256, 4096):
        P = np.exp(-rng.random(n) * 20.0)
        P[0] = 1.0                      # como deja build_target: max(P) = 1
        clasica = P / P.sum()
        cuantica = cmq.quantum_amplitude_normalization(P)
        assert np.array_equal(clasica, cuantica), (
            f"n={n}: la normalizacion cuantica no es bit a bit la clasica; "
            f"maxima diferencia {np.abs(clasica - cuantica).max():.3e}")


def test_b_eps15_sigue_normalizando():
    """Quitar el epsilon no debe romper lo obvio: la salida suma 1."""
    import numpy as np
    P = np.array([1.0, 0.5, 0.25, 0.125])
    out = cmq.quantum_amplitude_normalization(P)
    assert abs(float(out.sum()) - 1.0) < 1e-15
