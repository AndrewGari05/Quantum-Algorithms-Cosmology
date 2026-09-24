"""
test_triage.py — pruebas de `triage_campana.py`.

El triage no calcula fisica: decide QUE de una campana es citable y si las
celdas faithful siguen coincidiendo. Por eso lo que hay que blindar no son
numeros sino decisiones, y todas las de aqui fallaron la primera vez que el
script se corrio contra salidas de verdad.

Los esquemas de abajo son los REALES, copiados de la cabecera que escriben
los dos modulos. No coinciden entre si, y esa es justamente la trampa:

  * samplers  -> `Om_mean`, `H0_mean`, ... (nombres por parametro), sin
    columna `model`.
  * genetico  -> `p1_mean`..`p4_std` (nombres genericos), CON columna
    `model` y `params`.

Un triage que fije los nombres de columna a mano acierta en uno y falla en
silencio en el otro: `fila.get('p1_mean')` da None en ambos lados de una
comparacion de samplers, None == None sale True, y la celda faithful se
declara OK sin haber comprobado nada. Eso es peor que no comprobar.

Solo biblioteca estandar, igual que el script.
"""
import csv
import importlib.util
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_spec = importlib.util.spec_from_file_location(
    'triage_campana', os.path.join(ROOT, 'triage_campana.py'))
triage = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(triage)


# Cabecera real de cosmo_modular_quantum.py (samplers).
COLS_SAMPLERS = ['Method', 'Om_mean', 'Om_std', 'H0_mean', 'H0_std', 'Time_s',
                 'nqpp', 'chi2', 'n_data', 'chi2_red', 'AIC', 'BIC',
                 'acceptance', 'final_KL', 'ESS', 'dataset', 'prior', 'noise',
                 'proposal_route', 'seed', 'budget_mode', 'circuits_train',
                 'chi2_grid']

# Cabecera real de cosmo_genetic_optimizers.py (--sweep-all).
COLS_GENETICO = ['Method', 'model', 'params', 'p1_mean', 'p1_std', 'p2_mean',
                 'p2_std', 'p3_mean', 'p3_std', 'p4_mean', 'p4_std', 'Time_s',
                 'nqpp', 'chi2', 'n_data', 'chi2_red', 'AIC', 'BIC',
                 'acceptance', 'final_KL', 'ESS', 'dataset', 'prior', 'noise',
                 'proposal_route', 'seed', 'budget_mode', 'circuits_train',
                 'chi2_grid']

# Las mismas cabeceras antes del 2026-09-04: sin columnas de procedencia.
COLS_SAMPLERS_VIEJO = [c for c in COLS_SAMPLERS
                       if c not in triage.COLUMNAS_NUEVAS]

AJUSTE = {'chi2': '27.4691', 'n_data': '51', 'chi2_red': '0.5606',
          'AIC': '31.4691', 'BIC': '35.3328', 'dataset': 'CC+BAO',
          'prior': 'flat', 'noise': 'none'}


def _samplers(method, om='0.269366', h0='69.988624', nqpp='2', **extra):
    """Fila de samplers con los nombres de columna reales."""
    r = dict.fromkeys(COLS_SAMPLERS, '')
    r.update(AJUSTE)
    r.update({'Method': method, 'Om_mean': om, 'Om_std': '0.010',
              'H0_mean': h0, 'H0_std': '1.0', 'nqpp': nqpp,
              'seed': '42', 'budget_mode': 'circuits',
              'proposal_route': 'auto', 'circuits_train': '2281'})
    r.update(extra)
    return r


def _genetico(method, p1='0.262709', p2='70.388775', nqpp='4', **extra):
    """Fila del barrido genetico con los nombres de columna reales."""
    r = dict.fromkeys(COLS_GENETICO, '')
    r.update(AJUSTE)
    r.update({'Method': method, 'model': 'lcdm', 'params': 'Om|H0',
              'p1_mean': p1, 'p1_std': '0.001124', 'p2_mean': p2,
              'p2_std': '0.175031', 'nqpp': nqpp, 'seed': '42',
              'budget_mode': 'circuits', 'proposal_route': 'auto',
              'circuits_train': '0', 'chi2_grid': '27.5367'})
    r.update(extra)
    return r


def _escribe(path, cols, filas):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, cols, extrasaction='ignore')
        w.writeheader()
        w.writerows(filas)


def _captura(master):
    """Corre `revisar` sobre una carpeta y devuelve lo que imprimio."""
    import io
    from contextlib import redirect_stdout
    buf = io.StringIO()
    with redirect_stdout(buf):
        triage.revisar(master)
    return buf.getvalue()


# ── 1. estado de las tareas ──────────────────────────────────────────────

def test_log_cortado_en_la_firma_se_lee_como_oom(tmp_path):
    """El log que termina en la firma del QVMC es tarea muerta, no viva."""
    tarea = tmp_path / 'samplers_cpl_nqpp5_CC+BAO'
    tarea.mkdir()
    (tarea / 'task.log').write_text('previo\n[i] ' + triage.FIRMA_OOM + '\n')
    assert triage.estado_de_tarea(str(tarea))[0] == 'oom'


def test_log_con_progreso_se_lee_como_corriendo(tmp_path):
    tarea = tmp_path / 'samplers_lcdm_nqpp5_CC+BAO'
    tarea.mkdir()
    (tarea / 'task.log').write_text('paso 300/1000 chain 2\n')
    assert triage.estado_de_tarea(str(tarea))[0] == 'corriendo'


def test_con_csv_la_tarea_esta_terminada_aunque_el_log_asuste(tmp_path):
    tarea = tmp_path / 'genetic_lcdm_nb6_noise-none'
    tarea.mkdir()
    (tarea / 'task.log').write_text('[i] ' + triage.FIRMA_OOM + '\n')
    _escribe(str(tarea / 'resultados_lcdm.csv'), COLS_SAMPLERS,
             [_samplers('QMCMC 100%')])
    assert triage.estado_de_tarea(str(tarea))[0] == 'ok'


def test_el_sufijo_de_qubits_multiplica_por_la_dimension_del_modelo():
    # cpl tiene 4 parametros: 5 qubits por parametro son 20, que es justo
    # donde la campana de agosto se quedo sin RAM.
    assert triage._sufijo_qubits('samplers_cpl_nqpp5_CC+BAO') == '  (20 qubits)'
    assert triage._sufijo_qubits('samplers_lcdm_nqpp5_CC+BAO') == '  (10 qubits)'
    assert triage._sufijo_qubits('genetic_lcdm_nb6_noise-none') == ''


# ── 2. descubrimiento de columnas y del modelo ───────────────────────────

def test_las_columnas_de_parametros_se_descubren_no_se_fijan():
    """Los dos esquemas reales deben resolverse solos."""
    assert triage.columnas_de_parametros(set(COLS_SAMPLERS)) == [
        'H0_mean', 'H0_std', 'Om_mean', 'Om_std']
    assert triage.columnas_de_parametros(set(COLS_GENETICO)) == [
        'p1_mean', 'p1_std', 'p2_mean', 'p2_std',
        'p3_mean', 'p3_std', 'p4_mean', 'p4_std']


def test_comparar_por_columnas_inexistentes_no_puede_salir_ok():
    """El bug de la falsa celda faithful.

    Dos filas identicas comparadas por una columna que ninguna tiene daban
    `None == None` -> True, y el triage cantaba OK sin comprobar nada.
    """
    a = _samplers('QMCMC 50%')
    b = _samplers('QMCMC 100%')
    assert triage._iguales(a, b, ['Om_mean', 'H0_mean']) is True
    assert triage._iguales(a, b, ['p1_mean', 'p2_mean']) is False


def test_el_modelo_sale_de_la_ruta_cuando_el_csv_no_lo_trae():
    assert triage.modelo_de_ruta('run/model_wcdm/resultados_config.csv') == 'wcdm'
    assert triage.modelo_de_ruta('run/resultados_cpl.csv') == 'cpl'
    assert triage.modelo_de_ruta('run/resultados_config.csv') == '?'


# ── 3. veredicto de citabilidad ──────────────────────────────────────────

def test_campana_vieja_se_marca_como_no_citable(tmp_path):
    _escribe(str(tmp_path / 't1' / 'resultados_lcdm.csv'),
             COLS_SAMPLERS_VIEJO, [_samplers('QMCMC 100%')])
    salida = _captura(str(tmp_path))
    assert 'ANTERIOR al 2026-09-04' in salida
    assert 'NO citable' in salida


def test_esquemas_distintos_no_producen_falso_positivo(tmp_path):
    """La union de columnas entre los dos esquemas reales."""
    _escribe(str(tmp_path / 'samplers_lcdm' / 'resultados_config.csv'),
             COLS_SAMPLERS, [_samplers('QMCMC 100%')])
    _escribe(str(tmp_path / 'genetic_lcdm' / 'resultados_config.csv'),
             COLS_GENETICO, [_genetico('CGA')])
    salida = _captura(str(tmp_path))
    assert 'Todo citable' in salida
    assert 'ANTERIOR' not in salida


def test_carpeta_sin_subcarpetas_igual_lee_los_csv(tmp_path):
    """Una descarga parcial no debe abortar el diagnostico."""
    _escribe(str(tmp_path / 'resultados_lcdm.csv'), COLS_SAMPLERS,
             [_samplers('QMCMC 100%')])
    salida = _captura(str(tmp_path))
    assert 'no hay subcarpetas' in salida
    assert 'chi2_red' in salida          # llego a la seccion 2
    assert 'CITABLE' in salida           # y a la 3


# ── 4. celdas faithful ───────────────────────────────────────────────────

def test_par_faithful_roto_se_denuncia(tmp_path):
    """QVMC 67% y QVMC 100% deben coincidir: si no, hay un error real."""
    _escribe(str(tmp_path / 't1' / 'resultados_lcdm.csv'), COLS_SAMPLERS,
             [_samplers('QVMC 67%'),
              _samplers('QVMC 100%', om='0.999', h0='99.9')])
    assert 'REVISAR' in _captura(str(tmp_path))


def test_par_faithful_intacto_pasa(tmp_path):
    _escribe(str(tmp_path / 't1' / 'resultados_lcdm.csv'), COLS_SAMPLERS,
             [_samplers('QMCMC 50%'), _samplers('QMCMC 100%')])
    salida = _captura(str(tmp_path))
    assert 'REVISAR' not in salida
    assert '1/1  OK' in salida


def test_el_rung_clasico_no_se_pierde_por_tener_nqpp_distinto(tmp_path):
    """CGA escribe nqpp='—' y QGA (q=0%) escribe el numero.

    Agrupando por nqpp cada uno caia en su propio grupo y el par no llegaba
    a compararse NUNCA: el triage no decia 'mal', decia nada, que es la
    forma mas facil de que un error pase inadvertido.
    """
    _escribe(str(tmp_path / 'g' / 'resultados_config.csv'), COLS_GENETICO,
             [_genetico('CGA', nqpp='—'), _genetico('QGA (q=0%)', nqpp='4')])
    salida = _captura(str(tmp_path))
    assert 'CGA' in salida and '1/1  OK' in salida


def test_classical_vi_y_qvmc_33_no_se_exigen_iguales(tmp_path):
    """No son una celda faithful: el rung 33% cambia el muestreo.

    Medido: Om = 0.258460 (VI) contra 0.258262 (QVMC 33%). Exigirles
    igualdad marcaba un fallo falso en cada celda de cada campana.
    """
    _escribe(str(tmp_path / 't1' / 'resultados_lcdm.csv'), COLS_SAMPLERS,
             [_samplers('Classical VI', om='0.258460', nqpp='—'),
              _samplers('QVMC 33%', om='0.258262')])
    assert 'REVISAR' not in _captura(str(tmp_path))


def test_sin_columnas_de_parametros_avisa_en_vez_de_cantar_ok(tmp_path):
    _escribe(str(tmp_path / 't1' / 'resultados_lcdm.csv'),
             ['Method', 'chi2', 'n_data', 'chi2_red', 'AIC', 'BIC'],
             [{'Method': 'QMCMC 50%', **{k: AJUSTE[k] for k in
                                         ('chi2', 'n_data', 'chi2_red',
                                          'AIC', 'BIC')}}])
    salida = _captura(str(tmp_path))
    assert 'NO se puede comprobar' in salida
    assert 'OK' not in salida.split('4. CELDAS')[-1]


# ── 5. los ejemplos de los docstrings ────────────────────────────────────

def test_los_ejemplos_de_los_docstrings_corren():
    import doctest
    res = doctest.testmod(triage, verbose=False)
    assert res.failed == 0, f"{res.failed} doctests fallaron en triage_campana"


# ── 6. el eje de ruido no debe disparar falsas alarmas ───────────────────

def test_con_ruido_la_separacion_es_lo_esperado_no_un_fallo(tmp_path):
    """Sin ruido los dos rungs coinciden; con ruido DEBEN separarse.

    Contando ambos ejes juntos, una campana sana daba '40/62 REVISAR'
    (medido sobre hpc_20260903_161310: 20/20 celdas ideales identicas,
    11/11 ruidosas distintas, cero excepciones). Una alarma que salta
    siempre es una alarma que se deja de leer.
    """
    _escribe(str(tmp_path / 'ideal' / 'resultados_lcdm.csv'), COLS_SAMPLERS,
             [_samplers('QMCMC 50%', noise='none'),
              _samplers('QMCMC 100%', noise='none')])
    _escribe(str(tmp_path / 'ruido' / 'resultados_lcdm.csv'), COLS_SAMPLERS,
             [_samplers('QMCMC 50%', noise='readout'),
              _samplers('QMCMC 100%', om='0.2731', noise='readout')])
    salida = _captura(str(tmp_path))
    assert 'REVISAR' not in salida
    assert '1/1  OK' in salida.split('Sin ruido')[1].split('Con ruido')[0]


def test_un_rung_ruidoso_identico_al_ideal_se_denuncia(tmp_path):
    """Si el ruido no se aplico, los dos rungs salen iguales: eso SI es error."""
    _escribe(str(tmp_path / 'ruido' / 'resultados_lcdm.csv'), COLS_SAMPLERS,
             [_samplers('QMCMC 50%', noise='full'),
              _samplers('QMCMC 100%', noise='full')])
    salida = _captura(str(tmp_path))
    assert 'el ruido no llego' in salida


def test_none_counts_cuenta_como_eje_ideal(tmp_path):
    """La columna de control ideal-por-conteos sigue siendo faithful."""
    _escribe(str(tmp_path / 'c' / 'resultados_lcdm.csv'), COLS_SAMPLERS,
             [_samplers('QMCMC 50%', noise='none-counts'),
              _samplers('QMCMC 100%', noise='none-counts')])
    salida = _captura(str(tmp_path))
    assert 'Sin ruido' in salida and 'REVISAR' not in salida


def test_el_nivel_de_ruido_sale_de_la_carpeta_si_no_hay_columna(tmp_path):
    """Campana vieja: no hay columna `noise`, pero la carpeta lo dice.

    Sin esta caida a la ruta, hpc_20260903_161310 (que es anterior a
    [B-PROV]) mandaba sus 11 celdas ruidosas al cubo ideal y salia
    '40/62 REVISAR' sobre una campana sana.
    """
    _escribe(str(tmp_path / 'samplers_lcdm_nqpp5_noise-full' /
                 'resultados_lcdm.csv'), COLS_SAMPLERS_VIEJO,
             [_samplers('QMCMC 50%'),
              _samplers('QMCMC 100%', om='0.2731')])
    salida = _captura(str(tmp_path))
    assert 'REVISAR' not in salida
    assert 'Con ruido' in salida


def test_los_pares_inmunes_al_ruido_no_se_denuncian(tmp_path):
    """QVMC 67 vs 100 y CGA vs QGA(0%) NO deben separarse con ruido.

    La normalizacion ejecuta su circuito pero descarta el resultado y usa la
    suma exacta (ver quantum_amplitude_normalization), y en q=0% no hay
    componente cuantico. Exigirles separacion denunciaba 70 de 92 celdas
    sanas de hpc_20260903_161310.
    """
    _escribe(str(tmp_path / 'samplers_lcdm_nqpp5_noise-full' /
                 'resultados_lcdm.csv'), COLS_SAMPLERS_VIEJO,
             [_samplers('QVMC 67%'), _samplers('QVMC 100%')])
    _escribe(str(tmp_path / 'genetic_lcdm_nb5_noise-full' /
                 'resultados_g.csv'), COLS_GENETICO,
             [_genetico('CGA'), _genetico('QGA (q=0%)')])
    salida = _captura(str(tmp_path))
    assert 'REVISAR' not in salida
    assert 'INMUNES' in salida


def test_un_par_inmune_que_si_se_separa_se_denuncia(tmp_path):
    """Si la normalizacion empieza a depender del ruido, hay que enterarse."""
    _escribe(str(tmp_path / 'samplers_lcdm_nqpp5_noise-full' /
                 'resultados_lcdm.csv'), COLS_SAMPLERS_VIEJO,
             [_samplers('QVMC 67%'), _samplers('QVMC 100%', om='0.9')])
    assert 'REVISAR' in _captura(str(tmp_path))
