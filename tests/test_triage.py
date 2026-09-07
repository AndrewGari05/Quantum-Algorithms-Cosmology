"""
test_triage.py — pruebas de `triage_campana.py`.

El triage no calcula fisica: decide QUE de una campana es citable. Por eso lo
que hay que blindar no son numeros sino tres decisiones, y las tres fallaron
la primera vez que se corrio el script contra carpetas de verdad:

  1. Detectar el OOMKill por la firma del log (una tarea muerta que se cuenta
     como "aun corriendo" hace perder un dia esperandola).
  2. Leer la UNION de columnas de todos los CSV, no las de uno. Los CSV de
     samplers y de genetico tienen esquemas distintos, asi que mirar uno solo
     declaraba "campana vieja" incluso con el codigo corregido — un falso
     positivo que tiraria resultados buenos a la basura.
  3. No abandonar una carpeta sin subcarpetas de tareas: una descarga parcial
     sigue teniendo CSV que leer.

Solo biblioteca estandar, igual que el script.
"""
import csv
import importlib.util
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_spec = importlib.util.spec_from_file_location(
    'triage_campana', os.path.join(ROOT, 'triage_campana.py'))
triage = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(triage)


# Esquema de los CSV de samplers antes y despues del 2026-09-04.
COLS_SAMPLERS_VIEJO = ['Method', 'model', 'nqpp', 'dataset', 'p1_mean',
                       'p1_std', 'p2_mean', 'p2_std', 'chi2', 'n_data',
                       'chi2_red', 'AIC', 'BIC']
COLS_SAMPLERS_NUEVO = COLS_SAMPLERS_VIEJO + ['noise', 'proposal_route',
                                             'seed', 'budget_mode',
                                             'circuits_train']
COLS_GENETICO_NUEVO = COLS_SAMPLERS_VIEJO + ['seed', 'chi2_grid']


def _fila(method, **extra):
    """Una fila de resultados con valores plausibles."""
    base = dict(zip(COLS_SAMPLERS_VIEJO,
                    [method, 'lcdm', '6', 'CC+BAO+Pantheon', '0.300',
                     '0.010', '70.0', '1.0', '1105.23', '1099', '1.0057',
                     '1109.23', '1119.24']))
    base.update(extra)
    return base


def _escribe_csv(path, cols, filas):
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
    """El log que termina en la firma del QVMC es una tarea muerta, no viva."""
    tarea = tmp_path / 'samplers_cpl_nqpp5_CC+BAO'
    tarea.mkdir()
    (tarea / 'task.log').write_text(
        'paso previo\n[i] ' + triage.FIRMA_OOM + '\n')
    assert triage.estado_de_tarea(str(tarea))[0] == 'oom'


def test_log_con_progreso_se_lee_como_corriendo(tmp_path):
    tarea = tmp_path / 'samplers_lcdm_nqpp5_CC+BAO'
    tarea.mkdir()
    (tarea / 'task.log').write_text('paso 300/1000 chain 2\n')
    assert triage.estado_de_tarea(str(tarea))[0] == 'corriendo'


def test_con_csv_la_tarea_esta_terminada_aunque_el_log_asuste(tmp_path):
    """Un CSV escrito manda sobre cualquier lectura del log."""
    tarea = tmp_path / 'genetic_lcdm_nb6_noise-none'
    tarea.mkdir()
    (tarea / 'task.log').write_text('[i] ' + triage.FIRMA_OOM + '\n')
    _escribe_csv(str(tarea / 'resultados_lcdm.csv'),
                 COLS_SAMPLERS_VIEJO, [_fila('QMCMC 100%')])
    assert triage.estado_de_tarea(str(tarea))[0] == 'ok'


def test_el_sufijo_de_qubits_multiplica_por_la_dimension_del_modelo():
    # cpl tiene 4 parametros: 5 qubits por parametro son 20 en total, que es
    # justo donde la campana de agosto se quedo sin RAM.
    assert triage._sufijo_qubits('samplers_cpl_nqpp5_CC+BAO') == '  (20 qubits)'
    assert triage._sufijo_qubits('samplers_lcdm_nqpp5_CC+BAO') == '  (10 qubits)'
    assert triage._sufijo_qubits('genetic_lcdm_nb6_noise-none') == ''


# ── 2. veredicto de citabilidad ──────────────────────────────────────────

def test_campana_vieja_se_marca_como_no_citable(tmp_path):
    _escribe_csv(str(tmp_path / 't1' / 'resultados_lcdm.csv'),
                 COLS_SAMPLERS_VIEJO, [_fila('QMCMC 100%')])
    salida = _captura(str(tmp_path))
    assert 'ANTERIOR al 2026-09-04' in salida
    assert 'NO citable' in salida


def test_esquemas_distintos_no_producen_falso_positivo(tmp_path):
    """La union de columnas: samplers aporta `noise`, genetico `chi2_grid`.

    Mirando un solo archivo siempre faltaria la mitad y la campana buena se
    declararia vieja. Este es el bug que salio al correrlo de verdad.
    """
    _escribe_csv(str(tmp_path / 'samplers_lcdm' / 'resultados_lcdm.csv'),
                 COLS_SAMPLERS_NUEVO,
                 [_fila('QMCMC 100%', noise='readout', proposal_route='q',
                        seed='42', budget_mode='circuits',
                        circuits_train='2281')])
    _escribe_csv(str(tmp_path / 'genetic_lcdm' / 'resultados_ga.csv'),
                 COLS_GENETICO_NUEVO,
                 [_fila('CGA', seed='42', chi2_grid='27.6502')])
    salida = _captura(str(tmp_path))
    assert 'Todo citable' in salida
    assert 'ANTERIOR' not in salida


def test_carpeta_sin_subcarpetas_igual_lee_los_csv(tmp_path):
    """Una descarga parcial no debe abortar el diagnostico."""
    _escribe_csv(str(tmp_path / 'resultados_lcdm.csv'),
                 COLS_SAMPLERS_VIEJO, [_fila('QMCMC 100%')])
    salida = _captura(str(tmp_path))
    assert 'no hay subcarpetas' in salida
    assert 'chi2_red' in salida          # llego a la seccion 2
    assert 'CITABLE' in salida           # y a la 3


# ── 3. celdas faithful ───────────────────────────────────────────────────

def test_par_faithful_roto_se_denuncia(tmp_path):
    """QVMC 67% y QVMC 100% deben coincidir: si no, hay un error real."""
    filas = [_fila('QVMC 67%'),
             _fila('QVMC 100%', p1_mean='0.999', p2_mean='99.9')]
    _escribe_csv(str(tmp_path / 't1' / 'resultados_lcdm.csv'),
                 COLS_SAMPLERS_VIEJO, filas)
    salida = _captura(str(tmp_path))
    assert 'REVISAR' in salida


def test_par_faithful_intacto_pasa(tmp_path):
    filas = [_fila('QMCMC 50%'), _fila('QMCMC 100%')]
    _escribe_csv(str(tmp_path / 't1' / 'resultados_lcdm.csv'),
                 COLS_SAMPLERS_VIEJO, filas)
    salida = _captura(str(tmp_path))
    assert 'REVISAR' not in salida
    assert '1/1  OK' in salida


# ── 4. los ejemplos de los docstrings ────────────────────────────────────

def test_los_ejemplos_de_los_docstrings_corren():
    import doctest
    res = doctest.testmod(triage, verbose=False)
    assert res.failed == 0, f"{res.failed} doctests fallaron en triage_campana"
