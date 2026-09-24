"""
test_semillas.py — pruebas de `comparar_semillas.py` y del eje n_bits.

El comparador de semillas existe para responder una sola pregunta: ¿la
diferencia que quiero reportar es mayor que lo que el numero se mueve solo
por cambiar la semilla? Todo lo que se blinda aqui son las formas en que esa
respuesta puede salir mal SIN fallar:

  * emparejar mal los metodos (y entonces la comparacion nunca aparece);
  * comparar por columnas que no existen (y entonces todo sale "identico");
  * llamar significativa a una diferencia que cabe en la dispersion.

Solo biblioteca estandar, igual que el script.
"""
import csv
import importlib.util
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_spec = importlib.util.spec_from_file_location(
    'comparar_semillas', os.path.join(ROOT, 'comparar_semillas.py'))
cs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cs)

COLS = ['Method', 'Om_mean', 'Om_std', 'H0_mean', 'H0_std', 'Time_s', 'nqpp',
        'chi2', 'n_data', 'chi2_red', 'AIC', 'BIC', 'acceptance', 'final_KL',
        'ESS', 'dataset', 'prior', 'noise', 'proposal_route', 'seed',
        'budget_mode', 'circuits_train', 'chi2_grid']


def _fila(method, semilla, om='0.2764', kl='', acc='', nqpp='3', **extra):
    r = dict.fromkeys(COLS, '')
    r.update({'Method': method, 'Om_mean': om, 'Om_std': '0.0104',
              'H0_mean': '69.60', 'H0_std': '0.80', 'nqpp': nqpp,
              'chi2': '1064.61', 'n_data': '1099', 'chi2_red': '0.970',
              'AIC': '1068.6', 'BIC': '1078.6', 'dataset': 'CC+BAO+Pantheon',
              'prior': 'flat', 'noise': 'none', 'seed': str(semilla),
              'final_KL': kl, 'acceptance': acc, 'ESS': '6900',
              'budget_mode': 'circuits'})
    r.update(extra)
    return r


def _campana(raiz, semilla, filas, sub='model_lcdm'):
    """Escribe una corrida de una semilla con la estructura real."""
    d = os.path.join(raiz, f'semilla_{semilla}', sub)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, 'resultados_config.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, COLS)
        w.writeheader()
        w.writerows(filas)


def _captura(carpetas):
    import io
    from contextlib import redirect_stdout
    buf = io.StringIO()
    with redirect_stdout(buf):
        cs.revisar(carpetas)
    return buf.getvalue()


# ── estadistica ──────────────────────────────────────────────────────────

def test_la_dispersion_es_muestral_no_poblacional():
    """Con n semillas el divisor es n-1: subestimar la sigma infla la
    significancia de todo lo que se reporte."""
    m, s, n = cs.dispersion([1.0, 2.0, 3.0])
    assert (round(m, 6), round(s, 6), n) == (2.0, 1.0, 3)


def test_una_sola_semilla_no_tiene_dispersion():
    assert cs.dispersion([5.0]) == (5.0, 0.0, 1)
    assert cs.en_sigmas([1.0], [2.0])[2] is None


def test_diferencia_pequena_frente_a_la_dispersion_no_es_significativa():
    """Dos metodos que se solapan: la separacion debe salir por debajo de 1."""
    a = [1.6250, 1.6300, 1.6200]
    b = [1.6286, 1.6240, 1.6330]
    _, _, ns = cs.en_sigmas(a, b)
    assert abs(ns) < 1.0


def test_diferencia_grande_frente_a_la_dispersion_si_lo_es():
    _, _, ns = cs.en_sigmas([1.00, 1.01, 0.99], [2.00, 2.01, 1.99])
    assert abs(ns) > 3.0


# ── emparejado de celdas ─────────────────────────────────────────────────

def test_el_rung_clasico_se_empareja_pese_a_tener_nqpp_distinto(tmp_path):
    """`Classical VI` escribe nqpp='—' y `QVMC 100%` escribe el numero.

    Agrupando por nqpp cada uno caia en su propio grupo y la comparacion
    clasico-contra-cuantico no llegaba a hacerse nunca: el reporte no decia
    'no significativa', decia nada.
    """
    for s in (1, 2, 3):
        _campana(str(tmp_path), s, [
            _fila('Classical VI', s, kl=str(1.60 + 0.001 * s), nqpp='—'),
            _fila('QVMC 100%', s, kl=str(2.60 + 0.001 * s), nqpp='3'),
        ])
    salida = _captura([str(tmp_path / f'semilla_{s}') for s in (1, 2, 3)])
    assert 'Classical VI vs QVMC 100%' in salida
    assert 'sigmas' in salida


def test_celda_faithful_que_falla_en_una_sola_semilla_se_denuncia(tmp_path):
    """Coincidir en dos semillas y fallar en la tercera es un bug, no azar."""
    for s in (1, 2, 3):
        om_b = '0.2764' if s != 3 else '0.9999'
        _campana(str(tmp_path), s, [
            _fila('QMCMC 50%', s), _fila('QMCMC 100%', s, om=om_b)])
    salida = _captura([str(tmp_path / f'semilla_{s}') for s in (1, 2, 3)])
    assert 'REVISAR' in salida
    assert 'semilla=3' in salida


def test_celdas_faithful_intactas_pasan(tmp_path):
    for s in (1, 2, 3):
        _campana(str(tmp_path), s,
                 [_fila('QMCMC 50%', s), _fila('QMCMC 100%', s)])
    salida = _captura([str(tmp_path / f'semilla_{s}') for s in (1, 2, 3)])
    assert 'REVISAR' not in salida
    assert '3/3   OK' in salida


def test_una_sola_semilla_avisa_en_vez_de_inventar_barras(tmp_path):
    _campana(str(tmp_path), 42, [_fila('QMCMC 50%', 42)])
    salida = _captura([str(tmp_path / 'semilla_42')])
    assert 'UNA sola semilla' in salida


def test_campana_sin_columna_seed_se_rechaza(tmp_path):
    """Las campanas anteriores a [B-PROV] no sirven para este estudio."""
    d = tmp_path / 'vieja' / 'model_lcdm'
    d.mkdir(parents=True)
    sin_seed = [c for c in COLS if c != 'seed']
    with open(d / 'resultados_x.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, sin_seed)
        w.writeheader()
        w.writerow({k: v for k, v in _fila('QMCMC 50%', 1).items()
                    if k != 'seed'})
    salida = _captura([str(tmp_path / 'vieja')])
    assert 'seed' in salida and 'corregido' in salida


# ── [B-NBITS] el eje del genetico ────────────────────────────────────────

def test_b_nbits_las_familias_se_separan():
    """nqpp y n_bits no son la misma cantidad y no comparten eje."""
    import cosmo_hpc_runner as R
    assert R._familia_de_metodo('CGA') == 'genetic'
    assert R._familia_de_metodo('QGA (q=33%)') == 'genetic'
    assert R._familia_de_metodo('QMCMC 50%') == 'samplers'
    assert R._familia_de_metodo('QVMC 100%') == 'samplers'
    assert R._familia_de_metodo('Classical VI') == 'samplers'
    assert R._familia_de_metodo('Classical MCMC') == 'samplers'


@pytest.mark.parametrize('familia,sufijo,metodos', [
    ('samplers', '', ['QMCMC 50%', 'QMCMC 100%']),
    ('genetic', '_genetico', ['CGA', 'QGA (q=33%)']),
])
def test_b_nbits_cada_familia_produce_su_figura(tmp_path, familia, sufijo,
                                                metodos):
    """Y el eje x solo lleva los valores de rejilla de ESA familia.

    Sin el filtro, la figura del genetico heredaba los nqpp de los samplers
    y salia con media grafica vacia en una escala que no era de ninguna.
    """
    import matplotlib
    matplotlib.use('Agg')
    import cosmo_hpc_runner as R

    prefijo = 'samplers' if familia == 'samplers' else 'genetic'
    for g in (4, 5):
        # El runner deduce el modelo del nombre de la carpeta de tarea
        # (`samplers_<modelo>_...`), no de una columna del CSV.
        d = tmp_path / f'{prefijo}_lcdm_g{g}' / 'model_lcdm'
        d.mkdir(parents=True)
        with open(d / 'resultados_config.csv', 'w', newline='') as fh:
            w = csv.DictWriter(fh, COLS)
            w.writeheader()
            for m in metodos:
                w.writerow(_fila(m, 42, om=f'0.2{70+g}', nqpp=str(g)))
    salida = tmp_path / 'fig'
    hechas = R.generate_convergence_plots(
        str(tmp_path), outdir=str(salida),
        xlabel='nqpp' if familia == 'samplers' else 'n_bits',
        familia=familia, sufijo=sufijo)
    esperado = os.path.join(str(salida), f'convergence_lcdm{sufijo}.png')
    assert esperado in hechas, f"no se genero {esperado}: {hechas}"
    assert os.path.getsize(esperado) > 5000
