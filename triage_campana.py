#!/usr/bin/env python3
"""
triage_campana.py — Diagnostico rapido de una carpeta de resultados.

Contesta en un vistazo las tres preguntas que uno se hace al volver a una
campana larga:

    1. ¿Que tareas murieron, cuales siguen corriendo y cuales acabaron bien?
    2. ¿Son buenos los ajustes?
    3. ¿Que columnas de este CSV son citables y cuales no?

La tercera es la que importa mas y la que no se ve mirando archivos: una
campana corrida con codigo anterior al 2026-09-04 arrastra bugs conocidos que
invalidan columnas concretas, y este script lo detecta por la AUSENCIA de las
columnas de procedencia (`noise`, `budget_mode`, `chi2_grid`).

Uso:

    python triage_campana.py results/hpc_20260903_161310
    python triage_campana.py results/hpc_*            # varias a la vez

No necesita numpy ni qiskit: solo la biblioteca estandar, para poder correrlo
en cualquier sitio, incluso en el nodo de la HPC sin el entorno activado.
"""

from __future__ import annotations

import csv
import glob
import os
import re
import sys
from collections import defaultdict

#: Ultima linea que escribe una tarea de samplers antes de arrancar el QVMC.
#: Si el log termina AHI y no hay CSV, es la firma del OOM killer.
FIRMA_OOM = 'Adaptive QVMC grid window'

#: Columnas que solo existen desde el 2026-09-04. Su ausencia marca una
#: campana corrida con codigo que arrastra bugs conocidos.
COLUMNAS_NUEVAS = ('noise', 'proposal_route', 'seed', 'budget_mode',
                   'circuits_train', 'chi2_grid')


def _leer_csv(path):
    """Filas de un CSV de resultados, o lista vacia si no se puede leer."""
    try:
        with open(path, newline='') as fh:
            return list(csv.DictReader(fh))
    except Exception:
        return []


def _ultima_linea(path):
    """Ultima linea no vacia de un archivo de texto."""
    try:
        with open(path, errors='ignore') as fh:
            lineas = [l.rstrip('\n') for l in fh if l.strip()]
        return lineas[-1] if lineas else ''
    except Exception:
        return ''


#: Numero de parametros libres por modelo. Un nqpp de N qubits por parametro
#: da N * DIM_MODELO qubits en total, que es lo que decide si la tarea cabe
#: en RAM: el estado denso pesa 16 * 4**n_qubits bytes.
DIM_MODELO = {'lcdm': 2, 'pede': 2, 'wcdm': 3, 'gede': 3, 'cpl': 4}


def _sufijo_qubits(nombre):
    """Anota el total de qubits de una tarea a partir de su nombre.

    Args:
        nombre: nombre de la carpeta, p.ej. `samplers_cpl_nqpp5_CC+BAO`.

    Returns:
        Cadena `'  (20 qubits)'`, o `''` si el nombre no lo permite deducir.

    Examples:
        >>> _sufijo_qubits('samplers_cpl_nqpp5_CC+BAO')
        '  (20 qubits)'
        >>> _sufijo_qubits('genetic_lcdm_nb6_noise-none')
        ''
    """
    q = re.search(r'nqpp(\d+)', nombre)
    m = re.search(r'samplers_(\w+?)_', nombre)
    if q and m and m.group(1) in DIM_MODELO:
        return f"  ({int(q.group(1)) * DIM_MODELO[m.group(1)]} qubits)"
    return ''


def columnas_de_parametros(cols):
    """Columnas de parametros cosmologicos presentes en el CSV.

    Los nombres dependen del modelo (`Om_mean`, `H0_std`, `w0_mean`...), asi
    que no se pueden fijar de antemano: se descubren por el sufijo.

    Args:
        cols: conjunto de nombres de columna del CSV.

    Returns:
        Lista ordenada de las columnas que acaban en `_mean` o `_std`.

    Examples:
        >>> columnas_de_parametros({'Method', 'Om_mean', 'Om_std', 'chi2'})
        ['Om_mean', 'Om_std']
        >>> columnas_de_parametros({'Method', 'chi2'})
        []
    """
    return sorted(c for c in cols
                  if c.endswith('_mean') or c.endswith('_std'))


def _iguales(a, b, campos):
    """¿Coinciden dos filas en todos los campos dados?

    Compara solo los campos que EXISTEN en ambas filas y exige que haya al
    menos uno. Sin esa condicion, comparar dos filas por nombres de columna
    inexistentes da `None == None` para todo y la celda faithful sale "OK"
    sin haber comprobado nada — que es justo lo contrario de lo que sirve
    esta seccion.

    Args:
        a, b: las dos filas a comparar.
        campos: nombres de columna a cotejar.

    Returns:
        True si coinciden en todos los campos comunes; False si difieren o
        si no comparten ninguno.

    Examples:
        >>> _iguales({'x': '1'}, {'x': '1'}, ['x'])
        True
        >>> _iguales({'x': '1'}, {'x': '2'}, ['x'])
        False
        >>> _iguales({'x': '1'}, {'x': '1'}, ['no_existe'])
        False
    """
    comunes = [c for c in campos if c in a and c in b]
    if not comunes:
        return False
    return all(a[c] == b[c] for c in comunes)


def modelo_de_ruta(path):
    """Deduce el modelo cosmologico de la ruta de un CSV.

    El CSV no lleva columna `model`: el modelo esta en la carpeta
    (`model_lcdm/`) o en el nombre del archivo (`resultados_lcdm.csv`).

    Args:
        path: ruta del CSV.

    Returns:
        Clave del modelo, o `'?'` si la ruta no lo dice.

    Examples:
        >>> modelo_de_ruta('run/model_wcdm/resultados_config.csv')
        'wcdm'
        >>> modelo_de_ruta('run/resultados_cpl.csv')
        'cpl'
        >>> modelo_de_ruta('run/resultados_config.csv')
        '?'
    """
    partes = path.replace('\\', '/').split('/')
    for p in partes:
        m = re.match(r'model_(\w+)$', p)
        if m and m.group(1) in DIM_MODELO:
            return m.group(1)
    m = re.search(r'resultados_(\w+)\.csv$', partes[-1])
    if m and m.group(1) in DIM_MODELO:
        return m.group(1)
    for p in partes:
        m = re.search(r'_(' + '|'.join(DIM_MODELO) + r')_', p)
        if m:
            return m.group(1)
    return '?'


def nivel_de_ruido(fila, ruta):
    """Nivel de ruido de una fila: columna si existe, si no la ruta.

    En las campanas anteriores al 2026-09-04 no hay columna `noise` — es una
    de las que introdujo [B-PROV] —, pero el runner SI pone el nivel en el
    nombre de la carpeta (`samplers_lcdm_nqpp5_noise-full`). Sin esta caida
    a la ruta, toda campana vieja parece ideal y las celdas ruidosas se
    denuncian como error cuando estan haciendo justo lo que deben.

    Args:
        fila: la fila del CSV.
        ruta: ruta del archivo del que salio.

    Returns:
        El nivel como cadena; `'none'` si no se puede determinar.

    Examples:
        >>> nivel_de_ruido({'noise': 'readout'}, 'x/samplers_lcdm_noise-full/r.csv')
        'readout'
        >>> nivel_de_ruido({}, 'x/samplers_lcdm_nqpp5_noise-full/r.csv')
        'full'
        >>> nivel_de_ruido({'noise': ''}, 'x/genetic_lcdm_nb6/r.csv')
        'none'
    """
    if fila.get('noise'):
        return fila['noise']
    m = re.search(r'noise-([A-Za-z0-9_.-]+)', ruta.replace('\\', '/'))
    return m.group(1) if m else 'none'


def estado_de_tarea(carpeta):
    """Clasifica una carpeta de tarea.

    Args:
        carpeta: ruta de la subcarpeta de una tarea.

    Returns:
        `(estado, detalle)`, con estado en {'ok', 'oom', 'corriendo',
        'sin_log'}.
    """
    tiene_csv = bool(glob.glob(os.path.join(carpeta, 'resultados_*.csv')))
    logs = sorted(glob.glob(os.path.join(carpeta, '*.log')))
    logs = [l for l in logs if os.path.getsize(l) > 0]
    if tiene_csv:
        return 'ok', ''
    if not logs:
        return 'sin_log', 'la tarea no llego a escribir nada'
    ult = _ultima_linea(logs[-1])
    if FIRMA_OOM in ult:
        return 'oom', 'el log corta al arrancar el QVMC (SIGKILL del OOM killer)'
    return 'corriendo', ult[-70:]


def revisar(master_dir):
    """Imprime el diagnostico de una carpeta de campana."""
    print(f"\n{'='*74}\n{master_dir}\n{'='*74}")
    subs = sorted(d for d in glob.glob(os.path.join(master_dir, '*'))
                  if os.path.isdir(d))

    # ── 1. estado de cada tarea ──────────────────────────────────────────
    # Una campana puede no tener subcarpetas (descarga parcial, o una corrida
    # de una sola tarea que escribe en la raiz). En ese caso no hay nada que
    # clasificar, pero las secciones 2-4 siguen siendo utiles: los CSV se
    # buscan de forma recursiva desde la raiz.
    if not subs:
        print("\n1. TAREAS: no hay subcarpetas de tareas en esta carpeta")
        print("   (descarga parcial, o corrida de una sola tarea)")
    else:
        por_estado = defaultdict(list)
        for d in subs:
            est, det = estado_de_tarea(d)
            por_estado[est].append((os.path.basename(d), det))

        print(f"\n1. TAREAS  ({len(subs)} carpetas)")
        for est, etiqueta in (('ok', 'terminadas'),
                              ('oom', 'MUERTAS (OOMKill)'),
                              ('corriendo', 'aun corriendo'),
                              ('sin_log', 'sin log')):
            items = por_estado.get(est, [])
            if not items:
                continue
            print(f"   {etiqueta}: {len(items)}")
            if est in ('oom', 'sin_log'):
                for nombre, det in items:
                    print(f"      - {nombre}{_sufijo_qubits(nombre)}")
                    print(f"        {det}")
            elif est == 'corriendo':
                for nombre, det in items[:6]:
                    print(f"      - {nombre}: ...{det}")
                if len(items) > 6:
                    print(f"      ... y {len(items)-6} mas")

    # ── 2. ajuste ────────────────────────────────────────────────────────
    # Los CSV de samplers y de genetico NO comparten esquema: `noise` y
    # `budget_mode` solo existen en el primero, `chi2_grid` solo en el
    # segundo. Por eso la seccion 3 mira la UNION de columnas de todos los
    # archivos, no las de una fila cualquiera: mirar una sola daria siempre
    # "faltan columnas" aunque la campana sea del codigo corregido.
    filas, cols = [], set()
    for p in glob.glob(os.path.join(master_dir, '**', 'resultados_*.csv'),
                       recursive=True):
        nuevas = _leer_csv(p)
        # El modelo no viene como columna: se anota desde la ruta para poder
        # agrupar por modelo en las secciones 2 y 4.
        mod = modelo_de_ruta(p)
        for r in nuevas:
            # El CSV del barrido genetico SI trae columna `model`; el de
            # samplers no. Se prefiere la del archivo y se cae a la ruta.
            r['_modelo'] = r.get('model') or mod
            r['_archivo'] = p
            r['_ruido'] = nivel_de_ruido(r, p)
        filas += nuevas
        for r in nuevas[:1]:
            cols |= set(r.keys())
    if not filas:
        print("\n2. AJUSTE: aun no hay ningun CSV")
        return

    print("\n2. AJUSTE  (chi2 del mejor ajuste, identico en todos los rungs)")
    vistos = {}
    for r in filas:
        mod = r.get('_modelo') or '?'
        if mod not in vistos and r.get('chi2'):
            vistos[mod] = (r.get('chi2'), r.get('n_data'), r.get('chi2_red'),
                           r.get('AIC'), r.get('BIC'))
    print(f"   {'modelo':8s}{'chi2':>12s}{'n':>7s}{'chi2_red':>10s}"
          f"{'AIC':>11s}{'BIC':>11s}")
    for mod, v in sorted(vistos.items(), key=lambda x: float(x[1][3] or 0)):
        print(f"   {mod:8s}{v[0]:>12s}{v[1]:>7s}{v[2]:>10s}{v[3]:>11s}"
              f"{v[4]:>11s}")
    reds = [float(v[2]) for v in vistos.values() if v[2]]
    if reds:
        lo, hi = min(reds), max(reds)
        if 0.8 <= lo and hi <= 1.2:
            print(f"   -> chi2_red entre {lo:.3f} y {hi:.3f}: ajuste correcto.")
        elif hi < 0.8:
            print(f"   -> chi2_red {lo:.3f}-{hi:.3f}, por DEBAJO de 1: barras "
                  f"de error probablemente infladas (tipico de CC solo).")
        else:
            print(f"   -> chi2_red {lo:.3f}-{hi:.3f}: revisa el ajuste.")

    # ── 3. que se puede citar ────────────────────────────────────────────
    faltan = [c for c in COLUMNAS_NUEVAS if c not in cols]
    print("\n3. QUE ES CITABLE DE ESTA CAMPANA")
    if not faltan:
        print("   Tiene las columnas de procedencia: campana corrida con el")
        print("   codigo corregido. Todo citable, con las advertencias de")
        print("   lectura de siempre (no comparar contra `none` sin")
        print("   `none-counts`; el ESS no es metrica de calidad en el eje de")
        print("   ruido).")
    else:
        print(f"   Faltan columnas: {', '.join(faltan)}")
        print("   -> Campana corrida con codigo ANTERIOR al 2026-09-04.")
        print()
        print("   NO citable:")
        print("     * KL y ESS del QVMC  — [B-BUDGET] la rama cuantica recibio")
        print("       entre 57x y 225x mas evaluaciones de circuito que la")
        print("       clasica con el mismo --qvmc-iter, y [B-ESSCOMP] el ESS")
        print("       estaba comprimido.")
        print("     * chi2/AIC/BIC del genetico como comparacion entre rungs —")
        print("       [B-REFINE] el refinador lleva a los cuatro al mismo")
        print("       minimo; sin la columna chi2_grid no hay con que")
        print("       distinguirlos.")
        print()
        print("   SI citable:")
        print("     * Todo el QMCMC (medias, sigmas, aceptacion, R-hat). Ninguno")
        print("       de esos bugs lo toca.")
        print("     * Los chi2/AIC/BIC como calidad de ajuste de cada MODELO")
        print("       (que es para lo que sirven).")

    # ── 4. celdas faithful ───────────────────────────────────────────────
    # La clave de agrupacion NO incluye nqpp a proposito. Los rungs clasicos
    # ('Classical VI', 'CGA') escriben nqpp='—' mientras sus parejas cuanticas
    # escriben el numero, asi que agrupar por nqpp separaba a cada pareja y el
    # par CGA == QGA (q=0%) no llegaba a comprobarse nunca. Se usa el archivo
    # de origen en su lugar: dentro de un CSV de tarea el nqpp es unico.
    grupos = defaultdict(dict)
    for r in filas:
        clave = (r.get('_archivo'), r.get('_modelo'), r.get('dataset'),
                 r.get('_ruido'))
        grupos[clave][r.get('Method')] = r

    campos = columnas_de_parametros(cols)
    if not campos:
        print("\n4. CELDAS FAITHFUL: el CSV no trae ninguna columna *_mean /")
        print("   *_std, asi que NO se puede comprobar nada. Revisa el CSV.")
        return

    # Solo pares REALMENTE faithful. 'Classical VI' vs 'QVMC 33%' NO lo es y
    # estaba mal puesto aqui: el rung 33% cambia el muestreo (disparos en vez
    # de amplitudes), asi que difiere en la cuarta cifra por construccion.
    # Exigirle igualdad habria producido un fallo falso en cada celda.
    # El tercer campo dice si el par DEBE separarse al encender el ruido.
    # Solo es cierto cuando el componente que los distingue es un circuito
    # cuantico que el ruido toca:
    #
    #   QMCMC 50 -> 100 : anade la aceptacion cuantica, un circuito real.
    #                     El ruido lo degrada. DEBE separarse.
    #   QVMC  67 -> 100 : anade la normalizacion, que ejecuta el circuito
    #                     pero DESCARTA su resultado y usa la suma exacta
    #                     (ver quantum_amplitude_normalization). Es inmune
    #                     al ruido por construccion: seguir identico es lo
    #                     correcto, no un fallo.
    #   CGA  -> QGA 0%  : quantumness 0%, no hay componente cuantico que el
    #                     ruido pueda tocar.
    #
    # Sin esta distincion el triage denunciaba 'el ruido no llego' en 70 de
    # 92 celdas sanas.
    pares = [('QMCMC 50%', 'QMCMC 100%', True),
             ('QVMC 67%', 'QVMC 100%', False),
             ('CGA', 'QGA (q=0%)', False)]
    # La igualdad faithful SOLO se exige en el eje ideal. Con ruido, los dos
    # peldanos tienen que separarse: es el resultado del eje NISQ, no un
    # error. Mezclar ambos ejes en un solo contador daba '40/62 REVISAR' en
    # una campana perfectamente sana — y una alarma que salta siempre es una
    # alarma que se deja de leer. Al reves tambien informa: un peldano
    # ruidoso que coincide BIT A BIT con el ideal significa que el ruido no
    # se aplico, y eso si es un error.
    resumen = defaultdict(lambda: [0, 0])       # ideal: cuantos coinciden
    ruidosas = defaultdict(lambda: [0, 0])      # ruido: cuantos se separan
    inmunes = defaultdict(lambda: [0, 0])       # ruido: cuantos siguen iguales
    for (_arch, _mod, _ds, ruido), g in grupos.items():
        ideal = (ruido or 'none') in ('none', 'none-counts', '')
        for a, b, sensible in pares:
            if a not in g or b not in g:
                continue
            igual = _iguales(g[a], g[b], campos)
            if ideal:
                resumen[(a, b)][1] += 1
                resumen[(a, b)][0] += int(igual)
            elif sensible:
                ruidosas[(a, b)][1] += 1
                ruidosas[(a, b)][0] += int(not igual)
            else:
                inmunes[(a, b)][1] += 1
                inmunes[(a, b)][0] += int(igual)

    if resumen or ruidosas or inmunes:
        print("\n4. CELDAS FAITHFUL")
        print(f"   comparando: {', '.join(campos)}")
    if resumen:
        print("\n   Sin ruido — DEBEN coincidir exactamente:")
        for (a, b), (ok, tot) in sorted(resumen.items()):
            marca = 'OK' if ok == tot else '*** REVISAR ***'
            print(f"     {a:14s} == {b:14s}  {ok}/{tot}  {marca}")
    if ruidosas:
        print("\n   Con ruido — DEBEN separarse (que difieran es el resultado):")
        for (a, b), (sep, tot) in sorted(ruidosas.items()):
            marca = 'OK' if sep == tot else '*** REVISAR: el ruido no llego ***'
            print(f"     {a:14s} != {b:14s}  {sep}/{tot}  {marca}")
    if inmunes:
        print("\n   Con ruido, pares INMUNES por construccion "
              "(deben seguir iguales):")
        for (a, b), (ok, tot) in sorted(inmunes.items()):
            marca = 'OK' if ok == tot else '*** REVISAR ***'
            print(f"     {a:14s} == {b:14s}  {ok}/{tot}  {marca}")


def main(argv=None):
    """Punto de entrada.

    Args:
        argv: rutas de carpetas de campana; None usa `sys.argv`.

    Returns:
        Codigo de salida (0 siempre, salvo que no se pase ninguna ruta).
    """
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        print(__doc__)
        return 2
    for patron in args:
        for d in sorted(glob.glob(patron)) or [patron]:
            if os.path.isdir(d):
                revisar(d)
            else:
                print(f"no es una carpeta: {d}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
