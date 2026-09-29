#!/usr/bin/env python3
"""
comparar_semillas.py — ¿Que sobrevive al cambio de semilla?

Una campana con una sola semilla no puede decir si una diferencia es real o
es suerte del generador aleatorio. Este script toma varias corridas identicas
que solo difieren en `--seed` y responde tres cosas:

    1. ¿Cuanto se mueve cada numero al cambiar de semilla? (la dispersion)
    2. ¿Las diferencias que quiero reportar son MAYORES que esa dispersion?
    3. ¿Las celdas faithful siguen coincidiendo en CADA semilla?

La tercera es la mas importante y la que nadie suele comprobar: una celda
faithful debe dar identico dentro de una misma semilla. Si coincide en la
semilla 1 pero no en la 3, no es una fluctuacion — es un bug que la semilla
42 escondia.

La unidad de medida de todo el reporte es la propia dispersion entre
semillas. Decir "el KL clasico es 1.6250 y el cuantico 1.6286" no significa
nada por si solo; decir "difieren en 0.4 desviaciones de semilla" si.

Uso:

    python comparar_semillas.py results/semilla_*
    python comparar_semillas.py corrida_a corrida_b corrida_c

Cada argumento es la carpeta maestra de UNA semilla. La semilla se lee de la
columna `seed` del CSV, no del nombre de la carpeta.

Solo biblioteca estandar: corre en el nodo de la HPC sin activar el entorno.
"""

from __future__ import annotations

import csv
import glob

import campaign_io
import math
import os
import re
import sys
from collections import defaultdict

#: Pares que DEBEN coincidir exactamente dentro de una misma semilla, con la
#: razon por la que se exige. Son los mismos de `triage_campana.py`; si
#: cambias uno alla, cambialo aqui.
PARES_FAITHFUL = [
    ('QMCMC 50%', 'QMCMC 100%', 'la aceptacion cuantica reproduce Metropolis'),
    ('QVMC 67%', 'QVMC 100%', 'la normalizacion devuelve la suma exacta'),
    ('CGA', 'QGA (q=0%)', 'quantumness 0% son los operadores clasicos'),
]

#: Comparaciones cuyo interes es justamente que la diferencia NO sea cero, y
#: que solo se pueden defender si superan la dispersion entre semillas.
COMPARACIONES = [
    ('Classical VI', 'QVMC 100%', 'final_KL',
     'clasico contra cuantico a presupuesto igualado'),
    ('Classical MCMC', 'QMCMC 100%', 'acceptance',
     'aceptacion clasica contra cuantica'),
]


def columnas_de_parametros(cols):
    """Columnas de parametros cosmologicos presentes en el CSV.

    Los nombres dependen del esquema: `Om_mean` en samplers, `p1_mean` en el
    barrido genetico. Se descubren por el sufijo en vez de fijarse.

    Args:
        cols: conjunto de nombres de columna.

    Returns:
        Lista ordenada de las columnas que acaban en `_mean`.

    Examples:
        >>> columnas_de_parametros({'Method', 'Om_mean', 'Om_std', 'chi2'})
        ['Om_mean']
        >>> columnas_de_parametros({'Method', 'chi2'})
        []
    """
    return sorted(c for c in cols if c.endswith('_mean'))


def nivel_de_ruido(fila, ruta):
    """Nivel de ruido de una fila: columna si existe, si no la carpeta.

    Las campanas anteriores al 2026-09-04 no traen columna `noise`, pero el
    runner si pone el nivel en el nombre de la carpeta.

    Examples:
        >>> nivel_de_ruido({'noise': 'readout'}, 'x/noise-full/r.csv')
        'readout'
        >>> nivel_de_ruido({}, 'x/samplers_lcdm_noise-full/r.csv')
        'full'
        >>> nivel_de_ruido({'noise': ''}, 'x/genetic_lcdm_nb6/r.csv')
        'none'
    """
    if fila.get('noise'):
        return fila['noise']
    m = re.search(r'noise-([A-Za-z0-9_.-]+)', ruta.replace('\\', '/'))
    return m.group(1) if m else 'none'


def modelo_de_ruta(path, conocidos=('lcdm', 'pede', 'wcdm', 'gede', 'cpl')):
    """Deduce el modelo de la ruta cuando el CSV no trae columna `model`.

    Examples:
        >>> modelo_de_ruta('run/model_wcdm/resultados_config.csv')
        'wcdm'
        >>> modelo_de_ruta('run/resultados_config.csv')
        '?'
    """
    partes = path.replace('\\', '/').split('/')
    for p in partes:
        m = re.match(r'model_(\w+)$', p)
        if m and m.group(1) in conocidos:
            return m.group(1)
    for p in partes:
        m = re.search(r'_(' + '|'.join(conocidos) + r')_', p)
        if m:
            return m.group(1)
    return '?'


def _num(x):
    """Convierte a float, o None si no se puede."""
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def cargar(carpetas):
    """Lee todas las corridas y las indexa por celda y semilla.

    Args:
        carpetas: rutas de las carpetas maestras, una por semilla.

    Returns:
        `(celdas, semillas, columnas)`.

        `celdas` mapea `(tarea, modelo, dataset, ruido, metodo)` a un dict
        `semilla -> fila`. `semillas` es el conjunto de semillas vistas y
        `columnas` la union de nombres de columna.

    La identidad de la celda es la ruta del CSV **relativa a la carpeta de su
    semilla**, no el `nqpp`. Dos razones:

      * los rungs clasicos escriben `nqpp = '—'` mientras sus parejas
        cuanticas escriben el numero, asi que agrupar por nqpp separaba a
        cada pareja y las comparaciones clasico-contra-cuantico no llegaban
        a hacerse NUNCA — no fallaban, simplemente no aparecian;
      * la ruta relativa es identica entre semillas
        (`model_lcdm/resultados_config.csv`) y distinta entre tareas, que es
        exactamente lo que se necesita.
    """
    celdas = defaultdict(dict)
    semillas, columnas = set(), set()
    for carpeta in carpetas:
        raiz = os.path.abspath(carpeta)
        # [E-QPU6] Only the per-model CSV: the per-task cumulative file
        # repeats the same rows and used to double every cell.
        for p in campaign_io.result_csvs_any_layout(carpeta):
            try:
                filas = list(csv.DictReader(open(p, newline='')))
            except OSError:
                continue
            if not filas:
                continue
            columnas |= set(filas[0].keys())
            mod_ruta = modelo_de_ruta(p)
            tarea = os.path.relpath(os.path.abspath(p), raiz).replace('\\', '/')
            for r in filas:
                s = r.get('seed')
                if s in (None, ''):
                    continue
                semillas.add(s)
                clave = (tarea, r.get('model') or mod_ruta,
                         r.get('dataset', ''), nivel_de_ruido(r, p),
                         (r.get('Method') or '').strip())
                celdas[clave][s] = r
    return celdas, semillas, columnas


def dispersion(valores):
    """Media y desviacion muestral de una lista de numeros.

    Returns:
        `(media, desviacion, n)`. La desviacion es 0.0 con un solo valor.

    Examples:
        >>> m, s, n = dispersion([1.0, 2.0, 3.0])
        >>> round(m, 3), round(s, 3), n
        (2.0, 1.0, 3)
        >>> dispersion([5.0])
        (5.0, 0.0, 1)
        >>> dispersion([])
        (None, None, 0)
    """
    v = [x for x in valores if x is not None]
    if not v:
        return None, None, 0
    m = sum(v) / len(v)
    if len(v) == 1:
        return m, 0.0, 1
    var = sum((x - m) ** 2 for x in v) / (len(v) - 1)
    return m, math.sqrt(var), len(v)


def en_sigmas(a_vals, b_vals):
    """Paired comparison of two methods run with the same seeds.

    [E-QPU7] Both methods share every seed, so the right test is a paired
    t-test on the per-seed differences d_s = a_s - b_s. The previous version
    divided the difference of means by sqrt(sa^2 + sb^2), the per-seed
    spreads, which ignores the pairing and is sqrt(n) too conservative (a
    systematic offset at t = 20.8 was reported as 0.54 "sigmas").

    Args:
        a_vals, b_vals: values of each method, one per seed, same order.

    Returns:
        `(mean_diff, standard_error, t_statistic)`, or `(None, None, None)`
        with fewer than two usable pairs. `t_statistic` is None when every
        difference is identical (zero standard error).

    Examples:
        >>> d, se, t = en_sigmas([1.0, 1.1, 0.9], [2.0, 2.1, 1.9])
        >>> round(d, 3), t is None
        (-1.0, True)
        >>> d, se, t = en_sigmas([1.00, 1.12, 0.93], [0.99, 1.10, 0.92])
        >>> round(t, 2)
        4.0
        >>> en_sigmas([1.0], [2.0])[2] is None
        True
    """
    pares = [(a, b) for a, b in zip(a_vals, b_vals)
             if a is not None and b is not None]
    if len(pares) < 2:
        return None, None, None
    d = [a - b for a, b in pares]
    m, sd, n = dispersion(d)
    se = sd / math.sqrt(n)
    return m, se, (m / se if se > 1e-15 * max(1.0, abs(m)) else None)


def p_value_two_sided(t, n):
    """Two-sided p-value of a paired t statistic with n pairs (n-1 dof)."""
    from scipy import stats
    return float(2 * stats.t.sf(abs(t), n - 1))


def revisar(carpetas):
    """Imprime el reporte completo de comparacion entre semillas."""
    celdas, semillas, columnas = cargar(carpetas)
    print('=' * 74)
    print('COMPARACION ENTRE SEMILLAS')
    print('=' * 74)
    if not celdas:
        print('\nNo encontre ningun CSV con columna `seed` en esas carpetas.')
        print('Las campanas anteriores al 2026-09-04 no la traen: para el')
        print('estudio de semillas necesitas corridas del codigo corregido.')
        return 2

    orden = sorted(semillas, key=lambda s: (_num(s) is None, _num(s), s))
    print(f'\nSemillas encontradas: {len(orden)}  ->  {", ".join(orden)}')
    if len(orden) < 2:
        print('\nCon UNA sola semilla no hay dispersion que medir. Este')
        print('reporte necesita al menos dos corridas que solo difieran en')
        print('--seed; con tres o mas las barras de error son creibles.')

    campos = columnas_de_parametros(columnas)
    metricas = [c for c in ('final_KL', 'acceptance', 'ESS', 'chi2_grid')
                if c in columnas] + campos

    # ── 1. celdas faithful, semilla por semilla ──────────────────────────
    print('\n' + '-' * 74)
    print('1. CELDAS FAITHFUL — deben coincidir DENTRO de cada semilla')
    print('-' * 74)
    print('   Una celda faithful que coincide con una semilla y no con otra')
    print('   no es azar: es un error que esa semilla tapaba.')
    hubo = False
    for a, b, porque in PARES_FAITHFUL:
        ok = fallo = 0
        culpables = []
        for clave, por_semilla in celdas.items():
            if clave[4] != a:
                continue
            clave_b = clave[:4] + (b,)
            if clave_b not in celdas:
                continue
            # Solo tiene sentido exigir igualdad en el eje ideal.
            if clave[3] not in ('none', 'none-counts', ''):
                continue
            for s, fila_a in por_semilla.items():
                fila_b = celdas[clave_b].get(s)
                if fila_b is None:
                    continue
                comunes = [c for c in campos if c in fila_a and c in fila_b]
                if not comunes:
                    continue
                hubo = True
                if all(fila_a[c] == fila_b[c] for c in comunes):
                    ok += 1
                else:
                    fallo += 1
                    culpables.append(f'{clave[1]}/{clave[0]}/semilla={s}')
        if ok or fallo:
            marca = 'OK' if fallo == 0 else '*** REVISAR ***'
            print(f'\n   {a} == {b}   {ok}/{ok+fallo}   {marca}')
            print(f'      ({porque})')
            for c in culpables[:5]:
                print(f'      difiere en: {c}')
    if not hubo:
        print('\n   (no hay pares faithful comparables en estos datos)')

    # ── 2. cuanto se mueve cada numero ───────────────────────────────────
    print('\n' + '-' * 74)
    print('2. DISPERSION ENTRE SEMILLAS — cuanto se mueve cada numero')
    print('-' * 74)
    filas_disp = []
    for clave, por_semilla in sorted(celdas.items()):
        if len(por_semilla) < 2:
            continue
        for met in metricas:
            vals = [_num(f.get(met)) for f in por_semilla.values()]
            m, s, n = dispersion(vals)
            if m is None or n < 2:
                continue
            rel = abs(s / m) * 100 if m else float('nan')
            filas_disp.append((rel, clave, met, m, s, n))
    if not filas_disp:
        print('\n   (ninguna celda tiene dos o mas semillas)')
    else:
        filas_disp.sort(reverse=True)
        print(f'\n   Las 12 mas inestables (dispersion relativa mayor):\n')
        print(f'   {"celda":34s}{"metrica":12s}{"media":>12s}'
              f'{"sigma":>11s}{"%":>7s}')
        for rel, clave, met, m, s, n in filas_disp[:12]:
            et = f'{clave[1]}/{clave[3]}/{clave[4]}'[:33]
            print(f'   {et:34s}{met:12s}{m:12.5g}{s:11.4g}{rel:7.2f}')
        estables = sum(1 for f in filas_disp if f[0] < 1.0)
        print(f'\n   {estables} de {len(filas_disp)} combinaciones se mueven '
              f'menos del 1% entre semillas.')

    # ── 3. lo que se quiere reportar, medido en sigmas ───────────────────
    print('\n' + '-' * 74)
    print('3. ¿SOBREVIVEN LAS DIFERENCIAS QUE QUIERES REPORTAR?')
    print('-' * 74)
    print('   Paired t-test over seeds (both methods share every seed).')
    print('   A difference with p >= 0.05 cannot be defended.')
    algo = False
    for a, b, met, porque in COMPARACIONES:
        if met not in columnas:
            continue
        for clave, por_semilla in sorted(celdas.items()):
            if clave[4] != a:
                continue
            clave_b = clave[:4] + (b,)
            if clave_b not in celdas:
                continue
            sa = set(por_semilla) & set(celdas[clave_b])
            if len(sa) < 2:
                continue
            va = [_num(por_semilla[s].get(met)) for s in sorted(sa)]
            vb = [_num(celdas[clave_b][s].get(met)) for s in sorted(sa)]
            dif, sc, ns = en_sigmas(va, vb)
            if dif is None:
                continue
            algo = True
            et = f'{clave[1]}/{clave[3]}  [{clave[0]}]'
            # [E-QPU7] Verdict from the paired t-test p-value.
            if ns is None:
                juicio = ('identical difference on every seed: systematic'
                          if dif else 'no difference on any seed')
            else:
                pv = p_value_two_sided(ns, len(sa))
                if pv >= 0.05:
                    juicio = f'p = {pv:.3g}: within seed-to-seed noise, not reportable'
                elif pv >= 0.001:
                    juicio = f'p = {pv:.3g}: marginal, report with its error bar'
                else:
                    juicio = f'p = {pv:.3g}: solid'
            print(f'\n   {a} vs {b}  [{met}]  — {porque}')
            print(f'      {et}   ({len(sa)} seeds, paired)')
            print(f'      mean difference = {dif:+.6g}   SE = {sc:.3g}'
                  + (f'   ->  t = {ns:+.2f}' if ns is not None else ''))
            print(f'      {juicio}')
    if not algo:
        print('\n   (no hay pares comparables con dos o mas semillas)')

    print('\n' + '=' * 74)
    return 0


def main(argv=None):
    """Punto de entrada.

    Args:
        argv: carpetas maestras, una por semilla; None usa `sys.argv`.

    Returns:
        0 si el reporte se genero, 2 si no habia datos utilizables.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        print(__doc__)
        return 2
    carpetas = []
    for patron in args:
        carpetas += sorted(glob.glob(patron)) or [patron]
    carpetas = [c for c in carpetas if os.path.isdir(c)]
    if not carpetas:
        print('Ninguna de esas rutas es una carpeta.')
        return 2
    return revisar(carpetas)


if __name__ == '__main__':
    sys.exit(main())
