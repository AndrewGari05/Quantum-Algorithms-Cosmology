#!/usr/bin/env python3
"""comparar_algoritmos.py — con que numero se compara un metodo contra otro.

Lee resultados ya obtenidos; NO corre nada. Contesta la pregunta
"¿el cuantico gana o pierde?" con el estimador que corresponde a cada
familia, y NO con el mismo para todas, porque no miden lo mismo:

  * SAMPLERS (MCMC / QMCMC) -> ESS, tiempo, y ESS por segundo.
    El ESS (tamano de muestra efectivo) dice cuantas muestras
    INDEPENDIENTES equivalen a la cadena. Es la moneda estandar para
    comparar cadenas: una cadena con 10000 pasos muy correlacionados vale
    menos que una de 2000 pasos independientes.

  * VARIACIONALES (VI / QVMC) -> `final_KL` a presupuesto de circuitos
    igualado. [B-ESSCOMP] El ESS de esta familia es el numero de disparos
    (constante, 12288 en la campana) y NO mide calidad: usarlo como si la
    midiera es un error de lectura.

  * GENETICOS (CGA / QGA) -> `chi2_grid`, el chi2 del mejor punto de la
    REJILLA sin refinar. [B-REFINE] El chi2 refinado es identico en los
    cuatro peldanos y no distingue nada.

────────────────────────────────────────────────────────────────────────────
LA TRAMPA DE LOS SEGUNDOS
────────────────────────────────────────────────────────────────────────────
El cociente ESS/s castiga al cuantico por el costo del SIMULADOR, que no
tiene nada que ver con lo que costaria en hardware. Por eso el script
reporta las dos cosas por separado:

  * razon de ESS  -> independiente del hardware (a pasos igualados, cuantas
    muestras efectivas produjo cada uno). ES el numero que va al paper.
  * razon de ESS/s -> util para planear campanas, no para afirmar nada
    sobre computo cuantico.

Uso:
    python comparar_algoritmos.py CARPETA_CAMPANA [...] [--salida FIGS]
"""
from __future__ import annotations

import argparse
import csv
import glob

import campaign_io
import math
import os
import re
import statistics
import sys
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt          # noqa: E402

# Semantica heredada del documento de circuitos: indigo = cuantico,
# ocre = clasico. Aqui codifican POLARIDAD alrededor de 1 (quien gana),
# con gris para el empate — es una escala divergente, no categorica.
C_GANA_Q = '#5347C4'
C_GANA_C = '#A8651F'
C_EMPATE = '#8A8A94'
C_REF = '#3A3A44'

PAREJAS_SAMPLERS = [
    ('Classical MCMC', 'QMCMC 100%'),
    ('Classical MCMC', 'QMCMC 50%'),
]


def _guardar(fig, ruta: str) -> str:
    """[B-PDF] Guarda la figura en PNG y tambien en PDF.

    El PNG es para mirarla en pantalla; el PDF es vectorial y es el que
    pide el paper (un PNG a 150 dpi se ve pixelado impreso, y varias
    revistas lo rechazan). Los dos salen del MISMO objeto `fig`, asi que
    no pueden discrepar: no hay forma de que el PDF muestre una version
    vieja de lo que muestra el PNG.

    Args:
        fig: la figura de matplotlib, ya terminada.
        ruta: ruta del PNG. El PDF se escribe al lado, con la misma raiz.

    Returns:
        La ruta del PNG (lo que el resto del codigo ya esperaba).
    """
    fig.savefig(ruta, dpi=150)
    fig.savefig(os.path.splitext(ruta)[0] + '.pdf')
    return ruta


def _num(x):
    try:
        v = float(str(x).strip())
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def leer(raices) -> list:
    """All rows of every campaign, with model/grid/noise level resolved.

    [E-READ] Delegates to `campaign_io.read_campaign`, which accepts task
    folders with or without the grid and noise tags (HPC-8/QPU-10) and never
    reads the cumulative CSV twice (QPU-6).
    """
    return campaign_io.read_campaigns(raices)


def razones_samplers(filas, peldano='none') -> dict:
    """Razones cuantico/clasico por MODELO, pareadas celda a celda.

    Se agrupa por modelo y no por celda porque la propuesta cuantica usa
    d qubits (d = parametros del modelo) y es independiente de `nqpp`: las
    celdas que solo difieren en nqpp repiten exactamente la misma
    comparacion. Contarlas como observaciones independientes inflaria el
    tamano de muestra de 5 a 87.
    """
    celdas = defaultdict(dict)
    for r in filas:
        if r['_fam'] != 'samplers' or r['_noi'] != peldano:
            continue
        celdas[campaign_io.cell_key(r)][r['Method']] = r

    por_modelo = defaultdict(lambda: defaultdict(list))
    for key, v in celdas.items():
        mod, g = key[5], key[6]
        for clasico, cuantico in PAREJAS_SAMPLERS:
            c, q = v.get(clasico), v.get(cuantico)
            if not (c and q):
                continue
            ec, eq = _num(c['ESS']), _num(q['ESS'])
            tc, tq = _num(c['Time_s']), _num(q['Time_s'])
            if None in (ec, eq, tc, tq) or 0 in (ec, tc, tq):
                continue
            d = por_modelo[(clasico, cuantico)][mod]
            d.append({'ess': eq / ec, 'tiempo': tq / tc,
                      'eff': (eq / tq) / (ec / tc),
                      'acc_c': _num(c['acceptance']),
                      'acc_q': _num(q['acceptance'])})
    return por_modelo


def _med(vals):
    return statistics.median(vals) if vals else float('nan')


def tabla(por_modelo) -> None:
    for (clasico, cuantico), por_mod in por_modelo.items():
        print(f"\n{'='*72}\n  {cuantico}  contra  {clasico}"
              f"   (peldano ideal)\n{'='*72}")
        print(f"  {'modelo':8s} {'celdas':>7s} {'ESS q/c':>9s} "
              f"{'tiempo q/c':>11s} {'ESS/s q/c':>10s} "
              f"{'acept c':>8s} {'acept q':>8s}")
        todos_ess, todos_eff = [], []
        for mod in sorted(por_mod):
            v = por_mod[mod]
            e = _med([x['ess'] for x in v])
            t = _med([x['tiempo'] for x in v])
            f = _med([x['eff'] for x in v])
            ac = _med([x['acc_c'] for x in v if x['acc_c'] is not None])
            aq = _med([x['acc_q'] for x in v if x['acc_q'] is not None])
            todos_ess.append(e)
            todos_eff.append(f)
            print(f"  {mod:8s} {len(v):7d} {e:9.4f} {t:11.2f} {f:10.4f} "
                  f"{ac:8.4f} {aq:8.4f}")
        if todos_ess:
            gana = sum(1 for x in todos_ess if x > 1)
            print(f"\n  ESS:  mediana sobre modelos = {_med(todos_ess):.4f}   "
                  f"({gana}/{len(todos_ess)} modelos a favor del cuantico)")
            print(f"  ESS/s: mediana = {_med(todos_eff):.4f}  -> el clasico "
                  f"es {1/_med(todos_eff):.1f}x mas eficiente en el simulador")
            print("\n  Lectura: la razon de ESS es la comparacion honesta "
                  "(independiente\n  del hardware). La de ESS/s mide el costo "
                  "de Aer, no el del computo\n  cuantico, y no sostiene "
                  "ninguna afirmacion sobre hardware real.")


def figura(por_modelo, salida) -> list:
    """Una figura por pareja: razon de ESS y razon de eficiencia."""
    os.makedirs(salida, exist_ok=True)
    hechas = []
    for (clasico, cuantico), por_mod in por_modelo.items():
        mods = sorted(por_mod)
        if not mods:
            continue
        ess = [_med([x['ess'] for x in por_mod[m]]) for m in mods]
        eff = [_med([x['eff'] for x in por_mod[m]]) for m in mods]

        fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.6))
        y = range(len(mods))

        # (a) calidad por paso — escala lineal, el rango es estrecho
        ax = axes[0]
        for i, v in zip(y, ess):
            c = C_GANA_Q if v > 1.01 else (C_GANA_C if v < 0.99 else C_EMPATE)
            ax.barh(i, v - 1.0, left=1.0, height=.55, color=c)
        ax.axvline(1.0, color=C_REF, lw=1.1)
        ax.set_yticks(list(y))
        ax.set_yticklabels([m.upper() for m in mods], fontsize=9)
        ax.set_xlabel('ESS cuantico / ESS clasico', fontsize=9)
        ax.set_title('(a) calidad de muestreo, a pasos igualados',
                     fontsize=10, loc='left')
        lo, hi = min(ess), max(ess)
        pad = max(0.03, (hi - lo) * 0.45)
        ax.set_xlim(min(lo, 1.0) - pad, max(hi, 1.0) + pad)
        ax.grid(axis='x', alpha=.25, lw=.5)
        ax.tick_params(labelsize=8)
        for i, v in zip(y, ess):
            ax.text(v + (0.012 if v >= 1 else -0.012), i, f'{v:.3f}',
                    va='center', ha='left' if v >= 1 else 'right',
                    fontsize=8, color=C_REF)

        # (b) eficiencia de pared — log, el rango es de un orden
        ax = axes[1]
        for i, v in zip(y, eff):
            c = C_GANA_Q if v > 1 else C_GANA_C
            ax.barh(i, v, height=.55, color=c)
        ax.axvline(1.0, color=C_REF, lw=1.1)
        ax.set_xscale('log')
        ax.set_yticks(list(y))
        ax.set_yticklabels([])
        ax.set_xlabel('(ESS/s) cuantico / (ESS/s) clasico   [escala log]',
                      fontsize=9)
        ax.set_title('(b) eficiencia en el SIMULADOR', fontsize=10, loc='left')
        ax.grid(axis='x', alpha=.25, lw=.5)
        ax.tick_params(labelsize=8)
        for i, v in zip(y, eff):
            ax.text(v * 1.08, i, f'{v:.3f}', va='center', ha='left',
                    fontsize=8, color=C_REF)

        fig.suptitle(f'{cuantico} contra {clasico} — peldano ideal',
                     fontsize=11)
        # Leyenda por polaridad: identidad no queda solo en el color.
        from matplotlib.patches import Patch
        fig.legend(handles=[Patch(color=C_GANA_Q, label='gana el cuantico'),
                            Patch(color=C_GANA_C, label='gana el clasico'),
                            Patch(color=C_EMPATE, label='empate (±1%)')],
                   loc='lower center', ncol=3, fontsize=8, frameon=False)
        fig.tight_layout(rect=(0, .09, 1, .95))
        nombre = ('eficiencia_' +
                  cuantico.replace(' ', '').replace('%', '') + '.png')
        ruta = os.path.join(salida, nombre)
        _guardar(fig, ruta)
        plt.close(fig)
        hechas.append(ruta)
    return hechas


# ═════════════════════════════════════════════════════════════════════
# INCERTIDUMBRE REPORTADA
# ═════════════════════════════════════════════════════════════════════
#
# "¿Cual tiene menor incertidumbre?" es una pregunta trampa, y conviene
# tenerlo claro antes de leer la tabla: en inferencia bayesiana una sigma
# mas chica NO es mejor. La sigma correcta es la de la posterior. Un
# metodo que reporta menos incertidumbre de la que hay esta SESGADO, no
# es mas preciso.
#
# La referencia de oro es el MCMC clasico: con suficientes pasos y R-hat
# convergido, sus sigmas son las de la posterior. Todo lo demas se mide
# CONTRA esa, y lo que se busca es una razon cercana a 1 — no menor.
#
# Se espera que los variacionales salgan por debajo: minimizar
# KL(Q || P) es mode-seeking (zero-forcing), asi que Q sale mas angosta
# que P. Es un resultado de libro de texto de inferencia variacional, y
# aqui queda cuantificado sobre los datos del proyecto.

REFERENCIA = 'Classical MCMC'


def razones_sigma(filas, peldano='none') -> dict:
    """sigma de cada metodo / sigma del MCMC clasico, por parametro.

    Returns:
        dict[metodo][parametro] -> lista de razones (una por celda)
    """
    celdas = defaultdict(dict)
    for r in filas:
        if r['_fam'] == 'samplers' and r['_noi'] == peldano:
            celdas[campaign_io.cell_key(r)][r['Method']] = r

    rel = defaultdict(lambda: defaultdict(list))
    for v in celdas.values():
        ref = v.get(REFERENCIA)
        if not ref:
            continue
        pars = [c[:-4] for c in ref if c.endswith('_std')]
        for meth, r in v.items():
            if meth == REFERENCIA:
                continue
            for p in pars:
                a, b = _num(r.get(p + '_std')), _num(ref.get(p + '_std'))
                if a is not None and b:
                    rel[meth][p].append(a / b)
    return rel


ORDEN_METODOS = ['QMCMC 50%', 'QMCMC 100%', 'Classical VI',
                 'QVMC 33%', 'QVMC 67%', 'QVMC 100%']
ORDEN_PARS = ['Om', 'H0', 'w', 'w0', 'wa', 'Delta']


def tabla_sigma(rel) -> None:
    print(f"\n{'='*72}\n  INCERTIDUMBRE REPORTADA / la del {REFERENCIA}"
          f"\n{'='*72}")
    print("  Cerca de 1 = de acuerdo con la referencia.")
    print("  Por DEBAJO de 1 = reporta menos incertidumbre de la que hay.")
    print("  NO es mejor: es sesgo.\n")
    pars = [p for p in ORDEN_PARS if any(p in rel[m] for m in rel)]
    print(f"  {'Metodo':16s}" + ''.join(f"{p:>9s}" for p in pars))
    for meth in ORDEN_METODOS:
        if meth not in rel:
            continue
        linea = f"  {meth:16s}"
        for p in pars:
            v = rel[meth].get(p)
            linea += f"{_med(v):9.3f}" if v else f"{'-':>9s}"
        print(linea)


def figura_sigma(rel, salida) -> str | None:
    """Razon de sigmas: un grupo por metodo, un punto por parametro."""
    metodos = [m for m in ORDEN_METODOS if m in rel]
    if not metodos:
        return None
    os.makedirs(salida, exist_ok=True)
    pars = [p for p in ORDEN_PARS if any(p in rel[m] for m in metodos)]

    fig, ax = plt.subplots(figsize=(8.4, 4.0))
    for i, meth in enumerate(metodos):
        vals = [_med(rel[meth][p]) for p in pars if p in rel[meth]]
        etiquetas = [p for p in pars if p in rel[meth]]
        # el punto grande es la mediana sobre parametros; los chicos, cada uno
        # Las etiquetas se alternan arriba/abajo: varios parametros caen
        # casi en el mismo valor y encimadas no se leen.
        for j, (v, p) in enumerate(zip(vals, etiquetas)):
            ax.scatter(v, i, s=26, color=C_GANA_C if v < 0.95 else C_EMPATE,
                       zorder=3, alpha=.85)
            if v < 0.95:
                dy = -12 if j % 2 == 0 else 8
                ax.annotate(p, (v, i), textcoords='offset points',
                            xytext=(0, dy), ha='center', fontsize=7,
                            color=C_REF)
        if vals:
            ax.scatter(statistics.median(vals), i, s=120, marker='|',
                       color=C_REF, zorder=4, lw=1.6)
    ax.axvline(1.0, color=C_REF, lw=1.2)
    ax.axvspan(0.0, 0.95, color=C_GANA_C, alpha=.06)
    ax.set_yticks(range(len(metodos)))
    ax.set_yticklabels(metodos, fontsize=9)
    ax.set_xlabel(f'sigma del metodo / sigma del {REFERENCIA}', fontsize=9)
    ax.set_xlim(0.45, 1.12)
    ax.set_ylim(-0.6, len(metodos) - 0.4)
    ax.grid(axis='x', alpha=.25, lw=.5)
    ax.tick_params(labelsize=8)
    ax.set_title('Incertidumbre reportada: por debajo de 1 es sub-dispersion,'
                 ' no precision', fontsize=10, loc='left')
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([], [], marker='o', ls='', color=C_EMPATE,
               label='de acuerdo con la referencia'),
        Line2D([], [], marker='o', ls='', color=C_GANA_C,
               label='sub-dispersion (<0.95)'),
        Line2D([], [], marker='|', ls='', color=C_REF, markersize=10,
               label='mediana sobre parametros')],
        fontsize=8, frameon=False, loc='upper center',
        bbox_to_anchor=(0.5, -0.16), ncol=3)
    fig.tight_layout(rect=(0, .06, 1, 1))
    ruta = os.path.join(salida, 'incertidumbre_reportada.png')
    _guardar(fig, ruta)
    plt.close(fig)
    return ruta


# ═════════════════════════════════════════════════════════════════════
# LA FAMILIA GENETICA
# ═════════════════════════════════════════════════════════════════════
#
# El genetico NO se compara con las mismas reglas que los samplers, por
# dos razones que hay que tener claras antes de leer nada:
#
#   1. Su numero de merito es `chi2_grid` — el chi2 del mejor punto de la
#      REJILLA, sin refinar. [B-REFINE] El `chi2` reportado es el del
#      refinador local continuo, que lleva a los cuatro peldanos al MISMO
#      minimo y por tanto no distingue nada.
#
#   2. **Su columna `*_std` NO es una incertidumbre.** Un algoritmo
#      genetico converge a un PUNTO: esa desviacion es la dispersion de la
#      poblacion final, y se encoge conforme la poblacion converge. Medido
#      contra el MCMC clasico da entre 0.07 y 0.18 — no porque el genetico
#      sea 6-14 veces mas preciso, sino porque no esta midiendo lo mismo.
#      Ponerla en el mismo eje que las sigmas de los samplers seria un
#      error de lectura grave, asi que aqui se reporta APARTE y con aviso.
#
# Como chi2_grid depende de la resolucion de la rejilla, la comparacion
# solo tiene sentido a n_bits FIJO: se parea celda a celda contra el CGA.
# Y se reporta la DIFERENCIA, no la razon, porque es un chi2: lo que
# significa algo es cuantas unidades de chi2 se pierden, no el porcentaje.

BASE_GENETICO = 'CGA'
ORDEN_QGA = ['QGA (q=0%)', 'QGA (q=33%)', 'QGA (q=67%)', 'QGA (q=100%)']

# Rampa ordinal de un solo tono para la escalera (validada con el script
# de la skill de dataviz: L monotona, saltos >= 0.06, extremo claro sobre
# el fondo). La escalera 0->33->67->100 es ORDINAL, no categorica.
RAMPA_ESCALERA = ['#8fb4dd', '#5b8ecb', '#2f66ac', '#17406f']
# Categoricos para las facetas (validados con --pairs all).
C_CAT = ['#2a78d6', '#d95926', '#199e70']


def razones_genetico(filas, peldano='none') -> dict:
    """Delta chi2_grid de cada peldano contra el CGA, por (modelo, n_bits).

    Returns:
        dict[metodo][(modelo, n_bits)] -> Delta chi2_grid (positivo = peor)
    """
    celdas = defaultdict(dict)
    for r in filas:
        if r['_fam'] == 'genetic' and r['_noi'] == peldano:
            celdas[campaign_io.cell_key(r)][r['Method']] = r

    out = defaultdict(dict)
    for key, v in celdas.items():
        mod, g = key[5], key[6]
        base = v.get(BASE_GENETICO)
        if not base:
            continue
        c0 = _num(base.get('chi2_grid'))
        if c0 is None:
            continue
        for meth in ORDEN_QGA:
            x = _num(v.get(meth, {}).get('chi2_grid'))
            if x is not None:
                out[meth][(mod, g)] = x - c0
    return out


def tabla_genetico(gen) -> None:
    if not gen:
        return
    print(f"\n{'='*72}\n  GENETICOS — Delta chi2_grid contra {BASE_GENETICO}"
          f"\n{'='*72}")
    print("  Negativo = el cuantico encontro MEJOR punto de rejilla.")
    print("  Referencia de relevancia estadistica: Delta chi2 = 1.\n")
    celdas = sorted({k for m in gen for k in gen[m]})
    print(f"  {'modelo':7s} {'nb':>3s} " +
          ''.join(f"{m.replace('QGA (q=', '').replace(')', ''):>13s}"
                  for m in ORDEN_QGA))
    for mod, g in celdas:
        linea = f"  {mod:7s} {g:3d} "
        for meth in ORDEN_QGA:
            d = gen[meth].get((mod, g))
            linea += f"{'-':>13s}" if d is None else f"{d:+13.4f}"
        print(linea)

    print()
    for meth in ORDEN_QGA:
        v = list(gen[meth].values())
        if not v:
            continue
        peor = sum(1 for x in v if x > 1e-6)
        mejor = sum(1 for x in v if x < -1e-6)
        rel = sum(1 for x in v if abs(x) >= 1.0)
        print(f"  {meth:14s} mediana {_med(v):+8.4f} | mejor que CGA "
              f"{mejor:3d}/{len(v)} | peor {peor:3d}/{len(v)} | "
              f"|Delta|>=1 en {rel}/{len(v)}")
    print("\n  Lectura: un signo consistente en TODAS las celdas es un efecto\n"
          "  sistematico real, aunque la magnitud sea pequena. Son dos cosas\n"
          "  distintas y las dos hay que reportarlas.")


def aviso_sigma_genetico(filas, peldano='none') -> None:
    """La sigma del genetico contra la del MCMC, con el aviso que necesita."""
    refs, gen = {}, defaultdict(lambda: defaultdict(list))
    for r in filas:
        if r['_noi'] != peldano:
            continue
        if r['_fam'] == 'samplers' and r['Method'] == REFERENCIA:
            refs[(r['_camp'], r['_mod'])] = r
    for r in filas:
        if r['_fam'] != 'genetic' or r['_noi'] != peldano:
            continue
        R = refs.get((r['_camp'], r['_mod']))
        if not R:
            continue
        for c in R:
            if c.endswith('_std'):
                a, b = _num(r.get(c)), _num(R.get(c))
                if a is not None and b:
                    gen[r['Method']][c[:-4]].append(a / b)
    if not gen:
        return
    print(f"\n{'='*72}\n  GENETICOS — dispersion de la poblacion final"
          f"\n{'='*72}")
    print("  AVISO: esto NO es una incertidumbre y no se compara con las")
    print("  sigmas de los samplers. Un genetico converge a un PUNTO; la")
    print("  columna *_std mide cuanto se encogio la poblacion, no cuanta")
    print("  incertidumbre tiene el parametro.\n")
    pars = [p for p in ORDEN_PARS if any(p in gen[m] for m in gen)]
    print(f"  {'Metodo':16s}" + ''.join(f"{p:>9s}" for p in pars))
    for meth in [BASE_GENETICO] + ORDEN_QGA:
        if meth not in gen:
            continue
        linea = f"  {meth:16s}"
        for p in pars:
            v = gen[meth].get(p)
            linea += f"{_med(v):9.3f}" if v else f"{'-':>9s}"
        print(linea)


def figura_genetico(gen, salida) -> str | None:
    """Delta chi2_grid contra n_bits, una faceta por modelo.

    Se factoriza por modelo en vez de poner los cinco en un eje porque
    cinco series categoricas no pasan la separacion para daltonismo; con
    facetas cada panel lleva tres series, que si pasa.
    """
    activos = [m for m in ORDEN_QGA[1:] if gen.get(m)]
    if not activos:
        return None
    os.makedirs(salida, exist_ok=True)
    modelos = sorted({mod for m in activos for (mod, _) in gen[m]})
    ncol = min(len(modelos), 5)
    fig, axes = plt.subplots(1, ncol, figsize=(2.5 * ncol + 1.2, 3.4),
                             squeeze=False, sharey=True)
    for j, mod in enumerate(modelos):
        ax = axes[0][j]
        for k, meth in enumerate(activos):
            pts = sorted((g, d) for (m2, g), d in gen[meth].items()
                         if m2 == mod)
            if not pts:
                continue
            xs, ys = zip(*pts)
            # El 67% y el 100% coinciden en casi toda celda (el cruce
            # cuantico rara vez cambia el mejor punto encontrado). Se
            # dibuja el 67% mas grueso y discontinuo para que se vea
            # POR DEBAJO del 100% en vez de desaparecer bajo el.
            ancho, trazo = (3.0, (0, (4, 2))) if '67' in meth else (1.6, '-')
            ax.plot(xs, ys, marker='o', ms=4, lw=ancho, ls=trazo,
                    color=C_CAT[k % len(C_CAT)], alpha=.95,
                    label=meth.replace('QGA (q=', '').replace(')', ''))
        ax.axhline(1.0, color=C_REF, lw=1.0, ls=':')
        ax.set_yscale('symlog', linthresh=1e-3)
        ax.set_title(mod.upper(), fontsize=10)
        ax.set_xlabel('n_bits', fontsize=8)
        ax.xaxis.set_major_locator(
            matplotlib.ticker.MaxNLocator(integer=True))
        ax.grid(alpha=.22, lw=.5)
        ax.tick_params(labelsize=8)
        if j == 0:
            ax.set_ylabel(r'$\Delta\chi^2$ de rejilla contra CGA', fontsize=9)
    manejas, etiquetas = axes[0][0].get_legend_handles_labels()
    fig.legend(manejas, etiquetas, loc='lower center', ncol=len(etiquetas),
               fontsize=8, frameon=False, title='quantumness',
               title_fontsize=8)
    fig.suptitle('Genéticos: el costo del peldaño cuántico se desvanece al '
                 'refinar la rejilla   (punteada: Δχ² = 1; 67% y 100% '
                 'coinciden salvo en CPL)', fontsize=10)
    fig.tight_layout(rect=(0, .11, 1, .93))
    ruta = os.path.join(salida, 'genetico_chi2grid.png')
    _guardar(fig, ruta)
    plt.close(fig)
    return ruta


# ═════════════════════════════════════════════════════════════════════
# COMPARACION DE MODELOS COSMOLOGICOS  (otra pregunta, otras herramientas)
# ═════════════════════════════════════════════════════════════════════
#
# Todo lo anterior compara ALGORITMOS. Esto compara MODELOS, que es una
# pregunta distinta y ortogonal: ¿los datos prefieren LCDM o CPL?
#
# No se contesta con chi2_red. Los cinco modelos dan chi2_red ~ 0.97, o
# sea todos ajustan aceptablemente; eso no discrimina. El chi2 SIEMPRE
# baja al anadir parametros, asi que hay que penalizar la complejidad:
#
#     AIC = chi2 + 2k
#     BIC = chi2 + k ln(n)
#
# con k = parametros libres y n = numero de datos. Las dos columnas ya
# estan en el CSV. Lo que se compara son DIFERENCIAS contra el mejor.
#
# Con n = 1099, ln(n) = 7.0: el BIC cobra 7 unidades de chi2 por cada
# parametro extra y el AIC solo 2. Por eso pueden discrepar, y cuando
# discrepan la discrepancia ES informacion: quiere decir que el parametro
# extra compra una mejora intermedia — suficiente para el AIC, no para el
# BIC.
#
# LIMITE QUE HAY QUE DECLARAR: AIC y BIC son aproximaciones a la
# evidencia bayesiana. La respuesta propiamente bayesiana es el factor de
# Bayes, que pide muestreo anidado (MultiNest, PolyChord) y este marco no
# lo tiene. Ademas se evaluan en el MAP, asi que valen solo si el MAP se
# encontro de verdad.

# Escala de Jeffreys sobre Delta BIC, con su etiqueta. El color es una
# rampa ordinal de un solo tono (validada: L monotona, extremo claro
# sobre el fondo); la etiqueta viaja siempre con el color, nunca sola.
JEFFREYS = [(2.0,  'indistinguible del mejor',      RAMPA_ESCALERA[0]),
            (6.0,  'evidencia positiva en contra',  RAMPA_ESCALERA[1]),
            (10.0, 'evidencia fuerte en contra',    RAMPA_ESCALERA[2]),
            (1e18, 'evidencia muy fuerte en contra', RAMPA_ESCALERA[3])]


def _jeffreys(d):
    for tope, etiqueta, color in JEFFREYS:
        if d < tope:
            return etiqueta, color
    return JEFFREYS[-1][1], JEFFREYS[-1][2]


def comparar_modelos(filas, peldano='none', dataset=None, prior=None) -> list:
    """El mejor ajuste de cada modelo, con su AIC y su BIC.

    Se toma el chi2 MINIMO sobre todos los metodos y rejillas de ese
    modelo: es la mejor estimacion disponible del MAP, y por eso el
    control cruzado entre familias (que el genetico y los samplers
    aterricen en el mismo chi2) es lo que justifica usarlo.
    """
    # [E-QPU8] Model selection is only meaningful on one dataset and prior:
    # rows with different n_data must never share a table.
    grupos = {((r.get('dataset') or '').strip(), (r.get('prior') or '').strip())
              for r in filas if r['_noi'] == peldano}
    if dataset is None and len(grupos) > 1:
        raise ValueError(f"comparar_modelos: several (dataset, prior) groups "
                         f"{sorted(grupos)}; pass dataset=/prior= explicitly")
    mejor = {}
    for r in filas:
        if r['_noi'] != peldano:
            continue
        if dataset is not None and ((r.get('dataset') or '').strip() != dataset
                                    or (r.get('prior') or '').strip() != prior):
            continue
        c = _num(r.get('chi2'))
        if c is None:
            continue
        if r['_mod'] not in mejor or c < _num(mejor[r['_mod']]['chi2']):
            mejor[r['_mod']] = r

    out = []
    for mod, r in mejor.items():
        c, a, b = (_num(r['chi2']), _num(r['AIC']), _num(r['BIC']))
        n = int(_num(r['n_data']) or 0)
        if None in (c, a, b) or not n:
            continue
        out.append({'modelo': mod, 'k': round((a - c) / 2), 'chi2': c,
                    'chi2_red': _num(r['chi2_red']), 'aic': a, 'bic': b,
                    'n': n})
    return sorted(out, key=lambda x: x['bic'])


def tabla_modelos(mods) -> None:
    if not mods:
        return
    n = mods[0]['n']
    print(f"\n{'='*72}\n  MODELOS COSMOLOGICOS — cual prefieren los datos"
          f"\n{'='*72}")
    print(f"  n = {n} datos,  ln(n) = {math.log(n):.3f}")
    print(f"  AIC = chi2 + 2k     BIC = chi2 + k*ln(n)\n")
    print(f"  {'modelo':7s} {'k':>2s} {'chi2':>11s} {'chi2_red':>9s} "
          f"{'dAIC':>8s} {'dBIC':>8s}   veredicto (dBIC)")
    amin = min(m['aic'] for m in mods)
    bmin = min(m['bic'] for m in mods)
    for m in mods:
        et, _ = _jeffreys(m['bic'] - bmin)
        print(f"  {m['modelo']:7s} {m['k']:2d} {m['chi2']:11.4f} "
              f"{m['chi2_red']:9.4f} {m['aic']-amin:8.3f} "
              f"{m['bic']-bmin:8.3f}   {et}")
    ga = min(mods, key=lambda x: x['aic'])['modelo']
    gb = mods[0]['modelo']
    print(f"\n  Gana por AIC: {ga}      Gana por BIC: {gb}")
    if ga != gb:
        print("  Que discrepen no es un problema: el BIC cobra "
              f"{math.log(n):.1f} unidades de chi2\n  por parametro extra y "
              "el AIC solo 2. El parametro de mas compra una\n  mejora "
              "intermedia — le alcanza al AIC y no al BIC.")
    print("\n  chi2_red cercano a 1 en todos: los cinco ajustan de forma\n"
          "  aceptable. Por eso chi2_red NO sirve para elegir entre ellos.")


def figura_modelos(mods, salida) -> str | None:
    if len(mods) < 2:
        return None
    os.makedirs(salida, exist_ok=True)
    amin = min(m['aic'] for m in mods)
    bmin = min(m['bic'] for m in mods)

    # UN solo orden para los dos paneles. Con `sharey` el segundo panel
    # sobrescribe las etiquetas del primero, asi que ordenarlos por
    # separado emparejaba cada nombre con el valor del otro criterio.
    # Ademas asi cada renglon es el MISMO modelo en los dos paneles, que
    # es lo que hace visible la discrepancia entre AIC y BIC.
    orden = sorted(mods, key=lambda x: -x['bic'])
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.6), sharey=True)
    for ax, clave, mini, titulo, pen in (
            (axes[0], 'aic', amin, r'(a) $\Delta$AIC', '2 por parametro'),
            (axes[1], 'bic', bmin, r'(b) $\Delta$BIC',
             f"{math.log(mods[0]['n']):.1f} por parametro")):
        for i, m in enumerate(orden):
            d = m[clave] - mini
            _, color = _jeffreys(d)
            ax.barh(i, max(d, 0.0), height=.58, color=color)
            ax.text(d + 0.35, i, f'{d:.2f}', va='center', fontsize=8,
                    color=C_REF)
        ax.set_yticks(range(len(orden)))
        ax.set_yticklabels([f"{m['modelo'].upper()}  (k={m['k']})"
                            for m in orden], fontsize=9)
        for u in (2, 6, 10):
            ax.axvline(u, color=C_REF, lw=.8, ls=':')
        ax.set_xlabel(f'{titulo.split()[-1]}  —  penaliza {pen}', fontsize=9)
        ax.set_title(titulo, fontsize=10, loc='left')
        ax.grid(axis='x', alpha=.22, lw=.5)
        ax.tick_params(labelsize=8)
        ax.set_xlim(0, max(4.0, max(m[clave] - mini for m in mods) * 1.22))

    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=c, label=e) for _, e, c in JEFFREYS],
               loc='lower center', ncol=2, fontsize=8, frameon=False)
    fig.suptitle('Qué modelo prefieren los datos   '
                 '(0 = el mejor; las punteadas son 2, 6 y 10)', fontsize=10)
    fig.tight_layout(rect=(0, .16, 1, .93))
    ruta = os.path.join(salida, 'modelos_aic_bic.png')
    _guardar(fig, ruta)
    plt.close(fig)
    return ruta


# =============================================================================
# [B-SIGMANQPP] ¿La sub-dispersion del variacional se va al refinar la rejilla?
# =============================================================================
#
# Es la objecion obvia al resultado de `incertidumbre_reportada`: "reportas
# una sigma mas angosta porque tu rejilla es gruesa; con mas qubits se
# arregla". Esta seccion la contesta con datos, siguiendo la razon
# sigma(QVMC)/sigma(MCMC) a lo largo de TODO el barrido de nqpp.
#
# Ojo con lo que NO se puede comparar aqui: el `final_KL` cambia de
# significado con nqpp, porque la distribucion objetivo esta definida sobre
# la rejilla y la rejilla cambia. Dos KL calculados sobre rejillas distintas
# no son la misma cantidad. La razon de sigmas si es comparable: sigma es el
# ancho del mismo parametro en las mismas unidades, y la referencia clasica
# no depende de nqpp.

def sigma_contra_rejilla(filas, metodo='QVMC 100%', par='Om',
                         peldano='none') -> dict:
    """Razon sigma(metodo)/sigma(MCMC) en funcion de nqpp, por modelo.

    Args:
        filas: filas ya leidas por `leer`.
        metodo: el peldano cuantico a seguir.
        par: parametro cosmologico (todos los modelos tienen Om).
        peldano: nivel de ruido; el ideal por defecto.

    Returns:
        dict[modelo] -> lista de `(nqpp, razon)` ordenada por nqpp.
    """
    celdas = defaultdict(dict)
    for r in filas:
        if r['_fam'] == 'samplers' and r['_noi'] == peldano:
            celdas[campaign_io.cell_key(r)][r['Method']] = r

    # [E-QPU8] One series per (model, campaign, dataset, prior, seed); the
    # campaign is added to the label only when several are present.
    series = {k[:4] for k in celdas}
    fuera = defaultdict(list)
    for key, v in celdas.items():
        mod, g = key[5], key[6]
        if len(series) > 1:
            mod = f"{mod} [{key[0]}|{key[1]}|seed {key[3] or '?'}]"
        ref, q = v.get(REFERENCIA), v.get(metodo)
        if not ref or not q:
            continue
        a, b = _num(q.get(par + '_std')), _num(ref.get(par + '_std'))
        if a is None or not b or g is None:
            continue
        fuera[mod].append((g, a / b))
    return {m: sorted(v) for m, v in fuera.items() if len(v) >= 2}


def tabla_sigma_rejilla(serie, metodo='QVMC 100%') -> None:
    """Imprime la razon de sigmas a lo largo del barrido de nqpp."""
    if not serie:
        return
    print(f"\n{'='*72}\n  ¿SE ARREGLA LA SUB-DISPERSION CON MAS QUBITS?"
          f"\n{'='*72}")
    print(f"  razon sigma({metodo}) / sigma({REFERENCIA}) en Om,")
    print("  a lo largo del barrido de nqpp.\n")
    for mod in sorted(serie):
        pts = serie[mod]
        cuerpo = '  '.join(f"{g}:{r:.3f}" for g, r in pts)
        print(f"  {mod:6s}  {cuerpo}")
        if len(pts) >= 4:
            cola = [r for _, r in pts[-3:]]
            print(f"  {'':6s}  -> ultimos tres: "
                  f"{min(cola):.3f}–{max(cola):.3f}")
    print("\n  Si la razon subiera hacia 1 al crecer nqpp, la sub-dispersion")
    print("  seria un artefacto de rejilla gruesa. Si se aplana por debajo")
    print("  de 1, es un sesgo estructural del objetivo variacional y no se")
    print("  arregla con mas qubits.")


def figura_sigma_rejilla(serie, salida, metodo='QVMC 100%') -> str | None:
    """Una curva por modelo: razon de sigmas contra nqpp.

    Son CINCO modelos y la paleta validada tiene tres colores, asi que el
    color por si solo no basta (dos pares quedarian del mismo tono). Cada
    serie se distingue por la terna color+marcador+trazo, y ademas se
    etiqueta al final de la curva, que es lo que de verdad se lee cuando
    las curvas se juntan en la meseta.
    """
    if not serie:
        return None
    os.makedirs(salida, exist_ok=True)
    marcas = ['o', 's', '^', 'D', 'v']
    trazos = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]

    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    for i, mod in enumerate(sorted(serie)):
        pts = serie[mod]
        x = [g for g, _ in pts]
        y = [r for _, r in pts]
        col = C_CAT[i % len(C_CAT)]
        ax.plot(x, y, color=col, lw=1.7, ms=5,
                marker=marcas[i % len(marcas)],
                linestyle=trazos[i % len(trazos)], label=mod.upper())
        ax.annotate(f' {mod.upper()}', (x[-1], y[-1]), fontsize=8,
                    color=col, va='center', ha='left')

    ax.axhline(1.0, color=C_REF, lw=1.2, ls='--')
    ax.annotate('de acuerdo con el MCMC', (min(x), 1.0), fontsize=8,
                color=C_REF, va='bottom', ha='left')
    ax.set_xlabel('nqpp  (qubits por parametro de la rejilla)', fontsize=9)
    ax.set_ylabel(f'sigma({metodo}) / sigma({REFERENCIA})', fontsize=9)
    ax.set_title('La sub-dispersion del variacional NO se arregla con mas '
                 'qubits', fontsize=10, loc='left')
    ax.set_xticks(sorted({g for v in serie.values() for g, _ in v}))
    ax.set_ylim(top=max(1.06, ax.get_ylim()[1]))
    ax.margins(x=.10)
    ax.grid(alpha=.22, lw=.5)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    ruta = os.path.join(salida, 'sigma_contra_rejilla.png')
    _guardar(fig, ruta)
    plt.close(fig)
    return ruta


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description='Compara metodos con el estimador que corresponde a cada '
                    'familia. No corre nada.')
    p.add_argument('carpetas', nargs='+')
    p.add_argument('--salida', default='figuras_eficiencia')
    p.add_argument('--peldano', default='none',
                   help="peldano de ruido a comparar (por defecto 'none')")
    a = p.parse_args(argv)

    filas = leer([c for c in a.carpetas if os.path.isdir(c)])
    if not filas:
        print("  no se leyo ninguna fila.")
        return 1
    print(f"\n  {len(filas)} filas leidas de {len(a.carpetas)} campana(s)")
    por = razones_samplers(filas, peldano=a.peldano)
    tabla(por)
    hechas = figura(por, a.salida)

    rel = razones_sigma(filas, peldano=a.peldano)
    if rel:
        tabla_sigma(rel)
        r = figura_sigma(rel, a.salida)
        if r:
            hechas.append(r)

    # [B-SIGMANQPP] contesta la objecion de "rejilla gruesa".
    serie = sigma_contra_rejilla(filas, peldano=a.peldano)
    if serie:
        tabla_sigma_rejilla(serie)
        r = figura_sigma_rejilla(serie, a.salida)
        if r:
            hechas.append(r)

    grupos = sorted({((r.get('dataset') or '').strip(),
                      (r.get('prior') or '').strip())
                     for r in filas if r['_noi'] == a.peldano})
    for ds, pr in grupos:
        mods = comparar_modelos(filas, peldano=a.peldano, dataset=ds, prior=pr)
        if mods:
            print(f"\n  dataset = {ds or '?'}   prior = {pr or '?'}")
            tabla_modelos(mods)
            if len(grupos) == 1:
                r = figura_modelos(mods, a.salida)
                if r:
                    hechas.append(r)
    if len(grupos) > 1:
        print("  (model-selection figure skipped: several datasets present)")

    gen = razones_genetico(filas, peldano=a.peldano)
    if gen:
        tabla_genetico(gen)
        aviso_sigma_genetico(filas, peldano=a.peldano)
        r = figura_genetico(gen, a.salida)
        if r:
            hechas.append(r)

    print(f"\n  {len(hechas)} figuras en {os.path.abspath(a.salida)}/\n")
    return 0


if __name__ == '__main__':
    sys.exit(main())
