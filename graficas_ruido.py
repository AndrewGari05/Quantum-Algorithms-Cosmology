#!/usr/bin/env python3
"""graficas_ruido.py — el eje de ruido, leido de resultados ya obtenidos.

Este script NO corre ningun algoritmo ni toca ningun archivo de la campana.
Lee los CSV que la campana ya escribio y dibuja tres cosas:

  (1) metrica  vs  nqpp        — una curva por peldano de ruido  (samplers)
  (2) metrica  vs  n_bits      — una curva por peldano de ruido  (genetico)
  (3) a nqpp / n_bits FIJO     — los peldanos de ruido lado a lado, rung a rung

La pregunta que contestan las tres es la misma: ¿cuanto mueve el ruido al
resultado, y ese movimiento crece o no con el tamano del registro?

────────────────────────────────────────────────────────────────────────────
DE DONDE SALE EL PELDANO DE RUIDO
────────────────────────────────────────────────────────────────────────────
Del NOMBRE DE LA CARPETA (`..._noise-<peldano>`), no de la columna `noise`
del CSV, y esto es deliberado:

  * `none-counts` es un peldano del eje del proyecto pero NO es un valor de la
    columna: en el CSV se escribe `noise='none'` + `proposal_route='counts'`,
    porque eso es literalmente lo que corrio. El nombre de la carpeta es el
    unico sitio donde el control aparece como tal.
  * [B-PROV-GEN] hasta esta correccion, TODA fila genetica escribia
    `noise='none'` pasara lo que pasara, porque el escritor de CSV leia el
    global de otro modulo. Las campanas ya corridas tienen esa columna mal.

Asi que la carpeta manda y la columna se usa solo para AVISAR cuando las dos
no coinciden. Eso deja las campanas viejas perfectamente utilizables: el dato
cientifico nunca estuvo mal, solo la etiqueta, y la etiqueta se recupera del
nombre de la carpeta sin volver a correr nada.

────────────────────────────────────────────────────────────────────────────
QUE SE MIDE
────────────────────────────────────────────────────────────────────────────
  * `desplazamiento`  — |p(ruido) − p(ideal)| / sigma(ideal), por parametro,
    contra la MISMA celda (mismo modelo, mismo rung, mismo nqpp) corrida sin
    ruido. Es la unica metrica que dice si el ruido mueve la CIENCIA.
  * `acceptance`      — fraccion de aceptacion de Metropolis (solo QMCMC).
  * `final_KL`        — KL final del variacional (solo QVMC/VI).
  * `chi2_grid`       — [B-REFINE] chi2 del mejor punto de la REJILLA, sin
    refinar. Es el unico numero del genetico que distingue un rung de otro:
    el refinador local lleva a los cuatro al mismo minimo continuo.
  * `Time_s`          — costo. El eje de ruido es carisimo y conviene verlo.

Uso:
    python graficas_ruido.py CARPETA_CAMPANA [...] [--salida FIGS]
    python graficas_ruido.py results/hpc_20260907_130425 --salida figuras_ruido
"""
from __future__ import annotations

import argparse
import csv
import glob

import campaign_io
import math
import os
import re
import sys
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt          # noqa: E402

# ── vocabulario del eje ──────────────────────────────────────────────────
# El orden es el del eje, de ideal a mas ruidoso, y es el que llevan la
# leyenda y el eje x de las figuras de peldano fijo.
ORDEN_RUIDO = ['none', 'none-counts', 'readout', 'full']

# Color por peldano. `none` y `none-counts` comparten tono a proposito: son el
# MISMO nivel de ruido leido por dos rutas distintas (amplitud contra cuentas),
# no dos niveles. Se distinguen por el trazo, no por el color, para que la
# figura no sugiera una gradacion que no existe.
COLOR_RUIDO = {
    'none':         '#1f77b4',
    'none-counts':  '#1f77b4',
    'readout':      '#ff7f0e',
    'full':         '#d62728',
}
TRAZO_RUIDO = {
    'none':         'solid',
    'none-counts':  'dashed',
    'readout':      'solid',
    'full':         'solid',
}
# Los backends reales (FakeBrisbane, …) entran por este color de reserva: no
# son un peldano sintetico y no tienen sitio fijo en la escala.
COLOR_BACKEND = ['#2ca02c', '#9467bd', '#8c564b', '#e377c2']

# Columnas que NO son parametros cosmologicos aunque terminen en _mean/_std.
_NO_PARAM = {'Om_mean', 'Om_std'} - {'Om_mean', 'Om_std'}   # (ninguna)


# ═════════════════════════════════════════════════════════════════════════
# 1.  LECTURA
# ═════════════════════════════════════════════════════════════════════════

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


def _num(txt):
    """float o None. El CSV usa '' y '—' para 'aqui no aplica'."""
    if txt is None:
        return None
    t = str(txt).strip()
    if t in ('', '—', '-', 'nan', 'None'):
        return None
    try:
        v = float(t)
    except ValueError:
        return None
    return v if math.isfinite(v) else None


def columnas_de_parametros(campos) -> list:
    """Nombres de los parametros cosmologicos presentes en el encabezado.

    El esquema por corrida nombra las columnas (`Om_mean`, `H0_mean`,
    `w0_mean`…); el acumulado las llama `p1_mean`…`p4_mean`. Se aceptan los
    dos, porque distintas carpetas de la misma campana usan uno u otro.
    """
    return [c[:-5] for c in campos
            if c.endswith('_mean') and f'{c[:-5]}_std' in campos]


def leer_campana(raiz: str) -> list:
    """All rows of one campaign, each with its provenance resolved.

    [E-READ] Delegates to `campaign_io.read_campaign` (optional grid/noise
    tags, no double counting) and maps its keys to the names used here:
    `_familia`, `_modelo`, `_rejilla`, `_ruido` (from the folder),
    `_ruido_csv` (the CSV column, kept to cross-check) and `_camp`.
    """
    filas = []
    for r in campaign_io.read_campaign(raiz):
        r['_familia'] = r['_fam']
        r['_modelo'] = r['_mod']
        r['_rejilla'] = r['_g']
        r['_ruido'] = r['_noi']
        r['_ruido_csv'] = (r.get('noise') or '').strip()
        r['_ruta'] = r['_path']
        filas.append(r)
    return filas


def avisar_etiquetas_incoherentes(filas) -> None:
    """Denuncia las filas cuya columna `noise` no dice lo que dice su carpeta.

    No es cosmetico: si alguien agrupa por la columna en vez de por la
    carpeta, todos los peldanos del genetico caen en el mismo grupo y las
    diferencias por ruido desaparecen sin que nada falle.
    """
    malas = defaultdict(int)
    for r in filas:
        esperado = 'none' if r['_ruido'] == 'none-counts' else r['_ruido']
        if r['_ruido_csv'] and r['_ruido_csv'] != esperado:
            malas[(r['_familia'], r['_ruido'], r['_ruido_csv'])] += 1
    if not malas:
        return
    print("\n  AVISO [B-PROV-GEN] — la columna `noise` no coincide con la "
          "carpeta:")
    for (fam, carpeta, col), n in sorted(malas.items()):
        print(f"    {fam:8s}  carpeta dice '{carpeta}'  columna dice "
              f"'{col}'   ({n} filas)")
    print("    Se usa la CARPETA. El dato cientifico es valido; lo unico mal "
          "era la etiqueta,\n    y se recupera sin volver a correr nada.")


# ═════════════════════════════════════════════════════════════════════════
# 2.  METRICAS
# ═════════════════════════════════════════════════════════════════════════

def desplazamiento_en_sigmas(fila, ideal) -> float | None:
    """Cuanto movio el ruido al parametro, en sigmas de la corrida ideal.

    Se toma el MAXIMO sobre los parametros del modelo: si el ruido sesga
    aunque sea uno, el resultado ya esta sesgado. El divisor es la sigma
    IDEAL, no la ruidosa: el ruido suele ensanchar la posterior, y dividir
    entre la ruidosa escondería el sesgo detras de su propia dispersion.

    Args:
        fila: fila ruidosa.
        ideal: fila de la misma celda corrida sin ruido.

    Returns:
        float o None si no hay parametros comparables.
    """
    peor = None
    for p in columnas_de_parametros(fila.keys()):
        a, b = _num(fila.get(f'{p}_mean')), _num(ideal.get(f'{p}_mean'))
        s = _num(ideal.get(f'{p}_std'))
        if a is None or b is None or not s:
            continue
        d = abs(a - b) / s
        peor = d if peor is None else max(peor, d)
    return peor


METRICAS = {
    'desplazamiento': ('desplazamiento maximo  |Δp| / σ(ideal)', None),
    'acceptance':     ('fraccion de aceptacion', 'acceptance'),
    'final_KL':       ('KL final', 'final_KL'),
    'chi2_grid':      ('χ² del mejor punto de la rejilla (sin refinar)',
                       'chi2_grid'),
    'Time_s':         ('tiempo de pared (s)', 'Time_s'),
}


def valor(fila, metrica, indice_ideal):
    """Valor de una metrica en una fila, o None si no aplica ahi."""
    if metrica == 'desplazamiento':
        # [E-QPU9] For the genetic family `*_std` is the spread of the final
        # population, not an uncertainty, so a shift "in sigmas" is
        # meaningless there (it produced 223-sigma artefacts).
        if fila['_familia'] == 'genetic':
            return None
        ideal = indice_ideal.get(
            campaign_io.cell_key_without_noise(fila) + (fila['Method'],))
        return None if ideal is None else desplazamiento_en_sigmas(fila, ideal)
    return _num(fila.get(METRICAS[metrica][1]))


def indice_de_celdas_ideales(filas) -> dict:
    """La corrida SIN ruido de cada celda, para medir contra ella.

    La referencia es `none` — la ruta de amplitud —, no `none-counts`. Asi el
    desplazamiento de `none-counts` mide justamente el efecto de cambiar de
    ruta de lectura, que es para lo que existe ese control.
    """
    # [E-QPU8] Keyed by campaign, dataset, prior and seed as well, so a noisy
    # row is never measured against another campaign's ideal row.
    return {campaign_io.cell_key_without_noise(r) + (r['Method'],): r
            for r in filas if r['_ruido'] == 'none'}


# ═════════════════════════════════════════════════════════════════════════
# 3.  FIGURAS
# ═════════════════════════════════════════════════════════════════════════

def _estilo(peldano, reserva):
    if peldano in COLOR_RUIDO:
        return COLOR_RUIDO[peldano], TRAZO_RUIDO[peldano]
    return COLOR_BACKEND[reserva % len(COLOR_BACKEND)], 'dashdot'


def _leyenda_unida(axes):
    """Manejas y etiquetas de TODOS los paneles, sin repetir y en orden de eje.

    Leer la leyenda de un solo panel pierde los peldanos que no aparecen en
    ese panel — y esos son justo los interesantes, porque son los que solo se
    alcanzaron a correr en algunos rungs.
    """
    vistos = {}
    for fila in axes:
        for ax in fila:
            for h, e in zip(*ax.get_legend_handles_labels()):
                vistos.setdefault(e, h)
    orden = _orden(set(vistos))
    return [vistos[e] for e in orden], orden


def _orden(peldanos):
    """Peldanos sinteticos en el orden del eje; backends despues, alfabeticos."""
    conocidos = [p for p in ORDEN_RUIDO if p in peldanos]
    otros = sorted(p for p in peldanos if p not in ORDEN_RUIDO)
    return conocidos + otros


def figura_vs_rejilla(filas, familia, modelo, metrica, indice_ideal,
                      salida) -> str | None:
    """(1) y (2): la metrica contra nqpp / n_bits, una curva por peldano.

    Un panel por metodo (rung de la escalera), porque mezclar el clasico con
    el 100% cuantico en un solo eje esconde justo lo que se quiere ver.
    """
    sub = [r for r in filas
           if r['_familia'] == familia and r['_modelo'] == modelo]
    if not sub:
        return None

    metodos = sorted({r['Method'] for r in sub})
    datos = defaultdict(lambda: defaultdict(list))     # metodo → peldano → pts
    for r in sub:
        v = valor(r, metrica, indice_ideal)
        if v is not None and r['_rejilla'] is not None:
            datos[r['Method']][r['_ruido']].append((r['_rejilla'], v))
    metodos = [m for m in metodos if datos[m]]
    if not metodos:
        return None

    eje_x = 'nqpp  (qubits por parametro)' if familia == 'samplers' \
        else 'n_bits  (bits por parametro)'
    ncol = min(len(metodos), 3)
    nfil = math.ceil(len(metodos) / ncol)
    fig, axes = plt.subplots(nfil, ncol, figsize=(4.6 * ncol, 3.6 * nfil),
                             squeeze=False)
    peldanos = _orden({r['_ruido'] for r in sub})

    for k, met in enumerate(metodos):
        ax = axes[k // ncol][k % ncol]
        reserva = 0
        for p in peldanos:
            pts = sorted(datos[met].get(p, []))
            if not pts:
                continue
            c, ls = _estilo(p, reserva)
            if p not in COLOR_RUIDO:
                reserva += 1
            xs, ys = zip(*pts)
            ax.plot(xs, ys, marker='o', ms=4, lw=1.6, color=c, ls=ls, label=p)
        ax.set_title(met, fontsize=10)
        ax.set_xlabel(eje_x, fontsize=8)
        ax.set_ylabel(METRICAS[metrica][0], fontsize=8)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=.25, lw=.5)
        if metrica in ('Time_s', 'final_KL'):
            ax.set_yscale('log')
        if metrica == 'desplazamiento':
            # 1σ: por debajo de esta linea el ruido no mueve la ciencia.
            ax.axhline(1.0, color='0.4', lw=.9, ls=':')
    for k in range(len(metodos), nfil * ncol):
        axes[k // ncol][k % ncol].axis('off')

    # La leyenda se arma con la UNION de los paneles: un peldano que solo
    # aparece en un metodo (p.ej. el backend real, que solo se corrio en
    # algunos rungs) desaparecia de la leyenda si se leia solo el primer eje.
    manejas, etiquetas = _leyenda_unida(axes)
    fig.legend(manejas, etiquetas, loc='lower center', ncol=len(etiquetas),
               fontsize=8, frameon=False, title='peldano de ruido',
               title_fontsize=8)
    fig.suptitle(f'{modelo.upper()} — {METRICAS[metrica][0]} contra el '
                 f'tamano del registro', fontsize=11)
    fig.tight_layout(rect=(0, .07, 1, .96))
    nombre = f'ruido_vs_rejilla_{familia}_{modelo}_{metrica}.png'
    ruta = os.path.join(salida, nombre)
    _guardar(fig, ruta)
    plt.close(fig)
    return ruta


def figura_a_rejilla_fija(filas, familia, modelo, rejilla, metricas,
                          indice_ideal, salida) -> str | None:
    """(3): a nqpp / n_bits FIJO, los peldanos lado a lado, rung a rung.

    Barras agrupadas: un grupo por metodo de la escalera, una barra por
    peldano, un panel por metrica. Es la figura que contesta '¿cual de mis
    metodos aguanta el ruido?' sin que el tamano del registro se meta de por
    medio — y va en UNA figura por celda, no una por metrica, porque las
    metricas solo se leen bien comparadas entre si.
    """
    sub = [r for r in filas
           if r['_familia'] == familia and r['_modelo'] == modelo
           and r['_rejilla'] == rejilla]
    if not sub:
        return None
    peldanos = _orden({r['_ruido'] for r in sub})

    paneles = []
    for metrica in metricas:
        tabla = {}
        for r in sub:
            v = valor(r, metrica, indice_ideal)
            if v is not None:
                tabla[(r['Method'], r['_ruido'])] = v
        if tabla:
            metodos = [m for m in sorted({r['Method'] for r in sub})
                       if any((m, p) in tabla for p in peldanos)]
            paneles.append((metrica, tabla, metodos))
    if not paneles:
        return None

    ancho = 0.8 / max(len(peldanos), 1)
    anchura = max(1.5 * max(len(m) for _, _, m in paneles) + 3.0, 6.0)
    fig, axes = plt.subplots(len(paneles), 1, squeeze=False,
                             figsize=(anchura, 3.4 * len(paneles)))
    for k, (metrica, tabla, metodos) in enumerate(paneles):
        ax = axes[k][0]
        reserva = 0
        for j, p in enumerate(peldanos):
            c, _ = _estilo(p, reserva)
            if p not in COLOR_RUIDO:
                reserva += 1
            xs, ys = [], []
            for i, m in enumerate(metodos):
                if (m, p) in tabla:
                    xs.append(i + j * ancho - 0.4 + ancho / 2)
                    ys.append(tabla[(m, p)])
            # `none-counts` va hueca: comparte nivel de ruido con `none` y lo
            # que cambia es la ruta de lectura, no el peldano.
            ax.bar(xs, ys, width=ancho * .92, color=c,
                   label=p if xs else None,
                   hatch='//' if p == 'none-counts' else None,
                   edgecolor='white' if p == 'none-counts' else 'none',
                   alpha=.9)
        ax.set_xticks(range(len(metodos)))
        ax.set_xticklabels(metodos, rotation=15, ha='right', fontsize=8)
        ax.set_ylabel(METRICAS[metrica][0], fontsize=8)
        ax.tick_params(labelsize=8)
        ax.grid(axis='y', alpha=.25, lw=.5)
        if metrica in ('Time_s', 'final_KL'):
            ax.set_yscale('log')
        if metrica == 'desplazamiento':
            ax.axhline(1.0, color='0.4', lw=.9, ls=':')

    etiqueta = 'nqpp' if familia == 'samplers' else 'n_bits'
    manejas, etiquetas = _leyenda_unida(axes)
    fig.legend(manejas, etiquetas, loc='lower center', ncol=len(etiquetas),
               fontsize=8, frameon=False, title='peldano de ruido',
               title_fontsize=8)
    fig.suptitle(f'{modelo.upper()} — {etiqueta}={rejilla}: efecto del ruido '
                 f'peldano a peldano', fontsize=11)
    fig.tight_layout(rect=(0, .06, 1, .96))
    nombre = f'ruido_fijo_{familia}_{modelo}_{etiqueta}{rejilla}.png'
    ruta = os.path.join(salida, nombre)
    _guardar(fig, ruta)
    plt.close(fig)
    return ruta


# ═════════════════════════════════════════════════════════════════════════
# 4.  ORQUESTACION
# ═════════════════════════════════════════════════════════════════════════

def metricas_utiles(filas, familia) -> list:
    """Metricas que esta familia realmente llena; las vacias no se dibujan."""
    sub = [r for r in filas if r['_familia'] == familia]
    out = [] if familia == 'genetic' else ['desplazamiento']   # [E-QPU9]
    for m, (_, col) in METRICAS.items():
        if col and any(_num(r.get(col)) is not None for r in sub):
            out.append(m)
    return out


def construir(carpetas, salida) -> list:
    os.makedirs(salida, exist_ok=True)
    filas = []
    for c in carpetas:
        if not os.path.isdir(c):
            print(f"  no existe: {c}")
            continue
        f = leer_campana(c)
        print(f"  {os.path.basename(c)}: {len(f)} filas en "
              f"{len({r['_ruido'] for r in f})} peldanos")
        filas += f
    if not filas:
        print("  no se leyo ninguna fila con nombre de tarea reconocible.")
        return []

    avisar_etiquetas_incoherentes(filas)
    ideal = indice_de_celdas_ideales(filas)
    hechas = []

    for familia in ('samplers', 'genetic'):
        sub = [r for r in filas if r['_familia'] == familia]
        if not sub:
            continue
        modelos = sorted({r['_modelo'] for r in sub})
        metricas = metricas_utiles(filas, familia)
        print(f"\n  [{familia}] modelos={','.join(modelos)} | "
              f"metricas={','.join(metricas)}")
        for modelo in modelos:
            # Solo tiene sentido si ese modelo tiene MAS de un peldano.
            peldanos = {r['_ruido'] for r in sub if r['_modelo'] == modelo}
            if len(peldanos) < 2:
                print(f"    {modelo}: un solo peldano ({peldanos}), se omite")
                continue
            for metrica in metricas:
                r1 = figura_vs_rejilla(filas, familia, modelo, metrica,
                                       ideal, salida)
                if r1:
                    hechas.append(r1)
            # Peldano fijo: en las rejillas donde de verdad hay comparacion.
            for rej in sorted({r['_rejilla'] for r in sub
                               if r['_modelo'] == modelo
                               and r['_rejilla'] is not None}):
                cuantos = {r['_ruido'] for r in sub
                           if r['_modelo'] == modelo and r['_rejilla'] == rej}
                if len(cuantos) < 2:
                    continue
                r2 = figura_a_rejilla_fija(filas, familia, modelo, rej,
                                           metricas, ideal, salida)
                if r2:
                    hechas.append(r2)
    return hechas


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description='Figuras del eje de ruido a partir de resultados ya '
                    'obtenidos. No corre nada ni modifica la campana.')
    p.add_argument('carpetas', nargs='+',
                   help='carpetas maestras de campana (hpc_YYYYMMDD_HHMMSS)')
    p.add_argument('--salida', default='figuras_ruido',
                   help='carpeta donde dejar las figuras')
    a = p.parse_args(argv)

    print("\n" + "=" * 72)
    print("  EJE DE RUIDO — figuras desde resultados existentes")
    print("=" * 72)
    hechas = construir(a.carpetas, a.salida)
    print("\n" + "-" * 72)
    print(f"  {len(hechas)} figuras en {os.path.abspath(a.salida)}/")
    print("-" * 72 + "\n")
    return 0 if hechas else 1


if __name__ == '__main__':
    sys.exit(main())
