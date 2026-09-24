#!/usr/bin/env python3
"""recolectar_figuras.py — junta los ejemplos que ilustran la guia.

La guia (`guia_figuras.tex`) muestra una figura real de cada tipo. Este
script las recoge de una campana ya corrida y las copia a `figs/` con los
nombres que el .tex espera, para que la guia se pueda regenerar cuando
cambien las figuras del proyecto.

    python docs/guia_figuras/recolectar_figuras.py \\
        RUTA/A/hpc_20260907_182345 RUTA/A/figuras_2026-09-21

El primer argumento es la carpeta maestra de la campana; el segundo, la
carpeta de salida de `graficas_ruido.py` / `comparar_algoritmos.py` (la que
contiene `ruido_*` y `comparacion_*`).

No dibuja nada: solo copia. Si un ejemplo falta, lo dice y sigue, porque es
mejor una guia con una figura de menos que un fallo a medio camino.
"""
from __future__ import annotations

import glob
import os
import shutil
import sys

#: destino -> patron, relativo a la carpeta de la campana.
DE_CAMPANA = {
    'a_convergence.png': 'convergence_lcdm.png',
    'b_cost.png':        'cost_lcdm.png',
    'q_convergence_gen.png': 'convergence_lcdm_genetico.png',
    'r_cost_gen.png':    'cost_lcdm_genetico.png',
    'c_ladder_trends.png':
        'samplers_lcdm_nqpp3_noise-none/model_lcdm/ladder_trends_lcdm.png',
    'd_ladder_summary.png':
        'samplers_lcdm_nqpp3_noise-none/model_lcdm/ladder_summary_lcdm.png',
    'e_ladder_rhat.png':
        'samplers_lcdm_nqpp3_noise-none/model_lcdm/ladder_rhat_qmcmc_lcdm.png',
    # El de CPL, no el de LCDM: en CPL se ve mejor la separacion de los
    # peldanos en el eje logaritmico.
    'f_ladder_kl.png':
        'samplers_cpl_nqpp3_noise-none/model_cpl/ladder_kl_qvmc_cpl.png',
    'g_corner_ladder.png':
        'samplers_lcdm_nqpp3_noise-none/model_lcdm/corner_ladder_qmcmc_lcdm.png',
    'h_corner_1to1.png':
        'samplers_lcdm_nqpp3_noise-none/model_lcdm/'
        'corner_ladder_1to1_qvmc_lcdm_q100.png',
    'i_fitness.png':
        'genetic_lcdm_nb6_noise-none/model_lcdm/fitness_lcdm_fitness.png',
    'j_genetic_conv.png':
        'genetic_lcdm_nb6_noise-none/model_lcdm/'
        'genetic_convergence_lcdm_genetic_convergence.png',
}

#: destino -> patron, relativo a la carpeta de figuras derivadas.
DE_FIGURAS = {
    'k_ruido_vs_rejilla.png':
        'ruido_*/ruido_vs_rejilla_samplers_lcdm_desplazamiento.png',
    'l_ruido_fijo.png':  'ruido_*/ruido_fijo_samplers_lcdm_nqpp4.png',
    'm_eficiencia.png':  'comparacion_*/eficiencia_QMCMC100.png',
    'n_incertidumbre.png': 'comparacion_*/incertidumbre_reportada.png',
    'o_genetico.png':    'comparacion_*/genetico_chi2grid.png',
    'p_modelos.png':     'comparacion_*/modelos_aic_bic.png',
}


def copiar(raiz: str, mapa: dict, destino: str) -> tuple:
    """Copia cada patron de `mapa` a `destino`.

    Returns:
        `(copiadas, faltantes)`, dos listas de nombres.
    """
    ok, falta = [], []
    for nombre, patron in mapa.items():
        encontrados = sorted(glob.glob(os.path.join(raiz, patron)))
        if not encontrados:
            falta.append((nombre, patron))
            continue
        shutil.copyfile(encontrados[0], os.path.join(destino, nombre))
        ok.append(nombre)
    return ok, falta


def main(argv=None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 2:
        print(__doc__)
        return 2
    campana, figuras = args
    destino = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figs')
    os.makedirs(destino, exist_ok=True)

    ok1, falta1 = copiar(campana, DE_CAMPANA, destino)
    ok2, falta2 = copiar(figuras, DE_FIGURAS, destino)

    print(f"copiadas {len(ok1) + len(ok2)} de "
          f"{len(DE_CAMPANA) + len(DE_FIGURAS)} figuras -> {destino}/")
    for nombre, patron in falta1 + falta2:
        print(f"  FALTA {nombre}: no encontre {patron}")
    if falta1 or falta2:
        print("\n  Las que faltan salen como recuadro vacio en el PDF.")
        print("  Suele ser que la campana no tiene esa celda; ajusta el")
        print("  patron en este archivo y vuelve a correr.")
    print("\n  Ahora: cd docs/guia_figuras && latexmk -pdf guia_figuras.tex")
    return 0


if __name__ == '__main__':
    sys.exit(main())
