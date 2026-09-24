#!/usr/bin/env python3
"""barras_de_error.py — de dónde salen las barras de error.

Corre una cadena de verdad sobre la posterior de verdad (LCDM,
CC+BAO+Pantheon) y muestra, con números, los tres pasos que convierten una
lista de puntos en «Om = 0.2769 +/- 0.0104»:

  1. la cadena visita cada region en proporcion a su probabilidad,
  2. los puntos repetidos son los rechazos — asi es como se castiga lo
     improbable sin tirar nada,
  3. la media y la desviacion estandar de los puntos SON las columnas
     Om_mean / Om_std del CSV de la campana.

Ademas escribe docs/circuitos/figs/barras_de_error.{png,pdf}.

    python docs/circuitos/barras_de_error.py            # figura + numeros
    python docs/circuitos/barras_de_error.py --no-fig   # solo numeros

La salida de referencia esta en docs/circuitos/barras_salida.txt.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ))
os.environ.setdefault('MPLBACKEND', 'Agg')

import cosmo_modular_quantum as cmq          # noqa: E402
import cosmo_noise as cn                     # noqa: E402
from cosmo_core import MODELS, Posterior, ess_chains  # noqa: E402

N_PASOS = 3000
N_CADENAS = 4
BURN = 300
SEMILLA = 42

# Lo que la campaña reportó para esta misma celda (nqpp3, noise=none):
CAMPANA = {'Om_mean': 0.276444, 'Om_std': 0.010486,
           'H0_mean': 69.594941, 'H0_std': 0.800716,
           'acceptance': 0.4873, 'ESS': 6872.0}


def corre():
    cmq.set_noise(cn.NoiseSpec.from_level('none'), 'auto')
    cmq._reseed(SEMILLA)
    post = Posterior(MODELS['lcdm'], 'CC+BAO+Pantheon')
    q = cmq.QMCMCModular(post, {'proposal': False, 'acceptance': False},
                         n_chains=N_CADENAS, n_burn=0,
                         stop_on_convergence=False)
    return post, q.run(n_steps=N_PASOS, progress=False)


def rhat(chains: np.ndarray, j: int = 0) -> float:
    """R-hat simple (Gelman--Rubin) sobre el parametro j, tirando burn-in."""
    x = chains[:, BURN:, j]
    m, n = x.shape
    medias = x.mean(axis=1)
    W = x.var(axis=1, ddof=1).mean()
    B = n * medias.var(ddof=1)
    var = (n - 1) / n * W + B / n
    return float(np.sqrt(var / W))


def main(hacer_fig: bool = True) -> None:
    post, info = corre()
    ch = info['chains']
    # Las estadisticas se calculan SIN burn-in. Es la unica diferencia
    # entre 'info[flat]' (crudo, lo que la cadena anoto) y lo que se
    # reporta.
    flat = ch[:, BURN:, :].reshape(-1, ch.shape[2])
    om, h0 = flat[:, 0], flat[:, 1]
    om_crudo = info['flat'][:, 0]

    print("=" * 66)
    print("  DE LA CADENA A LA BARRA DE ERROR")
    print("=" * 66)
    print(f"  modelo LCDM, datos CC+BAO+Pantheon, {N_CADENAS} cadenas "
          f"x {N_PASOS} pasos")
    print(f"  aceptacion medida = {info['acceptance']:.4f}")

    # --- 1. los puntos repetidos son los rechazos ----------------------
    print("\n--- 1. los primeros 10 pasos de la cadena 0 ---")
    print(f"  {'paso':>5}  {'Om':>9}  {'H0':>8}   que paso")
    ant = None
    for t in range(10):
        o, h = ch[0, t, 0], ch[0, t, 1]
        que = "se quedo (rechazo)" if ant is not None and np.allclose(
            ch[0, t], ant) else "se movio (acepto)"
        print(f"  {t:5d}  {o:9.5f}  {h:8.3f}   {que}")
        ant = ch[0, t].copy()

    rep = sum(int(np.allclose(ch[c, t], ch[c, t - 1]))
              for c in range(ch.shape[0]) for t in range(1, ch.shape[1]))
    tot = ch.shape[0] * (ch.shape[1] - 1)
    print(f"\n  puntos repetidos: {rep} de {tot} = {rep/tot:.4f}")
    print(f"  1 - aceptacion  : {1 - info['acceptance']:.4f}   "
          "<- es lo mismo, por definicion")
    print("  Un rechazo NO tira el paso: vuelve a anotar el punto donde")
    print("  esta. Por eso las zonas improbables salen POCAS veces en la")
    print("  lista, que es exactamente lo que se quiere.")

    # --- 2. la media y la desviacion SON las columnas del CSV ----------
    print("\n--- 2. media y desviacion de la lista de puntos ---")
    for nom, x, k in (('Om', om, 'Om'), ('H0', h0, 'H0')):
        m, s = x.mean(), x.std(ddof=1)
        p16, p50, p84 = np.percentile(x, [16, 50, 84])
        print(f"  {nom}:  media = {m:.6f}   desv.est. = {s:.6f}")
        print(f"       percentiles 16/50/84 = {p16:.5f} / {p50:.5f} / "
              f"{p84:.5f}")
        print(f"       la campana reporto   = {CAMPANA[k+'_mean']:.6f} "
              f"+/- {CAMPANA[k+'_std']:.6f}")
    print("  (no son identicos porque son semillas y longitudes distintas;")
    print("   son la MISMA cantidad calculada de la MISMA manera)")
    print(f"\n  si NO se tira el burn-in:  Om = {om_crudo.mean():.6f} +/- "
          f"{om_crudo.std(ddof=1):.6f}")
    print(f"  tirandolo               :  Om = {om.mean():.6f} +/- "
          f"{om.std(ddof=1):.6f}")
    print("  el transitorio inicial infla la sigma; por eso se tira.")

    # --- 3. las tres condiciones para que esto valga -------------------
    print("\n--- 3. las tres condiciones ---")
    r = rhat(ch)
    ess = ess_chains(ch[:, BURN:, :])
    print(f"  (a) burn-in tirado      : {BURN} pasos de {N_PASOS}")
    print(f"  (b) R-hat (Om, sin burn): {r:.5f}   hace falta < 1.01")
    print(f"  (c) ESS                 : {ess:.0f} efectivas de "
          f"{len(om)} muestras (ya sin burn-in)")
    print(f"      -> cada {len(om)/ess:.1f} pasos vale 1 independiente")
    print(f"      -> el error de la media es s/sqrt(ESS) = "
          f"{om.std(ddof=1)/np.sqrt(ess):.6f}, no s/sqrt(N) = "
          f"{om.std(ddof=1)/np.sqrt(len(om)):.6f}")

    if hacer_fig:
        figura(ch, om)


def figura(ch: np.ndarray, om: np.ndarray) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    C, REF, OCRE = '#2a78d6', '#3A3A44', '#A8651F'
    fig, ax = plt.subplots(1, 2, figsize=(10.4, 3.5),
                           gridspec_kw={'width_ratios': [1.9, 1]})
    for c in range(ch.shape[0]):
        ax[0].plot(ch[c, :, 0], lw=.5, color=C, alpha=.5)
    ax[0].axvspan(0, BURN, color=OCRE, alpha=.12)
    ax[0].text(BURN / 2, ax[0].get_ylim()[1], ' burn-in\n (se tira)',
               va='top', ha='center', fontsize=8, color=OCRE)
    ax[0].set_xlabel('paso de la cadena', fontsize=9)
    ax[0].set_ylabel(r'$\Omega_m$', fontsize=10)
    ax[0].set_title('(a) las 4 cadenas recorriendo el espacio',
                    fontsize=10, loc='left')
    ax[0].grid(alpha=.2, lw=.5)
    ax[0].tick_params(labelsize=8)

    m, s = om.mean(), om.std(ddof=1)
    ax[1].hist(om, bins=60, color=C, alpha=.75, edgecolor='none')
    ax[1].axvline(m, color=REF, lw=1.6)
    for d in (-1, 1):
        ax[1].axvline(m + d * s, color=REF, lw=1.0, ls='--')
    ax[1].set_xlabel(r'$\Omega_m$', fontsize=10)
    ax[1].set_ylabel('cuántas veces la visitó', fontsize=9)
    ax[1].set_title('(b) el histograma ES la posterior',
                    fontsize=10, loc='left')
    ax[1].tick_params(labelsize=8)
    ax[1].grid(axis='y', alpha=.2, lw=.5)
    alto = ax[1].get_ylim()[1]
    ax[1].set_ylim(0, alto * 1.22)          # aire para las etiquetas
    ax[1].annotate(f'media\n{m:.4f}', (m, alto * 1.20), fontsize=8,
                   ha='center', va='top', color=REF)
    ax[1].annotate(f'$\\pm\\sigma$ = {s:.4f}', (m + s, alto * .60),
                   fontsize=8, ha='left', va='center', color=REF,
                   bbox=dict(fc='white', ec='none', alpha=.8, pad=1.2))
    fig.suptitle(r'De la cadena a  $\Omega_m = %.4f \pm %.4f$' % (m, s),
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .93))

    d = Path(__file__).resolve().parent / 'figs'
    d.mkdir(exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(d / f'barras_de_error.{ext}',
                    dpi=150 if ext == 'png' else None, bbox_inches='tight')
    print(f"\n  figura -> {d}/barras_de_error.{{png,pdf}}")


if __name__ == '__main__':
    main(hacer_fig='--no-fig' not in sys.argv)
