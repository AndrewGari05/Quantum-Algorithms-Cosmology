#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qpu_noisy_simulation.py — Gemelo ruidoso del pipeline QPU.
================================================================================

Corre el pipeline de `qpu_cosmo_samplers.py` — el mismo codigo, los mismos
circuitos, la misma logica por conteos — contra un simulador RUIDOSO local en
vez de contra hardware de IBM.

`qpu_cosmo_samplers.py` NO SE MODIFICA
--------------------------------------
Ese archivo se mantiene intacto para las corridas reales de QPU. Este modulo
no lo copia: lo IMPORTA y sustituye una sola pieza, la conexion. Un archivo
copiado empezaria a divergir del original en la primera correccion que se
aplicara a uno y no al otro, y la divergencia seria silenciosa — justo la
clase de fallo que el proyecto lleva persiguiendo. Aqui, cualquier arreglo en
el pipeline de hardware llega solo.

Lo unico que se reemplaza es de donde salen los conteos:

    QPUConnection      -> IBM Quantum (SamplerV2 sobre backend fisico)
    LocalNoisyConnection -> AerSimulator con NoiseModel, misma interfaz

Todo lo demas — `build_proposal_circuit`, `build_ansatz`, `GridEncoding`,
`kl_from_counts`, SPSA, `MCMC_QPU`, `QVMC_QPU`, las figuras — es literalmente
el codigo de hardware.

Por que este modulo existe
--------------------------
1. Es el PRIMER test extremo-a-extremo de `qpu_cosmo_samplers.py`. Hasta
   ahora ese modulo solo estaba validado en `--dry-run`, es decir con conteos
   uniformes sinteticos que no ejercitan la fisica: se comprobaban formas y
   decodificados, no resultados.
2. Da la version CON RUIDO DE DISPARO del eje de ruido. El eje que vive en
   `cosmo_modular_quantum` calcula las probabilidades exactas de `diag(rho)`
   para aislar la degradacion por ruido del ruido de muestreo; aqui se mide
   la combinacion de ambos, que es lo que de verdad ve el hardware.

[DD-INERTE] Advertencia que este modulo hace explicita
------------------------------------------------------
`QPUConnection` activa dynamical decoupling XY4 y Pauli twirling de compuertas
y medicion. En hardware real esas opciones actuan. Con un backend simulado
NO actuan: qiskit-ibm-runtime las descarta en local testing mode y lo dice
solo con un `UserWarning` que se pierde entre los logs de una corrida larga:

    UserWarning: Options {'dynamical_decoupling': ...,'twirling': ...}
    have no effect in local testing mode.

Consecuencia: esta corrida mide **ruido SIN supresion de error**, mientras que
la de hardware mide **ruido CON supresion de error**. No son el mismo
experimento. Es una lectura defendible — da una COTA INFERIOR de la calidad
alcanzable en hardware — pero tiene que quedar registrada, no inferirse. Por
eso `LocalNoisyConnection` no usa `SamplerV2` en absoluto: habla con
`AerSimulator` directamente y REPORTA en el log que la supresion de error esta
inerte, en vez de fijar unas opciones que sabe que se van a ignorar.

Uso
---
    # Peldano sintetico, QMCMC:
    python qpu_noisy_simulation.py --model lcdm --method qmcmc \\
        --noise full --steps 200 --chains 4

    # Backend calibrado real, QVMC:
    python qpu_noisy_simulation.py --model wcdm --method qvmc \\
        --noise FakeBrisbane --iters 30 --nqpp 3

    # Control: mismo pipeline SIN ruido (aisla el efecto del muestreo)
    python qpu_noisy_simulation.py --model lcdm --method qmcmc --noise none

Los flags de hardware (`--backend`, `--least-busy`, `--token`, `--session`)
se aceptan pero se ignoran: aqui no hay hardware al que apuntar.
"""

from __future__ import annotations

import argparse
import logging
import time
from typing import Dict, List, Optional

import numpy as np

import cosmo_noise as cnoise
import qpu_cosmo_samplers as qpu

__all__ = ['LocalNoisyConnection', 'build_parser', 'main']


# =============================================================================
# La unica pieza sustituida
# =============================================================================

class LocalNoisyConnection(qpu.QPUConnection):
    """`QPUConnection` que devuelve conteos de un Aer ruidoso local.

    Implementa la MISMA interfaz que la conexion de hardware —
    `transpile_isa`, `run_pub`, `close`, `timer`, `shots` — de modo que
    `QPUProposalEngine`, `MCMC_QPU` y `QVMC_QPU` funcionan sin cambio alguno.

    No hereda el `__init__` del padre porque ese abre una conexion a IBM.

    Args:
        noise: peldano del eje (`cosmo_noise.NoiseSpec`).
        shots: disparos por binding.
        seed: semilla del simulador, para reproducibilidad.
        logger: logger donde reportar la configuracion efectiva.
    """

    def __init__(self, noise: "cnoise.NoiseSpec", shots: int = 4096,
                 seed: Optional[int] = None,
                 logger: Optional[logging.Logger] = None):
        from cosmo_core import make_simulator

        self.shots = shots
        self.dry_run = False
        self.use_session = False
        self.log = logger or logging.getLogger("qpu-noisy")
        self.timer = qpu.TimingEstimator()
        self._context = None
        self.noise = noise
        self.seed = seed
        self.pm = None                      # sin backend fisico: sin ISA pass

        # [NOISE] counts_route=True: los circuitos de este pipeline MIDEN, asi
        # que Aer aplica el canal de lectura por si mismo. Aplicarlo ademas en
        # cerrado seria doble conteo.
        kwargs = noise.simulator_kwargs(counts_route=True)
        if seed is not None:
            kwargs['seed_simulator'] = int(seed)
        self.backend = make_simulator(**kwargs)

        self.log.info("Simulador local: metodo=%s | peldano de ruido='%s' | "
                      "shots=%d", kwargs.get('method'), noise.label, shots)
        # [DD-INERTE] Decirlo SIEMPRE y en el log de la corrida, no como
        # warning perdido: es la diferencia entre este experimento y el de
        # hardware, y sin ella la comparacion entre ambos es enganosa.
        self.log.warning(
            "[DD-INERTE] Esta corrida NO lleva dynamical decoupling ni Pauli "
            "twirling. En hardware real esas opciones si actuan, asi que los "
            "resultados de aqui son una COTA INFERIOR de la calidad "
            "alcanzable en QPU, no una prediccion de ella.")
        if noise.is_ideal:
            self.log.info("Peldano 'none': sin ruido de compuerta ni de "
                          "lectura. Lo que quede es ruido de DISPARO, que es "
                          "el control con el que comparar los demas "
                          "peldanos.")

    # ------------------------------------------------------------------ #
    def transpile_isa(self, qc):
        """Transpila al conjunto de compuertas del simulador.

        Sin backend fisico no hay mapa de acoplamiento que respetar, asi que
        esto es solo una traduccion a la base soportada. Se conserva el
        patron transpile-once del original: el llamador invoca esto una vez
        por plantilla, no por circuito.

        Args:
            qc: circuito logico.

        Returns:
            Circuito transpilado.
        """
        from qiskit import transpile
        return transpile(qc, self.backend, optimization_level=1)

    # ------------------------------------------------------------------ #
    def run_pub(self, isa_circuit, parameter_values: np.ndarray,
                shots: Optional[int] = None) -> List[Dict[str, int]]:
        """Un job con B bindings -> lista de B diccionarios de conteos.

        Reproduce el contrato de `QPUConnection.run_pub` exactamente: mismo
        orden de los bindings, mismas claves de conteo, mismo registro en el
        `TimingEstimator`. Ese contrato es lo que permite que el resto del
        pipeline no note la diferencia.

        Args:
            isa_circuit: circuito ya transpilado.
            parameter_values: array (B, n_phi) de bindings.
            shots: disparos por binding.

        Returns:
            Lista de B diccionarios {cadena de bits: frecuencia}.
        """
        shots = shots or self.shots
        pv = np.atleast_2d(np.asarray(parameter_values, dtype=float))
        b = int(pv.shape[0])
        params = list(isa_circuit.parameters)
        t0 = time.time()

        if params:
            if pv.shape[1] != len(params):
                raise ValueError(
                    f"run_pub: el circuito tiene {len(params)} parametros y "
                    f"llegaron bindings de ancho {pv.shape[1]}.")
            # Aer espera UNA dict por circuito, con la LISTA de B valores por
            # parametro (no B dicts): pasarlo al reves falla con AerError.
            binds = [{p: list(pv[:, i]) for i, p in enumerate(params)}]
        else:
            binds = None

        result = self.backend.run(isa_circuit, parameter_binds=binds,
                                  shots=shots).result()
        t_wall = time.time() - t0
        # Sin cola ni sobrecoste de API: el tiempo de pared ES el de
        # ejecucion, asi que se reporta como tal en vez de dejar que el
        # estimador lo derive de la heuristica de disparos del hardware.
        self.timer.record(b, shots, t_wall, t_exec_reported=t_wall)

        counts = result.get_counts()
        if isinstance(counts, dict):
            counts = [counts]
        if len(counts) != b:
            raise RuntimeError(
                f"run_pub: se esperaban {b} distribuciones de conteos y "
                f"llegaron {len(counts)}.")
        return [dict(c) for c in counts]

    # ------------------------------------------------------------------ #
    def close(self):
        """Nada que cerrar: no hay sesion ni lote remoto."""
        return None


# =============================================================================
# CLI
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    """Parser del gemelo: el del pipeline QPU mas el eje de ruido.

    Reutilizar `qpu.build_parser()` garantiza que cualquier flag nuevo del
    pipeline de hardware aparezca aqui sin tocar este archivo.

    Returns:
        `ArgumentParser` configurado.
    """
    p = qpu.build_parser()
    p.description = ("Gemelo ruidoso de qpu_cosmo_samplers: mismo pipeline "
                     "por conteos, contra Aer con ruido en vez de IBM.")
    cnoise.add_noise_cli(p)
    p.add_argument('--noise-seed', type=int, default=None,
                   help='semilla del simulador de Aer (reproducibilidad del '
                        'ruido de disparo). Por defecto usa --seed.')
    return p


def main(argv: Optional[List[str]] = None) -> int:
    """Punto de entrada: corre `qpu.main` con la conexion sustituida.

    La sustitucion se hace parcheando el nombre `QPUConnection` DENTRO del
    modulo de hardware mientras dura la llamada, y restaurandolo despues. De
    ese modo se reutiliza integro el `main` del pipeline real — presupuesto de
    jobs, logging, figuras, CSV — sin duplicar una sola linea de su logica, y
    sin dejar el modulo de hardware alterado para el resto del proceso.

    Args:
        argv: argumentos de linea de comandos.

    Returns:
        Codigo de salida.
    """
    args = build_parser().parse_args(argv)
    spec = cnoise.spec_from_args(args)

    total_q = args.nqpp * qpu.MODELS[args.model].n_params
    if not spec.is_ideal and total_q > cnoise.MAX_NOISY_QUBITS:
        print(f"[NOISE] {args.model} con nqpp={args.nqpp} son {total_q} "
              f"qubits, por encima del techo de "
              f"{cnoise.MAX_NOISY_QUBITS} del eje de ruido. La matriz de "
              f"densidad ocuparia "
              f"{cnoise.noisy_density_bytes(total_q) / 2**30:.1f} GB y el "
              f"job tardaria horas. Baja --nqpp o usa --noise none.")
        return 1

    seed = args.noise_seed if args.noise_seed is not None else args.seed

    def _factory(*_a, **kw):
        """Sustituto de QPUConnection: ignora los flags de hardware."""
        return LocalNoisyConnection(noise=spec,
                                    shots=kw.get('shots', args.shots),
                                    seed=seed, logger=kw.get('logger'))

    # Reenviar solo los flags que el parser de hardware conoce; los del eje de
    # ruido ya estan capturados en `spec` y `qpu.build_parser()` los rechazaria.
    passthrough = _strip_noise_flags(argv)

    original = qpu.QPUConnection
    qpu.QPUConnection = _factory
    try:
        return qpu.main(passthrough)
    finally:
        qpu.QPUConnection = original


def _strip_noise_flags(argv: Optional[List[str]]) -> Optional[List[str]]:
    """Quita del argv los flags propios del eje de ruido.

    `qpu.main` reconstruye sus argumentos con `qpu.build_parser()`, que no
    conoce `--noise*`; pasarselos abortaria con "unrecognized arguments".

    Args:
        argv: argumentos originales, o None para usar `sys.argv[1:]`.

    Returns:
        Lista filtrada, o None si `argv` era None y no habia nada que filtrar.
    """
    import sys

    source = list(sys.argv[1:] if argv is None else argv)
    noise_flags = {'--noise', '--noise-readout-p', '--noise-gate-p1',
                   '--noise-gate-p2', '--noise-seed'}
    out: List[str] = []
    skip = False
    for token in source:
        if skip:
            skip = False
            continue
        if token in noise_flags:
            skip = True
            continue
        if any(token.startswith(f + '=') for f in noise_flags):
            continue
        out.append(token)
    return out


if __name__ == '__main__':
    raise SystemExit(main())
