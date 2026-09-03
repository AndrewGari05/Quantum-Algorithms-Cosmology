#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
noise_feasibility_probe.py — Estudio de viabilidad del eje de ruido NISQ.
================================================================================

Script de VERIFICACION, no de produccion. No modifica ningun modulo del
proyecto y no entra en `cosmo_hpc_runner.py`: su unico proposito es sostener
con mediciones las decisiones de diseno de la Fase 4 (segundo eje de ablacion
= ruido), de modo que ninguna afirmacion del documento de diseno descanse en
lectura de codigo o en intuicion.

Contexto
--------
El framework produce hoy sus resultados con `AerSimulator(method='statevector')`
sin ruido. Anadir ruido choca con dos hechos:

  1. Aer no aplica ruido con `method='statevector'`.
  2. El simulador ideal lee amplitudes en sitios funcionalmente distintos
     (`QuantumProposalEngine._raw_block`, `hadamard_accept_log_batch`,
     `QVMCModular._kl_batch`, entre otros), y esas lecturas son justo lo que
     el ruido vuelve imposible.

Las sondas de abajo miden que rutas de escape existen y cuanto cuestan.

Sondas
------
A. `probe_local_testing_mode`
     Comprueba si `qiskit_ibm_runtime.SamplerV2` corre localmente cuando se le
     pasa un backend simulado (hipotesis del "atajo": reusar el pipeline por
     conteos de `qpu_cosmo_samplers.py` apuntandolo a Aer en vez de a IBM).
     Registra ADEMAS si las opciones de supresion de error del proyecto
     (dynamical decoupling XY4 + Pauli twirling) sobreviven a ese modo.

B. `probe_graded_axis`
     Comprueba que los cuatro niveles del eje de ruido propuesto
     (none / readout / full / backend real) son construibles y distinguibles.

C. `probe_cost`
     Mide tiempo por job y RAM pico de los dos metodos de simulacion con ruido
     (matriz de densidad vs trayectorias de statevector) sobre el ansatz real
     del proyecto, en los anchos de qubit que el runner realmente genera.

D. `probe_faithful_degradation`
     El resultado central. Verifica que rho[0,0] reproduce |psi_0|^2 de forma
     exacta sin ruido (el puente ideal->ruidoso no introduce error propio),
     reconfirma la identidad con min(1, e^Delta) contra la respuesta conocida,
     y mide como se degrada esa identidad canal por canal.

     Detecta ademas la trampa que invalidaria una columna entera de la matriz
     de ablacion: el error de LECTURA es un canal clasico posterior a la
     medicion y NO toca rho, de modo que leer rho[0,0] es ciego a el.

Uso
---
    python noise_feasibility_probe.py            # sondas rapidas (A, B, D)
    python noise_feasibility_probe.py --cost     # anade C (lento)

Requisitos: el stack verificado del proyecto (qiskit 2.4.2, qiskit-aer 0.17.2,
numpy 1.26.4) mas `qiskit-ibm-runtime` para la sonda A.
"""

from __future__ import annotations

import argparse
import functools
import sys
import time
import warnings
from typing import Dict, Optional

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error

warnings.filterwarnings("ignore")
print = functools.partial(print, flush=True)  # noqa: A001


# =============================================================================
# Modelos de ruido: los cuatro peldanos del eje propuesto
# =============================================================================

def noise_none() -> Optional[NoiseModel]:
    """Peldano 0 del eje: limite ideal (el de todos los resultados actuales)."""
    return None


def noise_readout(p: float = 0.03) -> NoiseModel:
    """Peldano 1: SOLO error de lectura, simetrico, igual en todos los qubits.

    Es un canal CLASICO aplicado despues de la medicion: voltea el bit
    reportado con probabilidad p. No altera el estado cuantico, y por eso
    resulta invisible a cualquier lectura de rho (ver
    `probe_faithful_degradation`, apartado d).

    Args:
        p: probabilidad de volteo del bit reportado.
    """
    nm = NoiseModel()
    nm.add_all_qubit_readout_error(ReadoutError([[1 - p, p], [p, 1 - p]]))
    return nm


def noise_full(p1: float = 1e-3, p2: float = 1e-2,
               pr: float = 0.03) -> NoiseModel:
    """Peldano 2: despolarizacion en compuertas de 1 y 2 qubits + lectura.

    Args:
        p1: probabilidad de despolarizacion en compuertas de un qubit.
        p2: idem en compuertas de dos qubits (tipicamente ~10x mayor).
        pr: probabilidad de volteo de lectura.
    """
    nm = noise_readout(pr)
    nm.add_all_qubit_quantum_error(depolarizing_error(p1, 1),
                                   ['ry', 'rz', 'sx', 'x', 'h', 'u'])
    nm.add_all_qubit_quantum_error(depolarizing_error(p2, 2),
                                   ['cx', 'cz', 'ecr'])
    return nm


def noise_backend(name: str = "FakeBrisbane") -> NoiseModel:
    """Peldano 3: modelo calibrado de un backend real de IBM.

    Args:
        name: clase de `qiskit_ibm_runtime.fake_provider` a usar.
    """
    from qiskit_ibm_runtime import fake_provider
    return NoiseModel.from_backend(getattr(fake_provider, name)())


# =============================================================================
# Sonda A — hipotesis del atajo
# =============================================================================

def probe_local_testing_mode() -> None:
    """Verifica si SamplerV2 corre local, y si DD/twirling sobreviven.

    La hipotesis a falsar es: "apuntando `qpu_cosmo_samplers.py` a un backend
    simulado en vez de a IBM se obtiene la corrida ruidosa sin tocar el
    simulador ideal".

    Lo que hay que mirar en la salida no es solo si corre, sino la advertencia
    que qiskit-ibm-runtime emite sobre las opciones que ignora.
    """
    from qiskit_ibm_runtime import Batch, Session, SamplerV2
    from qiskit_ibm_runtime.fake_provider import FakeBrisbane
    from qiskit.transpiler.preset_passmanagers import \
        generate_preset_pass_manager

    n = 3
    th = ParameterVector("t", n)
    qc = QuantumCircuit(n)
    for i in range(n):
        qc.ry(th[i], i)
    for i in range(n - 1):
        qc.cx(i, i + 1)
    qc.measure_all()

    pv = np.array([[0.1, 0.2, 0.3], [1.0, 1.1, 1.2]])
    backend = FakeBrisbane()
    pm = generate_preset_pass_manager(optimization_level=3, backend=backend)
    isa = pm.run(qc)

    for label, mode in (("mode=backend", backend),
                        ("mode=Batch", Batch(backend=backend)),
                        ("mode=Session", Session(backend=backend))):
        sampler = SamplerV2(mode=mode)
        # Exactamente las opciones que fija QPUConnection.__init__:
        opt = sampler.options
        opt.dynamical_decoupling.enable = True
        opt.dynamical_decoupling.sequence_type = "XY4"
        opt.twirling.enable_gates = True
        opt.twirling.enable_measure = True
        res = sampler.run([(isa, pv)], shots=512).result()
        data = res[0].data
        reg = getattr(data, 'meas', None) or getattr(data, 'c', None)
        c0, c1 = reg.get_counts(0), reg.get_counts(1)
        print(f"  {label:14s} OK | ISA {isa.num_qubits}q | "
              f"B=2 -> {len(c0)}/{len(c1)} resultados | "
              f"len(bitstring)={len(next(iter(c0)))}")


# =============================================================================
# Sonda B — el eje graduado es construible y distinguible
# =============================================================================

def probe_graded_axis() -> None:
    """Comprueba que los cuatro peldanos producen distribuciones distintas.

    Reporta la distancia de variacion total de cada peldano contra el ideal;
    un peldano que diera 0.0 seria indistinguible del ideal y no aportaria
    una columna real a la matriz de ablacion.
    """
    n = 4
    th = ParameterVector("t", n)
    qc = QuantumCircuit(n)
    for i in range(n):
        qc.ry(th[i], i)
    for i in range(n - 1):
        qc.cx(i, i + 1)
    for i in range(n):
        qc.ry(th[i], i)
    qc.measure_all()

    ref: Optional[Dict[str, int]] = None
    shots = 8192
    for label, nm in (("none", noise_none()), ("readout", noise_readout()),
                      ("full", noise_full()),
                      ("FakeBrisbane", noise_backend())):
        sim = AerSimulator(noise_model=nm, seed_simulator=7)
        isa = transpile(qc, sim, optimization_level=1)
        counts = sim.run(isa,
                         parameter_binds=[{p: [0.3] for p in isa.parameters}],
                         shots=shots).result().get_counts()
        if isinstance(counts, list):
            counts = counts[0]
        if ref is None:
            ref, tv = counts, 0.0
        else:
            keys = set(ref) | set(counts)
            tv = 0.5 * sum(abs(ref.get(k, 0) - counts.get(k, 0))
                           for k in keys) / shots
        print(f"  {label:14s} resultados={len(counts):3d}  "
              f"distancia de variacion total vs ideal = {tv:.4f}")


# =============================================================================
# Sonda C — costo real
# =============================================================================

def probe_cost(widths=(6, 8, 10, 12, 13)) -> None:
    """Tiempo por job y RAM pico: matriz de densidad vs trayectorias.

    Usa el ansatz real de `qpu_cosmo_samplers.build_ansatz` (3 capas), un job
    con B=2 bindings y 4096 disparos: exactamente la forma de una iteracion
    SPSA del QVMC-QPU.

    Args:
        widths: anchos de circuito (en qubits) a medir.
    """
    import gc
    import resource
    sys.path.insert(0, ".")
    from qpu_cosmo_samplers import build_ansatz

    nm = noise_backend()
    print(f"  {'n':>3} {'densidad(s)':>12} {'trayect.(s)':>12} "
          f"{'rho teorica':>14} {'RSS pico':>10}")
    for n in widths:
        qc = build_ansatz(n, n_layers=3)
        row = {}
        for method in ("density_matrix", "statevector"):
            sim = AerSimulator(noise_model=nm, method=method,
                               seed_simulator=7)
            isa = transpile(qc, sim, optimization_level=1)
            rng = np.random.default_rng(7)
            binds = [{p: list(rng.random(2)) for p in isa.parameters}]
            t0 = time.time()
            try:
                sim.run(isa, parameter_binds=binds, shots=4096).result()
                row[method] = time.time() - t0
            except Exception as exc:                       # noqa: BLE001
                row[method] = float('nan')
                print(f"      {method} fallo en n={n}: {type(exc).__name__}")
            gc.collect()
        rho_gb = (2 ** (2 * n)) * 16 / 2 ** 30
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        print(f"  {n:>3} {row['density_matrix']:>12.2f} "
              f"{row['statevector']:>12.2f} {rho_gb:>11.3f} GB "
              f"{rss:>8.0f}MB")


# =============================================================================
# Sonda D — la aceptacion FAITHFUL bajo ruido
# =============================================================================

def probe_faithful_degradation() -> None:
    """Curva de degradacion de la aceptacion Metropolis codificada en RY.

    La aceptacion se codifica como A = cos^2(theta/2) con
    theta = 2*arccos(sqrt(A)), y hoy se lee como |psi_0|^2. Bajo ruido esa
    amplitud deja de existir; la pregunta es por que la sustituimos.

    Apartados:
      (a) rho[0,0] == |psi_0|^2 sin ruido -> el cambio de lectura es exacto.
      (b) rho[0,0] == min(1, e^Delta) -> respuesta conocida, identidad intacta.
      (c) sesgo canal por canal. La despolarizacion lleva rho hacia I/2, de
          modo que el sesgo es lambda*(1/2 - A): la aceptacion se sesga HACIA
          1/2, aceptando de mas los movimientos malos y de menos los buenos.
          Es una distorsion direccional y predecible, no ruido simetrico.
      (d) el error de lectura es INVISIBLE a rho pero visible en conteos.
    """
    deltas = np.array([-6.0, -3.0, -1.0, -0.3, -0.05, 0.0, 0.5, 2.0])
    a_exact = np.minimum(1.0, np.exp(deltas))
    thetas = 2.0 * np.arccos(np.sqrt(np.clip(a_exact, 1e-12, 1.0)))

    par = ParameterVector('t', 1)
    base = QuantumCircuit(1)
    base.ry(par[0], 0)

    def via_statevector() -> np.ndarray:
        """Lectura actual del proyecto: |psi_0|^2 (solo existe sin ruido)."""
        qc = base.copy()
        qc.save_statevector()
        sim = AerSimulator(method='statevector')
        isa = transpile(qc, sim)
        res = sim.run([isa.assign_parameters({par[0]: float(t)})
                       for t in thetas]).result()
        return np.array([abs(np.asarray(res.get_statevector(k))[0]) ** 2
                         for k in range(len(thetas))])

    def via_density(nm: Optional[NoiseModel]) -> np.ndarray:
        """Lectura propuesta: rho[0,0]. Ve compuertas, NO ve lectura."""
        qc = base.copy()
        qc.save_density_matrix()
        sim = AerSimulator(method='density_matrix', noise_model=nm)
        isa = transpile(qc, sim)
        res = sim.run([isa.assign_parameters({par[0]: float(t)})
                       for t in thetas]).result()
        return np.array([
            float(np.real(np.asarray(res.data(k)['density_matrix'])[0, 0]))
            for k in range(len(thetas))])

    def via_counts(nm: Optional[NoiseModel],
                   shots: int = 100_000) -> np.ndarray:
        """Lectura por conteos: la unica que ve el canal de lectura."""
        qc = base.copy()
        qc.measure_all()
        sim = AerSimulator(noise_model=nm, seed_simulator=11)
        isa = transpile(qc, sim)
        res = sim.run([isa.assign_parameters({par[0]: float(t)})
                       for t in thetas], shots=shots).result()
        return np.array([res.get_counts(k).get('0', 0) / shots
                         for k in range(len(thetas))])

    sv, dm = via_statevector(), via_density(None)
    print("  (a) |psi_0|^2 vs rho[0,0] sin ruido : max|dif| = %.3e"
          % np.max(np.abs(sv - dm)))
    print("  (b) rho[0,0]  vs min(1,e^Delta)     : max|dif| = %.3e"
          % np.max(np.abs(dm - a_exact)))
    print()
    print("  (c) curva de degradacion:")
    print(f"      {'canal':30s} {'sesgo medio':>12s} {'sesgo rel medio':>16s}")
    for label, nm in (("ideal", noise_none()),
                      ("readout p=0.01", noise_readout(0.01)),
                      ("readout p=0.03", noise_readout(0.03)),
                      ("readout p=0.05", noise_readout(0.05)),
                      ("compuerta 1e-3 + lectura 0.03", noise_full(1e-3)),
                      ("compuerta 1e-2 + lectura 0.03", noise_full(1e-2))):
        d = via_density(nm) - a_exact
        print(f"      {label:30s} {d.mean():12.5f} "
              f"{np.mean(d / a_exact):16.5f}")
    print()
    p = 0.03
    cnt, den = via_counts(noise_readout(p)), via_density(noise_readout(p))
    print("  (d) conteos vs rho[0,0], mismo canal de lectura p=%.2f:" % p)
    print("      max|dif| = %.5f  |  ruido de disparo esperado ~ %.5f"
          % (np.max(np.abs(cnt - den)), 0.5 / np.sqrt(100_000)))
    print("      -> rho es CIEGO al error de lectura; los conteos no.")


# =============================================================================

def main(argv=None) -> int:
    """Ejecuta las sondas seleccionadas e imprime el reporte."""
    ap = argparse.ArgumentParser(
        description="Sondas de viabilidad del eje de ruido NISQ.")
    ap.add_argument('--cost', action='store_true',
                    help='incluye la sonda C (lenta)')
    args = ap.parse_args(argv)

    print("=" * 72)
    print("A. Modo de prueba local de SamplerV2 (hipotesis del atajo)")
    print("=" * 72)
    probe_local_testing_mode()

    print()
    print("=" * 72)
    print("B. Eje de ruido graduado")
    print("=" * 72)
    probe_graded_axis()

    print()
    print("=" * 72)
    print("D. Degradacion de la aceptacion FAITHFUL")
    print("=" * 72)
    probe_faithful_degradation()

    if args.cost:
        print()
        print("=" * 72)
        print("C. Costo: matriz de densidad vs trayectorias")
        print("=" * 72)
        probe_cost()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
