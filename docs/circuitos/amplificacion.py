#!/usr/bin/env python3
"""amplificacion.py — que le falta al circuito de aceptacion para acelerar.

NO es parte de la tesis. Es una demostracion de una sola idea, para
entenderla: el circuito de aceptacion del QMCMC ya prepara el estado que
necesita el muestreo por rechazo cuantico (uno de los candidatos que Harrow
lista para su algoritmo B), y lo unico que le falta para tener aceleracion
cuadratica es la amplificacion de amplitud encima.

    python docs/circuitos/amplificacion.py

LA IDEA EN DOS RENGLONES
------------------------
Clasicamente, si aceptar tiene probabilidad A, hay que tirar la moneda ~1/A
veces para lograr una aceptacion. Cuanticamente la amplitud de "aceptar" es
un objeto que se puede ROTAR antes de medir, y rotarla hasta casi 1 cuesta
~1/sqrt(A) pasos. Ahi esta toda la aceleracion.
"""
from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator

SIM = AerSimulator()


def prepara(theta: float) -> QuantumCircuit:
    """El circuito de aceptacion del QMCMC, con 'aceptar' = |1>.

    En `cosmo_modular_quantum` se usa la convencion 'aceptar' = |0> con
    theta_A = 2*arccos(sqrt(A)); aqui se voltea a |1> nada mas para que el
    oraculo sea una Z simple y la demostracion se lea. Es el mismo estado.
    """
    qc = QuantumCircuit(1)
    qc.ry(2 * theta, 0)          # -> sqrt(1-A)|0> + sqrt(A)|1>
    return qc


def con_amplificacion(theta: float, k: int) -> QuantumCircuit:
    """El mismo circuito mas k pasos del operador de Grover.

    Cada paso es: marcar el estado bueno, deshacer la preparacion,
    reflejar sobre |0>, rehacer la preparacion. El efecto neto es una
    ROTACION de 2*theta en el plano {|0>, |1>}: se empieza en theta y se
    quiere llegar a pi/2, de ahi que hagan falta ~(pi/4)/theta pasos.
    """
    qc = prepara(theta)
    for _ in range(k):
        qc.z(0)                              # marca 'aceptar'
        qc.ry(-2 * theta, 0)                 # A^dagger
        qc.x(0), qc.z(0), qc.x(0)            # refleja sobre |0>
        qc.ry(2 * theta, 0)                  # A
        qc.global_phase += np.pi             # el signo global de Q
    return qc


def main(A: float = 0.01) -> None:
    theta = np.arcsin(np.sqrt(A))
    sv = Statevector(prepara(theta)).data

    print(f"\nA = P(aceptar) = {A}")
    print(f"tu circuito prepara:  {sv[0].real:.4f}|0>  +  {sv[1].real:.4f}|1>")
    print(f"o sea la amplitud de 'aceptar' es sqrt(A) = {np.sqrt(A):.4f}\n")

    print(f"  {'k':>3}  {'P(aceptar)':>11}   {'sin^2((2k+1)theta)':>19}")
    print(f"  {'-'*3}  {'-'*11}   {'-'*19}")
    for k in range(9):
        p = abs(Statevector(con_amplificacion(theta, k)).data[1]) ** 2
        print(f"  {k:3d}  {p:11.4f}   {np.sin((2*k+1)*theta)**2:19.4f}")

    k_opt = int(round((np.pi / 4) / theta - 0.5))
    print(f"\n  k optimo = (pi/4)/arcsin(sqrt(A)) - 1/2 = {k_opt}")
    print(f"  clasico:  ~1/A      = {1/A:.0f} intentos")
    print(f"  cuantico: ~1/sqrt(A) = {k_opt} pasos"
          f"        ->  {1/A/max(k_opt, 1):.1f}x menos")

    qc = con_amplificacion(theta, k_opt)
    qc.measure_all()
    c = SIM.run(transpile(qc, SIM), shots=20_000,
                seed_simulator=7).result().get_counts()
    print(f"\n  medido de verdad, 20000 disparos, k={k_opt}: "
          f"acepta {c.get('1', 0)/20_000:.4f} de las veces")

    print("""
  POR QUE EL CLASICO NO PUEDE HACER ESTO
  Clasicamente la moneda solo se puede TIRAR: sale cara o cruz y ya.
  Cuanticamente la amplitud es un objeto que se puede rotar ANTES de
  medir. Esa es la unica cosa que lo cuantico hace aqui, y es de donde
  sale la raiz cuadrada.

  LO QUE ESTO NO ES
  La celda de aceptacion del proyecto es FIEL a proposito: reproduce
  min(1, e^Delta) exactamente, sin amplificar, y por eso cuesta lo mismo
  que la clasica. Anadir la amplificacion la convertiria en una celda
  ALGORITMICA con aceleracion medible — otra linea de trabajo, no esta.
""")


if __name__ == '__main__':
    main()
