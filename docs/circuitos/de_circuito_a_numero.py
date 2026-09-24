#!/usr/bin/env python3
"""de_circuito_a_numero.py — el puente, desde el caso mas pequeno posible.

Un circuito clasico no existe: una funcion clasica toma un numero y devuelve
un numero. Un circuito cuantico tambien toma numeros (los angulos) y tambien
acaba devolviendo numeros, pero en medio pasa por un objeto que NO se puede
leer. Este script hace visible esa cadena con un qubit, una compuerta, y
conteos de verdad.

    python docs/circuitos/de_circuito_a_numero.py
"""
from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator

SIM = AerSimulator()
RAYA = '=' * 70


def paso(n, titulo):
    print(f"\n{n}) {titulo}")


# ═════════════════════════════════════════════════════════════════════
def un_qubit(theta=1.0472):
    print(RAYA)
    print("  UN QUBIT, UNA COMPUERTA")
    print(RAYA)

    paso(1, f"ENTRA:  theta = {theta} rad")
    print("     Un float. Igual que el argumento de cualquier funcion.")

    qc = QuantumCircuit(1)
    qc.ry(theta, 0)
    paso(2, "EL CIRCUITO:  RY(theta) aplicado a |0>")

    a0, a1 = Statevector(qc).data
    paso(3, "EL ESTADO — existe, pero NO se puede leer")
    print(f"     |psi> = {a0.real:.4f}|0> + {a1.real:.4f}|1>")
    print(f"     que es  cos(theta/2)|0> + sin(theta/2)|1>")

    paso(4, "LA DISTRIBUCION — tampoco se lee; es lo que el circuito ES")
    print(f"     P(0) = |{a0.real:.4f}|^2 = {abs(a0)**2:.4f}")
    print(f"     P(1) = |{a1.real:.4f}|^2 = {abs(a1)**2:.4f}")
    print("     Aqui esta la diferencia con lo clasico: la salida del")
    print("     circuito NO es un numero, es una distribucion.")

    qc.measure_all()
    qc_t = transpile(qc, SIM)
    paso(5, "LO QUE SI SE VE: medir N veces y contar")
    for N in (10, 100, 1000, 100_000):
        c = SIM.run(qc_t, shots=N, seed_simulator=7).result().get_counts()
        n0 = c.get('0', 0)
        print(f"     N={N:>7}:  {n0:>6} ceros  ->  n0/N = {n0/N:.4f}")
    print("     La frecuencia converge a P(0) con error ~ 1/sqrt(N).")

    paso(6, f"SALE: el numero {abs(a0)**2:.4f}")
    print(f"     La funcion que el circuito calculo fue: theta -> cos^2(theta/2)")
    print(f"     Comprobacion clasica: cos^2({theta}/2) = {np.cos(theta/2)**2:.6f}")


# ═════════════════════════════════════════════════════════════════════
def al_reves(A=0.35):
    print("\n" + RAYA)
    print("  AL REVES: si quiero que salga A, ¿que angulo meto?")
    print(RAYA)
    th = 2 * np.arccos(np.sqrt(A))
    print(f"\n   Quiero P(0) = A = {A}")
    print(f"   Despejo de cos^2(theta/2) = A:")
    print(f"       theta = 2*arccos(sqrt(A)) = {th:.6f} rad")
    qc = QuantumCircuit(1)
    qc.ry(th, 0)
    qc.measure_all()
    c = SIM.run(transpile(qc, SIM), shots=100_000,
                seed_simulator=7).result().get_counts()
    n0 = c.get('0', 0)
    print(f"   Corro 100000 veces: {n0} ceros -> {n0/100_000:.4f}  (queria {A})")
    print("\n   ESO ES, LITERALMENTE, EL CIRCUITO DE ACEPTACION DEL QMCMC.")
    print("   Con A = min(1, e^Delta), medir 0 ES tirar la moneda de")
    print("   Metropolis. Por eso ese componente es FIEL: no aproxima la")
    print("   regla clasica, la calcula.")


# ═════════════════════════════════════════════════════════════════════
def dos_qubits():
    print("\n" + RAYA)
    print("  DOS QUBITS: lo mismo, pero la distribucion tiene 4 casillas")
    print(RAYA)
    qc = QuantumCircuit(2)
    qc.ry(1.0472, 0)
    qc.ry(0.6435, 1)
    qc.cx(0, 1)
    sv = Statevector(qc).data
    print("\n   ENTRA: dos angulos (1.0472, 0.6435)")
    print("   ESTADO (no se lee):")
    for i, a in enumerate(sv):
        print(f"       amplitud de |{i:02b}> = {a.real:+.4f}")
    print("   DISTRIBUCION (no se lee):")
    for i, a in enumerate(sv):
        print(f"       P({i:02b}) = {abs(a)**2:.4f}")
    qc.measure_all()
    c = SIM.run(transpile(qc, SIM), shots=100_000,
                seed_simulator=7).result().get_counts()
    print("   LO QUE SE VE (100000 disparos):")
    for k in sorted(c):
        print(f"       {k}: {c[k]:>6}  ->  {c[k]/100_000:.4f}")
    print("\n   Con n qubits hay 2^n casillas. El circuito sigue siendo una")
    print("   maquina que reparte probabilidad entre cadenas de bits; lo")
    print("   unico que cambia es cuantas cadenas hay.")


# ═════════════════════════════════════════════════════════════════════
def reglas():
    print("\n" + RAYA)
    print("  LAS REGLAS DE LECTURA DEL PROYECTO")
    print(RAYA)
    print("""
   Ya visto lo anterior, cada componente del marco es solo una eleccion
   de QUE numero se extrae de esa distribucion:

     QMCMC aceptacion  -> P(0)                    un escalar
     QMCMC propuesta   -> Re(psi_k)*sign(Im psi_k)   d numeros
                          (o <Z_q> = 1 - 2P(q=1) por la ruta de cuentas)
     QVMC ansatz       -> el vector |psi_x|^2 entero  (la distribucion Q)
     QGA mutacion      -> la cadena de bits medida    (el gen)
     QGA cruce         -> la cadena de bits medida    (el hijo)

   Eso es TODO el puente. Entran angulos, el circuito define una
   distribucion sobre cadenas de bits, y una regla fija convierte esa
   distribucion (o una muestra de ella) en el numero que el algoritmo
   clasico estaba esperando.
""")


if __name__ == '__main__':
    un_qubit()
    al_reves()
    dos_qubits()
    reglas()
