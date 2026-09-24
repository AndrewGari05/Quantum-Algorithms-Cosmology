#!/usr/bin/env python3
"""rechazo_cuantico.py — donde SI sirve la amplificacion (y donde no).

NO es parte de la tesis. Corrige un malentendido facil de tener:

  * Sobre la celda de aceptacion del QMCMC, amplificar NO SIRVE. Ahi A ya
    se calculo CLASICAMENTE (A = min(1, e^Delta), con Delta de los dos
    log-posteriores). Teniendo A como float, una linea de numpy decide
    aceptar en O(1). Gastar ~1/sqrt(A) operaciones cuanticas para eso es
    absurdo. Ademas A cambia en CADA paso de la cadena, asi que el k
    optimo cambiaria en cada paso, y para calcularlo hace falta... A.

  * Sobre el muestreo por rechazo de Harrow SI sirve, porque ahi NUNCA se
    calcula un A suelto: se prepara una superposicion sobre los m
    candidatos y la amplitud de aceptar se cuelga de una ancila
    CONDICIONADA a cada y. La amplificacion levanta todos los buenos a la
    vez. Por eso el ahorro es sobre m/Z (cuantos candidatos hay que
    probar), no sobre una moneda.

La misma compuerta RY(2 arcsin sqrt(p)) aparece en los dos casos. Lo que
cambia es a que se aplica: a un numero conocido, o a un registro en
superposicion.

    python docs/circuitos/rechazo_cuantico.py
"""
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
np.random.seed(3)
SIM = AerSimulator()

n = 5; m = 2**n                      # m=32 candidatos de y
F = np.random.uniform(-9.0, 0.0, m)  # el log-posterior de cada candidato
p = np.exp(F)                        # prob. de aceptar cada uno (en [0,1])
Z = p.sum()
print(f"m = {m} candidatos,  Z = sum exp(F) = {Z:.3f}")
print(f"prob. media de aceptar = Z/m = {Z/m:.4f}")
print(f"CLASICO: ~m/Z = {m/Z:.1f} intentos en promedio\n")

def prepara():
    """Superposicion uniforme sobre y, y la amplitud de aceptar colgada
    de una ancila CONDICIONADA a cada y. Nunca se calcula un A suelto."""
    qc = QuantumCircuit(n+1)
    qc.h(range(n))
    for y in range(m):                       # rotacion controlada por |y>
        ang = 2*np.arcsin(np.sqrt(p[y]))
        ctrl = QuantumCircuit(1); ctrl.ry(ang, 0)
        g = ctrl.to_gate().control(n, ctrl_state=y)
        qc.append(g, list(range(n))+[n])
    return qc

sv = Statevector(prepara()).data.reshape(2, m)   # [ancila, y]
P_buena = float(np.sum(np.abs(sv[1])**2))
print(f"P(ancila=1) antes de amplificar = {P_buena:.4f}   (= Z/m)")

def amplifica(k):
    A_op = prepara()
    qc = A_op.copy()
    for _ in range(k):
        qc.z(n)                                  # marca ancila=1
        qc.compose(A_op.inverse(), inplace=True)
        qc.x(range(n+1)); qc.h(n)
        qc.mcx(list(range(n)), n); qc.h(n); qc.x(range(n+1))
        qc.compose(A_op, inplace=True)
        qc.global_phase += np.pi
    return qc

theta = np.arcsin(np.sqrt(P_buena))
k_opt = max(1, int(round((np.pi/4)/theta - 0.5)))
print(f"CUANTICO: k optimo ~ (pi/4)/arcsin(sqrt(Z/m)) = {k_opt} pasos\n")
print(f"  {'k':>3}  {'P(aceptar)':>11}")
for k in range(0, k_opt+3):
    s = Statevector(amplifica(k)).data.reshape(2, m)
    print(f"  {k:3d}  {float(np.sum(np.abs(s[1])**2)):11.4f}")
print(f"\n  {m/Z:.1f} intentos clasicos  vs  {k_opt} pasos cuanticos"
      f"   ->  {m/Z/k_opt:.1f}x")
