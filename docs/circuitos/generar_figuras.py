"""Dibuja los SIETE circuitos del proyecto con el drawer de qiskit.

Los circuitos se construyen con el MISMO codigo que corre en la campana
(se importa `build_proposal_circuit` del modulo real) o replicando linea por
linea el constructor correspondiente, para que la figura del paper sea el
circuito que de verdad se ejecuta y no un dibujo parecido.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import UnitaryGate

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figs')
os.makedirs(OUT, exist_ok=True)

def guarda(qc, nombre, fold=-1, escala=1.0):
    fig = qc.draw('mpl', style='iqp', fold=fold, scale=escala,
                  initial_state=True)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(OUT, f'{nombre}.{ext}'),
                    bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f'  {nombre}: {qc.num_qubits} qubits, prof={qc.depth()}, '
          f'ops={dict(qc.count_ops())}')

# ── 1. QMCMC / propuesta ────────────────────────────────────────────────
import cosmo_modular_quantum as cmq
qc = cmq.build_proposal_circuit(n_qubits=3, n_layers=1)
guarda(qc, 'qmcmc_propuesta')

# ── 2. QMCMC / aceptacion (Hadamard test de un qubit) ───────────────────
par = ParameterVector('theta', 1)
qc = QuantumCircuit(1)
qc.ry(par[0], 0)
qc.measure_all()
guarda(qc, 'qmcmc_aceptacion', escala=1.4)

# ── 3. QVMC / ansatz hardware-efficient ─────────────────────────────────
n, L = 4, 1
n_p = L * n * 2 + n
phi = ParameterVector('φ', n_p)
qc = QuantumCircuit(n)
qc.h(range(n))
i = 0
for _ in range(L):
    for q in range(n):
        qc.ry(phi[i], q); i += 1
        qc.rz(phi[i], q); i += 1
    for q in range(n - 1):
        qc.cx(q, q + 1)
    qc.cx(n - 1, 0)
for q in range(n):
    qc.ry(phi[i], q); i += 1
qc.measure_all()
guarda(qc, 'qvmc_ansatz')

# ── 4. QVMC / normalizacion (QAE) ───────────────────────────────────────
n = 2
a = ParameterVector('a', 1)
qc = QuantumCircuit(n + 1)
qc.h(range(n + 1))
qc.ry(a[0], n)
guarda(qc, 'qvmc_normalizacion', escala=1.2)

# ── 5. QGA / inicializacion ─────────────────────────────────────────────
d, nb = 2, 3
qc = QuantumCircuit(d * nb)
qc.h(range(d * nb))
qc.measure_all()
guarda(qc, 'qga_init')

# ── 6. QGA / mutacion ───────────────────────────────────────────────────
m = ParameterVector('m', nb)
qc = QuantumCircuit(nb)
for q in range(nb):
    qc.ry(m[q], q)
qc.measure_all()
guarda(qc, 'qga_mutacion', escala=1.2)

# ── 7. QGA / cruce (SWAP^alpha) ─────────────────────────────────────────
alpha = 0.5
e = np.exp(1j * np.pi * alpha)
U = np.array([[1, 0, 0, 0],
              [0, (1 + e) / 2, (1 - e) / 2, 0],
              [0, (1 - e) / 2, (1 + e) / 2, 0],
              [0, 0, 0, 1]], dtype=complex)
pswap = UnitaryGate(U, label=r'SWAP$^{\alpha}$')
pa = ParameterVector('pa', nb)
pb = ParameterVector('pb', nb)
qc = QuantumCircuit(2 * nb)
for q in range(nb):
    qc.ry(pa[q], q)
    qc.ry(pb[q], nb + q)
for q in range(nb):
    qc.append(pswap, [q, nb + q])
qc.measure_all()
guarda(qc, 'qga_cruce')

print('\nlisto:', len(os.listdir(OUT)), 'archivos')


# ═════════════════════════════════════════════════════════════════════
# Figuras anadidas: el circuito REAL de LCDM y el desplazamiento de
# parametro. Las dos salieron de preguntas concretas — el dibujo generico
# de 3 qubits no correspondia a LCDM (que usa 2), y el "tercer circuito"
# del QVMC no estaba dibujado porque no es un circuito nuevo.
# ═════════════════════════════════════════════════════════════════════

# ── 1bis. QMCMC / propuesta, el circuito REAL de LCDM (n=2, L=3) ────────
qc = cmq.build_proposal_circuit(n_qubits=2, n_layers=3)
guarda(qc, 'qmcmc_propuesta_lcdm', escala=0.9)

# ── 3bis. QVMC / desplazamiento de parametro ────────────────────────────
#     El MISMO ansatz, tres veces, con phi[0] corrido +-pi/2. Se dibuja
#     pequeno (n=2, L=1) porque lo que hay que ver es que la estructura no
#     cambia: solo cambia un angulo.
def _ansatz_pequeno(desplaza_0=0.0, etiqueta=''):
    n, L = 2, 1
    qc = QuantumCircuit(n, name=etiqueta)
    qc.h(range(n))
    vals = [0.30, 1.10, 0.70, 2.40, 0.90, 1.70]
    i = 0
    for _ in range(L):
        for q in range(n):
            a = vals[i] + (desplaza_0 if i == 0 else 0.0); i += 1
            qc.ry(a, q)
            qc.rz(vals[i], q); i += 1
        for q in range(n - 1):
            qc.cx(q, q + 1)
        qc.cx(n - 1, 0)
    for q in range(n):
        qc.ry(vals[i % len(vals)], q); i += 1
    qc.measure_all()
    return qc

import matplotlib.pyplot as _plt                                  # noqa: E402
_figs = [('phi[0]', 0.0), ('phi[0] + pi/2', np.pi/2),
         ('phi[0] - pi/2', -np.pi/2)]
fig, axes = _plt.subplots(3, 1, figsize=(7.2, 6.6))
for ax, (tit, off) in zip(axes, _figs):
    _ansatz_pequeno(off).draw('mpl', style='iqp', ax=ax, initial_state=False)
    ax.set_title(tit, fontsize=10, loc='left')
fig.tight_layout()
for ext in ('pdf', 'png'):
    fig.savefig(os.path.join(OUT, f'qvmc_gradiente.{ext}'),
                bbox_inches='tight', dpi=200)
_plt.close(fig)
print('  qvmc_gradiente: el mismo ansatz tres veces, solo cambia phi[0]')
