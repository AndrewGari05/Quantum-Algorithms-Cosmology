"""Parameterized circuits used by the quantum components.

All circuits are returned without measurements; backends add measurements
or save instructions as needed.
"""
from __future__ import annotations

__all__ = ["random_proposal_circuit", "hardware_efficient_ansatz",
           "amplitude_encoding_circuit"]


def random_proposal_circuit(n_qubits: int, n_layers: int = 3):
    """Proposal circuit of Sarracino et al. (2025), Fig. 1.

    ``H^n -> [RY RZ on every qubit, chained CRY] x n_layers -> H^n -> RY``.
    Its angles are drawn uniformly in ``[0, 2 pi)`` for every proposal.

    Examples:
        >>> random_proposal_circuit(2, 3).num_parameters
        17
    """
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector
    n_par = n_layers * n_qubits * 2 + n_layers * (n_qubits - 1) + n_qubits
    phi = ParameterVector("phi", n_par)
    qc = QuantumCircuit(n_qubits)
    qc.h(range(n_qubits))
    i = 0
    for _ in range(n_layers):
        for q in range(n_qubits):
            qc.ry(phi[i], q)
            qc.rz(phi[i + 1], q)
            i += 2
        for q in range(n_qubits - 1):
            qc.cry(phi[i], q, q + 1)
            i += 1
    qc.h(range(n_qubits))
    for q in range(n_qubits):
        qc.ry(phi[i], q)
        i += 1
    return qc


def hardware_efficient_ansatz(n_qubits: int, n_layers: int = 3):
    """Hardware-efficient ansatz with circular CX entanglement.

    ``H^n -> [RY RZ on every qubit, CX chain, CX(n-1, 0)] x n_layers -> RY``.

    Examples:
        >>> hardware_efficient_ansatz(4, 3).num_parameters
        28
    """
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector
    n_par = n_layers * n_qubits * 2 + n_qubits
    phi = ParameterVector("phi", n_par)
    qc = QuantumCircuit(n_qubits)
    qc.h(range(n_qubits))
    i = 0
    for _ in range(n_layers):
        for q in range(n_qubits):
            qc.ry(phi[i], q)
            qc.rz(phi[i + 1], q)
            i += 2
        for q in range(n_qubits - 1):
            qc.cx(q, q + 1)
        if n_qubits > 1:
            qc.cx(n_qubits - 1, 0)
    for q in range(n_qubits):
        qc.ry(phi[i], q)
        i += 1
    return qc


def amplitude_encoding_circuit():
    """One qubit ``RY(t)``: ``P(0) = cos^2(t/2)``.

    With ``t = 2 arccos(sqrt(a))`` the probability of outcome 0 is ``a``.
    """
    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter
    qc = QuantumCircuit(1)
    qc.ry(Parameter("t"), 0)
    return qc
