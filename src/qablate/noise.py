"""Noise models for simulated backends.

A :class:`NoiseSpec` is one level of the noise axis used in the ablation
studies:

``none``
    Ideal simulation.
``readout``
    Symmetric bit-flip readout error with probability ``readout_p`` on every
    qubit.
``full``
    Depolarizing gate errors (``gate_p1`` on one-qubit gates, ``gate_p2`` on
    two-qubit gates) plus the readout error above.
``<fake backend>``
    The calibrated noise model of a ``qiskit_ibm_runtime.fake_provider``
    device (for example ``"FakeBrisbane"``). Circuits are placed on a
    connected path of physical qubits and routed against the device target,
    so every two-qubit gate lands on a pair that carries its calibrated error.

Readout error is a classical channel applied after measurement. Circuits that
are measured get it from Aer; for exact probabilities of unmeasured circuits
:meth:`NoiseSpec.apply_readout` applies it in closed form.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

__all__ = ["NoiseSpec", "NAMED_LEVELS"]

NAMED_LEVELS: tuple[str, ...] = ("none", "readout", "full")
DEFAULT_READOUT_P = 0.03
DEFAULT_GATE_P1 = 1e-3
DEFAULT_GATE_P2 = 1e-2
_GATES_1Q = ("id", "u", "u1", "u2", "u3", "rx", "ry", "rz", "x", "y", "z",
             "h", "s", "sdg", "t", "tdg", "sx", "sxdg", "p")
_GATES_2Q = ("cx", "cz", "cy", "ch", "swap", "ecr", "rzz", "rxx", "ryy")


def canonical_level(level: str | None) -> str:
    """Normalize a level name: named levels lower-case, backends without ``_``.

    Examples:
        >>> canonical_level(None), canonical_level("READOUT")
        ('none', 'readout')
        >>> canonical_level("Fake_Brisbane")
        'fakebrisbane'
    """
    if not level:
        return "none"
    s = str(level).strip()
    if s.lower() in NAMED_LEVELS:
        return s.lower()
    return re.sub(r"[_\-\s]", "", s).lower()


def _fake_backend(label: str):
    try:
        from qiskit_ibm_runtime import fake_provider
    except ImportError as exc:  # pragma: no cover
        raise ImportError(f"noise level {label!r} needs qiskit-ibm-runtime") from exc
    for name in dir(fake_provider):
        if name.startswith("Fake") and canonical_level(name) == label:
            return getattr(fake_provider, name)()
    raise ValueError(f"unknown noise level {label!r}: expected one of "
                     f"{NAMED_LEVELS} or a fake_provider backend name")


def _readout_matrices(noise_model) -> tuple[dict[int, np.ndarray], np.ndarray | None]:
    """Per-qubit readout confusion matrices ``M[a, b] = P(report a | true b)``.

    An all-qubit readout error has no ``gate_qubits`` entry in ``to_dict()``;
    it is returned as the default matrix, not as qubit 0's.
    """
    per_qubit: dict[int, np.ndarray] = {}
    default = None
    for entry in noise_model.to_dict().get("errors", []):
        if entry.get("type") != "roerror":
            continue
        m = np.asarray(entry["probabilities"], dtype=float).T
        groups = entry.get("gate_qubits")
        if not groups:
            default = m
        else:
            for qubits in groups:
                per_qubit[int(qubits[0])] = m
    return per_qubit, default


def _device_path(backend, n: int) -> list[int]:
    """Deterministic connected path of ``n`` physical qubits on ``backend``."""
    adj: dict[int, set] = {}
    for a, b in backend.coupling_map.get_edges():
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    best: list[int] = []

    def dfs(path: list[int]) -> bool:
        nonlocal best
        if len(path) > len(best):
            best = list(path)
        if len(path) >= n:
            return True
        for nb in sorted(adj.get(path[-1], ())):
            if nb not in path:
                path.append(nb)
                if dfs(path):
                    return True
                path.pop()
        return False

    dfs([min(adj)])
    if len(best) < n:
        raise ValueError(f"no connected path of {n} qubits on {backend.name}")
    return best[:n]


@dataclass
class NoiseSpec:
    """One resolved level of the noise axis. Build it with :meth:`from_level`.

    Attributes:
        label: Canonical level name (``'none'``, ``'readout'``, ``'full'`` or a
            normalized backend name).
        noise_model: The Aer ``NoiseModel``, or ``None`` when ideal.
        params: The channel strengths actually used, as text.
        device: The fake backend for device levels, else ``None``.
    """

    label: str = "none"
    noise_model: object | None = None
    params: str = ""
    device: object | None = None
    _readout: dict[int, np.ndarray] = field(default_factory=dict, repr=False)
    _default_readout: np.ndarray | None = field(default=None, repr=False)
    _target: object | None = field(default=None, repr=False)

    @classmethod
    def from_level(cls, level: str | None = "none", *,
                   readout_p: float = DEFAULT_READOUT_P,
                   gate_p1: float = DEFAULT_GATE_P1,
                   gate_p2: float = DEFAULT_GATE_P2) -> NoiseSpec:
        """Resolve a level name to a :class:`NoiseSpec`.

        Examples:
            >>> NoiseSpec.from_level("none").is_ideal
            True
            >>> NoiseSpec.from_level("readout", readout_p=0.05).params
            'readout_p=0.05'
        """
        lab = canonical_level(level)
        if lab == "none":
            return cls()
        from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error
        ro = ReadoutError([[1 - readout_p, readout_p], [readout_p, 1 - readout_p]])
        device = None
        if lab == "readout":
            nm = NoiseModel()
            nm.add_all_qubit_readout_error(ro)
            params = f"readout_p={readout_p:g}"
        elif lab == "full":
            nm = NoiseModel()
            if gate_p1 > 0:
                nm.add_all_qubit_quantum_error(depolarizing_error(gate_p1, 1), list(_GATES_1Q))
            if gate_p2 > 0:
                nm.add_all_qubit_quantum_error(depolarizing_error(gate_p2, 2), list(_GATES_2Q))
            nm.add_all_qubit_readout_error(ro)
            params = f"readout_p={readout_p:g};gate_p1={gate_p1:g};gate_p2={gate_p2:g}"
        else:
            device = _fake_backend(lab)
            nm = NoiseModel.from_backend(device)
            params = f"backend={lab}"
        per_qubit, default = _readout_matrices(nm)
        return cls(label=lab, noise_model=nm, params=params, device=device,
                   _readout=per_qubit, _default_readout=default)

    @property
    def is_ideal(self) -> bool:
        """True when there is no noise of any kind."""
        return self.noise_model is None

    @property
    def has_readout(self) -> bool:
        """True when the level includes a readout channel."""
        return bool(self._readout) or self._default_readout is not None

    def readout_matrix(self, qubit: int) -> np.ndarray | None:
        """Readout confusion matrix of a physical qubit, or ``None``."""
        return self._readout.get(qubit, self._default_readout)

    # ------------------------------------------------------------------ #
    def transpile(self, circuit, simulator, **kwargs):
        """Transpile ``circuit`` for this level.

        Ideal and synthetic levels: ``qiskit.transpile(circuit, simulator)``.
        Device levels: the circuit is placed on a connected path of physical
        qubits and routed against the device target; Aer ``save_*``
        instructions are re-attached to the final physical positions of the
        logical qubits (outputs stay in logical order) and the physical
        qubits are recorded in ``metadata['physical_qubits']``.
        """
        from qiskit import transpile
        if self.device is None:
            return transpile(circuit, simulator, **kwargs)
        if self._target is None:
            from qiskit_aer import AerSimulator
            self._target = AerSimulator.from_backend(self.device)
        saves = [ci for ci in circuit.data if ci.operation.name.startswith("save_")]
        core = circuit.copy_empty_like()
        for ci in circuit.data:
            if not ci.operation.name.startswith("save_"):
                core.append(ci)
        out = transpile(core, self._target,
                        initial_layout=_device_path(self.device, circuit.num_qubits),
                        optimization_level=1, seed_transpiler=0)
        final = out.layout.final_index_layout()
        for ci in saves:
            idx = [final[circuit.find_bit(q).index] for q in ci.qubits]
            out.append(ci.operation, [out.qubits[i] for i in idx])
        out.metadata = dict(out.metadata or {}, physical_qubits=list(final))
        return out

    def apply_readout(self, probs: np.ndarray, n_qubits: int,
                      physical_qubits: list[int] | None = None) -> np.ndarray:
        """Apply the readout channel in closed form to exact probabilities.

        Args:
            probs: ``(2**n,)`` or ``(B, 2**n)`` probabilities, Qiskit
                little-endian (bit ``q`` of index ``i`` is ``(i >> q) & 1``).
            n_qubits: Register width.
            physical_qubits: Physical qubit of each logical qubit, when the
                circuit was placed on a device.
        """
        if not self.has_readout:
            return probs
        p = np.asarray(probs, dtype=float)
        batched = p.ndim == 2
        p = p if batched else p[None, :]
        if p.shape[1] != 2 ** n_qubits:
            raise ValueError(f"expected {2 ** n_qubits} probabilities, got {p.shape[1]}")
        out = p.reshape((p.shape[0],) + (2,) * n_qubits)
        for q in range(n_qubits):
            m = self.readout_matrix(physical_qubits[q] if physical_qubits else q)
            if m is None:
                continue
            axis = n_qubits - q
            out = np.moveaxis(np.moveaxis(out, axis, -1) @ m.T, -1, axis)
        out = out.reshape(p.shape[0], 2 ** n_qubits)
        return out if batched else out[0]
