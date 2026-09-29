"""Execution backends for parameterized circuits.

Every quantum component in qablate talks to a backend through one method::

    backend.probabilities(circuit, parameter_values, rng, shots=None)

which returns a ``(B, 2**n)`` array: the outcome distribution of ``circuit``
for each of the ``B`` rows of ``parameter_values``. With ``shots=None`` the
distribution is exact (simulators only); with an integer it is the empirical
frequency of ``shots`` measurements. Outcomes use Qiskit's little-endian
order: bit ``q`` of index ``i`` is the result of qubit ``q``.

Columns of ``parameter_values`` follow ``circuit.parameters`` (Qiskit's
sorted order). Every random draw a backend makes (simulator seeds) comes from
the ``rng`` passed in, so results are reproducible from the caller's
generator. Exact runs make no random draws at all.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .noise import NoiseSpec

__all__ = ["Backend", "StatevectorBackend", "AerBackend", "IBMRuntimeBackend"]

_SEED_MAX = 2 ** 31 - 1


def _as_rows(parameter_values, n_params: int) -> np.ndarray:
    pv = np.asarray(parameter_values, dtype=float)
    if pv.ndim == 1:
        pv = pv[None, :] if n_params else pv.reshape(-1, 0)
    if pv.shape[1] != n_params:
        raise ValueError(f"circuit has {n_params} parameters, got rows of width {pv.shape[1]}")
    return pv


def _binds(circuit, pv: np.ndarray):
    params = list(circuit.parameters)
    if not params:
        return None
    return [{p: list(pv[:, i]) for i, p in enumerate(params)}]


def counts_to_frequencies(counts: dict[str, int], n_qubits: int) -> np.ndarray:
    """Counts dict -> frequency vector indexed by ``int(bitstring, 2)``."""
    f = np.zeros(2 ** n_qubits)
    total = 0
    for bits, c in counts.items():
        f[int(bits.replace(" ", ""), 2)] += c
        total += c
    return f / max(total, 1)


class Backend(ABC):
    """Interface shared by all backends."""

    #: False when the backend adds any noise beyond shot noise.
    noiseless: bool = True
    #: True when ``shots=None`` (exact probabilities) is supported.
    exact: bool = True
    #: Short name recorded in results.
    name: str = "backend"

    @abstractmethod
    def probabilities(self, circuit, parameter_values, rng: np.random.Generator,
                      shots: int | None = None) -> np.ndarray:
        """Outcome distribution for each parameter row, shape ``(B, 2**n)``."""

    def sample(self, circuit, parameter_values, rng: np.random.Generator,
               shots: int = 1) -> np.ndarray:
        """Measured outcomes (little-endian integers), shape ``(B, shots)``.

        The default implementation draws from :meth:`probabilities` with
        ``shots``; simulators override it to return the measured bitstrings.
        """
        f = self.probabilities(circuit, parameter_values, rng, shots=shots)
        counts = np.rint(f * shots).astype(int)
        return np.array([rng.permutation(np.repeat(np.arange(f.shape[1]), c))
                         for c in counts])


class AerBackend(Backend):
    """Qiskit Aer simulator, ideal or with a :class:`~qablate.noise.NoiseSpec`.

    Args:
        noise: Noise level; ``None`` or ``'none'`` for ideal simulation. A
            string is resolved with :meth:`NoiseSpec.from_level`.
        device: ``'CPU'`` or ``'GPU'``.

    Examples:
        >>> import numpy as np
        >>> from qiskit import QuantumCircuit
        >>> qc = QuantumCircuit(1); _ = qc.h(0)
        >>> AerBackend().probabilities(qc, np.zeros((1, 0)), np.random.default_rng(0))
        array([[0.5, 0.5]])
    """

    def __init__(self, noise=None, device: str = "CPU"):
        if noise is None or isinstance(noise, str):
            noise = NoiseSpec.from_level(noise or "none")
        self.noise = noise
        self.noiseless = noise.is_ideal
        self.name = f"aer:{noise.label}"
        self.device = device
        self._sims: dict[str, object] = {}
        self._cache: dict[tuple, object] = {}

    def _sim(self, method: str):
        if method not in self._sims:
            from qiskit_aer import AerSimulator
            opts = dict(method=method, device=self.device)
            if not self.noise.is_ideal:
                opts["noise_model"] = self.noise.noise_model
            self._sims[method] = AerSimulator(**opts)
        return self._sims[method]

    def _template(self, circuit, exact: bool):
        key = (id(circuit), exact)
        if key not in self._cache:
            import qiskit_aer  # noqa: F401  (registers the save_* circuit methods)
            qc = circuit.remove_final_measurements(inplace=False)
            if exact:
                method = "statevector" if self.noise.is_ideal else "density_matrix"
                qc.save_statevector() if self.noise.is_ideal else qc.save_probabilities()
            else:
                method = "statevector" if self.noise.is_ideal else "density_matrix"
                qc.measure_all()
            sim = self._sim(method)
            self._cache[key] = (self.noise.transpile(qc, sim), sim, circuit)
        return self._cache[key]

    def probabilities(self, circuit, parameter_values, rng, shots=None):
        pv = _as_rows(parameter_values, circuit.num_parameters)
        exact = shots is None
        t, sim, _ = self._template(circuit, exact)
        n = circuit.num_qubits
        if exact:
            # Exact probabilities involve no randomness: the caller's rng is
            # not touched, so swapping a classical component for an exact
            # quantum one leaves every other random draw unchanged.
            res = sim.run(t, parameter_binds=_binds(circuit, pv)).result()
            out = np.empty((len(pv), 2 ** n))
            phys = (t.metadata or {}).get("physical_qubits")
            for k in range(len(pv)):
                if self.noise.is_ideal:
                    out[k] = np.abs(np.asarray(res.get_statevector(k))) ** 2
                else:
                    out[k] = np.asarray(res.data(k)["probabilities"], dtype=float)
            return self.noise.apply_readout(out, n, phys) if not self.noise.is_ideal else out
        res = sim.run(t, parameter_binds=_binds(circuit, pv), shots=int(shots),
                      seed_simulator=int(rng.integers(_SEED_MAX))).result()
        return np.array([counts_to_frequencies(res.get_counts(k), n) for k in range(len(pv))])

    def sample(self, circuit, parameter_values, rng, shots: int = 1) -> np.ndarray:
        pv = _as_rows(parameter_values, circuit.num_parameters)
        t, sim, _ = self._template(circuit, False)
        res = sim.run(t, parameter_binds=_binds(circuit, pv), shots=int(shots), memory=True,
                      seed_simulator=int(rng.integers(_SEED_MAX))).result()
        return np.array([[int(b.replace(" ", ""), 2) for b in res.get_memory(k)]
                         for k in range(len(pv))])

    def statevectors(self, circuit, parameter_values, rng) -> np.ndarray:
        """Exact statevectors (ideal backends only), shape ``(B, 2**n)``."""
        if not self.noise.is_ideal:
            raise ValueError("statevectors are undefined under noise (mixed states)")
        pv = _as_rows(parameter_values, circuit.num_parameters)
        t, sim, _ = self._template(circuit, True)
        res = sim.run(t, parameter_binds=_binds(circuit, pv)).result()
        return np.array([np.asarray(res.get_statevector(k)) for k in range(len(pv))])


class StatevectorBackend(AerBackend):
    """Ideal Aer statevector simulation (alias of ``AerBackend(None)``)."""

    def __init__(self, device: str = "CPU"):
        super().__init__(None, device=device)
        self.name = "statevector"


class IBMRuntimeBackend(Backend):
    """IBM Quantum hardware (or a fake backend) through ``SamplerV2``.

    Only sampling is possible (``shots`` is required). Circuits are
    transpiled once to the device ISA and cached.

    Args:
        backend: A ``qiskit_ibm_runtime`` backend (real or fake).
        mode: Execution mode for ``SamplerV2`` (a ``Session``, ``Batch`` or the
            backend itself). Defaults to the backend.
        optimization_level: Preset pass-manager level.
    """

    exact = False

    def __init__(self, backend, mode=None, optimization_level: int = 3):
        from qiskit.transpiler import generate_preset_pass_manager
        from qiskit_ibm_runtime import SamplerV2
        self.backend = backend
        self.noiseless = False
        self.name = f"ibm:{getattr(backend, 'name', 'backend')}"
        self._pm = generate_preset_pass_manager(optimization_level=optimization_level,
                                                backend=backend)
        self._sampler = SamplerV2(mode=mode if mode is not None else backend)
        self._cache: dict[int, object] = {}

    def probabilities(self, circuit, parameter_values, rng, shots=None):
        if shots is None:
            raise ValueError("hardware backends can only sample: pass shots")
        pv = _as_rows(parameter_values, circuit.num_parameters)
        if id(circuit) not in self._cache:
            qc = circuit.remove_final_measurements(inplace=False)
            qc.measure_all()
            self._cache[id(circuit)] = self._pm.run(qc)
        isa = self._cache[id(circuit)]
        pub = (isa, pv) if circuit.num_parameters else (isa,)
        result = self._sampler.run([pub], shots=int(shots)).result()[0]
        data = result.data.meas
        n = circuit.num_qubits
        if circuit.num_parameters:
            return np.array([counts_to_frequencies(data.get_counts(k), n)
                             for k in range(len(pv))])
        return counts_to_frequencies(data.get_counts(), n)[None, :]
