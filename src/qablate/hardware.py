"""One compiled circuit, three places to run it: ideal simulator, noisy twin, device.

Timing a quantum algorithm on hardware against a simulator is only
meaningful when both execute the same process. :class:`DeviceCompiler`
transpiles every logical circuit once for a target device (its native
gates, connectivity and qubit layout); :class:`DeviceBackend` then runs that
same instruction-set (ISA) circuit at one of three locations:

``ideal``
    Aer without noise (the ISA circuit, so SWAPs and native gates included).
``noisy``
    Aer with the device's calibrated noise model
    (``AerSimulator.from_backend``): the noisy twin.
``device``
    The device itself through ``SamplerV2``.

All three only sample (``shots`` is required), so the estimator is the same
everywhere: frequencies of measured bitstrings. Every call is logged as a
:class:`JobTiming`: wall time, and on the device the queue time, the
execution span and the billed quantum seconds reported by IBM.

Example (offline, with a fake device)::

    from qiskit_ibm_runtime.fake_provider import FakeFez
    compiler = DeviceCompiler(FakeFez())
    ideal = DeviceBackend(compiler, "ideal")
    noisy = DeviceBackend(compiler, "noisy")
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import asdict, dataclass

import numpy as np

from .backends import Backend, _as_rows, counts_to_frequencies

__all__ = ["LOCATIONS", "DeviceCompiler", "DeviceBackend", "JobTiming", "QuantumBudgetExceeded"]

LOCATIONS: tuple[str, ...] = ("ideal", "noisy", "device")
_SEED_MAX = 2 ** 31 - 1


class QuantumBudgetExceeded(RuntimeError):
    """Raised before a device job when the quantum-seconds budget is spent."""


@dataclass
class JobTiming:
    """Timing of one backend call (one job).

    ``wall_s`` is measured locally from submission to result. On the device,
    ``queue_s`` (created -> running), ``execution_s`` (execution span on the
    QPU) and ``quantum_s`` (billed usage) come from IBM's job metadata; on
    simulators ``execution_s`` is the simulator's own run time and the other
    two are 0.
    """

    location: str
    circuit: str
    n_qubits: int
    depth: int
    two_qubit_gates: int
    n_circuits: int
    shots: int
    compile_s: float
    wall_s: float
    queue_s: float
    execution_s: float
    quantum_s: float
    job_id: str = ""
    label: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


def _fingerprint(circuit) -> str:
    """Content hash of a circuit: gates, qubits and parameter names/values.

    Circuit names, metadata and parameter UUIDs are ignored, so two circuits
    built independently from the same recipe share a fingerprint.
    """
    h = hashlib.sha256(f"{circuit.num_qubits},{circuit.num_clbits}".encode())
    for ins in circuit.data:
        qs = [circuit.find_bit(q).index for q in ins.qubits]
        cs = [circuit.find_bit(c).index for c in ins.clbits]
        ps = [np.round(np.asarray(p, dtype=complex), 12).tobytes() if isinstance(p, np.ndarray)
              else str(p) for p in ins.operation.params]
        h.update(repr((ins.operation.name, qs, cs, ps)).encode())
    return h.hexdigest()[:16]


class DeviceCompiler:
    """Transpile logical circuits once for a device and cache the ISA circuits.

    Share one compiler between the backends of the three locations so they
    run byte-identical circuits (checked through :meth:`fingerprint`).

    Args:
        device: A ``qiskit_ibm_runtime`` backend (real or fake).
        optimization_level: Preset pass-manager level.
        seed_transpiler: Seed of the layout and routing passes.
    """

    def __init__(self, device, optimization_level: int = 3, seed_transpiler: int = 0):
        from qiskit.transpiler import generate_preset_pass_manager
        self.device = device
        self.name = getattr(device, "name", type(device).__name__)
        self.optimization_level = optimization_level
        self.seed_transpiler = seed_transpiler
        self._pm = generate_preset_pass_manager(optimization_level=optimization_level,
                                                backend=device, seed_transpiler=seed_transpiler)
        self._cache: dict[str, tuple] = {}
        self._keys: dict[int, str] = {}
        self._alive: list = []

    def compile(self, circuit):
        """``(isa_circuit, column_order, compile_seconds)``; compile time is 0 when cached.

        ``column_order`` maps the columns of ``circuit.parameters`` onto the
        parameters that survive in the ISA circuit.
        """
        # Keyed by the logical circuit's content, so identical circuits built
        # by different objects (one per location) share one ISA circuit.
        key = self._keys.get(id(circuit))
        if key is None:
            key = _fingerprint(circuit)
            self._keys[id(circuit)] = key
            self._alive.append(circuit)            # keeps id() unique while cached
        if key in self._cache:
            isa, cols, _ = self._cache[key]
            return isa, cols, 0.0
        t0 = time.perf_counter()
        qc = circuit.remove_final_measurements(inplace=False)
        qc.measure_all()
        isa = self._pm.run(qc)
        dt = time.perf_counter() - t0
        names = [p.name for p in circuit.parameters]
        cols = np.array([names.index(p.name) for p in isa.parameters], dtype=int)
        self._cache[key] = (isa, cols, dt)
        return isa, cols, dt

    def fingerprint(self, circuit) -> str:
        """Short hash of the compiled circuit (identical across locations)."""
        return _fingerprint(self.compile(circuit)[0])

    def compiled(self) -> list[dict]:
        """One record per compiled circuit: fingerprint, depth, gate counts, compile time."""
        out = []
        for key, (isa, _, dt) in self._cache.items():
            out.append({"logical": key, "isa": _fingerprint(isa), "depth": isa.depth(),
                        "two_qubit_gates": self.two_qubit_gates(isa),
                        "physical_qubits": sorted({isa.find_bit(q).index for ins in isa.data
                                                   if ins.operation.name != "barrier"
                                                   for q in ins.qubits}),
                        "compile_s": dt})
        return out

    @staticmethod
    def two_qubit_gates(isa) -> int:
        return sum(1 for ins in isa.data if ins.operation.num_qubits == 2
                   and ins.operation.name not in ("barrier", "measure"))


class DeviceBackend(Backend):
    """Run circuits compiled for a device at one of :data:`LOCATIONS`.

    Args:
        compiler: The shared :class:`DeviceCompiler`.
        location: ``'ideal'``, ``'noisy'`` or ``'device'``.
        mode: For ``'device'``: the ``SamplerV2`` execution mode (a ``Batch``
            or ``Session``); defaults to the device itself (job mode).
        max_quantum_seconds: For ``'device'``: refuse to submit a job once the
            billed quantum seconds reach this value.
        log: List that receives one :class:`JobTiming` per job (a new list by
            default, available as ``backend.log``).
    """

    exact = False

    def __init__(self, compiler: DeviceCompiler, location: str, *, mode=None,
                 max_quantum_seconds: float | None = None, log: list | None = None):
        if location not in LOCATIONS:
            raise ValueError(f"location must be one of {LOCATIONS}")
        self.compiler = compiler
        self.location = location
        self.noiseless = location == "ideal"
        self.name = f"{location}:{compiler.name}"
        self.max_quantum_seconds = max_quantum_seconds
        self.log: list[JobTiming] = log if log is not None else []
        self.label = ""                      # set by callers to tag jobs (e.g. 'spsa')
        if location == "device":
            import warnings

            from qiskit_ibm_runtime import SamplerV2
            with warnings.catch_warnings():       # deprecated from 0.50; the extra pins <0.51
                warnings.simplefilter("ignore", DeprecationWarning)
                self._sampler = SamplerV2(mode=mode if mode is not None else compiler.device)
            # Raw execution, like the simulators: no dynamical decoupling, no twirling.
            opts = self._sampler.options
            opts.dynamical_decoupling.enable = False
            opts.twirling.enable_gates = False
            opts.twirling.enable_measure = False
        else:
            from qiskit_aer import AerSimulator
            self._sim = (AerSimulator() if location == "ideal"
                         else AerSimulator.from_backend(compiler.device))

    @property
    def quantum_seconds(self) -> float:
        """Billed quantum seconds used so far by this backend."""
        return float(sum(j.quantum_s for j in self.log))

    # ------------------------------------------------------------------ #
    def _run(self, circuit, parameter_values, rng, shots: int, memory: bool):
        if shots is None or int(shots) < 1:
            raise ValueError("DeviceBackend only samples: pass shots >= 1")
        shots = int(shots)
        pv = _as_rows(parameter_values, circuit.num_parameters)
        isa, cols, compile_s = self.compiler.compile(circuit)
        pv_isa = pv[:, cols]
        n_rows = len(pv)
        queue_s = execution_s = quantum_s = 0.0
        job_id = ""
        # Drawn at every location (and ignored on the device) so the caller's
        # random stream -- SPSA directions, proposal angles -- is identical in
        # the ideal, noisy and device runs.
        sim_seed = int(rng.integers(_SEED_MAX))
        t0 = time.perf_counter()
        if self.location == "device":
            if (self.max_quantum_seconds is not None
                    and self.quantum_seconds >= self.max_quantum_seconds):
                raise QuantumBudgetExceeded(
                    f"{self.quantum_seconds:.1f} s of the {self.max_quantum_seconds:.1f} s "
                    "quantum budget used; no further device jobs are submitted")
            pub = (isa, pv_isa) if isa.num_parameters else (isa,)
            job = self._sampler.run([pub], shots=shots)
            full = job.result()
            wall_s = time.perf_counter() - t0
            job_id = job.job_id()
            queue_s, execution_s, quantum_s = _ibm_times(job, full)
            data = full[0].data.meas
            if memory:
                rows = ([data.get_bitstrings(k) for k in range(n_rows)] if isa.num_parameters
                        else [data.get_bitstrings()] * n_rows)
            else:
                rows = ([data.get_counts(k) for k in range(n_rows)] if isa.num_parameters
                        else [data.get_counts()] * n_rows)
        else:
            binds = ([{p: list(pv_isa[:, i]) for i, p in enumerate(isa.parameters)}]
                     if isa.num_parameters else None)
            nb = n_rows if isa.num_parameters else 1
            res = self._sim.run(isa, parameter_binds=binds, shots=shots, memory=memory,
                                seed_simulator=sim_seed).result()
            wall_s = time.perf_counter() - t0
            execution_s = float(getattr(res, "time_taken", wall_s) or wall_s)
            get = res.get_memory if memory else res.get_counts
            rows = [get(k) for k in range(nb)]
            if nb == 1 and n_rows > 1:
                rows = rows * n_rows
        self.log.append(JobTiming(
            location=self.location, circuit=self.compiler._keys[id(circuit)], n_qubits=circuit.num_qubits,
            depth=isa.depth(), two_qubit_gates=self.compiler.two_qubit_gates(isa),
            n_circuits=n_rows, shots=shots, compile_s=compile_s, wall_s=wall_s,
            queue_s=queue_s, execution_s=execution_s, quantum_s=quantum_s, job_id=job_id,
            label=self.label))
        return rows

    def probabilities(self, circuit, parameter_values, rng, shots=None) -> np.ndarray:
        if shots is None:
            raise ValueError("DeviceBackend only samples: pass shots (the same at every location)")
        rows = self._run(circuit, parameter_values, rng, shots, memory=False)
        return np.array([counts_to_frequencies(c, circuit.num_qubits) for c in rows])

    def sample(self, circuit, parameter_values, rng, shots: int = 1) -> np.ndarray:
        rows = self._run(circuit, parameter_values, rng, shots, memory=True)
        return np.array([[int(b.replace(" ", ""), 2) for b in r] for r in rows])


def _ibm_times(job, result) -> tuple[float, float, float]:
    """(queue, execution span, billed quantum seconds) of a runtime job; 0 when unknown."""
    queue = execution = quantum = 0.0
    try:
        m = job.metrics()
        ts = m.get("timestamps", {})
        if ts.get("created") and ts.get("running"):
            from datetime import datetime
            f = lambda s: datetime.fromisoformat(s.replace("Z", "+00:00"))  # noqa: E731
            queue = (f(ts["running"]) - f(ts["created"])).total_seconds()
        quantum = float(m.get("usage", {}).get("quantum_seconds", 0.0) or 0.0)
    except Exception:  # noqa: BLE001  (metadata is best effort; never lose the result)
        pass
    try:
        spans = result.metadata["execution"]["execution_spans"]
        execution = float(sum((s.stop - s.start).total_seconds() for s in spans))
    except Exception:  # noqa: BLE001
        pass
    return queue, execution, quantum
