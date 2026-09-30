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
    QPU) and ``quantum_s`` (billed usage) come from IBM's job metadata, whose
    retrieval takes ``metadata_s``; ``quantum_estimated`` is True when the
    billed time was not yet final and IBM's estimate (or the execution span,
    whichever is larger) was charged instead. On simulators ``execution_s``
    is the simulator's own run time and the device-only fields are 0.
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
    metadata_s: float = 0.0
    quantum_estimated: bool = False
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


def _drop_idle_delays(isa):
    """Remove the delays that scheduling puts on qubits the circuit never uses.

    Padding covers every device qubit; on the unused ones it only makes the
    simulators carry (and apply relaxation to) the whole device.
    """
    used = {q for ins in isa.data if ins.operation.name not in ("delay", "barrier")
            for q in ins.qubits}
    out = isa.copy_empty_like()
    for ins in isa.data:
        if ins.operation.name == "delay" and not set(ins.qubits) & used:
            continue
        out.append(ins)
    out._op_start_times = None       # the schedule of the dropped delays no longer applies
    return out


class DeviceCompiler:
    """Transpile logical circuits once for a device and cache the ISA circuits.

    Share one compiler between the backends of the three locations so they
    run byte-identical circuits (checked through :meth:`fingerprint`).

    Args:
        device: A ``qiskit_ibm_runtime`` backend (real or fake).
        optimization_level: Preset pass-manager level.
        seed_transpiler: Seed of the layout and routing passes.

    Circuits are scheduled (ALAP) and idle periods padded with delays, so the
    noisy twin applies the same idle (T1/T2) noise the device experiences.
    """

    def __init__(self, device, optimization_level: int = 3, seed_transpiler: int = 0):
        from qiskit.transpiler import generate_preset_pass_manager
        self.device = device
        self.name = getattr(device, "name", type(device).__name__)
        self.optimization_level = optimization_level
        self.seed_transpiler = seed_transpiler
        self._pm = generate_preset_pass_manager(optimization_level=optimization_level,
                                                backend=device, seed_transpiler=seed_transpiler,
                                                scheduling_method="alap")
        try:
            self.calibration_date = str(device.properties().last_update_date)
        except Exception:  # noqa: BLE001  (not every backend exposes properties)
            self.calibration_date = ""
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
        isa = _drop_idle_delays(self._pm.run(qc))
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
                        "duration_us": self.duration_us(isa),
                        "physical_qubits": sorted({isa.find_bit(q).index for ins in isa.data
                                                   if ins.operation.name != "barrier"
                                                   for q in ins.qubits}),
                        "compile_s": dt})
        return out

    def duration_us(self, isa) -> float:
        """Scheduled duration of one shot in microseconds (0 if unknown)."""
        try:
            return round(float(isa.estimate_duration(self.device.target)) * 1e6, 3)
        except Exception:  # noqa: BLE001
            return 0.0

    def compile_all(self, circuits) -> float:
        """Compile ``circuits`` ahead of time; returns the seconds spent."""
        return float(sum(self.compile(c)[2] for c in circuits))

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
        seed: Seed of the simulator seeds (also for a fake device in local mode).

    Unlike the other backends, a ``DeviceBackend`` never draws from the
    caller's ``rng``: its simulator seeds come from its own ``seed``. The
    caller's random stream (SPSA directions, proposal angles, genetic draws)
    is therefore identical at every location whatever the measurement
    outcomes are.
    """

    exact = False

    def __init__(self, compiler: DeviceCompiler, location: str, *, mode=None,
                 max_quantum_seconds: float | None = None, log: list | None = None,
                 seed: int | None = None):
        if location not in LOCATIONS:
            raise ValueError(f"location must be one of {LOCATIONS}")
        self.compiler = compiler
        self.location = location
        self.noiseless = location == "ideal"
        self.name = f"{location}:{compiler.name}"
        self.max_quantum_seconds = max_quantum_seconds
        self.log: list[JobTiming] = log if log is not None else []
        self.label = ""                      # set by callers to tag jobs (e.g. 'spsa')
        self._seeds = np.random.default_rng(seed)
        self._local = location == "device" and getattr(compiler.device, "name", "").startswith("fake")
        if location == "device":
            import warnings

            from qiskit_ibm_runtime import SamplerV2
            with warnings.catch_warnings():       # deprecated from 0.50; the extra pins <0.51
                warnings.simplefilter("ignore", DeprecationWarning)
                if mode is None and self._local:
                    # Local testing mode on the same simulator as the noisy twin
                    # (the fake backend's own local noise model rejects qubits
                    # whose stored T2 exceeds 2 T1 once delays are present).
                    from qiskit_aer import AerSimulator
                    mode = AerSimulator.from_backend(compiler.device)
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

    def reseed(self, seed) -> None:
        """Restart the simulator-seed stream (e.g. once per algorithm, so an
        algorithm's results do not depend on which ones ran before it)."""
        self._seeds = np.random.default_rng(seed)

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
        n_rows = len(pv)
        if circuit.num_parameters == 0 and n_rows != 1:
            raise ValueError("a circuit without parameters runs once: pass one row and "
                             "ask for n * shots instead of n rows")
        isa, cols, compile_s = self.compiler.compile(circuit)
        pv_isa = pv[:, cols]
        queue_s = execution_s = quantum_s = metadata_s = 0.0
        estimated = False
        job_id = ""
        sim_seed = int(self._seeds.integers(_SEED_MAX))
        t0 = time.perf_counter()
        if self.location == "device":
            if (self.max_quantum_seconds is not None
                    and self.quantum_seconds >= self.max_quantum_seconds):
                raise QuantumBudgetExceeded(
                    f"{self.quantum_seconds:.1f} s of the {self.max_quantum_seconds:.1f} s "
                    "quantum budget used; no further device jobs are submitted")
            pub = (isa, pv_isa) if isa.num_parameters else (isa,)
            if self._local:
                self._sampler.options.simulator.seed_simulator = sim_seed
            job = self._sampler.run([pub], shots=shots)
            full = job.result()
            wall_s = time.perf_counter() - t0
            job_id = job.job_id()
            t1 = time.perf_counter()
            queue_s, execution_s, quantum_s, estimated = _ibm_times(job, full, local=self._local,
                                                                   n_executions=n_rows * shots)
            metadata_s = time.perf_counter() - t1
            data = full[0].data.meas
            get = data.get_bitstrings if memory else data.get_counts
            rows = [get(k) for k in range(n_rows)] if isa.num_parameters else [get()]
        else:
            binds = ([{p: list(pv_isa[:, i]) for i, p in enumerate(isa.parameters)}]
                     if isa.num_parameters else None)
            res = self._sim.run(isa, parameter_binds=binds, shots=shots, memory=memory,
                                seed_simulator=sim_seed).result()
            wall_s = time.perf_counter() - t0
            execution_s = float(getattr(res, "time_taken", wall_s) or wall_s)
            get = res.get_memory if memory else res.get_counts
            rows = [get(k) for k in range(n_rows)]
        self.log.append(JobTiming(
            location=self.location, circuit=self.compiler._keys[id(circuit)], n_qubits=circuit.num_qubits,
            depth=isa.depth(), two_qubit_gates=self.compiler.two_qubit_gates(isa),
            n_circuits=n_rows, shots=shots, compile_s=compile_s, wall_s=wall_s,
            queue_s=queue_s, execution_s=execution_s, quantum_s=quantum_s,
            metadata_s=metadata_s, quantum_estimated=estimated, job_id=job_id, label=self.label))
        return rows

    def probabilities(self, circuit, parameter_values, rng, shots=None) -> np.ndarray:
        if shots is None:
            raise ValueError("DeviceBackend only samples: pass shots (the same at every location)")
        rows = self._run(circuit, parameter_values, rng, shots, memory=False)
        return np.array([counts_to_frequencies(c, circuit.num_qubits) for c in rows])

    def sample(self, circuit, parameter_values, rng, shots: int = 1) -> np.ndarray:
        rows = self._run(circuit, parameter_values, rng, shots, memory=True)
        return np.array([[int(b.replace(" ", ""), 2) for b in r] for r in rows])


#: Conservative billed-time model used when IBM has not finalized a job's usage.
PER_JOB_S = 2.0
PER_SHOT_US = 250.0


def conservative_quantum_seconds(n_executions: int) -> float:
    """Pessimistic billed time of one job with ``n_executions`` = circuits x shots."""
    return PER_JOB_S + n_executions * PER_SHOT_US * 1e-6


def _ibm_times(job, result, *, local: bool = False, polls: int = 10, wait_s: float = 3.0,
               n_executions: int = 0) -> tuple[float, float, float, bool]:
    """(queue, execution span, billed quantum seconds, estimated?) of a runtime job.

    IBM finalizes the billed time shortly after the result is available
    (``usage.status == 'pending'`` until then), so it is polled (at most
    ``polls * wait_s`` seconds per job). If it is still unknown, the largest
    of IBM's usage estimate, the execution span and
    :func:`conservative_quantum_seconds` is charged and flagged, so a budget
    built on it stops early rather than late. Both the current
    (``qpu_charge_time_seconds`` + ``status``) and the older
    (``quantum_seconds``) usage schemas are read.
    """
    from datetime import datetime

    def ts(s):
        return datetime.fromisoformat(str(s).replace("Z", "+00:00"))

    queue = execution = 0.0
    try:
        spans = result.metadata["execution"]["execution_spans"]
        execution = float(sum((sp.stop - sp.start).total_seconds() for sp in spans))
    except Exception:  # noqa: BLE001  (metadata is best effort; never lose the result)
        pass
    if local:
        return queue, execution, 0.0, False
    usage: dict = {}
    for i in range(max(1, polls)):
        try:
            m = job.metrics()
        except Exception:  # noqa: BLE001
            m = {}
        stamps = m.get("timestamps", {}) or {}
        if stamps.get("created") and stamps.get("running"):
            queue = (ts(stamps["running"]) - ts(stamps["created"])).total_seconds()
        usage = m.get("usage", {}) or {}
        charge = usage.get("qpu_charge_time_seconds", usage.get("quantum_seconds"))
        status = usage.get("status", "completed" if "quantum_seconds" in usage else "pending")
        if status != "pending" and charge is not None and float(charge) > 0:   # a device job never bills 0 s
            return queue, execution, float(charge), False
        if i + 1 < polls:
            time.sleep(wait_s)
    estimate = 0.0
    try:
        estimate = float(job.usage_estimation().get("quantum_seconds") or 0.0)
    except Exception:  # noqa: BLE001
        pass
    partial = float(usage.get("qpu_charge_time_seconds") or usage.get("quantum_seconds") or 0.0)
    return queue, execution, max(estimate, execution, partial,
                                 conservative_quantum_seconds(n_executions)), True
