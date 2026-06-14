#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qpu_cosmo_samplers.py — Cosmological inference on real IBM Quantum hardware.
================================================================================

Script EXCLUSIVELY for real QPUs via `qiskit-ibm-runtime` (SamplerV2).
It does NOT use AerSimulator anywhere: every circuit runs on the selected
physical backend (or is planned in --dry-run mode).

The physics (ΛCDM/wCDM/CPL/PEDE/GEDE models, CC and Pantheon+ likelihoods,
priors and statistics) is imported in full from `cosmo_core`, so results
are directly comparable with the simulator pipeline
(`cosmo_modular_quantum.py`).

QUANTUM-ONLY by design  [QONLY]
-------------------------------
This script is STRICTLY for dispatching the quantum algorithms (QMCMC and
QVMC) to Qiskit Runtime primitives. It deliberately runs NO classical
method: no Classical MCMC, no Classical VI. Real QPU time is scarce and
queues are long, so spending hardware sessions (or even local CPU time
inside a hardware-oriented run) on classical baselines would be wasteful
and conceptually out of place here.

Classical baselines and the full classical-vs-quantum overlay study live
in the simulator pipeline `cosmo_modular_quantum.py`, which shares the same
physics module (`cosmo_core`). The recommended workflow is therefore:

  1. Explore the whole quantumness spectrum + classical baselines on the
     simulator (`cosmo_modular_quantum.py --benchmark` / Test Run).
  2. Once a promising configuration is identified, validate it on real
     hardware here with QMCMC / QVMC only.

Figures produced here are single-method quantum diagnostics (corner plot,
KL training curve, Gelman-Rubin R̂) annotated with the run metadata
(steps / iterations / nqpp).

Key design differences vs the simulator version
-----------------------------------------------
1.  [QPU] No statevector. All quantum information comes from measured
    COUNTS. The QVMC KL is estimated over the observed support and the
    QMCMC proposal displacements come from per-qubit <Z_q>.

2.  [QPU][OPT] SPSA instead of parameter-shift for QVMC. Parameter-shift
    needs 2·n_phi evaluations per iteration (~84 circuits for 42 angles);
    SPSA always needs 2, independent of n_phi. On hardware, where each job
    costs seconds-to-minutes of queue, this cuts the per-iteration cost by
    ~40x. Both perturbed parameter sets (phi+, phi-) travel in ONE PUB
    with parameter_values of shape (2, n_phi).

3.  [QPU][OPT] QMCMC proposals pre-generated in blocks: B random angle
    sets -> ONE job with B parameter bindings -> B displacements. The
    Metropolis chain consumes them sequentially from a local queue.

4.  [QPU] The Metropolis acceptance does NOT run on the QPU by default:
    it is inherently sequential (step t depends on t-1), so on hardware it
    would cost one job + one full queue wait PER STEP. We use the
    analytically equivalent expression P(ancilla=0) = e^D/(1+e^D) that the
    Hadamard test implements (Barker rule) — identical stationary
    distribution.

5.  Error suppression via SamplerV2 options: dynamical decoupling (XY4) +
    Pauli twirling of gates and measurement.

6.  `TimingEstimator` times every real job (API overhead + queue +
    execution) and extrapolates the total cost for 100/500/1000 steps.

Usage
-----
    # Planning without spending QPU time (no IBM account needed):
    python qpu_cosmo_samplers.py --model wcdm --method qvmc --iters 30 --dry-run

    # Real run (requires a saved IBM Quantum account or --token):
    python qpu_cosmo_samplers.py --model lcdm --dataset CC --method qvmc \\
        --iters 30 --shots 4096 --least-busy --log-file qpu_run.log

References
----------
* Sarracino et al. (2025) — QMCMC proposal circuit.
* Goliath et al. (2001) — analytic M_abs marginalization (in cosmo_core).
* Spall (1998) — SPSA.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

import matplotlib
matplotlib.use('Agg')                       # headless backend (HPC-safe)
import matplotlib.pyplot as plt
import corner                               # overlaid corner plots

import cosmo_core as core
from cosmo_core import (MODELS, CosmoModel, Posterior, ess_chains,
                        fit_statistics, fmt_theta, gelman_rubin_max,
                        setup_logger)

# Qiskit (circuit construction only; NO Aer)
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

# ── Contrasting color convention shared with cosmo_modular_quantum.py ────────
C_QUANTUM   = '#d62728'   # red    — QMCMC-QPU
C_QUANTUM2  = '#ff7f0e'   # orange — QVMC-QPU

# qiskit-ibm-runtime is imported lazily: --dry-run must work on machines
# with no credentials and no network access.
_RUNTIME_IMPORTED = False


def _import_runtime():
    """Import qiskit-ibm-runtime only when real hardware is requested."""
    global _RUNTIME_IMPORTED, QiskitRuntimeService, SamplerV2, Batch, Session
    global generate_preset_pass_manager
    if _RUNTIME_IMPORTED:
        return
    from qiskit_ibm_runtime import (QiskitRuntimeService, SamplerV2, Batch,
                                    Session)  # noqa: F401
    from qiskit.transpiler.preset_passmanagers import \
        generate_preset_pass_manager  # noqa: F401
    _RUNTIME_IMPORTED = True


RNG = np.random.default_rng()


def _reseed(seed: Optional[int]):
    """Reset the global RNG (here and in cosmo_core) to a fixed seed.

    Called before each quantum method so a run is reproducible given
    --seed (chain initialization and Metropolis uniforms).
    """
    global RNG
    rng = np.random.default_rng(seed)
    RNG = rng
    core.RNG = rng


# =============================================================================
# 1.  TIMING ESTIMATOR
# =============================================================================

@dataclass
class JobRecord:
    """Record of a single job executed on the QPU."""
    n_circuits: int
    shots: int
    t_submit: float      # s — API overhead until the job is accepted
    t_queue: float       # s — queue wait (estimated)
    t_exec: float        # s — reported/measured execution time
    t_total: float       # s — total wall time (submit -> result)


class TimingEstimator:
    """Times real jobs and extrapolates the cost of long runs.

    Every primitive call is wrapped with :meth:`record`, which splits the
    wall time into (API overhead, queue, execution). If the result exposes
    QPU usage metadata it is used; otherwise execution is estimated as
    shots x circuits x t_shot with t_shot ~ 100 us (typical Eagle/Heron)
    and the remaining wall time is attributed to queue + API.
    """

    T_SHOT = 1e-4          # s per shot (Eagle/Heron order of magnitude)
    T_API_DEFAULT = 2.0    # s API overhead per job (REST + serialization)
    T_QUEUE_DEFAULT = 60.0 # s queue per job on the open plan (highly variable)

    def __init__(self):
        self.jobs: List[JobRecord] = []

    # ------------------------------------------------------------------ #
    def record(self, n_circuits: int, shots: int, t_wall: float,
               t_exec_reported: Optional[float] = None) -> JobRecord:
        """Register an executed job and decompose its wall time."""
        t_exec = (t_exec_reported if t_exec_reported is not None
                  else n_circuits * shots * self.T_SHOT)
        t_api = min(self.T_API_DEFAULT, 0.2 * t_wall)
        t_queue = max(0.0, t_wall - t_exec - t_api)
        rec = JobRecord(n_circuits, shots, t_api, t_queue, t_exec, t_wall)
        self.jobs.append(rec)
        return rec

    # ------------------------------------------------------------------ #
    @property
    def mean_wall(self) -> float:
        """Mean wall time per job (s)."""
        if not self.jobs:
            return self.T_API_DEFAULT + self.T_QUEUE_DEFAULT
        return float(np.mean([j.t_total for j in self.jobs]))

    def project(self, jobs_needed: int) -> Dict[str, float]:
        """Time projection for `jobs_needed` additional jobs."""
        if self.jobs:
            api = float(np.mean([j.t_submit for j in self.jobs]))
            queue = float(np.mean([j.t_queue for j in self.jobs]))
            execu = float(np.mean([j.t_exec for j in self.jobs]))
        else:
            api, queue, execu = (self.T_API_DEFAULT, self.T_QUEUE_DEFAULT,
                                 0.5)
        return {'api': api * jobs_needed, 'queue': queue * jobs_needed,
                'exec': execu * jobs_needed,
                'total': (api + queue + execu) * jobs_needed}

    def report(self, jobs_per_unit: float, units=(100, 500, 1000),
               unit_name: str = "iterations") -> str:
        """Estimation table for several run lengths.

        Args:
            jobs_per_unit: QPU jobs required per step/iteration
                (e.g. 1.0 for QVMC-SPSA; chains/block for QMCMC).
            units: Step/iteration counts to tabulate.
            unit_name: Unit label.
        """
        lines = [f"{'#'+unit_name:>15} | {'jobs':>6} | {'API':>9} | "
                 f"{'queue':>9} | {'QPU':>9} | {'TOTAL':>10}",
                 "-" * 72]
        for n in units:
            jobs = int(np.ceil(n * jobs_per_unit))
            p = self.project(jobs)
            lines.append(
                f"{n:>15d} | {jobs:>6d} | {_fmt_t(p['api']):>9} | "
                f"{_fmt_t(p['queue']):>9} | {_fmt_t(p['exec']):>9} | "
                f"{_fmt_t(p['total']):>10}")
        return "\n".join(lines)


def _fmt_t(seconds: float) -> str:
    """Format seconds as human-readable s/min/h."""
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds/60:.1f}min"
    return f"{seconds/3600:.1f}h"


# =============================================================================
# 2.  HARDWARE CONNECTION
# =============================================================================

class QPUConnection:
    """Encapsulates service, backend, ISA transpilation and Sampler options.

    Args:
        backend_name: Backend name (e.g. 'ibm_brisbane') or None.
        least_busy: If True, pick the least-busy operational backend.
        token: API token; if None the saved account is used.
        use_session: Session mode (iterative, sustained priority) instead
            of Batch (independent jobs, recommended for the open plan).
        shots: Default shots per circuit.
        dry_run: If True, do NOT connect to IBM; "results" are returned as
            synthetic uniform counts, only to validate the workflow and to
            feed the TimingEstimator with the default values.
    """

    def __init__(self, backend_name: Optional[str] = None,
                 least_busy: bool = True, token: Optional[str] = None,
                 use_session: bool = False, shots: int = 4096,
                 dry_run: bool = False,
                 logger: Optional[logging.Logger] = None):
        self.shots = shots
        self.dry_run = dry_run
        self.use_session = use_session
        self.log = logger or logging.getLogger("qpu")
        self.timer = TimingEstimator()
        self._context = None       # active Batch or Session
        self.backend = None
        self.sampler = None
        self.pm = None

        if dry_run:
            self.log.warning("[DRY-RUN] No IBM Quantum connection: counts "
                             "will be synthetic and timings will use the "
                             "default estimates.")
            return

        _import_runtime()
        if token:
            service = QiskitRuntimeService(channel="ibm_quantum_platform",
                                           token=token)
        else:
            service = QiskitRuntimeService()
        if backend_name:
            self.backend = service.backend(backend_name)
        elif least_busy:
            self.backend = service.least_busy(operational=True,
                                              simulator=False)
        else:
            raise ValueError("Specify --backend or --least-busy.")
        self.log.info("Backend: %s (%d qubits)", self.backend.name,
                      self.backend.num_qubits)

        # [QPU] ISA transpilation: mandatory for V2 primitives.
        self.pm = generate_preset_pass_manager(optimization_level=3,
                                               backend=self.backend)

        # Execution context: Batch groups independent jobs behind a single
        # initial queue wait; Session keeps the QPU "reserved" between jobs
        # (ideal for iterative SPSA, requires a paid plan).
        ctx_cls = Session if use_session else Batch
        self._context = ctx_cls(backend=self.backend)
        self.sampler = SamplerV2(mode=self._context)

        # [QPU] Error suppression:
        opt = self.sampler.options
        opt.dynamical_decoupling.enable = True
        opt.dynamical_decoupling.sequence_type = "XY4"
        opt.twirling.enable_gates = True
        opt.twirling.enable_measure = True
        opt.default_shots = shots
        self.log.info("Mode: %s | DD=XY4 | gate+measure twirling | "
                      "shots=%d", "Session" if use_session else "Batch",
                      shots)

    # ------------------------------------------------------------------ #
    def transpile_isa(self, qc: QuantumCircuit) -> QuantumCircuit:
        """Transpile to the backend ISA (once per template)."""
        if self.dry_run:
            return qc
        return self.pm.run(qc)

    # ------------------------------------------------------------------ #
    def run_pub(self, isa_circuit: QuantumCircuit,
                parameter_values: np.ndarray,
                shots: Optional[int] = None) -> List[Dict[str, int]]:
        """Run ONE job with one PUB (circuit, B bindings) -> list of counts.

        [QPU][OPT] Single point of contact with the hardware: ALL batching
        goes through here, so every call = exactly one job measurable by
        the TimingEstimator.

        Args:
            isa_circuit: Circuit already transpiled to ISA.
            parameter_values: Array (B, n_phi) of bindings.
            shots: Shots per binding (default: self.shots).

        Returns:
            List of B count dictionaries {bitstring: frequency}.
        """
        shots = shots or self.shots
        B = int(parameter_values.shape[0])
        t0 = time.time()

        if self.dry_run:
            # Synthetic uniform counts: enough to validate shapes,
            # decoding and logging without touching the QPU.
            n_q = isa_circuit.num_qubits
            outs = []
            for _ in range(B):
                samples = RNG.integers(0, 2**min(n_q, 20), size=shots)
                cnt: Dict[str, int] = {}
                for s in samples:
                    k = format(int(s), f'0{n_q}b')
                    cnt[k] = cnt.get(k, 0) + 1
                outs.append(cnt)
            self.timer.record(B, shots, t_wall=self.timer.T_API_DEFAULT +
                              self.timer.T_QUEUE_DEFAULT +
                              B * shots * self.timer.T_SHOT)
            return outs

        pub = (isa_circuit, parameter_values)
        job = self.sampler.run([pub], shots=shots)
        result = job.result()
        t_wall = time.time() - t0

        # Usage metrics if the backend reports them
        t_exec = None
        try:
            t_exec = float(result.metadata.get('execution',
                                               {}).get('execution_spans',
                                                       None) or 0) or None
        except Exception:
            t_exec = None
        rec = self.timer.record(B, shots, t_wall, t_exec)
        self.log.debug("Job: %d circuits x %d shots | wall %.1fs "
                       "(queue~%.1fs, exec~%.1fs)", B, shots, rec.t_total,
                       rec.t_queue, rec.t_exec)

        outs = []
        for k in range(B):
            data = result[0].data
            # The classical register is named after the circuit creg
            # ('meas' with measure_all, 'c' otherwise).
            reg = getattr(data, 'meas', None) or getattr(data, 'c', None)
            outs.append(reg.get_counts(k) if B > 1 else reg.get_counts())
        return outs

    # ------------------------------------------------------------------ #
    def close(self):
        """Close the Batch/Session if open."""
        if self._context is not None:
            try:
                self._context.close()
            except Exception:
                pass


# =============================================================================
# 3.  CIRCUITS (same topology as the simulator version)
# =============================================================================

def build_proposal_circuit(n_qubits: int, n_layers: int = 3) -> QuantumCircuit:
    """Sarracino et al. (2025) proposal circuit WITH measurement.

    Same topology as in `cosmo_modular_quantum.py`, but terminated with
    `measure_all()`: on hardware there is no statevector, so the
    displacement is reconstructed from measured <Z_q> (see
    :meth:`QPUProposalEngine._counts_to_shift`).
    """
    n_params = n_layers * n_qubits * 2 + n_layers * (n_qubits - 1) + n_qubits
    phi = ParameterVector('phi', n_params)
    qc = QuantumCircuit(n_qubits)
    qc.h(range(n_qubits))
    idx = 0
    for _ in range(n_layers):
        for q in range(n_qubits):
            qc.ry(phi[idx], q); idx += 1
            qc.rz(phi[idx], q); idx += 1
        for q in range(n_qubits - 1):
            qc.cry(phi[idx], q, q + 1); idx += 1
    qc.h(range(n_qubits))
    for q in range(n_qubits):
        qc.ry(phi[idx], q); idx += 1
    qc.measure_all()
    return qc


def build_ansatz(n_qubits: int, n_layers: int = 3) -> QuantumCircuit:
    """Hardware-efficient ansatz (RY·RZ + chained CX) WITH measurement."""
    n_p = n_layers * n_qubits * 2 + n_qubits
    phi = ParameterVector('phi', n_p)
    qc = QuantumCircuit(n_qubits)
    qc.h(range(n_qubits))
    idx = 0
    for _ in range(n_layers):
        for q in range(n_qubits):
            qc.ry(phi[idx], q); idx += 1
            qc.rz(phi[idx], q); idx += 1
        for q in range(n_qubits - 1):
            qc.cx(q, q + 1)
    for q in range(n_qubits):
        qc.ry(phi[idx], q); idx += 1
    qc.measure_all()
    return qc


# =============================================================================
# 4.  SHARED HELPERS (acceptance rule, grid encoding)
# =============================================================================

def metropolis_log_accept(lp_cur: float, lp_prop: float) -> float:
    """Classical log P(accept), analytically equal to the Hadamard test.

    The Hadamard circuit with alpha = 2·arctan(e^{D/2}) yields
    P(ancilla=0) = e^D/(1+e^D) (Barker rule). [QPU] On hardware it does
    NOT run by default: the chain is sequential and would cost one job
    with its full queue wait PER STEP. This function reproduces the exact
    same number, so the stationary distribution is identical.
    """
    if not np.isfinite(lp_prop):
        return -np.inf
    delta = np.clip(lp_prop - lp_cur, -700, 700)
    return float(delta - np.logaddexp(0.0, delta))   # log sigmoid(D)


class GridEncoding:
    """Discrete grid encoding for QVMC-QPU (index <-> theta on a 2^n grid).

    Same bit convention as the simulator: parameter i occupies qubits
    [i·nqpp, (i+1)·nqpp), with the lowest bit position as the MSB of the
    chunk (Qiskit little-endian bitstrings).
    """

    def __init__(self, model: CosmoModel, nqpp: int):
        self.model = model
        self.nqpp = nqpp
        self.d = model.n_params
        self.n_qubits = self.d * nqpp
        self.n_grid = 2 ** nqpp
        self.n_states = 2 ** self.n_qubits
        self.grids = [np.linspace(b[0], b[1], self.n_grid)
                      for b in model.sample_box]
        self.theta_table = self._build_theta_table()

    def _build_theta_table(self) -> np.ndarray:
        """Table (n_states, d): theta for every basis-state index."""
        idx = np.arange(self.n_states)
        table = np.zeros((self.n_states, self.d))
        for i in range(self.d):
            val = np.zeros(self.n_states, dtype=int)
            for j in range(self.nqpp):
                bit = (idx >> (i * self.nqpp + j)) & 1
                val |= bit << (self.nqpp - 1 - j)
            table[:, i] = self.grids[i][val]
        return table

    def build_target(self, post: Posterior) -> np.ndarray:
        """Target posterior P on the grid (vectorized, classical)."""
        log_p = post.log_prob_batch(self.theta_table)
        valid = np.isfinite(log_p)
        P = np.zeros(self.n_states)
        if np.any(valid):
            log_p[valid] -= np.max(log_p[valid])
            P[valid] = np.exp(log_p[valid])
        return P / P.sum()


def kl_from_counts(counts: Dict[str, int], P: np.ndarray) -> float:
    """KL(Q_hat || P_smoothed) over the observed support of the counts.

    With finite shots Q_hat only has support on the measured bitstrings.
    KL = sum_{x: Q_hat(x)>0} Q_hat log(Q_hat / P_s), with P_s = P smoothed
    (+eps, renormalized) to avoid log(0). Biased low w.r.t. the exact KL,
    but monotonically correlated — valid as a cost function.
    """
    tot = sum(counts.values())
    eps = 1e-10
    P_s = (P + eps)
    P_s = P_s / P_s.sum()
    kl = 0.0
    for bits, c in counts.items():
        q = c / tot
        i = int(bits.replace(" ", ""), 2)
        kl += q * np.log(q / P_s[i])
    return float(max(kl, 0.0))


def counts_theta_mean(counts: Dict[str, int],
                      theta_table: np.ndarray) -> np.ndarray:
    """E_Q[theta] from measured (or synthetic-multinomial) counts."""
    tot = sum(counts.values())
    tm = np.zeros(theta_table.shape[1])
    for bits, c in counts.items():
        tm += (c / tot) * theta_table[int(bits.replace(" ", ""), 2)]
    return tm


def spsa_gains(k: int, a0: float, c0: float) -> Tuple[float, float]:
    """Standard SPSA gain schedule (Spall 1998): a_k, c_k at iteration k."""
    return a0 / (k + 1) ** 0.602, c0 / (k + 1) ** 0.101


# =============================================================================
# 5.  QMCMC ON QPU  +  MANDATORY CLASSICAL MCMC BASELINE
# =============================================================================

class QPUProposalEngine:
    """Block-batched quantum proposals measured on real hardware.

    [QPU][OPT] B random angle sets -> ONE job (PUB with parameter_values
    of shape (B, n_phi)) -> B count distributions. Each distribution is
    converted to a d-dimensional displacement via
    <Z_q> = 1 - 2·P(q=1) in [-1, 1] per qubit, replacing the real part of
    the statevector used in simulation (inaccessible on hardware).

    Args:
        conn: Active QPU connection.
        n_phys: Dimension of the physical parameter vector theta.
        n_layers: Proposal-circuit layers.
        block: Proposals per hardware job.
        shots_per_proposal: Shots used to estimate each <Z_q>.
    """

    def __init__(self, conn: QPUConnection, n_phys: int, n_layers: int = 3,
                 block: int = 64, shots_per_proposal: int = 128):
        self.conn = conn
        self.d = n_phys
        self.n_qubits = max(2, n_phys)
        self.block = block
        self.shots = shots_per_proposal
        qc = build_proposal_circuit(self.n_qubits, n_layers)
        self.n_phi = qc.num_parameters
        self.isa = conn.transpile_isa(qc)   # [OPT] ISA transpiled once
        self._queue: List[np.ndarray] = []

    def _counts_to_shift(self, counts: Dict[str, int]) -> np.ndarray:
        """Per-qubit <Z_q> from counts -> displacement in [-1, 1]^d."""
        tot = sum(counts.values())
        p1 = np.zeros(self.n_qubits)
        for bits, c in counts.items():
            b = bits.replace(" ", "")[::-1]    # Qiskit little-endian
            for q in range(self.n_qubits):
                if b[q] == '1':
                    p1[q] += c
        z = 1.0 - 2.0 * p1 / tot
        return z[:self.d].copy()

    def _refill(self):
        """Fill the queue: one job -> `block` displacements."""
        phis = RNG.uniform(0, 2 * np.pi, size=(self.block, self.n_phi))
        all_counts = self.conn.run_pub(self.isa, phis, shots=self.shots)
        for cnt in all_counts:
            self._queue.append(self._counts_to_shift(cnt))

    def next(self) -> np.ndarray:
        """Next displacement; refills in blocks when the queue is empty."""
        if not self._queue:
            self._refill()
        return self._queue.pop()


class MCMC_QPU:
    """Metropolis-Hastings (QMCMC-QPU) whose proposals come from the QPU.

    The proposal displacements are produced by :class:`QPUProposalEngine`,
    i.e. measured on real hardware (per-qubit <Z_q> from counts). The
    acceptance uses the Barker rule, analytically equivalent to the
    Hadamard test (see :func:`metropolis_log_accept`); it runs on the CPU
    because the chain is inherently sequential and a per-step QPU job would
    waste a full queue wait on every step.

    Args:
        post: cosmo_core posterior.
        engine: Quantum proposal engine (:class:`QPUProposalEngine`).
        n_chains: Parallel chains sharing the QPU proposal queue.
        step_frac: Step scale as a fraction of the sample box.
        rhat_every: Gelman-Rubin recording cadence.
        log_every: Detailed-logging cadence.
        tag: Label used in log lines.
    """

    def __init__(self, post: Posterior, engine, n_chains: int = 4,
                 step_frac: float = 0.06, rhat_every: int = 25,
                 log_every: int = 500, tag: str = 'QMCMC-QPU',
                 logger: Optional[logging.Logger] = None):
        self.post = post
        self.model = post.model
        self.engine = engine
        self.n_chains = n_chains
        widths = np.array([hi - lo for lo, hi in self.model.sample_box])
        self.step = step_frac * widths
        self.rhat_every = rhat_every
        self.log_every = log_every
        self.tag = tag
        self.log = logger or logging.getLogger("qpu")

    # ------------------------------------------------------------------ #
    def run(self, n_steps: int, n_burn: Optional[int] = None) -> dict:
        """Run the chains. Returns a dict with chains/flat/statistics."""
        d = self.model.n_params
        n_burn = n_burn if n_burn is not None else n_steps // 10
        lo = np.array([b[0] for b in self.model.sample_box])
        hi = np.array([b[1] for b in self.model.sample_box])
        cur = lo + (hi - lo) * RNG.uniform(0.3, 0.7, size=(self.n_chains, d))
        lp = np.array([self.post.log_prob(t) for t in cur])

        chains = np.zeros((self.n_chains, n_steps, d))
        n_acc = 0
        rhat_hist: List[Tuple[int, float]] = []
        t0 = time.time()

        for s in range(n_steps):
            for c in range(self.n_chains):
                shift = self.engine.next()      # [QPU] or [BASE] classical
                prop = cur[c] + self.step * shift
                lp_prop = self.post.log_prob(prop)
                if np.log(RNG.uniform() + 1e-300) < metropolis_log_accept(
                        lp[c], lp_prop):
                    cur[c], lp[c] = prop, lp_prop
                    n_acc += 1
                chains[c, s] = cur[c]

            if (s + 1) % self.rhat_every == 0 and s > 20:
                r = gelman_rubin_max(chains[:, max(0, s // 2):s + 1, :])
                rhat_hist.append((s + 1, r))

            if (s + 1) % self.log_every == 0:
                acc = n_acc / ((s + 1) * self.n_chains)
                tm = chains[:, max(0, s // 2):s + 1, :].reshape(-1, d).mean(0)
                self.log.info("[%s] step %d/%d | accept=%.3f | "
                              "Rhat_max=%.4f | theta_mean=%s | jobs=%d",
                              self.tag, s + 1, n_steps, acc,
                              rhat_hist[-1][1] if rhat_hist else np.nan,
                              fmt_theta(self.model, tm),
                              len(self.engine.conn.timer.jobs)
                              if hasattr(self.engine, 'conn') else 0)

        flat = chains[:, n_burn:, :].reshape(-1, d)
        return {
            'chains': chains, 'flat': flat,
            'acceptance': n_acc / (n_steps * self.n_chains),
            'elapsed': time.time() - t0,
            'rhat_hist': rhat_hist,
            'ess': ess_chains(chains[:, n_burn:, :]),
        }


# =============================================================================
# 6.  QVMC ON QPU (SPSA)  +  MANDATORY CLASSICAL VI BASELINE
# =============================================================================

class QVMC_QPU:
    """Variational Quantum Monte Carlo on real hardware with SPSA.

    Encodes the discretized posterior (grid of 2^{nqpp} points per
    parameter) into |psi(phi)|^2 and minimizes KL(Q||P) with Q estimated
    from measured counts.

    [QPU][OPT] SPSA optimization (Spall 1998): the full gradient is
    estimated from ONLY 2 cost evaluations per iteration (phi + c·Delta
    and phi - c·Delta, with Delta in {-1, +1}^{n_phi} random), packed into
    ONE PUB of shape (2, n_phi) -> 1 job per iteration, independent of the
    number of ansatz parameters. Parameter-shift would have cost
    2·n_phi ~ 84 evaluations/iteration.

    Args:
        post: cosmo_core posterior.
        conn: QPU connection.
        n_qubits_per_param: Qubits per parameter (grid of 2^n points).
        n_layers: Ansatz layers.
        a0, c0: Initial SPSA gains (learning rate and perturbation).
        log_every: Logging cadence.
    """

    def __init__(self, post: Posterior, conn: QPUConnection,
                 n_qubits_per_param: int = 3, n_layers: int = 2,
                 a0: float = 0.15, c0: float = 0.1, log_every: int = 500,
                 logger: Optional[logging.Logger] = None):
        self.post = post
        self.model = post.model
        self.conn = conn
        self.enc = GridEncoding(self.model, n_qubits_per_param)
        self.P = self.enc.build_target(post)
        self.a0, self.c0 = a0, c0
        self.log_every = log_every
        self.log = logger or logging.getLogger("qpu")

        qc = build_ansatz(self.enc.n_qubits, n_layers)
        self.n_phi = qc.num_parameters
        self.isa = conn.transpile_isa(qc)   # [OPT] ISA transpiled once
        self.history: List[dict] = []

    # ── SPSA training ───────────────────────────────────────────────────── #
    def train(self, n_iters: int) -> np.ndarray:
        """Optimize phi with SPSA: 1 hardware job per iteration."""
        phi = RNG.uniform(0, 2 * np.pi, self.n_phi)
        t0 = time.time()
        for k in range(n_iters):
            ak, ck = spsa_gains(k, self.a0, self.c0)
            delta = RNG.choice([-1.0, 1.0], size=self.n_phi)
            pv = np.vstack([phi + ck * delta, phi - ck * delta])
            c_plus, c_minus = self.conn.run_pub(self.isa, pv)   # [QPU] 1 job
            f_plus = kl_from_counts(c_plus, self.P)
            f_minus = kl_from_counts(c_minus, self.P)
            ghat = (f_plus - f_minus) / (2 * ck) * delta
            phi = phi - ak * ghat

            kl_mid = 0.5 * (f_plus + f_minus)
            # theta_mean estimated from the "+" side (avoids an extra job
            # used only for logging)
            tm = counts_theta_mean(c_plus, self.enc.theta_table)
            self.history.append({'it': k + 1, 'kl': kl_mid,
                                 'theta_mean': tm})
            if (k + 1) % self.log_every == 0 or k == n_iters - 1:
                self.log.info("[QVMC-QPU] iter %d/%d | KL~%.4f | "
                              "theta_mean=%s | jobs=%d | %.0fs", k + 1,
                              n_iters, kl_mid, fmt_theta(self.model, tm),
                              len(self.conn.timer.jobs), time.time() - t0)
        self.phi_opt = phi
        return phi

    # ── final sampling ──────────────────────────────────────────────────── #
    def sample(self, n_samples: int = 4000,
               shots_per_job: int = 4096) -> np.ndarray:
        """Sample theta from the optimized circuit by measuring on the QPU."""
        samples: List[np.ndarray] = []
        pv = self.phi_opt[None, :]
        while len(samples) < n_samples:
            counts = self.conn.run_pub(self.isa, pv,
                                       shots=shots_per_job)[0]
            for bits, c in counts.items():
                th = self.enc.theta_table[int(bits.replace(" ", ""), 2)]
                samples.extend([th] * c)
        return np.array(samples[:n_samples])


# =============================================================================
# 7.  QUANTUM-ONLY FIGURES (single method, annotated with run metadata)
# =============================================================================
#
# [QONLY] This script runs no classical method, so the figures show the
# quantum result alone. Each title/legend embeds the run metadata required
# for the analysis (requirement 2): MCMC steps for QMCMC; SPSA iterations
# AND nqpp (qubits per parameter) for QVMC.

def plot_corner_quantum(flat: np.ndarray, model: CosmoModel, outdir: str,
                        tag: str, title: str, label: str,
                        color: str = C_QUANTUM,
                        weights: Optional[np.ndarray] = None) -> str:
    """Single-distribution corner plot (corner.py) of a quantum posterior.

    2D contours (1sigma/2sigma) + 1D marginals, with Planck fiducials as
    dashed black lines. `title` carries the run metadata.
    """
    rng = [(flat[:, i].min(), flat[:, i].max())
           for i in range(model.n_params)]
    fig = corner.corner(
        flat, color=color, weights=weights, labels=list(model.param_latex),
        range=rng, plot_datapoints=False, plot_density=False,
        levels=(0.393, 0.865), smooth=1.0, bins=30,
        hist_kwargs={'density': True, 'lw': 2})
    corner.overplot_lines(fig, model.fiducial, color='k', ls='--', lw=1)
    handles = [plt.Line2D([], [], color=color, lw=2, label=label),
               plt.Line2D([], [], color='k', ls='--', lw=1,
                          label='Fiducial (Planck)')]
    fig.legend(handles=handles, loc='upper right',
               bbox_to_anchor=(0.98, 0.88), fontsize=11, frameon=True)
    fig.suptitle(title, fontsize=13, fontweight='bold', y=1.02)
    f = os.path.join(outdir, f'corner_{tag}.png')
    fig.savefig(f, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return f


def plot_kl_quantum(hist: List[dict], outdir: str, tag: str,
                    n_iters: int, nqpp: int) -> str:
    """QVMC-QPU KL training curve. Title embeds SPSA iterations and nqpp."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy([h['it'] for h in hist], [max(h['kl'], 1e-12) for h in hist],
                color=C_QUANTUM2, lw=2.2, label='QVMC-QPU (hardware)')
    ax.set_xlabel('SPSA iteration')
    ax.set_ylabel(r'KL$(\hat{Q}\,\|\,P_{\rm target})$  (shot-estimated)')
    ax.set_title(f'QVMC-QPU variational training\n'
                 f'(SPSA iterations = {n_iters},  nqpp = {nqpp} '
                 f'qubits/parameter)')
    ax.grid(alpha=0.3)
    ax.legend()
    f = os.path.join(outdir, f'kl_quantum_{tag}.png')
    fig.savefig(f, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return f


def plot_rhat_quantum(rh: List[Tuple[int, float]], outdir: str, tag: str,
                      n_steps: int, n_chains: int) -> str:
    """QMCMC-QPU Gelman-Rubin curve. Title embeds total steps and chains."""
    fig, ax = plt.subplots(figsize=(8, 5))
    if rh:
        ax.semilogy([p[0] for p in rh], [max(p[1] - 1.0, 1e-6) for p in rh],
                    color=C_QUANTUM, lw=2.2, marker='o', ms=4,
                    label='QMCMC-QPU (hardware)')
    ax.axhline(0.05, color='gray', ls='--', lw=1,
               label=r'$\hat{R} = 1.05$ threshold')
    ax.set_xlabel('MCMC step')
    ax.set_ylabel(r'$\hat{R}_{\max} - 1$')
    ax.set_title(f'QMCMC-QPU convergence\n'
                 f'(total steps = {n_steps},  chains = {n_chains})')
    ax.grid(alpha=0.3)
    ax.legend()
    f = os.path.join(outdir, f'rhat_quantum_{tag}.png')
    fig.savefig(f, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return f


# =============================================================================
# 8.  CLI
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    p = argparse.ArgumentParser(
        description="Cosmological inference on real IBM Quantum hardware "
                    "(SamplerV2). QUANTUM-ONLY: runs QMCMC and/or QVMC on "
                    "the QPU and runs NO classical method. Use --dry-run to "
                    "plan without an account. For classical baselines and "
                    "the full classical-vs-quantum overlay study, use "
                    "cosmo_modular_quantum.py.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--model', choices=list(MODELS), default='lcdm',
                   help='Cosmological model')
    p.add_argument('--dataset', choices=['CC', 'Pantheon+', 'CC+Pantheon+'],
                   default='CC', help='Observational dataset')
    p.add_argument('--prior', choices=['flat', 'gaussian'], default='flat',
                   help='Prior (gaussian = Planck 2018 on Om, H0)')
    p.add_argument('--method', choices=['qvmc', 'qmcmc', 'both'],
                   default='qvmc', help='QPU inference method')
    p.add_argument('--steps', type=int, default=200,
                   help='QMCMC steps (annotated on the R-hat figure)')
    p.add_argument('--iters', type=int, default=30,
                   help='QVMC SPSA iterations (= 1 QPU job each)')
    p.add_argument('--burn', type=int, default=None,
                   help='MCMC burn-in (default: 10%% of steps)')
    p.add_argument('--chains', type=int, default=4, help='MCMC chains')
    p.add_argument('--block', type=int, default=64,
                   help='QMCMC proposals per hardware job')
    p.add_argument('--nqpp', type=int, default=3,
                   help='Qubits per physical parameter (QVMC grid)')
    p.add_argument('--layers', type=int, default=2,
                   help='Ansatz / proposal-circuit layers')
    p.add_argument('--shots', type=int, default=4096,
                   help='Shots per circuit (KL estimation and sampling)')
    p.add_argument('--samples', type=int, default=4000,
                   help='Final posterior samples per method')
    # Hardware
    p.add_argument('--backend', type=str, default=None,
                   help="Specific backend, e.g. 'ibm_brisbane'")
    p.add_argument('--least-busy', action='store_true', default=True,
                   help='Pick the least-busy operational backend')
    p.add_argument('--token', type=str, default=None,
                   help='IBM Quantum token (if no saved account)')
    p.add_argument('--session', action='store_true',
                   help='Use Session instead of Batch (paid plans)')
    p.add_argument('--dry-run', action='store_true',
                   help='Plan and validate WITHOUT connecting to IBM')
    p.add_argument('--max-jobs', type=int, default=200,
                   help='Abort if the run would require more jobs than this')
    # Output
    p.add_argument('--log-file', type=str, default=None,
                   help='Log file (default: results/qpu_<ts>.log)')
    p.add_argument('--log-every', type=int, default=500,
                   help='Progress-logging cadence')
    p.add_argument('--outdir', type=str, default='results')
    p.add_argument('--seed', type=int, default=42,
                   help='RNG seed for the QPU run')
    p.add_argument('--no-plot', action='store_true',
                   help='Skip figure generation')
    return p


def estimate_jobs(args) -> int:
    """QPU jobs the requested run will need (for the --max-jobs guard).

    [QONLY] Only the quantum methods consume hardware; this script runs no
    classical code at all.
    """
    jobs = 0
    if args.method in ('qvmc', 'both'):
        jobs += args.iters                                # 1 SPSA job/iter
        jobs += int(np.ceil(args.samples / args.shots))   # final sampling
    if args.method in ('qmcmc', 'both'):
        proposals = args.steps * args.chains
        jobs += int(np.ceil(proposals / args.block))
    return jobs


def _summary(log, side: str, model: CosmoModel, mean: np.ndarray,
             std: np.ndarray, stats: dict, extra: str):
    """Log one method's summary block (mean ± std, chi2/AIC/BIC, extras)."""
    log.info("  | %s", side)
    for n, m, s in zip(model.param_names, mean, std):
        log.info("  |   %-5s = %.4f +/- %.4f", n, m, s)
    log.info("  |   chi2=%.2f  chi2_red=%.3f  AIC=%.2f  BIC=%.2f  %s",
             stats['chi2'], stats['chi2_red'], stats['AIC'], stats['BIC'],
             extra)


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point."""
    args = build_parser().parse_args(argv)

    os.makedirs(args.outdir, exist_ok=True)
    ts = time.strftime('%Y%m%d_%H%M%S')
    log_file = args.log_file or os.path.join(args.outdir, f"qpu_{ts}.log")
    log = setup_logger(log_file=log_file, name="qpu")
    # On QPU runs, progress matters on the console too (long wall times):
    for h in log.handlers:
        if isinstance(h, logging.StreamHandler) and not isinstance(
                h, logging.FileHandler):
            h.setLevel(logging.INFO)

    log.info("=" * 70)
    log.info("QPU COSMO SAMPLERS — model=%s dataset=%s prior=%s method=%s "
             "seed=%s", args.model, args.dataset, args.prior, args.method,
             args.seed)

    # Job budget BEFORE touching the hardware
    n_jobs = estimate_jobs(args)
    log.info("Estimated QPU jobs for this run: %d (quantum-only; no "
             "classical code runs here)", n_jobs)
    if n_jobs > args.max_jobs:
        log.error("The run would require %d jobs > --max-jobs=%d. "
                  "Reduce --steps/--iters or raise --block/--max-jobs.",
                  n_jobs, args.max_jobs)
        return 1

    model = MODELS[args.model]
    post = Posterior(model, dataset=args.dataset, prior_type=args.prior)
    log.info("Posterior: %d parameters, %d data points",
             model.n_params, post.n_data)

    conn = QPUConnection(backend_name=args.backend,
                         least_busy=args.least_busy, token=args.token,
                         use_session=args.session, shots=args.shots,
                         dry_run=args.dry_run, logger=log)
    try:
        results: Dict[str, dict] = {}
        figures: List[str] = []

        # ================= QMCMC (MCMC family) ========================== #
        if args.method in ('qmcmc', 'both'):
            log.info("-" * 70)
            log.info("QMCMC-QPU: %d steps x %d chains (blocks of %d)",
                     args.steps, args.chains, args.block)
            _reseed(args.seed)
            qm = MCMC_QPU(post, QPUProposalEngine(conn, model.n_params,
                                                  n_layers=args.layers,
                                                  block=args.block),
                          n_chains=args.chains, log_every=args.log_every,
                          tag='QMCMC-QPU', logger=log)
            r_q = qm.run(args.steps, args.burn)
            st_q = fit_statistics(post, r_q['flat'].mean(0))

            log.info("QMCMC-QPU — %d steps, %d chains:",
                     args.steps, args.chains)
            _summary(log, "QMCMC-QPU (hardware)", model,
                     r_q['flat'].mean(0), r_q['flat'].std(0), st_q,
                     f"accept={r_q['acceptance']:.3f} ESS={r_q['ess']:.0f} "
                     f"jobs={len(conn.timer.jobs)}")

            if not args.no_plot:
                tag = f"mcmc_{model.name}"
                # [QONLY] single-method figures, metadata in the titles
                figures.append(plot_corner_quantum(
                    r_q['flat'], model, args.outdir, tag,
                    title=(f"{model.label} — QMCMC-QPU\n"
                           f"(steps = {args.steps}, chains = {args.chains})"),
                    label='QMCMC-QPU (hardware)', color=C_QUANTUM))
                figures.append(plot_rhat_quantum(
                    r_q['rhat_hist'], args.outdir, model.name,
                    n_steps=args.steps, n_chains=args.chains))

            st = dict(st_q)
            st['theta_best'] = np.asarray(st['theta_best']).tolist()
            results['qmcmc_qpu'] = {
                'theta_mean': r_q['flat'].mean(0).tolist(),
                'theta_std': r_q['flat'].std(0).tolist(),
                **st, 'acceptance': r_q['acceptance'], 'ess': r_q['ess'],
                'n_steps': args.steps, 'n_chains': args.chains}

        # ================= QVMC (variational family) ==================== #
        if args.method in ('qvmc', 'both'):
            log.info("-" * 70)
            log.info("QVMC-QPU: %d SPSA iterations (1 job each), nqpp=%d "
                     "-> %d qubits", args.iters, args.nqpp,
                     model.n_params * args.nqpp)
            _reseed(args.seed)
            qv = QVMC_QPU(post, conn, n_qubits_per_param=args.nqpp,
                          n_layers=args.layers, log_every=args.log_every,
                          logger=log)
            qv.train(args.iters)
            S_q = qv.sample(args.samples, shots_per_job=args.shots)
            st_q = fit_statistics(post, S_q.mean(0))

            kl_q = qv.history[-1]['kl'] if qv.history else np.nan
            log.info("QVMC-QPU — %d iterations, nqpp=%d, %d shots:",
                     args.iters, args.nqpp, args.shots)
            _summary(log, "QVMC-QPU (hardware)", model, S_q.mean(0),
                     S_q.std(0), st_q, f"final KL~{kl_q:.4f}")

            if not args.no_plot:
                tag = f"qvmc_{model.name}"
                figures.append(plot_corner_quantum(
                    S_q, model, args.outdir, tag,
                    title=(f"{model.label} — QVMC-QPU\n"
                           f"(iterations = {args.iters}, "
                           f"nqpp = {args.nqpp} qubits/parameter)"),
                    label='QVMC-QPU (hardware)', color=C_QUANTUM2))
                figures.append(plot_kl_quantum(
                    qv.history, args.outdir, model.name,
                    n_iters=args.iters, nqpp=args.nqpp))

            st = dict(st_q)
            st['theta_best'] = np.asarray(st['theta_best']).tolist()
            results['qvmc_qpu'] = {
                'theta_mean': S_q.mean(0).tolist(),
                'theta_std': S_q.std(0).tolist(), **st, 'kl_final': kl_q,
                'nqpp': args.nqpp, 'n_iters': args.iters,
                'history': [{'it': h['it'], 'kl': h['kl']}
                            for h in qv.history]}

        # ── Timing report (measured + projection 100/500/1000) ─────────── #
        log.info("=" * 70)
        log.info("MEASURED TIMINGS: %d jobs, mean wall %.1fs/job",
                 len(conn.timer.jobs), conn.timer.mean_wall)
        if args.method in ('qvmc', 'both'):
            log.info("QVMC-QPU projection (1 job per SPSA iteration):\n%s",
                     conn.timer.report(1.0, unit_name="iterations"))
        if args.method in ('qmcmc', 'both'):
            jpu = args.chains / args.block
            log.info("QMCMC-QPU projection (%.3f jobs/step with %d chains, "
                     "block %d):\n%s", jpu, args.chains, args.block,
                     conn.timer.report(jpu, unit_name="steps"))

        if figures:
            log.info("Quantum figures: %s", ", ".join(figures))
        out_json = os.path.join(args.outdir, f"qpu_results_{ts}.json")
        with open(out_json, 'w') as f:
            json.dump({'args': vars(args), 'results': results}, f, indent=2,
                      default=float)
        log.info("Results -> %s | Log -> %s", out_json, log_file)
        return 0
    finally:
        conn.close()


if __name__ == '__main__':
    sys.exit(main())
