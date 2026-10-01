#!/usr/bin/env python3
""" cosmo_hpc_runner.py — Parallel HPC orchestrator (generic, multi-node)

 Designed to run on a SUPERCOMPUTER / compute node: it auto-detects the cores
 and RAM available on whatever node it lands on and distributes the work
 accordingly. Nothing is hard-wired to a specific machine.

 WHAT IT DOES
 ------------
 Runs the project's two pipelines IN PARALLEL:

     1) cosmo_modular_quantum.py    (QMCMC + QVMC, quantumness ladder)
     2) cosmo_genetic_optimizers.py (CGA + QGA, global optimization / MAP)

 without modifying a single line of those scripts: it invokes their existing,
 tested CLI (--sweep-all --sweep-models <model>), ONE MODEL PER PROCESS, so
 that each (script x model) combination is an independent task running in its
 own Python interpreter.

 WHY multiprocessing (subprocess) AND NOT multithreading
 -------------------------------------------------------
 * Python's GIL serializes all pure-Python code (the MCMC loop, the GA
   generational loop, circuit construction). Threads give you NO real
   parallelism for that part. Separate processes = separate GILs = real
   parallelism + crash isolation (if one model blows up, the rest continue).
 * The heavy compute (NumPy/BLAS and Qiskit-Aer in C++) is ALREADY
   multi-threaded internally via OpenMP. The real HPC risk is not "too few
   threads", it is OVERSUBSCRIPTION: if you launch W processes and each lets
   its BLAS/Aer grab all 80 cores, you end up with W*80 threads fighting over
   80 cores -> cache thrashing and the "parallel" version runs SLOWER.

 THE KEY: PARTITION THE CORES
 ----------------------------
 This orchestrator fixes, BEFORE starting each subprocess, the number of
 internal BLAS/OpenMP/Aer threads via environment variables (OMP_NUM_THREADS,
 etc.). With J concurrent processes and T threads each, it enforces J*T ~=
 total cores. Those env vars are read when NumPy/Qiskit are imported, which is
 why they must be set in a NEW interpreter (subprocess), not in threads of the
 same process.

 WHAT IT RETURNS
 ---------------
 It measures, per task and in aggregate: wall time and peak RAM of the process
 TREE (child process + its descendants), prints it as a table and saves it to
 master_profile.csv / .json. It also passes --profile to each child so each
 one produces its own resource_usage_*.png / profile_*.json. When a child
 wrote a profile_*.json, the orchestrator reports the MEASURED memory from
 cosmo_profiling rather than its own sampled estimate.

 TYPICAL USE ON A COMPUTE NODE
 -----------------------------
     # auto-detects the node's cores/RAM
     python cosmo_hpc_runner.py \
         --dataset CC+BAO+Pantheon+ \
         --steps 15000 --qvmc-iter 3000 --nqpp 3 \
         --generations 120 --population-size 200 --n-bits 6 \
         --threads-per-worker 8

 GRID SWEEP (convergence study)
 ------------------------------
     # one task per nqpp in {2,3,4,5} per model (QVMC/QMCMC):
     python cosmo_hpc_runner.py --nqpp-sweep 2 5 --only-samplers \
         --models lcdm wcdm --dataset CC+BAO+Pantheon+
     # and for the genetic optimizer (QGA), sweep the grid size n_bits:
     python cosmo_hpc_runner.py --nbits-sweep 3 6 --only-genetic --models lcdm

 REMEMBER: nqpp does NOT depend on the core count, it depends on RAM and on d
 (the number of parameters): the grid costs 2^(nqpp*d). Combinations exceeding
 --max-qubits are clamped down per model (or skipped with --strict-qubits);
 raise --max-qubits only if the RAM can take it
 (18q~=3.5GB . 20q~=14GB . 22q~=56GB . 24q~=224GB).

 Samplers only / genetic only:  --only-samplers  /  --only-genetic
 Dry run (see what it would launch without running):  --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cosmo_noise as cnoise

# psutil is a project dependency (cosmo_profiling). If missing, we degrade to
# /proc so we do not break the process-tree memory measurement.
try:
    import psutil
    _PSUTIL = True
except Exception:
    _PSUTIL = False


# --- Model geometry: d = number of parameters (for the RAM budget) ----------
# lcdm/pede=2, wcdm/gede=3, cpl=4  (matches the README memory table)
MODEL_DIM: Dict[str, int] = {
    'lcdm': 2, 'pede': 2, 'wcdm': 3, 'gede': 3, 'cpl': 4,
}
ALL_MODELS = list(MODEL_DIM)

# --- Per-state memory cost: DIFFERENT for the two pipelines -----------------
#
# [FIX] These used to be a single constant applied to both pipelines, which
# over-estimated the QGA by ~830x and made the clamp needlessly lower n_bits
# for 4-parameter models (e.g. CPL was silently dropped from n_bits 6 to 4,
# i.e. from a 64-level grid per axis to 16, to "save" memory that was never
# going to be used).
#
# SAMPLERS (cosmo_modular_quantum.py — QVMC/VI):
#   The grid is 2^(nqpp*d) states AND the likelihood builds auxiliary arrays
#   of shape (n_states, N_data) over it.
#
#   [B-MEM] The first version used ONE constant, 1660*8 = 13.3 kB/state,
#   inherited from the validation in cosmo_modular_quantum.py (_validate_args).
#   It is a design number that was NEVER calibrated against a measurement, and
#   the error is structural, not a calibration issue: the original comment said
#   the 1660 floats/state WERE the (n_states, N_data) arrays, but 1660 is a
#   constant — it does not depend on N_data. In other words, the model only
#   captured the dataset-independent part and dropped the dominant term.
#
#   Real consequence, measured in the CC+BAO+Pantheon campaign of 2026-08-31:
#   the planner estimated 3.7 GB for 18 qubits, 17.9–19.6 GB were measured
#   (5.3x), and three tasks (cpl/nqpp5 at 20q, gede/nqpp7 and wcdm/nqpp7 at
#   21q) hit the OOM killer — rc=-9, SIGKILL, no traceback or partial results.
#
#   Recalibrated against 26 peak-RSS measurements from two campaigns with
#   datasets of very different size (same machine, same code, clean
#   subprocess):
#
#       total_q    N_data=51 (CC+BAO)     N_data=1099 (CC+BAO+Pantheon)
#          12          --                      521 MB
#          14          --                     1157 MB
#          15          --                     2556 MB
#          16         1000 MB                 4822 MB
#          18         4036 MB                19586 MB
#          20          --                    >59580 MB  (OOMKill)
#          21          --                    >56366 MB  (OOMKill)
#
#   Subtracting the process baseline and dividing by 2^q, the per-state cost
#   converges from below to 14.4 kB with N_data=51 and to 73.7 kB with
#   N_data=1099. The two points fix a straight line in N_data, which is exactly
#   the shape predicted by the (n_states, N_data) array:
#
#       bytes/state = 11500 + 57 * N_data
#
#   The 57 B/datum are ~7 float64 per data point and per state (residual,
#   model, partial chi2 and their temporary copies); the fixed 11.5 kB are the
#   grid and its dataset-independent auxiliaries — and they are, incidentally,
#   the only thing the old constant captured.
#
#   (It also covers the QVMC training batch, which materializes 2*n_phi
#   statevectors at 16 B/state: smaller than the auxiliaries for every nqpp
#   the cap allows, so this model is still the limiting one.)
BYTES_PER_STATE_SAMPLERS_FIXED = 11_500
BYTES_PER_STATE_SAMPLERS_PER_DATUM = 57

#: Margin over the fit. The fit reproduces the measurements from above over
#: the whole useful range (12–18 q, both datasets), but a peak RSS is a sampled
#: high-water mark: the true maximum may be higher than the one observed.
#: Erring low here means a SIGKILL with no results; erring high, one fewer
#: task in parallel.
SAMPLERS_MEM_SAFETY = 1.15

#: Approximate N_data per dataset, ONLY for planning. The planner cannot load
#: the real data without dragging numpy/astropy into the parent process — that
#: was bug [B-PLAN] — so it sticks with this table.
DATASET_N_DATA: Dict[str, int] = {
    'CC+BAO': 51,
    'CC+BAO+Pantheon': 1099,
    'CC+BAO+Pantheon+': 1752,
}
#: Fallback for an unknown dataset: the largest in the table. Overestimating
#: is the safe direction.
DEFAULT_PLAN_N_DATA = max(DATASET_N_DATA.values())


def bytes_per_state_samplers(n_data: Optional[int] = None) -> float:
    """Memory cost per grid state for a samplers task.

    Args:
        n_data: number of data points of the active dataset. `None` uses
            `DEFAULT_PLAN_N_DATA` (the largest dataset in the table), which is
            the conservative direction.

    Returns:
        Bytes per state, safety margin included.
    Examples:
        The per-state cost depends on the DATASET, it is not a constant — that
        was bug [B-MEM]. With 51 points a state costs 16 kB; with 1099, 84 kB:

        >>> round(bytes_per_state_samplers(51) / 1024)
        16
        >>> round(bytes_per_state_samplers(1099) / 1024)
        83
    """
    n = DEFAULT_PLAN_N_DATA if n_data is None else int(n_data)
    return ((BYTES_PER_STATE_SAMPLERS_FIXED
             + BYTES_PER_STATE_SAMPLERS_PER_DATUM * n) * SAMPLERS_MEM_SAFETY)


def dataset_n_data(dataset: Optional[str]) -> int:
    """N_data of the dataset for planning, with a conservative fallback.

    Args:
        dataset: name of the dataset (or alias) whose size is looked up.

    Returns:
        int
    Examples:
        >>> dataset_n_data('CC+BAO')
        51
        >>> dataset_n_data('CC+BAO+Pantheon')
        1099

        An unknown dataset uses the largest in the table, which is the safe
        direction: overestimating costs one fewer task in parallel,
        underestimating costs an OOMKill.

        >>> dataset_n_data('CC+BAO+DESI') == DEFAULT_PLAN_N_DATA
        True
    """
    if not dataset:
        return DEFAULT_PLAN_N_DATA
    return DATASET_N_DATA.get(dataset, DEFAULT_PLAN_N_DATA)


#: Backward-compatible alias: the value the model gives for the largest
#: dataset. Any external script that imported the old name still receives a
#: number of the same kind, now calibrated.
BYTES_PER_STATE_SAMPLERS = bytes_per_state_samplers()
#
# GENETIC (cosmo_genetic_optimizers.py — QGA):
#   The QGA never evaluates the likelihood over the grid: fitness is computed
#   on the POPULATION (pop_size x d), not on 2^(n_bits*d) points. The only
#   d-dependent circuit is the quantum initialization (d*n_bits qubits in
#   superposition, measured); mutation uses n_bits qubits and crossover
#   2*n_bits, both independent of d. So the cost is one plain statevector:
#   one complex128 amplitude per state = 16 bytes.
#   Verified by measurement (peak RSS high-water mark, clean subprocess,
#   H^n + measure_all on the Aer statevector method):
#       n=20 -> 17.5 MB measured vs 16.8 MB theoretical
#       n=22 -> 65.6 MB measured vs 67.1 MB theoretical
#       n=24 -> 257.8 MB measured vs 268.4 MB theoretical
#   i.e. it converges to 16 B/state from below.
BYTES_PER_STATE_GENETIC = 16
#
# Fixed per-process cost (Python interpreter + numpy/scipy/qiskit/matplotlib
# imports + the module-level data load), which the estimate previously ignored
# entirely. With J processes in flight this is J*250 MB of real RAM that the
# aggregate budget must account for — on a 10-way split that is 2.5 GB.
# Calibrated end-to-end against real child runs (peak RSS of the process tree
# minus the statevector term):
#     genetic/lcdm, 12q: 260 MB peak - 0.1 MB statevector -> ~260 MB baseline
#     genetic/cpl,  24q: 510 MB peak - 268 MB statevector -> ~242 MB baseline
PROCESS_BASELINE_MB = 250.0

# Kind -> per-state cost, so callers can stay declarative. The samplers one
# depends on N_data, so it is resolved with the function below and not with
# the dictionary (which remains for the 'n_bits' case and for compatibility).
BYTES_PER_STATE_BY_KIND = {
    'nqpp': BYTES_PER_STATE_SAMPLERS,     # samplers tasks (default dataset)
    'n_bits': BYTES_PER_STATE_GENETIC,    # genetic tasks
}


def bytes_per_state(kind: str = 'nqpp', n_data: Optional[int] = None) -> float:
    """Per-state cost of the `kind` pipeline.

    The genetic optimizer does not touch the likelihood over the grid, so its
    cost does not depend on the dataset; the samplers one does — see [B-MEM]
    above.

    Args:
        kind: pipeline: 'nqpp' (samplers) or 'n_bits' (genetic). Default
            'nqpp'.
        n_data: number of data points of the dataset. Default None.

    Returns:
        float
    """
    if kind == 'n_bits':
        return float(BYTES_PER_STATE_GENETIC)
    return bytes_per_state_samplers(n_data)
#
# NOISY (third memory model — second ablation axis):
#   Simulating with noise requires a density matrix: 2^(2n) complex amplitudes
#   instead of 2^n, i.e. 16 * 4^n bytes. It does NOT REPLACE the two models
#   above, it is ADDED to them: the samplers task still builds its likelihood
#   grid (1660 floats/state) and additionally keeps rho.
#
#   The rho term dominates as soon as n grows:
#       n=10 ->  16.0 MB of rho   versus  13.6 MB of grid
#       n=12 -> 256.0 MB of rho   versus  54.4 MB of grid
#       n=13 ->   1.0 GB of rho   versus 108.9 MB of grid
#
#   But the REAL limiting factor of this axis is not memory but TIME, which is
#   why the ceiling (cosmo_noise.MAX_NOISY_QUBITS = 13) is an explicit constant
#   and not a value derived from the detected RAM like the other two. Measured
#   with the real ansatz (3 layers, B=2 bindings, 4096 shots, FakeBrisbane):
#       n=10 ->   3.8 s/job      n=12 ->  34.4 s/job      n=13 -> 216.5 s/job
#   At n=14 the density matrix is 4.0 GB and ~14 min per job; a machine with
#   more RAM does not move that limit, it only makes it more expensive.
BYTES_PER_STATE_NOISY_NOTE = (
    "rho = 16 * 4^n bytes, added to the task's model; TIME-bound ceiling")

# Backwards-compatible alias (any external script importing the old name keeps
# the samplers semantics it used to get).
BYTES_PER_STATE = BYTES_PER_STATE_SAMPLERS


def _cgroup_memory_limit_mb() -> Optional[float]:
    """RAM limit of the CONTAINER in MB, or None if there is none.

    [B-CGROUP] `psutil.virtual_memory().total` reports the RAM of the NODE, not
    of the container. In a Kubernetes pod with "Maximum memory 63Gi" on a node
    of, say, 512 GB, the runner believed it had 512 GB, admitted dozens of
    tasks at once and the runtime killed the pod for OOM.

    And an OOMKill is not a Python MemoryError: it is a SIGKILL. There is no
    traceback, no partial results, not a single log line explaining why. The
    whole run disappears.

    Reads cgroup v2 and v1. A gigantic value ('max', or the v1 sentinel) means
    no limit, and then it returns None so that the caller uses the node's RAM
    as before.

    Returns:
        Limit in MB, or None if there is no cgroup or it is unlimited.
    """
    candidates = (
        '/sys/fs/cgroup/memory.max',                       # cgroup v2
        '/sys/fs/cgroup/memory/memory.limit_in_bytes',     # cgroup v1
    )
    for path in candidates:
        try:
            raw = open(path).read().strip()
        except Exception:
            continue
        if raw == 'max':
            return None
        try:
            value = float(raw)
        except ValueError:
            continue
        # v1 uses 2^63-1 (or similar) as "no limit"; anything above 1 PB is
        # clearly a sentinel and not a real limit.
        if value <= 0 or value > 1e15:
            return None
        return value / 1e6
    return None


def _cgroup_cpu_limit() -> Optional[int]:
    """Cores the CONTAINER may use, or None if there is no limit.

    [B-CGROUP] Same problem as memory: `os.cpu_count()` returns the node's
    cores. With "Maximum CPU 15" on a 128-core node, the runner launched 128
    processes fighting over 15 cores — slower than doing it right, and with
    far more RAM in flight.

    Returns:
        Number of cores (>=1), or None if no limit is declared.
    """
    try:                                                   # cgroup v2
        quota, period = open('/sys/fs/cgroup/cpu.max').read().split()
        if quota != 'max':
            return max(1, int(float(quota) / float(period)))
        return None
    except Exception:
        pass
    try:                                                   # cgroup v1
        quota = float(open('/sys/fs/cgroup/cpu/cpu.cfs_quota_us').read())
        period = float(open('/sys/fs/cgroup/cpu/cpu.cfs_period_us').read())
        if quota > 0 and period > 0:
            return max(1, int(quota / period))
    except Exception:
        pass
    return None


def detected_cores() -> int:
    """Usable cores: the container limit if there is one, otherwise the node's."""
    return _cgroup_cpu_limit() or os.cpu_count() or 1


def detected_memory_mb() -> float:
    """Usable RAM in MB: the container limit if there is one, otherwise the node's."""
    limit = _cgroup_memory_limit_mb()
    if limit is not None:
        return limit
    try:
        import psutil
        return psutil.virtual_memory().total / 1e6
    except Exception:
        return 8_000.0


@dataclass
class Task:
    """One campaign task: what to run, with which arguments and how heavy it is.

    Groups everything the planner needs to know BEFORE launching it
    (estimated memory, qubits, noise level) and what is known AFTER (exit
    code, peak RSS, timings), so that `master_profile.csv` comes from a
    single place.
    """
    name: str                       # human-readable label
    script: str                     # cosmo_modular_quantum.py | cosmo_genetic_optimizers.py
    argv: List[str]                 # CLI arguments (without 'python' or the script)
    model: str
    total_qubits: int               # nqpp*d (samplers) or n_bits*d (genetic)
    est_mem_mb: float               # estimated peak RAM of the task
    outdir: str
    grid_value: int = 0             # EFFECTIVE nqpp (samplers) or n_bits (genetic)
    grid_kind: str = ""             # 'nqpp' | 'n_bits'
    # [NOISE] Second dimension of the ablation axis, treated like any other
    # (just like nqpp and n_bits): canonical label of the noise rung.
    noise: str = "none"
    # Results (filled in at run time):
    pid: Optional[int] = None
    rc: Optional[int] = None
    t_start: float = 0.0
    t_end: float = 0.0
    peak_rss_mb: float = 0.0
    peak_vram_mb: float = 0.0        # from cosmo_profiling (measured), if a GPU is used
    gpu_hours: float = 0.0           # from cosmo_profiling
    device: str = ""                 # 'CPU' | 'GPU' (what the child measured)
    mem_source: str = "sampled"      # 'measured' (profile json) | 'sampled' (psutil)
    log_path: str = ""

    @property
    def wall_s(self) -> float:
        """Wall-clock seconds of the task, or 0.0 if it has not finished yet."""
        if self.t_start and self.t_end:
            return self.t_end - self.t_start
        return 0.0


# =============================================================================
# 1.  BUILDING THE TASK LIST
# =============================================================================

def estimate_qubits_and_mem(total_q: int, kind: str = 'nqpp',
                            noisy: bool = False,
                            n_data: Optional[int] = None) -> float:
    """Estimated peak RAM (MB) of a task whose circuit/grid has 2^total_q
    states, plus the fixed per-process interpreter cost.

    `kind` selects the per-state cost: 'nqpp' (samplers — grid + likelihood
    auxiliaries, which depend on N_data) or 'n_bits' (genetic — a plain
    statevector, independent of the dataset). See the BYTES_PER_STATE_*
    comments: using the samplers constant for the QGA overestimated it ~830x,
    and using an N_data-independent constant for the samplers UNDERestimated
    it 5.3x, which is what sent three tasks to the OOM killer ([B-MEM]).

    Args:
        total_q: total circuit/grid qubits (nqpp*d or n_bits*d).
        kind: 'nqpp' | 'n_bits'.
        noisy: if True, ADDS the density matrix (16 * 4^total_q). It is a
            third additive model, not a replacement: the samplers task still
            needs its grid in addition to rho.
        n_data: data points of the active dataset; `None` uses the largest in
            `DATASET_N_DATA` (conservative). Ignored with kind='n_bits'.
    Examples:
        At 18 qubits with CC+BAO+Pantheon a task weighs ~22 GB (19.6 were
        measured); with CC+BAO, ~4.6 GB (4.0 were measured):

        >>> round(estimate_qubits_and_mem(18, 'nqpp', n_data=1099) / 1024, 1)
        22.1
        >>> round(estimate_qubits_and_mem(18, 'nqpp', n_data=51) / 1024, 1)
        4.5

        And at 20 qubits — the `cpl/nqpp5` task that was OOMKilled — it is
        88 GB, which the old model estimated at 14:

        >>> round(estimate_qubits_and_mem(20, 'nqpp', n_data=1099) / 1024)
        88

        The genetic optimizer does not evaluate the likelihood over the grid,
        so its cost does not depend on the dataset:

        >>> (estimate_qubits_and_mem(20, 'n_bits', n_data=51)
        ...  == estimate_qubits_and_mem(20, 'n_bits', n_data=1099))
        True
    """
    per_state = bytes_per_state(kind, n_data)
    mb = (2 ** total_q) * per_state / 1e6 + PROCESS_BASELINE_MB
    if noisy:
        # [REV] With quantum training the cost is not a single rho but the
        # parameter-shift batch (2*n_phi density matrices at once), which at
        # 12 qubits is 42 GB versus 256 MB for a single one. The genetic
        # optimizer has no such batch.
        factor = (cnoise.param_shift_batch_factor(total_q)
                  if kind == 'nqpp' else 1)
        mb += cnoise.noisy_density_bytes(total_q, factor) / 1e6
    return mb


def qubits_fitting_in(mem_mb: float, kind: str = 'nqpp',
                      n_data: Optional[int] = None) -> int:
    """Largest number of qubits whose 2^q states fit in mem_mb, for the given
    pipeline `kind` (the per-process baseline is reserved first).

    `n_data` only matters for 'nqpp': the per-state cost of the samplers grows
    with the size of the dataset ([B-MEM]).

    Args:
        mem_mb: available memory, in MB.
        kind: pipeline: 'nqpp' (samplers) or 'n_bits' (genetic). Default
            'nqpp'.
        n_data: number of data points of the dataset. Default None.

    Returns:
        int
    """
    usable = mem_mb - PROCESS_BASELINE_MB
    if usable <= 0:
        return 0
    per_state = bytes_per_state(kind, n_data)
    cap_states = usable * 1e6 / per_state
    q = 0
    while 2 ** (q + 1) <= cap_states:
        q += 1
    return q


def qubit_ceiling(max_qubits: Optional[int], mem_ceiling_mb: float,
                  kind: str = 'nqpp', noisy: bool = False,
                  n_data: Optional[int] = None,
                  generations: Optional[int] = None,
                  budget_hours: float = cnoise.DEFAULT_NOISY_TASK_HOURS
                  ) -> int:
    """EFFECTIVE per-task qubit ceiling for the given pipeline `kind`.

    [AUTO] `max_qubits=None` (the default — see build_parser) means "no user
    cap": the ceiling is derived PURELY from the RAM this node actually has,
    auto-detected via psutil. Pass an explicit --max-qubits[-genetic] only to
    be MORE conservative than what your RAM allows (e.g. a shared node where
    you want to leave headroom, or you want a faster/coarser run on purpose).
    You can never exceed what fits in RAM this way — it stays the ultimate
    backstop regardless of what you pass.

    [NOISE] With `noisy=True` the noise-axis ceiling is applied AS WELL, and
    it is also derived from RAM — like the other two, and unlike the first
    version of this axis, which hard-fixed it at 13 arguing a time limit. That
    was a bad generalization drawn from a small machine: on a large node with
    no hurry the limiting factor is memory again, and that does relax.

    What rules under noise is not a single rho but the parameter-shift BATCH
    of QVMC quantum training (`2 * n_phi` density matrices at once). Since a
    samplers task walks the whole ladder, that rung is the one that sets its
    ceiling. The genetic optimizer has no such batch and therefore gets
    `quantum_training=False`.

    Args:
        max_qubits: cap requested by the user, or None.
        mem_ceiling_mb: RAM available per task.
        kind: 'nqpp' | 'n_bits'.
        noisy: apply the noise-axis ceiling.
        n_data: data points of the active dataset (only affects 'nqpp').
        generations: planned generations, for the TIME ceiling of the noisy
            genetic optimizer ([B-TIME] in cosmo_noise). Only applies with
            kind='n_bits' and noisy=True.
        budget_hours: wall-clock budget per task for that ceiling.
    """
    ram_ceiling = qubits_fitting_in(mem_ceiling_mb, kind, n_data)
    ceiling = (ram_ceiling if max_qubits is None
               else min(max_qubits, ram_ceiling))
    if noisy:
        ceiling = min(ceiling, cnoise.noisy_qubit_ceiling(
            requested=None, mem_mb=mem_ceiling_mb,
            quantum_training=(kind == 'nqpp'),
            generations=generations, budget_hours=budget_hours))
    return ceiling


def grid_values_for_model(single: int, sweep: Optional[List[int]], d: int,
                          q_ceiling: int, strict: bool,
                          notices: List[str], kind: str, model: str
                          ) -> List[int]:
    """List of grid values (nqpp / n_bits) to use for ONE model.

    Implements the per-model clamp: the user sets a target (e.g. 6); if for THIS
    model 6 exceeds the RAM/qubit ceiling, it is lowered ONLY for this model to
    the largest value that fits (q_ceiling // d), leaving the other models at
    their requested value. With --strict-qubits nothing is clamped: whatever
    exceeds the ceiling is SKIPPED (later flagged as SKIP).

    For a sweep [LO, HI] the upper bound is trimmed to fit_max per model; if not
    even LO fits, the model produces no tasks (with a notice).

    Args:
        single: single value requested by the user.
        sweep: requested range, or None.
        d: number of free parameters of the model.
        q_ceiling: qubit ceiling per task.
        strict: do not clamp; fail if it does not fit.
        notices: list where the per-model clamp notices accumulate.
        kind: pipeline: 'nqpp' (samplers) or 'n_bits' (genetic).
        model: cosmological model (`cosmo_core.CosmoModel`).

    Returns:
        List[int]
    """
    fit_max = max(1, q_ceiling // d)            # largest grid that fits for this d
    if sweep:
        lo, hi = sorted((int(sweep[0]), int(sweep[1])))
        if strict:
            vals = [v for v in range(lo, hi + 1) if v * d <= q_ceiling]
        else:
            hi_eff = min(hi, fit_max)
            vals = list(range(lo, hi_eff + 1))
            if hi_eff < hi:
                notices.append(
                    f"{model}: {kind} sweep {lo}..{hi} -> {lo}..{hi_eff} "
                    f"(d={d}; {hi}x{d}={hi*d}q exceeds the {q_ceiling}q ceiling)")
        if not vals:
            notices.append(
                f"{model}: NO {kind} in the sweep {lo}..{hi} fits in "
                f"{q_ceiling}q (d={d}). Model skipped. Raise --max-qubits or "
                f"lower the range.")
        return vals
    # single value with clamp
    if strict:
        return [single]
    eff = min(single, fit_max)
    if eff < single:
        notices.append(
            f"{model}: {kind} {single} -> {eff} "
            f"(d={d}; {single}x{d}={single*d}q exceeds the {q_ceiling}q ceiling; "
            f"the other models stay at {single})")
    return [eff]


def build_tasks(args, master_dir: str, q_ceiling: int,
                notices: List[str], q_ceiling_genetic: Optional[int] = None,
                noise_levels: Optional[List[str]] = None,
                noisy_q_ceiling: Optional[int] = None,
                noisy_q_ceiling_genetic: Optional[int] = None
                ) -> List[Task]:
    """One task per (script x model x grid value x NOISE LEVEL).

    When a sweep is requested (--nqpp-sweep / --nbits-sweep) one task is
    generated per value, to measure how the grid size affects the results. The
    grid value is trimmed per model according to `q_ceiling` (unless
    --strict-qubits), so a heavy model (CPL, d=4) is lowered on its own while
    the light ones (LCDM, d=2) stay at the target value.

    [NOISE] The noise level is ONE MORE task dimension, exactly like nqpp and
    n_bits: `--noise-sweep none,readout,full` generates the two-dimensional
    matrix (quantumness x noise) in a single invocation. Each noisy rung
    carries its OWN qubit ceiling and its own memory model, because the
    density matrix changes both: the ceiling drops to 13 and the RAM cost
    gains a 16*4^n term that dominates the grid term.

    Args:
        args: namespace of the runner's parser.
        master_dir: master folder of the run.
        q_ceiling: samplers qubit ceiling WITHOUT noise.
        notices: list where clamp notices accumulate.
        q_ceiling_genetic: genetic qubit ceiling WITHOUT noise.
        noise_levels: rungs to generate. None is equivalent to `['none']`.
        noisy_q_ceiling: samplers ceiling WITH noise.
        noisy_q_ceiling_genetic: genetic ceiling WITH noise.

    Returns:
        List of tasks.
    """
    tasks: List[Task] = []
    models = args.models or ALL_MODELS
    strict = bool(getattr(args, 'strict_qubits', False))

    common_data = ['--dataset', args.dataset, '--prior', args.prior,
                   '--seed', str(args.seed)]
    if args.profile:
        common_data.append('--profile')
    if args.gpu:
        common_data.append('--gpu')

    sweeping_nqpp = bool(args.nqpp_sweep)
    sweeping_nbits = bool(args.nbits_sweep)

    levels = [cnoise.canonical_level(x) for x in (noise_levels or ['none'])]
    sweeping_noise = len(levels) > 1

    # [NOISE-CONTROL] The control column.
    #
    # The ideal rung uses the amplitude route by default and any noisy rung
    # uses the counts route, because Re(psi)*sign(Im(psi)) does not exist for
    # a mixed state. Comparing the ideal column against the noisy ones then
    # mixes TWO effects: the noise and the change of readout operator.
    #
    # This is not theoretical. In the first run of the axis, reading
    # 'none' -> 'readout' suggested that readout noise NARROWS the posterior
    # and IMPROVES mixing (sigma 0.0208 -> 0.0176, ESS 101 -> 141). With the
    # control one sees that ideal-by-counts and readout are identical: the
    # readout channel does not touch the proposal and the whole jump was the
    # change of route.
    #
    # That is why the control is added AUTOMATICALLY — a noise sweep without
    # it produces a matrix that invites inverted conclusions.
    # `--no-noise-control` disables it for whoever explicitly wants that.
    route = getattr(args, 'proposal_route', 'auto')
    plan: List[Tuple[str, str, str]] = [
        (lvl, route, f"noise-{lvl}" if sweeping_noise else "") for lvl in levels]
    want_control = (sweeping_noise
                    and not getattr(args, 'no_noise_control', False)
                    and 'none' in levels
                    and route == 'auto'
                    and not args.only_genetic)
    if want_control:
        plan.append(('none', 'counts', 'noise-none-counts'))

    for noise, proposal_route, ntag in plan:
        noisy = (noise != 'none')
        # The control shares the 'none' rung but is a column of its own: it is
        # labeled differently so it does not collide in the CSV or on disk.
        noise_col = 'none-counts' if proposal_route == 'counts' and not noisy \
            else noise
        # [NOISE] The rung's own ceilings and memory model.
        q_cap = (noisy_q_ceiling if noisy and noisy_q_ceiling is not None
                 else q_ceiling)
        g_cap_base = (q_ceiling if q_ceiling_genetic is None
                      else q_ceiling_genetic)
        g_cap = (noisy_q_ceiling_genetic
                 if noisy and noisy_q_ceiling_genetic is not None
                 else g_cap_base)
        # The rung only enters the name/folder when there is more than one, so
        # that an ideal run produces EXACTLY the same output paths as before
        # this axis and the previous CSVs still line up.
        noise_argv = (['--noise', noise] if noisy or sweeping_noise else [])
        # [E-HPC2] Forward the channel strengths. They used to be parsed here
        # and dropped, so every child ran with the defaults. Only non-default
        # values are forwarded, which keeps default commands unchanged.
        for flag, attr, default in (
                ('--noise-readout-p', 'noise_readout_p', cnoise.DEFAULT_READOUT_P),
                ('--noise-gate-p1', 'noise_gate_p1', cnoise.DEFAULT_GATE_P1),
                ('--noise-gate-p2', 'noise_gate_p2', cnoise.DEFAULT_GATE_P2)):
            val = getattr(args, attr, default)
            if noisy and val != default:
                noise_argv += [flag, repr(float(val))]
        route_argv = (['--proposal-route', proposal_route]
                      if proposal_route != 'auto' else [])

        for m in models:
            d = MODEL_DIM[m]

            # ---- Samplers tasks (QMCMC + QVMC), one per nqpp value ----
            # [E-RUNGS] A QMCMC-only re-run builds no QVMC grid: its circuits
            # use max(2, d) qubits whatever nqpp is, so the grid ceiling must
            # not clamp nqpp (that would relabel the rows' grid column).
            _rungs = getattr(args, 'rungs', None) or []
            mcmc_only = bool(_rungs) and not any(
                t in _rungs for t in ('C-VI', 'QVMC33', 'QVMC67', 'QVMC100'))
            s_cap = 64 if mcmc_only else q_cap
            if not args.only_genetic:
                for nqpp in grid_values_for_model(
                        args.nqpp, args.nqpp_sweep, d, s_cap, strict,
                        notices, 'nqpp', m):
                    total_q = max(2, d) if mcmc_only else nqpp * d
                    # visible tag if sweeping or if the clamp changed the value
                    tag = (f"nqpp{nqpp}"
                           if (sweeping_nqpp or nqpp != args.nqpp) else "")
                    parts = [p for p in (tag, ntag) if p]
                    suffix = ("/" + "/".join(parts)) if parts else ""
                    dsuffix = ("_" + "_".join(parts)) if parts else ""
                    name = f"samplers/{m}{suffix}"
                    outdir = os.path.join(master_dir,
                                          f"samplers_{m}{dsuffix}")
                    argv = ['--sweep-all', '--sweep-models', m,
                            '--steps', str(args.steps),
                            '--qvmc-iter', str(args.qvmc_iter),
                            '--nqpp', str(nqpp),
                            '--chains', str(args.chains),
                            '--shots', str(args.shots),
                            # [AUTO] pass the ACTUAL computed ceiling, not the
                            # raw --max-qubits (which may be unset): correct
                            # whether the cap came from RAM-auto-detection or
                            # from an explicit user override.
                            '--max-qubits', str(max(q_cap, nqpp * d) if mcmc_only else q_cap),
                            '--outdir', outdir] + common_data + noise_argv \
                        + route_argv \
                        + (['--rungs'] + list(args.rungs)
                           if getattr(args, 'rungs', None) else [])
                    tasks.append(Task(
                        name=name, script='cosmo_modular_quantum.py',
                        argv=argv, model=m, total_qubits=total_q,
                        est_mem_mb=estimate_qubits_and_mem(
                            total_q, 'nqpp', noisy=noisy,
                            n_data=dataset_n_data(getattr(args, 'dataset',
                                                          None))),
                        outdir=outdir, grid_value=nqpp, grid_kind='nqpp',
                        noise=noise_col))

            # ---- Genetic tasks (CGA + QGA), one per n_bits value ----
            # [NOISE-CONTROL] The control is a column of the QMCMC PROPOSAL;
            # the QGA has no switchable readout route, so duplicating it here
            # would only repeat the ideal column.
            if not args.only_samplers and proposal_route == 'auto':
                # [FIX] The genetic pipeline has its OWN ceiling: the QGA does
                # not build the 2^(n*d) likelihood grid the samplers do, so it
                # is bounded by a plain statevector (16 B/state), not by the
                # samplers' auxiliary arrays (13.3 kB/state).
                for nbits in grid_values_for_model(
                        args.n_bits, args.nbits_sweep, d, g_cap,
                        strict, notices, 'n_bits', m):
                    total_q = nbits * d
                    tag = (f"nb{nbits}"
                           if (sweeping_nbits or nbits != args.n_bits) else "")
                    parts = [p for p in (tag, ntag) if p]
                    suffix = ("/" + "/".join(parts)) if parts else ""
                    dsuffix = ("_" + "_".join(parts)) if parts else ""
                    name = f"genetic/{m}{suffix}"
                    outdir = os.path.join(master_dir,
                                          f"genetic_{m}{dsuffix}")
                    argv = ['--sweep-all', '--sweep-models', m,
                            '--generations', str(args.generations),
                            '--population-size', str(args.population_size),
                            '--n-bits', str(nbits),
                            # [AUTO] pass the ACTUAL computed ceiling (RAM-auto
                            # or user override) so the child's own validation
                            # agrees with the plan built here.
                            '--max-qubits', str(g_cap),
                            '--outdir', outdir] + common_data + noise_argv \
                        + (['--sweep-qga-levels'] + [str(q) for q in args.qga_levels]
                           if getattr(args, 'qga_levels', None) else [])
                    tasks.append(Task(
                        name=name, script='cosmo_genetic_optimizers.py',
                        argv=argv, model=m, total_qubits=total_q,
                        est_mem_mb=estimate_qubits_and_mem(total_q, 'n_bits',
                                                           noisy=noisy),
                        outdir=outdir, grid_value=nbits, grid_kind='n_bits',
                        noise=noise_col))

    return tasks


# =============================================================================
# 2.  SCHEDULER (subprocess pool with concurrency and memory caps)
# =============================================================================

def child_env(threads_per_worker: int) -> Dict[str, str]:
    """Subprocess environment with the internal threads CAPPED.

    These variables are read by BLAS (OpenBLAS/MKL), NumExpr, Aer's OpenMP and
    Qiskit's rayon WHEN THEY ARE IMPORTED. That is why they must be set in the
    new interpreter's environment (subprocess), not after importing NumPy. This
    is the safety belt against oversubscription.

    Args:
        threads_per_worker: threads assigned to each subprocess.

    Returns:
        Dict[str, str]
    """
    env = os.environ.copy()
    t = str(max(1, threads_per_worker))
    for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
              'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
              'RAYON_NUM_THREADS'):
        env[k] = t
    # Aer also honors OMP_NUM_THREADS; reinforce just in case.
    env.setdefault('QISKIT_NUM_PROCS', t)
    return env


def proc_tree_rss_mb(pid: int) -> float:
    """Resident memory (MB) of the process + descendants. psutil if present,
    else /proc as a fallback (Linux). Returns 0 if the process is already

    Args:
        pid: identifier of the root process of the tree.

    Returns:
        float
    gone."""
    if _PSUTIL:
        try:
            p = psutil.Process(pid)
            procs = [p] + p.children(recursive=True)
            rss = 0
            for q in procs:
                try:
                    rss += q.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            return rss / 1e6
        except psutil.NoSuchProcess:
            return 0.0
    # Minimal fallback (process only, no tree):
    try:
        with open(f"/proc/{pid}/statm") as fh:
            pages = int(fh.read().split()[1])
        return pages * os.sysconf('SC_PAGE_SIZE') / 1e6
    except Exception:
        return 0.0


def run_pool(tasks: List[Task], max_parallel: int, threads_per_worker: int,
             mem_budget_mb: float, max_qubits: int, project_dir: str,
             poll: float = 0.5, max_qubits_genetic: Optional[int] = None
             ) -> None:
    """Run the tasks with at most `max_parallel` in flight, honoring an

    Args:
        tasks: list of campaign tasks.
        max_parallel: maximum number of simultaneous tasks.
        threads_per_worker: threads assigned to each subprocess.
        mem_budget_mb: aggregate memory budget, in MB.
        max_qubits: qubit cap per task.
        project_dir: project folder where the scripts live.
        poll: seconds between polls of the children's state. Default 0.5.
        max_qubits_genetic: qubit cap for the genetic tasks. Default
            None.
    aggregate RAM budget, and sample each tree's peak RSS."""
    pending = list(tasks)
    running: List[Task] = []
    skipped: List[Task] = []
    procs: Dict[int, subprocess.Popen] = {}

    def admitted_mem() -> float:
        """Estimated memory (MB) of the tasks currently running."""
        return sum(t.est_mem_mb for t in running)

    print(f"\n{'='*74}\nPLAN: {len(tasks)} tasks | "
          f"max_parallel={max_parallel} | threads/worker={threads_per_worker} | "
          f"RAM budget={mem_budget_mb/1024:.0f} GB\n{'='*74}")

    while pending or running:
        # --- launch tasks while there is both slot AND memory headroom ---
        i = 0
        while i < len(pending) and len(running) < max_parallel:
            t = pending[i]

            # Qubit cap: do not launch what the script itself would reject.
            # [FIX] Per-pipeline cap: the genetic tasks are bounded by their
            # own (higher) cap, since a QGA circuit costs 16 B/state, not the
            # samplers' 13.3 kB/state. `is not None` (not truthy) so a
            # degenerate 0-qubit genetic ceiling on a near-zero-RAM node
            # doesn't silently fall back to the samplers cap.
            cap = (max_qubits_genetic
                   if (t.grid_kind == 'n_bits' and max_qubits_genetic is not None)
                   else max_qubits)
            if t.total_qubits > cap:
                print(f"  [SKIP] {t.name}: {t.total_qubits} qubits "
                      f"(> cap {cap}). "
                      f"Lower n_bits/nqpp for {t.model} or raise the cap "
                      f"(mind the RAM).")
                t.rc = -2
                skipped.append(t)
                pending.pop(i)
                continue

            # Aggregate RAM budget (always let at least one task in).
            if running and admitted_mem() + t.est_mem_mb > mem_budget_mb:
                i += 1  # try the next one; maybe a lighter task fits
                continue

            os.makedirs(t.outdir, exist_ok=True)
            t.log_path = os.path.join(t.outdir, "stdout.log")
            cmd = [sys.executable, os.path.join(project_dir, t.script)] + t.argv
            logfh = open(t.log_path, 'w')
            t.t_start = time.time()
            p = subprocess.Popen(cmd, cwd=t.outdir, stdout=logfh,
                                  stderr=subprocess.STDOUT,
                                  env=child_env(threads_per_worker))
            t.pid = p.pid
            procs[t.pid] = p
            p._logfh = logfh  # type: ignore  (close on completion)
            running.append(t)
            pending.pop(i)
            print(f"  [START] {t.name:18s} pid={t.pid:<7d} "
                  f"~{t.total_qubits}q ~{t.est_mem_mb:.0f}MB  -> {t.outdir}")

        # --- sample RSS and reap finished tasks ---
        time.sleep(poll)
        for t in list(running):
            p = procs[t.pid]
            t.peak_rss_mb = max(t.peak_rss_mb, proc_tree_rss_mb(t.pid))
            rc = p.poll()
            if rc is not None:
                t.t_end = time.time()
                # [E-HPC1] Children written before this fix exit 0 even when a
                # model failed inside --sweep-all; treat a "FAILED —" line in
                # the task log as a failure as well.
                if rc == 0 and _log_reports_failure(t.log_path):
                    rc = 3
                t.rc = rc
                try:
                    p._logfh.close()  # type: ignore
                except Exception:
                    pass
                running.remove(t)
                # Prefer the memory MEASURED by cosmo_profiling (profile_*.json
                # the child wrote with --profile) over our own sampling.
                _apply_child_profile(t)
                status = "OK" if rc == 0 else f"FAILED (rc={rc})"
                extra = (f"  VRAM={t.peak_vram_mb/1024:.2f}GB  "
                         f"{t.gpu_hours:.4f} GPU-h" if t.device == 'GPU' else "")
                print(f"  [DONE]  {t.name:18s} {status:14s} "
                      f"wall={t.wall_s/60:6.1f} min  "
                      f"peakRSS={t.peak_rss_mb/1024:5.2f} GB "
                      f"({t.mem_source}){extra}")
                if rc != 0:
                    _tail(t.log_path)

    # return skipped tasks to the global list for the report
    for t in skipped:
        if t not in tasks:
            tasks.append(t)


def _log_reports_failure(path: str) -> bool:
    """True if a child's log contains a per-model failure line.

    The sweep drivers print ``<model>: FAILED — <reason>`` when a model raises
    and the batch continues. Used as a second check on top of the exit code.
    """
    try:
        with open(path, encoding='utf-8', errors='replace') as fh:
            return any(': FAILED — ' in line for line in fh)
    except OSError:
        return False


def _apply_child_profile(t: Task) -> None:
    """Read the profile_*.json the child wrote with --profile (the memory
    MEASURED by cosmo_profiling) and use it as the authoritative figure. If
    none is present, keep the peak sampled by psutil (proc_tree_rss_mb).

    Each json carries as_row(): wall_s, peak_rss_mb, peak_vram_mb, gpu_hours,
    mean_gpu_util_pct, device, gpu_name. A task (one model) may leave several
    (e.g. one per ladder); we aggregate: max RSS, max VRAM, sum of GPU-h.
    """
    import glob as _glob
    files = _glob.glob(os.path.join(t.outdir, '**', 'profile_*.json'),
                       recursive=True)
    if not files:
        return
    rss = vram = gpuh = 0.0
    dev = 'CPU'
    found = False
    for f in files:
        try:
            with open(f) as fh:
                d = json.load(fh)
            rss = max(rss, float(d.get('peak_rss_mb', 0.0)))
            vram = max(vram, float(d.get('peak_vram_mb', 0.0)))
            gpuh += float(d.get('gpu_hours', 0.0))
            if str(d.get('device', 'CPU')).upper() == 'GPU':
                dev = 'GPU'
            found = True
        except Exception:
            pass
    if found and rss > 0:
        t.peak_rss_mb = rss          # measured > sampled
        t.peak_vram_mb = vram
        t.gpu_hours = gpuh
        t.device = dev
        t.mem_source = 'measured'


def _tail(path: str, n: int = 12) -> None:
    """Print the last n lines of a failed task's log."""
    try:
        with open(path) as fh:
            lines = fh.readlines()
        print("    +- last log lines ------------------------------------")
        for ln in lines[-n:]:
            print("    | " + ln.rstrip())
        print("    +-----------------------------------------------------")
    except Exception:
        pass


# =============================================================================
# 3.  FINAL REPORT (memory + time) and persistence
# =============================================================================

def report(tasks: List[Task], master_dir: str, t_wall0: float) -> None:
    """Print the final campaign summary: status, time and RSS per task.

    Args:
        tasks: list of campaign tasks.
        master_dir: root folder of the campaign.
        t_wall0: timestamp of the start of the campaign.
    """
    total_wall = time.time() - t_wall0
    print(f"\n{'='*74}\nSUMMARY - total wall time: "
          f"{total_wall/60:.1f} min\n{'='*74}")
    hdr = f"{'task':20s} {'status':10s} {'wall(min)':>10s} {'peakRSS(GB)':>12s}"
    print(hdr)
    print("-" * len(hdr))
    rows = []
    for t in tasks:
        if t.rc == -2:
            status = "SKIP"
        elif t.rc == 0:
            status = "OK"
        elif t.rc is None:
            status = "?"
        else:
            status = f"rc={t.rc}"
        print(f"{t.name:20s} {status:10s} {t.wall_s/60:10.1f} "
              f"{t.peak_rss_mb/1024:12.2f}")
        rows.append({
            'task': t.name, 'script': t.script, 'model': t.model,
            'status': status, 'returncode': t.rc,
            'total_qubits': t.total_qubits,
            'grid_kind': t.grid_kind, 'grid_value': t.grid_value,
            # [NOISE] Second coordinate of the ablation axis. It is ALWAYS
            # written, also in ideal runs ('none'), so that the
            # two-dimensional matrix can be pivoted without special cases.
            'noise': t.noise,
            'wall_s': round(t.wall_s, 1),
            'wall_min': round(t.wall_s / 60, 2),
            'peak_rss_mb': round(t.peak_rss_mb, 1),
            'peak_rss_gb': round(t.peak_rss_mb / 1024, 3),
            'mem_source': t.mem_source,
            'peak_vram_gb': round(t.peak_vram_mb / 1024, 3),
            'gpu_hours': round(t.gpu_hours, 5),
            'device': t.device or 'CPU',
            'est_mem_mb': round(t.est_mem_mb, 1),
            'outdir': t.outdir, 'log': t.log_path,
        })

    ok = sum(1 for t in tasks if t.rc == 0)
    peak_concurrent = max((t.peak_rss_mb for t in tasks), default=0.0) / 1024
    print("-" * len(hdr))
    print(f"  {ok}/{len(tasks)} tasks OK | "
          f"peak RSS of the heaviest task: {peak_concurrent:.2f} GB")

    summary = {
        'total_wall_s': round(total_wall, 1),
        'total_wall_min': round(total_wall / 60, 2),
        'n_tasks': len(tasks), 'n_ok': ok,
        'tasks': rows,
    }
    with open(os.path.join(master_dir, 'master_profile.json'), 'w') as fh:
        json.dump(summary, fh, indent=2)
    # flat CSV
    import csv as _csv
    csv_path = os.path.join(master_dir, 'master_profile.csv')
    with open(csv_path, 'w', newline='') as fh:
        w = _csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else
                            ['task'])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\n  Master profile: {csv_path}")
    print(f"  Master profile: {os.path.join(master_dir, 'master_profile.json')}")
    print(f"  (Each task also left its resource_usage_*.png in its subfolder "
          f"if you used --profile.)")


# =============================================================================
# 3b. CONVERGENCE-vs-GRID PLOTS  (merged: formerly plot_grid_convergence.py)
# =============================================================================
#
#  Runs AUTOMATICALLY at the end of a sweep (--nqpp-sweep), or on demand with
#  python cosmo_hpc_runner.py --plot-only results/hpc_<ts>/ . It reads the
#  `nqpp` column already present in the result CSVs and, per model, plots each
#  parameter +/- sigma vs nqpp (one line per method, with reference lines) and
#  the cost (wall time and peak RAM) vs nqpp. numpy/matplotlib are imported
#  here, not at the top, so the orchestrator does not depend on them when it is
#  only launching processes.

# Reference values for the guide lines.
#  * Planck: reuse the project's values (cosmo_core: OM_MU=0.3111, H0_MU=67.66,
#    Planck 2018), so the Planck line is consistent with the prior you actually
#    use.
#  * SH0ES/Riess: the LOCAL H0 measurement (Riess et al. 2022), the other end of
#    the Hubble tension, so you can see which one each method approaches. Edit
#    here to use other values or add references.
REFERENCES = {
    'Planck 2018':       {'color': '#222222', 'ls': '--',
                          'vals': {'Om': (0.3111, 0.0056), 'H0': (67.66, 0.42)}},
    'SH0ES (Riess+ 22)': {'color': '#d62728', 'ls': ':',
                          'vals': {'H0': (73.04, 1.04)}},
}
PARAM_LATEX = {'Om': r'$\Omega_m$', 'H0': r'$H_0$', 'w': r'$w$',
               'w0': r'$w_0$', 'wa': r'$w_a$', 'Delta': r'$\Delta$'}


def _to_float(s) -> float:
    """Tolerant float(s): returns nan instead of raising if it cannot convert."""
    try:
        return float(s)
    except (TypeError, ValueError):
        return float('nan')


def _find_result_csvs(master_dir: str) -> List[str]:
    """Per-model result CSVs under `master_dir`.

    [E-QPU6] The cumulative `results_all_models.csv` (legacy name
    `resultados_TODOS_los_modelos.csv`) repeats the rows of the per-model
    files, so reading both counted every run twice. Both the current and the
    legacy file names are accepted; the cumulative file is never read.
    """
    import campaign_io
    return campaign_io.result_csvs_any_layout(master_dir)


def _infer_model(path: str) -> str:
    """Model name inferred from the path of the task's folder."""
    for part in path.split(os.sep):
        if part.startswith(('samplers_', 'genetic_')):
            return part.split('_')[1]
    return '?'


def _parse_result_rows(csv_paths: List[str]) -> List[dict]:
    """Flatten the CSVs into records {model, method, grid, param, mean, std,...}.
    Supports the generic schema (params + p1_mean...) and the named one
    (<p>_mean)."""
    import csv as _csv
    records: List[dict] = []
    for path in csv_paths:
        try:
            with open(path, newline='') as fh:
                reader = _csv.DictReader(fh)
                cols = reader.fieldnames or []
                generic = 'params' in cols and 'p1_mean' in cols
                for row in reader:
                    try:
                        grid = int(float(row.get('nqpp', '')))
                    except (TypeError, ValueError):
                        continue
                    common = dict(
                        model=row.get('model', '') or _infer_model(path),
                        method=row.get('Method', '?'), grid=grid,
                        chi2_red=_to_float(row.get('chi2_red', 'nan')),
                        final_KL=_to_float(row.get('final_KL', 'nan')),
                        ESS=_to_float(row.get('ESS', 'nan')),
                        time_s=_to_float(row.get('Time_s', 'nan')))
                    if generic:
                        for i, pname in enumerate(
                                (row.get('params', '') or '').split('|'), 1):
                            if pname:
                                records.append(dict(
                                    common, param=pname,
                                    mean=_to_float(row.get(f'p{i}_mean', '')),
                                    std=_to_float(row.get(f'p{i}_std', ''))))
                    else:
                        for c in cols:
                            if c.endswith('_mean'):
                                p = c[:-5]
                                records.append(dict(
                                    common, param=p,
                                    mean=_to_float(row.get(c, '')),
                                    std=_to_float(row.get(f'{p}_std', ''))))
        except Exception as e:
            print(f"  ! Could not read {path}: {e}")
    return records


def _read_master_profile_rss(master_dir: str):
    """Peak RAM (GB) per (model, nqpp) from master_profile.csv, if present."""
    import csv as _csv
    rss = {}
    mp = os.path.join(master_dir, 'master_profile.csv')
    if not os.path.exists(mp):
        return rss
    with open(mp, newline='') as fh:
        for row in _csv.DictReader(fh):
            if row.get('grid_kind') != 'nqpp':
                continue
            try:
                rss[(row['model'], int(row['grid_value']))] = _to_float(
                    row.get('peak_rss_gb', 'nan'))
            except (KeyError, ValueError):
                pass
    return rss


def _method_family(method: str) -> str:
    """Family a method belongs to: 'genetic' or 'samplers'.

    [B-NBITS] It is needed because the two families report their resolution
    in the SAME CSV column (`nqpp`) but measure different things: for the
    samplers it is qubits per parameter of the posterior grid, for the genetic
    optimizer it is bits per gene of the encoding. Drawing them on a single
    axis labeled 'nqpp' — which is what used to be done — puts together two
    numbers that are not comparable.

    Args:
        method: method name as it appears in the `Method` column.

    Returns:
        'genetic' or 'samplers'.

    Examples:
        >>> _method_family('CGA'), _method_family('QGA (q=33%)')
        ('genetic', 'genetic')
        >>> _method_family('QMCMC 50%'), _method_family('Classical VI')
        ('samplers', 'samplers')
    """
    m = method.upper()
    return 'genetic' if ('GA' in m.replace('CLASSICAL', '')) else 'samplers'


def _is_grid_method(method: str) -> bool:
    """True if the method lives on the grid (QVMC or classical VI).

    Used to avoid mixing grid and grid-free methods in one figure: nqpp only
    means something for the former.
    """
    m = method.lower()
    return ('vmc' in m) or ('vi' in m) or ('varia' in m)


def _method_styles(methods, plt):
    """Assign each method a STABLE, highly distinguishable
    (color, marker, line-style), so runs do not blend into each other, not even
    when two give almost the same value. Combines several qualitative palettes
    (~26 colors) and cycles markers and line styles."""
    base = (list(plt.get_cmap('tab10').colors)
            + list(plt.get_cmap('Set2').colors)
            + list(plt.get_cmap('Dark2').colors))
    markers = ['o', 's', '^', 'D', 'v', 'P', 'X', '*', 'h', '<', '>', 'p']
    lss = ['-', '--', '-.', ':']
    style = {}
    for i, mth in enumerate(sorted(methods)):
        style[mth] = (base[i % len(base)], markers[i % len(markers)],
                      lss[(i // len(markers)) % len(lss)])
    return style


NOISE_ORDER = ['none', 'none-counts', 'readout', 'full']


def _infer_noise(path: str) -> str:
    """Noise rung of a task, inferred from the name of its folder.

    Task folders are named `<pipeline>_<model>[_<tag>]_noise-<level>` when
    there is a noise sweep, and without the suffix when there is not. That
    absence means 'none' — an ideal run keeps the usual paths.

    Args:
        path: path of the results CSV.

    Returns:
        Rung label.
    """
    for part in path.split(os.sep):
        if '_noise-' in part:
            return part.split('_noise-')[-1]
    return 'none'


def _noise_sort_key(level: str):
    """Axis order: named rungs first, backends last."""
    return (NOISE_ORDER.index(level) if level in NOISE_ORDER
            else len(NOISE_ORDER), level)


def _infer_grid(path: str, row: dict) -> str:
    """Resolution of the task ('nqpp3', 'nb5', ...) or '' if it cannot be known.

    It is taken from the folder name, which the runner tags when there is a
    sweep; if the folder carries no tag, it falls back to the CSV's `nqpp`
    column.

    Args:
        path: path of the CSV.
        row: already parsed row (unused; kept for compatibility).

    Returns:
        Resolution label, or empty string.
    """
    import re as _re
    for part in path.split(os.sep):
        m = _re.search(r'_(nqpp\d+|nb\d+)(?:_|$)', part)
        if m:
            return m.group(1)
    # Without a tag on the folder there was no sweep, so ALL rows of that CSV
    # have the same resolution. Inferring it row by row from the `nqpp` column
    # would be worse: classical MCMC leaves it empty, so a single CSV was split
    # into two groups ('' and 'nqpp3') and duplicate figures came out with
    # half of the methods each.
    return ''


def _parse_noise_rows(csv_paths: List[str]) -> List[dict]:
    """Flatten the CSVs into records {model, method, noise, param, mean, std, ...}.

    Unlike `_parse_result_rows`, it does NOT require the `nqpp` column: the
    genetic tasks do not have it and here they do matter, because the QGA is
    one of the ladders the noise axis must compare.
    """
    import csv as _csv
    out: List[dict] = []
    for path in csv_paths:
        noise = _infer_noise(path)
        try:
            with open(path, newline='') as fh:
                reader = _csv.DictReader(fh)
                cols = reader.fieldnames or []
                generic = 'params' in cols and 'p1_mean' in cols
                for row in reader:
                    common = dict(
                        model=row.get('model', '') or _infer_model(path),
                        method=row.get('Method', '?'), noise=noise,
                        # [B-GRID] The resolution HAS to travel in the
                        # record. Without it, comparing along the noise axis
                        # mixed different nqpp: with --nqpp-sweep and
                        # --noise-sweep together, the noisy ceiling clamps
                        # some rungs and not others, so the ideal column
                        # could keep nqpp=5 and the noisy one nqpp=3. The
                        # figure then attributed to noise what was a change
                        # of resolution.
                        grid=_infer_grid(path, row),
                        chi2_red=_to_float(row.get('chi2_red', 'nan')),
                        final_KL=_to_float(row.get('final_KL', 'nan')),
                        ESS=_to_float(row.get('ESS', 'nan')))
                    if generic:
                        names = (row.get('params', '') or '').split('|')
                        for i, pname in enumerate(names, 1):
                            if pname:
                                out.append(dict(
                                    common, param=pname,
                                    mean=_to_float(row.get(f'p{i}_mean', '')),
                                    std=_to_float(row.get(f'p{i}_std', ''))))
                    else:
                        for col in cols:
                            if col.endswith('_mean'):
                                p = col[:-5]
                                out.append(dict(
                                    common, param=p,
                                    mean=_to_float(row.get(col, '')),
                                    std=_to_float(row.get(f'{p}_std', ''))))
        except Exception:
            continue
    return [r for r in out if r['mean'] == r['mean']]      # drop NaN


def generate_noise_comparison_plots(master_dir: str,
                                    outdir: Optional[str] = None
                                    ) -> List[str]:
    """One figure per model comparing EVERY rung with and without noise.

    [PLOT-NOISE] Directly answers "I want to see every quantumness level of
    every model with and without noise". The x axis is the noise rung in
    order, each line is a quantumness rung, and there is one panel per
    parameter plus one for fit quality. That way the degradation of a rung
    reads as the slope of its own line, and the comparison between rungs as
    the separation between lines — the two questions of the axis, in a single
    image.

    One line per rung rather than one panel per rung is deliberate: what
    matters is not the shape of each curve on its own but whether some rungs
    withstand noise better than others, and that is only visible by
    overlaying them.

    The `none-counts` column, when present, is drawn as part of the axis: it
    is the control that separates the effect of noise from the effect of the
    change of readout operator, and without it the slope between `none` and
    `readout` mixes both.

    Args:
        master_dir: master folder of the run.
        outdir: where to write (by default, the same folder).

    Returns:
        List of generated paths.
    """
    try:
        import numpy as np
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as e:                                  # pragma: no cover
        print(f"  (no noise plots: could not import matplotlib: {e})")
        return []

    outdir = outdir or master_dir
    records = _parse_noise_rows(_find_result_csvs(master_dir))
    if not records:
        print("  (no noise plots: found no readable rows)")
        return []

    levels = sorted({r['noise'] for r in records}, key=_noise_sort_key)
    if len(levels) < 2:
        print(f"  (no noise plots: only one rung, '{levels[0]}')")
        return []

    # [PLOT-NOISE] Samplers and genetic go in SEPARATE figures, and not for
    # aesthetics: their sigmas are not the same quantity.
    #
    # For MCMC/VI, sigma is the width of the posterior — a credible interval —
    # so "shift in units of sigma" answers "did it move more than what the
    # method knows about the parameter?".
    #
    # For a genetic optimizer, sigma is the spread of an ALREADY CONVERGED
    # population: a tiny quantity that only measures how tightly the
    # individuals clustered. Dividing by it produces shifts of 5 or 9 sigmas
    # that mean nothing — they showed up in the first run and looked like a
    # huge noise effect when they were a near-zero denominator. The genetic
    # optimizer is therefore drawn in ABSOLUTE units, which is what makes
    # sense for an optimizer whose result is a point.
    def _is_genetic(method: str) -> bool:
        """True if the method label belongs to the genetic family (CGA or QGA)."""
        m = (method or '').upper()
        return m.startswith('CGA') or m.startswith('QGA')

    made: List[str] = []
    for model in sorted({r['model'] for r in records}):
        for kind, keep, suffix in (
                ('samplers', lambda m: not _is_genetic(m), ''),
                ('genetic', _is_genetic, '_genetic')):
            pool = [r for r in records if r['model'] == model
                    and keep(r['method'])]
            if not pool:
                continue
            # [B-GRID] One figure per RESOLUTION. Mixing different nqpp on the
            # same noise axis turns a ceiling clamp into what looks like a
            # noise effect.
            grids = sorted({r['grid'] for r in pool})
            # With a single resolution there is nothing to split: the name and
            # the title stay as before, so as not to break earlier references.
            split = len(grids) > 1
            for g in grids:
                sub = [r for r in pool if r['grid'] == g]
                # Comparing only makes sense if that resolution exists in more
                # than one rung; if the ceiling clamped it in the noisy ones,
                # there is no comparison to make and drawing it would mislead.
                if len({r['noise'] for r in sub}) < 2:
                    print(f"  . {model} [{kind}] {g or 'no-grid'}: only "
                          f"1 noise rung, no comparison (skipped)")
                    continue
                gsuf = f"{suffix}_{g}" if (split and g) else suffix
                label = f"{model} ({g})" if (split and g) else model
                pth = _one_noise_figure(model, sub, levels, outdir, gsuf,
                                        pull=(kind == 'samplers'),
                                        np=np, plt=plt, title_label=label)
                if pth:
                    made.append(pth)
                    print(f"  . {model} [{kind}]"
                          f"{' ' + g if split and g else ''}: {pth}")
    return made


def _one_noise_figure(model, rows, levels, outdir, suffix, pull, np, plt,
                      title_label=None):
    """Draw ONE noise comparison figure.

    Args:
        model: model name.
        rows: records of a single pipeline (samplers or genetic).
        levels: noise rungs, already sorted.
        outdir: output folder.
        suffix: file name suffix ('' or '_genetic').
        pull: True to draw the shift in units of sigma (only meaningful when
            sigma is the width of a posterior); False for absolute values
            (genetic).
        np, plt: modules already imported by the caller.

    Returns:
        Path of the figure, or empty string if there was nothing to draw.
    """
    if True:
        params, seen = [], set()
        for r in rows:
            if r['param'] not in seen:
                seen.add(r['param']); params.append(r['param'])
        methods = sorted({r['method'] for r in rows})
        if not params or not methods:
            return ""

        # Quality metric: the KL if the model produces it (QVMC), else chi2.
        has_kl = any(r['final_KL'] == r['final_KL'] for r in rows)
        metric, mlabel = (('final_KL', 'final KL (lower = better fit)')
                          if has_kl else ('chi2_red', r'reduced $\chi^2$'))

        npan = len(params) + 1
        ncol = min(3, npan)
        nrow = int(np.ceil(npan / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(5.4 * ncol, 4.2 * nrow),
                                 squeeze=False)
        flat = [axes[i // ncol][i % ncol] for i in range(nrow * ncol)]
        for ax in flat[npan:]:
            ax.set_visible(False)

        x = np.arange(len(levels))
        # Color and marker per method, fixed across all panels: identity never
        # depends on color alone.
        cyc = ['#1f77b4', '#d62728', '#ff7f0e', '#2ca02c', '#9467bd',
               '#17becf', '#8c564b', '#e377c2']
        mks = ['o', 's', '^', 'D', 'v', 'P', 'X', '*']
        style = {m: (cyc[i % len(cyc)], mks[i % len(mks)])
                 for i, m in enumerate(methods)}

        def series(method, key, sub=None):
            """(value, error) series of a method along the noise axis.

            Args:
                method: method name.
                key: key of the quantity to extract.
                sub: parameter name to select within the key. Default
                    None.

            Returns:
                See the description above.
            """
            ys, es = [], []
            for lvl in levels:
                sel = [r for r in rows if r['method'] == method
                       and r['noise'] == lvl
                       and (sub is None or r['param'] == sub)]
                ys.append(sel[-1][key] if sel else float('nan'))
                es.append(sel[-1].get('std', float('nan')) if sel
                          else float('nan'))
            return np.array(ys, float), np.array(es, float)

        # [PLOT-NOISE] The parameter panels show the SHIFT in units of sigma
        # relative to the ideal rung, not the absolute value.
        #
        # With the absolute value the posterior's sigma bar (~0.016 in Om) is
        # an order of magnitude larger than what noise moves the mean
        # (~0.002), so all lines get squashed into a common band and the
        # figure cannot tell one rung from another. The real question is not
        # "what is Om" — that is already in the corner plots — but "how much
        # did noise move it compared with what the method knows about it".
        # That is exactly (mean - mean_ideal) / sigma_ideal.
        #
        # The grey +-1 sigma band gives the scale: a line inside it moved less
        # than the method's own uncertainty, i.e. noise did not shift it
        # detectably.
        for j, p in enumerate(params):
            ax = flat[j]
            if pull:
                ax.axhspan(-1, 1, color='0.85', alpha=0.55, zorder=0, lw=0)
                ax.axhline(0, color='0.45', lw=1.0, ls=':', zorder=1)
            for m in methods:
                y, e = series(m, 'mean', sub=p)
                if not np.any(np.isfinite(y)):
                    continue
                col, mk = style[m]
                if pull:
                    ref = y[0]
                    sig = e[0] if (len(e) and np.isfinite(e[0]) and e[0] > 0) \
                        else np.nanmax(e) if np.any(np.isfinite(e)) else np.nan
                    if not np.isfinite(sig) or sig <= 0:
                        continue
                    y = (y - ref) / sig
                ax.plot(x, y, color=col, marker=mk, ms=7, lw=2,
                        alpha=0.9, zorder=3, label=m if j == 0 else None)
            ax.set_ylabel(f'{p}:  shift from ideal  [$\\sigma$]' if pull
                          else f'{p}  (MAP)', fontsize=10)
            ax.set_xticks(x)
            ax.set_xticklabels(levels, rotation=20, ha='right', fontsize=9)
            ax.grid(True, axis='y', alpha=0.25, lw=0.6)
            for s in ('top', 'right'):
                ax.spines[s].set_visible(False)

        ax = flat[len(params)]
        for m in methods:
            y, _ = series(m, metric)
            if not np.any(np.isfinite(y)):
                continue
            col, mk = style[m]
            ax.plot(x, y, color=col, marker=mk, ms=7, lw=2, alpha=0.9)
        ax.set_ylabel(mlabel, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(levels, rotation=20, ha='right', fontsize=9)
        ax.grid(True, alpha=0.25, lw=0.6)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)

        handles, labels = flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center',
                   ncol=min(len(labels), 4), fontsize=9, frameon=False,
                   bbox_to_anchor=(0.5, -0.02))
        sub_t = ("parameters: shift from the ideal rung in units of its own "
                 "sigma (grey band = +-1 sigma)" if pull else
                 "genetic: absolute MAP — a GA's spread is convergence, not a "
                 "credible interval, so sigma units would be meaningless")
        fig.suptitle(
            f"{title_label or model} — every quantumness rung, with and "
            f"without noise\n"
            f"{sub_t}   ·   right: fit quality",
            fontsize=13, fontweight='bold')
        fig.tight_layout(rect=(0, 0.06, 1, 0.90))
        path = os.path.join(outdir, f"noise_comparison{suffix}_{model}.png")
        fig.savefig(path, dpi=150, bbox_inches='tight')
        fig.savefig(path.replace('.png', '.pdf'), bbox_inches='tight')
        plt.close(fig)
        return path


def generate_convergence_plots(master_dir: str, outdir: Optional[str] = None,
                               only_grid_methods: bool = False,
                               xlabel: str = 'nqpp',
                               family: Optional[str] = None,
                               suffix: str = '') -> List[str]:
    """Generate convergence_<model>.png and cost_<model>.png for each model with
    >=2 grid values. Returns the paths created. Does not raise if matplotlib is

    Args:
        master_dir: root folder of the campaign.
        outdir: folder to write the output to. Default None.
        only_grid_methods: restrict to the methods that live on the grid.
            Default False.
        xlabel: x-axis label. Default 'nqpp'.
        family: [B-NBITS] restricts to one family of methods: 'samplers'
            (QMCMC/QVMC/classical) or 'genetic' (CGA/QGA). None draws all of
            them, which is what it did before and was WRONG: the genetic x
            axis is n_bits (bits per gene) and the samplers one is nqpp
            (qubits per grid parameter). They are different quantities and
            sharing an axis makes them incomparable. Default None.
        suffix: appended to the file name so as not to overwrite the other
            family's figure. Default ''.

    Returns:
        List[str]
    missing: it warns and returns []."""
    try:
        import numpy as np
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"  (no plots: could not import matplotlib/numpy: {e})")
        return []

    outdir = outdir or master_dir
    csvs = _find_result_csvs(master_dir)
    if not csvs:
        print("  (no plots: no result CSVs found)")
        return []
    records = _parse_result_rows(csvs)
    if not records:
        print("  (no plots: CSVs had no readable rows)")
        return []
    rss_map = _read_master_profile_rss(master_dir)
    models = sorted({r['model'] for r in records})

    def series(model, param):
        """Group the records of a (model, parameter) by method and resolution.

        Args:
            model: cosmological model (`cosmo_core.CosmoModel`).
            param: parameter name.

        Returns:
            See the description above.
        """
        from collections import defaultdict
        by = defaultdict(list)
        for r in records:
            if r['model'] == model and r['param'] == param:
                by[r['method']].append((r['grid'], r['mean'], r['std']))
        out = {}
        for method, pts in by.items():
            pts = sorted(set(pts))
            out[method] = (np.array([p[0] for p in pts], float),
                           np.array([p[1] for p in pts], float),
                           np.array([p[2] for p in pts], float))
        return out

    made: List[str] = []
    for model in models:
        # [B-NBITS] The x axis is set ONLY by the records of the family being
        # drawn. Without this filter the genetic figure inherited the nqpp
        # values of the samplers (e.g. 2 and 3) and the axis ran from 2 to 5
        # with data only at 4 and 5 — half an empty plot and a scale that
        # belongs to neither family.
        grids = sorted({r['grid'] for r in records if r['model'] == model
                        and (not family
                             or _method_family(r['method']) == family)})
        if len(grids) < 2:
            print(f"  . {model}: only {len(grids)} {xlabel} value(s); "
                  f"no convergence to plot (skipped).")
            continue
        params, seen = [], set()
        for r in records:
            if r['model'] == model and r['param'] not in seen:
                seen.add(r['param']); params.append(r['param'])

        # --- convergence figure (one panel per parameter) ---
        ncol = min(2, len(params)); nrow = int(np.ceil(len(params) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(6.4 * ncol, 4.2 * nrow),
                                 squeeze=False)
        # stable per-method styles (same color/marker across all panels)
        all_methods = sorted({r['method'] for r in records
                              if r['model'] == model})
        style = _method_styles(all_methods, plt)
        for idx, param in enumerate(params):
            ax = axes[idx // ncol][idx % ncol]
            for method in sorted(series(model, param)):
                if only_grid_methods and not _is_grid_method(method):
                    continue
                if family and _method_family(method) != family:
                    continue
                g, m, s = series(model, param)[method]
                col, mk, ls = style[method]
                finite = m[~np.isnan(m)]
                # Does the method depend on the grid? If its estimate is (almost)
                # identical across all nqpp -> it is grid-independent (MCMC/QMCMC
                # do not use the grid). We draw it ONCE as a horizontal line,
                # not a curve repeated at every nqpp.
                spread = (finite.max() - finite.min()) if finite.size else 0.0
                scale = abs(np.nanmedian(m)) + 1e-12
                grid_independent = finite.size >= 2 and spread <= 1e-6 * scale
                if grid_independent:
                    val = float(np.nanmean(m))
                    ax.plot([grids[0], grids[-1]], [val, val], ls=ls,
                            marker=mk, color=col, lw=1.8, ms=6,
                            label=f'{method} ({xlabel}-independent)')
                    sval = float(np.nanmean(s)) if (~np.isnan(s)).any() else 0.0
                    if sval > 0:
                        ax.axhspan(val - sval, val + sval, color=col, alpha=0.08)
                else:
                    ax.plot(g, m, ls=ls, marker=mk, color=col, lw=1.8, ms=6,
                            label=method)
                    if (~np.isnan(s)).any():
                        ax.fill_between(g, m - s, m + s, color=col, alpha=0.12)
            # reference lines: Planck (CMB) and SH0ES/Riess (local) - the two
            # ends of the Hubble tension, to see which one each method approaches
            for ref_name, ref in REFERENCES.items():
                if param in ref['vals']:
                    v, sg = ref['vals'][param]
                    ax.axhline(v, ls=ref['ls'], color=ref['color'], lw=1.6,
                               label=f"{ref_name}: {param}={v}")
                    ax.axhspan(v - sg, v + sg, color=ref['color'], alpha=0.06)
            ax.set_xlabel(xlabel); ax.set_xticks(grids)
            ax.set_ylabel(PARAM_LATEX.get(param, param))
            ax.set_title(f'{PARAM_LATEX.get(param, param)} vs {xlabel}')
            ax.grid(True, alpha=0.3); ax.legend(fontsize=7, loc='best')
        for j in range(len(params), nrow * ncol):
            axes[j // ncol][j % ncol].axis('off')
        fig.suptitle(f'Convergence of the estimates - model {model.upper()}',
                     fontsize=13, fontweight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        os.makedirs(outdir, exist_ok=True)
        p1 = os.path.join(outdir, f'convergence_{model}{suffix}.png')
        fig.savefig(p1, dpi=150, bbox_inches='tight')
        fig.savefig(p1.replace('.png', '.pdf'), bbox_inches='tight')
        plt.close(fig); made.append(p1)

        # --- cost figure (time + RAM) vs grid ---
        from collections import defaultdict
        times = defaultdict(list)
        for r in records:
            if r['model'] != model or np.isnan(r['time_s']):
                continue
            # [B-NBITS2] The SAME family filter the convergence figure uses.
            # Without it, the samplers cost figure also drew the genetic
            # curves —whose x axis is n_bits, not nqpp— on an axis labeled
            # 'nqpp' with the ticks placed at the other family's values: the
            # genetic points fell to the right of the last tick, unlabeled.
            # It is the same error that [B-NBITS] fixed above, which had been
            # left behind here.
            if family and _method_family(r['method']) != family:
                continue
            times[r['method']].append((r['grid'], r['time_s']))
        grids_rss = sorted([(g, v) for (mm, g), v in rss_map.items()
                            if mm == model])
        if times or grids_rss:
            fig, ax = plt.subplots(figsize=(7.2, 4.6))
            for method in sorted(times):
                pts = sorted(set(times[method]))
                col, mk, ls = style[method]
                ax.plot([p[0] for p in pts], [p[1] for p in pts], ls=ls,
                        marker=mk, color=col, lw=1.6, ms=5,
                        label=f'{method} (s)')
            ax.set_xlabel(xlabel); ax.set_ylabel('Time [s]')
            ax.set_xticks(grids); ax.grid(True, alpha=0.3)
            if grids_rss:
                ax2 = ax.twinx()
                ax2.plot([p[0] for p in grids_rss], [p[1] for p in grids_rss],
                         '-s', color='black', lw=1.8, ms=5,
                         label='peak RAM [GB]')
                ax2.set_ylabel('peak RAM [GB]')
                ax2.legend(loc='upper left', fontsize=8)
            ax.legend(loc='upper right', fontsize=8)
            fig.suptitle(f'Cost vs resolution - model {model.upper()}',
                         fontsize=12, fontweight='bold')
            fig.tight_layout()
            p2 = os.path.join(outdir, f'cost_{model}{suffix}.png')
            fig.savefig(p2, dpi=150, bbox_inches='tight')
            # [B-PDF] Also as PDF, like the convergence figure above. It was
            # missing only here, and the paper needs vector graphics.
            fig.savefig(p2.replace('.png', '.pdf'), bbox_inches='tight')
            plt.close(fig); made.append(p2)

    if made:
        print(f"  Convergence plots: {len(made)} figures in {outdir}")
    return made


# =============================================================================
# 4.  CLI
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    """Build the orchestrator's command-line parser."""
    p = argparse.ArgumentParser(
        prog='cosmo_hpc_runner.py',
        description="Parallel orchestrator for the samplers and genetic "
                    "pipelines on an HPC compute node (auto-detects cores and "
                    "RAM).",
        formatter_class=argparse.RawTextHelpFormatter)

    # --- what to run ---
    sel = p.add_mutually_exclusive_group()
    sel.add_argument('--only-samplers', action='store_true',
                     help='cosmo_modular_quantum.py only')
    sel.add_argument('--only-genetic', action='store_true',
                     help='cosmo_genetic_optimizers.py only')
    p.add_argument('--models', nargs='+', choices=ALL_MODELS, default=None,
                   help='Models to sweep (default: all)')
    p.add_argument('--rungs', nargs='+', default=None, metavar='TAG',
                   help='[E-RUNGS] samplers: run only these ladder rungs '
                        '(C-MCMC QMCMC50 QMCMC100 C-VI QVMC33 QVMC67 QVMC100). '
                        'Used to re-run only the rungs an errata fix affects.')
    p.add_argument('--qga-levels', nargs='+', type=int, default=None,
                   metavar='PCT',
                   help='[E-RUNGS] genetic: run only these QGA quantumness '
                        'levels (0 33 67 100); CGA is always included.')

    # --- node resources ---
    p.add_argument('--total-cores', type=int, default=detected_cores(),
                   help='Node cores (default: those detected on the node)')
    p.add_argument('--max-parallel', type=int, default=None,
                   help='Max concurrent processes (default: cores//threads)')
    p.add_argument('--threads-per-worker', type=int, default=None,
                   help='BLAS/Aer threads per process (default: ~cores//tasks)')
    p.add_argument('--mem-budget-gb', type=float, default=None,
                   help='Aggregate RAM budget (default: 85%% of RAM)')

    # --- samplers hyperparameters ---
    p.add_argument('--dataset', default='CC+BAO',
                   help='Shared dataset (e.g. CC+BAO+Pantheon+)')
    p.add_argument('--prior', choices=['flat', 'gaussian'], default='flat')
    p.add_argument('--steps', type=int, default=4000)
    p.add_argument('--qvmc-iter', type=int, default=300)
    p.add_argument('--nqpp', type=int, default=3)
    p.add_argument('--nqpp-sweep', nargs=2, type=int, metavar=('LO', 'HI'),
                   default=None,
                   help='Sweep nqpp from LO to HI (one samplers task per '
                        'value). Studies the effect of the grid size.')
    p.add_argument('--chains', type=int, default=6)
    p.add_argument('--shots', type=int, default=2000)

    # --- genetic hyperparameters ---
    p.add_argument('--generations', type=int, default=80)
    p.add_argument('--population-size', type=int, default=120)
    p.add_argument('--n-bits', type=int, default=6)
    p.add_argument('--nbits-sweep', nargs=2, type=int, metavar=('LO', 'HI'),
                   default=None,
                   help='Sweep n_bits from LO to HI (one genetic task per '
                        'value). The QGA analogue of --nqpp-sweep.')

    # --- [NOISE] second ablation axis ---
    cnoise.add_noise_cli(p)
    p.add_argument('--proposal-route', type=str, default='auto',
                   choices=('auto', 'amplitude', 'counts'),
                   help="readout route of the proposal engine, forwarded to "
                        "the samplers tasks. Default 'auto'.")
    p.add_argument('--no-noise-control', action='store_true',
                   help="Do NOT generate the ideal-by-counts control column "
                        "when sweeping noise. It is generated by default: "
                        "without it, comparing the ideal rung (which reads "
                        "amplitudes) against the noisy ones (which read "
                        "counts) mixes the effect of noise with that of the "
                        "change of readout operator, and in the first run of "
                        "the axis that inverted three conclusions.")
    p.add_argument('--noise-sweep', type=str, default=None, metavar='LIST',
                   help="comma-separated noise rungs, e.g. "
                        "'none,readout,full' or 'none,FakeBrisbane'. Generates "
                        "one task per rung, just as --nqpp-sweep generates "
                        "one per nqpp value: together they produce the "
                        "two-dimensional quantumness x noise matrix. Replaces "
                        "--noise when given.")

    # --- shared ---
    p.add_argument('--max-qubits', type=int, default=None, metavar='N',
                   help='[AUTO by default] Per-task qubit cap for the '
                        'SAMPLERS pipeline (nqpp*d). Left unset, the cap is '
                        'derived PURELY from the RAM this node actually has '
                        '(auto-detected) at 13.3 kB/state — you never have '
                        'to set this to use your own RAM. Pass a number only '
                        'to be MORE conservative than your RAM allows.')
    p.add_argument('--max-qubits-genetic', type=int, default=None,
                   metavar='N',
                   help='[AUTO by default] Per-task qubit cap for the '
                        'GENETIC pipeline (n_bits*d, the width of the '
                        'quantum-init circuit). Left unset, derived from '
                        'detected RAM at 16 B/state (a plain statevector — '
                        'the QGA never builds the samplers grid, so it '
                        'affords many more qubits at the same RAM). Pass a '
                        'number only to be more conservative.')
    p.add_argument('--noisy-task-hours', type=float,
                   default=cnoise.DEFAULT_NOISY_TASK_HOURS, metavar='H',
                   help='[B-TIME] Wall-clock budget per noisy GENETIC task, '
                        'in hours (default %(default)s). The density-matrix '
                        'QGA is held back by time, not RAM: it grows x4.4 per '
                        'qubit (measured: 73 s/gen at 10 q, 1425 s/gen at '
                        '12 q, ~7.7 h/gen at 14 q). This budget and '
                        '--generations set the noise-axis qubit ceiling of '
                        'the genetic pipeline. Raising it admits wider and '
                        'slower cells.')
    p.add_argument('--max-task-gb', type=float, default=None,
                   help='Max RAM per task for the clamp (default: the aggregate '
                        'budget). Together with --max-qubits it sets the '
                        'effective per-model grid ceiling.')
    p.add_argument('--strict-qubits', action='store_true',
                   help='Do not clamp: combinations exceeding the ceiling are '
                        'SKIPPED instead of having their grid lowered.')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--gpu-check', action='store_true',
                   help='print why the GPU is or is not available and exit, without running anything. Useful on a new HPC: --gpu degrades to CPU when Aer does not expose a GPU, and without this check that is only noticeable from the wall time.')
    p.add_argument('--gpu', action='store_true',
                   help='Pass --gpu to each child (a single GPU is shared: mind '
                        'the concurrency)')
    p.add_argument('--no-profile', action='store_true',
                   help='Do not pass --profile to the children')
    p.add_argument('--outdir', default=None,
                   help='Master folder (default: results/hpc_<ts>)')
    p.add_argument('--project-dir', default='.',
                   help='Folder where the project .py files live')
    p.add_argument('--dry-run', action='store_true',
                   help='Show the plan and commands, without running')

    # --- convergence plots (built in) ---
    p.add_argument('--no-plots', action='store_true',
                   help='Do not generate the convergence plots at the end')
    p.add_argument('--only-grid-methods', action='store_true',
                   help='In the plots, only methods that use the grid (VI/QVMC)')
    p.add_argument('--plot-only', metavar='DIR', default=None,
                   help='Run nothing: only (re)generate the convergence plots '
                        'of an existing master folder.')
    return p


def main() -> int:
    """Entry point: plans the campaign, runs it and generates the figures.

    Returns:
        Process exit code (0 if everything went well).
    """
    args = build_parser().parse_args()
    args.profile = not args.no_profile

    # --- plot-only mode: regenerate figures from a previous run and exit ---
    if args.plot_only:
        if not os.path.isdir(args.plot_only):
            sys.stderr.write(f"--plot-only: {args.plot_only} does not exist\n")
            return 2
        made = generate_convergence_plots(
            args.plot_only, only_grid_methods=args.only_grid_methods,
            xlabel='nqpp', family='samplers')
        made += generate_convergence_plots(
            args.plot_only, only_grid_methods=args.only_grid_methods,
            xlabel='n_bits', family='genetic', suffix='_genetic')
        if not made:
            print("Nothing to plot: you need >=2 grid values per model "
                  "(run a sweep with --nqpp-sweep).")
            return 1
        return 0

    project_dir = os.path.abspath(args.project_dir)
    for s in ('cosmo_modular_quantum.py', 'cosmo_genetic_optimizers.py'):
        if not os.path.exists(os.path.join(project_dir, s)):
            sys.stderr.write(f"Cannot find {s} in {project_dir}. "
                             f"Use --project-dir.\n")
            return 2

    ts = time.strftime('%Y%m%d_%H%M%S')
    master_dir = os.path.abspath(
        args.outdir or os.path.join('results', f'hpc_{ts}'))
    os.makedirs(master_dir, exist_ok=True)

    # --- (aggregate) RAM budget ---
    if args.mem_budget_gb:
        mem_budget_mb = args.mem_budget_gb * 1024
    elif _PSUTIL:
        # [B-CGROUP] Container limit if there is one; otherwise the node's RAM.
        mem_budget_mb = detected_memory_mb() * 0.85
    else:
        mem_budget_mb = 125 * 1024 * 0.85

    if getattr(args, 'gpu_check', False):
        from cosmo_core import gpu_diagnosis
        print(gpu_diagnosis())
        return 0

    # --- EFFECTIVE per-task qubit ceiling (for the per-model clamp) ---
    # The most restrictive of --max-qubits and the per-task RAM (--max-task-gb,
    # or, if not given, the aggregate budget as a single-task cap).
    task_mem_ceiling = (args.max_task_gb * 1024 if args.max_task_gb
                        else mem_budget_mb)
    # [B-MEM] The samplers ceiling depends on the dataset: with CC+BAO (51
    # points) a state costs 14 kB and with CC+BAO+Pantheon (1099) it costs
    # 74 kB, so the same --max-task-gb grants ~2 fewer qubits in the latter.
    plan_n_data = dataset_n_data(getattr(args, 'dataset', None))
    q_ceiling = qubit_ceiling(args.max_qubits, task_mem_ceiling, 'nqpp',
                              n_data=plan_n_data)
    # [FIX] The genetic pipeline gets its own ceiling from its own per-state
    # cost; sharing the samplers' ceiling used to clamp n_bits for 4-parameter
    # models (CPL 6 -> 4) to save memory that was never going to be used.
    q_ceiling_genetic = qubit_ceiling(args.max_qubits_genetic,
                                      task_mem_ceiling, 'n_bits')

    # [NOISE] Requested rungs, and their OWN ceilings. The noisy ceilings are
    # computed with the density-matrix memory model and are also bounded by
    # MAX_NOISY_QUBITS, which is a TIME limit and does not relax with more
    # RAM.
    if args.noise_sweep:
        noise_levels = [cnoise.canonical_level(s)
                        for s in args.noise_sweep.split(',') if s.strip()]
    else:
        noise_levels = [cnoise.canonical_level(args.noise)]
    # Validate early: a misspelled backend name must fail here, not halfway
    # through a campaign that takes hours.
    for lvl in noise_levels:
        cnoise.NoiseSpec.from_level(lvl)
    any_noisy = any(lvl != 'none' for lvl in noise_levels)

    noisy_q_ceiling = qubit_ceiling(args.max_qubits, task_mem_ceiling,
                                    'nqpp', noisy=True, n_data=plan_n_data)
    # [B-TIME] The noisy genetic optimizer is bounded by the CLOCK, not the
    # RAM: at 14 qubits a rho is 4.3 GB (fits) but ~7.7 h per generation
    # (fits in no campaign). The ceiling follows from the per-task budget and
    # from how many generations were requested.
    noisy_q_ceiling_genetic = qubit_ceiling(
        args.max_qubits_genetic, task_mem_ceiling, 'n_bits', noisy=True,
        generations=args.generations, budget_hours=args.noisy_task_hours)

    # --- build tasks (with per-model clamp) ---
    clamp_notices: List[str] = []
    tasks = build_tasks(args, master_dir, q_ceiling, clamp_notices,
                        q_ceiling_genetic=q_ceiling_genetic,
                        noise_levels=noise_levels,
                        noisy_q_ceiling=noisy_q_ceiling,
                        noisy_q_ceiling_genetic=noisy_q_ceiling_genetic)
    n_tasks = len(tasks) or 1

    # --- split the cores: J*T ~= total_cores, without oversubscribing ---
    if args.threads_per_worker:
        T = args.threads_per_worker
        J = args.max_parallel or max(1, args.total_cores // T)
    elif args.max_parallel:
        J = args.max_parallel
        T = max(1, args.total_cores // J)
    else:
        # default: as many processes as tasks (up to filling the node),
        # splitting the cores evenly.
        J = min(n_tasks, args.total_cores)
        T = max(1, args.total_cores // J)
    # final correction in case J*T exceeds the node
    if J * T > args.total_cores and T > 1:
        T = max(1, args.total_cores // J)

    print(f"Node: {args.total_cores} cores | "
          f"RAM budget: {mem_budget_mb/1024:.0f} GB | "
          f"psutil={'yes' if _PSUTIL else 'no (limited measurement)'}")
    # [AUTO] Report WHY each ceiling is what it is: derived purely from
    # detected RAM (the default, no flag needed), or lowered by an explicit
    # user override.
    # [B-BUDGETMISMATCH] A --max-task-gb larger than --mem-budget-gb is a
    # silent contradiction. The planner bounds the qubits with the former, but
    # the pool ALWAYS admits at least one task even if it does not fit in the
    # aggregate budget (otherwise a task larger than the budget would block
    # the campaign forever). So with this combination a task exceeding the
    # whole budget can be launched, and the only signal would be an OOMKill
    # hours later.
    #
    # Measured: --max-task-gb 40 --mem-budget-gb 10 with CC+BAO+Pantheon
    # granted 18 qubits, i.e. ~22 GB for ONE task, against a budget of 10.
    if args.max_task_gb and args.max_task_gb * 1024 > mem_budget_mb:
        print(f"\n  !! WARNING: --max-task-gb ({args.max_task_gb:.0f} GB) is LARGER "
              f"than the aggregate budget ({mem_budget_mb/1024:.0f} GB).\n"
              f"     The qubit ceiling comes from --max-task-gb, but the pool "
              f"always admits at least\n"
              f"     one task even if it does not fit in the budget: you may "
              f"launch a task that\n"
              f"     overflows it and end in an OOMKill. Lower --max-task-gb to "
              f"{mem_budget_mb/1024:.0f} GB or less, or raise --mem-budget-gb.\n")

    src_s = (f"user override --max-qubits={args.max_qubits}"
             if args.max_qubits is not None else
             "auto, derived from detected RAM (no --max-qubits set)")
    print(f"Grid ceiling (samplers): {q_ceiling} qubits/task  [{src_s}]\n"
          f"    ({task_mem_ceiling/1024:.0f} GB available -> "
          f"{qubits_fitting_in(task_mem_ceiling, 'nqpp', plan_n_data)}q fit at "
          f"{bytes_per_state_samplers(plan_n_data)/1024:.1f} kB/state, "
          f"N_data={plan_n_data})")
    if not args.only_samplers:
        src_g = (f"user override --max-qubits-genetic={args.max_qubits_genetic}"
                 if args.max_qubits_genetic is not None else
                 "auto, derived from detected RAM (no --max-qubits-genetic set)")
        print(f"Grid ceiling (genetic):  {q_ceiling_genetic} qubits/task  [{src_g}]\n"
              f"    ({task_mem_ceiling/1024:.0f} GB available -> "
              f"{qubits_fitting_in(task_mem_ceiling, 'n_bits')}q fit at "
              f"{BYTES_PER_STATE_GENETIC} B/state)")
    # [NOISE] Report the axis and WHY its ceiling is different: it is not set
    # by RAM alone but by the density-matrix simulation cost.
    if any_noisy:
        print(f"Noise axis: {', '.join(noise_levels)}")
        f_s = cnoise.param_shift_batch_factor(max(noisy_q_ceiling, 1))
        print(f"    noisy ceiling (derived from RAM, not a constant):")
        print(f"      samplers  {noisy_q_ceiling}q  -- set by the "
              f"parameter-shift BATCH of QVMC quantum training "
              f"({f_s}x rho = "
              f"{cnoise.noisy_density_bytes(noisy_q_ceiling, f_s)/2**30:.0f} "
              f"GB), not a single rho "
              f"({cnoise.noisy_density_bytes(noisy_q_ceiling)/2**30:.2f} GB)")
        print(f"      genetic   {noisy_q_ceiling_genetic}q  -- one rho per "
              f"operator, no batch")
        print(f"      QMCMC is not listed: its engine uses max(2,d) qubits "
              f"(2-4) and the acceptance 1, so nqpp does not touch its "
              f"circuits and noise comes free at any resolution.")
    print(f"Split: J={J} processes x T={T} threads = {J*T} cores "
          f"(of {args.total_cores})")
    print(f"Master folder: {master_dir}")
    if clamp_notices:
        print("Per-model grid adjustments (clamp):")
        for n in clamp_notices:
            print(f"  * {n}")

    if args.dry_run:
        print(f"\n--- DRY RUN: {len(tasks)} commands ---")
        for t in tasks:
            cmd = [sys.executable, os.path.join(project_dir, t.script)] + t.argv
            _cap = (q_ceiling_genetic if t.grid_kind == 'n_bits'
                    else q_ceiling)
            flag = (f"  (SKIP: exceeds the {_cap}q cap)"
                    if t.total_qubits > _cap else "")
            print(f"\n# {t.name}  ({t.grid_kind}={t.grid_value}, "
                  f"~{t.total_qubits}q, ~{t.est_mem_mb:.0f} MB){flag}\n"
                  f"OMP_NUM_THREADS={T} {shlex.join(cmd)}")
        return 0

    t_wall0 = time.time()
    run_pool(tasks, max_parallel=J, threads_per_worker=T,
             mem_budget_mb=mem_budget_mb,
             # [FIX] pass the COMPUTED ceilings (always concrete ints), not
             # the raw CLI args (which default to None = auto).
             max_qubits=q_ceiling, project_dir=project_dir,
             max_qubits_genetic=q_ceiling_genetic)
    report(tasks, master_dir, t_wall0)

    # --- convergence plots at the end of ALL the runs ---
    if not args.no_plots:
        print("\nGenerating convergence plots...")
        # [B-NBITS] Two figures, one per family: the genetic x axis is n_bits
        # and the samplers one is nqpp, and mixing them on a single axis
        # labeled 'nqpp' put two different quantities side by side.
        generate_convergence_plots(
            master_dir, only_grid_methods=args.only_grid_methods,
            xlabel='nqpp', family='samplers')
        generate_convergence_plots(
            master_dir, only_grid_methods=args.only_grid_methods,
            xlabel='n_bits', family='genetic', suffix='_genetic')
        # [PLOT-NOISE] Only produces something if there was more than one
        # rung; with an ideal run it skips itself and does not clutter the
        # folder.
        if any_noisy or len(noise_levels) > 1:
            print("\nGenerating noise-comparison plots...")
            generate_noise_comparison_plots(master_dir)

    return 0 if all(t.rc in (0, -2) for t in tasks) else 1


if __name__ == '__main__':
    sys.exit(main())
