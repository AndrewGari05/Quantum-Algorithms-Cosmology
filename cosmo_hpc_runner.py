#!/usr/bin/env python3
# =============================================================================
#  cosmo_hpc_runner.py — Parallel HPC orchestrator (generic, multi-node)
# =============================================================================
#
#  Designed to run on a SUPERCOMPUTER / compute node: it auto-detects the cores
#  and RAM available on whatever node it lands on and distributes the work
#  accordingly. Nothing is hard-wired to a specific machine.
#
#  WHAT IT DOES
#  ------------
#  Runs the project's two pipelines IN PARALLEL:
#
#      1) cosmo_modular_quantum.py    (QMCMC + QVMC, quantumness ladder)
#      2) cosmo_genetic_optimizers.py (CGA + QGA, global optimization / MAP)
#
#  without modifying a single line of those scripts: it invokes their existing,
#  tested CLI (--sweep-all --sweep-models <model>), ONE MODEL PER PROCESS, so
#  that each (script x model) combination is an independent task running in its
#  own Python interpreter.
#
#  WHY multiprocessing (subprocess) AND NOT multithreading
#  -------------------------------------------------------
#  * Python's GIL serializes all pure-Python code (the MCMC loop, the GA
#    generational loop, circuit construction). Threads give you NO real
#    parallelism for that part. Separate processes = separate GILs = real
#    parallelism + crash isolation (if one model blows up, the rest continue).
#  * The heavy compute (NumPy/BLAS and Qiskit-Aer in C++) is ALREADY
#    multi-threaded internally via OpenMP. The real HPC risk is not "too few
#    threads", it is OVERSUBSCRIPTION: if you launch W processes and each lets
#    its BLAS/Aer grab all 80 cores, you end up with W*80 threads fighting over
#    80 cores -> cache thrashing and the "parallel" version runs SLOWER.
#
#  THE KEY: PARTITION THE CORES
#  ----------------------------
#  This orchestrator fixes, BEFORE starting each subprocess, the number of
#  internal BLAS/OpenMP/Aer threads via environment variables (OMP_NUM_THREADS,
#  etc.). With J concurrent processes and T threads each, it enforces J*T ~=
#  total cores. Those env vars are read when NumPy/Qiskit are imported, which is
#  why they must be set in a NEW interpreter (subprocess), not in threads of the
#  same process.
#
#  WHAT IT RETURNS
#  ---------------
#  It measures, per task and in aggregate: wall time and peak RAM of the process
#  TREE (child process + its descendants), prints it as a table and saves it to
#  master_profile.csv / .json. It also passes --profile to each child so each
#  one produces its own resource_usage_*.png / profile_*.json. When a child
#  wrote a profile_*.json, the orchestrator reports the MEASURED memory from
#  cosmo_profiling rather than its own sampled estimate.
#
#  TYPICAL USE ON A COMPUTE NODE
#  -----------------------------
#      # auto-detects the node's cores/RAM
#      python cosmo_hpc_runner.py \
#          --dataset CC+BAO+Pantheon+ \
#          --steps 15000 --qvmc-iter 3000 --nqpp 3 \
#          --generations 120 --population-size 200 --n-bits 6 \
#          --threads-per-worker 8
#
#  GRID SWEEP (convergence study)
#  ------------------------------
#      # one task per nqpp in {2,3,4,5} per model (QVMC/QMCMC):
#      python cosmo_hpc_runner.py --nqpp-sweep 2 5 --only-samplers \
#          --models lcdm wcdm --dataset CC+BAO+Pantheon+
#      # and for the genetic optimizer (QGA), sweep the grid size n_bits:
#      python cosmo_hpc_runner.py --nbits-sweep 3 6 --only-genetic --models lcdm
#
#  REMEMBER: nqpp does NOT depend on the core count, it depends on RAM and on d
#  (the number of parameters): the grid costs 2^(nqpp*d). Combinations exceeding
#  --max-qubits are clamped down per model (or skipped with --strict-qubits);
#  raise --max-qubits only if the RAM can take it
#  (18q~=3.5GB . 20q~=14GB . 22q~=56GB . 24q~=224GB).
#
#  Samplers only / genetic only:  --only-samplers  /  --only-genetic
#  Dry run (see what it would launch without running):  --dry-run
# =============================================================================

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
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
#   of shape (n_states, N_data) over it. Uses EXACTLY the same constant as
#   the validation in cosmo_modular_quantum.py (_validate_args:
#   2**total_q * 1660 * 8), to avoid having two different magic numbers. The
#   1660 floats/state are those auxiliary arrays; x8 bytes (float64).
#   (This also covers the QVMC training batch, which materializes 2*n_phi
#   statevectors at 16 B/state — smaller than the auxiliary arrays at every
#   nqpp the cap allows, so this constant stays the binding one.)
BYTES_PER_STATE_SAMPLERS = 1660 * 8
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

# Kind -> per-state cost, so callers can stay declarative.
BYTES_PER_STATE_BY_KIND = {
    'nqpp': BYTES_PER_STATE_SAMPLERS,     # samplers tasks
    'n_bits': BYTES_PER_STATE_GENETIC,    # genetic tasks
}
#
# NOISY (tercer modelo de memoria — segundo eje de ablacion):
#   Simular con ruido exige matriz de densidad: 2^(2n) amplitudes complejas en
#   vez de 2^n, es decir 16 * 4^n bytes. No SUSTITUYE a los dos modelos de
#   arriba, se SUMA a ellos: la tarea de samplers sigue construyendo su grid de
#   verosimilitud (1660 floats/estado) y ademas mantiene rho.
#
#   El termino de rho domina en cuanto n crece:
#       n=10 ->  16.0 MB de rho   frente a  13.6 MB de grid
#       n=12 -> 256.0 MB de rho   frente a  54.4 MB de grid
#       n=13 ->   1.0 GB de rho   frente a 108.9 MB de grid
#
#   Pero el limitante REAL de este eje no es la memoria sino el TIEMPO, y por
#   eso el techo (cosmo_noise.MAX_NOISY_QUBITS = 13) es una constante explicita
#   y no un valor derivado de la RAM detectada como los otros dos. Medido con
#   el ansatz real (3 capas, B=2 bindings, 4096 disparos, FakeBrisbane):
#       n=10 ->   3.8 s/job      n=12 ->  34.4 s/job      n=13 -> 216.5 s/job
#   A n=14 la matriz de densidad son 4.0 GB y ~14 min por job; una maquina con
#   mas RAM no mueve ese limite, solo lo hace mas caro.
BYTES_PER_STATE_NOISY_NOTE = (
    "rho = 16 * 4^n bytes, sumado al modelo de la tarea; techo por TIEMPO")

# Backwards-compatible alias (any external script importing the old name keeps
# the samplers semantics it used to get).
BYTES_PER_STATE = BYTES_PER_STATE_SAMPLERS


@dataclass
class Task:
    name: str                       # human-readable label
    script: str                     # cosmo_modular_quantum.py | cosmo_genetic_optimizers.py
    argv: List[str]                 # CLI arguments (without 'python' or the script)
    model: str
    total_qubits: int               # nqpp*d (samplers) or n_bits*d (genetic)
    est_mem_mb: float               # estimated peak RAM of the task
    outdir: str
    grid_value: int = 0             # EFFECTIVE nqpp (samplers) or n_bits (genetic)
    grid_kind: str = ""             # 'nqpp' | 'n_bits'
    # [NOISE] Segunda dimension del eje de ablacion, tratada como una mas
    # (igual que nqpp y n_bits): etiqueta canonica del peldano de ruido.
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
        if self.t_start and self.t_end:
            return self.t_end - self.t_start
        return 0.0


# =============================================================================
# 1.  BUILDING THE TASK LIST
# =============================================================================

def estimate_qubits_and_mem(total_q: int, kind: str = 'nqpp',
                            noisy: bool = False) -> float:
    """Estimated peak RAM (MB) of a task whose circuit/grid has 2^total_q
    states, plus the fixed per-process interpreter cost.

    `kind` selects the per-state cost: 'nqpp' (samplers — grid + likelihood
    auxiliaries) or 'n_bits' (genetic — one plain statevector). See the
    BYTES_PER_STATE_* comments: using the samplers constant for the QGA
    over-estimated it by ~830x.

    Args:
        total_q: total circuit/grid qubits (nqpp*d or n_bits*d).
        kind: 'nqpp' | 'n_bits'.
        noisy: si True, ANADE la matriz de densidad (16 * 4^total_q). Es un
            tercer modelo aditivo, no un sustituto: la tarea de samplers sigue
            necesitando su grid ademas de rho.
    """
    per_state = BYTES_PER_STATE_BY_KIND.get(kind, BYTES_PER_STATE_SAMPLERS)
    mb = (2 ** total_q) * per_state / 1e6 + PROCESS_BASELINE_MB
    if noisy:
        # [REV] Con entrenamiento cuantico el coste no es rho suelta sino el
        # lote de parameter-shift (2*n_phi matrices de densidad a la vez), que
        # a 12 qubits son 42 GB frente a los 256 MB de una sola. El genetico
        # no tiene ese lote.
        factor = (cnoise.param_shift_batch_factor(total_q)
                  if kind == 'nqpp' else 1)
        mb += cnoise.noisy_density_bytes(total_q, factor) / 1e6
    return mb


def qubits_fitting_in(mem_mb: float, kind: str = 'nqpp') -> int:
    """Largest number of qubits whose 2^q states fit in mem_mb, for the given
    pipeline `kind` (the per-process baseline is reserved first)."""
    usable = mem_mb - PROCESS_BASELINE_MB
    if usable <= 0:
        return 0
    per_state = BYTES_PER_STATE_BY_KIND.get(kind, BYTES_PER_STATE_SAMPLERS)
    cap_states = usable * 1e6 / per_state
    q = 0
    while 2 ** (q + 1) <= cap_states:
        q += 1
    return q


def qubit_ceiling(max_qubits: Optional[int], mem_ceiling_mb: float,
                  kind: str = 'nqpp', noisy: bool = False) -> int:
    """EFFECTIVE per-task qubit ceiling for the given pipeline `kind`.

    [AUTO] `max_qubits=None` (the default — see build_parser) means "no user
    cap": the ceiling is derived PURELY from the RAM this node actually has,
    auto-detected via psutil. Pass an explicit --max-qubits[-genetic] only to
    be MORE conservative than what your RAM allows (e.g. a shared node where
    you want to leave headroom, or you want a faster/coarser run on purpose).
    You can never exceed what fits in RAM this way — it stays the ultimate
    backstop regardless of what you pass.

    [NOISE] Con `noisy=True` se aplica ADEMAS el techo del eje de ruido, que
    tambien se deriva de la RAM — igual que los otros dos, y a diferencia de
    lo que hacia la primera version de este eje, que lo fijaba en 13 duro
    argumentando un limite de tiempo. Era una mala generalizacion sacada de
    una maquina pequena: en un nodo grande y sin prisa el limitante vuelve a
    ser la memoria, y esa si se relaja.

    Lo que manda con ruido no es rho suelta sino el LOTE de parameter-shift
    del entrenamiento cuantico del QVMC (`2 * n_phi` matrices de densidad a la
    vez). Como una tarea de samplers recorre la escalera entera, ese peldano
    es el que fija su techo. El genetico no tiene ese lote y por eso recibe
    `quantum_training=False`.

    Args:
        max_qubits: tope pedido por el usuario, o None.
        mem_ceiling_mb: RAM disponible por tarea.
        kind: 'nqpp' | 'n_bits'.
        noisy: aplica el techo del eje de ruido.
    """
    ram_ceiling = qubits_fitting_in(mem_ceiling_mb, kind)
    ceiling = (ram_ceiling if max_qubits is None
               else min(max_qubits, ram_ceiling))
    if noisy:
        ceiling = min(ceiling, cnoise.noisy_qubit_ceiling(
            requested=None, mem_mb=mem_ceiling_mb,
            quantum_training=(kind == 'nqpp')))
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

    [NOISE] El nivel de ruido es una dimension de tarea MAS, exactamente igual
    que nqpp y n_bits: `--noise-sweep none,readout,full` genera la matriz
    bidimensional (quantumness x ruido) en una sola invocacion. Cada peldano
    ruidoso lleva su PROPIO techo de qubits y su propio modelo de memoria,
    porque la matriz de densidad cambia ambos: el techo baja a 13 y el coste
    de RAM gana un termino 16*4^n que domina al del grid.

    Args:
        args: namespace del parser del runner.
        master_dir: carpeta maestra de la corrida.
        q_ceiling: techo de qubits de samplers SIN ruido.
        notices: lista donde acumular avisos de recorte.
        q_ceiling_genetic: techo de qubits genetico SIN ruido.
        noise_levels: peldanos a generar. None equivale a `['none']`.
        noisy_q_ceiling: techo de samplers CON ruido.
        noisy_q_ceiling_genetic: techo genetico CON ruido.

    Returns:
        Lista de tareas.
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

    # [NOISE-CONTROL] La columna de control.
    #
    # El peldano ideal usa por defecto la ruta de amplitudes y cualquier
    # peldano con ruido usa la ruta por conteos, porque Re(psi)*sign(Im(psi))
    # no existe para un estado mezclado. Comparar la columna ideal contra las
    # ruidosas mezcla entonces DOS efectos: el ruido y el cambio de operador
    # de lectura.
    #
    # No es teorico. En la primera corrida del eje, leer 'none' -> 'readout'
    # daba que el ruido de lectura ESTRECHA el posterior y MEJORA el mezclado
    # (sigma 0.0208 -> 0.0176, ESS 101 -> 141). Con el control se ve que
    # ideal-por-conteos y readout son identicas: el canal de lectura no toca
    # la propuesta y todo el salto era el cambio de ruta.
    #
    # Por eso el control se anade SOLO — una barrida de ruido sin el produce
    # una matriz que invita a conclusiones invertidas. `--no-noise-control`
    # lo desactiva para quien lo quiera explicitamente.
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
        # El control comparte el peldano 'none' pero es una columna propia:
        # se etiqueta distinto para que no colisione en el CSV ni en disco.
        noise_col = 'none-counts' if proposal_route == 'counts' and not noisy \
            else noise
        # [NOISE] Techos y modelo de memoria propios del peldano.
        q_cap = (noisy_q_ceiling if noisy and noisy_q_ceiling is not None
                 else q_ceiling)
        g_cap_base = (q_ceiling if q_ceiling_genetic is None
                      else q_ceiling_genetic)
        g_cap = (noisy_q_ceiling_genetic
                 if noisy and noisy_q_ceiling_genetic is not None
                 else g_cap_base)
        # El peldano solo entra en el nombre/carpeta cuando hay mas de uno, de
        # modo que una corrida ideal produce EXACTAMENTE las mismas rutas de
        # salida que antes de este eje y los CSV previos siguen alineando.
        noise_argv = (['--noise', noise] if noisy or sweeping_noise else [])
        route_argv = (['--proposal-route', proposal_route]
                      if proposal_route != 'auto' else [])

        for m in models:
            d = MODEL_DIM[m]

            # ---- Samplers tasks (QMCMC + QVMC), one per nqpp value ----
            if not args.only_genetic:
                for nqpp in grid_values_for_model(
                        args.nqpp, args.nqpp_sweep, d, q_cap, strict,
                        notices, 'nqpp', m):
                    total_q = nqpp * d
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
                            '--max-qubits', str(q_cap),
                            '--outdir', outdir] + common_data + noise_argv \
                        + route_argv
                    tasks.append(Task(
                        name=name, script='cosmo_modular_quantum.py',
                        argv=argv, model=m, total_qubits=total_q,
                        est_mem_mb=estimate_qubits_and_mem(total_q, 'nqpp',
                                                           noisy=noisy),
                        outdir=outdir, grid_value=nqpp, grid_kind='nqpp',
                        noise=noise_col))

            # ---- Genetic tasks (CGA + QGA), one per n_bits value ----
            # [NOISE-CONTROL] El control es una columna de la PROPUESTA del
            # QMCMC; el QGA no tiene ruta de lectura conmutable, asi que
            # duplicarlo aqui solo repetiria la columna ideal.
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
                            '--outdir', outdir] + common_data + noise_argv
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
    aggregate RAM budget, and sample each tree's peak RSS."""
    pending = list(tasks)
    running: List[Task] = []
    skipped: List[Task] = []
    procs: Dict[int, subprocess.Popen] = {}

    def admitted_mem() -> float:
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
            # [NOISE] Segunda coordenada del eje de ablacion. Se escribe
            # SIEMPRE, tambien en corridas ideales ('none'), para que la
            # matriz bidimensional se pueda pivotar sin casos especiales.
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
    try:
        return float(s)
    except (TypeError, ValueError):
        return float('nan')


def _find_result_csvs(master_dir: str) -> List[str]:
    import glob
    found, seen, out = [], set(), []
    for pat in ('resultados_TODOS_los_modelos.csv', 'resultados_config.csv'):
        found += glob.glob(os.path.join(master_dir, '**', pat), recursive=True)
    for f in found:
        if f not in seen:
            seen.add(f); out.append(f)
    return out


def _infer_model(path: str) -> str:
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


def _is_grid_method(method: str) -> bool:
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
    """Peldano de ruido de una tarea, deducido del nombre de su carpeta.

    Las carpetas de tarea se llaman `<pipeline>_<modelo>[_<tag>]_noise-<nivel>`
    cuando hay barrido de ruido, y sin el sufijo cuando no lo hay. Esa ausencia
    significa 'none' — una corrida ideal conserva las rutas de siempre.

    Args:
        path: ruta del CSV de resultados.

    Returns:
        Etiqueta del peldano.
    """
    for part in path.split(os.sep):
        if '_noise-' in part:
            return part.split('_noise-')[-1]
    return 'none'


def _noise_sort_key(level: str):
    """Orden del eje: los peldanos con nombre primero, backends al final."""
    return (NOISE_ORDER.index(level) if level in NOISE_ORDER
            else len(NOISE_ORDER), level)


def _parse_noise_rows(csv_paths: List[str]) -> List[dict]:
    """Aplana los CSV en registros {model, method, noise, param, mean, std, ...}.

    A diferencia de `_parse_result_rows`, NO exige la columna `nqpp`: las
    tareas geneticas no la tienen y aqui si interesan, porque el QGA es una de
    las escaleras que el eje de ruido debe comparar.
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
    return [r for r in out if r['mean'] == r['mean']]      # descarta NaN


def generate_noise_comparison_plots(master_dir: str,
                                    outdir: Optional[str] = None
                                    ) -> List[str]:
    """Una figura por modelo comparando CADA rung con y sin ruido.

    [PLOT-NOISE] Responde directamente a "quiero ver cada quantumness de cada
    modelo con y sin ruido". El eje x es el peldano de ruido en orden, cada
    linea es un rung de quantumness, y hay un panel por parametro mas uno de
    calidad de ajuste. Asi la degradacion de un rung se lee como la pendiente
    de su propia linea, y la comparacion entre rungs como la separacion entre
    lineas — las dos preguntas del eje, en una sola imagen.

    Se usa una linea por rung y no un panel por rung a proposito: lo que
    interesa no es la forma de cada curva por separado sino si unos rungs
    aguantan el ruido mejor que otros, y eso solo se ve superponiendolos.

    La columna `none-counts`, cuando existe, se dibuja como parte del eje: es
    el control que separa el efecto del ruido del efecto del cambio de
    operador de lectura, y sin ella la pendiente entre `none` y `readout`
    mezcla ambos.

    Args:
        master_dir: carpeta maestra de la corrida.
        outdir: donde escribir (por defecto, la misma).

    Returns:
        Lista de rutas generadas.
    """
    try:
        import numpy as np
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as e:                                  # pragma: no cover
        print(f"  (sin graficas de ruido: no pude importar matplotlib: {e})")
        return []

    outdir = outdir or master_dir
    records = _parse_noise_rows(_find_result_csvs(master_dir))
    if not records:
        print("  (sin graficas de ruido: no encontre filas legibles)")
        return []

    levels = sorted({r['noise'] for r in records}, key=_noise_sort_key)
    if len(levels) < 2:
        print(f"  (sin graficas de ruido: solo un peldano, '{levels[0]}')")
        return []

    # [PLOT-NOISE] Samplers y genetico van en figuras SEPARADAS, y no por
    # estetica: sus sigmas no son la misma cantidad.
    #
    # Para MCMC/VI, sigma es la anchura del posterior — un intervalo de
    # credibilidad — asi que "desplazamiento en unidades de sigma" responde
    # "se movio mas de lo que el metodo sabe del parametro?".
    #
    # Para un genetico, sigma es la dispersion de una poblacion YA CONVERGIDA:
    # una cantidad diminuta que solo mide cuanto se apretaron los individuos.
    # Dividir por ella produce desplazamientos de 5 o 9 sigmas que no
    # significan nada — se vieron en la primera corrida y parecian un efecto
    # enorme del ruido cuando eran un denominador casi nulo. El genetico se
    # dibuja por tanto en unidades ABSOLUTAS, que es lo que tiene sentido para
    # un optimizador cuyo resultado es un punto.
    def _is_genetic(method: str) -> bool:
        m = (method or '').upper()
        return m.startswith('CGA') or m.startswith('QGA')

    made: List[str] = []
    for model in sorted({r['model'] for r in records}):
        for kind, keep, suffix in (
                ('samplers', lambda m: not _is_genetic(m), ''),
                ('genetic', _is_genetic, '_genetic')):
            sub = [r for r in records if r['model'] == model
                   and keep(r['method'])]
            if sub:
                pth = _one_noise_figure(model, sub, levels, outdir, suffix,
                                        pull=(kind == 'samplers'),
                                        np=np, plt=plt)
                if pth:
                    made.append(pth)
                    print(f"  . {model} [{kind}]: {pth}")
    return made


def _one_noise_figure(model, rows, levels, outdir, suffix, pull, np, plt):
    """Dibuja UNA figura de comparacion de ruido.

    Args:
        model: nombre del modelo.
        rows: registros de un solo pipeline (samplers o genetico).
        levels: peldanos de ruido, ya ordenados.
        outdir: carpeta de salida.
        suffix: sufijo del nombre de archivo ('' o '_genetic').
        pull: True para dibujar el desplazamiento en unidades de sigma (solo
            tiene sentido cuando sigma es la anchura de un posterior); False
            para valores absolutos (genetico).
        np, plt: modulos ya importados por el llamador.

    Returns:
        Ruta de la figura, o cadena vacia si no habia nada que dibujar.
    """
    if True:
        params, seen = [], set()
        for r in rows:
            if r['param'] not in seen:
                seen.add(r['param']); params.append(r['param'])
        methods = sorted({r['method'] for r in rows})
        if not params or not methods:
            return ""

        # Metrica de calidad: la KL si el modelo la produce (QVMC), si no chi2.
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
        # Color y marcador por metodo, fijos en todos los paneles: la identidad
        # nunca depende solo del color.
        cyc = ['#1f77b4', '#d62728', '#ff7f0e', '#2ca02c', '#9467bd',
               '#17becf', '#8c564b', '#e377c2']
        mks = ['o', 's', '^', 'D', 'v', 'P', 'X', '*']
        style = {m: (cyc[i % len(cyc)], mks[i % len(mks)])
                 for i, m in enumerate(methods)}

        def series(method, key, sub=None):
            ys, es = [], []
            for lvl in levels:
                sel = [r for r in rows if r['method'] == method
                       and r['noise'] == lvl
                       and (sub is None or r['param'] == sub)]
                ys.append(sel[-1][key] if sel else float('nan'))
                es.append(sel[-1].get('std', float('nan')) if sel
                          else float('nan'))
            return np.array(ys, float), np.array(es, float)

        # [PLOT-NOISE] Los paneles de parametro muestran el DESPLAZAMIENTO en
        # unidades de sigma respecto al peldano ideal, no el valor absoluto.
        #
        # Con el valor absoluto la barra de sigma del posterior (~0.016 en Om)
        # es un orden de magnitud mayor que lo que el ruido mueve la media
        # (~0.002), asi que todas las lineas se aplastan en una banda comun y
        # la figura no distingue un rung de otro. La pregunta real no es
        # "cuanto vale Om" — eso ya esta en los corner plots — sino "cuanto lo
        # movio el ruido comparado con lo que el metodo sabe de el". Eso es
        # exactamente (mean - mean_ideal) / sigma_ideal.
        #
        # La banda gris de +-1 sigma da la escala: una linea dentro de ella se
        # movio menos que la propia incertidumbre del metodo, es decir, el
        # ruido no la desplazo de forma detectable.
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
            f"{model} — every quantumness rung, with and without noise\n"
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
                               xlabel: str = 'nqpp') -> List[str]:
    """Generate convergence_<model>.png and cost_<model>.png for each model with
    >=2 grid values. Returns the paths created. Does not raise if matplotlib is
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
        grids = sorted({r['grid'] for r in records if r['model'] == model})
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
        p1 = os.path.join(outdir, f'convergence_{model}.png')
        fig.savefig(p1, dpi=150, bbox_inches='tight')
        fig.savefig(p1.replace('.png', '.pdf'), bbox_inches='tight')
        plt.close(fig); made.append(p1)

        # --- cost figure (time + RAM) vs grid ---
        from collections import defaultdict
        times = defaultdict(list)
        for r in records:
            if r['model'] == model and not np.isnan(r['time_s']):
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
            p2 = os.path.join(outdir, f'cost_{model}.png')
            fig.savefig(p2, dpi=150, bbox_inches='tight')
            plt.close(fig); made.append(p2)

    if made:
        print(f"  Convergence plots: {len(made)} figures in {outdir}")
    return made


# =============================================================================
# 4.  CLI
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
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

    # --- node resources ---
    p.add_argument('--total-cores', type=int, default=os.cpu_count() or 1,
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

    # --- [NOISE] segundo eje de ablacion ---
    cnoise.add_noise_cli(p)
    p.add_argument('--proposal-route', type=str, default='auto',
                   choices=('auto', 'amplitude', 'counts'),
                   help="ruta de lectura del motor de propuesta, reenviada a "
                        "las tareas de samplers. Por defecto 'auto'.")
    p.add_argument('--no-noise-control', action='store_true',
                   help="NO generar la columna de control ideal-por-conteos "
                        "al barrer ruido. Por defecto se genera: sin ella, "
                        "comparar el peldano ideal (que lee amplitudes) "
                        "contra los ruidosos (que leen conteos) mezcla el "
                        "efecto del ruido con el del cambio de operador de "
                        "lectura, y en la primera corrida del eje eso "
                        "invirtio tres conclusiones.")
    p.add_argument('--noise-sweep', type=str, default=None, metavar='LISTA',
                   help="peldanos de ruido separados por comas, p. ej. "
                        "'none,readout,full' o 'none,FakeBrisbane'. Genera "
                        "una tarea por peldano, igual que --nqpp-sweep genera "
                        "una por valor de nqpp: juntos producen la matriz "
                        "bidimensional quantumness x ruido. Sustituye a "
                        "--noise cuando se da.")

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
    p.add_argument('--max-task-gb', type=float, default=None,
                   help='Max RAM per task for the clamp (default: the aggregate '
                        'budget). Together with --max-qubits it sets the '
                        'effective per-model grid ceiling.')
    p.add_argument('--strict-qubits', action='store_true',
                   help='Do not clamp: combinations exceeding the ceiling are '
                        'SKIPPED instead of having their grid lowered.')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--gpu-check', action='store_true',
                   help='imprime por que la GPU esta o no disponible y sale, sin correr nada. Util en una HPC nueva: --gpu degrada a CPU cuando Aer no expone GPU, y sin este chequeo eso solo se nota por el tiempo de pared.')
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
    args = build_parser().parse_args()
    args.profile = not args.no_profile

    # --- plot-only mode: regenerate figures from a previous run and exit ---
    if args.plot_only:
        if not os.path.isdir(args.plot_only):
            sys.stderr.write(f"--plot-only: {args.plot_only} does not exist\n")
            return 2
        made = generate_convergence_plots(
            args.plot_only, only_grid_methods=args.only_grid_methods)
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
        mem_budget_mb = psutil.virtual_memory().total / 1e6 * 0.85
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
    q_ceiling = qubit_ceiling(args.max_qubits, task_mem_ceiling, 'nqpp')
    # [FIX] The genetic pipeline gets its own ceiling from its own per-state
    # cost; sharing the samplers' ceiling used to clamp n_bits for 4-parameter
    # models (CPL 6 -> 4) to save memory that was never going to be used.
    q_ceiling_genetic = qubit_ceiling(args.max_qubits_genetic,
                                      task_mem_ceiling, 'n_bits')

    # [NOISE] Peldanos pedidos, y sus techos PROPIOS. Los techos ruidosos se
    # calculan con el modelo de memoria de la matriz de densidad y ademas
    # quedan acotados por MAX_NOISY_QUBITS, que es un limite de TIEMPO y no se
    # relaja con mas RAM.
    if args.noise_sweep:
        noise_levels = [cnoise.canonical_level(s)
                        for s in args.noise_sweep.split(',') if s.strip()]
    else:
        noise_levels = [cnoise.canonical_level(args.noise)]
    # Validar temprano: un nombre de backend mal escrito debe fallar aqui, no
    # a mitad de una campana de horas.
    for lvl in noise_levels:
        cnoise.NoiseSpec.from_level(lvl)
    any_noisy = any(lvl != 'none' for lvl in noise_levels)

    noisy_q_ceiling = qubit_ceiling(args.max_qubits, task_mem_ceiling,
                                    'nqpp', noisy=True)
    noisy_q_ceiling_genetic = qubit_ceiling(args.max_qubits_genetic,
                                            task_mem_ceiling, 'n_bits',
                                            noisy=True)

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
    src_s = (f"user override --max-qubits={args.max_qubits}"
             if args.max_qubits is not None else
             "auto, derived from detected RAM (no --max-qubits set)")
    print(f"Grid ceiling (samplers): {q_ceiling} qubits/task  [{src_s}]\n"
          f"    ({task_mem_ceiling/1024:.0f} GB available -> "
          f"{qubits_fitting_in(task_mem_ceiling, 'nqpp')}q fit at "
          f"{BYTES_PER_STATE_SAMPLERS/1024:.1f} kB/state)")
    if not args.only_samplers:
        src_g = (f"user override --max-qubits-genetic={args.max_qubits_genetic}"
                 if args.max_qubits_genetic is not None else
                 "auto, derived from detected RAM (no --max-qubits-genetic set)")
        print(f"Grid ceiling (genetic):  {q_ceiling_genetic} qubits/task  [{src_g}]\n"
              f"    ({task_mem_ceiling/1024:.0f} GB available -> "
              f"{qubits_fitting_in(task_mem_ceiling, 'n_bits')}q fit at "
              f"{BYTES_PER_STATE_GENETIC} B/state)")
    # [NOISE] Reportar el eje y POR QUE su techo es distinto: no lo fija la
    # RAM sino el tiempo de simulacion con matriz de densidad.
    if any_noisy:
        print(f"Eje de ruido: {', '.join(noise_levels)}")
        f_s = cnoise.param_shift_batch_factor(max(noisy_q_ceiling, 1))
        print(f"    techo con ruido (derivado de la RAM, no una constante):")
        print(f"      samplers  {noisy_q_ceiling}q  -- lo fija el LOTE de "
              f"parameter-shift del entrenamiento cuantico del QVMC "
              f"({f_s}x rho = "
              f"{cnoise.noisy_density_bytes(noisy_q_ceiling, f_s)/2**30:.0f} "
              f"GB), no rho suelta "
              f"({cnoise.noisy_density_bytes(noisy_q_ceiling)/2**30:.2f} GB)")
        print(f"      genetico  {noisy_q_ceiling_genetic}q  -- una rho por "
              f"operador, sin lote")
        print(f"      QMCMC no aparece: su motor usa max(2,d) qubits (2-4) y "
              f"la aceptacion 1, asi que nqpp no toca sus circuitos y el "
              f"ruido le sale gratis a cualquier resolucion.")
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
        generate_convergence_plots(
            master_dir, only_grid_methods=args.only_grid_methods)
        # [PLOT-NOISE] Solo produce algo si hubo mas de un peldano; con una
        # corrida ideal se salta sola y no ensucia la carpeta.
        if any_noisy or len(noise_levels) > 1:
            print("\nGenerating noise-comparison plots...")
            generate_noise_comparison_plots(master_dir)

    return 0 if all(t.rc in (0, -2) for t in tasks) else 1


if __name__ == '__main__':
    sys.exit(main())
