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
from typing import Dict, List, Optional

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

# Worst-case memory of the statevector + likelihood auxiliary arrays. Uses
# EXACTLY the same constant as the validation in cosmo_modular_quantum.py
# (_validate_args: 2**total_q * 1660 * 8), to avoid having two different magic
# numbers. The 1660 floats/state are the auxiliary arrays the likelihood builds
# over the grid; x8 bytes (float64).
BYTES_PER_STATE = 1660 * 8


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

def estimate_qubits_and_mem(total_q: int) -> float:
    """Estimated peak RAM (MB) for a grid of 2^total_q states (worst case)."""
    return (2 ** total_q) * BYTES_PER_STATE / 1e6


def qubits_fitting_in(mem_mb: float) -> int:
    """Largest number of qubits whose grid (2^q states) fits in mem_mb."""
    if mem_mb <= 0:
        return 0
    cap_states = mem_mb * 1e6 / BYTES_PER_STATE
    q = 0
    while 2 ** (q + 1) <= cap_states:
        q += 1
    return q


def qubit_ceiling(max_qubits: int, mem_ceiling_mb: float) -> int:
    """EFFECTIVE per-task qubit ceiling: the most restrictive of the
    --max-qubits cap (enforced by the scripts) and what fits in the per-task
    RAM."""
    return min(max_qubits, qubits_fitting_in(mem_ceiling_mb))


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
                notices: List[str]) -> List[Task]:
    """One task per (script x model x grid value), with per-model clamping.

    When a sweep is requested (--nqpp-sweep / --nbits-sweep) one task is
    generated per value, to measure how the grid size affects the results. The
    grid value is trimmed per model according to `q_ceiling` (unless
    --strict-qubits), so a heavy model (CPL, d=4) is lowered on its own while
    the light ones (LCDM, d=2) stay at the target value.
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

    for m in models:
        d = MODEL_DIM[m]

        # ---- Samplers tasks (QMCMC + QVMC), one per nqpp value ----
        if not args.only_genetic:
            for nqpp in grid_values_for_model(
                    args.nqpp, args.nqpp_sweep, d, q_ceiling, strict,
                    notices, 'nqpp', m):
                total_q = nqpp * d
                # visible tag if sweeping or if the clamp changed the value
                tag = (f"nqpp{nqpp}"
                       if (sweeping_nqpp or nqpp != args.nqpp) else "")
                name = f"samplers/{m}" + (f"/{tag}" if tag else "")
                outdir = os.path.join(
                    master_dir, f"samplers_{m}" + (f"_{tag}" if tag else ""))
                argv = ['--sweep-all', '--sweep-models', m,
                        '--steps', str(args.steps),
                        '--qvmc-iter', str(args.qvmc_iter),
                        '--nqpp', str(nqpp),
                        '--chains', str(args.chains),
                        '--shots', str(args.shots),
                        '--max-qubits', str(args.max_qubits),
                        '--outdir', outdir] + common_data
                tasks.append(Task(
                    name=name, script='cosmo_modular_quantum.py',
                    argv=argv, model=m, total_qubits=total_q,
                    est_mem_mb=estimate_qubits_and_mem(total_q),
                    outdir=outdir, grid_value=nqpp, grid_kind='nqpp'))

        # ---- Genetic tasks (CGA + QGA), one per n_bits value ----
        if not args.only_samplers:
            for nbits in grid_values_for_model(
                    args.n_bits, args.nbits_sweep, d, q_ceiling, strict,
                    notices, 'n_bits', m):
                total_q = nbits * d
                tag = (f"nb{nbits}"
                       if (sweeping_nbits or nbits != args.n_bits) else "")
                name = f"genetic/{m}" + (f"/{tag}" if tag else "")
                outdir = os.path.join(
                    master_dir, f"genetic_{m}" + (f"_{tag}" if tag else ""))
                argv = ['--sweep-all', '--sweep-models', m,
                        '--generations', str(args.generations),
                        '--population-size', str(args.population_size),
                        '--n-bits', str(nbits),
                        '--max-qubits', str(args.max_qubits),
                        '--outdir', outdir] + common_data
                tasks.append(Task(
                    name=name, script='cosmo_genetic_optimizers.py',
                    argv=argv, model=m, total_qubits=total_q,
                    est_mem_mb=estimate_qubits_and_mem(total_q),
                    outdir=outdir, grid_value=nbits, grid_kind='n_bits'))

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
             poll: float = 0.5) -> None:
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
            if t.total_qubits > max_qubits:
                print(f"  [SKIP] {t.name}: {t.total_qubits} qubits "
                      f"(> --max-qubits {max_qubits}). "
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

    # --- shared ---
    p.add_argument('--max-qubits', type=int, default=18,
                   help='Per-task qubit cap (same as the scripts)')
    p.add_argument('--max-task-gb', type=float, default=None,
                   help='Max RAM per task for the clamp (default: the aggregate '
                        'budget). Together with --max-qubits it sets the '
                        'effective per-model grid ceiling.')
    p.add_argument('--strict-qubits', action='store_true',
                   help='Do not clamp: combinations exceeding the ceiling are '
                        'SKIPPED instead of having their grid lowered.')
    p.add_argument('--seed', type=int, default=42)
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

    # --- EFFECTIVE per-task qubit ceiling (for the per-model clamp) ---
    # The most restrictive of --max-qubits and the per-task RAM (--max-task-gb,
    # or, if not given, the aggregate budget as a single-task cap).
    task_mem_ceiling = (args.max_task_gb * 1024 if args.max_task_gb
                        else mem_budget_mb)
    q_ceiling = qubit_ceiling(args.max_qubits, task_mem_ceiling)

    # --- build tasks (with per-model clamp) ---
    clamp_notices: List[str] = []
    tasks = build_tasks(args, master_dir, q_ceiling, clamp_notices)
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
    print(f"Grid ceiling: {q_ceiling} qubits/task "
          f"(min of --max-qubits {args.max_qubits} and "
          f"{qubits_fitting_in(task_mem_ceiling)}q that fit in "
          f"{task_mem_ceiling/1024:.0f} GB)")
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
            flag = ("  (SKIP: exceeds max-qubits)"
                    if t.total_qubits > args.max_qubits else "")
            print(f"\n# {t.name}  ({t.grid_kind}={t.grid_value}, "
                  f"~{t.total_qubits}q, ~{t.est_mem_mb:.0f} MB){flag}\n"
                  f"OMP_NUM_THREADS={T} {shlex.join(cmd)}")
        return 0

    t_wall0 = time.time()
    run_pool(tasks, max_parallel=J, threads_per_worker=T,
             mem_budget_mb=mem_budget_mb, max_qubits=args.max_qubits,
             project_dir=project_dir)
    report(tasks, master_dir, t_wall0)

    # --- convergence plots at the end of ALL the runs ---
    if not args.no_plots:
        print("\nGenerating convergence plots...")
        generate_convergence_plots(
            master_dir, only_grid_methods=args.only_grid_methods)

    return 0 if all(t.rc in (0, -2) for t in tasks) else 1


if __name__ == '__main__':
    sys.exit(main())
