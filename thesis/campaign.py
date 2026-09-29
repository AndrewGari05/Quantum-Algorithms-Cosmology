"""Plan and run a campaign of ladder tasks on one machine (workstation or HPC node).

A campaign is the product ``models x grids x noise levels x families``. Each
task runs in its own process (``python -m thesis run ...``), with one BLAS
thread, and writes ``<campaign>/<task>/results.csv`` plus a log. Tasks are
admitted while both the core count and the memory budget allow; the
available cores respect CPU affinity and quotas, and the memory budget
respects the cgroup limit of this process (SLURM, containers).

``status.csv`` is appended when each task ends, so a campaign that is killed
keeps the record of what finished; SIGINT/SIGTERM terminate the children.
A task fails when its exit code is non-zero.
"""
from __future__ import annotations

import csv
import itertools
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field

BYTES = 16                     # complex128 amplitude / density-matrix entry
BASE_MB = 400.0                # interpreter + Qiskit + data, measured ~340 MB


def available_cores() -> int:
    """Cores this process may use (affinity and cgroup CPU quota)."""
    n = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1
    try:
        quota, period = open("/sys/fs/cgroup/cpu.max").read().split()
        if quota != "max":
            n = min(n, max(1, int(int(quota) / int(period))))
    except (OSError, ValueError):
        pass
    return max(1, n)


def available_memory_mb() -> float:
    """Physical memory, capped by every cgroup limit on this process's path."""
    total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e6
    limits = []
    try:
        for line in open("/proc/self/cgroup"):
            _, ctrl, path = line.strip().split(":", 2)
            if ctrl in ("", "memory") or "memory" in ctrl.split(","):
                parts = path.strip("/").split("/") if path.strip("/") else []
                for root in ("/sys/fs/cgroup", "/sys/fs/cgroup/memory"):
                    for i in range(len(parts), -1, -1):
                        d = os.path.join(root, *parts[:i])
                        for f in ("memory.max", "memory.limit_in_bytes"):
                            try:
                                v = open(os.path.join(d, f)).read().strip()
                                if v != "max" and int(v) < 2 ** 60:
                                    limits.append(int(v) / 1e6)
                            except (OSError, ValueError):
                                pass
    except OSError:
        pass
    return min([total] + limits)


def estimate_mb(family: str, ndim: int, grid: int, noisy: bool) -> float:
    """Peak memory estimate of one task [MB]."""
    if family == "mcmc":
        n = max(2, ndim)
        return BASE_MB + (BYTES * 4 ** n if noisy else BYTES * 2 ** n) / 1e6
    if family == "vi":
        n = ndim * grid
        n_params = 3 * 2 * n + n
        state = BYTES * 4 ** n if noisy else BYTES * 2 ** n
        return BASE_MB + (state + 8 * 2 ** n * (1 + 2 * n_params) + 8 * 2 ** n * 1200 * 3 / 16) / 1e6
    n = 2 * grid                                    # genetic crossover circuit
    return BASE_MB + (BYTES * 4 ** n if noisy else BYTES * 2 ** n) / 1e6


@dataclass
class Task:
    name: str
    argv: list[str]
    mem_mb: float
    outdir: str
    proc: subprocess.Popen | None = None
    t0: float = 0.0
    rc: int | None = None
    extra: dict[str, object] = field(default_factory=dict)


def plan(args) -> list[Task]:
    from qablate.cosmology import get_model
    tasks = []
    grids = {"mcmc": [0], "vi": args.grids, "genetic": args.bits}
    for model, noise, family in itertools.product(args.models, args.noise, args.families):
        d = get_model(model).ndim
        for g in grids[family]:
            name = f"{family}_{model}" + (f"_g{g}" if family != "mcmc" else "") + f"_noise-{noise}"
            outdir = os.path.join(args.campaign, name)
            argv = [sys.executable, "-m", "thesis", "run", family, "--model", model,
                    "--dataset", args.dataset, "--prior", args.prior, "--seed", str(args.seed),
                    "--noise", noise, "--out", outdir, "--campaign",
                    os.path.basename(os.path.normpath(args.campaign)), "--task", name]
            if family == "mcmc":
                argv += ["--steps", str(args.steps), "--chains", str(args.chains)]
            elif family == "vi":
                argv += ["--grid", str(g), "--iters", str(args.iters), "--samples", str(args.samples)]
            else:
                argv += ["--grid", str(g), "--population", str(args.population),
                         "--generations", str(args.generations)]
            tasks.append(Task(name, argv, estimate_mb(family, d, g, noise != "none"), outdir))
    return tasks


def run(tasks: list[Task], campaign: str, cores: int | None = None,
        mem_mb: float | None = None, poll: float = 2.0) -> int:
    """Run tasks with at most ``cores`` processes inside ``mem_mb``; returns #failed."""
    cores = cores or available_cores()
    mem_mb = mem_mb or 0.9 * available_memory_mb()
    too_big = [t for t in tasks if t.mem_mb > mem_mb]
    for t in too_big:
        print(f"  [SKIP] {t.name}: needs ~{t.mem_mb:.0f} MB > budget {mem_mb:.0f} MB")
    queue = sorted([t for t in tasks if t not in too_big], key=lambda t: -t.mem_mb)
    running: list[Task] = []
    os.makedirs(campaign, exist_ok=True)
    status = os.path.join(campaign, "status.csv")
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               QISKIT_NUM_PROCS="1", PYTHONUNBUFFERED="1")

    def stop(signum, _frame):
        for t in running:
            t.proc.terminate()
        print(f"\n  signal {signum}: {len(running)} task(s) terminated")
        sys.exit(128 + signum)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    failed = 0
    while queue or running:
        used = sum(t.mem_mb for t in running)
        for t in list(queue):
            if len(running) < cores and (used + t.mem_mb <= mem_mb or not running):
                os.makedirs(t.outdir, exist_ok=True)
                log = open(os.path.join(t.outdir, "task.log"), "w")
                t.proc = subprocess.Popen(t.argv, stdout=log, stderr=subprocess.STDOUT, env=env)
                t.t0 = time.time()
                running.append(t)
                queue.remove(t)
                used += t.mem_mb
                print(f"  [START] {t.name} (~{t.mem_mb:.0f} MB)")
        time.sleep(poll)
        for t in list(running):
            rc = t.proc.poll()
            if rc is None:
                continue
            running.remove(t)
            t.rc = rc
            failed += rc != 0
            new = not os.path.exists(status)
            with open(status, "a", newline="") as fh:
                w = csv.writer(fh)
                if new:
                    w.writerow(["task", "rc", "wall_s", "est_mb"])
                w.writerow([t.name, rc, f"{time.time() - t.t0:.1f}", f"{t.mem_mb:.0f}"])
            print(f"  [{'OK' if rc == 0 else f'FAILED rc={rc}'}] {t.name} "
                  f"{(time.time() - t.t0) / 60:.1f} min")
    return failed + len(too_big)
