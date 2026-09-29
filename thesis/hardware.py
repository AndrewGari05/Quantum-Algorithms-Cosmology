"""Hardware protocol: QMCMC, QVMC and QGA with one recipe at three locations.

The question is what a run costs, and how much fidelity it keeps, when the
circuits execute on an ideal simulator, on the device's noisy twin and on
the device. For the comparison to mean anything the three runs have to be
the same process, so this module fixes:

* the circuits: every logical circuit is transpiled once for the target
  device and that ISA circuit runs at every location
  (:class:`qablate.hardware.DeviceCompiler`);
* the estimator: all locations sample with the same number of shots (no
  exact probabilities anywhere, also not on the ideal simulator);
* the random stream: each algorithm starts from the same seed at every
  location, and backends consume it identically, so SPSA directions,
  proposal angles and genetic draws coincide; only measurement outcomes
  differ;
* the budget: steps, iterations, generations, shots and job sizes.

Per algorithm and location it records the posterior summary, the quality
measure of the algorithm and a timing breakdown: wall time, time inside
backend jobs, queue, execution and billed quantum seconds (device only),
and the classical remainder. Compilation is done once and reported once.

What is run (defaults, LCDM on CC+BAO):

``vi``      Born machine on a 2**3 x 2**3 grid (6 qubits, the thesis ansatz,
            3 layers), trained with SPSA (2 circuits per iteration), 60
            iterations x 1024 shots, from a common warm start (best of 4
            COBYLA runs on the exact KL, computed once on a statevector).
            Converging from scratch takes ~1000 SPSA iterations, beyond a
            10-minute device budget; ``vi_init='cold'`` runs that experiment
            anyway. The exact KL of every iterate is evaluated afterwards.
``mcmc``    Metropolis with random-circuit proposals (counts route, 64 shots
            per increment, 1024 increments per job), classical acceptance,
            2000 steps x 4 chains. The amplitude-encoded acceptance is not
            run on hardware: it is one job per step.
``genetic`` Grid genome (4 bits per parameter), circuit-sampled init,
            mutation and crossover, population 60, 20 generations.

Classical references (local, not part of the location comparison): a long
Gaussian-proposal MCMC for the posterior, the same short MCMC with a
Gaussian proposal, and the classical-operator grid GA.
"""
from __future__ import annotations

import csv
import json
import math
import os
import time
from dataclasses import asdict, dataclass, field

import numpy as np

from qablate import (
    AerBackend,
    BornMachineVI,
    GaussianProposal,
    MetropolisHastings,
    RandomCircuitProposal,
    diagnostics,
)
from qablate.cosmology import Posterior
from qablate.hardware import LOCATIONS, DeviceBackend, DeviceCompiler, QuantumBudgetExceeded
from qablate.variational import reverse_kl

from .ladders import STEP_FRACTION, _init_positions, grid_floor, prefit_window

ALGORITHMS = ("vi", "genetic", "mcmc")


@dataclass
class HardwareConfig:
    """Everything that defines the recipe (identical at every location)."""

    model: str = "lcdm"
    dataset: str = "CC+BAO"
    prior: str = "flat"
    seed: int = 42
    # VI
    vi_grid: int = 3
    vi_layers: int = 3
    vi_iters: int = 60
    vi_shots: int = 1024
    vi_samples: int = 4096
    vi_init: str = "warm"                 # 'warm' (common COBYLA start) or 'cold'
    vi_warm_evals: int = 5000             # per restart
    vi_warm_restarts: int = 4
    spsa_a: float | None = None           # None: 0.02 from a warm start, 0.15 from a cold one
    spsa_c: float | None = None           # None: 0.05 warm, 0.1 cold

    def spsa_gains(self) -> tuple[float, float]:
        warm = self.vi_init == "warm"
        return (self.spsa_a if self.spsa_a is not None else (0.02 if warm else 0.15),
                self.spsa_c if self.spsa_c is not None else (0.05 if warm else 0.1))
    # MCMC
    mcmc_steps: int = 2000
    mcmc_chains: int = 4
    mcmc_shots: int = 64
    mcmc_block: int = 1024
    mcmc_calibration: int = 256
    # genetic
    ga_bits: int = 4
    ga_population: int = 60
    ga_generations: int = 20
    ga_operators: tuple = ("init", "mutation", "crossover")
    # reference
    reference_steps: int = 20000
    reference_chains: int = 8
    extra: dict = field(default_factory=dict)

    def mcmc_burn(self) -> int:
        return max(50, int(0.1 * self.mcmc_steps))


# --------------------------------------------------------------------------- #
def plan(cfg: HardwareConfig, post: Posterior) -> dict[str, dict]:
    """Jobs, circuits and shots each algorithm submits per location (upper bounds for GA)."""
    out = {}
    out["vi"] = {"jobs": cfg.vi_iters + 2, "circuits": 2 * cfg.vi_iters + 2,
                 "shots": 2 * cfg.vi_iters * cfg.vi_shots + cfg.vi_shots + cfg.vi_samples,
                 "qubits": post.ndim * cfg.vi_grid}
    n_prop = cfg.mcmc_chains * (cfg.mcmc_burn() + cfg.mcmc_steps)
    blocks = math.ceil(n_prop / cfg.mcmc_block)
    out["mcmc"] = {"jobs": 1 + blocks, "circuits": cfg.mcmc_calibration + blocks * cfg.mcmc_block,
                   "shots": (cfg.mcmc_calibration + blocks * cfg.mcmc_block) * cfg.mcmc_shots,
                   "qubits": max(2, post.ndim)}
    n_child = cfg.ga_population - max(1, round(0.1 * cfg.ga_population))
    ops = set(cfg.ga_operators)
    jobs = (post.ndim * 2 if "init" in ops else 0) + cfg.ga_generations * (
        ("crossover" in ops) + ("mutation" in ops))
    circ = cfg.ga_generations * n_child * post.ndim * (("crossover" in ops) + ("mutation" in ops))
    out["genetic"] = {"jobs": jobs, "circuits": circ + (post.ndim * 2 if "init" in ops else 0),
                      "shots": circ + (4 * cfg.ga_population * post.ndim if "init" in ops else 0),
                      "qubits": 2 * cfg.ga_bits if "crossover" in ops else cfg.ga_bits}
    return out


def estimate_quantum_seconds(p: dict, per_job_s: float = 2.0, per_shot_us: float = 250.0) -> float:
    """Rough billed-time estimate: fixed cost per job plus a per-shot cost.

    The two constants are deliberately pessimistic placeholders; the real
    values depend on the device and are reported per job after a run.
    """
    return p["jobs"] * per_job_s + p["shots"] * per_shot_us * 1e-6


# --------------------------------------------------------------------------- #
def _seed(cfg: HardwareConfig, algorithm: str) -> np.random.SeedSequence:
    return np.random.SeedSequence([cfg.seed, ALGORITHMS.index(algorithm)])


def _timing(backend: DeviceBackend, n0: int, wall: float) -> dict:
    jobs = backend.log[n0:]
    inside = sum(j.wall_s for j in jobs)
    return {"jobs": len(jobs), "circuits": sum(j.n_circuits for j in jobs),
            "shots_total": sum(j.n_circuits * j.shots for j in jobs),
            "wall_s": wall, "backend_wall_s": inside, "classical_s": wall - inside,
            "queue_s": sum(j.queue_s for j in jobs),
            "execution_s": sum(j.execution_s for j in jobs),
            "quantum_s": sum(j.quantum_s for j in jobs)}


def _summary(post: Posterior, samples: np.ndarray, ref: dict) -> dict:
    mu, sd = samples.mean(axis=0), samples.std(axis=0)
    row = {}
    for i, n in enumerate(post.model.param_names):
        row[f"mean_{n}"] = float(mu[i])
        row[f"std_{n}"] = float(sd[i])
        row[f"shift_{n}_sigma"] = float((mu[i] - ref["mean"][i]) / ref["std"][i])
        row[f"width_ratio_{n}"] = float(sd[i] / ref["std"][i])
    row["chi2_estimate"] = float(post.chi2(mu))
    return row


def reference(post: Posterior, cfg: HardwareConfig) -> dict:
    """Long classical MCMC used as the posterior reference."""
    rng = np.random.default_rng(cfg.seed)
    scale = STEP_FRACTION * np.array([hi - lo for lo, hi in post.model.sample_box])
    mh = MetropolisHastings(post.log_prob, post.ndim, cfg.reference_chains,
                            proposal=GaussianProposal(scale), rng=rng)
    burn = max(50, int(0.1 * cfg.reference_steps))
    mh.run_mcmc(_init_positions(post, cfg.reference_chains, rng), burn + cfg.reference_steps)
    ch = mh.get_chain(discard=burn)
    flat = ch.reshape(-1, post.ndim)
    return {"mean": flat.mean(axis=0), "std": flat.std(axis=0),
            "rhat_max": float(np.max(diagnostics.rhat(np.swapaxes(ch, 0, 1))))}


# --------------------------------------------------------------------------- #
class VIProtocol:
    """Shared grid, target, warm start and exact evaluator for the VI runs."""

    def __init__(self, post: Posterior, cfg: HardwareConfig):
        self.post, self.cfg = post, cfg
        self.grid, self.outside = prefit_window(post, cfg.vi_grid, cfg.seed)
        self.p = self.grid.target(post.log_prob)
        self.exact = AerBackend()
        t0 = time.perf_counter()
        probe = BornMachineVI(self.grid, n_layers=cfg.vi_layers)
        rng = np.random.default_rng(_seed(cfg, "vi"))
        if cfg.vi_init == "warm":
            # best of several COBYLA restarts on the exact (statevector) KL;
            # a single run often stalls in a poor local minimum
            best = None
            for _ in range(cfg.vi_warm_restarts):
                r = probe.fit(self.p, max(1, cfg.vi_warm_evals // (1 + 2 * probe.n_params)), rng)
                if best is None or r.kl < best.kl:
                    best = r
            self.phi0 = best.phi
        else:
            self.phi0 = 0.1 * rng.standard_normal(probe.n_params)
        self.warm_start_s = time.perf_counter() - t0
        self.circuit = probe.circuit
        self.kl0 = self.exact_kl(self.phi0[None, :])[0]

    def exact_kl(self, phis: np.ndarray) -> np.ndarray:
        q = self.exact.probabilities(self.circuit, phis, np.random.default_rng(0))
        return reverse_kl(q, self.p)


def run_vi(proto: VIProtocol, backend: DeviceBackend, ref: dict) -> tuple[dict, list[dict]]:
    cfg, post = proto.cfg, proto.post
    rng = np.random.default_rng(np.random.SeedSequence([cfg.seed, ALGORITHMS.index("vi"), 1]))
    vi = BornMachineVI(proto.grid, n_layers=cfg.vi_layers, optimizer="spsa", backend=backend,
                       shots=cfg.vi_shots, spsa_gains=cfg.spsa_gains())
    n0, t0 = len(backend.log), time.perf_counter()
    backend.label = "vi-spsa"
    res = vi.fit(proto.p, cfg.vi_iters, rng, initial=proto.phi0)
    backend.label = "vi-samples"
    samples = vi.sample(res, cfg.vi_samples, rng, shots_backend=backend)
    wall = time.perf_counter() - t0
    kl_exact = proto.exact_kl(res.phi_history)
    row = {"algorithm": "vi", "kl_start_exact": float(proto.kl0),
           "kl_final_exact": float(kl_exact[-1]), "kl_final_shots": float(res.history[-1]),
           "iterations": cfg.vi_iters, "circuit_evaluations": res.circuit_evaluations,
           "grid_mass_outside": proto.outside, "warm_start_s": proto.warm_start_s,
           **_summary(post, samples, ref), **_timing(backend, n0, wall)}
    # per-iteration trace with cumulative time (time to reach a KL level)
    jobs = [j for j in backend.log[n0:] if j.label == "vi-spsa"]
    cw = np.cumsum([j.wall_s for j in jobs])
    cq = np.cumsum([j.quantum_s for j in jobs])
    trace = [{"iteration": k + 1, "kl_shots": float(res.history[k]), "kl_exact": float(kl_exact[k]),
              "backend_wall_s": float(cw[k]) if k < len(cw) else float("nan"),
              "quantum_s": float(cq[k]) if k < len(cq) else float("nan")}
             for k in range(cfg.vi_iters)]
    return row, trace


# --------------------------------------------------------------------------- #
def run_mcmc(post: Posterior, cfg: HardwareConfig, backend, ref: dict) -> dict:
    rng = np.random.default_rng(_seed(cfg, "mcmc"))
    scale = STEP_FRACTION * np.array([hi - lo for lo, hi in post.model.sample_box])
    if backend is None:                     # classical baseline, same recipe
        proposal = GaussianProposal(scale)
    else:
        proposal = RandomCircuitProposal(scale, backend=backend, route="counts",
                                         shots=cfg.mcmc_shots, block_size=cfg.mcmc_block,
                                         n_calibration=cfg.mcmc_calibration)
        backend.label = "mcmc-proposals"
    n0 = len(backend.log) if backend is not None else 0
    t0 = time.perf_counter()
    mh = MetropolisHastings(post.log_prob, post.ndim, cfg.mcmc_chains, proposal=proposal, rng=rng)
    burn = cfg.mcmc_burn()
    mh.run_mcmc(_init_positions(post, cfg.mcmc_chains, rng), burn + cfg.mcmc_steps)
    wall = time.perf_counter() - t0
    ch = mh.get_chain(discard=burn)
    per_chain = np.swapaxes(ch, 0, 1)
    row = {"algorithm": "mcmc", "acceptance": float(np.mean(mh.acceptance_fraction)),
           "rhat_max": float(np.max(diagnostics.rhat(per_chain))),
           "ess_min": float(np.min(diagnostics.ess(per_chain))),
           "steps": cfg.mcmc_steps, "chains": cfg.mcmc_chains, "burn": burn,
           **_summary(post, ch.reshape(-1, post.ndim), ref)}
    row.update(_timing(backend, n0, wall) if backend is not None else
               {"wall_s": wall, "classical_s": wall, "jobs": 0})
    return row


# --------------------------------------------------------------------------- #
def run_genetic(post: Posterior, cfg: HardwareConfig, backend, floor) -> dict:
    from qablate.cosmology import fit_statistics
    from qablate.experimental import GAConfig, GeneticAlgorithm
    rng = np.random.default_rng(_seed(cfg, "genetic"))
    quantum = list(cfg.ga_operators) if backend is not None else []
    n0 = len(backend.log) if backend is not None else 0
    if backend is not None:
        backend.label = "ga-operators"
    t0 = time.perf_counter()
    ga = GeneticAlgorithm(post.log_prob, post.model.bounds,
                          GAConfig(pop_size=cfg.ga_population, n_generations=cfg.ga_generations),
                          init_box=post.model.sample_box, genome_bits=cfg.ga_bits,
                          quantum=quantum, backend=backend, rng=rng)
    r = ga.run()
    wall = time.perf_counter() - t0
    fs = fit_statistics(post, r.theta_best)
    row = {"algorithm": "genetic", "operators": "+".join(quantum) or "classical",
           "chi2_estimate": float(post.chi2(r.theta_best)), "chi2_map": fs["chi2"],
           "chi2_grid_floor": floor, "generations": cfg.ga_generations,
           "population": cfg.ga_population}
    for n, v in zip(post.model.param_names, r.theta_best, strict=True):
        row[f"mean_{n}"] = float(v)
    row.update(_timing(backend, n0, wall) if backend is not None else
               {"wall_s": wall, "classical_s": wall, "jobs": 0})
    return row


# --------------------------------------------------------------------------- #
def resolve_device(name: str, *, channel: str | None = None, instance: str | None = None):
    """A fake backend (``fake_fez``, ``FakeTorino``...) or an IBM device by name.

    ``least_busy`` picks the least busy operational device of the account.
    """
    from qablate.noise import canonical_level
    if canonical_level(name).startswith("fake"):
        from qablate.noise import _fake_backend
        return _fake_backend(canonical_level(name))
    from qiskit_ibm_runtime import QiskitRuntimeService
    kw = {k: v for k, v in (("channel", channel), ("instance", instance)) if v}
    service = QiskitRuntimeService(**kw)
    if name == "least_busy":
        return service.least_busy(operational=True, simulator=False)
    return service.backend(name)


def _write_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    cols: list[str] = []
    for r in rows:
        cols += [k for k in r if k not in cols]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def run(cfg: HardwareConfig, device, *, locations=LOCATIONS, algorithms=ALGORITHMS,
        out: str = "hardware", max_quantum_seconds: float | None = None,
        optimization_level: int = 3, log=print) -> list[dict]:
    """Run the protocol and write ``results.csv``, ``jobs.csv``, ``vi_trace.csv``,
    ``circuits.json`` and ``summary.md`` into ``out``."""
    os.makedirs(out, exist_ok=True)
    post = Posterior(cfg.model, cfg.dataset, cfg.prior)
    compiler = DeviceCompiler(device, optimization_level=optimization_level,
                              seed_transpiler=cfg.seed)
    log(f"  device {compiler.name} | locations {', '.join(locations)} | "
        f"algorithms {', '.join(algorithms)}")
    ref = reference(post, cfg)
    log(f"  reference posterior (classical MCMC {cfg.reference_steps} x {cfg.reference_chains}): "
        + ", ".join(f"{n}={m:.4f}+-{s:.4f}" for n, m, s in
                    zip(post.model.param_names, ref["mean"], ref["std"], strict=True)))
    base = {"model": cfg.model, "dataset": cfg.dataset, "seed": cfg.seed,
            "device": compiler.name}
    rows: list[dict] = []
    traces: list[dict] = []
    jobs: list[dict] = []
    proto = VIProtocol(post, cfg) if "vi" in algorithms else None
    floor = grid_floor(post, cfg.ga_bits) if "genetic" in algorithms else None

    # classical baselines (same recipe, no circuits)
    if "mcmc" in algorithms:
        rows.append({**base, "location": "classical", **run_mcmc(post, cfg, None, ref)})
    if "genetic" in algorithms:
        rows.append({**base, "location": "classical", **run_genetic(post, cfg, None, floor)})

    for loc in locations:
        mode_ctx = None
        mode = None
        if loc == "device":
            from qiskit_ibm_runtime import Batch
            if not getattr(device, "name", "").startswith("fake"):
                mode_ctx = Batch(backend=device)
                mode = mode_ctx
        backend = DeviceBackend(compiler, loc, mode=mode,
                                max_quantum_seconds=max_quantum_seconds if loc == "device" else None)
        try:
            for alg in algorithms:
                log(f"  [{loc}] {alg} ...")
                t0 = time.perf_counter()
                try:
                    if alg == "vi":
                        row, trace = run_vi(proto, backend, ref)
                        traces += [{**base, "location": loc, **t} for t in trace]
                    elif alg == "mcmc":
                        row = run_mcmc(post, cfg, backend, ref)
                    else:
                        row = run_genetic(post, cfg, backend, floor)
                except QuantumBudgetExceeded as exc:
                    log(f"  [{loc}] {alg} stopped: {exc}")
                    row = {"algorithm": alg, "error": str(exc)}
                rows.append({**base, "location": loc, **row})
                log(f"  [{loc}] {alg} done in {time.perf_counter() - t0:.1f} s"
                    + (f" | quantum {backend.quantum_seconds:.1f} s" if loc == "device" else ""))
        finally:
            jobs += [{**base, **j.as_dict()} for j in backend.log]
            if mode_ctx is not None:
                mode_ctx.close()
            _write_all(out, cfg, compiler, rows, jobs, traces, ref)
    _write_all(out, cfg, compiler, rows, jobs, traces, ref)
    return rows


def _write_all(out, cfg, compiler, rows, jobs, traces, ref):
    _write_csv(os.path.join(out, "results.csv"), rows)
    _write_csv(os.path.join(out, "jobs.csv"), jobs)
    _write_csv(os.path.join(out, "vi_trace.csv"), traces)
    from qablate import __version__
    with open(os.path.join(out, "circuits.json"), "w") as fh:
        json.dump({"device": compiler.name, "optimization_level": compiler.optimization_level,
                   "seed_transpiler": compiler.seed_transpiler, "qablate": __version__,
                   "config": {k: (list(v) if isinstance(v, tuple) else v)
                              for k, v in asdict(cfg).items()},
                   "reference": {k: np.asarray(v).tolist() for k, v in ref.items()},
                   "circuits": compiler.compiled()}, fh, indent=2)
    with open(os.path.join(out, "summary.md"), "w") as fh:
        fh.write(summary_table(rows))


def summary_table(rows: list[dict]) -> str:
    """Markdown table: one line per algorithm and location."""
    cols = [("algorithm", "{}"), ("location", "{}"), ("jobs", "{}"), ("wall_s", "{:.1f}"),
            ("backend_wall_s", "{:.1f}"), ("classical_s", "{:.1f}"), ("queue_s", "{:.1f}"),
            ("execution_s", "{:.2f}"), ("quantum_s", "{:.1f}"), ("kl_final_exact", "{:.3f}"),
            ("rhat_max", "{:.3f}"), ("chi2_estimate", "{:.3f}")]
    shift = sorted({k for r in rows for k in r if k.startswith("shift_")})
    cols += [(k, "{:+.2f}") for k in shift]
    lines = ["| " + " | ".join(c for c, _ in cols) + " |",
             "|" + "---|" * len(cols)]
    for r in rows:
        cells = []
        for c, f in cols:
            v = r.get(c, "")
            try:
                cells.append(f.format(v) if v != "" and v is not None else "")
            except (ValueError, TypeError):
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"
