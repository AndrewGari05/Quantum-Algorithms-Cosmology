"""Command-line interface of the thesis workflow.

Examples::

    python -m thesis run mcmc --model lcdm --dataset CC+BAO+Pantheon --steps 20000 --chains 8
    python -m thesis run vi --model cpl --grid 3 --iters 15000 --noise readout
    python -m thesis run genetic --model wcdm --grid 6
    python -m thesis campaign --campaign results/c1 --models lcdm cpl --noise none readout
    python -m thesis analyze results/c1
    python -m thesis legacy refit <old campaign folder>
    python -m thesis hardware --device fake_fez --locations ideal noisy --out results/hw
    python -m thesis hardware --device ibm_fez --out results/hw --dry-run

Every flag is used by the command that accepts it; there are no parsed but
ignored options.
"""
from __future__ import annotations

import argparse
import os
import sys

from qablate.cosmology import MODELS


def _common(p):
    p.add_argument("--model", choices=sorted(MODELS), required=True)
    p.add_argument("--dataset", default="CC+BAO+Pantheon")
    p.add_argument("--prior", choices=["flat", "gaussian"], default="flat")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--noise", default="none",
                   help="none | readout | full | a fake backend name (e.g. FakeBrisbane)")
    p.add_argument("--readout-p", type=float, default=0.03)
    p.add_argument("--gate-p1", type=float, default=1e-3)
    p.add_argument("--gate-p2", type=float, default=1e-2)
    p.add_argument("--out", required=True, help="output folder (results.csv is appended)")
    p.add_argument("--campaign", default="")
    p.add_argument("--task", default="")
    p.add_argument("--rungs", nargs="+", default=None)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m thesis", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="run one ladder")
    fam = run.add_subparsers(dest="family", required=True)
    m = fam.add_parser("mcmc")
    _common(m)
    m.add_argument("--steps", type=int, default=20000)
    m.add_argument("--chains", type=int, default=8)
    m.add_argument("--route", choices=["counts", "statevector"], default="counts")
    m.add_argument("--shots", type=int, default=None)
    v = fam.add_parser("vi")
    _common(v)
    v.add_argument("--grid", type=int, required=True, help="qubits per parameter")
    v.add_argument("--iters", type=int, default=15000)
    v.add_argument("--samples", type=int, default=4096)
    g = fam.add_parser("genetic")
    _common(g)
    g.add_argument("--grid", type=int, required=True, help="bits per parameter")
    g.add_argument("--population", type=int, default=120)
    g.add_argument("--generations", type=int, default=80)

    c = sub.add_parser("campaign", help="plan and run many tasks")
    c.add_argument("--campaign", required=True)
    c.add_argument("--models", nargs="+", default=sorted(MODELS), choices=sorted(MODELS))
    c.add_argument("--families", nargs="+", default=["mcmc", "vi", "genetic"],
                   choices=["mcmc", "vi", "genetic"])
    c.add_argument("--noise", nargs="+", default=["none"])
    c.add_argument("--dataset", default="CC+BAO+Pantheon")
    c.add_argument("--prior", choices=["flat", "gaussian"], default="flat")
    c.add_argument("--seed", type=int, default=42)
    c.add_argument("--grids", nargs="+", type=int, default=[3, 4, 5])
    c.add_argument("--bits", nargs="+", type=int, default=[4, 6])
    c.add_argument("--steps", type=int, default=20000)
    c.add_argument("--chains", type=int, default=8)
    c.add_argument("--iters", type=int, default=15000)
    c.add_argument("--samples", type=int, default=4096)
    c.add_argument("--population", type=int, default=120)
    c.add_argument("--generations", type=int, default=80)
    c.add_argument("--cores", type=int, default=None)
    c.add_argument("--mem-gb", type=float, default=None)
    c.add_argument("--dry-run", action="store_true")

    a = sub.add_parser("analyze", help="tables and figures from results.csv files")
    a.add_argument("roots", nargs="+")
    a.add_argument("--out", default="analysis")

    h = sub.add_parser("hardware", help="QMCMC, QVMC and QGA with one recipe on the ideal "
                       "simulator, the noisy twin and an IBM device")
    h.add_argument("--device", required=True,
                   help="IBM device name (ibm_fez, ...), least_busy, or a fake backend (fake_fez)")
    h.add_argument("--locations", nargs="+", default=["ideal", "noisy", "device"],
                   choices=["ideal", "noisy", "device"])
    h.add_argument("--algorithms", nargs="+", default=["vi", "genetic", "mcmc"],
                   choices=["vi", "genetic", "mcmc"])
    h.add_argument("--out", required=True)
    h.add_argument("--model", choices=sorted(MODELS), default="lcdm")
    h.add_argument("--dataset", default="CC+BAO")
    h.add_argument("--seed", type=int, default=42)
    h.add_argument("--max-quantum-seconds", type=float, default=480.0,
                   help="stop submitting device jobs past this billed time (default 8 min)")
    h.add_argument("--channel", default=None, help="IBM channel (default: saved account)")
    h.add_argument("--instance", default=None, help="IBM instance/CRN (default: saved account)")
    h.add_argument("--optimization-level", type=int, default=3)
    h.add_argument("--dry-run", action="store_true",
                   help="compile and print jobs, shots and a rough time estimate; run nothing")
    for flag, typ in (("vi-grid", int), ("vi-iters", int), ("vi-shots", int), ("vi-samples", int),
                      ("vi-warm-evals", int), ("vi-warm-restarts", int), ("mcmc-steps", int), ("mcmc-chains", int),
                      ("mcmc-shots", int), ("mcmc-block", int), ("ga-bits", int),
                      ("ga-population", int), ("ga-generations", int)):
        h.add_argument(f"--{flag}", type=typ, default=None)
    h.add_argument("--vi-init", choices=["warm", "cold"], default=None)
    h.add_argument("--spsa-a", type=float, default=None)
    h.add_argument("--spsa-c", type=float, default=None)

    lg = sub.add_parser("legacy", help="tools for campaigns written by the v0.8 thesis code")
    lg.add_argument("args", nargs=argparse.REMAINDER)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "run":
        from qablate import NoiseSpec
        from qablate.cosmology import Posterior

        from . import ladders, results
        noise = NoiseSpec.from_level(args.noise, readout_p=args.readout_p,
                                     gate_p1=args.gate_p1, gate_p2=args.gate_p2)
        post = Posterior(args.model, args.dataset, args.prior)
        kw = dict(seed=args.seed, noise=noise)
        if args.rungs:
            kw["rungs"] = args.rungs
        if args.family == "mcmc":
            rows = ladders.mcmc_ladder(post, n_steps=args.steps, n_chains=args.chains,
                                       route=args.route, shots=args.shots, **kw)
        elif args.family == "vi":
            rows = ladders.vi_ladder(post, grid_bits=args.grid, max_iter=args.iters,
                                     n_samples=args.samples, **kw)
        else:
            rows = ladders.genetic_ladder(post, grid_bits=args.grid, pop_size=args.population,
                                          n_generations=args.generations, **kw)
        for r in rows:
            r.update(campaign=args.campaign, task=args.task)
        os.makedirs(args.out, exist_ok=True)
        results.append(os.path.join(args.out, "results.csv"), rows)
        for r in rows:
            print(f"  {r['rung']:34s} chi2_map={r['chi2_map']:.4f}")
        return 0
    if args.cmd == "campaign":
        from . import campaign
        tasks = campaign.plan(args)
        print(f"  {len(tasks)} tasks | cores {args.cores or campaign.available_cores()} | "
              f"memory budget {args.mem_gb or 0.9 * campaign.available_memory_mb() / 1e3:.1f} GB")
        if args.dry_run:
            for t in tasks:
                print(f"  {t.name:40s} ~{t.mem_mb:8.0f} MB  {' '.join(t.argv[3:])}")
            return 0
        failed = campaign.run(tasks, args.campaign, cores=args.cores,
                              mem_mb=args.mem_gb * 1e3 if args.mem_gb else None)
        return 1 if failed else 0
    if args.cmd == "analyze":
        from . import analysis
        return analysis.main(args.roots, args.out)
    if args.cmd == "hardware":
        return _hardware(args)
    from .legacy import errata_tools
    return errata_tools.main(args.args)


def _hardware(args) -> int:
    from qablate.cosmology import Posterior

    from . import hardware as hw
    cfg = hw.HardwareConfig(model=args.model, dataset=args.dataset, seed=args.seed)
    for k in vars(cfg):
        v = getattr(args, k, None)
        if v is not None and k not in ("model", "dataset", "seed"):
            setattr(cfg, k, v)
    device = hw.resolve_device(args.device, channel=args.channel, instance=args.instance)
    post = Posterior(cfg.model, cfg.dataset, cfg.prior)
    plan = hw.plan(cfg, post)
    total = 0.0
    print(f"  device {getattr(device, 'name', args.device)}; per location:")
    for alg in args.algorithms:
        p = plan[alg]
        est = hw.estimate_quantum_seconds(p)
        total += est
        print(f"  {alg:8s} {p['qubits']:2d} qubits | {p['jobs']:4d} jobs | "
              f"{p['circuits']:6d} circuits | {p['shots']:9d} shots | device ~{est / 60:.1f} min")
    if "device" in args.locations:
        print(f"  device total ~{total / 60:.1f} min (rough; the run stops submitting at "
              f"{args.max_quantum_seconds / 60:.1f} min of billed time)")
    if args.dry_run:
        from qablate.circuits import hardware_efficient_ansatz, random_proposal_circuit
        from qablate.hardware import DeviceCompiler
        comp = DeviceCompiler(device, args.optimization_level, cfg.seed)
        for name, qc in (("vi ansatz", hardware_efficient_ansatz(post.ndim * cfg.vi_grid,
                                                                 cfg.vi_layers)),
                         ("mcmc proposal", random_proposal_circuit(max(2, post.ndim), 3))):
            isa, _, dt = comp.compile(qc)
            print(f"  {name:14s} -> depth {isa.depth():4d}, "
                  f"{comp.two_qubit_gates(isa):3d} two-qubit gates (compiled in {dt:.1f} s)")
        return 0
    rows = hw.run(cfg, device, locations=args.locations, algorithms=args.algorithms,
                  out=args.out, max_quantum_seconds=args.max_quantum_seconds,
                  optimization_level=args.optimization_level)
    print(hw.summary_table(rows))
    print(f"  written to {args.out}/ (results.csv, jobs.csv, vi_trace.csv, circuits.json, "
          "summary.md)")
    return 1 if any("error" in r for r in rows) else 0


if __name__ == "__main__":
    sys.exit(main())
