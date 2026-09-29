"""Command-line interface of the thesis workflow.

Examples::

    python -m thesis run mcmc --model lcdm --dataset CC+BAO+Pantheon --steps 20000 --chains 8
    python -m thesis run vi --model cpl --grid 3 --iters 15000 --noise readout
    python -m thesis run genetic --model wcdm --grid 6
    python -m thesis campaign --campaign results/c1 --models lcdm cpl --noise none readout
    python -m thesis analyze results/c1
    python -m thesis legacy refit <old campaign folder>

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
    from .legacy import errata_tools
    return errata_tools.main(args.args)


if __name__ == "__main__":
    sys.exit(main())
