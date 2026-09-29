"""The three ablation ladders of the thesis, built on qablate.

Each ladder switches one component of an algorithm from its classical to
its circuit-based implementation per rung; every rung starts from the same
seed. Rows are returned in the schema of :mod:`thesis.results`.

MCMC ladder (``family='mcmc'``)
    ``mcmc``           Gaussian random-walk proposal, Metropolis acceptance
    ``qmcmc-proposal`` random-circuit proposal, Metropolis acceptance
    ``qmcmc-full``     random-circuit proposal, amplitude-encoded acceptance
    On an ideal simulator with exact probabilities the last two rungs are
    identical by construction (a plumbing check, verified on every run).

VI ladder (``family='vi'``), Born machine on a ``2**grid`` per-parameter grid
    ``vi-cobyla-exact``  COBYLA, samples drawn classically from exact Q
    ``vi-cobyla-shots``  COBYLA, samples measured from the circuit
    ``vi-pshift-shots``  parameter-shift gradient descent, measured samples
    COBYLA receives as many circuit evaluations as parameter-shift.

Genetic ladder (``family='genetic'``), ``grid`` bits per parameter
    ``ga-continuous``    continuous genome (reference optimum)
    ``ga-grid``          grid genome, classical operators
    ``ga-grid+init`` ... cumulative circuit-sampled operators
    Compare circuit rungs with ``ga-grid`` (same search space); the gap
    between ``ga-grid`` and ``ga-continuous`` is the grid resolution, and
    ``chi2_grid_floor`` is the best chi2 any grid rung can reach.
"""
from __future__ import annotations

import time
from collections.abc import Sequence

import numpy as np

from qablate import (
    AerBackend,
    AmplitudeEncodedMetropolis,
    BornMachineVI,
    GaussianProposal,
    Grid,
    MetropolisAcceptance,
    MetropolisHastings,
    NoiseSpec,
    RandomCircuitProposal,
    diagnostics,
)
from qablate.cosmology import Posterior, fit_statistics

MCMC_RUNGS = ("mcmc", "qmcmc-proposal", "qmcmc-full")
VI_RUNGS = ("vi-cobyla-exact", "vi-cobyla-shots", "vi-pshift-shots")
GA_RUNGS = ("ga-continuous", "ga-grid", "ga-grid+init", "ga-grid+init+mutation",
            "ga-grid+init+mutation+crossover")
STEP_FRACTION = 0.06        #: proposal scale as a fraction of the model's sample box


def _base(post: Posterior, noise: NoiseSpec, **kw) -> dict[str, object]:
    return {"model": post.model.name, "dataset": post.dataset, "prior": post.prior,
            "n_data": post.n_data, "noise": noise.label, "noise_params": noise.params, **kw}


def _moments(row: dict[str, object], names, mean, std) -> None:
    for n, m, s in zip(names, mean, std, strict=True):
        row[f"mean_{n}"] = float(m)
        row[f"std_{n}"] = float(s)


def _init_positions(post: Posterior, nchains: int, rng) -> np.ndarray:
    box = np.array(post.model.sample_box)
    return box[:, 0] + rng.random((nchains, post.ndim)) * (box[:, 1] - box[:, 0])


# --------------------------------------------------------------------------- #
def mcmc_ladder(post: Posterior, *, n_steps: int, n_chains: int = 6, seed: int = 42,
                noise: NoiseSpec | None = None, route: str = "counts",
                shots: int | None = None, rungs: Sequence[str] = MCMC_RUNGS) -> list[dict]:
    """Run the MCMC rungs and return one result row per rung."""
    noise = noise or NoiseSpec.from_level("none")
    backend = AerBackend(noise)
    scale = STEP_FRACTION * np.array([hi - lo for lo, hi in post.model.sample_box])
    burn = max(50, int(0.1 * n_steps))
    rows, chains = [], {}
    for rung in rungs:
        if rung not in MCMC_RUNGS:
            raise ValueError(f"unknown MCMC rung {rung!r}")
        rng = np.random.default_rng(seed)
        prop = (GaussianProposal(scale) if rung == "mcmc" else
                RandomCircuitProposal(scale, backend=backend, route=route, shots=shots))
        acc = (AmplitudeEncodedMetropolis(backend=backend, shots=shots)
               if rung == "qmcmc-full" else MetropolisAcceptance())
        t0 = time.time()
        mh = MetropolisHastings(post.log_prob, post.ndim, n_chains, proposal=prop,
                                acceptance=acc, rng=rng)
        mh.run_mcmc(_init_positions(post, n_chains, rng), burn + n_steps)
        ch = mh.get_chain(discard=burn)
        chains[rung] = ch
        flat = ch.reshape(-1, post.ndim)
        per_chain = np.swapaxes(ch, 0, 1)
        fs_est = post.chi2(flat.mean(axis=0))
        fs = fit_statistics(post, flat.mean(axis=0))
        row = _base(post, noise, family="mcmc", rung=rung,
                    components={"mcmc": "", "qmcmc-proposal": "proposal",
                                "qmcmc-full": "proposal+acceptance"}[rung],
                    seed=seed, route="" if rung == "mcmc" else route, backend=backend.name,
                    shots=shots, n_steps=n_steps, n_chains=n_chains, burn=burn,
                    acceptance=float(mh.acceptance_fraction.mean()),
                    detailed_balance=bool(acc.preserves_detailed_balance),
                    rhat_max=float(np.max(diagnostics.rhat(per_chain))),
                    ess_min=float(np.min(diagnostics.ess(per_chain))),
                    corr_01=float(np.corrcoef(flat[:, 0], flat[:, 1])[0, 1]),
                    chi2_estimate=fs_est, chi2_map=fs["chi2"], chi2_red=fs["chi2_red"],
                    AIC=fs["AIC"], BIC=fs["BIC"], time_s=time.time() - t0)
        row["converged"] = bool(row["rhat_max"] < diagnostics.RHAT_THRESHOLD
                                and row["ess_min"] >= diagnostics.ESS_MIN)
        _moments(row, post.model.param_names, flat.mean(axis=0), flat.std(axis=0))
        rows.append(row)
    if {"qmcmc-proposal", "qmcmc-full"} <= set(chains) and noise.is_ideal and shots is None:
        if not np.array_equal(chains["qmcmc-proposal"], chains["qmcmc-full"]):
            raise AssertionError("plumbing check failed: exact amplitude-encoded acceptance "
                                 "differs from Metropolis on an ideal backend")
    return rows


# --------------------------------------------------------------------------- #
def prefit_window(post: Posterior, grid_bits: int, seed: int, sigma_mult: float = 4.0):
    """Grid window from a short classical MCMC: mean +- sigma_mult sd, clipped.

    Returns the :class:`~qablate.Grid` and the fraction of prefit samples
    that fall outside it (a check on truncated posterior mass).
    """
    rng = np.random.default_rng(seed)
    scale = STEP_FRACTION * np.array([hi - lo for lo, hi in post.model.sample_box])
    mh = MetropolisHastings(post.log_prob, post.ndim, 4, proposal=GaussianProposal(scale), rng=rng)
    mh.run_mcmc(_init_positions(post, 4, rng), 1500)
    s = mh.get_chain(discard=500, flat=True)
    grid = Grid.around(s, grid_bits, sigma_mult=sigma_mult, bounds=post.model.bounds)
    lo, hi = np.array(grid.window).T
    outside = float(np.mean(np.any((s < lo) | (s > hi), axis=1)))
    return grid, outside


def vi_ladder(post: Posterior, *, grid_bits: int, max_iter: int, n_samples: int = 4096,
              seed: int = 42, noise: NoiseSpec | None = None, n_layers: int = 3,
              rungs: Sequence[str] = VI_RUNGS) -> list[dict]:
    """Run the Born-machine VI rungs on one shared grid."""
    noise = noise or NoiseSpec.from_level("none")
    backend = AerBackend(noise)
    grid, outside = prefit_window(post, grid_bits, seed)
    p = grid.target(post.log_prob)
    rows = []
    for rung in rungs:
        if rung not in VI_RUNGS:
            raise ValueError(f"unknown VI rung {rung!r}")
        rng = np.random.default_rng(seed + 1)
        t0 = time.time()
        opt = "parameter-shift" if rung == "vi-pshift-shots" else "cobyla"
        vi = BornMachineVI(grid, n_layers=n_layers, optimizer=opt, backend=backend)
        res = vi.fit(p, max_iter, rng)
        samples = vi.sample(res, n_samples, rng,
                            shots_backend=None if rung == "vi-cobyla-exact" else backend)
        mean, std = samples.mean(axis=0), samples.std(axis=0)
        fs = fit_statistics(post, mean)
        row = _base(post, noise, family="vi", rung=rung,
                    components={"vi-cobyla-exact": "", "vi-cobyla-shots": "sampling",
                                "vi-pshift-shots": "sampling+training"}[rung],
                    seed=seed, backend=backend.name, grid=grid_bits, shots=n_samples,
                    max_iter=max_iter, kl=res.kl, grid_mass_outside=outside,
                    corr_01=float(res.correlation[0, 1]),
                    circuit_evaluations=res.circuit_evaluations,
                    chi2_estimate=post.chi2(mean), chi2_map=fs["chi2"], chi2_red=fs["chi2_red"],
                    AIC=fs["AIC"], BIC=fs["BIC"], time_s=time.time() - t0)
        _moments(row, post.model.param_names, mean, std)
        rows.append(row)
    return rows


# --------------------------------------------------------------------------- #
def grid_floor(post: Posterior, grid_bits: int, max_points: int = 2 ** 22) -> float | None:
    """Lowest chi2 over all cell centres of the genetic grid (``None`` if too big)."""
    import itertools
    lo = np.array([b[0] for b in post.model.bounds])
    hi = np.array([b[1] for b in post.model.bounds])
    n = 2 ** grid_bits
    if n ** post.ndim > max_points:
        return None
    axes = [lo[i] + (np.arange(n) + 0.5) / n * (hi[i] - lo[i]) for i in range(post.ndim)]
    pts = np.array(list(itertools.product(*axes)))
    return float(min(np.min(post.chi2_batch(c)) for c in np.array_split(pts, max(1, len(pts) // 20000))))


def genetic_ladder(post: Posterior, *, grid_bits: int, pop_size: int = 120,
                   n_generations: int = 80, seed: int = 42, noise: NoiseSpec | None = None,
                   rungs: Sequence[str] = GA_RUNGS) -> list[dict]:
    """Run the genetic rungs. ``chi2_estimate`` is chi2 at the best individual."""
    from qablate.experimental import GAConfig, GeneticAlgorithm
    noise = noise or NoiseSpec.from_level("none")
    backend = AerBackend(noise)
    floor = grid_floor(post, grid_bits)
    rows = []
    for rung in rungs:
        if rung not in GA_RUNGS:
            raise ValueError(f"unknown genetic rung {rung!r}")
        quantum = rung.split("+")[1:] if rung.startswith("ga-grid+") else []
        t0 = time.time()
        ga = GeneticAlgorithm(post.log_prob, post.model.bounds,
                              GAConfig(pop_size=pop_size, n_generations=n_generations),
                              init_box=post.model.sample_box,
                              genome_bits=None if rung == "ga-continuous" else grid_bits,
                              quantum=quantum, backend=backend, rng=seed)
        r = ga.run()
        fs = fit_statistics(post, r.theta_best)
        row = _base(post, noise, family="genetic", rung=rung, components="+".join(quantum),
                    seed=seed, backend=backend.name,
                    grid="" if rung == "ga-continuous" else grid_bits,
                    pop_size=pop_size, n_generations=n_generations,
                    chi2_estimate=post.chi2(r.theta_best), chi2_map=fs["chi2"],
                    chi2_red=fs["chi2_red"], AIC=fs["AIC"], BIC=fs["BIC"],
                    chi2_grid_floor=None if rung == "ga-continuous" else floor,
                    time_s=time.time() - t0)
        # population spread is not an uncertainty: stored as mean only
        for n, v in zip(post.model.param_names, r.theta_best, strict=True):
            row[f"mean_{n}"] = float(v)
        rows.append(row)
    return rows
