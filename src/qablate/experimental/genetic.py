"""Genetic algorithms with classical and circuit-sampled operators (experimental).

A genetic algorithm maximizes ``log_prob`` with a population evolved by
tournament selection, elitism, crossover and mutation. Two genomes exist:

``genome_bits=None`` (continuous)
    Individuals are real vectors; crossover is a random convex blend and
    mutation adds Gaussian noise. Only classical operators are available.
``genome_bits=n`` (grid)
    Every parameter is an ``n``-bit index into ``2**n`` cells over the prior
    box; individuals are cell centres. Each operator has a classical and a
    circuit-sampled implementation with the same output distribution:

    ======================  ===============================  ====================================
    operator                classical                        circuit-sampled
    ======================  ===============================  ====================================
    initialization          uniform cell index               ``H`` on ``n`` qubits, measured
    mutation (per gene)     flip each bit with prob ``p``    ``RY`` per qubit, measured
    crossover (per gene)    keep agreeing bits; take A's     ``SWAP**alpha`` on (A_q, B_q) pairs,
                            bit w.p. ``cos^2(pi alpha/2)``   register A measured
    ======================  ===============================  ====================================

    The circuits start in basis states and are measured in the same basis,
    so their outputs are classical random bits; swapping them in tests the
    plumbing, and any difference from the classical twin is Monte Carlo
    noise. Compare a circuit-sampled run with the classical run on the SAME
    grid genome: comparing with the continuous genome mixes in the grid
    resolution (the best cell centre can be well above the continuous
    optimum).

Only the genes that are selected for mutation, and only the pairs that are
selected for crossover, are changed.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

__all__ = ["GAConfig", "GeneticAlgorithm", "GAResult", "OPERATORS"]

OPERATORS = ("init", "mutation", "crossover")


@dataclass
class GAConfig:
    """Hyper-parameters (validated on construction).

    Attributes:
        pop_size: Individuals per generation.
        n_generations: Generations.
        crossover_rate: Probability that a child is a crossover of two parents.
        mutation_rate: Probability that a gene mutates.
        mutation_scale: Gaussian mutation std as a fraction of the box width
            (continuous genome); on the grid genome it sets the bit-flip
            probability so that the mean absolute step matches.
        elite_frac: Fraction copied unchanged to the next generation (< 1).
        tournament_k: Tournament size (>= 1).
        crossover_alpha: ``SWAP**alpha`` exponent of the grid crossover.
    """

    pop_size: int = 120
    n_generations: int = 80
    crossover_rate: float = 0.9
    mutation_rate: float = 0.20
    mutation_scale: float = 0.12
    elite_frac: float = 0.08
    tournament_k: int = 3
    crossover_alpha: float = 0.5

    def __post_init__(self):
        if self.pop_size < 2 or self.n_generations < 1:
            raise ValueError("pop_size >= 2 and n_generations >= 1 required")
        for name in ("crossover_rate", "mutation_rate"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if not 0.0 <= self.elite_frac < 1.0:
            raise ValueError("elite_frac must be in [0, 1)")
        if self.tournament_k < 1:
            raise ValueError("tournament_k must be >= 1")
        if not 0.0 < self.crossover_alpha <= 1.0:
            raise ValueError("crossover_alpha must be in (0, 1]")


@dataclass
class GAResult:
    """Outcome of :meth:`GeneticAlgorithm.run`.

    ``population_spread`` is the fitness-weighted spread of the final
    population. It reflects selection pressure and mutation, not posterior
    uncertainty; do not report it as an error bar.
    """

    theta_best: np.ndarray
    log_prob_best: float
    population: np.ndarray
    log_prob: np.ndarray
    population_spread: np.ndarray
    history: list[dict[str, float]] = field(default_factory=list)
    config: dict[str, bool] = field(default_factory=dict)


def _bits(v: np.ndarray, n: int) -> np.ndarray:
    """Integers -> (..., n) bits, least-significant first (qubit order)."""
    return (np.asarray(v)[..., None] >> np.arange(n)) & 1


def _ints(b: np.ndarray) -> np.ndarray:
    """(..., n) bits, least-significant first -> integers."""
    return np.sum(np.asarray(b) << np.arange(b.shape[-1]), axis=-1)


def flip_probability(mutation_scale: float, n_bits: int, samples: int = 200_000) -> float:
    """Bit-flip probability whose mean absolute cell step equals the
    Gaussian mutation's ``E|dx| = mutation_scale * sqrt(2 / pi)`` (in box widths).

    Solved by bisection on a fixed-seed Monte Carlo estimate (deterministic).
    """
    target = mutation_scale * np.sqrt(2.0 / np.pi)
    rng = np.random.default_rng(12345)
    g = rng.integers(0, 2 ** n_bits, samples)
    u = rng.random((samples, n_bits))

    def step(p):
        flips = (u < p).astype(int)
        return np.mean(np.abs(_ints(_bits(g, n_bits) ^ flips) - g)) / 2 ** n_bits

    lo, hi = 1e-4, 0.5
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if step(mid) < target else (lo, mid)
    return 0.5 * (lo + hi)


class GeneticAlgorithm:
    """Genetic maximization of a vectorized ``log_prob``.

    Args:
        log_prob: ``(n, ndim) -> (n,)``; ``nan`` is treated as ``-inf``.
        bounds: Prior box ``[(lo, hi), ...]``.
        config: :class:`GAConfig`.
        init_box: Box where the initial population is drawn (defaults to
            ``bounds``). On the grid genome, only cells whose centre lies in
            it are drawn, for classical and circuit-sampled init alike.
        genome_bits: ``None`` for the continuous genome, or bits per parameter.
        quantum: Operators to run as circuits, a subset of
            ``('init', 'mutation', 'crossover')`` (grid genome only).
        backend: Backend for circuit-sampled operators (ideal Aer by default).
        rng: Generator or seed.
    """

    def __init__(self, log_prob: Callable, bounds: Sequence, config: GAConfig | None = None, *,
                 init_box: Sequence | None = None, genome_bits: int | None = None,
                 quantum: Sequence[str] = (), backend=None, rng=None):
        self.log_prob_fn = log_prob
        self.cfg = config or GAConfig()
        self.lo = np.array([b[0] for b in bounds], dtype=float)
        self.hi = np.array([b[1] for b in bounds], dtype=float)
        self.width = self.hi - self.lo
        self.ndim = len(self.lo)
        box = np.asarray(init_box if init_box is not None else bounds, dtype=float)
        self.ilo, self.ihi = box[:, 0], box[:, 1]
        self.nb = genome_bits
        self.quantum = set(quantum)
        if self.quantum - set(OPERATORS):
            raise ValueError(f"quantum operators must be among {OPERATORS}")
        if self.quantum and self.nb is None:
            raise ValueError("circuit-sampled operators need the grid genome (genome_bits)")
        self.rng = np.random.default_rng(rng)
        self.n_elite = max(1, int(round(self.cfg.elite_frac * self.cfg.pop_size)))
        if self.n_elite >= self.cfg.pop_size:
            raise ValueError("elite_frac leaves no room for children")
        if self.nb is not None:
            self.levels = 2 ** self.nb
            self.p_flip = flip_probability(self.cfg.mutation_scale, self.nb)
            centres = [self._centres(j) for j in range(self.ndim)]
            self._init_cells = [np.flatnonzero((c >= self.ilo[j]) & (c <= self.ihi[j]))
                                for j, c in enumerate(centres)]
            for j, cells in enumerate(self._init_cells):
                if len(cells) == 0:
                    raise ValueError(f"init_box has no cell centre in dimension {j}")
        self.backend = backend
        if self.quantum and self.backend is None:
            from ..backends import AerBackend
            self.backend = AerBackend()
        self._circuits = {}

    # ------------------------------------------------------------------ #
    def _centres(self, j: int) -> np.ndarray:
        return self.lo[j] + (np.arange(self.levels) + 0.5) / self.levels * self.width[j]

    def decode(self, genes: np.ndarray) -> np.ndarray:
        """Cell indices ``(P, d)`` -> cell centres."""
        return self.lo + (np.asarray(genes, dtype=float) + 0.5) / self.levels * self.width

    def fitness(self, pop: np.ndarray) -> np.ndarray:
        lp = np.asarray(self.log_prob_fn(pop), dtype=float)
        return np.where(np.isnan(lp), -np.inf, lp)

    def _reflect(self, x: np.ndarray) -> np.ndarray:
        eps = 1e-9 * self.width
        x = np.where(x < self.lo, 2 * self.lo - x, x)
        x = np.where(x > self.hi, 2 * self.hi - x, x)
        return np.clip(x, self.lo + eps, self.hi - eps)

    # -- circuits -------------------------------------------------------- #
    def _circuit(self, kind: str):
        if kind in self._circuits:
            return self._circuits[kind]
        from qiskit import QuantumCircuit
        from qiskit.circuit import ParameterVector
        from qiskit.circuit.library import UnitaryGate
        nb = self.nb
        if kind == "init":
            qc = QuantumCircuit(nb)
            qc.h(range(nb))
        elif kind == "mutation":
            th = ParameterVector("m", nb)
            qc = QuantumCircuit(nb)
            for q in range(nb):
                qc.ry(th[q], q)
        else:
            a, b = ParameterVector("a", nb), ParameterVector("b", nb)
            e = np.exp(1j * np.pi * self.cfg.crossover_alpha)
            swap_alpha = UnitaryGate(np.array([[1, 0, 0, 0], [0, (1 + e) / 2, (1 - e) / 2, 0],
                                               [0, (1 - e) / 2, (1 + e) / 2, 0], [0, 0, 0, 1]]),
                                     label="SWAP^a")
            qc = QuantumCircuit(2 * nb)
            for q in range(nb):
                qc.ry(a[q], q)
                qc.ry(b[q], nb + q)
            for q in range(nb):
                qc.append(swap_alpha, [q, nb + q])
        self._circuits[kind] = qc
        return qc

    # -- operators --------------------------------------------------------- #
    def init_population(self) -> np.ndarray:
        P, d = self.cfg.pop_size, self.ndim
        if self.nb is None:
            return self.ilo + self.rng.random((P, d)) * (self.ihi - self.ilo)
        genes = np.empty((P, d), dtype=int)
        for j, cells in enumerate(self._init_cells):
            if "init" in self.quantum:
                # rejection sampling of uniform circuit outcomes onto init_box cells
                out: list[int] = []
                while len(out) < P:
                    s = self.backend.sample(self._circuit("init"), np.zeros((1, 0)),
                                            self.rng, shots=2 * P)[0]
                    out += [int(v) for v in s if v in set(cells)]
                genes[:, j] = out[:P]
            else:
                genes[:, j] = self.rng.choice(cells, size=P)
        return self.decode(genes)

    def _tournament(self, pop, fit, n):
        idx = self.rng.integers(0, len(pop), size=(n, self.cfg.tournament_k))
        return pop[idx[np.arange(n), np.argmax(fit[idx], axis=1)]]

    def crossover(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        P, d = a.shape
        do = self.rng.random(P) < self.cfg.crossover_rate
        if self.nb is None:
            alpha = self.rng.random((P, d))
            return np.where(do[:, None], alpha * a + (1 - alpha) * b, a)
        ga, gb = self._encode(a), self._encode(b)
        child = ga.copy()
        rows = np.flatnonzero(do)
        if len(rows):
            ba, bb = _bits(ga[rows], self.nb), _bits(gb[rows], self.nb)   # (R, d, nb)
            if "crossover" in self.quantum:
                pv = np.concatenate([np.pi * ba, np.pi * bb], axis=2).reshape(-1, 2 * self.nb)
                s = self.backend.sample(self._circuit("crossover"), pv, self.rng, shots=1)[:, 0]
                new = (s & (self.levels - 1)).reshape(len(rows), d)
            else:
                keep_a = self.rng.random(ba.shape) < np.cos(np.pi * self.cfg.crossover_alpha / 2) ** 2
                new = _ints(np.where(ba == bb, ba, np.where(keep_a, ba, bb)))
            child[rows] = new
        return self.decode(child)

    def mutate(self, pop: np.ndarray) -> np.ndarray:
        P, d = pop.shape
        mask = self.rng.random((P, d)) < self.cfg.mutation_rate
        if self.nb is None:
            noise = self.rng.standard_normal((P, d)) * (self.cfg.mutation_scale * self.width)
            return self._reflect(pop + np.where(mask, noise, 0.0))
        genes = self._encode(pop)
        sel = np.argwhere(mask)
        if len(sel):
            g = genes[sel[:, 0], sel[:, 1]]
            b = _bits(g, self.nb)
            if "mutation" in self.quantum:
                p1 = np.where(b == 1, 1 - self.p_flip, self.p_flip)
                s = self.backend.sample(self._circuit("mutation"), 2 * np.arcsin(np.sqrt(p1)),
                                        self.rng, shots=1)[:, 0]
            else:
                s = _ints(b ^ (self.rng.random(b.shape) < self.p_flip))
            genes[sel[:, 0], sel[:, 1]] = s
        return self.decode(genes)

    def _encode(self, pop: np.ndarray) -> np.ndarray:
        frac = np.clip((pop - self.lo) / self.width, 0.0, 1.0 - 1e-12)
        return np.floor(frac * self.levels).astype(int)

    # ------------------------------------------------------------------ #
    def run(self) -> GAResult:
        """Evolve the population and return the best individual found."""
        cfg = self.cfg
        pop = self.init_population()
        fit = self.fitness(pop)
        history = []
        for gen in range(cfg.n_generations):
            order = np.argsort(-np.where(np.isfinite(fit), fit, -np.inf), kind="stable")
            elite = pop[order[:self.n_elite]]
            n_child = cfg.pop_size - self.n_elite
            children = self.mutate(self.crossover(self._tournament(pop, fit, n_child),
                                                  self._tournament(pop, fit, n_child)))
            pop = np.vstack([elite, children])
            fit = self.fitness(pop)
            fin = fit[np.isfinite(fit)]
            history.append({"generation": gen, "best_log_prob": float(np.max(fit)),
                            "mean_log_prob": float(np.mean(fin)) if len(fin) else float("nan")})
        best = int(np.argmax(fit))
        w = np.exp(fit - np.max(fit)) if np.isfinite(np.max(fit)) else np.zeros_like(fit)
        w = np.where(np.isfinite(w), w, 0.0)
        w = w / w.sum() if w.sum() > 0 else w
        mu = w @ pop
        spread = np.sqrt(np.maximum(w @ (pop - mu) ** 2, 0.0))
        return GAResult(theta_best=pop[best].copy(), log_prob_best=float(fit[best]),
                        population=pop, log_prob=fit, population_spread=spread,
                        history=history, config={op: op in self.quantum for op in OPERATORS})
