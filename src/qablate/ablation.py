"""Classical-quantum ablation studies.

An ablation study names the interchangeable components of an algorithm,
each with a classical and a quantum implementation, runs configurations
that switch components one at a time (a *ladder*), and records what
changes. Two kinds of comparison are distinguished:

* **equivalence checks** — pairs of configurations that must give identical
  outputs by construction (for example an exact amplitude-encoded acceptance
  versus classical Metropolis on the same random stream). They verify the
  implementation; they are not evidence about quantum hardware.
* **treatments** — every other pair, whose differences are the result.

Every configuration is run from the same seed, so paired configurations
share their random streams wherever their components make the same draws.

Examples:
    >>> import numpy as np
    >>> from qablate import (Study, MetropolisHastings, GaussianProposal,
    ...                      MetropolisAcceptance, AmplitudeEncodedMetropolis)
    >>> log_prob = lambda x: -0.5 * np.sum(x**2, axis=1)
    >>> def run(parts, rng):
    ...     mh = MetropolisHastings(log_prob, 1, 2, rng=rng,
    ...                             proposal=GaussianProposal(1.0), **parts)
    ...     mh.run_mcmc(np.zeros((2, 1)), 50)
    ...     return {"chain": mh.get_chain()}
    >>> study = Study({"acceptance": (MetropolisAcceptance, AmplitudeEncodedMetropolis)},
    ...               run, equivalent=[({}, {"acceptance": True})])
    >>> results = study.run(study.ladder(), seed=1)
    >>> study.check_equivalences(results, keys=["chain"])
    [('acceptance=C', 'acceptance=Q', True)]
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

__all__ = ["Study", "Run"]


@dataclass
class Run:
    """One configuration of a study and its outputs."""

    config: dict[str, bool]
    outputs: dict[str, object]
    label: str = ""
    quantum_fraction: float = 0.0
    seed: int | None = None
    extra: dict[str, object] = field(default_factory=dict)


class Study:
    """A set of interchangeable components and a function that runs them.

    Args:
        components: ``{name: (classical_factory, quantum_factory)}``. A
            factory is called with no arguments for every run, so stateful
            components start fresh.
        run: ``run(parts, rng) -> dict`` where ``parts`` maps each component
            name to the instance chosen for this configuration.
        equivalent: Pairs of configurations (``{name: is_quantum}``, missing
            names are classical) declared identical by construction.
    """

    def __init__(self, components: dict[str, tuple[Callable, Callable]],
                 run: Callable[[dict[str, object], np.random.Generator], dict[str, object]],
                 equivalent: Sequence[tuple[dict[str, bool], dict[str, bool]]] = ()):
        self.components = dict(components)
        self._run = run
        self.equivalent = [(self._full(a), self._full(b)) for a, b in equivalent]

    def _full(self, config: dict[str, bool]) -> dict[str, bool]:
        unknown = set(config) - set(self.components)
        if unknown:
            raise ValueError(f"unknown components {sorted(unknown)}")
        return {n: bool(config.get(n, False)) for n in self.components}

    def label(self, config: dict[str, bool]) -> str:
        """Readable label, e.g. ``'proposal=Q,acceptance=C'``."""
        c = self._full(config)
        return ",".join(f"{n}={'Q' if q else 'C'}" for n, q in c.items())

    def quantum_fraction(self, config: dict[str, bool]) -> float:
        """Fraction of components that are quantum (bookkeeping only)."""
        c = self._full(config)
        return sum(c.values()) / len(c)

    def ladder(self, order: Sequence[str] | None = None) -> list[dict[str, bool]]:
        """Configurations that switch components to quantum one at a time."""
        order = list(order) if order is not None else list(self.components)
        rungs, cur = [self._full({})], {}
        for name in order:
            cur = dict(cur, **{name: True})
            rungs.append(self._full(cur))
        return rungs

    def run(self, configs: Sequence[dict[str, bool]], seed: int) -> list[Run]:
        """Run each configuration from ``numpy.random.default_rng(seed)``."""
        out = []
        for cfg in configs:
            full = self._full(cfg)
            parts = {n: self.components[n][1 if q else 0]() for n, q in full.items()}
            outputs = self._run(parts, np.random.default_rng(seed))
            out.append(Run(config=full, outputs=outputs, label=self.label(full),
                           quantum_fraction=self.quantum_fraction(full), seed=seed))
        return out

    def check_equivalences(self, runs: Sequence[Run], keys: Sequence[str],
                           strict: bool = False) -> list[tuple[str, str, bool]]:
        """Compare declared-equivalent pairs output by output (exact equality).

        Returns ``(label_a, label_b, identical)`` for every declared pair
        present in ``runs``; with ``strict=True`` raises on the first failure.
        """
        by_label = {r.label: r for r in runs}
        report = []
        for a, b in self.equivalent:
            ra, rb = by_label.get(self.label(a)), by_label.get(self.label(b))
            if ra is None or rb is None:
                continue
            same = all(np.array_equal(np.asarray(ra.outputs[k]), np.asarray(rb.outputs[k]))
                       for k in keys)
            if strict and not same:
                raise AssertionError(f"declared equivalent but different: {ra.label} vs {rb.label}")
            report.append((ra.label, rb.label, bool(same)))
        return report
