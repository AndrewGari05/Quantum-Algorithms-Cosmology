"""Proposal distributions for :class:`~qablate.mcmc.MetropolisHastings`.

A proposal maps the current positions ``x`` (shape ``(nchains, ndim)``) to
candidates and returns the Hastings term
``log q(x | x') - log q(x' | x)`` for each chain (zero for symmetric
proposals). Randomness comes only from the ``rng`` argument.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .backends import AerBackend, Backend

__all__ = ["Proposal", "GaussianProposal", "RandomCircuitProposal"]


class Proposal(ABC):
    """Interface of a proposal distribution."""

    #: True when q(x' | x) = q(x | x'); the Hastings term is then zero.
    symmetric: bool = True

    @abstractmethod
    def propose(self, x: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(candidates, log_q_ratio)`` for positions ``x``."""


class GaussianProposal(Proposal):
    """Random-walk proposal ``x' = x + scale * N(0, I)``.

    Args:
        scale: Step size, a scalar or one value per parameter.
    """

    def __init__(self, scale=1.0):
        self.scale = np.asarray(scale, dtype=float)

    def propose(self, x, rng):
        x = np.asarray(x, dtype=float)
        return x + self.scale * rng.standard_normal(x.shape), np.zeros(len(x))


class RandomCircuitProposal(Proposal):
    """Random walk whose increments are read from a random quantum circuit.

    For every proposal the angles of :func:`~qablate.circuits.random_proposal_circuit`
    (``max(2, ndim)`` qubits) are drawn uniformly and one ``ndim``-vector is
    read from the circuit:

    ``route='counts'``
        ``<Z_q> = 1 - 2 P(q = 1)`` for the first ``ndim`` qubits. This is
        what can be measured on hardware (exactly with ``shots=None`` on a
        simulator, or from ``shots`` measurements).
    ``route='statevector'``
        ``Re(psi_i) * sign(Im(psi_i))`` of the first ``ndim`` amplitudes
        (Sarracino et al. 2025). Requires an ideal simulator; it cannot be
        realized on hardware.

    The increment does not depend on the current position, so this is a
    random-walk Metropolis sampler with circuit-generated increments; no
    quantum advantage is implied. Each increment is scaled to unit root mean
    square per coordinate (constants estimated once from ``n_calibration``
    draws) and multiplied by an independent random sign, which makes the
    proposal exactly symmetric on any backend, including devices with
    asymmetric readout error. (Subtracting an estimated mean instead would
    leave a constant drift and bias the chain.)

    Args:
        scale: Step size, a scalar or one value per parameter.
        backend: Where circuits run; defaults to an ideal
            :class:`~qablate.backends.AerBackend`.
        route: ``'counts'`` or ``'statevector'``.
        n_layers: Layers of the proposal circuit.
        shots: Shots per increment (``None`` for exact probabilities).
        block_size: Increments generated per backend job.
        n_calibration: Draws used to estimate the scale constants.
    """

    def __init__(self, scale=1.0, *, backend: Backend | None = None,
                 route: str = "counts", n_layers: int = 3, shots: int | None = None,
                 block_size: int = 256, n_calibration: int = 1024):
        if route not in ("counts", "statevector"):
            raise ValueError("route must be 'counts' or 'statevector'")
        self.scale = np.asarray(scale, dtype=float)
        self.backend = backend if backend is not None else AerBackend()
        if route == "statevector" and not (self.backend.noiseless and hasattr(self.backend, "statevectors")):
            raise ValueError("route='statevector' needs an ideal simulator backend")
        if shots is None and not self.backend.exact:
            raise ValueError("this backend can only sample: pass shots")
        self.route = route
        self.n_layers = int(n_layers)
        self.shots = shots
        self.block_size = int(block_size)
        self.n_calibration = int(n_calibration)
        self.ndim: int | None = None
        self.rms: np.ndarray | None = None
        self._circuit = None
        self._queue: list[np.ndarray] = []

    def reset(self) -> None:
        """Forget the calibration and queued increments (called by the sampler)."""
        self.ndim = None
        self.rms = None
        self._queue = []

    def _raw(self, n: int, rng) -> np.ndarray:
        phis = rng.uniform(0.0, 2.0 * np.pi, size=(n, self._circuit.num_parameters))
        if self.route == "statevector":
            sv = self.backend.statevectors(self._circuit, phis, rng)
            return np.real(sv[:, :self.ndim]) * np.where(np.imag(sv[:, :self.ndim]) >= 0, 1.0, -1.0)
        probs = self.backend.probabilities(self._circuit, phis, rng, shots=self.shots)
        idx = np.arange(probs.shape[1])
        p1 = np.stack([probs[:, ((idx >> q) & 1) == 1].sum(axis=1)
                       for q in range(self.ndim)], axis=1)
        return 1.0 - 2.0 * p1

    def _setup(self, ndim: int, rng) -> None:
        from .circuits import random_proposal_circuit
        self.ndim = ndim
        self._circuit = random_proposal_circuit(max(2, ndim), self.n_layers)
        raw = self._raw(max(self.n_calibration, 2), rng)
        rms = np.sqrt(np.mean(raw ** 2, axis=0))
        self.rms = np.where(rms > 1e-12, rms, 1.0)

    def increments(self, n: int, rng) -> np.ndarray:
        """``n`` unit-scale increments (zero mean, unit RMS, symmetric)."""
        out = []
        while len(out) < n:
            if not self._queue:
                raw = self._raw(self.block_size, rng) / self.rms
                signs = 2.0 * rng.integers(0, 2, size=(self.block_size, 1)) - 1.0
                self._queue = list(signs * raw)
            out.append(self._queue.pop())
        return np.array(out)

    def propose(self, x, rng):
        x = np.asarray(x, dtype=float)
        if self.ndim is None:
            self._setup(x.shape[1], rng)
        elif x.shape[1] != self.ndim:
            raise ValueError("dimension changed between calls")
        return x + self.scale * self.increments(len(x), rng), np.zeros(len(x))
