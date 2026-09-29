"""Acceptance rules for :class:`~qablate.mcmc.MetropolisHastings`.

An acceptance rule receives the log acceptance ratio
``log_alpha = log p(x') - log p(x) + log q(x | x') - log q(x' | x)`` of every
chain and returns which candidates are accepted.

A rule preserves detailed balance with respect to ``p`` when its acceptance
probability ``h`` satisfies ``h(a) / h(-a) = exp(a)`` for every ``a``.
Metropolis, ``h = min(1, e^a)``, does. :attr:`Acceptance.preserves_detailed_balance`
reports whether a configured rule does; the sampler warns when it does not.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .backends import AerBackend, Backend

__all__ = ["Acceptance", "MetropolisAcceptance", "AmplitudeEncodedMetropolis"]


class Acceptance(ABC):
    """Interface of an acceptance rule."""

    preserves_detailed_balance: bool = True

    @abstractmethod
    def accept(self, log_alpha: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Boolean mask of accepted candidates."""

    def probability(self, log_alpha: np.ndarray) -> np.ndarray:
        """Acceptance probability for each ``log_alpha`` (for diagnostics)."""
        return np.exp(np.minimum(np.asarray(log_alpha, dtype=float), 0.0))


class MetropolisAcceptance(Acceptance):
    """Metropolis rule: accept when ``log u < log_alpha``, ``u ~ U(0, 1)``."""

    def accept(self, log_alpha, rng):
        log_alpha = np.asarray(log_alpha, dtype=float)
        return np.log(rng.random(log_alpha.shape)) < log_alpha


class AmplitudeEncodedMetropolis(Acceptance):
    """Metropolis acceptance probability loaded into, and read from, a qubit.

    ``A = min(1, e^log_alpha)`` is computed classically and encoded as
    ``RY(2 arccos(sqrt(A)))``, whose probability of outcome 0 is ``A``; that
    probability ``P0`` is read back from the backend and the candidate is
    accepted when ``log u < log P0``. On an ideal backend with exact
    probabilities this reproduces :class:`MetropolisAcceptance` decision for
    decision (same ``u``); it is a consistency check of the quantum
    plumbing, not a Hadamard test and not a source of speed-up. With
    ``shots`` the estimate of ``P0`` is unbiased, so the chain still targets
    ``p``.

    Under noise the backend returns ``P0 = f(A)`` with ``f`` affine but not
    identity (for a symmetric readout flip ``p``: ``(1 - 2p) A + p``), which
    breaks detailed balance: every candidate is accepted with probability at
    least ``p``. ``readout_mitigation=True`` inverts the readout confusion
    matrix of the measured qubit; that restores detailed balance only when
    readout is the sole noise source (with ``shots``, the mitigated estimate
    is clipped to [0, 1] and is then no longer exactly unbiased).

    Args:
        backend: Where the one-qubit circuits run; ideal Aer by default.
        shots: Shots per chain and step (``None`` for exact probabilities).
        readout_mitigation: Invert the readout matrix of the measured qubit.
    """

    def __init__(self, backend: Backend | None = None, shots: int | None = None,
                 readout_mitigation: bool = False):
        from .circuits import amplitude_encoding_circuit
        self.backend = backend if backend is not None else AerBackend()
        if shots is None and not self.backend.exact:
            raise ValueError("this backend can only sample: pass shots")
        self.shots = shots
        self.readout_mitigation = bool(readout_mitigation)
        self._circuit = amplitude_encoding_circuit()
        noise = getattr(self.backend, "noise", None)
        only_readout = noise is not None and noise.label == "readout"
        self.preserves_detailed_balance = bool(
            self.backend.noiseless or (self.readout_mitigation and only_readout))
        self._m = None
        if self.readout_mitigation and noise is not None and noise.has_readout:
            self._m = noise.readout_matrix(0)

    def estimate(self, a: np.ndarray, rng) -> np.ndarray:
        """Backend estimate of ``P0`` for acceptance probabilities ``a``."""
        t = 2.0 * np.arccos(np.sqrt(np.clip(a, 0.0, 1.0)))
        p0 = self.backend.probabilities(self._circuit, t[:, None], rng, shots=self.shots)[:, 0]
        if self._m is not None:
            m00, m01 = self._m[0, 0], self._m[0, 1]
            p0 = np.clip((p0 - m01) / (m00 - m01), 0.0, 1.0)
        return p0

    def accept(self, log_alpha, rng):
        log_alpha = np.asarray(log_alpha, dtype=float)
        u = rng.random(log_alpha.shape)
        p0 = self.estimate(np.exp(np.minimum(log_alpha, 0.0)), rng)
        with np.errstate(divide="ignore"):
            return np.log(u) < np.log(p0)
