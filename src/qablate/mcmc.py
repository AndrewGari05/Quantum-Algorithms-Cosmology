"""Metropolis-Hastings with swappable proposal and acceptance rules.

The transition kernel is owned by this module so that each of its two
components can be replaced independently:

* a :class:`~qablate.proposals.Proposal` draws a candidate and reports the
  Hastings term ``log q(x | x') - log q(x' | x)``;
* an :class:`~qablate.acceptance.Acceptance` decides, from the log acceptance
  ratio, which candidates are kept.

The public interface follows emcee's naming (``run_mcmc``, ``get_chain``,
``get_log_prob``, ``acceptance_fraction``) so that existing analysis code
works unchanged. Chains are stored with shape ``(nsteps, nchains, ndim)``.
"""
from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .acceptance import Acceptance, MetropolisAcceptance
from .proposals import GaussianProposal, Proposal

__all__ = ["MetropolisHastings", "State"]


@dataclass
class State:
    """Positions and log-probabilities of every chain at one step."""

    coords: np.ndarray
    log_prob: np.ndarray


class MetropolisHastings:
    """Parallel Metropolis-Hastings chains with pluggable components.

    Args:
        log_prob: Log target density. With ``vectorized=True`` (default) it
            maps an ``(n, ndim)`` array to an ``(n,)`` array; otherwise it
            maps one ``(ndim,)`` point to a float. It may return ``-inf``;
            ``nan`` is treated as ``-inf``.
        ndim: Number of parameters.
        nchains: Number of independent chains advanced in lockstep.
        proposal: Candidate generator. Defaults to an isotropic
            :class:`~qablate.proposals.GaussianProposal` with unit scale.
        acceptance: Acceptance rule. Defaults to
            :class:`~qablate.acceptance.MetropolisAcceptance`.
        rng: A :class:`numpy.random.Generator` (or seed). It drives every
            random decision of the sampler, including those of the proposal
            and the acceptance rule, so a run is reproducible from it alone.
        vectorized: Whether ``log_prob`` accepts a batch of points.

    Raises:
        ValueError: If ``ndim`` or ``nchains`` is not positive.

    Examples:
        >>> import numpy as np
        >>> from qablate import MetropolisHastings, GaussianProposal
        >>> log_prob = lambda x: -0.5 * np.sum(x**2, axis=1)
        >>> mh = MetropolisHastings(log_prob, ndim=2, nchains=4,
        ...                         proposal=GaussianProposal(0.8), rng=0)
        >>> _ = mh.run_mcmc(np.zeros((4, 2)), 2000)
        >>> mh.get_chain(discard=200, flat=True).shape
        (7200, 2)
    """

    def __init__(self, log_prob: Callable, ndim: int, nchains: int = 4, *,
                 proposal: Proposal | None = None,
                 acceptance: Acceptance | None = None,
                 rng=None, vectorized: bool = True):
        if int(ndim) < 1 or int(nchains) < 1:
            raise ValueError("ndim and nchains must be positive integers")
        self.ndim = int(ndim)
        self.nchains = int(nchains)
        self.proposal = proposal if proposal is not None else GaussianProposal(1.0)
        self.acceptance = acceptance if acceptance is not None else MetropolisAcceptance()
        self.rng = np.random.default_rng(rng)
        self.vectorized = bool(vectorized)
        self._log_prob = log_prob
        self._chain = np.empty((0, self.nchains, self.ndim))
        self._lnp = np.empty((0, self.nchains))
        self._accepted = np.zeros(self.nchains, dtype=int)
        self._state: State | None = None
        if not self.proposal.symmetric and not getattr(
                self.proposal, "returns_hastings_term", True):
            raise ValueError("an asymmetric proposal must return its Hastings term")
        if not self.acceptance.preserves_detailed_balance:
            warnings.warn(
                f"{type(self.acceptance).__name__} does not satisfy detailed "
                "balance in its current configuration; the chain will not "
                "sample log_prob exactly.", UserWarning, stacklevel=2)

    # ------------------------------------------------------------------ #
    def compute_log_prob(self, coords: np.ndarray) -> np.ndarray:
        """Evaluate ``log_prob`` on an ``(n, ndim)`` array; ``nan`` -> ``-inf``."""
        coords = np.atleast_2d(np.asarray(coords, dtype=float))
        if self.vectorized:
            lp = np.asarray(self._log_prob(coords), dtype=float).reshape(-1)
        else:
            lp = np.array([float(self._log_prob(c)) for c in coords])
        if lp.shape != (coords.shape[0],):
            raise ValueError(f"log_prob returned shape {lp.shape}, "
                             f"expected ({coords.shape[0]},)")
        return np.where(np.isnan(lp), -np.inf, lp)

    def run_mcmc(self, initial_state, nsteps: int, *, progress: bool = False) -> State:
        """Advance every chain ``nsteps`` times and store the samples.

        Args:
            initial_state: ``(nchains, ndim)`` starting positions, a
                :class:`State`, or ``None`` to continue from the last state.
            nsteps: Number of steps to take (stored samples).
            progress: Show a progress bar (requires ``tqdm``).

        Returns:
            The final :class:`State`.
        """
        nsteps = int(nsteps)
        if nsteps < 1:
            raise ValueError("nsteps must be a positive integer")
        if initial_state is None:
            if self._state is None:
                raise ValueError("no previous state; pass initial_state")
            state = self._state
        elif isinstance(initial_state, State):
            state = initial_state
        else:
            x = np.array(initial_state, dtype=float)
            if x.shape != (self.nchains, self.ndim):
                raise ValueError(f"initial_state must have shape "
                                 f"({self.nchains}, {self.ndim}), got {x.shape}")
            state = State(x, self.compute_log_prob(x))
        if not np.all(np.isfinite(state.log_prob)):
            raise ValueError("log_prob is not finite at some initial positions")

        chain = np.empty((nsteps, self.nchains, self.ndim))
        lnp = np.empty((nsteps, self.nchains))
        x, lp = state.coords.copy(), state.log_prob.copy()
        steps = range(nsteps)
        if progress:
            from tqdm.auto import tqdm
            steps = tqdm(steps, leave=False)
        for t in steps:
            x_new, log_q_ratio = self.proposal.propose(x, self.rng)
            lp_new = self.compute_log_prob(x_new)
            log_alpha = lp_new - lp + log_q_ratio
            log_alpha = np.where(np.isfinite(lp_new), log_alpha, -np.inf)
            accept = self.acceptance.accept(log_alpha, self.rng)
            x[accept] = x_new[accept]
            lp[accept] = lp_new[accept]
            self._accepted += accept
            chain[t], lnp[t] = x, lp
        self._chain = np.concatenate([self._chain, chain])
        self._lnp = np.concatenate([self._lnp, lnp])
        self._state = State(x.copy(), lp.copy())
        return self._state

    # ------------------------------------------------------------------ #
    def get_chain(self, discard: int = 0, thin: int = 1, flat: bool = False) -> np.ndarray:
        """Stored samples, shape ``(nsteps, nchains, ndim)`` or flattened."""
        c = self._chain[discard::thin]
        return c.reshape(-1, self.ndim) if flat else c

    def get_log_prob(self, discard: int = 0, thin: int = 1, flat: bool = False) -> np.ndarray:
        """Stored log-probabilities, shape ``(nsteps, nchains)`` or flattened."""
        v = self._lnp[discard::thin]
        return v.reshape(-1) if flat else v

    @property
    def iteration(self) -> int:
        """Number of stored steps."""
        return self._chain.shape[0]

    @property
    def acceptance_fraction(self) -> np.ndarray:
        """Fraction of accepted proposals, per chain."""
        return self._accepted / max(self.iteration, 1)

    def to_inference_data(self, discard: int = 0, param_names=None):
        """Samples as an ``arviz.InferenceData`` (requires arviz)."""
        import arviz as az
        chain = np.swapaxes(self.get_chain(discard=discard), 0, 1)
        names = list(param_names) if param_names else [f"x{i}" for i in range(self.ndim)]
        return az.from_dict(posterior={n: chain[..., i] for i, n in enumerate(names)})

    def reset(self) -> None:
        """Discard stored samples (the last state is kept)."""
        self._chain = np.empty((0, self.nchains, self.ndim))
        self._lnp = np.empty((0, self.nchains))
        self._accepted[:] = 0
