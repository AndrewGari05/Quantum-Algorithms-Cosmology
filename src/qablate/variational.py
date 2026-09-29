"""Variational inference on a discretized parameter grid with a Born machine.

The posterior is discretized on a regular grid with ``2**n`` points per
parameter; parameter ``i`` is encoded in qubits ``[i*n, (i+1)*n)`` (Qiskit
little-endian, the lowest qubit of each block is the most significant bit
of that parameter's index). A parameterized circuit ``U(phi)`` defines the
variational distribution ``Q_phi(k) = |<k|U(phi)|0>|^2`` (a Born machine),
trained to minimize the reverse Kullback-Leibler divergence ``KL(Q || P)``
to the grid target ``P``.

Reverse KL is mode seeking: when ``Q`` cannot represent the correlations of
``P`` it under-estimates marginal widths (Bishop 2006, section 10.1.2).
Report :attr:`VIResult.correlation` next to the widths.

Three optimizers are provided: ``'cobyla'`` (gradient free, SciPy),
``'parameter-shift'`` (exact gradients of ``KL`` from ``2 * n_params``
shifted circuits, chain rule through ``Q``) and ``'spsa'`` (Spall 1998: a
gradient estimate from 2 circuits per iteration whatever the number of
parameters). COBYLA and parameter-shift need exact probabilities; SPSA also
trains from measured frequencies (``shots``), which is what hardware
provides, and then minimizes the plug-in estimate ``KL(Q_hat || P)``
(biased low for few shots). Every circuit evaluation is counted in
:attr:`VIResult.circuit_evaluations`, so budgets can be matched.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from .backends import AerBackend, Backend

__all__ = ["Grid", "BornMachineVI", "VIResult"]


class Grid:
    """Regular grid over a box, with the qubit encoding used by the Born machine.

    Args:
        window: ``(lo, hi)`` per parameter.
        n_qubits_per_param: Grid points per parameter are ``2**n``.

    Examples:
        >>> g = Grid([(0.0, 1.0), (10.0, 20.0)], 2)
        >>> g.points.shape
        (16, 2)
        >>> g.points[0].tolist(), g.points[1].tolist()   # qubit 0 is the MSB of parameter 0
        ([0.0, 10.0], [0.6666666666666666, 10.0])
    """

    def __init__(self, window: Sequence, n_qubits_per_param: int):
        self.window = [(float(lo), float(hi)) for lo, hi in window]
        self.ndim = len(self.window)
        self.nqpp = int(n_qubits_per_param)
        self.n_qubits = self.ndim * self.nqpp
        self.n_per_axis = 2 ** self.nqpp
        self.axes = [np.linspace(lo, hi, self.n_per_axis) for lo, hi in self.window]
        idx = np.arange(2 ** self.n_qubits)
        pts = np.zeros((len(idx), self.ndim))
        for i in range(self.ndim):
            val = np.zeros(len(idx), dtype=int)
            for j in range(self.nqpp):
                val |= ((idx >> (i * self.nqpp + j)) & 1) << (self.nqpp - 1 - j)
            pts[:, i] = self.axes[i][val]
        self.points = pts

    @classmethod
    def around(cls, samples: np.ndarray, n_qubits_per_param: int, *,
               sigma_mult: float = 4.0, bounds: Sequence | None = None,
               quantiles: Sequence[float] | None = None) -> Grid:
        """Grid centred on samples: mean +- ``sigma_mult`` std, clipped to ``bounds``.

        With ``quantiles=(q_lo, q_hi)`` the window is the sample quantile
        range widened by ``sigma_mult`` times a quarter of its width instead,
        which is robust for skewed posteriors.
        """
        s = np.asarray(samples, dtype=float)
        if quantiles is not None:
            lo, hi = np.quantile(s, quantiles, axis=0)
            pad = 0.25 * sigma_mult * (hi - lo)
            lo, hi = lo - pad, hi + pad
        else:
            mu, sd = s.mean(axis=0), s.std(axis=0)
            lo, hi = mu - sigma_mult * sd, mu + sigma_mult * sd
        if bounds is not None:
            b = np.asarray(bounds, dtype=float)
            lo, hi = np.maximum(lo, b[:, 0]), np.minimum(hi, b[:, 1])
        return cls(list(zip(lo, hi, strict=True)), n_qubits_per_param)

    def target(self, log_prob: Callable) -> np.ndarray:
        """Normalized target ``P`` on the grid from a vectorized ``log_prob``.

        Raises:
            ValueError: If no grid point has finite probability.
        """
        lp = np.concatenate([np.asarray(log_prob(c), dtype=float)       # bounded memory
                             for c in np.array_split(self.points, max(1, len(self.points) // 4096))])
        ok = np.isfinite(lp)
        if not np.any(ok):
            raise ValueError("no grid point has finite log-probability; check the window")
        p = np.zeros(len(lp))
        p[ok] = np.exp(lp[ok] - lp[ok].max())
        return p / p.sum()

    def moments(self, q: np.ndarray):
        """Exact mean, standard deviations and correlation matrix under ``q``."""
        q = np.asarray(q, dtype=float) / np.sum(q)
        mu = q @ self.points
        d = self.points - mu
        cov = (d * q[:, None]).T @ d
        sd = np.sqrt(np.diag(cov))
        corr = cov / np.outer(sd, sd)
        return mu, sd, corr


def reverse_kl(q: np.ndarray, p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """``KL(Q || P)`` over the full support with ``P`` smoothed by ``eps``.

    Mass that ``Q`` puts where ``P`` is zero is penalized (it is not
    renormalized away). ``q`` may be ``(n,)`` or ``(B, n)``.
    """
    ps = (p + eps) / np.sum(p + eps)
    q = np.atleast_2d(q)
    return np.sum(q * (np.log(np.clip(q, eps, None)) - np.log(ps)), axis=1)


@dataclass
class VIResult:
    """Outcome of :meth:`BornMachineVI.fit`."""

    phi: np.ndarray
    q: np.ndarray
    kl: float
    mean: np.ndarray
    std: np.ndarray
    correlation: np.ndarray
    history: list[float] = field(default_factory=list)
    circuit_evaluations: int = 0
    #: SPSA only: the parameters after every iteration (for post-hoc exact KL).
    phi_history: np.ndarray | None = None
    samples: np.ndarray | None = None


class BornMachineVI:
    """Reverse-KL variational inference with a Born-machine ansatz on a grid.

    Args:
        grid: The :class:`Grid`.
        n_layers: Layers of :func:`~qablate.circuits.hardware_efficient_ansatz`.
        optimizer: ``'cobyla'``, ``'parameter-shift'`` or ``'spsa'``.
        backend: Where circuits run; ideal Aer by default. COBYLA and
            parameter-shift need exact probabilities (a simulator).
        shots: Shots per circuit evaluation (``None``: exact probabilities).
            Only ``'spsa'`` accepts shots.
        spsa_gains: ``(a, c)`` of the SPSA schedule ``a_k = a / (k + 1)**0.602``,
            ``c_k = c / (k + 1)**0.101`` (the thesis hardware values).
        learning_rate: Initial step of parameter-shift gradient descent; the
            step decays as ``lr / (1 + decay * i)`` and the gradient norm is
            clipped to 1 (the schedule of the thesis runs).
        decay: Step decay; defaults to ``0.02 * max(1, n_params / 42)``.
    """

    def __init__(self, grid: Grid, *, n_layers: int = 3, optimizer: str = "cobyla",
                 backend: Backend | None = None, learning_rate: float = 0.05,
                 decay: float | None = None, shots: int | None = None,
                 spsa_gains: tuple[float, float] = (0.15, 0.1)):
        from .circuits import hardware_efficient_ansatz
        if optimizer not in ("cobyla", "parameter-shift", "spsa"):
            raise ValueError("optimizer must be 'cobyla', 'parameter-shift' or 'spsa'")
        self.grid = grid
        self.optimizer = optimizer
        self.backend = backend if backend is not None else AerBackend()
        if shots is not None and optimizer != "spsa":
            raise ValueError("only optimizer='spsa' trains from shots")
        if shots is None and not self.backend.exact:
            raise ValueError("this backend only samples: use optimizer='spsa' with shots")
        self.shots = None if shots is None else int(shots)
        self.spsa_gains = (float(spsa_gains[0]), float(spsa_gains[1]))
        self.circuit = hardware_efficient_ansatz(grid.n_qubits, n_layers)
        self.n_params = self.circuit.num_parameters
        self.learning_rate = float(learning_rate)
        self.decay = (0.02 * max(1.0, self.n_params / 42.0) if decay is None else float(decay))
        self._evals = 0

    def distributions(self, phis: np.ndarray, rng) -> np.ndarray:
        """``Q_phi`` for each row of ``phis`` (counted as circuit evaluations)."""
        phis = np.atleast_2d(phis)
        self._evals += len(phis)
        return self.backend.probabilities(self.circuit, phis, rng, shots=self.shots)

    def fit(self, p: np.ndarray, max_iter: int, rng, *, budget: str = "circuits",
            initial: np.ndarray | None = None) -> VIResult:
        """Train on target ``p``.

        Args:
            p: Grid target from :meth:`Grid.target`.
            max_iter: Iterations of parameter-shift descent or SPSA (2 circuit
                evaluations each, plus one final evaluation). COBYLA gets the
                same number of circuit evaluations (``budget='circuits'``,
                ``max_iter * (1 + 2 * n_params)``) or of iterations
                (``budget='iterations'``).
            rng: ``numpy.random.Generator``; draws the initial angles.
            initial: Start from these angles instead (the initial draw is
                still made, so the rest of the random stream is unchanged).
        """
        from scipy.optimize import minimize
        self._evals = 0
        rng = np.random.default_rng(rng)
        phi = 0.1 * rng.standard_normal(self.n_params)
        if initial is not None:
            phi = np.array(initial, dtype=float).reshape(self.n_params)
        history: list[float] = []
        phi_history = None
        if self.optimizer == "spsa":
            a, c = self.spsa_gains
            phi_history = np.empty((int(max_iter), self.n_params))
            for k in range(int(max_iter)):
                ak, ck = a / (k + 1) ** 0.602, c / (k + 1) ** 0.101
                delta = rng.choice([-1.0, 1.0], size=self.n_params)
                qs = self.distributions(np.vstack([phi + ck * delta, phi - ck * delta]), rng)
                f_plus, f_minus = reverse_kl(qs, p)
                phi = phi - ak * (f_plus - f_minus) / (2.0 * ck) * delta
                history.append(0.5 * float(f_plus + f_minus))
                phi_history[k] = phi
        elif self.optimizer == "parameter-shift":
            best_kl, best_phi = np.inf, phi.copy()
            ps = (p + 1e-12) / np.sum(p + 1e-12)
            for i in range(int(max_iter)):
                q = self.distributions(phi, rng)[0]
                kl = float(reverse_kl(q, p)[0])
                history.append(kl)
                if kl < best_kl:
                    best_kl, best_phi = kl, phi.copy()
                shifts = np.repeat(phi[None, :], 2 * self.n_params, axis=0)
                j = np.arange(self.n_params)
                shifts[2 * j, j] += np.pi / 2
                shifts[2 * j + 1, j] -= np.pi / 2
                qs = self.distributions(shifts, rng)
                dq = (qs[0::2] - qs[1::2]) / 2.0
                grad = dq @ (np.log(np.clip(q, 1e-12, None)) - np.log(ps))
                gnorm = float(np.linalg.norm(grad))
                if gnorm > 1.0:
                    grad /= gnorm
                phi = phi - self.learning_rate / (1.0 + self.decay * i) * grad
            phi = best_phi
        else:
            cap = int(max_iter) * ((1 + 2 * self.n_params) if budget == "circuits" else 1)
            cap = max(cap, self.n_params + 2)

            def cost(ph):
                kl = float(reverse_kl(self.distributions(ph, rng)[0], p)[0])
                history.append(kl)
                return kl
            phi = minimize(cost, phi, method="COBYLA",
                           options={"maxiter": cap, "rhobeg": 0.3}).x
        q = self.distributions(phi, rng)[0]
        mu, sd, corr = self.grid.moments(q)
        return VIResult(phi=phi, q=q, kl=float(reverse_kl(q, p)[0]), mean=mu, std=sd,
                        correlation=corr, history=history,
                        circuit_evaluations=self._evals, phi_history=phi_history)

    def sample(self, result: VIResult, n: int, rng, *, shots_backend: Backend | None = None) -> np.ndarray:
        """Draw ``n`` grid points from the trained distribution.

        With ``shots_backend`` the samples are measurement outcomes of the
        trained circuit on that backend (possibly noisy hardware); otherwise
        they are drawn classically from the exact ``Q``.
        """
        if shots_backend is None:
            idx = rng.choice(len(result.q), size=n, p=result.q / result.q.sum())
        else:
            f = shots_backend.probabilities(self.circuit, result.phi, rng, shots=n)[0]
            idx = np.repeat(np.arange(len(f)), np.rint(f * n).astype(int))
        return self.grid.points[idx]
