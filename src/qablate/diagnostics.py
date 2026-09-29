"""Convergence diagnostics.

R-hat and effective sample size are computed by ArviZ (rank-normalized
split R-hat with its folded tail term, bulk ESS; Vehtari et al. 2021). The
integrated autocorrelation time follows emcee's FFT estimator with Sokal's
automatic window. A parameter that never moves is reported as not converged
(R-hat = inf, ESS = 0) instead of being silently skipped.

Chains are ``(nchains, nsteps)`` for one parameter or
``(nchains, nsteps, ndim)`` for several.
"""
from __future__ import annotations

import numpy as np

__all__ = ["rhat", "ess", "autocorr_time", "summary", "RHAT_THRESHOLD", "ESS_MIN"]

#: Vehtari et al. (2021): R-hat below 1.01 ...
RHAT_THRESHOLD = 1.01
#: ... and bulk ESS above 400 before trusting summaries.
ESS_MIN = 400


def _as_3d(chains) -> np.ndarray:
    c = np.asarray(chains, dtype=float)
    if c.ndim == 2:
        c = c[:, :, None]
    if c.ndim != 3:
        raise ValueError("chains must have shape (nchains, nsteps[, ndim])")
    return c


def _frozen(c: np.ndarray) -> np.ndarray:
    """True for parameters that take a single value across all chains."""
    return np.array([np.ptp(c[:, :, j]) == 0.0 for j in range(c.shape[2])], dtype=bool)


def rhat(chains) -> np.ndarray:
    """Rank-normalized split R-hat per parameter (max of bulk and tail).

    Examples:
        >>> rng = np.random.default_rng(0)
        >>> bool(rhat(rng.standard_normal((4, 1000)))[0] < 1.01)
        True
    """
    import arviz as az
    c = _as_3d(chains)
    out = np.array([float(az.rhat(c[:, :, j], method="rank")) for j in range(c.shape[2])])
    out[_frozen(c)] = np.inf
    return out


def ess(chains) -> np.ndarray:
    """Bulk effective sample size per parameter (accounts for between-chain spread)."""
    import arviz as az
    c = _as_3d(chains)
    out = np.array([float(az.ess(c[:, :, j], method="bulk")) for j in range(c.shape[2])])
    out[_frozen(c)] = 0.0
    return out


def _autocorr_1d(x: np.ndarray) -> np.ndarray:
    n = len(x)
    f = np.fft.rfft(x - x.mean(), n=2 * n)
    acf = np.fft.irfft(f * np.conjugate(f))[:n]
    return acf / acf[0]


def autocorr_time(chains, c: float = 5.0) -> np.ndarray:
    """Integrated autocorrelation time per parameter (emcee's estimator).

    The autocorrelation function is averaged over chains before the window
    is chosen (Goodman & Weare 2010, as in ``emcee.autocorr``). A frozen
    parameter gives ``inf``.
    """
    ch = _as_3d(chains)
    out = np.empty(ch.shape[2])
    for j in range(ch.shape[2]):
        if np.ptp(ch[:, :, j]) == 0.0:
            out[j] = np.inf
            continue
        f = np.mean([_autocorr_1d(ch[k, :, j]) for k in range(ch.shape[0])], axis=0)
        taus = 2.0 * np.cumsum(f) - 1.0
        m = np.arange(len(taus)) < c * taus
        window = int(np.argmin(m)) if np.any(~m) else len(taus) - 1
        out[j] = taus[window]
    return out


def summary(chains, param_names=None) -> dict[str, dict[str, float]]:
    """Mean, standard deviation, R-hat, ESS and a convergence flag per parameter.

    ``converged`` is ``R-hat < 1.01 and ESS >= 400``.
    """
    c = _as_3d(chains)
    names = list(param_names) if param_names else [f"x{i}" for i in range(c.shape[2])]
    r, e = rhat(c), ess(c)
    flat = c.reshape(-1, c.shape[2])
    return {n: {"mean": float(flat[:, j].mean()), "std": float(flat[:, j].std(ddof=1)),
                "rhat": float(r[j]), "ess": float(e[j]),
                "converged": bool(r[j] < RHAT_THRESHOLD and e[j] >= ESS_MIN)}
            for j, n in enumerate(names)}
