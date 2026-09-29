"""Posterior of a background model given H(z) and supernova data.

``log p(theta) = log prior(theta) - chi2(theta) / 2`` with

* H(z): ``chi2 = sum(((H_obs - H(z)) / sigma)^2)``;
* supernovae: the absolute magnitude is marginalized analytically
  (Goliath et al. 2001), ``chi2 = A - B^2 / C`` with
  ``A = d' C^-1 d``, ``B = d' C^-1 1``, ``C = 1' C^-1 1``,
  ``d = m_obs - mu(z)``. The luminosity distance uses the cumulative
  trapezoid rule on a 1200-point redshift grid (error below 1e-4 mag).

The flat prior is the open box ``model.bounds``; the Gaussian prior adds the
Planck 2018 constraints on ``Omega_m`` and ``H0``.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import cumulative_trapezoid

from . import data as _data
from .models import C_LIGHT, PLANCK2018, CosmoModel, get_model

__all__ = ["Posterior", "fit_statistics"]


class Posterior:
    """Vectorized log-posterior of a model on one or more datasets.

    Args:
        model: Model name or :class:`~qablate.cosmology.models.CosmoModel`.
        datasets: Dataset spec, e.g. ``"CC+BAO"`` or ``"CC+BAO+Pantheon"``.
        prior: ``'flat'`` or ``'gaussian'``.
        n_zgrid: Redshift grid size for the distance integral.

    Examples:
        >>> post = Posterior("lcdm", "CC+BAO")
        >>> post.ndim, post.n_data
        (2, 51)
        >>> post.log_prob([[0.3, 70.0], [0.1, 70.0]]).shape
        (2,)
    """

    def __init__(self, model, datasets="CC+BAO", prior: str = "flat", n_zgrid: int = 1200):
        if prior not in ("flat", "gaussian"):
            raise ValueError("prior must be 'flat' or 'gaussian'")
        self.model: CosmoModel = get_model(model)
        self.components = _data.resolve(datasets)
        self.dataset = "+".join(self.components)
        self.prior = prior
        self.hz, self.sn = _data.load(self.components)
        self._lo = np.array([b[0] for b in self.model.bounds])
        self._hi = np.array([b[1] for b in self.model.bounds])
        if self.sn is not None:
            self._zg = np.linspace(0.0, float(self.sn.z.max()) * 1.02, n_zgrid)
            if self.sn.cov is None:
                self._w = 1.0 / self.sn.dmb ** 2
                self._cw = float(np.sum(self._w))
            else:
                self._cinv = np.linalg.inv(self.sn.cov)
                self._cinv1 = self._cinv @ np.ones(len(self.sn))
                self._c11 = float(np.sum(self._cinv1))

    @property
    def ndim(self) -> int:
        """Number of parameters."""
        return self.model.ndim

    @property
    def n_data(self) -> int:
        """Number of data points."""
        return (len(self.hz) if self.hz is not None else 0) + (len(self.sn) if self.sn is not None else 0)

    # ------------------------------------------------------------------ #
    def chi2_batch(self, thetas) -> np.ndarray:
        """``chi2`` for each row of ``thetas``; ``inf`` where unphysical."""
        t = np.atleast_2d(np.asarray(thetas, dtype=float))
        cols = [t[:, j:j + 1] for j in range(t.shape[1])]
        chi2 = np.zeros(len(t))
        if self.hz is not None:
            e2 = self.model.E2(self.hz.z[None, :], cols)
            bad = np.any(~np.isfinite(e2) | (e2 <= 0.0), axis=1)
            h = t[:, 1:2] * np.sqrt(np.where(e2 > 0.0, e2, 1.0))
            c = np.sum(((self.hz.H[None, :] - h) / self.hz.sigma[None, :]) ** 2, axis=1)
            chi2 += np.where(bad, np.inf, c)
        if self.sn is not None:
            e2 = self.model.E2(self._zg[None, :], cols)
            bad = np.any(~np.isfinite(e2) | (e2 <= 0.0), axis=1)
            integral = cumulative_trapezoid(1.0 / np.sqrt(np.where(e2 > 0.0, e2, 1.0)),
                                            self._zg, axis=1, initial=0.0)
            i_sn = np.array([np.interp(self.sn.z, self._zg, row) for row in integral])
            dl = (C_LIGHT / t[:, 1:2]) * (1.0 + self.sn.z)[None, :] * i_sn
            mu = 5.0 * np.log10(np.clip(dl, 1e-10, None)) + 25.0
            d = self.sn.mb[None, :] - mu
            if self.sn.cov is None:
                a = np.sum(d ** 2 * self._w[None, :], axis=1)
                b = np.sum(d * self._w[None, :], axis=1)
                c = a - b ** 2 / self._cw
            else:
                cd = d @ self._cinv
                c = np.sum(cd * d, axis=1) - (d @ self._cinv1) ** 2 / self._c11
            chi2 += np.where(bad, np.inf, c)
        return chi2

    def chi2(self, theta) -> float:
        """``chi2`` at one parameter vector."""
        return float(self.chi2_batch(np.asarray(theta, dtype=float)[None, :])[0])

    def log_prior(self, thetas) -> np.ndarray:
        """Log prior for each row (``-inf`` outside the open box)."""
        t = np.atleast_2d(np.asarray(thetas, dtype=float))
        inside = np.all((t > self._lo) & (t < self._hi), axis=1)
        lp = np.zeros(len(t))
        if self.prior == "gaussian":
            (om, som), (h0, sh0) = PLANCK2018["Om"], PLANCK2018["H0"]
            lp += -0.5 * ((t[:, 0] - om) / som) ** 2 - 0.5 * ((t[:, 1] - h0) / sh0) ** 2
        return np.where(inside, lp, -np.inf)

    def log_prob(self, thetas) -> np.ndarray:
        """Unnormalized log posterior for each row of ``thetas`` (shape ``(n,)``)."""
        t = np.atleast_2d(np.asarray(thetas, dtype=float))
        lp = self.log_prior(t)
        out = np.full(len(t), -np.inf)
        ok = np.isfinite(lp)
        if np.any(ok):
            c = self.chi2_batch(t[ok])
            out[ok] = np.where(np.isfinite(c), lp[ok] - 0.5 * c, -np.inf)
        return out

    __call__ = log_prob


def fit_statistics(post: Posterior, theta0, refine: bool = True) -> dict[str, object]:
    """Best fit inside the prior box and chi2, reduced chi2, AIC and BIC.

    The search starts at ``theta0`` and never returns a worse point: L-BFGS-B
    in box-normalized coordinates, then a bounded Nelder-Mead polish. An
    optimum on the box edge is moved inside by 1e-9 of the box width.

    Returns:
        ``theta_best``, ``chi2``, ``chi2_red``, ``AIC``, ``BIC``, ``k``, ``n_data``.

    Examples:
        >>> post = Posterior("lcdm", "CC+BAO")
        >>> st = fit_statistics(post, [0.3, 70.0])
        >>> round(st["chi2"], 4), st["k"], st["n_data"]
        (27.4691, 2, 51)
    """
    from scipy.optimize import minimize
    theta = np.asarray(theta0, dtype=float).copy()
    if refine:
        lo, hi = post._lo, post._hi
        width = hi - lo
        eps = 1e-9 * width

        def to_theta(u):
            return np.clip(lo + np.asarray(u, dtype=float) * width, lo + eps, hi - eps)

        def f(u):
            c = post.chi2(to_theta(u))
            return c if np.isfinite(c) else 1e300

        u0 = np.clip((theta - lo) / width, 0.0, 1.0)
        unit = [(0.0, 1.0)] * len(u0)
        best_u, best_f = u0, f(u0)
        r1 = minimize(f, u0, method="L-BFGS-B", bounds=unit,
                      options={"maxiter": 2000, "ftol": 1e-14, "gtol": 1e-10})
        if np.isfinite(r1.fun) and r1.fun < best_f:
            best_u, best_f = r1.x, r1.fun
        r2 = minimize(f, best_u, method="Nelder-Mead", bounds=unit,
                      options={"maxiter": 4000, "xatol": 1e-10, "fatol": 1e-10})
        if np.isfinite(r2.fun) and r2.fun < best_f:
            best_u = r2.x
        cand = to_theta(best_u)
        if np.isfinite(post.log_prior(cand)[0]):
            theta = cand
    chi2 = post.chi2(theta)
    n, k = post.n_data, post.ndim
    return {"theta_best": theta, "chi2": chi2, "chi2_red": chi2 / max(n - k, 1),
            "AIC": chi2 + 2 * k, "BIC": chi2 + k * np.log(n), "k": k, "n_data": n}
