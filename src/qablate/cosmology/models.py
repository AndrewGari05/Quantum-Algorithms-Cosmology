"""Flat background cosmological models.

A model only has to provide ``E^2(z; theta) = H^2(z) / H0^2``; distances and
likelihoods are derived from it. By convention ``theta[0] = Omega_m`` and
``theta[1] = H0`` [km/s/Mpc]. Radiation is fixed at ``Omega_r = 9.4e-5`` and
all models are spatially flat.

``E2`` is vectorized: ``z`` may broadcast against columns of ``theta``
(each ``theta[i]`` may be an ``(n, 1)`` array), so a whole batch of
parameter vectors is evaluated in one call.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

__all__ = ["CosmoModel", "MODELS", "get_model", "C_LIGHT", "OMEGA_R0",
           "PLANCK2018", "SH0ES2022"]

C_LIGHT = 299792.458          #: speed of light [km/s]
OMEGA_R0 = 9.4e-5             #: photons + relativistic neutrinos (fixed)
#: Planck 2018 TT,TE,EE+lowE+lensing (mean, sd); used by the Gaussian prior.
PLANCK2018 = {"Om": (0.3111, 0.0056), "H0": (67.66, 0.42)}
#: Riess et al. (2022), SH0ES.
SH0ES2022 = {"H0": (73.04, 1.04)}


@dataclass(frozen=True)
class CosmoModel:
    """A flat background model.

    Attributes:
        name: Identifier (``'lcdm'``, ``'wcdm'``, ``'cpl'``, ``'pede'``, ``'gede'``).
        label: Human-readable name.
        param_names: Parameter names; the first two are ``('Om', 'H0')``.
        param_latex: LaTeX labels.
        bounds: Flat-prior box per parameter (open interval).
        sample_box: Narrower box used to initialize samplers.
        fiducial: Reference values.
        E2: ``E2(z, theta) -> ndarray``.
        reference: Literature reference for the model.
    """

    name: str
    label: str
    param_names: tuple[str, ...]
    param_latex: tuple[str, ...]
    bounds: tuple[tuple[float, float], ...]
    sample_box: tuple[tuple[float, float], ...]
    fiducial: tuple[float, ...]
    E2: Callable
    reference: str = ""

    @property
    def ndim(self) -> int:
        """Number of free parameters."""
        return len(self.param_names)

    def H(self, z, theta) -> np.ndarray:
        """``H(z)`` [km/s/Mpc]; ``nan`` where ``E^2 <= 0`` (unphysical).

        Examples:
            >>> import numpy as np
            >>> z = np.array([0.5, 1.0, 2.0])
            >>> l = MODELS['lcdm'].H(z, [0.3, 70.0])
            >>> w = MODELS['wcdm'].H(z, [0.3, 70.0, -1.0])
            >>> c = MODELS['cpl'].H(z, [0.3, 70.0, -1.0, 0.0])
            >>> bool(np.array_equal(l, w) and np.array_equal(l, c))
            True
        """
        e2 = self.E2(np.asarray(z), theta)
        return theta[1] * np.sqrt(np.where(e2 > 0.0, e2, np.nan))


def _e2_lcdm(z, th):
    om = th[0]
    zp1 = 1.0 + z
    return om * zp1**3 + OMEGA_R0 * zp1**4 + (1.0 - om - OMEGA_R0)


def _e2_wcdm(z, th):
    om, w = th[0], th[2]
    zp1 = 1.0 + z
    return om * zp1**3 + OMEGA_R0 * zp1**4 + (1.0 - om - OMEGA_R0) * zp1**(3.0 * (1.0 + w))


def _e2_cpl(z, th):
    om, w0, wa = th[0], th[2], th[3]
    zp1 = 1.0 + z
    f_de = zp1**(3.0 * (1.0 + w0 + wa)) * np.exp(-3.0 * wa * z / zp1)
    return om * zp1**3 + OMEGA_R0 * zp1**4 + (1.0 - om - OMEGA_R0) * f_de


def _e2_pede(z, th):
    om = th[0]
    zp1 = 1.0 + z
    f_de = 1.0 - np.tanh(np.log10(zp1))
    return om * zp1**3 + OMEGA_R0 * zp1**4 + (1.0 - om - OMEGA_R0) * f_de


def _e2_gede(z, th):
    # Transition redshift from the LCDM matter-DE equality, (1 + z_t)^3 =
    # (1 - Om) / Om, for every Delta. Li & Shafieloo (2020) define z_t with
    # the GEDE density itself; the difference in H(z) is below 1.8 % inside
    # the prior box. f_DE(0) = 1 exactly.
    om, delta = th[0], th[2]
    zp1 = 1.0 + z
    zt1 = ((1.0 - om) / np.clip(om, 1e-6, None))**(1.0 / 3.0)
    num = 1.0 - np.tanh(delta * np.log10(zp1 / zt1))
    den = 1.0 + np.tanh(delta * np.log10(zt1))
    return om * zp1**3 + OMEGA_R0 * zp1**4 + (1.0 - om - OMEGA_R0) * num / np.clip(den, 1e-12, None)


_OM, _H0 = PLANCK2018["Om"][0], PLANCK2018["H0"][0]
_BASE_B = ((0.18, 0.50), (60.0, 82.0))
_BASE_S = ((0.25, 0.38), (64.0, 76.0))
_LAT = (r"$\Omega_m$", r"$H_0$")

MODELS: dict[str, CosmoModel] = {m.name: m for m in (
    CosmoModel("lcdm", "Flat LCDM", ("Om", "H0"), _LAT, _BASE_B, _BASE_S, (_OM, _H0), _e2_lcdm),
    CosmoModel("wcdm", "wCDM (constant w)", ("Om", "H0", "w"), _LAT + (r"$w$",),
               _BASE_B + ((-2.0, -0.3),), _BASE_S + ((-1.4, -0.6),), (_OM, _H0, -1.0), _e2_wcdm),
    CosmoModel("cpl", "CPL w(z) = w0 + wa z/(1+z)", ("Om", "H0", "w0", "wa"),
               _LAT + (r"$w_0$", r"$w_a$"), _BASE_B + ((-2.0, -0.3), (-3.0, 2.0)),
               _BASE_S + ((-1.4, -0.6), (-1.5, 1.0)), (_OM, _H0, -1.0, 0.0), _e2_cpl,
               "Chevallier & Polarski 2001; Linder 2003"),
    CosmoModel("pede", "PEDE", ("Om", "H0"), _LAT, _BASE_B, ((0.25, 0.38), (64.0, 78.0)),
               (_OM, _H0), _e2_pede, "Li & Shafieloo 2019"),
    CosmoModel("gede", "GEDE", ("Om", "H0", "Delta"), _LAT + (r"$\Delta$",),
               _BASE_B + ((-3.0, 6.0),), _BASE_S + ((-1.0, 3.0),), (_OM, _H0, 1.0), _e2_gede,
               "Li & Shafieloo 2020 (LCDM transition redshift)"),
)}


def get_model(name) -> CosmoModel:
    """Model by name (or the model itself)."""
    if isinstance(name, CosmoModel):
        return name
    try:
        return MODELS[str(name).lower()]
    except KeyError:
        raise ValueError(f"unknown model {name!r}; available: {sorted(MODELS)}") from None


def param_names(model) -> list[str]:
    """Parameter names of a model."""
    return list(get_model(model).param_names)
