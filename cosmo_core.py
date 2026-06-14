# =============================================================================
#  cosmo_core.py — Shared physics, data and statistics
# =============================================================================
#
#  Core module of the quantum-classical cosmological inference pipeline.
#  ALL the physics lives here, strictly separated from the sampling logic
#  (MCMC/QVMC), which lives in `cosmo_modular_quantum.py` and
#  `qpu_cosmo_samplers.py`.
#
#  Contents:
#    1. Modular registry of cosmological models (ΛCDM, wCDM, CPL, PEDE, GEDE)
#    2. Observational data (Cosmic Chronometers + Pantheon+ SNe Ia)
#    3. Likelihoods and N-dimensional posterior (any number of parameters)
#    4. Statistical estimators: χ², reduced χ², AIC, BIC, ESS, Gelman-Rubin, τ
#    5. Logging utilities for the non-interactive CLI mode
#
#  [KEY CHANGE vs previous version]
#  The Pantheon+ likelihood NO LONGER uses a precomputed lookup grid in Ωm
#  (which was only valid for flat ΛCDM). The comoving distance is now
#  computed with a vectorized cumulative trapezoid over a fine z grid,
#  valid for ANY model E²(z; θ). This is what enables CPL/wCDM without
#  touching the sampling code, and is also ~10× faster than the original
#  point-by-point scipy.integrate.quad.
# =============================================================================

from __future__ import annotations

import logging
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import minimize

# ── Physical constants and Planck 2018 values ────────────────────────────────
C_LIGHT  = 299792.458        # km/s
OMEGA_R0 = 9.4e-5            # radiation density today (fixed)

H0_MU, H0_SIG = 67.66, 0.42      # Gaussian prior on H0  (Planck 2018, TT,TE,EE+lowE+lensing)
OM_MU, OM_SIG = 0.3111, 0.0056   # Gaussian prior on Ωm  (Planck 2018)

RNG = np.random.default_rng(42)


# =============================================================================
# 1. MODULAR REGISTRY OF COSMOLOGICAL MODELS
# =============================================================================

@dataclass
class CosmoModel:
    """Complete specification of a background cosmological model.

    The only physics a model needs to expose is E²(z; θ), the square of
    the dimensionless Hubble function. Everything else (H(z), d_L(z),
    likelihoods, χ²) is derived generically.

    Attributes:
        name: Short identifier used in the CLI and file names.
        label: Human-readable name for figure titles.
        param_names: ASCII names of the free parameters, in order.
            By convention the first two are always ('Om', 'H0').
        param_latex: LaTeX labels for figure axes.
        bounds: Flat-prior box (lo, hi) per parameter. Outside the box
            the log-posterior is -inf.
        sample_box: Narrower box used to initialize chains and to build
            the discrete grid for QVMC/VI.
        fiducial: Reference values (Planck or paper) for guide lines.
        E2: Vectorized function E²(z, θ) -> ndarray. Must be > 0 in the
            physical domain; unphysical values may return <= 0 and the
            likelihood penalizes them automatically.
    """
    name: str
    label: str
    param_names: List[str]
    param_latex: List[str]
    bounds: List[Tuple[float, float]]
    sample_box: List[Tuple[float, float]]
    fiducial: List[float]
    E2: Callable[[np.ndarray, np.ndarray], np.ndarray]

    @property
    def n_params(self) -> int:
        """Number of free parameters of the model."""
        return len(self.param_names)

    def H(self, z: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """H(z) in km/s/Mpc. θ[1] is always H0 by convention."""
        e2 = self.E2(np.asarray(z), theta)
        return theta[1] * np.sqrt(np.clip(e2, 1e-12, None))


def _E2_lcdm(z, th):
    """Flat ΛCDM: E² = Ωm(1+z)³ + Ωr(1+z)⁴ + ΩΛ."""
    Om = th[0]
    OL = 1.0 - Om - OMEGA_R0
    zp1 = 1.0 + z
    return Om * zp1**3 + OMEGA_R0 * zp1**4 + OL


def _E2_wcdm(z, th):
    """Flat wCDM: dark energy with constant equation of state w.

    E² = Ωm(1+z)³ + Ωr(1+z)⁴ + Ω_DE (1+z)^{3(1+w)}
    """
    Om, _, w = th[0], th[1], th[2]
    ODE = 1.0 - Om - OMEGA_R0
    zp1 = 1.0 + z
    return Om * zp1**3 + OMEGA_R0 * zp1**4 + ODE * zp1**(3.0 * (1.0 + w))


def _E2_cpl(z, th):
    """CPL (Chevallier-Polarski-Linder): w(z) = w0 + wa·z/(1+z).

    Integrating the continuity equation:
      ρ_DE/ρ_DE0 = (1+z)^{3(1+w0+wa)} · exp(-3 wa z/(1+z))
    """
    Om, _, w0, wa = th[0], th[1], th[2], th[3]
    ODE = 1.0 - Om - OMEGA_R0
    zp1 = 1.0 + z
    f_de = zp1**(3.0 * (1.0 + w0 + wa)) * np.exp(-3.0 * wa * z / zp1)
    return Om * zp1**3 + OMEGA_R0 * zp1**4 + ODE * f_de


def _E2_pede(z, th):
    """PEDE (Phenomenologically Emergent Dark Energy, Li & Shafieloo 2019).

    f_DE(z) = 1 - tanh(log10(1+z));  f_DE(0) = 1 by construction.
    No extra parameters: same (Ωm, H0) space as ΛCDM.
    """
    Om = th[0]
    ODE = 1.0 - Om - OMEGA_R0
    zp1 = 1.0 + z
    f_de = 1.0 - np.tanh(np.log10(zp1))
    return Om * zp1**3 + OMEGA_R0 * zp1**4 + ODE * f_de


def _E2_gede(z, th):
    """GEDE (Generalized Emergent Dark Energy, Li & Shafieloo 2020).

    f_DE(z; Δ) = [1 - tanh(Δ·log10((1+z)/(1+z_t)))] /
                 [1 + tanh(Δ·log10(1+z_t))]
    with z_t the matter-DE equality epoch: (1+z_t) = ((1-Ωm)/Ωm)^{1/3}.
    Δ=1 with z_t→0 recovers PEDE; Δ=0 recovers ΛCDM.
    """
    Om, _, Delta = th[0], th[1], th[2]
    ODE = 1.0 - Om - OMEGA_R0
    zp1 = 1.0 + z
    zt1 = ((1.0 - Om) / np.clip(Om, 1e-6, None))**(1.0 / 3.0)
    num = 1.0 - np.tanh(Delta * np.log10(zp1 / zt1))
    den = 1.0 + np.tanh(Delta * np.log10(zt1))
    return Om * zp1**3 + OMEGA_R0 * zp1**4 + ODE * num / np.clip(den, 1e-12, None)


#: Global model registry. To inject a new model (e.g. VC):
#:   MODELS['vc'] = CosmoModel(name='vc', ..., E2=_E2_vc)
#: and the whole pipeline (MCMC, QVMC, plots, statistics) supports it
#: without further changes.
MODELS: Dict[str, CosmoModel] = {
    'lcdm': CosmoModel(
        name='lcdm', label='Flat ΛCDM',
        param_names=['Om', 'H0'],
        param_latex=[r'$\Omega_m$', r'$H_0$ [km/s/Mpc]'],
        bounds=[(0.18, 0.50), (60.0, 82.0)],
        sample_box=[(0.25, 0.38), (64.0, 76.0)],
        fiducial=[OM_MU, H0_MU],
        E2=_E2_lcdm,
    ),
    'wcdm': CosmoModel(
        name='wcdm', label='wCDM (constant w)',
        param_names=['Om', 'H0', 'w'],
        param_latex=[r'$\Omega_m$', r'$H_0$ [km/s/Mpc]', r'$w$'],
        bounds=[(0.18, 0.50), (60.0, 82.0), (-2.0, -0.3)],
        sample_box=[(0.25, 0.38), (64.0, 76.0), (-1.4, -0.6)],
        fiducial=[OM_MU, H0_MU, -1.0],
        E2=_E2_wcdm,
    ),
    'cpl': CosmoModel(
        name='cpl', label='CPL  w(z)=w0+wa·z/(1+z)',
        param_names=['Om', 'H0', 'w0', 'wa'],
        param_latex=[r'$\Omega_m$', r'$H_0$ [km/s/Mpc]', r'$w_0$', r'$w_a$'],
        bounds=[(0.18, 0.50), (60.0, 82.0), (-2.0, -0.3), (-3.0, 2.0)],
        sample_box=[(0.25, 0.38), (64.0, 76.0), (-1.4, -0.6), (-1.5, 1.0)],
        fiducial=[OM_MU, H0_MU, -1.0, 0.0],
        E2=_E2_cpl,
    ),
    'pede': CosmoModel(
        name='pede', label='PEDE',
        param_names=['Om', 'H0'],
        param_latex=[r'$\Omega_m$', r'$H_0$ [km/s/Mpc]'],
        bounds=[(0.18, 0.50), (60.0, 82.0)],
        sample_box=[(0.25, 0.38), (64.0, 78.0)],
        fiducial=[OM_MU, H0_MU],
        E2=_E2_pede,
    ),
    'gede': CosmoModel(
        name='gede', label='GEDE',
        param_names=['Om', 'H0', 'Delta'],
        param_latex=[r'$\Omega_m$', r'$H_0$ [km/s/Mpc]', r'$\Delta$'],
        bounds=[(0.18, 0.50), (60.0, 82.0), (-3.0, 6.0)],
        sample_box=[(0.25, 0.38), (64.0, 76.0), (-1.0, 3.0)],
        fiducial=[OM_MU, H0_MU, 1.0],
        E2=_E2_gede,
    ),
}


# =============================================================================
# 2. OBSERVATIONAL DATA
# =============================================================================

# 51 Cosmic Chronometer points (standard compilation, H(z) in km/s/Mpc)
_CC_EMBEDDED = np.array([
    [0.07,  69.00, 19.60], [0.10,  69.00, 12.00], [0.12,  68.60, 26.20],
    [0.17,  83.00,  8.00], [0.1791, 75.00, 4.00], [0.1993, 75.00, 5.00],
    [0.20,  72.90, 29.60], [0.240, 79.69,  2.65], [0.27,  77.00, 14.00],
    [0.28,  88.80, 36.60], [0.300, 81.70,  6.22], [0.31,  78.17,  4.74],
    [0.350, 82.70,  8.40], [0.3519, 83.00, 14.00], [0.36,  79.93,  3.39],
    [0.38,  81.50,  1.90], [0.3802, 83.00, 13.50], [0.40,  95.00, 17.00],
    [0.4004, 77.00, 10.20], [0.4247, 87.10, 11.20], [0.43,  86.45,  3.68],
    [0.44,  82.60,  7.80], [0.4497, 92.80, 12.90], [0.47,  89.00, 34.00],
    [0.4783, 80.90,  9.00], [0.48,  97.00, 62.00], [0.51,  90.40,  1.90],
    [0.52,  94.35,  2.65], [0.56,  93.33,  2.32], [0.570, 92.90,  7.85],
    [0.59,  98.48,  3.19], [0.5929, 104.00, 13.00], [0.60,  87.90,  6.10],
    [0.61,  97.30,  2.10], [0.64,  98.82,  2.99], [0.6797, 92.00,  8.00],
    [0.73,  97.30,  7.00], [0.7812, 105.00, 12.00], [0.8754, 125.00, 17.00],
    [0.88,  90.00, 40.00], [0.90,  117.00, 23.00], [1.037, 154.00, 20.00],
    [1.30,  168.00, 17.00], [1.363, 160.00, 33.60], [1.43,  177.00, 18.00],
    [1.53,  140.00, 14.00], [1.75,  202.00, 40.00], [1.965, 186.50, 50.40],
    [2.33,  224.00,  8.00], [2.34,  222.00,  7.00], [2.360, 226.00,  8.00],
])


def load_cc(path: str = "cosmic_chronometers.txt") -> np.ndarray:
    """Load CC data from file (z, H, sigma); fallback to the embedded array.

    Args:
        path: Path to the three-column text file.

    Returns:
        ndarray of shape (N, 3): columns (z, H_obs, sigma).
    """
    for d in (os.path.dirname(os.path.abspath(__file__)), os.getcwd()):
        p = os.path.join(d, path)
        if os.path.exists(p):
            try:
                arr = np.loadtxt(p, comments='#')
                if arr.ndim == 2 and arr.shape[1] >= 3:
                    print(f"  ✓ CC loaded from file: {p}  ({len(arr)} pts)")
                    return arr[:, :3]
            except Exception as e:
                print(f"  ⚠  Error reading {p}: {e} — using embedded data")
    return _CC_EMBEDDED


def load_pantheon(search_dirs: Optional[Sequence[str]] = None) -> Optional[dict]:
    """Load the Pantheon+ catalog (Scolnic et al. 2022 / Brout et al. 2022).

    Searches for `pantheon_full_parameters.txt` (and variants) in the
    script folder and the cwd. Expected format: name zcmb zhel dz mb dmb.

    Returns:
        dict with arrays 'z', 'mb', 'dmb' sorted by z, or None if the
        file is not found.
    """
    names = ["pantheon_full_parameters.txt", "Pantheon_full_parameters.txt",
             "pantheon_plus.txt", "pantheon_plus.csv",
             "PantheonPlus.txt", "PantheonPlus.csv"]
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        script_dir = os.getcwd()
    dirs = [script_dir, os.getcwd()] + list(search_dirs or [])

    for d in dirs:
        for name in names:
            path = os.path.join(d, name)
            if not os.path.exists(path):
                continue
            try:
                raw = np.genfromtxt(path, comments='#', dtype=None, encoding='utf-8')
                if raw.ndim == 1:
                    raw = raw.reshape(1, -1)
                if raw.dtype.names:
                    z = raw['f1'].astype(float)
                    mb = raw['f4'].astype(float)
                    dmb = raw['f5'].astype(float)
                else:
                    z = raw[:, 1].astype(float)
                    mb = raw[:, 4].astype(float)
                    dmb = raw[:, 5].astype(float)
                mask = (z > 0) & (dmb > 0) & np.isfinite(mb)
                z, mb, dmb = z[mask], mb[mask], dmb[mask]
                idx = np.argsort(z)
                print(f"  ✓ Pantheon+ loaded: {path}  ({len(z)} SNe Ia, "
                      f"z ∈ [{z[idx[0]]:.3f}, {z[idx[-1]]:.3f}])")
                return {'z': z[idx], 'mb': mb[idx], 'dmb': dmb[idx], 'path': path}
            except Exception as e:
                print(f"  ⚠  Error reading {path}: {e}")
    return None


# =============================================================================
# 3. LIKELIHOODS AND POSTERIOR (model-agnostic, N-dimensional)
# =============================================================================

class Posterior:
    """N-dimensional log-posterior tying together model + datasets + prior.

    This class is the ONLY contact point between the physics and the
    samplers: any sampler receives an instance and evaluates it as
    `post.log_prob(theta)`. That guarantees the strict physics ↔
    inference separation required by the design.

    Args:
        model: CosmoModel instance (from the MODELS registry).
        dataset: 'CC', 'Pantheon+' or 'CC+Pantheon+'.
        prior_type: 'flat' (box) or 'gaussian' (Planck on Ωm and H0,
            flat on the extra parameters within bounds).
        cc_data: (N,3) array of Cosmic Chronometers.
        pantheon: dict from load_pantheon() or None.
        n_zgrid: Resolution of the z grid for the SNe comoving-distance
            integral (cumulative trapezoid).
    """

    def __init__(self, model: CosmoModel, dataset: str = 'CC',
                 prior_type: str = 'flat',
                 cc_data: Optional[np.ndarray] = None,
                 pantheon: Optional[dict] = None,
                 n_zgrid: int = 1200):
        self.model = model
        self.dataset = dataset
        self.prior_type = prior_type

        self.cc = cc_data if cc_data is not None else load_cc()
        self.z_cc, self.H_cc, self.sig_cc = self.cc[:, 0], self.cc[:, 1], self.cc[:, 2]

        self.pantheon = pantheon
        if 'Pantheon' in dataset and pantheon is None:
            self.pantheon = load_pantheon()
            if self.pantheon is None:
                raise FileNotFoundError(
                    "Dataset includes Pantheon+ but "
                    "pantheon_full_parameters.txt was not found")

        # Fine z grid for the cumulative trapezoid of d_C(z).
        # [KEY CHANGE] replaces the precomputed Ωm lookup grid (ΛCDM-only)
        # with an on-the-fly computation valid for any E²(z;θ).
        if self.pantheon is not None:
            zmax = float(self.pantheon['z'].max()) * 1.02
            self._zg = np.linspace(0.0, zmax, n_zgrid)
            self._inv_s2 = 1.0 / self.pantheon['dmb']**2
            self._C_marg = float(np.sum(self._inv_s2))

    # ── total number of data points (for reduced χ² and BIC) ────────────────
    @property
    def n_data(self) -> int:
        """Total number of observational points of the active dataset."""
        n = 0
        if self.dataset in ('CC', 'CC+Pantheon+'):
            n += len(self.z_cc)
        if self.dataset in ('Pantheon+', 'CC+Pantheon+') and self.pantheon:
            n += len(self.pantheon['z'])
        return n

    # ── χ² components ────────────────────────────────────────────────────────
    def chi2_cc(self, theta: np.ndarray) -> float:
        """Cosmic Chronometers χ² for the active model."""
        Hm = self.model.H(self.z_cc, theta)
        return float(np.sum(((self.H_cc - Hm) / self.sig_cc)**2))

    def chi2_pantheon(self, theta: np.ndarray) -> float:
        """Pantheon+ effective χ² with analytic marginalization over M_abs.

        Implements χ²_eff = A − B²/C (Goliath et al. 2001):
            A = Σ Δ²/σ²,  B = Σ Δ/σ²,  C = Σ 1/σ²,  Δ = μ_obs − μ_th.
        The comoving distance is obtained with a vectorized cumulative
        trapezoid over self._zg, valid for any registered model.
        """
        e2 = self.model.E2(self._zg, theta)
        if np.any(e2 <= 0):           # unphysical region (e.g. extreme CPL)
            return np.inf
        invE = 1.0 / np.sqrt(e2)
        I = cumulative_trapezoid(invE, self._zg, initial=0.0)
        z_sn = self.pantheon['z']
        I_sn = np.interp(z_sn, self._zg, I)
        dL = (C_LIGHT / theta[1]) * (1.0 + z_sn) * I_sn          # Mpc
        mu_th = 5.0 * np.log10(np.clip(dL, 1e-10, None)) + 25.0
        delta = self.pantheon['mb'] - mu_th                       # μ_obs − μ_th + M_abs
        A = float(np.sum(delta**2 * self._inv_s2))
        B = float(np.sum(delta * self._inv_s2))
        return A - B**2 / self._C_marg

    def chi2(self, theta: np.ndarray) -> Tuple[float, int]:
        """Total χ² of the active dataset. Returns (chi2, n_data)."""
        c = 0.0
        if self.dataset in ('CC', 'CC+Pantheon+'):
            c += self.chi2_cc(theta)
        if self.dataset in ('Pantheon+', 'CC+Pantheon+') and self.pantheon:
            c += self.chi2_pantheon(theta)
        return c, self.n_data

    # ── prior and posterior ──────────────────────────────────────────────────
    def log_prior(self, theta: np.ndarray) -> float:
        """Log-prior: hard box + optional Planck Gaussian on (Ωm, H0)."""
        for v, (lo, hi) in zip(theta, self.model.bounds):
            if not (lo < v < hi):
                return -np.inf
        if self.prior_type == 'gaussian':
            return (-0.5 * ((theta[0] - OM_MU) / OM_SIG)**2
                    - 0.5 * ((theta[1] - H0_MU) / H0_SIG)**2)
        return 0.0

    def log_prob(self, theta: np.ndarray) -> float:
        """(Unnormalized) log-posterior at θ."""
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        c, _ = self.chi2(theta)
        if not np.isfinite(c):
            return -np.inf
        return lp - 0.5 * c

    def __call__(self, theta: np.ndarray) -> float:
        return self.log_prob(theta)

    # ── batched (vectorized) evaluation ──────────────────────────────────────
    def log_prob_batch(self, thetas: np.ndarray) -> np.ndarray:
        """Log-posterior for a parameter batch, shape (B, d) -> (B,).

        [OPT] Used by QVMC.build_target(): evaluates the full grid
        (up to 2^{n_qubits} states) in ONE vectorized pass instead of a
        Python loop. For CPL with nqpp=3 (4096 states) this reduces the
        target construction from ~1 s to ~30 ms.

        It works for any model because the E²(z;θ) functions are written
        with elementary broadcasting operations: θ_i components with
        shape (B,1) are passed against z with shape (Nz,).
        """
        thetas = np.atleast_2d(np.asarray(thetas, dtype=float))
        B = len(thetas)
        out = np.full(B, -np.inf)

        # 1) box mask
        ok = np.ones(B, dtype=bool)
        for j, (lo, hi) in enumerate(self.model.bounds):
            ok &= (thetas[:, j] > lo) & (thetas[:, j] < hi)
        if not np.any(ok):
            return out
        T = thetas[ok]                                    # (Bv, d)
        th_cols = [T[:, j:j + 1] for j in range(T.shape[1])]  # each (Bv,1)

        lp = np.zeros(len(T))
        if self.prior_type == 'gaussian':
            lp += (-0.5 * ((T[:, 0] - OM_MU) / OM_SIG)**2
                   - 0.5 * ((T[:, 1] - H0_MU) / H0_SIG)**2)

        # 2) vectorized CC: E2 with broadcasting (Bv, Ncc)
        if self.dataset in ('CC', 'CC+Pantheon+'):
            e2 = self.model.E2(self.z_cc[None, :], th_cols)
            Hm = T[:, 1:2] * np.sqrt(np.clip(e2, 1e-12, None))
            lp += -0.5 * np.sum(((self.H_cc[None, :] - Hm)
                                 / self.sig_cc[None, :])**2, axis=1)

        # 3) vectorized Pantheon+: row-wise cumulative trapezoid (Bv, Nzg)
        if self.dataset in ('Pantheon+', 'CC+Pantheon+') and self.pantheon:
            e2 = self.model.E2(self._zg[None, :], th_cols)
            bad = np.any(e2 <= 0, axis=1)
            e2 = np.clip(e2, 1e-12, None)
            I = cumulative_trapezoid(1.0 / np.sqrt(e2), self._zg,
                                     axis=1, initial=0.0)
            z_sn = self.pantheon['z']
            # row-wise interpolation
            idx = np.searchsorted(self._zg, z_sn).clip(1, len(self._zg) - 1)
            z0, z1 = self._zg[idx - 1], self._zg[idx]
            w = (z_sn - z0) / (z1 - z0)
            I_sn = I[:, idx - 1] * (1 - w)[None, :] + I[:, idx] * w[None, :]
            dL = (C_LIGHT / T[:, 1:2]) * (1.0 + z_sn)[None, :] * I_sn
            mu_th = 5.0 * np.log10(np.clip(dL, 1e-10, None)) + 25.0
            delta = self.pantheon['mb'][None, :] - mu_th
            A = np.sum(delta**2 * self._inv_s2[None, :], axis=1)
            Bm = np.sum(delta * self._inv_s2[None, :], axis=1)
            chi2p = A - Bm**2 / self._C_marg
            chi2p[bad] = np.inf
            lp += -0.5 * chi2p

        out[ok] = lp
        return out


# =============================================================================
# 4. STATISTICAL AND MODEL-SELECTION ESTIMATORS
# =============================================================================

def autocorr_time_fft(x: np.ndarray) -> float:
    """Integrated autocorrelation time τ via FFT, O(N log N).

    For N = 160 000 samples this is ~250× faster than np.correlate
    (mode='full'), avoiding the hang when building the results table.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 5:
        return 1.0
    x = x - x.mean()
    f = np.fft.rfft(x, n=2 * n)
    acf = np.fft.irfft(f * np.conj(f))[:n].real
    if acf[0] <= 0:
        return 1.0
    acf /= acf[0]
    w = int(np.argmax(acf < 0.05))
    if w == 0:
        w = n // 4
    return float(max(1.0, 1 + 2 * np.sum(acf[1:max(w, 2)])))


def ess_chains(chains: np.ndarray) -> float:
    """Effective Sample Size for MCMC chains, shape (M, N, d).

    ESS = M·N / τ_max, with τ_max the worst τ across parameters
    (conservative).
    """
    M, N, d = chains.shape
    taus = []
    for p in range(d):
        taus.append(np.mean([autocorr_time_fft(chains[c, :, p]) for c in range(M)]))
    return float(M * N / max(taus))


def ess_weights(w: np.ndarray) -> float:
    """Kish ESS for weighted samples (QVMC/VI): (Σw)²/Σw²."""
    w = np.asarray(w, dtype=float)
    s = w.sum()
    return float(s * s / np.sum(w * w)) if s > 0 else 0.0


def gelman_rubin(chains: np.ndarray) -> float:
    """Gelman-Rubin R̂ statistic for one parameter, shape (M, N)."""
    M, N = chains.shape
    mu_j = chains.mean(axis=1)
    B = N * np.var(mu_j, ddof=1)
    W = np.mean(np.var(chains, axis=1, ddof=1))
    var_hat = (1 - 1 / N) * W + B / N
    return float(np.sqrt(var_hat / W)) if W > 1e-12 else np.nan


def gelman_rubin_max(chains: np.ndarray) -> float:
    """Maximum R̂ over all parameters, shape (M, N, d)."""
    return max(gelman_rubin(chains[:, :, p]) for p in range(chains.shape[2]))


def fit_statistics(post: Posterior, theta_mean: np.ndarray,
                   refine: bool = True) -> dict:
    """Compute χ², reduced χ², AIC and BIC at the best fit.

    Starts from the posterior mean and optionally refines with
    Nelder-Mead to find the χ² minimum (true best fit).

    Args:
        post: Active posterior.
        theta_mean: Posterior mean estimated by the sampler.
        refine: If True, locally minimize χ² starting from theta_mean.

    Returns:
        dict with theta_best, chi2, chi2_red, AIC, BIC, k, n_data.
    """
    theta_best = np.asarray(theta_mean, dtype=float).copy()
    if refine:
        res = minimize(lambda t: post.chi2(t)[0], theta_best,
                       method='Nelder-Mead',
                       options={'maxiter': 800, 'xatol': 1e-6, 'fatol': 1e-6})
        if np.isfinite(res.fun):
            theta_best = res.x
    chi2, n = post.chi2(theta_best)
    k = post.model.n_params
    dof = max(n - k, 1)
    return {
        'theta_best': theta_best,
        'chi2': chi2,
        'chi2_red': chi2 / dof,
        'AIC': chi2 + 2 * k,
        'BIC': chi2 + k * np.log(n),
        'k': k,
        'n_data': n,
    }


# =============================================================================
# 5. LOGGING (non-interactive CLI mode)
# =============================================================================

def setup_logger(log_file: Optional[str] = None,
                 name: str = "qcosmo") -> logging.Logger:
    """Configure a dual logger: detailed file + minimal console.

    In interactive mode (log_file=None) everything goes to the console.
    In CLI mode, the detail goes to the file and the console only
    receives WARNING+.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s",
                            datefmt="%H:%M:%S")
    if log_file:
        fh = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.WARNING)
        ch.setFormatter(fmt)
        logger.addHandler(ch)
        logger.info(f"Log started: {log_file}")
    else:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(ch)
    return logger


def fmt_theta(model: CosmoModel, theta: np.ndarray) -> str:
    """Format θ with parameter names for readable logs."""
    return "  ".join(f"{n}={v:.4f}" for n, v in zip(model.param_names, theta))
