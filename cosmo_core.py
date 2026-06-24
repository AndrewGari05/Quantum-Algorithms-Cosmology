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
# [P2] Radiation density today, FIXED. This is the photon term only; the
# relativistic-neutrino contribution (a factor 1 + 7/8·(4/11)^{4/3}·N_eff ≈
# 1.68 for N_eff = 3.046) is NOT included. Over the redshift range probed here
# (CC+BAO to z≈2.3, SNe to z≈2.3) radiation is sub-percent in E², so this is a
# deliberate, declared approximation rather than an omission; for high-z
# (CMB-distance) extensions Ωr must be upgraded to include neutrinos.
OMEGA_R0 = 9.4e-5            # radiation density today (photons only, fixed)

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

    [P1 — normalization, verified] At z=0, (1+z)/(1+z_t) = 1/(1+z_t), so
    num = 1 - tanh(-Δ·log10(1+z_t)) = 1 + tanh(Δ·log10(1+z_t)) = den, hence
    f_DE(0) = 1 EXACTLY and E²(0) = Ωm + Ωr + Ω_DE = 1 by construction (the
    `tests/` suite asserts this for all Ωm, Δ). The sign convention of z_t
    follows Li & Shafieloo (2020) Eq. (2): an inverted sign would also give
    f_DE(0)=1 but the OPPOSITE high-z evolution, so the unit test compares
    f_DE at a non-zero z against the paper's convention, not only at z=0.
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
    """Load the H(z) measurements (Cosmic Chronometers + BAO) from file.

    [LABEL FIX] These H(z) points combine Cosmic Chronometers AND BAO-derived
    H(z) measurements in a single (z, H, sigma) table; that is why there are
    more points than a CC-only compilation. The dataset is therefore labelled
    "CC+BAO" throughout the project. The file format is unchanged
    (three columns: z, H_obs, sigma), treated with independent (diagonal)
    Gaussian errors.

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
                    arr = arr[:, :3]
                    # [ROBUSTNESS] Drop NaN/inf rows and non-positive sigma
                    # instead of silently loading them — a corrupt line would
                    # otherwise poison every χ² downstream with no warning.
                    finite = np.all(np.isfinite(arr), axis=1)
                    if not np.all(finite):
                        print(f"  ⚠  {int(np.sum(~finite))} non-finite row(s) "
                              f"in {p} dropped (NaN/inf)")
                        arr = arr[finite]
                    if np.any(arr[:, 2] <= 0):
                        print(f"  ⚠  {int(np.sum(arr[:, 2] <= 0))} row(s) in "
                              f"{p} with non-positive sigma dropped")
                        arr = arr[arr[:, 2] > 0]
                    if len(arr) == 0:
                        print(f"  ⚠  {p} had no valid rows — using embedded")
                        return _CC_EMBEDDED
                    print(f"  ✓ CC+BAO H(z) loaded from file: {p}  "
                          f"({len(arr)} pts)")
                    return arr
            except Exception as e:
                print(f"  ⚠  Error reading {p}: {e} — using embedded data")
    return _CC_EMBEDDED


def load_pantheon(search_dirs: Optional[Sequence[str]] = None) -> Optional[dict]:
    """Load the Pantheon (2018) SNe Ia catalog with DIAGONAL errors.

    [LABEL FIX] This is the original Pantheon compilation (Scolnic et al. 2018,
    1048 SNe Ia), NOT Pantheon+. It is treated with independent per-SN errors
    (the `dmb` column), and M_abs is marginalized analytically (Goliath 2001).
    For the full Pantheon+ release with its covariance matrix use
    `load_pantheon_plus` and the 'Pantheon+' dataset instead.

    Searches for `pantheon_full_parameters.txt` (and variants) in the script
    folder and the cwd. Expected format: name zcmb zhel dz mb dmb.

    Returns:
        dict with arrays 'z', 'mb', 'dmb' sorted by z, or None if the
        file is not found.
    """
    names = ["pantheon_full_parameters.txt", "Pantheon_full_parameters.txt",
             "pantheon.txt", "Pantheon.txt"]
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
                print(f"  ✓ Pantheon (2018) loaded: {path}  ({len(z)} SNe Ia, "
                      f"z ∈ [{z[idx[0]]:.3f}, {z[idx[-1]]:.3f}], diagonal errors)")
                return {'z': z[idx], 'mb': mb[idx], 'dmb': dmb[idx],
                        'cov': None, 'path': path}
            except Exception as e:
                print(f"  ⚠  Error reading {path}: {e}")
    return None


def _pantheon_plus_files_present(search_dirs: Optional[Sequence[str]] = None
                                 ) -> Optional[str]:
    """Return a short note if BOTH Pantheon+ files exist on disk, else None.

    Lets the Posterior give a precise error: 'files present but unloadable'
    (bad covariance) vs 'files missing' have different fixes.
    """
    names_data = ["Pantheon+SH0ES.dat", "PantheonPlusSH0ES.dat",
                  "Pantheon+_data.txt"]
    names_cov = ["Pantheon+SH0ES_STAT+SYS.cov",
                 "Pantheon+SH0ES_STAT+SYS.txt", "PantheonPlus.cov"]
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        script_dir = os.getcwd()
    dirs = [script_dir, os.getcwd()] + list(search_dirs or [])

    def _find(cands):
        for d in dirs:
            for nm in cands:
                p = os.path.join(d, nm)
                if os.path.exists(p):
                    return p
        return None

    dp, cp = _find(names_data), _find(names_cov)
    if dp and cp:
        return f"{os.path.basename(dp)} + {os.path.basename(cp)}"
    return None


def load_pantheon_plus(data_name: str = "Pantheon+SH0ES.dat",
                       cov_name: str = "Pantheon+SH0ES_STAT+SYS.cov",
                       search_dirs: Optional[Sequence[str]] = None
                       ) -> Optional[dict]:
    """Load the full Pantheon+ (2022) catalog WITH its covariance matrix.

    The crucial statistical difference versus the 2018 Pantheon: Pantheon+
    ships a full N×N covariance matrix C (statistical + correlated
    systematics), so the χ² is the proper quadratic form
        χ² = Δᵀ C⁻¹ Δ,
    not a sum of independent terms. Ignoring the off-diagonal terms would
    underestimate the true uncertainties.

    File formats (as released by the Pantheon+ team, Brout et al. 2022):
      * Data table `Pantheon+SH0ES.dat`: a header row of column names followed
        by rows; we use the redshift column (zHD, with zCMB / zcmb as
        fallbacks) and the distance-modulus columns (MU_SH0ES and its error,
        with m_b_corr fallbacks).
      * Covariance `...STAT+SYS.cov`: first line = N, then N*N entries (one per
        line) in row-major order.

    For the cleanest cosmology fit, SNe with very low redshift (z < 0.01) are
    usually cut to avoid peculiar-velocity dominance; we apply z > 0.01 and
    slice the covariance to the surviving indices.

    Returns:
        dict with 'z', 'mu' (distance modulus), 'cov' (N×N), 'cov_inv',
        and helper sums for analytic M_abs marginalization; or None if the
        files are not found.
    """
    names_data = [data_name, "Pantheon+SH0ES.dat", "PantheonPlusSH0ES.dat",
                  "Pantheon+_data.txt"]
    names_cov = [cov_name, "Pantheon+SH0ES_STAT+SYS.cov",
                 "Pantheon+SH0ES_STAT+SYS.txt", "PantheonPlus.cov"]
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        script_dir = os.getcwd()
    dirs = [script_dir, os.getcwd()] + list(search_dirs or [])

    def _find(cands):
        for d in dirs:
            for nm in cands:
                p = os.path.join(d, nm)
                if os.path.exists(p):
                    return p
        return None

    data_path = _find(names_data)
    cov_path = _find(names_cov)
    if data_path is None or cov_path is None:
        return None

    try:
        # --- data table: header row of names, then values ---
        with open(data_path) as fh:
            header = fh.readline().split()
        tbl = np.genfromtxt(data_path, names=header, skip_header=1,
                            dtype=float, encoding='utf-8')
        cols = {n.lower(): n for n in tbl.dtype.names}

        def pick(options):
            for o in options:
                if o.lower() in cols:
                    return tbl[cols[o.lower()]].astype(float)
            return None

        z = pick(['zHD', 'zCMB', 'zcmb', 'zhel'])
        mu = pick(['MU_SH0ES', 'mu', 'MU'])
        if mu is None:                       # fall back to apparent magnitude
            mu = pick(['m_b_corr', 'mB', 'mb'])
        if z is None or mu is None:
            print("  ⚠  Pantheon+ data columns not recognized "
                  f"(have: {list(tbl.dtype.names)})")
            return None

        # --- covariance: first line N, then N*N entries row-major ---
        with open(cov_path) as fh:
            first = fh.readline().split()
        n_cov = int(first[0])
        flat = np.loadtxt(cov_path, skiprows=1)
        cov = flat.reshape(n_cov, n_cov)
        if n_cov != len(z):
            print(f"  ⚠  Pantheon+ size mismatch: data={len(z)} cov={n_cov}")
            return None

        # --- low-z cut (peculiar velocities) ---
        keep = z > 0.01
        z, mu = z[keep], mu[keep]
        cov = cov[np.ix_(keep, keep)]

        idx = np.argsort(z)
        z, mu = z[idx], mu[idx]
        cov = cov[np.ix_(idx, idx)]

        cov_inv = np.linalg.inv(cov)
        ones = np.ones(len(z))
        C_marg = float(ones @ cov_inv @ ones)        # 1ᵀ C⁻¹ 1
        print(f"  ✓ Pantheon+ (2022) loaded: {len(z)} SNe Ia with FULL "
              f"covariance, z ∈ [{z[0]:.3f}, {z[-1]:.3f}]")
        return {'z': z, 'mu': mu, 'cov': cov, 'cov_inv': cov_inv,
                'C_marg': C_marg, 'ones': ones,
                'path': data_path, 'cov_path': cov_path}
    except Exception as e:
        print(f"  ⚠  Error reading Pantheon+ ({data_path}, {cov_path}): {e}")
        return None


# =============================================================================
# 3. LIKELIHOODS AND POSTERIOR (model-agnostic, N-dimensional)
# =============================================================================

#: Canonical dataset keys and the components each one activates.
#  'cc'   -> the CC+BAO H(z) table (diagonal)
#  'sn'   -> Pantheon 2018 SNe (diagonal, M_abs marginalized)
#  'snp'  -> Pantheon+ 2022 SNe (full covariance, M_abs marginalized)
DATASET_COMPONENTS: Dict[str, set] = {
    'CC+BAO':            {'cc'},
    'Pantheon':          {'sn'},
    'Pantheon+':         {'snp'},
    'CC+BAO+Pantheon':   {'cc', 'sn'},
    'CC+BAO+Pantheon+':  {'cc', 'snp'},
}

#: Backward-compatible aliases so old commands / CSVs keep working.
DATASET_ALIASES: Dict[str, str] = {
    'CC': 'CC+BAO',
    'CC+Pantheon+': 'CC+BAO+Pantheon',   # old "Pantheon+" was really Pantheon
}


def canonical_dataset(name: str) -> str:
    """Map a dataset name (possibly an old alias) to its canonical key."""
    return DATASET_ALIASES.get(name, name)


class Posterior:
    """N-dimensional log-posterior tying together model + datasets + prior.

    This class is the ONLY contact point between the physics and the
    samplers: any sampler receives an instance and evaluates it as
    `post.log_prob(theta)`. That guarantees the strict physics ↔
    inference separation required by the design.

    Args:
        model: CosmoModel instance (from the MODELS registry).
        dataset: one of DATASET_COMPONENTS (or an old alias):
            'CC+BAO', 'Pantheon', 'Pantheon+', 'CC+BAO+Pantheon',
            'CC+BAO+Pantheon+'. The legacy names 'CC' and 'CC+Pantheon+'
            are accepted as aliases.
        prior_type: 'flat' (box) or 'gaussian' (Planck on Ωm and H0,
            flat on the extra parameters within bounds).
        cc_data: (N,3) array of CC+BAO H(z) points.
        pantheon: dict from load_pantheon()/load_pantheon_plus() or None.
        n_zgrid: Resolution of the z grid for the SNe comoving-distance
            integral (cumulative trapezoid).
    """

    def __init__(self, model: CosmoModel, dataset: str = 'CC+BAO',
                 prior_type: str = 'flat',
                 cc_data: Optional[np.ndarray] = None,
                 pantheon: Optional[dict] = None,
                 n_zgrid: int = 1200):
        self.model = model
        self.dataset = canonical_dataset(dataset)
        if self.dataset not in DATASET_COMPONENTS:
            raise ValueError(
                f"Unknown dataset '{dataset}'. Valid: "
                f"{list(DATASET_COMPONENTS)} (or aliases {list(DATASET_ALIASES)})")
        self.components = DATASET_COMPONENTS[self.dataset]
        self.prior_type = prior_type

        # CC+BAO H(z) table (always loaded; used only if 'cc' is active).
        self.cc = cc_data if cc_data is not None else load_cc()
        self.z_cc, self.H_cc, self.sig_cc = (self.cc[:, 0], self.cc[:, 1],
                                             self.cc[:, 2])

        # SNe: pick the right loader for the requested component.
        self.pantheon = pantheon
        if 'sn' in self.components and self.pantheon is None:
            self.pantheon = load_pantheon()
            if self.pantheon is None:
                raise FileNotFoundError(
                    "Dataset includes Pantheon (2018) but its data file "
                    "(pantheon_full_parameters.txt) was not found")
        if 'snp' in self.components and self.pantheon is None:
            self.pantheon = load_pantheon_plus()
            if self.pantheon is None:
                found = _pantheon_plus_files_present()
                if found:
                    raise ValueError(
                        "Dataset includes Pantheon+ (2022): the files were "
                        f"found ({found}) but could NOT be loaded — check the "
                        "covariance size matches the data and is invertible "
                        "(see the warning printed above).")
                raise FileNotFoundError(
                    "Dataset includes Pantheon+ (2022) but its data and/or "
                    "covariance files (Pantheon+SH0ES.dat and "
                    "Pantheon+SH0ES_STAT+SYS.cov) were not found")

        # Fine z grid for the cumulative trapezoid of d_C(z), valid for ANY
        # E²(z;θ). Built whenever an SNe component is active.
        if self.pantheon is not None:
            zmax = float(self.pantheon['z'].max()) * 1.02
            self._zg = np.linspace(0.0, zmax, n_zgrid)
            if 'sn' in self.components:           # diagonal Pantheon 2018
                self._inv_s2 = 1.0 / self.pantheon['dmb']**2
                self._C_marg = float(np.sum(self._inv_s2))

    # ── total number of data points (for reduced χ² and BIC) ────────────────
    @property
    def n_data(self) -> int:
        """Total number of observational points of the active dataset."""
        n = 0
        if 'cc' in self.components:
            n += len(self.z_cc)
        if ('sn' in self.components or 'snp' in self.components) and self.pantheon:
            n += len(self.pantheon['z'])
        return n

    # ── χ² components ────────────────────────────────────────────────────────
    def chi2_cc(self, theta: np.ndarray) -> float:
        """CC+BAO H(z) χ² for the active model (diagonal errors)."""
        Hm = self.model.H(self.z_cc, theta)
        return float(np.sum(((self.H_cc - Hm) / self.sig_cc)**2))

    def _mu_theory(self, z_sn: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """Theoretical distance modulus μ_th(z; θ) for the active model.

        [P3 — FLAT universe only] The comoving (line-of-sight) distance is
        d_C = (c/H0)∫dz/E, and here the transverse comoving distance is taken
        equal to it, d_M = d_C, so d_L = (1+z)·d_C. This is correct ONLY for
        spatially flat models (Ω_k = 0), which is the case for ALL models in
        the registry (ΛCDM, wCDM, CPL, PEDE, GEDE are flat by construction).
        A curved extension (Ω_k ≠ 0, e.g. a "Variable Curvature" model) would
        need the generalized d_M:
            d_M = (c/(H0√|Ω_k|)) · sinn(√|Ω_k|·H0·d_C/c),
        with sinn = sinh for open (Ω_k>0) and sin for closed (Ω_k<0). Adding a
        non-flat model therefore requires generalizing THIS method; the flat
        form below would otherwise give wrong distances.
        """
        e2 = self.model.E2(self._zg, theta)
        if np.any(e2 <= 0):
            return None
        invE = 1.0 / np.sqrt(e2)
        I = cumulative_trapezoid(invE, self._zg, initial=0.0)
        I_sn = np.interp(z_sn, self._zg, I)
        dL = (C_LIGHT / theta[1]) * (1.0 + z_sn) * I_sn          # Mpc
        return 5.0 * np.log10(np.clip(dL, 1e-10, None)) + 25.0

    def chi2_pantheon(self, theta: np.ndarray) -> float:
        """Pantheon (2018) effective χ², DIAGONAL, M_abs marginalized.

        χ²_eff = A − B²/C (Goliath et al. 2001):
            A = Σ Δ²/σ², B = Σ Δ/σ², C = Σ 1/σ², Δ = m_obs − μ_th.
        """
        mu_th = self._mu_theory(self.pantheon['z'], theta)
        if mu_th is None:
            return np.inf
        delta = self.pantheon['mb'] - mu_th       # m_obs − μ_th (+ M_abs offset)
        A = float(np.sum(delta**2 * self._inv_s2))
        B = float(np.sum(delta * self._inv_s2))
        return A - B**2 / self._C_marg

    def chi2_pantheon_plus(self, theta: np.ndarray) -> float:
        """Pantheon+ (2022) effective χ² with FULL covariance, M_abs marginalized.

        The proper quadratic form with the covariance matrix C:
            χ²_eff = A − B²/Cn,
            A = Δᵀ C⁻¹ Δ,  B = Δᵀ C⁻¹ 1,  Cn = 1ᵀ C⁻¹ 1,
        which is the matrix generalization of Goliath et al. (2001): it
        analytically marginalizes the constant M_abs offset while keeping the
        correlated systematics encoded in C.
        """
        mu_th = self._mu_theory(self.pantheon['z'], theta)
        if mu_th is None:
            return np.inf
        delta = self.pantheon['mu'] - mu_th
        Ci = self.pantheon['cov_inv']
        Cid = Ci @ delta
        A = float(delta @ Cid)
        B = float(self.pantheon['ones'] @ Cid)
        return A - B**2 / self.pantheon['C_marg']

    def chi2(self, theta: np.ndarray) -> Tuple[float, int]:
        """Total χ² of the active dataset. Returns (chi2, n_data).

        [B4 — combined datasets and M_abs] The SNe blocks marginalize the
        absolute-magnitude offset M_abs ANALYTICALLY (Goliath 2001), which
        also marginalizes the M_abs–H0 degeneracy WITHIN the SNe block: SNe
        alone constrain only the shape E(z), not H0. When an H(z) block
        (CC+BAO) is added, the total χ² = χ²_CC(θ) + χ²_SN(θ) and H0 is then
        constrained DIRECTLY by CC+BAO (which does not marginalize it). This
        is intentional and is the point of combining the datasets — CC+BAO is
        what anchors H0 — but it means that for a combined dataset H0 is NOT
        marginalized away; only the SNe nuisance offset is. Reduced-χ² and
        BIC use the full n_data across both blocks.
        """
        c = 0.0
        if 'cc' in self.components:
            c += self.chi2_cc(theta)
        if 'sn' in self.components and self.pantheon:
            c += self.chi2_pantheon(theta)
        if 'snp' in self.components and self.pantheon:
            c += self.chi2_pantheon_plus(theta)
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

        # 2) vectorized CC+BAO: E2 with broadcasting (Bv, Ncc)
        if 'cc' in self.components:
            e2 = self.model.E2(self.z_cc[None, :], th_cols)
            Hm = T[:, 1:2] * np.sqrt(np.clip(e2, 1e-12, None))
            lp += -0.5 * np.sum(((self.H_cc[None, :] - Hm)
                                 / self.sig_cc[None, :])**2, axis=1)

        # 3) vectorized SNe (Pantheon 2018 diagonal OR Pantheon+ covariance):
        #    row-wise cumulative trapezoid for the comoving distance (Bv, Nzg)
        if ('sn' in self.components or 'snp' in self.components) and self.pantheon:
            e2 = self.model.E2(self._zg[None, :], th_cols)
            bad = np.any(e2 <= 0, axis=1)
            e2 = np.clip(e2, 1e-12, None)
            I = cumulative_trapezoid(1.0 / np.sqrt(e2), self._zg,
                                     axis=1, initial=0.0)
            z_sn = self.pantheon['z']
            idx = np.searchsorted(self._zg, z_sn).clip(1, len(self._zg) - 1)
            z0, z1 = self._zg[idx - 1], self._zg[idx]
            w = (z_sn - z0) / (z1 - z0)
            I_sn = I[:, idx - 1] * (1 - w)[None, :] + I[:, idx] * w[None, :]
            dL = (C_LIGHT / T[:, 1:2]) * (1.0 + z_sn)[None, :] * I_sn
            mu_th = 5.0 * np.log10(np.clip(dL, 1e-10, None)) + 25.0

            if 'sn' in self.components:        # diagonal (Pantheon 2018)
                delta = self.pantheon['mb'][None, :] - mu_th
                A = np.sum(delta**2 * self._inv_s2[None, :], axis=1)
                Bm = np.sum(delta * self._inv_s2[None, :], axis=1)
                chi2p = A - Bm**2 / self._C_marg
            else:                              # full covariance (Pantheon+)
                delta = self.pantheon['mu'][None, :] - mu_th       # (Bv, Nsn)
                Ci = self.pantheon['cov_inv']
                CiD = delta @ Ci                                   # (Bv, Nsn)
                A = np.sum(CiD * delta, axis=1)                    # Δᵀ C⁻¹ Δ
                Bm = CiD @ self.pantheon['ones']                   # Δᵀ C⁻¹ 1
                chi2p = A - Bm**2 / self.pantheon['C_marg']
            chi2p[bad] = np.inf
            lp += -0.5 * chi2p

        out[ok] = lp
        return out


# =============================================================================
# 4. STATISTICAL AND MODEL-SELECTION ESTIMATORS
# =============================================================================

def autocorr_time_fft(x: np.ndarray, c: float = 5.0) -> float:
    r"""Integrated autocorrelation time τ via FFT, with Sokal windowing.

    Definition. For a stationary chain x_t with normalized autocorrelation
    function ρ(t), the integrated autocorrelation time is

        τ = 1 + 2 Σ_{t=1}^{∞} ρ(t).

    The sum cannot be carried to ∞ on a finite chain: the tail of ρ(t) is
    pure noise (variance ~1/N per lag) and adding it injects variance that
    grows with the truncation window. The estimator must therefore truncate
    at some window M, trading bias (M too small) against variance (M too
    large).

    Windowing rule (Sokal). We accumulate the partial sums

        τ(M) = 1 + 2 Σ_{t=1}^{M} ρ(t)

    and stop at the FIRST window M satisfying  M ≥ c · τ(M)  (default c = 5,
    the value recommended by Sokal and used by emcee). This self-consistent
    rule adapts the window to the chain's own correlation length and is the
    community standard; it replaces the earlier ad-hoc "first lag below
    0.05, else N/4" cutoff, which mis-fired in two opposite regimes (a chain
    whose ρ never drops below 0.05 and a chain already below 0.05 at lag 1
    both returned argmax = 0, conflating a terrible chain with an excellent
    one).

    Complexity. The autocovariance is obtained from a single zero-padded FFT
    and its inverse, so the cost is O(N log N) in time and O(N) in space —
    for N = 1.6×10^5 this is ~250× faster than an O(N²) np.correlate.

    Args:
        x: 1-D chain of samples for ONE parameter.
        c: Sokal window constant (window must reach c·τ before stopping).

    Returns:
        τ ≥ 1. Returns 1.0 for degenerate input (n < 5 or zero variance).

    Reference:
        A. D. Sokal, "Monte Carlo Methods in Statistical Mechanics" (1996);
        Foreman-Mackey et al. (2013), emcee, autocorr module.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 5:
        return 1.0
    x = x - x.mean()
    # autocovariance via FFT (Wiener-Khinchin), normalized to ρ(0) = 1
    f = np.fft.rfft(x, n=2 * n)
    acf = np.fft.irfft(f * np.conj(f))[:n].real
    if acf[0] <= 0:
        return 1.0
    acf /= acf[0]
    # cumulative τ(M) = 1 + 2 Σ_{t=1..M} ρ(t); Sokal stop at M ≥ c·τ(M).
    tau_cum = 1.0 + 2.0 * np.cumsum(acf[1:])          # tau_cum[M-1] = τ(M)
    windows = np.arange(1, n)                          # M = 1, 2, ...
    reached = windows >= c * tau_cum
    if np.any(reached):
        M = int(np.argmax(reached))                    # first True
        tau = float(tau_cum[M])
    else:
        # ρ never decorrelates within the chain: report the full-window τ as
        # a (conservative, high) estimate rather than silently truncating.
        tau = float(tau_cum[-1])
    return max(1.0, tau)


def autocorr_time_max(chains: np.ndarray) -> float:
    """Worst-case τ across chains and parameters for a (M, N, d) array.

    Averages τ over the M chains per parameter, then takes the maximum over
    the d parameters (the conservative choice that drives ESS).
    """
    M, N, d = chains.shape
    taus = [np.mean([autocorr_time_fft(chains[c, :, p]) for c in range(M)])
            for p in range(d)]
    return float(max(taus))


def ess_chains(chains: np.ndarray) -> float:
    r"""Effective Sample Size for MCMC chains, shape (M, N, d).

    Formula. With M chains of length N in d dimensions,

        ESS = (M · N) / τ_max,

    where τ_max = max_p ( (1/M) Σ_c τ(x_{c,·,p}) ) is the per-chain-averaged
    integrated autocorrelation time of the WORST-mixing parameter (see
    `autocorr_time_max`). Taking the max over parameters (rather than a
    per-parameter ESS) yields a single conservative scalar: the number of
    effectively independent draws is governed by the slowest direction.

    This is the same MN/τ convention used by emcee; it is intentionally
    distinct from ArviZ's rank-normalized bulk-ESS, which is reported
    separately by the diagnostics layer when needed.
    """
    M, N, _ = chains.shape
    return float(M * N / autocorr_time_max(chains))


def estimate_grid_window(post, sigma_mult: float = 4.0,
                         n_steps: int = 400, n_chains: int = 4,
                         use_median: bool = True) -> List[tuple]:
    """Quick classical pre-fit that positions a discrete inference grid.

    [ADAPTIVE GRID] A grid-based method (QVMC, classical VI, the QPU
    GridEncoding) represents the posterior on a discrete 2^nqpp grid per
    parameter. Spanning the full (wide) sample_box, the cosmological
    posterior — far narrower than the grid spacing — collapses onto one or
    two cells and can never look smooth, independent of the iteration budget.
    This helper runs a short vectorized Metropolis chain to locate the
    posterior centre and per-parameter width, then returns per-parameter
    windows [centre − k·σ, centre + k·σ] (k = `sigma_mult`), clipped to the
    model's physical bounds, so a few qubits actually RESOLVE the posterior.

    [B3] Lives in cosmo_core so BOTH the simulator pipeline
    (cosmo_modular_quantum) and the real-hardware pipeline
    (qpu_cosmo_samplers) build their grids the SAME way — otherwise the QPU
    KL is computed on a different (coarser) grid than the simulator and the
    two are not comparable.

    [S1] The centre defaults to the per-parameter MEDIAN (robust to the
    asymmetric posteriors of wCDM/GEDE, where the mean can fall outside the
    high-density region); pass use_median=False for the mean.

    The pre-fit only places/scales the grid (an adaptive-grid technique); it
    does not feed the downstream result itself.
    """
    model = post.model
    d = model.n_params
    lo = np.array([b[0] for b in model.sample_box])
    hi = np.array([b[1] for b in model.sample_box])
    step = 0.06 * (hi - lo)
    theta = lo + (hi - lo) * RNG.uniform(0.3, 0.7, size=(n_chains, d))
    lp = post.log_prob_batch(theta)
    samples = []
    for s in range(n_steps):
        prop = theta + step * RNG.normal(size=(n_chains, d))
        lpp = post.log_prob_batch(prop)
        acc = np.log(RNG.uniform(size=n_chains) + 1e-300) < (lpp - lp)
        theta[acc] = prop[acc]
        lp[acc] = lpp[acc]
        if s >= n_steps // 3:                  # discard burn-in third
            samples.append(theta.copy())
    flat = np.concatenate(samples, axis=0)
    centre = np.median(flat, axis=0) if use_median else flat.mean(0)
    sd = flat.std(0) + 1e-9
    blo = np.array([b[0] for b in model.bounds])
    bhi = np.array([b[1] for b in model.bounds])
    win_lo = np.clip(centre - sigma_mult * sd, blo, bhi)
    win_hi = np.clip(centre + sigma_mult * sd, blo, bhi)
    for i in range(d):                         # guard a degenerate window
        if win_hi[i] - win_lo[i] < 1e-6:
            win_lo[i], win_hi[i] = model.sample_box[i]
    return list(zip(win_lo.tolist(), win_hi.tolist()))


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
    """Maximum classical R̂ over all parameters, shape (M, N, d)."""
    return max(gelman_rubin(chains[:, :, p]) for p in range(chains.shape[2]))


def _split_chains(chains: np.ndarray) -> np.ndarray:
    """Split each chain in half: (M, N, d) -> (2M, N//2, d).

    Splitting is what lets R̂ detect WITHIN-chain non-stationarity (e.g. a
    slow trend): a drifting chain disagrees with its own two halves, which a
    whole-chain R̂ cannot see. Required for the modern split-R̂ of Vehtari
    et al. (2021). If N is odd the last sample is dropped.
    """
    M, N, d = chains.shape
    half = N // 2
    if half < 2:
        return chains
    first, second = chains[:, :half, :], chains[:, half:2 * half, :]
    return np.concatenate([first, second], axis=0)


def _rank_normalize(x: np.ndarray) -> np.ndarray:
    """Rank-normalize a flat array to approximate normal scores.

    Replaces each value by Φ⁻¹((r − 3/8)/(n − 1/4)) with r its (1-based,
    average-tie) rank. This makes R̂ robust to heavy tails and
    non-Gaussian marginals (the rank-normalized R̂ of Vehtari 2021); a
    purely Gaussian estimator can otherwise look "converged" on a skewed
    posterior it has not actually explored.
    """
    from scipy.stats import norm, rankdata
    n = x.size
    r = rankdata(x.ravel(), method='average')
    z = norm.ppf((r - 0.375) / (n - 0.25))
    return z.reshape(x.shape)


def split_rhat(chains: np.ndarray, rank_normalize: bool = True) -> float:
    r"""Rank-normalized split-R̂ over all parameters (Vehtari et al. 2021).

    Pipeline per parameter: split each chain in half (catches drift),
    optionally rank-normalize the pooled draws (robust to heavy tails),
    then apply the standard between/within variance ratio

        R̂ = sqrt( ((N-1)/N · W + B/N) / W ).

    The reported value is the MAX over parameters. The community
    convergence threshold for this statistic is R̂ < 1.01 (see
    `mcmc_converged`), markedly stricter than the legacy 1.05.

    Args:
        chains: (M, N, d) array.
        rank_normalize: apply the rank-normal transform before R̂.

    Returns:
        max-over-parameters split-R̂ (≥ 1; ~1 means converged).
    """
    sc = _split_chains(chains)
    M, N, d = sc.shape
    if N < 2:
        return np.nan
    rhats = []
    for p in range(d):
        x = sc[:, :, p]
        if rank_normalize:
            x = _rank_normalize(x)
        mu_j = x.mean(axis=1)
        B = N * np.var(mu_j, ddof=1)
        W = np.mean(np.var(x, axis=1, ddof=1))
        if W <= 1e-12:
            rhats.append(np.nan)
            continue
        var_hat = (1.0 - 1.0 / N) * W + B / N
        rhats.append(float(np.sqrt(var_hat / W)))
    finite = [r for r in rhats if np.isfinite(r)]
    return float(max(finite)) if finite else np.nan


#: Community-standard convergence threshold for rank-normalized split-R̂
#: (Vehtari et al. 2021). Stricter than the legacy 1.05; with 1.05 a chain
#: that has not actually mixed can be declared converged, which would make
#: the project's central "quantum reproduces classical" comparison vacuous
#: (agreement between two unconverged chains proves nothing).
RHAT_THRESHOLD = 1.01


def mcmc_converged(chains: np.ndarray, threshold: float = RHAT_THRESHOLD
                   ) -> bool:
    """True if rank-normalized split-R̂ is below `threshold` (default 1.01)."""
    r = split_rhat(chains)
    return bool(np.isfinite(r) and r < threshold)


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


# =============================================================================
# 8. AER SIMULATOR FACTORY (CPU/GPU device selection, single source of truth)
# =============================================================================
#
#  All quantum simulators in the project are created through `make_simulator`
#  so the CPU/GPU choice lives in ONE place. On a CUDA node, pass
#  prefer_gpu=True; if Aer exposes a 'GPU' device the simulator runs on it,
#  otherwise it transparently falls back to CPU. This is what the `--gpu` flag
#  toggles in both executable scripts.
#
#  [GPU BACKEND] On the development machine the GPU statevector path is
#  provided by NVIDIA cuQuantum / cuStateVec (packages cuquantum-cu12,
#  custatevec-cu12), against which qiskit-aer 0.17.x is built — NOT by a
#  separate `qiskit-aer-gpu` wheel. Either way, `AerSimulator().
#  available_devices()` reports 'GPU' when a usable CUDA device is present, so
#  the probe below is backend-agnostic. On a CPU-only node (e.g. Nicte-Ha) it
#  returns CPU and the project runs unchanged.

_GPU_PROBED = False
_GPU_AVAILABLE = False


def gpu_available() -> bool:
    """True if qiskit-aer exposes a 'GPU' device (probed once and cached).

    Works for both GPU backends: a cuQuantum/cuStateVec-enabled qiskit-aer
    (this project's setup) and a legacy `qiskit-aer-gpu` build. On a CPU-only
    node this returns False and the project runs on CPU.
    """
    global _GPU_PROBED, _GPU_AVAILABLE
    if _GPU_PROBED:
        return _GPU_AVAILABLE
    _GPU_PROBED = True
    try:
        from qiskit_aer import AerSimulator
        devices = AerSimulator().available_devices()
        _GPU_AVAILABLE = 'GPU' in devices
    except Exception:
        _GPU_AVAILABLE = False
    return _GPU_AVAILABLE


def resolve_device(prefer_gpu: bool) -> str:
    """Return the device string ('GPU' or 'CPU') actually usable.

    prefer_gpu=True downgrades to 'CPU' (with the caller free to warn) when no
    GPU is present, so a run never crashes for asking for a GPU that isn't
    there.
    """
    if prefer_gpu and gpu_available():
        return 'GPU'
    return 'CPU'


def make_simulator(method: str = 'statevector', prefer_gpu: bool = False,
                   n_qubits: Optional[int] = None, **kwargs):
    """Create an AerSimulator on the resolved device (single source of truth).

    Args:
        method: Aer simulation method (default 'statevector').
        prefer_gpu: if True and a GPU is available, run on it; else CPU.
        n_qubits: optional circuit width hint. Blocking (which splits the
            statevector across the GPU) only helps for LARGE circuits; for the
            many tiny circuits of QMCMC/QVMC/QGA it adds pure overhead, so it
            is enabled only when n_qubits is large (or unknown). [A4]
        **kwargs: forwarded to AerSimulator (e.g. precision, blocking options).

    Returns:
        A configured AerSimulator. On GPU, batched-shots distribution is
        enabled (it helps the many-small-circuits workload); statevector
        blocking is enabled only for wide circuits, where it actually pays off.
        cuStateVec is requested when available (the project's cuQuantum
        backend); Aer ignores the flag if the build lacks cuStateVec.
    """
    from qiskit_aer import AerSimulator
    device = resolve_device(prefer_gpu)
    opts = dict(method=method, device=device)
    if device == 'GPU':
        # Batched shots distribute many circuits/shots across the GPU — always
        # useful for our workload.
        opts.update(batched_shots_gpu=True)
        # Prefer the cuStateVec kernels when the Aer build supports them
        # (cuQuantum). Harmlessly ignored otherwise.
        opts.setdefault('cuStateVec_enable', True)
        # [A4] Statevector blocking only helps for WIDE statevectors; for the
        # tiny proposal/acceptance circuits it is wasted overhead. Enable it
        # only when the circuit is large or the width is unknown.
        blocking_qubits = kwargs.pop('blocking_qubits', 22)
        if n_qubits is None or n_qubits >= blocking_qubits:
            opts.update(blocking_enable=True, blocking_qubits=blocking_qubits)
    opts.update(kwargs)
    return AerSimulator(**opts)


def fmt_theta(model: CosmoModel, theta: np.ndarray) -> str:
    """Format θ with parameter names for readable logs."""
    return "  ".join(f"{n}={v:.4f}" for n, v in zip(model.param_names, theta))

# =============================================================================
# 6. RUN-DIRECTORY UTILITIES (timestamped output folders)
# =============================================================================
#
#  Every executable script funnels its figures, logs and per-run CSV into a
#  UNIQUE timestamped folder so results from different runs never overwrite or
#  mix. This single helper is the one source of truth for that convention,
#  shared by all scripts.

def make_run_dir(base: str = "results", tag: Optional[str] = None,
                 timestamp: bool = True) -> str:
    """Create and return a unique output directory for one run.

    Layout:  <base>/run_<YYYYMMDD_HHMMSS>[_<tag>]/

    Args:
        base: Parent directory that collects all runs (created if missing).
        tag: Optional short suffix (e.g. the model name) to make the folder
            self-describing at a glance.
        timestamp: If True (default), embed a second-resolution timestamp so
            sequential runs get distinct folders. If False, the directory is
            just <base>/<tag> (or <base> when tag is None).

    Returns:
        Path to the freshly created run directory.
    """
    if timestamp:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        name = f"run_{stamp}_{tag}" if tag else f"run_{stamp}"
    else:
        name = tag if tag else ""
    run_dir = os.path.join(base, name) if name else base

    # Avoid a same-second collision by appending a counter.
    final = run_dir
    k = 1
    while timestamp and os.path.exists(final):
        final = f"{run_dir}_{k}"
        k += 1

    os.makedirs(final, exist_ok=True)
    return final
