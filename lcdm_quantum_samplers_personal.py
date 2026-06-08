# =============================================================================
# ΛCDM Parameter Estimation — QMCMC and QVMC
# =============================================================================
#
# Both algorithms fit flat ΛCDM to Cosmic Chronometer H(z) data.
# Parameters: θ = (Ωm, H0)
#
# ┌─────────────────────────────────────────────────────────────────────────┐
# │  WHAT IS QUANTUM vs CLASSICAL IN EACH ALGORITHM                        │
# │                                                                         │
# │  QMCMC                                                                  │
# │    QUANTUM : proposes the STEP (direction + size) via circuit          │
# │    CLASSICAL: evaluates log-posterior, accepts/rejects (Metropolis)    │
# │    CLASSICAL: convergence diagnostics (τ, Gelman-Rubin)               │
# │    The chain IS classical; only the proposal mechanism is quantum.     │
# │                                                                         │
# │  QVMC                                                                   │
# │    CLASSICAL: builds the target posterior on a discrete grid           │
# │    QUANTUM : represents and samples from the posterior (circuit)       │
# │    CLASSICAL: trains the circuit (COBYLA minimises KL divergence)     │
# │    QUANTUM : each shot IS a posterior sample                           │
# │    The training IS classical; the sampler IS quantum.                  │
# └─────────────────────────────────────────────────────────────────────────┘
#
# HOW TO MAKE IT MORE QUANTUM (progression roadmap at bottom of file)
#
# Requirements: pip install qiskit qiskit-aer scipy numpy matplotlib
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import minimize
from scipy.stats import multivariate_normal
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

try:
    import corner
    HAS_CORNER = True
except ImportError:
    HAS_CORNER = False
    print("  ⚠  'corner' no instalado — corner plots desactivados.")
    print("     Instala con: pip install corner")

from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit_aer import AerSimulator

# ── reproducibility ───────────────────────────────────────────────────────────
RNG = np.random.default_rng(42)

# =============================================================================
# 1. DATA AND PHYSICS
# =============================================================================
# Everything in this section is purely classical.

OMEGA_R0 = 9.4e-5   # radiation density today (Planck 2018, fixed)

# ── CC data (Cosmic Chronometers) ─────────────────────────────────────────────
_CC_DATA = np.array([
    [0.07,  69.00, 19.60], [0.10,  69.00, 12.00], [0.12,  68.60, 26.20],
    [0.17,  83.00,  8.00], [0.1791, 75.00,  4.00], [0.1993, 75.00,  5.00],
    [0.20,  72.90, 29.60], [0.240, 79.69,  2.65], [0.27,  77.00, 14.00],
    [0.28,  88.80, 36.60], [0.300, 81.70,  6.22], [0.31,  78.17,  4.74],
    [0.350, 82.70,  8.40], [0.3519,83.00, 14.00], [0.36,  79.93,  3.39],
    [0.38,  81.50,  1.90], [0.3802,83.00, 13.50], [0.40,  95.00, 17.00],
    [0.4004,77.00, 10.20], [0.4247,87.10, 11.20], [0.43,  86.45,  3.68],
    [0.44,  82.60,  7.80], [0.4497,92.80, 12.90], [0.47,  89.00, 34.00],
    [0.4783,80.90,  9.00], [0.48,  97.00, 62.00], [0.51,  90.40,  1.90],
    [0.52,  94.35,  2.65], [0.56,  93.33,  2.32], [0.570, 92.90,  7.85],
    [0.59,  98.48,  3.19], [0.5929,104.00,13.00], [0.60,  87.90,  6.10],
    [0.61,  97.30,  2.10], [0.64,  98.82,  2.99], [0.6797,92.00,  8.00],
    [0.73,  97.30,  7.00], [0.7812,105.00,12.00], [0.8754,125.00,17.00],
    [0.88,  90.00, 40.00], [0.90,  117.00,23.00], [1.037, 154.00,20.00],
    [1.30,  168.00,17.00], [1.363, 160.00,33.60], [1.43,  177.00,18.00],
    [1.53,  140.00,14.00], [1.75,  202.00,40.00], [1.965, 186.50,50.40],
    [2.33,  224.00, 8.00], [2.34,  222.00, 7.00], [2.360, 226.00, 8.00],
])


def _load_cc_file(path="cosmic_chronometers.txt"):
    """
    [FIX 3] Lee Cronometros Cosmicos de un archivo de texto si existe.
    Formato: columnas  z  H(z)  sigma_H  ('#' = comentario).
    Si no existe, escribe el array embebido a 'path' y lo usa (asi en
    corridas futuras ya hay archivo). Retorna un array (N,3).
    """
    import os as _os
    for c in [path, "cc_hz_data.txt", "CC.txt"]:
        if c and _os.path.exists(c):
            try:
                raw = np.genfromtxt(c, comments="#")
                if raw.ndim == 1:
                    raw = raw.reshape(1, -1)
                raw = raw[(raw[:, 0] > 0) & (raw[:, 2] > 0)]
                raw = raw[np.argsort(raw[:, 0])]
                print(f"  ✓ CC cargado de archivo: {c}  ({len(raw)} puntos)")
                return raw[:, :3]
            except Exception as e:
                print(f"  ⚠  Error leyendo {c}: {e}")
    try:
        with open(path, "w") as f:
            f.write("# z   H(z)[km/s/Mpc]   sigma_H\n")
            for row in _CC_DATA_BUILTIN:
                f.write(f"{row[0]:.4f}  {row[1]:.2f}  {row[2]:.2f}\n")
        print(f"  ℹ  CC: archivo no encontrado; escribi el array embebido a {path}")
    except Exception:
        pass
    return _CC_DATA_BUILTIN


_CC_DATA_BUILTIN = _CC_DATA.copy()
_CC_DATA = _load_cc_file()


# ── Pantheon+ SNe Ia — se lee de archivo externo ──────────────────────────────
#
# Archivo: pantheon_full_parameters.txt  (Scolnic et al. 2022 / Brout et al. 2022)
# Formato: columnas separadas por espacios, primera línea es cabecera con #
#   #name  zcmb  zhel  dz  mb  dmb
#   nombre z_CMB z_hel dz  m_b dm_b
#
# mb  = magnitud aparente corregida de la SNe Ia
# dmb = incertidumbre en mb (estadística + sistemática)
#
# La likelihood de Pantheon+ se implementa correctamente usando el módulo
# de distancia, NO como H(z). La física es:
#
#   mu_obs  = mb - M_abs          (módulo de distancia observado)
#   mu_th   = 5*log10(dL) + 25    (módulo de distancia teórico)
#   dL(z)   = (c/H0)*(1+z)*integral_0^z dz'/E(z')   [Mpc]
#   E(z)    = sqrt(Om*(1+z)^3 + (1-Om))
#
#   chi2_SNe = sum_i [(mu_obs_i - mu_th_i) / dmb_i]^2
#
# M_abs es un parámetro de nuisance degenerado con H0.
# Se trata con marginalización analítica (Goliath et al. 2001):
#   chi2_eff = A - B^2/C   donde:
#   A = sum[(mu_obs - mu_th)^2 / sigma^2]
#   B = sum[(mu_obs - mu_th)   / sigma^2]
#   C = sum[1                  / sigma^2]
# Esto elimina M_abs de la inferencia.

from scipy.integrate import quad as _quad

_C_LIGHT = 299792.458   # km/s

def _load_pantheon_file(search_dirs=None):
    """
    Busca pantheon_full_parameters.txt en la carpeta del script y en cwd.
    Retorna dict con z, mb, dmb arrays, o None si no se encuentra.
    """
    import os
    names = [
        "pantheon_full_parameters.txt",
        "Pantheon_full_parameters.txt",
        "pantheon_plus.txt", "pantheon_plus.csv",
        "PantheonPlus.txt",  "PantheonPlus.csv",
    ]
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        script_dir = os.getcwd()
    dirs = [script_dir, os.getcwd()]
    if search_dirs:
        dirs += search_dirs

    for d in dirs:
        for name in names:
            path = os.path.join(d, name)
            if not os.path.exists(path):
                continue
            try:
                raw = np.genfromtxt(path, comments='#',
                                    dtype=None, encoding='utf-8')
                # Formato: name zcmb zhel dz mb dmb
                if raw.ndim == 1:
                    raw = raw.reshape(1, -1)
                # Puede ser structured o unstructured
                if raw.dtype.names:
                    z   = raw['f1'].astype(float)   # zcmb
                    mb  = raw['f4'].astype(float)   # mb
                    dmb = raw['f5'].astype(float)   # dmb
                else:
                    z   = raw[:, 1].astype(float)
                    mb  = raw[:, 4].astype(float)
                    dmb = raw[:, 5].astype(float)
                # Filtrar z > 0 y dmb > 0
                mask = (z > 0) & (dmb > 0) & np.isfinite(mb)
                z, mb, dmb = z[mask], mb[mask], dmb[mask]
                # Ordenar por z
                idx = np.argsort(z)
                z, mb, dmb = z[idx], mb[idx], dmb[idx]
                print(f"  ✓ Pantheon+ cargado: {path}")
                print(f"    {len(z)} SNe Ia | z ∈ [{z.min():.3f}, {z.max():.3f}]")
                return {'z': z, 'mb': mb, 'dmb': dmb, 'path': path}
            except Exception as e:
                print(f"  ⚠  Error leyendo {path}: {e}")
    return None

_PANTHEON_RAW = _load_pantheon_file()

# Pre-calcular integrales de luminosidad en una grilla de Om para velocidad
# (igual que Sarracino et al. 2022 y 2602.15459)
_OM_GRID_PAN  = np.linspace(0.18, 0.50, 300)
_Z_PAN        = _PANTHEON_RAW['z']   if _PANTHEON_RAW else None
_INTEGRAL_PAN = None   # se llena la primera vez que se usa

def _precompute_pantheon_integrals():
    """
    Precalcula I(z, Om) = integral_0^z dz'/E(z',Om) para la grilla de Om.
    Shape: (len(_OM_GRID_PAN), len(_Z_PAN))
    Se llama una sola vez al inicio.
    """
    global _INTEGRAL_PAN
    if _INTEGRAL_PAN is not None or _Z_PAN is None:
        return
    print("  Precalculando integrales de luminosidad Pantheon+ ... ", end="", flush=True)
    n_om = len(_OM_GRID_PAN)
    n_z  = len(_Z_PAN)
    _INTEGRAL_PAN = np.zeros((n_om, n_z))
    for i, Om in enumerate(_OM_GRID_PAN):
        OmL = 1.0 - Om
        for j, z in enumerate(_Z_PAN):
            val, _ = _quad(lambda zp: 1.0/np.sqrt(Om*(1+zp)**3 + OmL),
                           0, z, limit=50)
            _INTEGRAL_PAN[i, j] = val
    print(f"listo ({n_om}×{n_z} puntos)")

def log_likelihood_pantheon(Om, H0):
    """
    Log-likelihood de Pantheon+ con marginalización analítica sobre M_abs.
    Usa integrales precalculadas para velocidad.
    Implementa chi2_eff = A - B^2/C (Goliath et al. 2001).
    """
    if _PANTHEON_RAW is None or _INTEGRAL_PAN is None:
        return 0.0

    # [FIX 2] Interpolacion LINEAL en Om entre las dos filas vecinas de la malla
    # (antes era un lookup escalonado con searchsorted, sesgado hacia arriba).
    g    = _OM_GRID_PAN
    Om_c = np.clip(Om, g[0], g[-1])
    k    = int(np.clip(np.searchsorted(g, Om_c) - 1, 0, len(g) - 2))
    t    = (Om_c - g[k]) / (g[k + 1] - g[k])
    I_vals = (1.0 - t) * _INTEGRAL_PAN[k] + t * _INTEGRAL_PAN[k + 1]

    # Distancia de luminosidad y módulo de distancia teórico
    dL    = (_C_LIGHT / H0) * (1 + _Z_PAN) * I_vals   # Mpc
    mu_th = 5.0 * np.log10(np.clip(dL, 1e-10, None)) + 25.0

    # Módulo observado y varianza
    mu_obs  = _PANTHEON_RAW['mb']
    sigma2  = _PANTHEON_RAW['dmb']**2

    # Delta = mu_obs - mu_th  (sin M_abs — se marginaliza)
    delta = mu_obs - mu_th

    # Marginalización analítica sobre M_abs
    A = np.sum(delta**2     / sigma2)
    B = np.sum(delta        / sigma2)
    C = np.sum(1.0          / sigma2)
    chi2_eff = A - B**2 / C

    return -0.5 * chi2_eff

_PANTHEON_DATA = _PANTHEON_RAW   # alias para compatibilidad con menú

# ── Variables globales activas (se fijan en _main) ────────────────────────────
z_obs     = _CC_DATA[:, 0]
H_obs     = _CC_DATA[:, 1]
sigma_obs = _CC_DATA[:, 2]
N_DATA    = len(z_obs)
DATASET   = "CC"   # "CC" | "Pantheon+" | "CC+Pantheon+"

# ΛCDM model — completely classical
def H_lcdm(z, Om, H0):
    """H(z) [km/s/Mpc] for flat ΛCDM with radiation."""
    OmL = 1.0 - Om - OMEGA_R0
    return H0 * np.sqrt(Om*(1+z)**3 + OMEGA_R0*(1+z)**4 + OmL)

# Log-posterior — completely classical
H0_MU, H0_SIG = 67.66,  0.42     # prior gaussiano H0 (Planck 2018)
OM_MU, OM_SIG = 0.3111, 0.0056   # prior gaussiano Om (Planck 2018)

# Variable global que controla el tipo de prior — se fija en __main__
PRIOR_TYPE = "flat"   # "flat" | "gaussian"

def log_posterior(Om, H0):           # ← CLASSICAL
    """
    Log-posterior para (Ωm, H0).
    Usa DATASET global para combinar likelihoods:
      'CC'           → solo Cosmic Chronometers H(z)
      'Pantheon+'    → solo SNe Ia módulo de distancia
      'CC+Pantheon+' → ambos combinados
    """
    if not (0.18 < Om < 0.50 and 60.0 < H0 < 82.0):
        return -np.inf

    # Prior
    if PRIOR_TYPE == "gaussian":
        lp  = -0.5 * ((Om - OM_MU) / OM_SIG)**2
        lp += -0.5 * ((H0 - H0_MU) / H0_SIG)**2
    else:
        lp = 0.0

    # Log-likelihood CC
    if DATASET in ("CC", "CC+Pantheon+"):
        Hm  = H_lcdm(z_obs, Om, H0)
        lp += -0.5 * np.sum(((H_obs - Hm) / sigma_obs)**2)

    # Log-likelihood Pantheon+
    if DATASET in ("Pantheon+", "CC+Pantheon+") and _PANTHEON_RAW is not None:
        lp += log_likelihood_pantheon(Om, H0)

    return lp


def chi2_at(Om, H0):
    """
    chi2 total en (Om,H0) para el DATASET activo. Retorna (chi2, n_data).
    Sirve para chi2_reducido, AIC y BIC en la tabla de resultados.
    """
    chi2, n = 0.0, 0
    if DATASET in ("CC", "CC+Pantheon+"):
        Hm    = H_lcdm(z_obs, Om, H0)
        chi2 += float(np.sum(((H_obs - Hm) / sigma_obs) ** 2))
        n    += len(z_obs)
    if DATASET in ("Pantheon+", "CC+Pantheon+") and _PANTHEON_RAW is not None:
        chi2 += -2.0 * log_likelihood_pantheon(Om, H0)   # loglike = -0.5 chi2_eff
        n    += len(_PANTHEON_RAW["z"])
    return chi2, n


def _autocorr_time_flat(x):
    """
    Tiempo de autocorrelacion integrado (para ESS en metodos MCMC).
    Usa FFT para correlacion  ->  O(N log N) en lugar de O(N^2).
    Para N=160,000 muestras (20k pasos x 8 cadenas) esto es ~250x mas rapido
    que np.correlate(mode='full'), evitando el cuelgue al calcular la tabla.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 5:
        return 1.0
    x = x - x.mean()
    # Autocorrelacion via FFT (O(N log N))
    f   = np.fft.rfft(x, n=2 * n)
    acf = np.fft.irfft(f * np.conj(f))[:n].real
    if acf[0] <= 0:
        return 1.0
    acf = acf / acf[0]
    w = int(np.argmax(acf < 0.05))
    if w == 0:
        w = n // 4
    return float(max(1.0, 1 + 2 * np.sum(acf[1:max(w, 2)])))

def build_proposal_circuit(n_qubits: int, n_layers: int = 3) -> QuantumCircuit:
    """
    ── QUANTUM ──
    Build the proposal circuit from Sarracino et al. (2025), Fig. 1.

    Architecture:
      H on all qubits                          (uniform superposition)
      × n_layers of:
        RY(φ) RZ(φ) on each qubit             (arbitrary single-qubit rotation)
        CRY(φ) chain q0→q1→...→q_{n-1}       (entanglement)
      H on all qubits                          (final Hadamard)

    This circuit has NO fixed parameters — it is run fresh every step,
    producing a different statevector each time due to random parameter
    initialisation. That randomness IS the quantum proposal.

    Parameters
    ----------
    n_qubits : int   number of qubits = ⌈log₂(d)⌉, d = parameter dimensions
    n_layers : int   circuit depth (more layers → richer proposals)

    Returns
    -------
    qc : QuantumCircuit with NO measurements (statevector read directly)
    """
    n_params = n_layers * n_qubits * 2 + n_layers * (n_qubits - 1) + n_qubits
    phi = ParameterVector('φ', n_params)
    qc  = QuantumCircuit(n_qubits)

    qc.h(range(n_qubits))   # opening Hadamard layer

    idx = 0
    for _ in range(n_layers):
        # Single-qubit rotations
        for q in range(n_qubits):
            qc.ry(phi[idx], q); idx += 1
            qc.rz(phi[idx], q); idx += 1
        # Entangling CRY chain
        for q in range(n_qubits - 1):
            qc.cry(phi[idx], q, q+1); idx += 1

    qc.h(range(n_qubits))   # closing Hadamard layer

    # Final RY layer (matches paper Fig. 1 structure)
    for q in range(n_qubits):
        qc.ry(phi[idx], q); idx += 1

    return qc

def get_statevector(qc: QuantumCircuit,
                    phi_values: np.ndarray,
                    sim: AerSimulator) -> np.ndarray:
    """
    ── QUANTUM ──
    Run the circuit with given angles and return the complex statevector.
    Uses exact statevector simulation (no shot noise).
    """
    bound = qc.assign_parameters(dict(zip(qc.parameters, phi_values)))
    bound.save_statevector()
    result = sim.run(transpile(bound, sim)).result()
    return np.array(result.get_statevector())


# =============================================================================
# 3. QMCMC — Quantum Markov Chain Monte Carlo
# =============================================================================
#
# KEY IDEA: Replace the classical Gaussian random-walk proposal
#   θ_new = θ_old + N(0, σ²)
# with a quantum-circuit proposal
#   θ_new = θ_old + step_size × Re(v) × f(Im(v))
# where v is the statevector of a freshly-run quantum circuit.
#
# Every step the circuit is run with NEW random angles → each proposal
# is independent of the chain history. This is the quantum advantage:
# classical random walks have memory (correlations); this does not.
#
# What is quantum:  the PROPOSAL STEP (section 3.1)
# What is classical: the ACCEPTANCE (section 3.2), convergence diagnostics

# =============================================================================
# 3. CLASSICAL MCMC — emcee-style Metropolis-Hastings
# =============================================================================

class ClassicalMCMC:
    """
    MCMC clásico con propuesta gaussiana adaptativa.
    Equivalente directo del QMCMC pero sin ningún componente cuántico.
    Útil para comparar resultados y tiempos de convergencia.
    """

    def __init__(self,
                 n_chains:    int   = 8,
                 step_size:   float = 0.5,
                 n_burn:      int   = 500,
                 check_every: int   = 200):
        self.n_chains    = n_chains
        self.step_size   = step_size
        self.n_burn      = n_burn
        self.check_every = check_every

    def _propose(self, theta: np.ndarray) -> np.ndarray:
        """Propuesta gaussiana clásica con escalas adaptadas por parámetro."""
        scales = np.array([0.015, 0.6])   # Ωm tiene rango ~0.1, H0 ~10
        return theta + RNG.normal(0, scales)

    def _accept(self, lp_old: float, lp_new: float) -> bool:
        """Criterio de Metropolis-Hastings."""
        return np.log(RNG.uniform()) < (lp_new - lp_old)

    def _gelman_rubin(self, chains: np.ndarray) -> float:
        M, N   = chains.shape
        mu_j   = chains.mean(axis=1)
        mu_bar = mu_j.mean()
        B      = N * np.var(mu_j, ddof=1)
        W      = np.mean(np.var(chains, axis=1, ddof=1))
        var_hat = (1 - 1/N)*W + B/N
        return float(np.sqrt(var_hat / W)) if W > 1e-12 else np.nan

    def _autocorr_time(self, x: np.ndarray) -> float:
        n   = len(x)
        x   = x - x.mean()
        acf = np.correlate(x, x, mode='full')[n-1:]
        acf = acf / acf[0]
        for i, a in enumerate(acf):
            if a < 0.05:
                return max(1.0, float(2*i - 1))
        return float(n)

    def run(self, n_steps: int = 3000):
        print(f"\n── Classical MCMC ({self.n_chains} chains | "
              f"burn-in: {self.n_burn} | sampling: {n_steps}) ──")
        d = 2

        theta = np.column_stack([
            RNG.uniform(0.29, 0.33, self.n_chains),   # Ωm cerca de 0.31
            RNG.uniform(68.0, 72.0, self.n_chains),   # H0 cerca de 70
        ])
        log_p        = np.array([log_posterior(*t) for t in theta])
        burn_chains  = np.zeros((self.n_chains, self.n_burn, d))
        post_chains  = np.zeros((self.n_chains, n_steps, d))
        accept_count = np.zeros(self.n_chains)

        t0    = time.time()
        total = self.n_burn + n_steps
        pbar  = tqdm(total=total, desc="Classical MCMC [burn-in]",
                     bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} "
                                "[{elapsed}<{remaining}, {rate_fmt}]",
                     ncols=80)

        # Burn-in
        for step in range(self.n_burn):
            for c in range(self.n_chains):
                prop  = self._propose(theta[c])
                lp_p  = log_posterior(*prop)
                if np.isfinite(lp_p) and self._accept(log_p[c], lp_p):
                    theta[c] = prop
                    log_p[c] = lp_p
                burn_chains[c, step] = theta[c]
            pbar.update(1)

        # Sampling
        pbar.set_description("Classical MCMC [sampling]")
        converged = False
        rhat_hist = []

        for step in range(n_steps):
            for c in range(self.n_chains):
                prop  = self._propose(theta[c])
                lp_p  = log_posterior(*prop)
                if np.isfinite(lp_p) and self._accept(log_p[c], lp_p):
                    theta[c]        = prop
                    log_p[c]        = lp_p
                    accept_count[c] += 1
                post_chains[c, step] = theta[c]
            pbar.update(1)

            if (step + 1) % self.check_every == 0 and step > 50:
                rhat = max(
                    self._gelman_rubin(post_chains[:, :step+1, 0]),
                    self._gelman_rubin(post_chains[:, :step+1, 1])
                )
                rhat_hist.append(rhat)
                tau = max(
                    np.mean([self._autocorr_time(post_chains[c, :step+1, 0])
                             for c in range(self.n_chains)]),
                    np.mean([self._autocorr_time(post_chains[c, :step+1, 1])
                             for c in range(self.n_chains)])
                )
                acc = accept_count.mean() / (step + 1)
                tqdm.write(
                    f"  ↳ step {step+1:5d}/{n_steps} | "
                    f"R̂-1={rhat-1:.4f} | τ={tau:.1f} | acc={acc:.2f}"
                    + (" ✓ converged" if rhat-1 < 0.05 and step > 50*tau else "")
                )
                if rhat - 1 < 0.05 and step > 50*tau:
                    converged   = True
                    post_chains = post_chains[:, :step+1, :]
                    pbar.update(n_steps - step - 1)
                    break

        pbar.close()
        elapsed   = time.time() - t0
        flat      = post_chains.reshape(-1, d)
        mu        = flat.mean(axis=0)
        std       = flat.std(axis=0)

        print(f"\n  Classical MCMC results:")
        print(f"    Ωm = {mu[0]:.4f} ± {std[0]:.4f}")
        print(f"    H0 = {mu[1]:.4f} ± {std[1]:.4f}")
        print(f"    Converged: {converged}  |  Time: {elapsed:.1f}s")

        return post_chains, {
            'flat':      flat,
            'mu':        mu,
            'std':       std,
            'converged': converged,
            'elapsed':   elapsed,
            'rhat_hist': rhat_hist,
            'label':     'Classical MCMC',
        }


# =============================================================================
# 4. CLASSICAL VARIATIONAL INFERENCE (VI)
# =============================================================================

class ClassicalVI:
    """
    Inferencia variacional clásica con familia gaussiana diagonal.
    Minimiza KL(q_φ ‖ P_target) donde q_φ = N(μ, diag(σ²)).
    Equivalente directo del QVMC pero sin circuitos cuánticos.

    Limitación: solo puede representar distribuciones gaussianas.
    El QVMC puede representar cualquier forma.
    """

    PARAM_NAMES  = ["Ωm", "H0"]
    PARAM_RANGES = [(0.20, 0.42), (64.0, 78.0)]
    N_PARAMS_PHY = 2

    def __init__(self,
                 n_qubits_per_param: int = 3):
        """
        n_qubits_per_param : misma resolución de grilla que el QVMC
                             para comparación justa.
        """
        self.nqpp     = n_qubits_per_param
        self.n_grid   = 2**n_qubits_per_param
        self.n_states = self.n_grid ** self.N_PARAMS_PHY
        self.grids    = [np.linspace(lo, hi, self.n_grid)
                         for lo, hi in self.PARAM_RANGES]

        print(f"Classical VI initialised")
        print(f"  grid points = {self.n_grid} per param")
        print(f"  total states = {self.n_states}")

    def _build_grid(self):
        """Construye la grilla 2D de todos los (Ωm, H0) en orden correcto."""
        points = []
        for om in self.grids[0]:
            for h0 in self.grids[1]:
                points.append([om, h0])
        pts = np.array(points)
        return pts[:, 0], pts[:, 1]

    def build_target(self) -> np.ndarray:
        """Evalúa el posterior en toda la grilla — igual que QVMC."""
        print(f"\n  Building target posterior ({self.n_states} grid points)...")
        Om_grid, H0_grid = self._build_grid()
        log_p = np.array([log_posterior(Om, H0)
                          for Om, H0 in tqdm(zip(Om_grid, H0_grid),
                                             total=self.n_states,
                                             desc="Target Posterior")])
        valid          = np.isfinite(log_p)
        log_p[~valid]  = -np.inf
        log_p[valid]  -= log_p[valid].max()
        P              = np.zeros(self.n_states)
        P[valid]       = np.exp(log_p[valid])
        P             /= P.sum()
        print(f"  Valid states: {valid.sum()} / {self.n_states}")
        # Muestra el máximo de la grilla para verificar
        best_idx = np.argmax(P)
        print(f"  MAP estimate: Ωm={Om_grid[best_idx]:.4f}, "
              f"H0={H0_grid[best_idx]:.4f}")
        return P, Om_grid, H0_grid

    def run(self, max_iter: int = 500):
        """
        Minimiza KL(q ‖ P) donde q es gaussiana diagonal.
        Parámetros variacionales: φ = (μ_Ωm, μ_H0, log_σ_Ωm, log_σ_H0)
        """
        t0 = time.time()
        P_target, Om_grid, H0_grid = self.build_target()
        points = np.column_stack([Om_grid, H0_grid])

        # Inicializar en el MAP de la grilla
        best_idx = np.argmax(P_target)
        mu0  = np.array([Om_grid[best_idx], H0_grid[best_idx]])
        sig0 = np.array([0.02, 1.5])
        phi0 = np.concatenate([mu0, np.log(sig0)])

        print(f"  MAP inicial: Ωm={mu0[0]:.4f}, H0={mu0[1]:.4f}")

        history = []
        it      = [0]
        eps     = 1e-10

        # Rangos físicos para clipping de mu
        om_lo, om_hi = self.PARAM_RANGES[0]
        h0_lo, h0_hi = self.PARAM_RANGES[1]

        pbar = tqdm(total=max_iter, desc="Classical VI [training]",
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} "
                               "[{elapsed}<{remaining}, {rate_fmt}]",
                    ncols=80)

        def kl_gaussian(phi):
            mu  = phi[:2]
            sig = np.exp(np.clip(phi[2:], -4, 2))  # σ entre e^-4 y e^2

            # Penalidad suave en lugar de salto duro a 1e6
            # Esto evita el pico que domina la gráfica de KL
            penalty = 0.0
            if mu[0] < om_lo: penalty += 1e4 * (om_lo - mu[0])**2
            if mu[0] > om_hi: penalty += 1e4 * (mu[0] - om_hi)**2
            if mu[1] < h0_lo: penalty += 1e4 * (h0_lo - mu[1])**2
            if mu[1] > h0_hi: penalty += 1e4 * (mu[1] - h0_hi)**2
            if penalty > 0:
                kl_val = penalty
                history.append(kl_val)
                pbar.update(1)
                it[0] += 1
                return kl_val

            # Evalúa gaussiana en cada punto de la grilla
            diff  = points - mu
            expon = -0.5 * np.sum((diff / sig)**2, axis=1)
            expon -= expon.max()
            Q = np.exp(expon)

            # [FIX 1] renormalizar Q y P sobre el soporte P>eps  ->  KL >= 0
            # (antes ni Q ni P se renormalizaban tras enmascarar, lo que
            #  violaba la desigualdad de Gibbs y producia KL negativo).
            mask = P_target > eps
            Qm   = np.clip(Q[mask], eps, None); Qm /= Qm.sum()
            Pm   = np.clip(P_target[mask], eps, None); Pm /= Pm.sum()
            kl   = float(np.sum(Qm * np.log(Qm / Pm)))

            history.append(kl)
            pbar.update(1)
            if it[0] % 50 == 0:
                tqdm.write(f"  ↳ iter {it[0]:4d}/{max_iter}  "
                           f"KL={kl:.4f}  Ωm={mu[0]:.4f}  H0={mu[1]:.4f}  "
                           f"σΩm={sig[0]:.4f}  σH0={sig[1]:.4f}")
            it[0] += 1
            return kl

        # L-BFGS-B con bounds explícitos — más robusto que Nelder-Mead para VI
        bounds_opt = [
            (om_lo, om_hi),       # μ_Ωm
            (h0_lo, h0_hi),       # μ_H0
            (-4.0,  2.0),         # log σ_Ωm
            (-4.0,  2.0),         # log σ_H0
        ]
        res = minimize(kl_gaussian, phi0, method='L-BFGS-B',
                       bounds=bounds_opt,
                       options={'maxiter': max_iter * 10,
                                'ftol': 1e-9,
                                'gtol': 1e-6})
        pbar.close()

        mu_opt  = res.x[:2]
        sig_opt = np.exp(np.clip(res.x[2:], -4, 2))

        # Muestrear del gaussiano óptimo
        n_samples = 3000
        samples   = RNG.multivariate_normal(
            mu_opt,
            np.diag(sig_opt**2),
            size=n_samples
        )
        # Filtrar fuera de bounds
        mask    = ((samples[:,0] > 0.20) & (samples[:,0] < 0.45) &
                   (samples[:,1] > 60.0) & (samples[:,1] < 80.0))
        samples = samples[mask]
        weights = np.ones(len(samples)) / len(samples)

        elapsed = time.time() - t0
        print(f"\n  Classical VI results:")
        print(f"    Ωm = {mu_opt[0]:.4f} ± {sig_opt[0]:.4f}")
        print(f"    H0 = {mu_opt[1]:.4f} ± {sig_opt[1]:.4f}")
        print(f"    Final KL = {res.fun:.6f}  |  Time: {elapsed:.1f}s")

        return {
            'S_all':   samples,
            'W_all':   weights,
            'mu':      mu_opt,
            'std':     sig_opt,
            'history': history,
            'elapsed': elapsed,
            'label':   'Classical VI',
        }


class QMCMC:
    """
    Quantum MCMC for ΛCDM parameter estimation.
    Implements Algorithm 1 from Sarracino et al. (2025).
    """

    def __init__(self,
                 n_chains:    int   = 8,
                 step_size:   float = 0.05,
                 n_burn:      int   = 500,
                 n_layers:    int   = 3,
                 check_every: int   = 200):
        """
        Parameters
        ----------
        n_chains    : number of parallel chains
        step_size   : initial step size i (Eq. 7 of paper)
        n_burn      : burn-in steps (discarded)
        n_layers    : depth of quantum proposal circuit
        check_every : how often to compute convergence diagnostics
        """
        self.n_chains    = n_chains
        self.step_size   = step_size
        self.n_burn      = n_burn
        self.n_layers    = n_layers
        self.check_every = check_every

        # ΛCDM has d=2 parameters → n_qubits = ⌈log₂(2)⌉ = 1
        # But the paper uses ⌈log₂(d)⌉ with minimum 1.
        # For d=2 a single qubit gives a 2-component statevector.
        # We use 2 qubits so the statevector has 4 real + 4 imag components,
        # giving richer proposals for 2 parameters.
        self.n_qubits    = 2
        self.n_params_qc = (n_layers * self.n_qubits * 2
                           + n_layers * (self.n_qubits - 1)
                           + self.n_qubits)

        # Build circuit template (parameters filled fresh each step)
        self.qc  = build_proposal_circuit(self.n_qubits, n_layers)
        self.sim = AerSimulator(method='statevector')
        # Transpila la plantilla UNA sola vez (acelera mucho 2000+ pasos).
        # No cambia la fisica: solo evita re-transpilar en cada propuesta.
        self._qc_t = transpile(self.qc, self.sim)

        print(f"QMCMC initialised")
        print(f"  n_chains    = {n_chains}")
        print(f"  n_qubits    = {self.n_qubits}  (log₂(d) = log₂(2) = 1, using 2 for richness)")
        print(f"  step_size   = {step_size}")
        print(f"  circuit depth (decomposed) = {self.qc.decompose().depth()}")

    # ── 3.1  QUANTUM PROPOSAL ──────────────────────────────────────────────

    def _quantum_proposal(self, theta_current: np.ndarray) -> np.ndarray:
        """
        ── QUANTUM ──
        Propose θ_new using Eq. 7 of Sarracino et al.:
          s = i · Re(v) · f(Im(v))
          θ_new = θ_current + s[:d]

        The circuit angles are sampled fresh every call (uniform in [0, 2π]).
        This means every proposal is independent of the chain history.
        """
        # Random angles for this step (NOT the same as circuit training)
        phi = RNG.uniform(0, 2*np.pi, self.n_params_qc)

        # ── QUANTUM: run circuit, get statevector ──
        # ── QUANTUM: corre el circuito ya transpilado, lee el statevector ──
        bound = self._qc_t.assign_parameters(dict(zip(self._qc_t.parameters, phi)))
        bound.save_statevector()
        sv = np.array(self.sim.run(bound).result().get_statevector())

        # Extract real and imaginary parts (Eq. 7)
        re = np.real(sv)     # shape (2^n_qubits,)
        im = np.imag(sv)

        # f(Im(v)): step function — multiply step size if Im > 0, divide if Im < 0
        # This gives the proposal variable length depending on quantum state
        f_im = np.where(im >= 0, 1.0, -1.0)

        # Shift vector (take first d=2 components, one per parameter)
        # Escalas físicas adaptadas por parámetro
        # Target acceptance ~0.23-0.45; acc=0.96 → escalas muy pequeñas
        param_scales = np.array([0.15, 6.0])   # 10x más grande que antes
        shift = self.step_size * re[:2] * f_im[:2] * param_scales

        return theta_current + shift

    # ── 3.2  CLASSICAL ACCEPTANCE ─────────────────────────────────────────

    @staticmethod
    def _accept(log_p_current: float,
                log_p_proposed: float) -> bool:
        """
        ── CLASSICAL ──
        Standard Metropolis-Hastings acceptance rule:
          accept if log(U) < log_p_proposed - log_p_current
        """
        log_alpha = log_p_proposed - log_p_current
        return np.log(RNG.uniform()) < log_alpha

    # ── 3.3  CONVERGENCE DIAGNOSTICS ─────────────────────────────────────

    @staticmethod
    def _autocorrelation_time(chain: np.ndarray) -> float:
        """
        ── CLASSICAL ──
        Integrated autocorrelation time τ via the standard windowed estimator.
        Convergence criterion: N > 50·τ (Goodman & Weare 2010).
        """
        x   = chain - chain.mean()
        N   = len(x)
        acf = np.correlate(x, x, mode='full')[N-1:]
        acf = acf / acf[0]
        # Sum until the window where acf < 0.05
        window = np.argmax(acf < 0.05)
        if window == 0:
            window = N // 4
        return float(1 + 2*np.sum(acf[1:window]))

    @staticmethod
    def _gelman_rubin(chains: np.ndarray) -> float:
        """
        ── CLASSICAL ──
        Gelman-Rubin R̂ for a single parameter across M chains.
        chains : shape (M, N)
        Returns R̂. Convergence if R̂ - 1 < 0.05.
        """
        M, N     = chains.shape
        mu_j     = chains.mean(axis=1)          # per-chain mean
        mu_bar   = mu_j.mean()                  # grand mean
        B        = N * np.var(mu_j, ddof=1)     # between-chain var
        W        = np.mean(np.var(chains, axis=1, ddof=1))  # within-chain var
        var_hat  = (1 - 1/N)*W + B/N
        return float(np.sqrt(var_hat / W)) if W > 1e-12 else np.nan

    # ── 3.4  MAIN SAMPLER ─────────────────────────────────────────────────

    def run(self, n_steps: int = 3000):
        """
        Run the QMCMC.

        Returns
        -------
        samples : ndarray shape (n_chains, n_steps, 2)  post-burn-in chains
        info    : dict with acceptance rates, τ, R̂
        """
        print(f"\n── QMCMC ({self.n_chains} chains | burn-in: {self.n_burn} | sampling: {n_steps}) ──")
        d = 2   # Ωm, H0

        # Initialise chains near expected posterior
        theta = np.column_stack([
            RNG.uniform(0.29, 0.33, self.n_chains),   # Ωm cerca de 0.31
            RNG.uniform(68.0, 72.0, self.n_chains),   # H0 cerca de 70
        ])
        log_p = np.array([log_posterior(*t) for t in theta])

        # Storage
        burn_chains  = np.zeros((self.n_chains, self.n_burn,  d))
        post_chains  = np.zeros((self.n_chains, n_steps, d))
        accept_count = np.zeros(self.n_chains)

        t0 = time.time()

        # Barra de progreso única que cubre burn-in + sampling
        total_steps = self.n_burn + n_steps
        pbar = tqdm(total=total_steps, desc="QMCMC",
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} "
                               "[{elapsed}<{remaining}, {rate_fmt}]",
                    ncols=80)

        # ── Burn-in ───────────────────────────────────────────────────────
        pbar.set_description("QMCMC [burn-in]")
        for step in range(self.n_burn):
            for c in range(self.n_chains):
                theta_prop = self._quantum_proposal(theta[c])
                lp_prop    = log_posterior(*theta_prop)
                if np.isfinite(lp_prop) and self._accept(log_p[c], lp_prop):
                    theta[c] = theta_prop
                    log_p[c] = lp_prop
                burn_chains[c, step] = theta[c]
            pbar.update(1)

        # ── Post-burn-in sampling ─────────────────────────────────────────
        pbar.set_description("QMCMC [sampling]")
        converged = False
        rhat_hist = []

        for step in range(n_steps):
            for c in range(self.n_chains):
                theta_prop = self._quantum_proposal(theta[c])
                lp_prop    = log_posterior(*theta_prop)
                if np.isfinite(lp_prop) and self._accept(log_p[c], lp_prop):
                    theta[c]       = theta_prop
                    log_p[c]       = lp_prop
                    accept_count[c] += 1
                post_chains[c, step] = theta[c]
            pbar.update(1)

            # Convergence diagnostics every check_every steps
            if (step + 1) % self.check_every == 0 and step > 50:
                rhat_om = self._gelman_rubin(post_chains[:, :step+1, 0])
                rhat_h0 = self._gelman_rubin(post_chains[:, :step+1, 1])
                rhat    = max(rhat_om, rhat_h0)
                rhat_hist.append(rhat)

                tau_om = np.mean([self._autocorrelation_time(
                                  post_chains[c, :step+1, 0])
                                  for c in range(self.n_chains)])
                tau_h0 = np.mean([self._autocorrelation_time(
                                  post_chains[c, :step+1, 1])
                                  for c in range(self.n_chains)])
                tau = max(tau_om, tau_h0)
                acc = accept_count.mean() / (step + 1)

                tqdm.write(
                    f"  ↳ step {step+1:5d}/{n_steps} | "
                    f"R̂-1={rhat-1:.4f} | τ={tau:.1f} | acc={acc:.2f}"
                    + (" ✓ converged" if rhat-1 < 0.05 and step > 50*tau else "")
                )

                if rhat - 1 < 0.05 and step > 50*tau:
                    converged   = True
                    post_chains = post_chains[:, :step+1, :]
                    pbar.update(n_steps - step - 1)  # cierra la barra
                    break

        pbar.close()

        elapsed = time.time() - t0
        acc_rates = accept_count / n_steps

        print(f"\n  Elapsed       : {elapsed:.1f} s")
        print(f"  Acceptance    : {acc_rates.mean():.3f} ± {acc_rates.std():.3f}")
        print(f"  Converged     : {converged}")

        flat = post_chains.reshape(-1, d)
        self._print_summary(flat)

        return post_chains, {
            'flat': flat,
            'acceptance': acc_rates,
            'rhat_history': rhat_hist,
            'elapsed': elapsed,
            'converged': converged,
        }

    @staticmethod
    def _print_summary(flat):
        names = ["Ωm", "H0"]
        print(f"\n  Posterior summary (QMCMC):")
        for i, name in enumerate(names):
            q = np.percentile(flat[:, i], [16, 50, 84])
            print(f"  {name:4s}: {q[1]:.4f} +{q[2]-q[1]:.4f} -{q[1]-q[0]:.4f}")


# =============================================================================
# 4. QVMC — Quantum Variational Monte Carlo
# =============================================================================
#
# KEY IDEA: Encode the posterior directly into a quantum circuit.
# Train the circuit so that measuring it gives posterior samples.
#
# What is quantum:  the CIRCUIT that REPRESENTS the posterior
# What is classical: building the target grid, training (COBYLA),
#                    evaluating KL divergence
#
# The circuit has n_qubits = N_QUBITS_PER_PARAM × d qubits.
# Each parameter is discretised into 2^N_QUBITS_PER_PARAM grid values.
# The circuit amplitude squared |⟨x|ψ⟩|² approximates P(θ_x | data).

class QVMC:
    """
    Quantum Variational Monte Carlo for ΛCDM parameter estimation.
    """

    # Parameter grid definition
    PARAM_NAMES  = ["Ωm", "H0"]
    PARAM_RANGES = [(0.20, 0.42), (64.0, 78.0)]
    N_PARAMS_PHY = 2

    def __init__(self,
                 n_qubits_per_param: int = 3,
                 n_layers:           int = 3,
                 n_shots:            int = 4000):
        """
        Parameters
        ----------
        n_qubits_per_param : grid resolution per parameter
                             2 → 4 values/param (fast, coarse)
                             3 → 8 values/param (recommended)
                             4 → 16 values/param (slow but fine)
        n_layers           : ansatz depth
        n_shots            : shots for posterior sampling after training
        """
        self.nqpp    = n_qubits_per_param
        self.n_qubits = self.N_PARAMS_PHY * n_qubits_per_param
        self.n_grid   = 2**n_qubits_per_param
        self.n_states = 2**self.n_qubits
        self.n_layers = n_layers
        self.n_shots  = n_shots

        self.grids = [np.linspace(lo, hi, self.n_grid)
                      for lo, hi in self.PARAM_RANGES]

        self.sim = AerSimulator(method='statevector')

        print(f"QVMC initialised")
        print(f"  n_qubits_per_param = {n_qubits_per_param}")
        print(f"  total qubits       = {self.n_qubits}")
        print(f"  grid points        = {self.n_grid} per param")
        print(f"  total states       = {self.n_states}")

    # ── 4.1  GRID ENCODING ────────────────────────────────────────────────

    def _decode(self, bitstring: str) -> np.ndarray:
        """
        ── CLASSICAL ──
        Decode a bitstring → (Ωm, H0) parameter vector.
        Qiskit returns bitstrings reversed; we correct for it.
        """
        bits = bitstring[::-1]
        theta = np.zeros(self.N_PARAMS_PHY)
        for i in range(self.N_PARAMS_PHY):
            chunk    = bits[i*self.nqpp : (i+1)*self.nqpp]
            theta[i] = self.grids[i][int(chunk, 2)]
        return theta

    # ── 4.2  BUILD TARGET POSTERIOR ──────────────────────────────────────

    def build_target(self) -> np.ndarray:
        """
        ── CLASSICAL ──
        Evaluate log_posterior at every grid point and normalise.
        This is the distribution the circuit will learn to represent.

        Returns P_target : ndarray shape (n_states,)
        """

        print(f"\n  Building target posterior ({self.n_states} grid points)...")
        log_p = np.full(self.n_states, -np.inf)

        for idx in tqdm(range(self.n_states), desc="Target Posterior"):
            bs        = format(idx, f'0{self.n_qubits}b')
            Om, H0    = self._decode(bs)
            lp        = log_posterior(Om, H0)
            if np.isfinite(lp):
                log_p[idx] = lp

        valid = np.isfinite(log_p)
        print(f"  Valid states : {valid.sum()} / {self.n_states}")
        log_p[valid] -= np.max(log_p[valid])   # numerical stability

        P = np.zeros(self.n_states)
        P[valid] = np.exp(log_p[valid])
        P /= P.sum()

        # Show MAP estimate
        best_bs = format(np.argmax(P), f'0{self.n_qubits}b')
        Om_map, H0_map = self._decode(best_bs)
        print(f"  MAP : Ωm={Om_map:.4f}, H0={H0_map:.4f}")
        return P

    # ── 4.3  QUANTUM ANSATZ ───────────────────────────────────────────────

    def _build_ansatz(self) -> tuple:
        """
        ── QUANTUM ──
        Hardware-efficient ansatz for QVMC.
        Architecture per layer:
          RY(φ) RZ(φ) on each qubit        → amplitude shaping
          CNOT chain with wrap-around       → entanglement
        Final layer: RY only.

        The circuit is TRAINED (φ optimised to minimise KL vs posterior).
        After training, |⟨x|ψ(φ*)⟩|² ≈ P_target(x) for all x.
        """
        n   = self.n_qubits
        n_p = self.n_layers * n * 2 + n
        phi = ParameterVector('φ', n_p)
        qc  = QuantumCircuit(n)

        qc.h(range(n))   # uniform superposition

        idx = 0
        for _ in range(self.n_layers):
            for q in range(n):
                qc.ry(phi[idx], q); idx += 1
                qc.rz(phi[idx], q); idx += 1
            for q in range(n-1):
                qc.cx(q, q+1)
            qc.cx(n-1, 0)   # wrap-around entanglement

        for q in range(n):
            qc.ry(phi[idx], q); idx += 1

        qc.measure_all()
        return qc, n_p

    # ── 4.4  CLASSICAL TRAINING ───────────────────────────────────────────

    def _kl_divergence(self,
                       phi: np.ndarray,
                       qc:  QuantumCircuit,
                       P_target: np.ndarray,
                       eps: float = 1e-12) -> float:
        """
        ── CLASSICAL cost, evaluated on QUANTUM circuit output ──
        KL(q_φ ‖ P_target) = Σ_x q_φ(x) log[q_φ(x)/P_target(x)]

        q_φ(x) = |⟨x|ψ(φ)⟩|² from exact statevector simulation.
        """
        bound  = qc.assign_parameters(dict(zip(qc.parameters, phi)))
        sv_qc  = bound.remove_final_measurements(inplace=False)
        sv_qc.save_statevector()
        sv     = self.sim.run(transpile(sv_qc, self.sim)).result().get_statevector()
        Q      = np.abs(np.array(sv))**2

        # [FIX 1] renormalizar Q y P sobre el soporte P>eps  ->  KL >= 0
        # (antes ni Q ni P se renormalizaban tras enmascarar, lo que violaba
        #  la desigualdad de Gibbs y producia un KL negativo durante COBYLA).
        mask = P_target > eps
        Qm   = np.clip(Q[mask], eps, None); Qm /= Qm.sum()
        Pm   = np.clip(P_target[mask], eps, None); Pm /= Pm.sum()
        return float(np.sum(Qm * np.log(Qm / Pm)))

    def train(self, P_target: np.ndarray,
              max_iter: int = 600) -> tuple:
        """
        ── CLASSICAL optimisation of QUANTUM circuit ──
        Minimise KL(q_φ ‖ P_target) over φ using COBYLA.

        COBYLA is gradient-free. Gradient-based training would use the
        parameter-shift rule (a quantum-native gradient), which is more
        efficient but more complex to implement. See roadmap at EOF.

        Returns (phi_optimal, qc, history)
        """
        print(f"\n  Training ansatz (COBYLA, max {max_iter} iter)...")
        qc, n_p = self._build_ansatz()
        phi0    = 0.1 * RNG.standard_normal(n_p)
        history = []
        it      = [0]

        pbar = tqdm(total=max_iter, desc="QVMC [training]",
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} "
                               "[{elapsed}<{remaining}, {rate_fmt}]",
                    ncols=80)

        def cost(phi):
            kl = self._kl_divergence(phi, qc, P_target)
            history.append(kl)
            pbar.update(1)
            if it[0] % 50 == 0:
                tqdm.write(f"  ↳ iter {it[0]:4d}/{max_iter}  KL = {kl:.6f}")
            it[0] += 1
            return kl

        t0  = time.time()
        res = minimize(cost, phi0, method='COBYLA',
                       options={'maxiter': max_iter, 'rhobeg': 0.3})
        
        pbar.close() 
        
        elapsed = time.time() - t0
        print(f"  Final KL : {res.fun:.6f}  |  elapsed : {elapsed:.1f} s")
        print(f"  Converged: {res.success}  |  iters   : {len(history)}")
        return res.x, qc, history

    # ── 4.5  QUANTUM SAMPLING ─────────────────────────────────────────────

    def sample(self, phi_opt: np.ndarray,
               qc: QuantumCircuit,
               n_chains: int = 4) -> tuple:
        """
        ── QUANTUM ──
        Draw posterior samples from the trained circuit.
        Each shot IS a sample: the quantum measurement collapses the
        superposition and returns a bitstring → (Ωm, H0).

        Uses multiple seeds (chains) for Gelman-Rubin compatibility.
        """
        print(f"\n  Sampling from trained circuit "
              f"({n_chains} chains × {self.n_shots} shots)...")
        bound  = qc.assign_parameters(dict(zip(qc.parameters, phi_opt)))
        all_chains = []

        for c in range(n_chains):
            counts = self.sim.run(
                transpile(bound, self.sim),
                shots=self.n_shots,
                seed_simulator=1000 + c*137
            ).result().get_counts()

            S, W = [], []
            for bs, cnt in counts.items():
                S.append(self._decode(bs))
                W.append(cnt)
            all_chains.append((np.array(S), np.array(W, dtype=float)/self.n_shots))
            print(f"  chain {c+1}: {len(S)} unique states sampled")

        return all_chains

    # ── 4.6  GELMAN-RUBIN FOR QVMC CHAINS ────────────────────────────────

    @staticmethod
    def gelman_rubin(chains_and_weights: list,
                     param_names: list) -> dict:
        """
        ── CLASSICAL ──
        Gelman-Rubin R̂ across QVMC chains (tests shot-noise stability).
        """
        M = len(chains_and_weights)
        n_p = chains_and_weights[0][0].shape[1]
        Rhat = {}
        print(f"\n  Gelman-Rubin R̂  ({M} chains):")
        for p in range(n_p):
            means, variances, ns = [], [], []
            for S, W in chains_and_weights:
                w  = W / W.sum()
                mu = np.average(S[:, p], weights=w)
                s2 = np.average((S[:, p]-mu)**2, weights=w)
                means.append(mu); variances.append(s2); ns.append(len(S))
            N_eff   = M / np.sum(1.0/np.array(ns))
            B       = N_eff * np.var(means, ddof=1)
            W_      = np.mean(variances)
            var_hat = (1-1/N_eff)*W_ + B/N_eff
            rhat    = float(np.sqrt(var_hat/W_)) if W_ > 1e-15 else np.nan
            name    = param_names[p]
            Rhat[name] = rhat
            status  = "✓" if rhat < 1.05 else "⚠"
            print(f"  {status} {name:4s}: R̂ = {rhat:.4f}")
        return Rhat

    # ── 4.7  FULL PIPELINE ────────────────────────────────────────────────

    def run(self, max_iter: int = 600, n_chains: int = 4):
        """Run the full QVMC pipeline: build target → train → sample."""
        t0               = time.time()
        P_target         = self.build_target()
        phi_opt, qc, hist = self.train(P_target, max_iter)
        chains           = self.sample(phi_opt, qc, n_chains)
        Rhat             = self.gelman_rubin(chains, self.PARAM_NAMES)

        # Merge chains for summary
        S_all = np.concatenate([s for s,_ in chains])
        W_all = np.concatenate([w for _,w in chains])
        W_all /= W_all.sum()
        elapsed = time.time() - t0
        print(f"\n  Posterior summary (QVMC):")
        for i, name in enumerate(self.PARAM_NAMES):
            mu  = np.average(S_all[:,i], weights=W_all)
            std = np.sqrt(np.average((S_all[:,i]-mu)**2, weights=W_all))
            print(f"  {name:4s}: {mu:.4f} ± {std:.4f}")
        print(f"  Total elapsed: {elapsed:.1f} s")

        return {
            'chains':   chains,
            'S_all':    S_all,
            'W_all':    W_all,
            'P_target': P_target,
            'phi_opt':  phi_opt,
            'history':  hist,
            'Rhat':     Rhat,
            'elapsed':  elapsed,
        }


# =============================================================================
# 5. PLOTS
# =============================================================================

def get_global_corner_range(active_models, get_samples_func, margin=0.05,
                             use_percentiles=True):
    """
    Calcula el rango global [min, max] para los ejes de los corner plots
    evaluando todas las muestras de todos los metodos activos.
    Asi todos los subplots comparten la misma escala y no se enciman.

    use_percentiles=True  ->  recorta outliers extremos (mas limpio).
    use_percentiles=False ->  usa min/max absoluto.
    """
    all_om, all_h0 = [], []
    for label, info, _ in active_models:
        S, _ = get_samples_func(info, label)
        all_om.append(S[:, 0])
        all_h0.append(S[:, 1])
    all_om = np.concatenate(all_om)
    all_h0 = np.concatenate(all_h0)
    if use_percentiles:
        om_min, om_max = np.percentile(all_om, [0.5, 99.5])
        h0_min, h0_max = np.percentile(all_h0, [0.5, 99.5])
    else:
        om_min, om_max = np.min(all_om), np.max(all_om)
        h0_min, h0_max = np.min(all_h0), np.max(all_h0)
    pad_om = (om_max - om_min) * margin
    pad_h0 = (h0_max - h0_min) * margin
    return [(om_min - pad_om, om_max + pad_om),
            (h0_min - pad_h0, h0_max + pad_h0)]

def plot_results(cmcmc_info=None, cvi_info=None,
                 qmcmc_info=None, qvmc_info=None,
                 nqpp=3, dataset_label="CC", run_cfg=None):
    """
    Genera 5 figuras + CSV + grafica de tabla:
      1. H(z) posterior predictive
      2. Marginales 1D
      3. Corner plots SEPARADOS (2x2 subfiguras)
      4. Corner plots SUPERPUESTOS (solo contornos, limpio)
      5. KL training curves
      + CSV y grafica de barras con resultados
    """
    import csv, os

    methods = [
        ('Classical MCMC', cmcmc_info, 'C0'),
        ('Classical VI',   cvi_info,   'C2'),
        ('QMCMC',          qmcmc_info, 'C1'),
        ('QVMC',           qvmc_info,  'C3'),
    ]
    active = [(lbl, info, col) for lbl, info, col in methods if info is not None]
    if not active:
        print("  no hay nada que graficar.")
        return

    n_qubits_total = 2 * nqpp
    n_states       = 4 ** nqpp
    tag = (f"nqpp={nqpp} | qubits={n_qubits_total} | "
           f"estados={n_states:,} | datos={dataset_label} | prior={PRIOR_TYPE}")

    z_plot = np.linspace(0.0, 2.4, 300)

    def get_samples(info, label):
        if "flat" in info:
            S = info["flat"]
            W = np.ones(len(S)) / len(S)
        else:
            S = info["S_all"]
            W = info["W_all"]
        W = np.array(W, dtype=np.float64)
        W = np.clip(W, 0, None)
        W = W / W.sum()
        return S, W

    def safe_choice(n, size, W):
        W = np.array(W, dtype=np.float64)
        W = np.clip(W, 0, None); W = W / W.sum()
        sz = min(size, n)
        return RNG.choice(n, size=sz, p=W, replace=(sz >= n))

    # =========================================================
    # FIG 1 — H(z)
    # =========================================================
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.errorbar(z_obs, H_obs, yerr=sigma_obs,
                 fmt=".k", capsize=3, ms=4, zorder=5,
                 label=f"Datos {dataset_label}")
    for label, info, color in active:
        S, W = get_samples(info, label)
        idx  = safe_choice(len(S), 60, W)
        for i in idx:
            ax1.plot(z_plot, H_lcdm(z_plot, S[i,0], S[i,1]),
                     color=color, alpha=0.06, lw=0.7)
        mu = np.average(S, weights=W, axis=0)
        ax1.plot(z_plot, H_lcdm(z_plot, mu[0], mu[1]),
                 color=color, lw=2.5, label=label)
    ax1.set_xlabel(r"$z$", fontsize=13)
    ax1.set_ylabel(r"$H(z)$ [km s$^{-1}$ Mpc$^{-1}$]", fontsize=13)
    ax1.set_title(r"$\Lambda$CDM — $H(z)$ posterior predictive" + f"\n{tag}", fontsize=10)
    ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("lcdm_hz_comparison.pdf", dpi=150, bbox_inches="tight")
    print("  Saved: lcdm_hz_comparison.pdf"); plt.show()

    # =========================================================
    # FIG 2 — Marginales
    # =========================================================
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4))
    for label, info, color in active:
        S, W = get_samples(info, label)
        axes2[0].hist(S[:,0], weights=W, bins=30, color=color,
                      alpha=0.45, density=True, label=label, edgecolor="white", lw=0.5)
        axes2[1].hist(S[:,1], weights=W, bins=30, color=color,
                      alpha=0.45, density=True, label=label, edgecolor="white", lw=0.5)
    axes2[0].axvline(0.3111, color="k",    ls="--", lw=1.5, label="Planck Om")
    axes2[1].axvline(73.04,  color="k",    ls="--", lw=1.5, label="SH0ES H0")
    axes2[1].axvline(67.66,  color="gray", ls=":",  lw=1.5, label="Planck H0")
    axes2[0].set_xlabel(r"$\Omega_m$", fontsize=13)
    axes2[1].set_xlabel(r"$H_0$ [km s$^{-1}$ Mpc$^{-1}$]", fontsize=13)
    for ax in axes2:
        ax.set_ylabel("Density", fontsize=12); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    fig2.suptitle(r"$\Lambda$CDM — Marginales" + f"\n{tag}", fontsize=11)
    plt.tight_layout()
    plt.savefig("lcdm_marginals_comparison.pdf", dpi=150, bbox_inches="tight")
    print("  Saved: lcdm_marginals_comparison.pdf"); plt.show()

    # =========================================================
    # PRE-CALCULO del rango unificado para corner plots
    # (evita que los subplots se encimen cuando los posteriors
    #  tienen escalas muy diferentes entre metodos)
    # =========================================================
    if HAS_CORNER:
        global_range = get_global_corner_range(active, get_samples)

    # =========================================================
    # FIG 3 — Corner SEPARADOS
    # =========================================================
    if HAS_CORNER:
        n_active = len(active)
        if   n_active == 1: nrows, ncols = 1, 1
        elif n_active == 2: nrows, ncols = 1, 2
        elif n_active == 3: nrows, ncols = 1, 3
        else:               nrows, ncols = 2, 2

        fig3    = plt.figure(figsize=(7*ncols, 7*nrows))
        subfigs = fig3.subfigures(nrows, ncols, wspace=0.10, hspace=0.15)
        if n_active == 1:
            subfigs = np.array([[subfigs]])
        elif nrows == 1:
            subfigs = np.array([np.atleast_1d(subfigs)])
        elif ncols == 1:
            subfigs = np.array([[sf] for sf in subfigs])

        for idx_m, (label, info, color) in enumerate(active):
            row = idx_m // ncols; col = idx_m % ncols
            sf  = subfigs[row, col]
            S, W = get_samples(info, label)
            mu   = np.average(S, weights=W, axis=0)
            std  = np.sqrt(np.average((S-mu)**2, weights=W, axis=0))
            corner.corner(S, weights=W,
                labels=[r"$\Omega_m$", r"$H_0$"],
                color=color, truth_color="k", truths=[0.3111, 67.66],
                show_titles=False,          # titulo en sf.suptitle (mas limpio)
                label_kwargs={"fontsize": 11},
                quantiles=[0.16, 0.50, 0.84],
                plot_contours=True, fill_contours=True, smooth=1.0,
                range=global_range,         # escala unificada entre metodos
                fig=sf)
            sf.suptitle(
                f"{label}\n"
                f"$\\Omega_m={mu[0]:.4f}\\pm{std[0]:.4f}$  "
                f"$H_0={mu[1]:.4f}\\pm{std[1]:.4f}$",
                fontsize=11, color=color, fontweight="bold", y=1.05)

        fig3.suptitle(
            r"$\Lambda$CDM — Corner plots separados" + f"\n{tag}",
            fontsize=12, fontweight="bold", y=1.05)
        plt.savefig("corner_separate.pdf", dpi=150, bbox_inches="tight")
        print("  Saved: corner_separate.pdf"); plt.show()

    # =========================================================
    # FIG 4 — Corner SUPERPUESTOS (solo contornos, sin fill)
    # =========================================================
    if HAS_CORNER:
        fig4     = None
        handles4 = []
        for label, info, color in active:
            S, W = get_samples(info, label)
            mu   = np.average(S, weights=W, axis=0)
            std  = np.sqrt(np.average((S-mu)**2, weights=W, axis=0))
            fig4 = corner.corner(
                S, weights=W,
                labels=[r"$\Omega_m$", r"$H_0$"],
                color=color, truth_color="k", truths=[0.3111, 67.66],
                show_titles=False, label_kwargs={"fontsize": 13},
                quantiles=[],
                plot_contours=True, fill_contours=False,
                smooth=1.5, no_fill_contours=True,
                contour_kwargs={"linewidths": 2.0},
                range=global_range,         # mantiene todas las elipses encuadradas
                fig=fig4)
            handles4.append(plt.Line2D([0],[0], color=color, lw=2.5,
                label=f"{label}: $\\Omega_m={mu[0]:.3f}\\pm{std[0]:.3f}$"
                      f"  $H_0={mu[1]:.2f}\\pm{std[1]:.2f}$"))
        handles4.append(plt.Line2D([0],[0], color="k", lw=1.5, ls="--",
            label="Planck 2018: $\\Omega_m=0.3111$, $H_0=67.66$"))
        caxes = fig4.get_axes()
        if len(caxes) >= 2:
            caxes[1].legend(handles=handles4, loc="center left", fontsize=9,
                framealpha=0.95, edgecolor="gray",
                title=f"nqpp={nqpp} | {n_qubits_total}q | {dataset_label}",
                title_fontsize=8)
            caxes[1].axis("off")
        fig4.suptitle(r"$\Lambda$CDM — Corner superpuestos" + f"\n{tag}",
            fontsize=12, fontweight="bold", y=1.02)
        plt.savefig("corner_overlay.pdf", dpi=150, bbox_inches="tight")
        print("  Saved: corner_overlay.pdf"); plt.show()

    # =========================================================
    # FIG 5 — KL training
    # =========================================================
    training = [(lbl, info, col) for lbl, info, col in active if "history" in info]
    if training:
        fig5, ax5 = plt.subplots(figsize=(9, 4))
        for label, info, color in training:
            hist = np.array(info["history"])
            hist_clean = np.where(np.abs(hist) > 100, np.nan, hist)
            ax5.plot(hist_clean, color=color, lw=1.5, label=label)
        ax5.set_xlabel("Iteracion", fontsize=12)
        ax5.set_ylabel("KL Divergencia", fontsize=12)
        ax5.set_title(f"Convergencia variacional\n{tag}", fontsize=11)
        ax5.legend(fontsize=10); ax5.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("lcdm_kl_training.pdf", dpi=150, bbox_inches="tight")
        print("  Saved: lcdm_kl_training.pdf"); plt.show()

    # =========================================================
    # TABLA — terminal + CSV + grafica de barras
    # =========================================================
    rows = []
    print("\n" + "="*82)
    print(f"  {'Metodo':<20} {'Om':>16} {'H0':>18} {'Tiempo':>10} {'Qubits':>8}")
    print("="*82)
    for label, info, _ in active:
        S, W  = get_samples(info, label)
        mu    = np.average(S, weights=W, axis=0)
        std   = np.sqrt(np.average((S-mu)**2, weights=W, axis=0))
        t     = info.get("elapsed", None)
        t_str = f"{t:.1f}s" if (t is not None and np.isfinite(t)) else "—"
        q_str = str(n_qubits_total) if label in ("QVMC","Classical VI") else "—"
        print(f"  {label:<20} {mu[0]:.4f} +/- {std[0]:.4f}  "
              f"{mu[1]:.4f} +/- {std[1]:.4f}  {t_str:>10}  {q_str:>8}")

        # ── Metricas adicionales (chi2, chi2_red, AIC, BIC, ESS, etc.) ─────
        chi2, n_data = chi2_at(mu[0], mu[1])
        k        = 2
        dof      = max(n_data - k, 1)
        chi2_red = chi2 / dof
        aic      = chi2 + 2 * k
        bic      = chi2 + k * np.log(max(n_data, 2))
        acc      = info.get("acceptance", None)
        if isinstance(acc, np.ndarray):
            acc = float(np.mean(acc))
        kl_fin   = info.get("history", None)
        kl_fin   = float(kl_fin[-1]) if (kl_fin is not None and len(kl_fin)) else ""
        rhat     = info.get("converged", "")
        if "flat" in info:
            tau_om = _autocorr_time_flat(S[:, 0])
            tau_h0 = _autocorr_time_flat(S[:, 1])
            ess_om = len(S) / max(tau_om, 1e-9)
            ess_h0 = len(S) / max(tau_h0, 1e-9)
        else:
            ess_om = ess_h0 = float(len(S))

        rows.append({"Method":label, "Om_mean":f"{mu[0]:.4f}",
            "Om_std":f"{std[0]:.4f}", "H0_mean":f"{mu[1]:.4f}",
            "H0_std":f"{std[1]:.4f}", "Time_s":t_str,
            "nqpp":str(nqpp) if label in ("QVMC","Classical VI") else "—",
            "n_qubits":str(n_qubits_total) if label in ("QVMC","Classical VI") else "—",
            "n_states":str(n_states) if label in ("QVMC","Classical VI") else "—",
            "chi2":f"{chi2:.4f}", "n_data":str(n_data),
            "chi2_red":f"{chi2_red:.4f}", "AIC":f"{aic:.4f}", "BIC":f"{bic:.4f}",
            "acceptance":(f"{acc:.4f}" if acc is not None else ""),
            "final_KL":(f"{kl_fin:.6f}" if kl_fin != "" else ""),
            "ESS_Om":f"{ess_om:.1f}", "ESS_H0":f"{ess_h0:.1f}",
            "dataset":dataset_label, "prior":PRIOR_TYPE})
    print("="*82)
    print(f"  Planck 2018:  Om = 0.3111 +/- 0.0056  |  H0 = 67.66 +/- 0.42")
    print("="*82)

    # CSV — acumula corridas (append)
    csv_fname = "resultados_config.csv"
    fieldnames = ["Method","Om_mean","Om_std","H0_mean","H0_std",
                  "Time_s","nqpp","n_qubits","n_states",
                  "chi2","n_data","chi2_red","AIC","BIC",
                  "acceptance","final_KL","ESS_Om","ESS_H0",
                  "dataset","prior"]
    write_header = not os.path.exists(csv_fname)
    with open(csv_fname, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header: w.writeheader()
        w.writerows(rows)
    print(f"\n  CSV guardado/actualizado: {csv_fname}")

    # Grafica de barras horizontales
    labels_t = [r["Method"] for r in rows]
    om_means = [float(r["Om_mean"]) for r in rows]
    om_stds  = [float(r["Om_std"])  for r in rows]
    h0_means = [float(r["H0_mean"]) for r in rows]
    h0_stds  = [float(r["H0_std"])  for r in rows]
    colors_t = [m[2] for m in methods if m[1] is not None]

    fig6, (ax_om, ax_h0) = plt.subplots(1, 2, figsize=(13, 4))
    x = np.arange(len(labels_t))
    ax_om.barh(x, om_means, xerr=om_stds, color=colors_t,
               alpha=0.75, height=0.5, capsize=5)
    ax_om.axvline(0.3111, color="k", ls="--", lw=1.5, label="Planck")
    ax_om.axvspan(0.3111-0.0056, 0.3111+0.0056,
                  alpha=0.15, color="k", label="Planck 1sigma")
    ax_om.set_yticks(x); ax_om.set_yticklabels(labels_t, fontsize=11)
    ax_om.set_xlabel(r"$\Omega_m$", fontsize=13)
    ax_om.legend(fontsize=9); ax_om.grid(True, alpha=0.3, axis="x")

    ax_h0.barh(x, h0_means, xerr=h0_stds, color=colors_t,
               alpha=0.75, height=0.5, capsize=5)
    ax_h0.axvline(67.66, color="gray", ls=":", lw=1.5, label="Planck H0")
    ax_h0.axvline(73.04, color="k",   ls="--", lw=1.5, label="SH0ES H0")
    ax_h0.set_yticks(x); ax_h0.set_yticklabels(labels_t, fontsize=11)
    ax_h0.set_xlabel(r"$H_0$ [km s$^{-1}$ Mpc$^{-1}$]", fontsize=13)
    ax_h0.legend(fontsize=9); ax_h0.grid(True, alpha=0.3, axis="x")

    fig6.suptitle(r"$\Lambda$CDM — Resumen de parametros" + f"\n{tag}",
                  fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig("tabla_resultados.pdf", dpi=150, bbox_inches="tight")
    print("  Saved: tabla_resultados.pdf"); plt.show()



# =============================================================================
# 6. MAIN — Interactivo desde terminal
# =============================================================================

# ── Tabla de referencia de qubits por hardware ────────────────────────────────
#
#  nqpp │ total qubits │ estados │ RAM simulación │ hardware recomendado
#  ─────┼──────────────┼─────────┼────────────────┼──────────────────────
#   2   │      4       │    16   │    < 1 MB      │ Laptop / cualquier PC
#   3   │      6       │    64   │    < 1 MB      │ Laptop / cualquier PC   ← DEFAULT
#   4   │      8       │   256   │    < 1 MB      │ PC con ≥ 8 GB RAM
#   5   │     10       │  1024   │    ~ 8 MB      │ PC con ≥ 16 GB RAM
#   6   │     12       │  4096   │    ~ 32 MB     │ PC potente / servidor
#   7   │     14       │ 16384   │    ~ 128 MB    │ Servidor / HPC         ← lento
#   8   │     16       │ 65536   │    ~ 512 MB    │ HPC (≥ 64 GB RAM)
#   9   │     18       │ 262144  │    ~ 2 GB      │ HPC (≥ 128 GB RAM)
#  10   │     20       │ 1048576 │    ~ 8 GB      │ HPC / supercomputadora
#  11+  │     22+      │  > 4M   │    > 32 GB     │ Supercomputadora / IBM Quantum real
#
# En hardware cuántico real (IBM Eagle 127q, Heron 133q):
#   cualquier nqpp es viable porque no hay simulación clásica de statevector.
#   El límite es la profundidad del circuito y el ruido de compuertas.

_QUBIT_TABLE = {
    # nqpp: (total_q, n_states, ram_str, nivel, recomendado_para)
    2:  ( 4,      16, "< 1 MB",   "OK",      "Laptop / cualquier PC"),
    3:  ( 6,      64, "< 1 MB",   "OK",      "Laptop / cualquier PC  ← recomendado"),
    4:  ( 8,     256, "< 1 MB",   "OK",      "PC con ≥ 8 GB RAM"),
    5:  (10,    1024, "~8 MB",    "OK",      "PC con ≥ 16 GB RAM"),
    6:  (12,    4096, "~32 MB",   "OK",      "PC potente / servidor"),
    7:  (14,   16384, "~128 MB",  "LENTO",   "Servidor / HPC"),
    8:  (16,   65536, "~512 MB",  "LENTO",   "HPC (≥ 64 GB RAM)"),
    9:  (18,  262144, "~2 GB",    "MUY LENTO","HPC (≥ 128 GB RAM)"),
    10: (20, 1048576, "~8 GB",    "MUY LENTO","HPC / Supercomputadora"),
}


def ask_dataset():
    """Pregunta que dataset usar para la inferencia."""
    pan_ok = _PANTHEON_DATA is not None
    n_pan  = len(_PANTHEON_DATA['z']) if pan_ok else 0

    print("\n  Dataset para la inferencia:")
    print(f"  [1] CC solamente          ({len(_CC_DATA)} Cosmic Chronometers H(z))")
    if pan_ok:
        print(f"  [2] Pantheon+ solamente   ({n_pan} SNe Ia, archivo encontrado)")
        print(f"  [3] CC + Pantheon+        (combinacion de ambos)")
    else:
        print("  [2] Pantheon+ solamente   archivo NO encontrado")
        print("      -> Pon pantheon_full_parameters.txt en la misma carpeta")
        print("  [3] CC + Pantheon+        archivo NO encontrado")

    while True:
        raw = input("  Seleccion (1/2/3) [default: 1]: ").strip()
        if raw in ("", "1"):
            return "CC"
        if raw == "2":
            if not pan_ok:
                print("  Archivo no encontrado. Usando CC.")
                return "CC"
            return "Pantheon+"
        if raw == "3":
            if not pan_ok:
                print("  Archivo no encontrado. Usando CC.")
                return "CC"
            return "CC+Pantheon+"
        print("  Ingresa 1, 2 o 3.")


def ask_int(prompt, min_val=1, default=None):
    """Pide un entero al usuario con validacion."""
    while True:
        suffix = f" [default: {default}]" if default is not None else ""
        raw = input(f"{prompt}{suffix}: ").strip()
        if raw == "" and default is not None:
            return default
        try:
            val = int(raw)
            if val >= min_val:
                return val
            print(f"  debe ser >= {min_val}. Intenta de nuevo.")
        except ValueError:
            print("  Ingresa un numero entero valido.")

def ask_method():
    """Pregunta qué método(s) correr."""
    print("\n  ¿Qué método(s) quieres correr?")
    print("  [1] Classical MCMC")
    print("  [2] Classical VI  (Variational Inference)")
    print("  [3] QMCMC")
    print("  [4] QVMC")
    print("  [5] Todos (recomendado para comparar)")
    print("  [6] Solo cuánticos (QMCMC + QVMC)")
    print("  [7] Solo clásicos  (Classical MCMC + Classical VI)")
    while True:
        raw = input("  Selección (1-7) [default: 5]: ").strip()
        if raw in ("", "5"):
            return {"cmcmc": True, "cvi": True, "qmcmc": True, "qvmc": True}
        if raw == "1":
            return {"cmcmc": True,  "cvi": False, "qmcmc": False, "qvmc": False}
        if raw == "2":
            return {"cmcmc": False, "cvi": True,  "qmcmc": False, "qvmc": False}
        if raw == "3":
            return {"cmcmc": False, "cvi": False, "qmcmc": True,  "qvmc": False}
        if raw == "4":
            return {"cmcmc": False, "cvi": False, "qmcmc": False, "qvmc": True}
        if raw == "6":
            return {"cmcmc": False, "cvi": False, "qmcmc": True,  "qvmc": True}
        if raw == "7":
            return {"cmcmc": True,  "cvi": True,  "qmcmc": False, "qvmc": False}
        print("  ⚠  Ingresa un número del 1 al 7.")


def ask_prior():
    """Pregunta el tipo de prior."""
    print("\n  ¿Qué tipo de prior quieres usar?")
    print("  [1] Flat      (uniforme — solo bounds duros)")
    print(f"  [2] Gaussiano (Ωm: {OM_MU} ± {OM_SIG}  |  H0: {H0_MU} ± {H0_SIG})")
    while True:
        raw = input("  Selección (1/2) [default: 1]: ").strip()
        if raw in ("", "1"):
            return "flat"
        if raw == "2":
            return "gaussian"
        print("  ⚠  Ingresa 1 o 2.")


def ask_hardware():
    """Pregunta en qué tipo de hardware va a correr."""
    print("\n  ¿En qué hardware vas a correr esto?")
    print("  [1] Laptop  (≤ 16 GB RAM)")
    print("  [2] PC      (16–64 GB RAM)")
    print("  [3] Servidor / HPC")
    print("  [4] Supercomputadora")
    print("  [5] Computadora cuántica real (IBM Quantum)")
    while True:
        raw = input("  Selección (1-5) [default: 1]: ").strip()
        if raw in ("", "1"):
            return "laptop"
        if raw == "2":
            return "pc"
        if raw == "3":
            return "hpc"
        if raw == "4":
            return "super"
        if raw == "5":
            return "quantum"
        print("  ⚠  Ingresa un número del 1 al 5.")


def ask_nqpp(hardware: str) -> int:
    """
    Pregunta el número de qubits por parámetro (nqpp).
    Muestra la tabla de referencia filtrada por hardware y advierte si el
    usuario elige algo fuera del rango recomendado.
    """
    # Rangos recomendados por hardware
    recommended = {
        "laptop":  (2, 4),
        "pc":      (3, 6),
        "hpc":     (5, 9),
        "super":   (7, 11),
        "quantum": (2, 15),   # sin límite práctico en hardware real
    }
    hw_names = {
        "laptop":  "Laptop",
        "pc":      "PC",
        "hpc":     "Servidor/HPC",
        "super":   "Supercomputadora",
        "quantum": "Computadora cuántica real",
    }
    lo, hi = recommended[hardware]
    default = min(3, hi)

    print(f"\n  Tabla de referencia para {hw_names[hardware]}:")
    print(f"  {'nqpp':>4}  {'qubits':>6}  {'estados':>8}  {'RAM sim.':>9}  {'velocidad':>9}  Hardware")
    print("  " + "─" * 70)

    for nq, (tq, ns, ram, vel, hw) in _QUBIT_TABLE.items():
        marker = "← recomendado" if lo <= nq <= hi else ""
        warn   = "⚠" if nq > hi else " "
        print(f"  {warn}{nq:>4}  {tq:>6}  {ns:>8,}  {ram:>9}  {vel:>9}  {hw}  {marker}")

    if hardware == "quantum":
        print("   11+     22+     > 4M       > 32 GB  (hardware real — sin límite práctico)")

    print(f"\n  Rango recomendado para tu hardware: nqpp = {lo} a {hi}")

    while True:
        raw = input(f"  Qubits por parámetro (nqpp) [default: {default}]: ").strip()
        if raw == "":
            nqpp = default
        else:
            try:
                nqpp = int(raw)
                if nqpp < 2:
                    print("  ⚠  Mínimo es 2. Intenta de nuevo.")
                    continue
            except ValueError:
                print("  ⚠  Ingresa un número entero.")
                continue

        # Advertencias según la elección
        total_q  = 2 * nqpp          # 2 parámetros: Ωm, H0
        n_states = 2 ** total_q
        info     = _QUBIT_TABLE.get(nqpp)
        ram_str  = info[2] if info else "> 32 GB"
        vel_str  = info[3] if info else "EXTREMADAMENTE LENTO"

        print(f"\n  → nqpp = {nqpp}  |  qubits totales = {total_q}  "
              f"|  estados = {n_states:,}  |  RAM ≈ {ram_str}")

        if hardware in ("laptop", "pc") and nqpp > hi:
            print(f"\n  ╔══════════════════════════════════════════════════════╗")
            print(f"  ║  ⚠  ADVERTENCIA — nqpp={nqpp} puede ser demasiado     ║")
            print(f"  ║  para tu hardware ({hw_names[hardware]}).               ║")
            print(f"  ║  Con {n_states:,} estados y simulación statevector,    ║")
            print(f"  ║  la RAM requerida es ≈ {ram_str} por iteración.         ║")
            print(f"  ║  Velocidad esperada: {vel_str:<12}               ║")
            print(f"  ║  Recomendado para tu hardware: nqpp ≤ {hi}             ║")
            print(f"  ╚══════════════════════════════════════════════════════╝")
            confirm = input("  ¿Continuar de todas formas? (s/n) [default: n]: ").strip().lower()
            if confirm not in ("s", "si", "sí", "yes", "y"):
                print("  Elige un valor menor.")
                continue

        elif hardware == "hpc" and nqpp > hi:
            print(f"  ⚠  nqpp={nqpp} requiere ≈ {ram_str} de RAM y será {vel_str}.")
            print(f"     Recomendado para HPC: nqpp ≤ {hi}.")
            confirm = input("  ¿Continuar? (s/n) [default: n]: ").strip().lower()
            if confirm not in ("s", "si", "sí", "yes", "y"):
                continue

        elif hardware == "quantum":
            print(f"  ✓ En hardware cuántico real no hay límite por RAM.")
            print(f"    El límite es la profundidad del circuito y el ruido de compuertas.")
            if nqpp > 10:
                print(f"  ⚠  nqpp={nqpp} genera circuitos muy profundos.")
                print(f"    En hardware NISQ actual (IBM Eagle/Heron) puede haber")
                print(f"    demasiado ruido para resultados confiables sin mitigación.")

        elif nqpp <= hi:
            print(f"  ✓ Elección dentro del rango recomendado para tu hardware.")

        return nqpp


def _main():
    """Función principal — evita problemas con 'global' en __main__."""
    import sys
    mod = sys.modules[__name__]   # referencia al módulo para modificar PRIOR_TYPE

    print("=" * 60)
    print("  ΛCDM — Classical & Quantum Parameter Estimation")
    print("  Parámetros: Ωm ∈ (0.20, 0.45)  |  H0 ∈ (60, 80) km/s/Mpc")
    print("=" * 60)

    # ── 0. Dataset ────────────────────────────────────────────────────────
    dataset_choice = ask_dataset()

    if dataset_choice == "Pantheon+":
        # Solo Pantheon+ — CC array se mantiene pero no se usa en log_posterior
        mod.z_obs     = _CC_DATA[:, 0]   # para las graficas H(z)
        mod.H_obs     = _CC_DATA[:, 1]
        mod.sigma_obs = _CC_DATA[:, 2]
        mod.N_DATA    = len(mod.z_obs)
        mod.DATASET   = "Pantheon+"
        _precompute_pantheon_integrals()
        print(f"  -> Dataset: solo Pantheon+ ({len(_PANTHEON_DATA['z'])} SNe Ia)")
        print(f"     Likelihood: modulo de distancia con marginalizacion sobre M_abs")

    elif dataset_choice == "CC+Pantheon+":
        mod.z_obs     = _CC_DATA[:, 0]
        mod.H_obs     = _CC_DATA[:, 1]
        mod.sigma_obs = _CC_DATA[:, 2]
        mod.N_DATA    = len(mod.z_obs)
        mod.DATASET   = "CC+Pantheon+"
        _precompute_pantheon_integrals()
        print(f"  -> Dataset: CC ({mod.N_DATA} pts) + Pantheon+ "
              f"({len(_PANTHEON_DATA['z'])} SNe Ia)")

    # ── 1. Hardware ───────────────────────────────────────────────────────
    hardware = ask_hardware()

    # ── 2. Prior — modificamos la variable global del módulo directamente
    mod.PRIOR_TYPE = ask_prior()
    print(f"  → Prior: {mod.PRIOR_TYPE.upper()}")

    # ── 3. Métodos ────────────────────────────────────────────────────────
    run       = ask_method()
    run_cmcmc = run["cmcmc"]
    run_cvi   = run["cvi"]
    run_qmcmc = run["qmcmc"]
    run_qvmc  = run["qvmc"]

    # ── 4. Qubits (VI y QVMC usan grilla discreta) ───────────────────────
    nqpp = 3
    if run_qvmc or run_cvi:
        nqpp = ask_nqpp(hardware)

    # ── 5. Pasos MCMC ────────────────────────────────────────────────────
    n_steps_mcmc  = 2000
    n_burn_mcmc   = 200
    n_steps_qmcmc = 2000
    n_burn_qmcmc  = 200

    if run_cmcmc:
        print("\n─" * 30)
        print("  Configuración Classical MCMC")
        print("─" * 30)
        n_steps_mcmc = ask_int("  Pasos de sampling", min_val=100, default=2000)
        n_burn_mcmc  = max(1, int(round(0.10 * n_steps_mcmc)))
        print(f"  → Burn-in: {n_burn_mcmc} pasos (10%)")

    if run_qmcmc:
        print("\n─" * 30)
        print("  Configuración QMCMC")
        print("─" * 30)
        n_steps_qmcmc = ask_int("  Pasos de sampling", min_val=100, default=2000)
        n_burn_qmcmc  = max(1, int(round(0.10 * n_steps_qmcmc)))
        print(f"  → Burn-in: {n_burn_qmcmc} pasos (10%)")

    # ── 6. Iteraciones variacionales ──────────────────────────────────────
    n_iter_cvi  = 500
    n_iter_qvmc = 500

    if run_cvi:
        print("\n─" * 30)
        print("  Configuración Classical VI")
        print("─" * 30)
        n_iter_cvi = ask_int("  Iteraciones de entrenamiento", min_val=50, default=500)

    if run_qvmc:
        print("\n─" * 30)
        print("  Configuración QVMC")
        print("─" * 30)
        n_iter_qvmc = ask_int("  Iteraciones de entrenamiento", min_val=50, default=500)

    # ── 7. Resumen ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Resumen de configuración")
    print("=" * 60)
    print(f"  Hardware : {hardware}  |  Prior: {mod.PRIOR_TYPE}")
    if run_cmcmc:
        print(f"  Cl. MCMC : {n_steps_mcmc} pasos + {n_burn_mcmc} burn-in")
    if run_cvi:
        print(f"  Cl. VI   : {n_iter_cvi} iters  |  nqpp={nqpp} (grilla)")
    if run_qmcmc:
        print(f"  QMCMC    : {n_steps_qmcmc} pasos + {n_burn_qmcmc} burn-in")
    if run_qvmc:
        print(f"  QVMC     : {n_iter_qvmc} iters  |  nqpp={nqpp}  "
              f"|  qubits={2*nqpp}  |  estados={4**nqpp:,}")
    print("=" * 60)
    input("  Presiona Enter para iniciar...")

    # ── 8. Correr métodos ─────────────────────────────────────────────────
    cmcmc_info = None
    if run_cmcmc:
        print("\n" + "─"*60)
        print(f"  Classical MCMC  [prior: {mod.PRIOR_TYPE}]")
        print("─"*60)
        cmcmc = ClassicalMCMC(n_chains=8, step_size=0.5, n_burn=n_burn_mcmc)
        _, cmcmc_info = cmcmc.run(n_steps=n_steps_mcmc)

    cvi_info = None
    if run_cvi:
        print("\n" + "─"*60)
        print(f"  Classical VI  [prior: {mod.PRIOR_TYPE}  |  nqpp={nqpp}]")
        print("─"*60)
        cvi      = ClassicalVI(n_qubits_per_param=nqpp)
        cvi_info = cvi.run(max_iter=n_iter_cvi)

    qmcmc_info = None
    if run_qmcmc:
        print("\n" + "─"*60)
        print(f"  QMCMC  [prior: {mod.PRIOR_TYPE}]")
        print("─"*60)
        qmcmc = QMCMC(n_chains=8, step_size=0.05,
                      n_burn=n_burn_qmcmc, n_layers=3)
        _, qmcmc_info = qmcmc.run(n_steps=n_steps_qmcmc)

    qvmc_info = None
    if run_qvmc:
        print("\n" + "─"*60)
        print(f"  QVMC  [prior: {mod.PRIOR_TYPE}  |  nqpp={nqpp}]")
        print("─"*60)
        qvmc      = QVMC(n_qubits_per_param=nqpp, n_layers=3, n_shots=3000)
        qvmc_info = qvmc.run(max_iter=n_iter_qvmc, n_chains=4)

    # ── 9. Plots ──────────────────────────────────────────────────────────
    print("\n" + "─"*60)
    print("  Generando gráficas...")
    print("─"*60)
    plot_results(cmcmc_info=cmcmc_info, cvi_info=cvi_info,
                 qmcmc_info=qmcmc_info, qvmc_info=qvmc_info,
                 nqpp=nqpp, dataset_label=mod.DATASET)

    print("\n" + "=" * 60)
    print("  ROADMAP: HOW TO MAKE IT MORE QUANTUM")
    print("=" * 60)
    print("""
  Current state:
    Classical MCMC : Metropolis-Hastings con propuesta gaussiana
    Classical VI   : Gaussiana variacional optimizada con COBYLA
    QMCMC          : propuesta cuántica + aceptación clásica
    QVMC           : posterior representado como estado cuántico |ψ(φ)⟩

  Next steps:
    STEP 1 — Parameter-shift rule para gradientes nativos cuánticos
    STEP 2 — Hadamard test para evaluar la likelihood cuánticamente
    STEP 3 — Quantum amplitude estimation para normalización
    STEP 4 — IBM Quantum real hardware
  """)


if __name__ == "__main__":
    _main()
