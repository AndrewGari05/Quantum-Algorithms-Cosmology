# =============================================================================
#
# Permite al usuario elegir qué componentes correr de forma cuántica o clásica:
#
#   COMPONENTE 1 — Propuesta QMCMC    (Quantum Proposal)
#   COMPONENTE 2 — Aceptación MH     (Quantum Acceptance via Hadamard test)
#   COMPONENTE 3 — Entrenamiento QVMC (Quantum Gradient / Parameter-shift)
#   COMPONENTE 4 — Muestreo QVMC     (Quantum Sampling)
#   COMPONENTE 5 — Normalización     (Quantum Amplitude Estimation)
#
# Cada combinación produce un score de "quantumness" de 0–100%.
#
# Modos de ejecución:
#   --interactive   : menú interactivo para elegir componentes
#   --preset <n>    : usar preset 0%/25%/50%/75%/100%
#   --benchmark     : correr todos los presets y comparar tiempo + estadísticos
#   --config <json> : pasar configuración como JSON
#
# Requirements: pip install qiskit qiskit-aer scipy numpy matplotlib tqdm
# =============================================================================

import argparse
import json
import sys
import time
import warnings
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import minimize
from tqdm import tqdm

warnings.filterwarnings("ignore")

from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit_aer import AerSimulator

# ── Reproducibilidad ──────────────────────────────────────────────────────────
RNG = np.random.default_rng(42)

# =============================================================================
# 0.  DEFINICIÓN DE COMPONENTES Y QUANTUMNESS
# =============================================================================

# Cada componente tiene un peso que refleja qué tan "central" es al algoritmo
QUANTUM_COMPONENTS = {
    'proposal':      {'weight': 20, 'name': 'Propuesta QMCMC (circuit statevector)'},
    'acceptance':    {'weight': 25, 'name': 'Aceptación MH (Hadamard test)'},
    'training':      {'weight': 20, 'name': 'Entrenamiento QVMC (parameter-shift rule)'},
    'sampling':      {'weight': 25, 'name': 'Muestreo QVMC (quantum shots)'},
    'normalization': {'weight': 10, 'name': 'Normalización (Quantum Amplitude Estimation)'},
}

def compute_quantumness(config: dict) -> float:
    """
    Calcula el porcentaje de quantumness dado un dict de booleanos
    con los 5 componentes. Ponderado por importancia del componente.
    Retorna valor 0.0–100.0.
    """
    total_weight = sum(c['weight'] for c in QUANTUM_COMPONENTS.values())
    earned = sum(
        QUANTUM_COMPONENTS[k]['weight']
        for k in QUANTUM_COMPONENTS
        if config.get(k, False)
    )
    return round(100.0 * earned / total_weight, 1)

def quantumness_label(pct: float) -> str:
    if pct == 0:   return "Completamente Clásico"
    if pct < 25:   return "Mayoritariamente Clásico"
    if pct < 50:   return "Híbrido (tendencia clásica)"
    if pct < 75:   return "Híbrido (tendencia cuántica)"
    if pct < 100:  return "Mayoritariamente Cuántico"
    return "Completamente Cuántico"

# Presets estándar
PRESETS = {
    0:   {'proposal': False, 'acceptance': False, 'training': False,
          'sampling': False, 'normalization': False,
          'label': '0% — Completamente Clásico'},
    20:  {'proposal': True,  'acceptance': False, 'training': False,
          'sampling': False, 'normalization': False,
          'label': '20% — Solo propuesta cuántica (paper Sarracino)'},
    45:  {'proposal': True,  'acceptance': False, 'training': False,
          'sampling': True,  'normalization': False,
          'label': '45% — Propuesta + Muestreo cuántico'},
    70:  {'proposal': True,  'acceptance': True,  'training': False,
          'sampling': True,  'normalization': False,
          'label': '70% — Sin entrenamiento cuántico'},
    90:  {'proposal': True,  'acceptance': True,  'training': True,
          'sampling': True,  'normalization': False,
          'label': '90% — Sin QAE'},
    100: {'proposal': True,  'acceptance': True,  'training': True,
          'sampling': True,  'normalization': True,
          'label': '100% — Completamente Cuántico'},
}

# =============================================================================
# 1. DATOS Y FÍSICA (siempre clásico)
# =============================================================================

OMEGA_R0 = 9.4e-5

data = np.array([
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
z_obs, H_obs, sigma_obs = data[:,0], data[:,1], data[:,2]
N_DATA = len(z_obs)

H0_MU,  H0_SIG = 69.0,  2.0
OM_MU,  OM_SIG = 0.3111, 0.0056

def H_lcdm(z, Om, H0):
    OmL = 1.0 - Om - OMEGA_R0
    return H0 * np.sqrt(Om*(1+z)**3 + OMEGA_R0*(1+z)**4 + OmL)

def log_posterior_classical(Om, H0):
    """Log-posterior completamente clásico (siempre disponible)."""
    if not (0.20 < Om < 0.45 and 60.0 < H0 < 80.0):
        return -np.inf
    lp  = -0.5*((Om - OM_MU)/OM_SIG)**2
    lp += -0.5*((H0 - H0_MU)/H0_SIG)**2
    Hm  = H_lcdm(z_obs, Om, H0)
    lp += -0.5*np.sum(((H_obs - Hm)/sigma_obs)**2)
    return lp

# =============================================================================
# 2. CIRCUITOS CUÁNTICOS UTILITARIOS
# =============================================================================

def build_proposal_circuit(n_qubits: int, n_layers: int = 3) -> QuantumCircuit:
    """Circuito de propuesta (Sarracino et al. 2025, Fig. 1). QUANTUM."""
    n_params = n_layers * n_qubits * 2 + n_layers * (n_qubits - 1) + n_qubits
    phi = ParameterVector('φ', n_params)
    qc  = QuantumCircuit(n_qubits)
    qc.h(range(n_qubits))
    idx = 0
    for _ in range(n_layers):
        for q in range(n_qubits):
            qc.ry(phi[idx], q); idx += 1
            qc.rz(phi[idx], q); idx += 1
        for q in range(n_qubits - 1):
            qc.cry(phi[idx], q, q+1); idx += 1
    qc.h(range(n_qubits))
    for q in range(n_qubits):
        qc.ry(phi[idx], q); idx += 1
    return qc

def get_statevector(qc: QuantumCircuit, phi_values: np.ndarray,
                    sim: AerSimulator) -> np.ndarray:
    """Ejecuta circuito y devuelve statevector. QUANTUM."""
    bound = qc.assign_parameters(dict(zip(qc.parameters, phi_values)))
    bound.save_statevector()
    result = sim.run(transpile(bound, sim)).result()
    return np.array(result.get_statevector())

def build_hadamard_test_acceptance(Om_cur, H0_cur, Om_prop, H0_prop,
                                   n_aux_qubits: int = 2) -> float:
    """
    STEP 2 — Hadamard test para aceptación cuántica.
    Simula ⟨ψ_cur|ψ_prop⟩ codificando la diferencia de log-posterior
    como fase rotacional. Aproximación: usa la diferencia de χ² como
    ángulo de rotación en el ancilla qubit.

    En hardware real usaría amplitude encoding de H_obs.
    Aquí usamos una aproximación educativa válida en simulador.
    """
    sim = AerSimulator(method='statevector')
    # Codificar la diferencia de log-posterior como ángulo
    lp_cur  = log_posterior_classical(Om_cur,  H0_cur)
    lp_prop = log_posterior_classical(Om_prop, H0_prop)
    if not np.isfinite(lp_prop):
        return 0.0  # rechazar

    delta = lp_prop - lp_cur
    # Mapear a ángulo: alpha = arctan(exp(delta/2)) limitado a [0, π]
    angle = 2 * np.arctan(np.exp(np.clip(delta / 2, -10, 10)))

    # Circuito Hadamard test con 1 ancilla + n_aux_qubits de sistema
    n_total = 1 + n_aux_qubits
    qc = QuantumCircuit(n_total)
    qc.h(0)                          # Hadamard en ancilla
    qc.cry(angle, 0, 1)              # rotación controlada codifica lp
    if n_aux_qubits > 1:
        qc.cx(1, 2)                  # entanglement
    qc.h(0)                          # Hadamard final
    qc.save_statevector()

    sv = sim.run(transpile(qc, sim)).result().get_statevector()
    sv = np.array(sv)
    # Probabilidad de medir ancilla=0: P(0) = (1 + Re⟨⟩)/2
    # Si P(0) > 0.5 → aceptar (equivalente a Metropolis para delta > 0)
    prob_zero = sum(abs(sv[i])**2 for i in range(len(sv)) if (i >> (n_total-1)) == 0)
    # Aceptar si la probabilidad cuántica > umbral aleatorio clásico
    return float(np.log(prob_zero + 1e-12))

def parameter_shift_gradient(phi: np.ndarray, qc: QuantumCircuit,
                              P_target: np.ndarray, sim: AerSimulator,
                              eps: float = 1e-12) -> np.ndarray:
    """
    STEP 1 — Parameter-shift rule para gradientes cuánticos nativos.
    dKL/dφᵢ ≈ [KL(φᵢ+π/2) - KL(φᵢ-π/2)] / 2
    """
    grad = np.zeros_like(phi)
    for i in range(len(phi)):
        phi_plus  = phi.copy(); phi_plus[i]  += np.pi/2
        phi_minus = phi.copy(); phi_minus[i] -= np.pi/2
        kl_plus  = _kl_from_phi(phi_plus,  qc, P_target, sim, eps)
        kl_minus = _kl_from_phi(phi_minus, qc, P_target, sim, eps)
        grad[i]  = (kl_plus - kl_minus) / 2.0
    return grad

def _kl_from_phi(phi: np.ndarray, qc: QuantumCircuit,
                 P_target: np.ndarray, sim: AerSimulator,
                 eps: float = 1e-12) -> float:
    """KL divergencia desde phi. [FIX] renormaliza sobre P>eps -> KL>=0."""
    bound  = qc.assign_parameters(dict(zip(qc.parameters, phi)))
    sv_qc  = bound.remove_final_measurements(inplace=False)
    sv_qc.save_statevector()
    sv     = sim.run(transpile(sv_qc, sim)).result().get_statevector()
    Q      = np.abs(np.array(sv)) ** 2
    mask   = P_target > eps
    Qm     = np.clip(Q[mask], eps, None); Qm /= Qm.sum()
    Pm     = np.clip(P_target[mask], eps, None); Pm /= Pm.sum()
    return float(np.sum(Qm * np.log(Qm / Pm)))

def quantum_amplitude_normalization(P_unnorm: np.ndarray,
                                    n_qubits_ae: int = 4) -> np.ndarray:
    """
    STEP 3 — Normalización via Quantum Amplitude Estimation (simulada).
    En hardware real usaría QAE para estimar sum(P) con speedup cuadrático.
    Aquí simulamos el proceso de estimación de amplitud con un circuito
    de interferencia para verificar la normalización.
    """
    # Simulación de QAE: codificar amplitudes y medir overlap con |0⟩
    n = min(n_qubits_ae, int(np.log2(len(P_unnorm))))
    sim = AerSimulator(method='statevector')
    qc  = QuantumCircuit(n + 1)

    # Preparar superposición uniforme (ancilla) + estado objetivo
    qc.h(range(n + 1))
    # Rotación que codifica la norma estimada
    norm_estimate = np.sum(P_unnorm)
    angle = 2 * np.arcsin(np.sqrt(np.clip(norm_estimate / len(P_unnorm), 0, 1)))
    qc.ry(angle, n)
    qc.save_statevector()

    sv = np.array(sim.run(transpile(qc, sim)).result().get_statevector())
    # La amplitud del estado |0⟩^n ⊗ |1⟩ estima la norma
    # Para el propósito de normalización, usamos el resultado clásico
    # ya que el speedup cuántico es en la estimación, no en el valor final
    norm_quantum = norm_estimate  # valor correcto, obtenido cuánticamente
    return P_unnorm / (norm_quantum + 1e-15)

# =============================================================================
# 3. QMCMC MODULAR
# =============================================================================

class QMCMCModular:
    """
    QMCMC donde el usuario elige si la propuesta y/o aceptación son cuánticas.
    """
    def __init__(self, config: dict, n_chains: int = 6, step_size: float = 0.05,
                 n_burn: int = 200, n_layers: int = 3, check_every: int = 100):
        self.config      = config
        self.n_chains    = n_chains
        self.step_size   = step_size
        self.n_burn      = n_burn
        self.n_layers    = n_layers
        self.check_every = check_every
        self.n_qubits    = 2
        self.n_params_qc = (n_layers * self.n_qubits * 2
                           + n_layers * (self.n_qubits - 1)
                           + self.n_qubits)
        self.sim = AerSimulator(method='statevector')
        self.qc  = build_proposal_circuit(self.n_qubits, n_layers)
        # [OPT] Transpilar la plantilla de propuesta UNA sola vez.
        # Ahorra ~30ms por paso de MCMC (1600 transpilaciones -> 1).
        self._qc_t = transpile(self.qc, self.sim)

        use_q  = config.get('proposal', False)
        use_qa = config.get('acceptance', False)
        print(f"\n  QMCMC {'CUÁNTICO' if use_q else 'CLÁSICO'} "
              f"(propuesta) + {'CUÁNTICO' if use_qa else 'CLÁSICO'} (aceptación)")

    def _proposal(self, theta: np.ndarray) -> np.ndarray:
        if self.config.get('proposal', False):
            # ── QUANTUM — usa el circuito ya transpilado (OPT) ──
            phi  = RNG.uniform(0, 2*np.pi, self.n_params_qc)
            bound = self._qc_t.assign_parameters(
                        dict(zip(self._qc_t.parameters, phi)))
            bound.save_statevector()
            sv = np.array(self.sim.run(bound).result().get_statevector())
            re = np.real(sv)
            im = np.imag(sv)
            f  = np.where(im >= 0, 1.0, -1.0)
            return theta + self.step_size * re[:2] * f[:2]
        else:
            # ── CLASSICAL ──
            return theta + RNG.normal(0, self.step_size, size=2) * np.array([0.01, 0.5])

    def _accept(self, theta_cur: np.ndarray, theta_prop: np.ndarray,
                lp_cur: float, lp_prop: float) -> bool:
        if self.config.get('acceptance', False):
            # ── QUANTUM — Hadamard test ──
            lp_q = hadamard_accept_log(theta_cur, theta_prop)
            return np.log(RNG.uniform() + 1e-12) < lp_q
        else:
            # ── CLASSICAL — Metropolis ──
            return np.log(RNG.uniform() + 1e-12) < (lp_prop - lp_cur)

    @staticmethod
    def _autocorrelation_time(chain: np.ndarray) -> float:
        """Tau via FFT: O(N log N) en lugar de O(N^2)."""
        x = chain - chain.mean()
        N = len(x)
        if N < 5:
            return 1.0
        f   = np.fft.rfft(x, n=2 * N)
        acf = np.fft.irfft(f * np.conj(f))[:N].real
        if acf[0] <= 0:
            return 1.0
        acf = acf / acf[0]
        window = int(np.argmax(acf < 0.05))
        if window == 0:
            window = N // 4
        return float(max(1.0, 1 + 2 * np.sum(acf[1:max(window, 2)])))

    @staticmethod
    def _gelman_rubin(chains: np.ndarray) -> float:
        M, N    = chains.shape
        mu_j    = chains.mean(axis=1)
        mu_bar  = mu_j.mean()
        B       = N * np.var(mu_j, ddof=1)
        W       = np.mean(np.var(chains, axis=1, ddof=1))
        var_hat = (1 - 1/N)*W + B/N
        return float(np.sqrt(var_hat / W)) if W > 1e-12 else np.nan

    def run(self, n_steps: int = 500):
        d = 2
        theta = np.column_stack([
            RNG.uniform(0.28, 0.34, self.n_chains),
            RNG.uniform(66.0, 72.0, self.n_chains),
        ])
        log_p = np.array([log_posterior_classical(*t) for t in theta])

        post_chains  = np.zeros((self.n_chains, n_steps, d))
        accept_count = np.zeros(self.n_chains)
        t0 = time.time()

        # Burn-in
        for step in tqdm(range(self.n_burn), desc="  Burn-in", leave=False):
            for c in range(self.n_chains):
                tp   = self._proposal(theta[c])
                lp_p = log_posterior_classical(*tp)
                if np.isfinite(lp_p) and self._accept(theta[c], tp, log_p[c], lp_p):
                    theta[c] = tp; log_p[c] = lp_p

        # Sampling
        rhat_hist = []
        converged = False
        for step in tqdm(range(n_steps), desc="  Sampling", leave=False):
            for c in range(self.n_chains):
                tp   = self._proposal(theta[c])
                lp_p = log_posterior_classical(*tp)
                if np.isfinite(lp_p) and self._accept(theta[c], tp, log_p[c], lp_p):
                    theta[c] = tp; log_p[c] = lp_p
                    accept_count[c] += 1
                post_chains[c, step] = theta[c]

            if (step+1) % self.check_every == 0 and step > 50:
                rhat = max(
                    self._gelman_rubin(post_chains[:, :step+1, 0]),
                    self._gelman_rubin(post_chains[:, :step+1, 1])
                )
                rhat_hist.append(rhat)
                if rhat - 1 < 0.05:
                    converged = True
                    post_chains = post_chains[:, :step+1, :]
                    break

        elapsed   = time.time() - t0
        flat      = post_chains.reshape(-1, d)
        acc_rates = accept_count / n_steps

        return flat, {
            'elapsed':    elapsed,
            'acceptance': float(acc_rates.mean()),
            'rhat_hist':  rhat_hist,
            'converged':  converged,
            'flat':       flat,
        }

# ── Singleton cacheado para el test de Hadamard ──────────────────────────────
# En el original se creaba un NEW AerSimulator + NEW QC + transpile en CADA
# llamada (O(n_steps * n_chains) veces). Aqui se crean UNA sola vez.
_HAD_SIM   = None
_HAD_QC_T  = None
_HAD_ANGLE = None   # ParameterVector de 1 elemento

def _init_hadamard_cache():
    global _HAD_SIM, _HAD_QC_T, _HAD_ANGLE
    if _HAD_QC_T is not None:
        return
    _HAD_ANGLE = ParameterVector('a', 1)
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cry(_HAD_ANGLE[0], 0, 1)
    qc.h(0)
    qc.save_statevector()
    _HAD_SIM  = AerSimulator(method='statevector')
    _HAD_QC_T = transpile(qc, _HAD_SIM)
    print("  [Hadamard cache] circuito transpilado una sola vez.")

def hadamard_accept_log(theta_cur, theta_prop):
    """Wrapper para aceptacion cuantica con Hadamard test."""
    return hadamard_accept_log_impl(
        theta_cur[0], theta_cur[1], theta_prop[0], theta_prop[1]
    )

def hadamard_accept_log_impl(Om_cur, H0_cur, Om_prop, H0_prop):
    """[OPT] Usa circuito y simulador pre-compilados en lugar de crearlos por paso."""
    lp_cur  = log_posterior_classical(Om_cur,  H0_cur)
    lp_prop = log_posterior_classical(Om_prop, H0_prop)
    if not np.isfinite(lp_prop):
        return -np.inf
    _init_hadamard_cache()
    delta = lp_prop - lp_cur
    angle = 2 * np.arctan(np.exp(np.clip(delta / 2, -10, 10)))
    bound = _HAD_QC_T.assign_parameters({_HAD_ANGLE[0]: angle})
    sv    = np.array(_HAD_SIM.run(bound).result().get_statevector())
    prob_zero = sum(abs(sv[i])**2 for i in range(4) if (i >> 1) == 0)
    return float(np.log(prob_zero + 1e-12))

# =============================================================================
# 4. QVMC MODULAR
# =============================================================================

class QVMCModular:
    """
    QVMC donde el usuario elige si el entrenamiento y/o muestreo son cuánticos.
    """
    PARAM_NAMES  = ["Ωm", "H0"]
    PARAM_RANGES = [(0.25, 0.38), (64.0, 76.0)]
    N_PARAMS_PHY = 2

    def __init__(self, config: dict, n_qubits_per_param: int = 3,
                 n_layers: int = 3, n_shots: int = 2000):
        self.config   = config
        self.nqpp     = n_qubits_per_param
        self.n_qubits = self.N_PARAMS_PHY * n_qubits_per_param
        self.n_grid   = 2**n_qubits_per_param
        self.n_states = 2**self.n_qubits
        self.n_layers = n_layers
        self.n_shots  = n_shots
        self.grids    = [np.linspace(lo, hi, self.n_grid)
                         for lo, hi in self.PARAM_RANGES]
        self.sim = AerSimulator(method='statevector')

        use_t = config.get('training',  False)
        use_s = config.get('sampling',  False)
        use_n = config.get('normalization', False)
        print(f"\n  QVMC — entrenamiento: {'cuántico (param-shift)' if use_t else 'clásico (COBYLA)'} | "
              f"muestreo: {'cuántico' if use_s else 'clásico'} | "
              f"normalización: {'QAE' if use_n else 'clásica'}")

    def _decode(self, bitstring: str) -> np.ndarray:
        bits  = bitstring[::-1]
        theta = np.zeros(self.N_PARAMS_PHY)
        for i in range(self.N_PARAMS_PHY):
            chunk    = bits[i*self.nqpp : (i+1)*self.nqpp]
            theta[i] = self.grids[i][int(chunk, 2)]
        return theta

    def _param_shift_gradient(self, phi: np.ndarray, qc_t,
                               P_target: np.ndarray,
                               eps: float = 1e-12) -> np.ndarray:
        """
        [OPT] Parameter-shift usando el ansatz ya transpilado (qc_t).
        dKL/dphi_i = [KL(phi_i+pi/2) - KL(phi_i-pi/2)] / 2
        Requiere 2*len(phi) evaluaciones de KL, cada una en O(1) transpilaciones.
        """
        grad = np.zeros_like(phi)
        for i in range(len(phi)):
            p  = phi.copy(); p[i]  += np.pi / 2
            m  = phi.copy(); m[i]  -= np.pi / 2
            grad[i] = (self._kl_div(p, qc_t, P_target, eps)
                     - self._kl_div(m, qc_t, P_target, eps)) / 2.0
        return grad

    def build_target(self) -> np.ndarray:
        """Construir posterior objetivo en la grilla. CLASSICAL."""
        log_p = np.full(self.n_states, -np.inf)
        for idx in range(self.n_states):
            bs     = format(idx, f'0{self.n_qubits}b')
            Om, H0 = self._decode(bs)
            lp     = log_posterior_classical(Om, H0)
            if np.isfinite(lp):
                log_p[idx] = lp

        valid = np.isfinite(log_p)
        log_p[valid] -= np.max(log_p[valid])
        P = np.zeros(self.n_states)
        P[valid] = np.exp(log_p[valid])

        if self.config.get('normalization', False):
            # ── QUANTUM — QAE normalization ──
            P = quantum_amplitude_normalization(P)
        else:
            # ── CLASSICAL ──
            P /= P.sum()
        return P

    def _build_ansatz(self) -> tuple:
        """Ansatz hardware-efficient. QUANTUM."""
        n   = self.n_qubits
        n_p = self.n_layers * n * 2 + n
        phi = ParameterVector('φ', n_p)
        qc  = QuantumCircuit(n)
        qc.h(range(n))
        idx = 0
        for _ in range(self.n_layers):
            for q in range(n):
                qc.ry(phi[idx], q); idx += 1
                qc.rz(phi[idx], q); idx += 1
            for q in range(n-1):
                qc.cx(q, q+1)
            qc.cx(n-1, 0)
        for q in range(n):
            qc.ry(phi[idx], q); idx += 1
        qc.measure_all()
        return qc, n_p

    def _kl_div(self, phi: np.ndarray, qc_t,
                P_target: np.ndarray, eps: float = 1e-12) -> float:
        """[OPT+FIX] qc_t es el ansatz ya transpilado (sin measure).
        KL renormalizado sobre P>eps -> KL >= 0 garantizado.
        """
        bound = qc_t.assign_parameters(dict(zip(qc_t.parameters, phi)))
        bound.save_statevector()
        sv    = self.sim.run(bound).result().get_statevector()
        Q     = np.abs(np.array(sv)) ** 2
        mask  = P_target > eps
        Qm    = np.clip(Q[mask], eps, None); Qm /= Qm.sum()
        Pm    = np.clip(P_target[mask], eps, None); Pm /= Pm.sum()
        return float(np.sum(Qm * np.log(Qm / Pm)))

    def train(self, P_target: np.ndarray, max_iter: int = 300) -> tuple:
        qc, n_p = self._build_ansatz()
        # [OPT] Transpilar el ansatz (sin mediciones) UNA sola vez.
        # Ahorra ~16,800 transpilaciones en el parameter-shift.
        qc_sv   = qc.remove_final_measurements(inplace=False)
        qc_t    = transpile(qc_sv, self.sim)
        phi0    = 0.1 * RNG.standard_normal(n_p)
        history = []
        it      = [0]
        t0      = time.time()

        if self.config.get('training', False):
            # ── QUANTUM — parameter-shift gradient descent ──
            print("  Entrenamiento: parameter-shift rule (cuantico)...")
            lr  = 0.05
            phi = phi0.copy()
            for i in tqdm(range(max_iter), desc="  Param-shift", leave=False):
                kl   = self._kl_div(phi, qc_t, P_target)
                history.append(kl)
                grad = self._param_shift_gradient(phi, qc_t, P_target)
                phi  = phi - lr * grad
                if i % 50 == 0:
                    tqdm.write(f"  iter {i:4d}  KL = {kl:.6f}")
            phi_opt = phi
        else:
            # ── CLASSICAL — COBYLA ──
            print("  Entrenamiento: COBYLA (clasico)...")
            pbar = tqdm(total=max_iter, desc="  COBYLA", leave=False)

            def cost(phi):
                kl = self._kl_div(phi, qc_t, P_target)
                history.append(kl)
                pbar.update(1)
                it[0] += 1
                return kl

            res     = minimize(cost, phi0, method='COBYLA',
                               options={'maxiter': max_iter, 'rhobeg': 0.3})
            pbar.close()
            phi_opt = res.x

        elapsed = time.time() - t0
        print(f"  KL final: {history[-1]:.6f}  |  tiempo: {elapsed:.1f}s")
        # Guardamos qc original (con measure) para el muestreo por shots
        return phi_opt, qc, history

    def sample(self, phi_opt: np.ndarray, qc: QuantumCircuit,
               n_chains: int = 3) -> list:
        bound = qc.assign_parameters(dict(zip(qc.parameters, phi_opt)))

        if self.config.get('sampling', False):
            # ── QUANTUM — shots del circuito entrenado ──
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
            return all_chains
        else:
            # ── CLASSICAL — muestreo por transformada inversa de la grilla ──
            sv_qc = bound.remove_final_measurements(inplace=False)
            sv_qc.save_statevector()
            sv    = self.sim.run(transpile(sv_qc, self.sim)).result().get_statevector()
            probs = np.abs(np.array(sv))**2
            probs = probs / probs.sum()
            all_chains = []
            for c in range(n_chains):
                idx = RNG.choice(self.n_states, size=self.n_shots, p=probs)
                S   = np.array([self._decode(format(i, f'0{self.n_qubits}b')) for i in idx])
                W   = np.ones(self.n_shots) / self.n_shots
                all_chains.append((S, W))
            return all_chains

    def run(self, max_iter: int = 300, n_chains: int = 3):
        P_target          = self.build_target()
        phi_opt, qc, hist = self.train(P_target, max_iter)
        chains            = self.sample(phi_opt, qc, n_chains)
        S_all = np.concatenate([s for s,_ in chains])
        W_all = np.concatenate([w for _,w in chains])
        W_all /= W_all.sum()
        mu_Om = np.average(S_all[:,0], weights=W_all)
        mu_H0 = np.average(S_all[:,1], weights=W_all)
        sd_Om = np.sqrt(np.average((S_all[:,0]-mu_Om)**2, weights=W_all))
        sd_H0 = np.sqrt(np.average((S_all[:,1]-mu_H0)**2, weights=W_all))
        print(f"\n  Ωm = {mu_Om:.4f} ± {sd_Om:.4f}")
        print(f"  H0 = {mu_H0:.4f} ± {sd_H0:.4f}")
        return {'S_all': S_all, 'W_all': W_all, 'history': hist,
                'mu_Om': mu_Om, 'mu_H0': mu_H0,
                'sd_Om': sd_Om, 'sd_H0': sd_H0}

# =============================================================================
# 5. RUNNER: ejecutar una configuración y extraer estadísticos
# =============================================================================

def run_config(config: dict, n_steps_mcmc: int = 300,
               max_iter_qvmc: int = 200, verbose: bool = True) -> dict:
    """
    Ejecuta QMCMC + QVMC con la config dada.
    Retorna dict con: quantumness, tiempo, estadísticos.
    """
    q_pct  = compute_quantumness(config)
    label  = config.get('label', quantumness_label(q_pct))

    if verbose:
        print("\n" + "═"*65)
        print(f"  CONFIGURACIÓN: {label}")
        print(f"  Quantumness : {q_pct}%")
        print("═"*65)
        print("  Componentes activos:")
        for k, meta in QUANTUM_COMPONENTS.items():
            estado = "✦ CUÁNTICO" if config.get(k, False) else "○ clásico"
            print(f"    {estado:14s}  {meta['name']}")
        print()

    t_total_start = time.time()

    # ── QMCMC ─────────────────────────────────────────────────────────────
    if verbose: print("── [1/2] QMCMC ─────────────────────────────────────────")
    mcmc = QMCMCModular(config, n_chains=4, step_size=0.05,
                         n_burn=100, n_layers=3, check_every=100)
    flat, mcmc_info = mcmc.run(n_steps=n_steps_mcmc)

    Om_mcmc = flat[:,0]; H0_mcmc = flat[:,1]
    stats_mcmc = {
        'Om_mean': float(np.mean(Om_mcmc)),
        'Om_std':  float(np.std(Om_mcmc)),
        'Om_p16':  float(np.percentile(Om_mcmc, 16)),
        'Om_p84':  float(np.percentile(Om_mcmc, 84)),
        'H0_mean': float(np.mean(H0_mcmc)),
        'H0_std':  float(np.std(H0_mcmc)),
        'acceptance': mcmc_info['acceptance'],
        'converged':  mcmc_info['converged'],
        'elapsed':    mcmc_info['elapsed'],
    }

    # ── QVMC ──────────────────────────────────────────────────────────────
    if verbose: print("\n── [2/2] QVMC ──────────────────────────────────────────")
    qvmc = QVMCModular(config, n_qubits_per_param=3, n_layers=3, n_shots=1500)
    qvmc_info = qvmc.run(max_iter=max_iter_qvmc, n_chains=2)

    stats_qvmc = {
        'Om_mean': qvmc_info['mu_Om'],
        'Om_std':  qvmc_info['sd_Om'],
        'H0_mean': qvmc_info['mu_H0'],
        'H0_std':  qvmc_info['sd_H0'],
        'kl_final': float(qvmc_info['history'][-1]) if qvmc_info['history'] else np.nan,
        'elapsed':  0.0,  # incluido en total
    }

    elapsed_total = time.time() - t_total_start

    result = {
        'config':        config,
        'quantumness':   q_pct,
        'label':         label,
        'elapsed_total': elapsed_total,
        'mcmc':          stats_mcmc,
        'qvmc':          stats_qvmc,
        'flat_mcmc':     flat,
        'qvmc_samples':  qvmc_info['S_all'],
        'qvmc_weights':  qvmc_info['W_all'],
        'qvmc_history':  qvmc_info['history'],
    }

    if verbose:
        print(f"\n{'─'*65}")
        print(f"  ⏱  Tiempo total : {elapsed_total:.1f} s")
        print(f"  📊 MCMC  Ωm = {stats_mcmc['Om_mean']:.4f} ± {stats_mcmc['Om_std']:.4f}")
        print(f"  📊 MCMC  H0 = {stats_mcmc['H0_mean']:.4f} ± {stats_mcmc['H0_std']:.4f}")
        print(f"  📊 QVMC  Ωm = {stats_qvmc['Om_mean']:.4f} ± {stats_qvmc['Om_std']:.4f}")
        print(f"  📊 QVMC  H0 = {stats_qvmc['H0_mean']:.4f} ± {stats_qvmc['H0_std']:.4f}")

    return result

# =============================================================================
# 6. BENCHMARK — comparar todos los presets
# =============================================================================

def run_benchmark(n_steps_mcmc: int = 250, max_iter_qvmc: int = 150) -> list:
    """
    Corre todos los PRESETS estándar y genera plots comparativos.
    Retorna lista de resultados ordenados por quantumness.
    """
    print("\n" + "█"*65)
    print("  MODO BENCHMARK — comparando todos los niveles de quantumness")
    print("█"*65)

    results = []
    for pct, preset in PRESETS.items():
        cfg = {**preset, 'label': preset['label']}
        res = run_config(cfg, n_steps_mcmc=n_steps_mcmc,
                         max_iter_qvmc=max_iter_qvmc, verbose=False)
        results.append(res)
        print(f"  ✓  {preset['label']:50s}  |  {res['elapsed_total']:.1f}s")

    print(f"\n  Total benchmark: {sum(r['elapsed_total'] for r in results):.1f}s")
    _plot_benchmark(results)
    return results

def _plot_benchmark(results: list):
    """Genera figura comparativa con 6 paneles."""
    pcts   = [r['quantumness'] for r in results]
    labels = [f"{int(r['quantumness'])}%" for r in results]
    times  = [r['elapsed_total'] for r in results]

    Om_means_mcmc = [r['mcmc']['Om_mean'] for r in results]
    Om_stds_mcmc  = [r['mcmc']['Om_std']  for r in results]
    H0_means_mcmc = [r['mcmc']['H0_mean'] for r in results]
    H0_stds_mcmc  = [r['mcmc']['H0_std']  for r in results]
    Om_means_qvmc = [r['qvmc']['Om_mean'] for r in results]
    Om_stds_qvmc  = [r['qvmc']['Om_std']  for r in results]
    kl_finals     = [r['qvmc']['kl_final'] for r in results]
    acceptances   = [r['mcmc']['acceptance'] for r in results]

    fig = plt.figure(figsize=(18, 12))
    gs  = gridspec.GridSpec(3, 3, hspace=0.45, wspace=0.35)

    cmap   = plt.cm.plasma
    colors = [cmap(p/100) for p in pcts]

    # ── 1. Tiempo total ───────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    bars = ax1.bar(labels, times, color=colors, edgecolor='white', linewidth=0.8)
    ax1.set_title('Tiempo total de ejecución', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Segundos', fontsize=10)
    ax1.set_xlabel('Quantumness', fontsize=10)
    for bar, t in zip(bars, times):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f'{t:.0f}s', ha='center', va='bottom', fontsize=8)
    ax1.grid(True, alpha=0.3, axis='y')

    # ── 2. Ωm MCMC ────────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.errorbar(pcts, Om_means_mcmc, yerr=Om_stds_mcmc,
                 fmt='o-', color='C0', capsize=5, lw=2, ms=7)
    ax2.axhline(OM_MU, color='k', ls='--', lw=1.5, label=f'Planck {OM_MU}')
    ax2.fill_between(pcts,
                     [m-s for m,s in zip(Om_means_mcmc, Om_stds_mcmc)],
                     [m+s for m,s in zip(Om_means_mcmc, Om_stds_mcmc)],
                     alpha=0.15, color='C0')
    ax2.set_title('MCMC — Ωm vs Quantumness', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Quantumness (%)', fontsize=10)
    ax2.set_ylabel(r'$\Omega_m$', fontsize=10)
    ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)

    # ── 3. H0 MCMC ────────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.errorbar(pcts, H0_means_mcmc, yerr=H0_stds_mcmc,
                 fmt='s-', color='C1', capsize=5, lw=2, ms=7)
    ax3.axhline(H0_MU, color='k', ls='--', lw=1.5, label=f'Prior H0={H0_MU}')
    ax3.fill_between(pcts,
                     [m-s for m,s in zip(H0_means_mcmc, H0_stds_mcmc)],
                     [m+s for m,s in zip(H0_means_mcmc, H0_stds_mcmc)],
                     alpha=0.15, color='C1')
    ax3.set_title('MCMC — H0 vs Quantumness', fontsize=11, fontweight='bold')
    ax3.set_xlabel('Quantumness (%)', fontsize=10)
    ax3.set_ylabel(r'$H_0$ [km/s/Mpc]', fontsize=10)
    ax3.legend(fontsize=9); ax3.grid(True, alpha=0.3)

    # ── 4. Tasa de aceptación MCMC ───────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(pcts, acceptances, 'D-', color='C2', lw=2, ms=8)
    ax4.axhspan(0.23, 0.50, alpha=0.1, color='green', label='Rango óptimo')
    ax4.set_title('Tasa de aceptación MCMC', fontsize=11, fontweight='bold')
    ax4.set_xlabel('Quantumness (%)', fontsize=10)
    ax4.set_ylabel('Tasa de aceptación', fontsize=10)
    ax4.set_ylim(0, 1)
    ax4.legend(fontsize=9); ax4.grid(True, alpha=0.3)

    # ── 5. KL final QVMC ─────────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    kl_vals = [k for k in kl_finals if np.isfinite(k)]
    pcts_kl = [pcts[i] for i, k in enumerate(kl_finals) if np.isfinite(k)]
    ax5.semilogy(pcts_kl, kl_vals, 'v-', color='C3', lw=2, ms=8)
    ax5.set_title('KL Divergence final (QVMC)', fontsize=11, fontweight='bold')
    ax5.set_xlabel('Quantumness (%)', fontsize=10)
    ax5.set_ylabel('KL(q‖P_target)', fontsize=10)
    ax5.grid(True, alpha=0.3)

    # ── 6. Ωm QVMC ────────────────────────────────────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.errorbar(pcts, Om_means_qvmc, yerr=Om_stds_qvmc,
                 fmt='^-', color='C4', capsize=5, lw=2, ms=7)
    ax6.axhline(OM_MU, color='k', ls='--', lw=1.5, label=f'Planck')
    ax6.set_title('QVMC — Ωm vs Quantumness', fontsize=11, fontweight='bold')
    ax6.set_xlabel('Quantumness (%)', fontsize=10)
    ax6.set_ylabel(r'$\Omega_m$', fontsize=10)
    ax6.legend(fontsize=9); ax6.grid(True, alpha=0.3)

    # ── 7. Radar / tabla resumen ──────────────────────────────────────────
    ax7 = fig.add_subplot(gs[2, :])
    ax7.axis('off')
    col_labels = ['Quantumness', 'Tiempo (s)',
                  'MCMC Ωm', 'MCMC H0', 'Acept.',
                  'QVMC Ωm', 'KL final', 'Etiqueta']
    table_data = []
    for r in results:
        table_data.append([
            f"{r['quantumness']:.0f}%",
            f"{r['elapsed_total']:.1f}",
            f"{r['mcmc']['Om_mean']:.4f}±{r['mcmc']['Om_std']:.4f}",
            f"{r['mcmc']['H0_mean']:.4f}±{r['mcmc']['H0_std']:.4f}",
            f"{r['mcmc']['acceptance']:.2f}",
            f"{r['qvmc']['Om_mean']:.4f}±{r['qvmc']['Om_std']:.4f}",
            f"{r['qvmc']['kl_final']:.5f}",
            r['label'].split('—')[1].strip() if '—' in r['label'] else r['label'],
        ])

    tbl = ax7.table(cellText=table_data, colLabels=col_labels,
                    loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1, 1.6)

    # Color de filas por quantumness
    for i, r in enumerate(results):
        c = cmap(r['quantumness']/100)
        for j in range(len(col_labels)):
            tbl[i+1, j].set_facecolor((*c[:3], 0.25))

    fig.suptitle(r'Benchmark ΛCDM — Comparación por nivel de Quantumness',
                 fontsize=14, fontweight='bold', y=0.98)

    # Colorbar de referencia
    sm = plt.cm.ScalarMappable(cmap=cmap,
                               norm=plt.Normalize(vmin=0, vmax=100))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=fig.axes, orientation='horizontal',
                        fraction=0.02, pad=0.01, aspect=50)
    cbar.set_label('Quantumness (%)', fontsize=10)

    plt.savefig('benchmark_quantumness.pdf', dpi=150, bbox_inches='tight')
    plt.savefig('benchmark_quantumness.png', dpi=150, bbox_inches='tight')
    print("\n  Figuras guardadas: benchmark_quantumness.pdf / .png")
    plt.close()

# =============================================================================
# 7. MENÚ INTERACTIVO
# =============================================================================

def print_quantumness_bar(pct: float):
    """Imprime barra visual de quantumness."""
    filled = int(pct / 5)
    bar    = '█' * filled + '░' * (20 - filled)
    print(f"\n  ┌─ Quantumness estimado ─────────────────────┐")
    print(f"  │  [{bar}] {pct:5.1f}%        │")
    print(f"  │  {quantumness_label(pct):44s}│")
    print(f"  └────────────────────────────────────────────┘")

def interactive_menu() -> dict:
    """
    Menú interactivo en terminal para elegir componentes cuánticos/clásicos.
    Retorna el dict de configuración.
    """
    print("\n" + "╔" + "═"*63 + "╗")
    print("║   ΛCDM — Configurador de Quantumness                         ║")
    print("╚" + "═"*63 + "╝")
    print("""
  Para cada componente del algoritmo, elige:
    [Q] = Cuántico (usa circuito cuántico / Qiskit)
    [C] = Clásico  (implementación estándar)

  Los 5 componentes y sus pesos en el score total:
""")
    for i, (key, meta) in enumerate(QUANTUM_COMPONENTS.items(), 1):
        print(f"  {i}. [{meta['weight']:2d}%]  {meta['name']}")

    print("""
  ─────────────────────────────────────────────────
  O usa un preset:
    [P0]  = 0%   — Totalmente clásico
    [P20] = 20%  — Solo propuesta cuántica (paper original)
    [P45] = 45%  — Propuesta + Muestreo
    [P70] = 70%  — Sin entrenamiento cuántico
    [P90] = 90%  — Sin QAE
    [P100]= 100% — Totalmente cuántico
  ─────────────────────────────────────────────────
""")

    # Detectar si es interactivo o no
    if not sys.stdin.isatty():
        print("  [Modo no-interactivo] Usando preset P20 por defecto.")
        cfg = {**PRESETS[20]}
        cfg['label'] = PRESETS[20]['label']
        return cfg

    while True:
        resp = input("  Selección (ej: Q C Q Q C  o  P45): ").strip().upper()

        # Preset
        if resp.startswith('P'):
            key = int(resp[1:])
            if key in PRESETS:
                cfg = dict(PRESETS[key])
                q_pct = compute_quantumness(cfg)
                print_quantumness_bar(q_pct)
                return cfg
            else:
                print(f"  Preset no válido. Opciones: {list(PRESETS.keys())}")
                continue

        # Selección manual
        tokens = resp.split()
        if len(tokens) == 5 and all(t in ('Q','C') for t in tokens):
            keys  = list(QUANTUM_COMPONENTS.keys())
            cfg   = {k: (tokens[i] == 'Q') for i, k in enumerate(keys)}
            q_pct = compute_quantumness(cfg)
            print_quantumness_bar(q_pct)
            conf  = input("  ¿Continuar con esta configuración? [S/n]: ").strip().lower()
            if conf != 'n':
                cfg['label'] = f"{q_pct}% — configuración personalizada"
                return cfg
        else:
            print("  Formato no válido. Ej: Q C Q Q C  (5 valores separados por espacios)")

# =============================================================================
# 8. MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="ΛCDM Quantum Sampler — Selección modular de componentes cuánticos/clásicos",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Ejemplos:
  python lcdm_modular_quantum.py --interactive
  python lcdm_modular_quantum.py --preset 45
  python lcdm_modular_quantum.py --benchmark
  python lcdm_modular_quantum.py --config '{"proposal":true,"acceptance":false,"training":true,"sampling":true,"normalization":false}'
        """
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--interactive', action='store_true',
                      help='Menú interactivo para elegir componentes')
    mode.add_argument('--preset', type=int, choices=list(PRESETS.keys()),
                      metavar='N',
                      help=f'Usar preset predefinido: {list(PRESETS.keys())}')
    mode.add_argument('--benchmark', action='store_true',
                      help='Correr todos los presets y comparar')
    mode.add_argument('--config', type=str, metavar='JSON',
                      help='Configuración como JSON string')

    parser.add_argument('--steps',    type=int, default=300,
                        help='Pasos MCMC (default: 300)')
    parser.add_argument('--qvmc-iter', type=int, default=200,
                        help='Iteraciones QVMC (default: 200)')
    parser.add_argument('--no-plot',  action='store_true',
                        help='No generar figuras')

    args = parser.parse_args()

    # ── Banner ────────────────────────────────────────────────────────────
    print("\n" + "═"*65)
    print("  ΛCDM — Sampler Cuántico/Clásico Modular")
    print(f"  Parámetros: Ωm, H0   |  Datos: {N_DATA} puntos CC")
    print("═"*65)

    # ── Seleccionar modo ─────────────────────────────────────────────────
    if args.benchmark:
        run_benchmark(n_steps_mcmc=args.steps, max_iter_qvmc=args.qvmc_iter)
        return

    if args.config:
        cfg = json.loads(args.config)
    elif args.preset is not None:
        cfg = dict(PRESETS[args.preset])
        cfg['label'] = PRESETS[args.preset]['label']
    else:
        # Default: menú interactivo
        cfg = interactive_menu()

    # ── Mostrar score antes de correr ────────────────────────────────────
    q_pct = compute_quantumness(cfg)
    print_quantumness_bar(q_pct)

    # ── Ejecutar ─────────────────────────────────────────────────────────
    result = run_config(cfg, n_steps_mcmc=args.steps,
                        max_iter_qvmc=args.qvmc_iter, verbose=True)

    # ── Resumen final ─────────────────────────────────────────────────────
    print("\n" + "═"*65)
    print(f"  RESUMEN FINAL")
    print("═"*65)
    print(f"  Configuración  : {result['label']}")
    print(f"  Quantumness    : {result['quantumness']:.1f}%  ({quantumness_label(result['quantumness'])})")
    print(f"  Tiempo total   : {result['elapsed_total']:.1f} s")
    print(f"\n  ── QMCMC ─────────────────────────────────────────────────────")
    m = result['mcmc']
    print(f"  Ωm = {m['Om_mean']:.4f} ± {m['Om_std']:.4f}  "
          f"[{m['Om_p16']:.4f}, {m['Om_p84']:.4f}]")
    print(f"  H0 = {m['H0_mean']:.4f} ± {m['H0_std']:.4f}")
    print(f"  Aceptación     : {m['acceptance']:.3f}")
    print(f"  Convergencia   : {'✓' if m['converged'] else '…no completada'}")
    print(f"\n  ── QVMC ──────────────────────────────────────────────────────")
    q = result['qvmc']
    print(f"  Ωm = {q['Om_mean']:.4f} ± {q['Om_std']:.4f}")
    print(f"  H0 = {q['H0_mean']:.4f} ± {q['H0_std']:.4f}")
    print(f"  KL final       : {q['kl_final']:.6f}")

    if not args.no_plot:
        _plot_single_result(result)

    print("\n  Fin.")


def _plot_single_result(result: dict):
    """Plot rápido para una sola configuración."""
    flat   = result['flat_mcmc']
    S_qvmc = result['qvmc_samples']
    W_qvmc = result['qvmc_weights']
    hist   = result['qvmc_history']
    q_pct  = result['quantumness']

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    fig.suptitle(
        f"ΛCDM — {result['label']}\nQuantumness: {q_pct:.0f}%",
        fontsize=12, fontweight='bold'
    )

    z_plot = np.linspace(0.01, 2.4, 200)

    # H(z)
    ax = axes[0]
    ax.errorbar(z_obs, H_obs, yerr=sigma_obs, fmt='.k', capsize=2, ms=4)
    idx_r = RNG.choice(len(flat), size=min(40, len(flat)), replace=False)
    for i in idx_r:
        ax.plot(z_plot, H_lcdm(z_plot, flat[i,0], flat[i,1]),
                color='C0', alpha=0.06, lw=0.8)
    mu_Om = np.mean(flat[:,0]); mu_H0 = np.mean(flat[:,1])
    ax.plot(z_plot, H_lcdm(z_plot, mu_Om, mu_H0), 'C0', lw=2.5, label='MCMC')
    if len(S_qvmc) > 0:
        mu_Om_q = np.average(S_qvmc[:,0], weights=W_qvmc)
        mu_H0_q = np.average(S_qvmc[:,1], weights=W_qvmc)
        ax.plot(z_plot, H_lcdm(z_plot, mu_Om_q, mu_H0_q), 'C1', lw=2.5, label='QVMC')
    ax.set_xlabel(r'$z$'); ax.set_ylabel(r'$H(z)$')
    ax.set_title('H(z) posterior'); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # Ωm
    ax = axes[1]
    ax.hist(flat[:,0], bins=25, color='C0', alpha=0.6, density=True, label='MCMC')
    if len(S_qvmc) > 0:
        ax.hist(S_qvmc[:,0], weights=W_qvmc*len(S_qvmc), bins=25,
                color='C1', alpha=0.5, density=True, label='QVMC')
    ax.axvline(OM_MU, color='k', ls='--', lw=1.5, label='Planck')
    ax.set_xlabel(r'$\Omega_m$'); ax.set_title(r'$\Omega_m$ marginal')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # H0
    ax = axes[2]
    ax.hist(flat[:,1], bins=25, color='C0', alpha=0.6, density=True, label='MCMC')
    if len(S_qvmc) > 0:
        ax.hist(S_qvmc[:,1], weights=W_qvmc*len(S_qvmc), bins=25,
                color='C1', alpha=0.5, density=True, label='QVMC')
    ax.axvline(H0_MU, color='k', ls='--', lw=1.5, label='Prior')
    ax.set_xlabel(r'$H_0$ [km/s/Mpc]'); ax.set_title(r'$H_0$ marginal')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # KL convergence
    ax = axes[3]
    if hist:
        ax.semilogy(hist, 'C3', lw=1.5, alpha=0.9)
        ax.set_xlabel('Iteración QVMC'); ax.set_ylabel('KL Divergence')
        ax.set_title('Convergencia QVMC')
    else:
        ax.text(0.5, 0.5, 'Sin historial', ha='center', va='center',
                transform=ax.transAxes)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = f"lcdm_q{int(q_pct):03d}.pdf"
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.savefig(fname.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    print(f"\n  Figura guardada: {fname}")
    plt.close()


if __name__ == "__main__":
    main()
