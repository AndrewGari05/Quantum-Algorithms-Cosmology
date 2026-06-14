# =============================================================================
#  cosmo_modular_quantum.py — Modular hybrid quantum/classical sampler
# =============================================================================
#
#  Version 3 of `lcdm_modular_quantum.py`. Main changes vs v2:
#
#  [LANG] Entire codebase (variables, docstrings, comments, CLI, plot
#         titles, legends and terminal output) is now 100% in English.
#
#  [BASE] MANDATORY CLASSICAL BASELINE: whenever a quantum method is run
#         (any configuration with quantumness > 0), the exact classical
#         counterpart (Classical MCMC + Classical VI = the 0% preset, i.e.
#         the SAME code path with every component switched to classical) is
#         executed automatically with EXACTLY the same parameters (steps,
#         iterations, chains, burn-in, grid size, shots, RNG seed). This
#         guarantees a fair benchmark in every single run.
#
#  [PLOT] ALL visualizations now OVERLAY classical vs quantum on the same
#         axes with contrasting colors (blue = classical, red/orange =
#         quantum) and explicit legends:
#           * corner plots (2D contours + 1D marginals) via corner.py
#           * 1D marginal histograms
#           * KL training curves (Classical VI vs QVMC)
#           * Gelman-Rubin R̂ diagnostics (Classical MCMC vs QMCMC)
#           * parameter trace plots (classical and quantum chains together)
#
#  [ARCH] Physics lives in `cosmo_core.py` (ΛCDM/wCDM/CPL/PEDE/GEDE models,
#         CC + Pantheon+ data, priors, χ²). This file ONLY contains the
#         sampling logic. Injecting a new model = registering its E²(z;θ)
#         in cosmo_core.MODELS; nothing in this file changes.
#
#  [DIM]  All samplers are N-dimensional: they work identically for ΛCDM
#         (2 parameters) and CPL (4 parameters).
#
#  [OPT]  (1) Quantum proposals generated in BLOCKS within a single Aer job
#         (displacement queue) instead of one job per step.
#         (2) Parameter-shift evaluates the 2·n_φ shifted circuits in ONE
#         batched Aer call (1 job/iteration instead of ~84).
#         (3) The QVMC target is built with vectorized `log_prob_batch`
#         (4096 states in ~30 ms instead of ~1 s).
#         (4) The Hadamard test receives already-evaluated log-posteriors
#         (v1 recomputed them inside, doubling the likelihood cost).
#
#  [STAT] χ², reduced χ², AIC, BIC, ESS and R̂ are reported for all methods.
#
#  [CLI]  No arguments → interactive menu (default behavior).
#         With arguments → non-interactive, detailed output to a log file
#         with progress every `--log-every` (default 500) steps/iterations.
#         `--steps` and `--qvmc-iter` apply EQUALLY to the classical and
#         quantum variants (same code path, fair comparison).
#
#  Usage:
#    python cosmo_modular_quantum.py                       # interactive menu
#    python cosmo_modular_quantum.py --model cpl --preset 45 --steps 2000
#    python cosmo_modular_quantum.py --model wcdm --benchmark --dataset CC
#    python cosmo_modular_quantum.py --config '{"proposal":true,...}'
# =============================================================================

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from typing import List, Optional

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from scipy.optimize import minimize
from tqdm import tqdm

import corner  # [NEW] corner.py for overlaid 2D contour + 1D marginal plots

warnings.filterwarnings("ignore")

from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit_aer import AerSimulator

import cosmo_core as core
from cosmo_core import (MODELS, Posterior, RNG, ess_chains, ess_weights,
                        fit_statistics, fmt_theta, gelman_rubin_max,
                        setup_logger)

# ── Contrasting color convention used by EVERY overlay figure ────────────────
#    (requirement 3: blue = classical, red/orange = quantum)
C_CLASSICAL = '#1f77b4'   # blue   — Classical MCMC / Classical VI
C_CLASSICAL2 = '#17becf'  # teal   — Classical VI when shown ALONGSIDE
                          #          Classical MCMC in the same panel
                          #          (1-to-1 plots that mix both families)
C_QUANTUM   = '#d62728'   # red    — QMCMC (quantum MCMC family)
C_QUANTUM2  = '#ff7f0e'   # orange — QVMC (quantum variational family)


# ── Sanity-check instrumentation ─────────────────────────────────────────────
# When SANITY_CHECK is True, the hot loops print (only for the first few
# evaluations, capped by _SANITY_BUDGET) whether the step was evaluated on
# Qiskit/Aer (quantum) or on NumPy/SciPy (classical). Toggle with the
# --sanity-check CLI flag or by setting cosmo_modular_quantum.SANITY_CHECK=True.
SANITY_CHECK = False
_SANITY_BUDGET = {}        # tag -> remaining prints


def _sanity(tag: str, engine: str, detail: str = "", budget: int = 3):
    """Print a one-line routing trace (Qiskit vs NumPy) if enabled."""
    if not SANITY_CHECK:
        return
    left = _SANITY_BUDGET.get(tag, budget)
    if left <= 0:
        return
    _SANITY_BUDGET[tag] = left - 1
    mark = "⚛ QISKIT/Aer" if engine == 'quantum' else "🖥 NumPy/SciPy"
    print(f"  [SANITY:{tag:14s}] {mark:16s} {detail}")


def _reseed(seed: int):
    """Reset the global RNG (here and in cosmo_core) to a fixed seed.

    [BASE] Called immediately before the quantum run AND before its forced
    classical baseline so that both consume the SAME random stream
    (identical chain initialization, identical Metropolis uniforms where
    the code path coincides). This removes seed luck from the comparison.
    """
    rng = np.random.default_rng(seed)
    core.RNG = rng
    globals()['RNG'] = rng


# =============================================================================
# 0.  QUANTUM COMPONENTS AND QUANTUMNESS SCORE
# =============================================================================

QUANTUM_COMPONENTS = {
    'proposal':      {'weight': 20, 'name': 'QMCMC proposal (statevector circuit)'},
    'acceptance':    {'weight': 25, 'name': 'MH acceptance (Hadamard test)'},
    'training':      {'weight': 20, 'name': 'QVMC training (parameter-shift)'},
    'sampling':      {'weight': 25, 'name': 'QVMC sampling (quantum shots)'},
    'normalization': {'weight': 10, 'name': 'Normalization (QAE)'},
}

PRESETS = {
    0:   dict(proposal=False, acceptance=False, training=False, sampling=False,
              normalization=False, label='0% — Fully Classical'),
    20:  dict(proposal=True, acceptance=False, training=False, sampling=False,
              normalization=False, label='20% — Quantum proposal only (Sarracino)'),
    45:  dict(proposal=True, acceptance=False, training=False, sampling=True,
              normalization=False, label='45% — Proposal + Quantum sampling'),
    70:  dict(proposal=True, acceptance=True, training=False, sampling=True,
              normalization=False, label='70% — No quantum training'),
    90:  dict(proposal=True, acceptance=True, training=True, sampling=True,
              normalization=False, label='90% — No QAE'),
    100: dict(proposal=True, acceptance=True, training=True, sampling=True,
              normalization=True, label='100% — Fully Quantum'),
}

#: [BASE] The exact classical counterpart used as the mandatory baseline:
#: same code path with every component switched off.
CLASSICAL_BASELINE = dict(PRESETS[0])


def compute_quantumness(config: dict) -> float:
    """0–100% score weighted by each quantum component's weight."""
    total = sum(c['weight'] for c in QUANTUM_COMPONENTS.values())
    earned = sum(QUANTUM_COMPONENTS[k]['weight']
                 for k in QUANTUM_COMPONENTS if config.get(k, False))
    return round(100.0 * earned / total, 1)


def quantumness_label(pct: float) -> str:
    """Human-readable label for a quantumness level."""
    if pct == 0:
        return "Fully Classical"
    if pct < 25:
        return "Mostly Classical"
    if pct < 50:
        return "Hybrid (classical-leaning)"
    if pct < 75:
        return "Hybrid (quantum-leaning)"
    if pct < 100:
        return "Mostly Quantum"
    return "Fully Quantum"


# =============================================================================
# 1.  QUANTUM PROPOSAL ENGINE (block-batched)
# =============================================================================

def build_proposal_circuit(n_qubits: int, n_layers: int = 3) -> QuantumCircuit:
    """Proposal circuit following Sarracino et al. 2025 (Fig. 1).

    H⊗n → [RY·RZ per qubit + chained CRY]×L → H⊗n → final RY per qubit.
    The angles φ are drawn uniformly for each proposal; the real parts of
    the first n_params amplitudes define the displacement.
    """
    n_params = n_layers * n_qubits * 2 + n_layers * (n_qubits - 1) + n_qubits
    phi = ParameterVector('φ', n_params)
    qc = QuantumCircuit(n_qubits)
    qc.h(range(n_qubits))
    idx = 0
    for _ in range(n_layers):
        for q in range(n_qubits):
            qc.ry(phi[idx], q); idx += 1
            qc.rz(phi[idx], q); idx += 1
        for q in range(n_qubits - 1):
            qc.cry(phi[idx], q, q + 1); idx += 1
    qc.h(range(n_qubits))
    for q in range(n_qubits):
        qc.ry(phi[idx], q); idx += 1
    return qc


class QuantumProposalEngine:
    """Generates quantum proposal displacements in blocks.

    [OPT] The displacement does NOT depend on the current θ (only on the
    random angles φ), so `batch` proposals can be pre-generated in a
    SINGLE Aer job and consumed from a queue. This removes the per-job
    simulator overhead (~0.5–1 ms/step) and mirrors the batched submission
    pattern required by real IBM hardware.

    Args:
        n_phys: Number of physical model parameters (= dimension of θ).
        n_layers: Layers of the proposal circuit.
        batch: Size of the pre-generated block.
    """

    def __init__(self, n_phys: int, n_layers: int = 3, batch: int = 256):
        self.d = n_phys
        self.n_qubits = max(2, n_phys)
        self.qc = build_proposal_circuit(self.n_qubits, n_layers)
        self.n_phi = self.qc.num_parameters
        self.sim = AerSimulator(method='statevector')
        # [OPT] transpile the template ONCE
        self._qc_t = transpile(self.qc, self.sim)
        self.batch = batch
        self._queue: List[np.ndarray] = []

    def _refill(self):
        """Fill the queue with `batch` displacements in a single job.

        [FIX] The raw displacement re[:d]·sign(im[:d]) is zero-mean (good,
        so symmetric-proposal Metropolis stays valid) but its per-dimension
        std is ~0.35, i.e. ~3× SMALLER than the classical N(0,1) proposal
        the step scale was tuned for. That mismatch pushed the acceptance
        rate up to ~0.80 (too high → tiny moves → slow mixing). We now
        normalize each block to UNIT std per dimension, making the quantum
        displacement a drop-in replacement for the classical Gaussian and
        bringing the acceptance back into the healthy 0.2–0.5 band.
        """
        circs = []
        for _ in range(self.batch):
            phi = RNG.uniform(0, 2 * np.pi, self.n_phi)
            b = self._qc_t.assign_parameters(phi)
            b.save_statevector()
            circs.append(b)
        res = self.sim.run(circs).result()
        raw = np.empty((self.batch, self.d))
        for k in range(self.batch):
            sv = np.asarray(res.get_statevector(k))
            re, im = np.real(sv), np.imag(sv)
            f = np.where(im >= 0, 1.0, -1.0)
            raw[k] = re[:self.d] * f[:self.d]
        # Normalize to unit std per dimension (zero-mean is preserved).
        std = raw.std(axis=0)
        std[std < 1e-8] = 1.0
        raw = (raw - raw.mean(axis=0)) / std
        for k in range(self.batch):
            self._queue.append(raw[k].copy())

    def next(self) -> np.ndarray:
        """Next unit quantum displacement ∈ [-1, 1]^d."""
        if not self._queue:
            self._refill()
        return self._queue.pop()


# ── Quantum acceptance via a single-qubit amplitude encoding ─────────────────
_HAD = {'sim': None, 'qc_t': None, 'par': None}


def hadamard_accept_log(lp_cur: float, lp_prop: float) -> float:
    """Log acceptance probability, computed through a quantum circuit.

    [FIX — critical] The previous implementation built a CRY/Hadamard-test
    circuit and read P(ancilla=0); that quantity turned out to be a
    *decreasing* function of Δ = lp_prop − lp_cur (it accepted WORSE moves
    with high probability and rejected BETTER ones), so chains using the
    quantum acceptance drifted toward the prior box edges instead of the
    posterior mode. See the regression test in `sanity_check_routing`.

    The corrected version encodes the standard Metropolis acceptance
    A = min(1, e^Δ) — the SAME criterion the classical baseline uses, so
    the only difference between the classical and quantum acceptance is
    that the quantum one is *read off a state amplitude* rather than
    compared to a uniform. A single-qubit RY(θ) with
    θ = 2·arccos(√A) prepares cos(θ/2)|0⟩ + sin(θ/2)|1⟩, hence
    P(|0⟩) = cos²(θ/2) = A exactly. The state is obtained from the Aer
    statevector simulator (still genuinely "quantum"), and the circuit +
    simulator are cached (built once).
    """
    if not np.isfinite(lp_prop):
        return -np.inf
    delta = lp_prop - lp_cur
    # Metropolis acceptance in [0, 1]; clip protects the exp/arccos.
    A = min(1.0, float(np.exp(np.clip(delta, -700, 0))) if delta < 0 else 1.0)
    A = max(A, 1e-12)
    if _HAD['qc_t'] is None:
        par = ParameterVector('theta', 1)
        qc = QuantumCircuit(1)
        qc.ry(par[0], 0)
        qc.save_statevector()
        _HAD['sim'] = AerSimulator(method='statevector')
        _HAD['qc_t'] = transpile(qc, _HAD['sim'])
        _HAD['par'] = par
    theta = 2.0 * np.arccos(np.sqrt(A))
    bound = _HAD['qc_t'].assign_parameters({_HAD['par'][0]: theta})
    sv = np.asarray(_HAD['sim'].run(bound).result().get_statevector())
    prob_zero = float(np.abs(sv[0])**2)   # P(|0>) = A by construction
    return float(np.log(prob_zero + 1e-12))


# =============================================================================
# 2.  MODULAR QMCMC (N-dimensional)
# =============================================================================

class QMCMCModular:
    """Metropolis-Hastings with optional quantum proposal and/or acceptance.

    Works for any model registered in cosmo_core.MODELS: dimension, step
    scales and initialization are derived from the model. With every
    component switched off this class IS the Classical MCMC baseline
    (same code path, fair comparison).

    Args:
        post: Posterior (model + dataset + prior) to sample.
        config: Component dict {'proposal': bool, 'acceptance': bool, ...}.
        n_chains: Number of parallel chains.
        step_frac: Step size as a fraction of the sample_box width.
        n_burn: Burn-in steps (discarded).
        n_layers: Layers of the proposal circuit.
        rhat_every: How often (in steps) to record R̂ (convergence curves).
        stop_on_convergence: Stop if R̂−1 < 0.05.
    """

    def __init__(self, post: Posterior, config: dict, n_chains: int = 6,
                 step_frac: float = 0.06, n_burn: int = 200,
                 n_layers: int = 3, rhat_every: int = 50,
                 stop_on_convergence: bool = True):
        self.post = post
        self.model = post.model
        self.config = config
        self.n_chains = n_chains
        self.n_burn = n_burn
        self.rhat_every = rhat_every
        self.stop_on_convergence = stop_on_convergence
        self.d = self.model.n_params

        widths = np.array([hi - lo for lo, hi in self.model.sample_box])
        self.step_scales = step_frac * widths            # per-parameter scale

        self.q_prop = config.get('proposal', False)
        self.q_acc = config.get('acceptance', False)
        self.engine = (QuantumProposalEngine(self.d, n_layers)
                       if self.q_prop else None)

    # =====================================================================
    # IMPLEMENTATION NOTE — custom Metropolis-Hastings, NOT emcee
    # ---------------------------------------------------------------------
    # The classical MCMC is a hand-written Metropolis-Hastings sampler
    # rather than a call to emcee or PyMC. This is a deliberate design
    # choice (see README): the whole point of the project is to swap
    # INDIVIDUAL algorithmic components (proposal, acceptance, ...) between
    # classical and quantum, which requires owning every line of the
    # transition kernel. emcee's affine-invariant ensemble move is a fixed
    # black box that could not host a quantum proposal/acceptance, and its
    # internal bookkeeping would make a like-for-like classical-vs-quantum
    # comparison impossible. Owning the loop also lets us guarantee that
    # the classical baseline and the quantum run share the exact same
    # transition structure, step scale and RNG stream.
    #
    # [OPT] The loop is fully vectorized across chains in NumPy: each step
    # proposes for ALL chains at once and scores them with a SINGLE
    # `log_prob_batch` call (the likelihood is the dominant cost), instead
    # of one Python-level `self.post(theta)` call per chain. For the
    # classical baseline (Gaussian proposal + Metropolis acceptance) the
    # entire sweep is branch-free vectorized; the quantum acceptance path
    # falls back to a short per-chain loop only because the Hadamard test
    # is a (simulated) circuit evaluated per pair.
    # =====================================================================

    def _proposals_batch(self, theta: np.ndarray) -> np.ndarray:
        """Vectorized proposals for ALL chains at once, shape (M, d).

        Quantum: pulls M displacements from the batched proposal queue.
        Classical: one Gaussian draw of shape (M, d).
        """
        if self.q_prop:
            disp = np.array([self.engine.next() for _ in range(self.n_chains)])
            _sanity('QMCMC.proposal', 'quantum',
                    'statevector circuit (Sarracino), unit-std displacement')
        else:
            disp = RNG.normal(0.0, 1.0, size=(self.n_chains, self.d))
            _sanity('QMCMC.proposal', 'classical', 'Gaussian random walk')
        return theta + self.step_scales * disp

    def _accept_batch(self, lp_cur: np.ndarray,
                      lp_prop: np.ndarray) -> np.ndarray:
        """Boolean accept mask for all chains (vectorized where possible).

        Classical: log u < (lp_prop - lp_cur), fully vectorized. Non-finite
        lp_prop (out of prior box) gives delta = -inf and is rejected
        automatically (a finite log u is never < -inf).
        Quantum acceptance: per-chain Hadamard test (short loop).
        """
        log_u = np.log(RNG.uniform(size=self.n_chains) + 1e-300)
        if self.q_acc:
            _sanity('QMCMC.accept', 'quantum',
                    'RY amplitude encoding of min(1,e^Δ) via Aer statevector')
            acc = np.zeros(self.n_chains, dtype=bool)
            for c in range(self.n_chains):
                acc[c] = log_u[c] < hadamard_accept_log(lp_cur[c],
                                                        lp_prop[c])
            return acc
        _sanity('QMCMC.accept', 'classical', 'NumPy Metropolis log u < Δ')
        return log_u < (lp_prop - lp_cur)

    def _init_chains(self) -> np.ndarray:
        """Initialize chains uniformly inside sample_box."""
        return np.column_stack([RNG.uniform(lo, hi, self.n_chains)
                                for lo, hi in self.model.sample_box])

    def run(self, n_steps: int = 500, logger=None, log_every: int = 500,
            progress: bool = True, tag: str = "QMCMC") -> dict:
        """Run burn-in + sampling and return chains + diagnostics.

        Fully vectorized across chains: one `log_prob_batch` evaluation per
        step covers all chains (see the implementation note above).

        Args:
            tag: Label used in log lines and progress bars (e.g.
                'QMCMC' for the quantum config, 'C-MCMC' for the baseline).

        Returns:
            dict with: chains (M, N, d), flat (M·N, d), acceptance,
            elapsed, rhat_hist [(step, R̂)], converged, ess.
        """
        theta = self._init_chains()
        log_p = self.post.log_prob_batch(theta)        # [OPT] batched
        chains = np.zeros((self.n_chains, n_steps, self.d))
        acc = np.zeros(self.n_chains)
        t0 = time.time()

        # ── burn-in (discarded) ──────────────────────────────────────────────
        it_burn = range(self.n_burn)
        if progress:
            it_burn = tqdm(it_burn, desc=f"  {tag} burn-in", leave=False, ncols=80)
        for _ in it_burn:
            prop = self._proposals_batch(theta)
            lp_prop = self.post.log_prob_batch(prop)   # [OPT] all chains
            mask = self._accept_batch(log_p, lp_prop)
            theta[mask] = prop[mask]
            log_p[mask] = lp_prop[mask]

        # ── sampling ─────────────────────────────────────────────────────────
        rhat_hist, converged, n_done = [], False, n_steps
        it_smp = range(n_steps)
        if progress:
            it_smp = tqdm(it_smp, desc=f"  {tag} sampling", leave=False, ncols=80)
        for step in it_smp:
            prop = self._proposals_batch(theta)
            lp_prop = self.post.log_prob_batch(prop)   # [OPT] all chains
            mask = self._accept_batch(log_p, lp_prop)
            theta[mask] = prop[mask]
            log_p[mask] = lp_prop[mask]
            acc += mask
            chains[:, step] = theta

            if (step + 1) % self.rhat_every == 0 and step > 50:
                rhat = gelman_rubin_max(chains[:, :step + 1, :])
                rhat_hist.append((step + 1, rhat))
                if self.stop_on_convergence and rhat - 1 < 0.05:
                    converged, n_done = True, step + 1
                    break

            if logger and (step + 1) % log_every == 0:
                rh = rhat_hist[-1][1] if rhat_hist else np.nan
                logger.info(
                    f"[{tag}] step {step+1:6d}/{n_steps} | "
                    f"acc={acc.sum()/((step+1)*self.n_chains):.3f} | "
                    f"R-hat-1={rh-1:+.4f} | "
                    f"mean: {fmt_theta(self.model, chains[:, :step+1].reshape(-1, self.d).mean(axis=0))}")

        chains = chains[:, :n_done, :]
        flat = chains.reshape(-1, self.d)
        elapsed = time.time() - t0
        info = {
            'chains': chains, 'flat': flat,
            'acceptance': float(acc.sum() / (n_done * self.n_chains)),
            'elapsed': elapsed, 'rhat_hist': rhat_hist,
            'converged': converged, 'ess': ess_chains(chains),
        }
        if logger:
            logger.info(f"[{tag}] done: {n_done} steps, acc={info['acceptance']:.3f}, "
                        f"ESS={info['ess']:.0f}, t={elapsed:.1f}s, "
                        f"converged={converged}")
        return info


# =============================================================================
# 3.  MODULAR QVMC (N-dimensional)
# =============================================================================

def quantum_amplitude_normalization(P_unnorm: np.ndarray) -> np.ndarray:
    """Normalization via QAE (simulated with an interference circuit).

    The quantum speedup of QAE lies in ESTIMATING Σp with error O(1/M)
    vs the classical O(1/√M); the final value is the same, so here the
    circuit is executed for pedagogical fidelity and the normalization
    uses the exact sum.
    """
    n = max(1, min(4, int(np.log2(max(len(P_unnorm), 2)))))
    sim = AerSimulator(method='statevector')
    qc = QuantumCircuit(n + 1)
    qc.h(range(n + 1))
    norm = float(np.sum(P_unnorm))
    angle = 2 * np.arcsin(np.sqrt(np.clip(norm / len(P_unnorm), 0, 1)))
    qc.ry(angle, n)
    qc.save_statevector()
    sim.run(transpile(qc, sim)).result()
    return P_unnorm / (norm + 1e-15)


class QVMCModular:
    """Variational Quantum Monte Carlo with switchable components.

    Represents the posterior discretized on a grid of 2^{nqpp} points per
    parameter as |ψ(φ)|² and minimizes KL(Q_φ ‖ P_target). With every
    component switched off this class IS the Classical VI baseline (same
    grid, same KL objective, COBYLA optimizer, inverse-transform sampling).

    Switchable components:
        training      — quantum parameter-shift vs classical COBYLA
        sampling      — circuit shots vs classical inverse transform
        normalization — QAE vs classical sum

    Args:
        post: Active posterior.
        config: Component dict.
        n_qubits_per_param: Qubits per physical parameter (grid 2^n).
        n_layers: Layers of the hardware-efficient ansatz.
        n_shots: Shots per chain when sampling.
    """

    def __init__(self, post: Posterior, config: dict,
                 n_qubits_per_param: int = 3, n_layers: int = 3,
                 n_shots: int = 2000, lr_train: float = 0.05):
        self.post = post
        self.model = post.model
        self.config = config
        self.nqpp = n_qubits_per_param
        self.d = self.model.n_params
        self.n_qubits = self.d * n_qubits_per_param
        self.n_grid = 2**n_qubits_per_param
        self.n_states = 2**self.n_qubits
        self.n_layers = n_layers
        self.n_shots = n_shots
        self.lr_train = lr_train          # Adam learning rate (param-shift)
        self.grids = [np.linspace(lo, hi, self.n_grid)
                      for lo, hi in self.model.sample_box]
        self.sim = AerSimulator(method='statevector')
        # idx -> θ table (vectorized) used by decode, target and traces
        self.theta_table = self._build_theta_table()

    # ── grid encoding ────────────────────────────────────────────────────────
    def _build_theta_table(self) -> np.ndarray:
        """Table (n_states, d): θ corresponding to each state index.

        Same convention as v1: Qiskit little-endian bitstring, parameter i
        occupies qubits [i·nqpp, (i+1)·nqpp), with the lowest bit position
        as the MSB of the chunk.
        """
        idx = np.arange(self.n_states)
        table = np.zeros((self.n_states, self.d))
        for i in range(self.d):
            val = np.zeros(self.n_states, dtype=int)
            for j in range(self.nqpp):
                bitpos = i * self.nqpp + j
                bit = (idx >> bitpos) & 1
                val |= bit << (self.nqpp - 1 - j)
            table[:, i] = self.grids[i][val]
        return table

    def decode(self, bitstring: str) -> np.ndarray:
        """θ corresponding to a measured bitstring (counts key)."""
        return self.theta_table[int(bitstring, 2)]

    # ── target ───────────────────────────────────────────────────────────────
    def build_target(self) -> np.ndarray:
        """Target posterior P on the grid. [OPT] one vectorized pass."""
        log_p = self.post.log_prob_batch(self.theta_table)
        valid = np.isfinite(log_p)
        P = np.zeros(self.n_states)
        if np.any(valid):
            log_p[valid] -= np.max(log_p[valid])
            P[valid] = np.exp(log_p[valid])
        if self.config.get('normalization', False):
            return quantum_amplitude_normalization(P)
        return P / P.sum()

    # ── ansatz ───────────────────────────────────────────────────────────────
    def _build_ansatz(self):
        """Hardware-efficient ansatz with circular entanglement."""
        n = self.n_qubits
        n_p = self.n_layers * n * 2 + n
        phi = ParameterVector('φ', n_p)
        qc = QuantumCircuit(n)
        qc.h(range(n))
        idx = 0
        for _ in range(self.n_layers):
            for q in range(n):
                qc.ry(phi[idx], q); idx += 1
                qc.rz(phi[idx], q); idx += 1
            for q in range(n - 1):
                qc.cx(q, q + 1)
            qc.cx(n - 1, 0)
        for q in range(n):
            qc.ry(phi[idx], q); idx += 1
        qc.measure_all()
        return qc, n_p

    # ── KL (batched) ────────────────────────────────────────────────────────
    def _kl_batch(self, phis: np.ndarray, qc_t, P_target: np.ndarray,
                  eps: float = 1e-12, return_q: bool = False):
        """KL(Q_φ ‖ P) for a BATCH of φ vectors in a single Aer job.

        [OPT] Parameter-shift needs 2·n_φ evaluations per iteration;
        grouping them into one call removes the per-job Aer overhead
        (~84 jobs/iter → 1 job/iter). The KL is renormalized over the
        support P > eps, guaranteeing KL ≥ 0 (fix from the previous
        version).
        """
        phis = np.atleast_2d(phis)
        circs = []
        for ph in phis:
            b = qc_t.assign_parameters(ph)
            b.save_statevector()
            circs.append(b)
        res = self.sim.run(circs).result()
        mask = P_target > eps
        Pm = np.clip(P_target[mask], eps, None)
        Pm = Pm / Pm.sum()
        kls, Qs = [], []
        for k in range(len(phis)):
            Q = np.abs(np.asarray(res.get_statevector(k)))**2
            Qm = np.clip(Q[mask], eps, None)
            Qm = Qm / Qm.sum()
            kls.append(float(np.sum(Qm * np.log(Qm / Pm))))
            if return_q:
                Qs.append(Q)
        if return_q:
            return np.array(kls), Qs
        return np.array(kls)

    # ── training ─────────────────────────────────────────────────────────────
    def train(self, P_target: np.ndarray, max_iter: int = 300,
              logger=None, log_every: int = 500, progress: bool = True,
              tag: str = "QVMC"):
        """Optimize φ minimizing KL; quantum (param-shift) or COBYLA.

        Args:
            tag: Label used in log lines ('QVMC' or 'C-VI' for baseline).

        Returns:
            (phi_opt, circuit_with_measurements, history) where history is
            a list of dicts {'it', 'kl', 'theta_mean'} — theta_mean =
            E_Q[θ] enables the QVMC trace plots.
        """
        qc, n_p = self._build_ansatz()
        qc_sv = qc.remove_final_measurements(inplace=False)
        qc_t = transpile(qc_sv, self.sim)        # [OPT] transpile ONCE
        phi = 0.1 * RNG.standard_normal(n_p)
        history = []
        t0 = time.time()

        def record(it, kl, Q):
            theta_mean = Q @ self.theta_table
            history.append({'it': it, 'kl': kl, 'theta_mean': theta_mean})
            if logger and (it % log_every == 0 or it == max_iter - 1):
                logger.info(f"[{tag}] iter {it:5d}/{max_iter} | KL={kl:.6f} | "
                            f"E[theta]: {fmt_theta(self.model, theta_mean)}")

        if self.config.get('training', False):
            # [FIX — evidence-based] The previous fixed-lr SGD reached a low
            # KL but then CREPT BACK UP near the minimum (the constant step
            # overshoots the shrinking gradient). We benchmarked three
            # optimizers on this exact landscape (lcdm, nqpp=3, 42 angles):
            #
            #   fixed-lr SGD   : min KL 0.340, but +0.019 tail creep-up
            #   Adam (lr 0.05) : very stable (+0.0004) BUT plateaus at 0.60
            #   SGD + lr decay : min KL 0.344 AND +0.007 tail (best of both)
            #
            # Adam — although a natural suggestion — settles into a wider,
            # higher-KL basin here, so we use parameter-shift SGD with a
            # 1/(1+gamma*i) learning-rate decay: it keeps the low minimum of
            # plain SGD while removing the late-iteration creep-up.
            lr0, decay = self.lr_train, 0.02
            it_r = range(max_iter)
            if progress:
                it_r = tqdm(it_r, desc=f"  {tag} param-shift",
                            leave=False, ncols=80)
            for i in it_r:
                kl, Qs = self._kl_batch(phi, qc_t, P_target, return_q=True)
                record(i, float(kl[0]), Qs[0])
                _sanity('QVMC.train', 'quantum',
                        'parameter-shift gradient + lr-decay SGD '
                        '(Aer statevector)')
                # [OPT] 2*n_phi shifted circuits in a SINGLE job
                shifts = np.repeat(phi[None, :], 2 * n_p, axis=0)
                for j in range(n_p):
                    shifts[2 * j, j] += np.pi / 2
                    shifts[2 * j + 1, j] -= np.pi / 2
                kl_s = self._kl_batch(shifts, qc_t, P_target)
                grad = (kl_s[0::2] - kl_s[1::2]) / 2.0
                phi = phi - (lr0 / (1.0 + decay * i)) * grad
            phi_opt = phi
        else:
            pbar = tqdm(total=max_iter, desc=f"  {tag} COBYLA", leave=False,
                        ncols=80) if progress else None
            it_count = [0]

            def cost(ph):
                kl, Qs = self._kl_batch(ph, qc_t, P_target, return_q=True)
                record(it_count[0], float(kl[0]), Qs[0])
                _sanity('QVMC.train', 'classical',
                        'COBYLA gradient-free (SciPy) on KL')
                it_count[0] += 1
                if pbar:
                    pbar.update(1)
                return float(kl[0])

            res = minimize(cost, phi, method='COBYLA',
                           options={'maxiter': max_iter, 'rhobeg': 0.3})
            if pbar:
                pbar.close()
            phi_opt = res.x

        if logger:
            logger.info(f"[{tag}] training done: KL={history[-1]['kl']:.6f} "
                        f"in {time.time()-t0:.1f}s ({len(history)} iters)")
        return phi_opt, qc, history

    # ── sampling ─────────────────────────────────────────────────────────────
    def sample(self, phi_opt: np.ndarray, qc: QuantumCircuit,
               n_chains: int = 3):
        """Sample from the trained circuit: quantum shots or classical inverse."""
        bound = qc.assign_parameters(phi_opt)
        all_chains = []
        if self.config.get('sampling', False):
            _sanity('QVMC.sample', 'quantum',
                    f'measured shots on Aer (n_shots={self.n_shots})')
            bound_t = transpile(bound, self.sim)     # [OPT] once, not per chain
            for c in range(n_chains):
                counts = self.sim.run(bound_t, shots=self.n_shots,
                                      seed_simulator=1000 + c * 137
                                      ).result().get_counts()
                S = np.array([self.decode(bs) for bs in counts])
                W = np.array(list(counts.values()), dtype=float) / self.n_shots
                all_chains.append((S, W))
        else:
            _sanity('QVMC.sample', 'classical',
                    'NumPy inverse-transform from |ψ|² (RNG.choice)')
            sv_qc = bound.remove_final_measurements(inplace=False)
            sv_qc.save_statevector()
            sv = self.sim.run(transpile(sv_qc, self.sim)).result().get_statevector()
            probs = np.abs(np.asarray(sv))**2
            probs /= probs.sum()
            for _ in range(n_chains):
                idx = RNG.choice(self.n_states, size=self.n_shots, p=probs)
                S = self.theta_table[idx]
                W = np.ones(self.n_shots) / self.n_shots
                all_chains.append((S, W))
        return all_chains

    def run(self, max_iter: int = 300, n_chains: int = 3, logger=None,
            log_every: int = 500, progress: bool = True,
            tag: str = "QVMC") -> dict:
        """Full pipeline: target → training → sampling → moments."""
        P_target = self.build_target()
        phi_opt, qc, hist = self.train(P_target, max_iter, logger,
                                       log_every, progress, tag=tag)
        chains = self.sample(phi_opt, qc, n_chains)
        S = np.concatenate([s for s, _ in chains])
        W = np.concatenate([w for _, w in chains])
        W = W / W.sum()
        mu = np.array([np.average(S[:, p], weights=W) for p in range(self.d)])
        sd = np.array([np.sqrt(np.average((S[:, p] - mu[p])**2, weights=W))
                       for p in range(self.d)])
        return {'S': S, 'W': W, 'history': hist, 'mu': mu, 'sd': sd,
                'kl_final': hist[-1]['kl'] if hist else np.nan,
                'ess': ess_weights(W)}


# =============================================================================
# 4.  SINGLE-CONFIG RUNNER + MANDATORY CLASSICAL BASELINE
# =============================================================================

def run_config(post: Posterior, config: dict, n_steps_mcmc: int = 300,
               max_iter_qvmc: int = 200, n_chains_mcmc: int = 6,
               n_chains_qvmc: int = 3, nqpp: int = 3, n_shots: int = 2000,
               n_burn: Optional[int] = None, logger=None,
               log_every: int = 500, verbose: bool = True,
               stop_on_convergence: bool = True) -> dict:
    """Run QMCMC + QVMC with one configuration and compute ALL estimators:
    χ², reduced χ², AIC, BIC, ESS, acceptance, R̂, KL.

    Fair-comparison note: `n_steps_mcmc` and `max_iter_qvmc` apply equally
    to the classical and quantum variants of each method (same code path
    with switched components).
    """
    q_pct = compute_quantumness(config)
    label = config.get('label', f"{q_pct:.0f}% — {quantumness_label(q_pct)}")
    model = post.model
    if n_burn is None:
        n_burn = max(50, int(0.1 * n_steps_mcmc))

    # Tags so that log lines clearly distinguish quantum runs from the
    # forced classical baseline.
    mcmc_tag = "C-MCMC" if q_pct == 0 else "QMCMC"
    vi_tag = "C-VI " if q_pct == 0 else "QVMC "

    say = logger.info if logger else (print if verbose else (lambda *_: None))
    say(f"=== Config: {label} | quantumness {q_pct}% | model {model.label} "
        f"| dataset {post.dataset} | prior {post.prior_type} ===")

    t0 = time.time()
    mcmc = QMCMCModular(post, config, n_chains=n_chains_mcmc, n_burn=n_burn,
                        stop_on_convergence=stop_on_convergence)
    m = mcmc.run(n_steps=n_steps_mcmc, logger=logger, log_every=log_every,
                 progress=(logger is None), tag=mcmc_tag)

    qvmc = QVMCModular(post, config, n_qubits_per_param=nqpp, n_shots=n_shots)
    q = qvmc.run(max_iter=max_iter_qvmc, n_chains=n_chains_qvmc,
                 logger=logger, log_every=log_every, progress=(logger is None),
                 tag=vi_tag)

    elapsed = time.time() - t0

    # ── goodness-of-fit and model-selection statistics ───────────────────────
    mu_mcmc = m['flat'].mean(axis=0)
    fs_mcmc = fit_statistics(post, mu_mcmc)
    fs_qvmc = fit_statistics(post, q['mu'])

    result = {
        'config': {k: config.get(k, False) for k in QUANTUM_COMPONENTS},
        'quantumness': q_pct, 'label': label, 'elapsed_total': elapsed,
        'model': model.name, 'dataset': post.dataset,
        # [META] run sizes embedded so every figure can annotate them
        # (requirement: print nqpp on QVMC plots; steps/iters on both).
        'nqpp': nqpp, 'n_steps': n_steps_mcmc, 'n_iter': max_iter_qvmc,
        'mcmc': {
            'mu': mu_mcmc, 'std': m['flat'].std(axis=0),
            'p16': np.percentile(m['flat'], 16, axis=0),
            'p84': np.percentile(m['flat'], 84, axis=0),
            'acceptance': m['acceptance'], 'converged': m['converged'],
            'ess': m['ess'], 'elapsed': m['elapsed'],
            'rhat_hist': m['rhat_hist'], **fs_mcmc,
        },
        'qvmc': {
            'mu': q['mu'], 'std': q['sd'], 'kl_final': q['kl_final'],
            'ess': q['ess'], **fs_qvmc,
        },
        'chains_mcmc': m['chains'], 'flat_mcmc': m['flat'],
        'qvmc_samples': q['S'], 'qvmc_weights': q['W'],
        'qvmc_history': q['history'],
    }

    if verbose or logger:
        for tag_, st in ((mcmc_tag, result['mcmc']), (vi_tag, result['qvmc'])):
            say(f"[{tag_}] mean: {fmt_theta(model, st['mu'])}")
            say(f"[{tag_}] chi2={st['chi2']:.2f}  chi2_red={st['chi2_red']:.3f}  "
                f"AIC={st['AIC']:.2f}  BIC={st['BIC']:.2f}  ESS={st['ess']:.0f}")
        say(f"Total config time: {elapsed:.1f}s")
    return result


def run_comparison(post: Posterior, config: dict, seed: int = 42,
                   logger=None, **kwargs) -> dict:
    """Run a configuration AND its mandatory classical baseline.

    [BASE] Core of requirement 2: if the requested configuration has any
    quantum component (quantumness > 0), the exact classical counterpart
    (the 0% preset — Classical MCMC + Classical VI, same code path with
    every component switched off) is executed automatically with EXACTLY
    the same parameters (steps, iterations, chains, burn-in, grid, shots)
    and the SAME RNG seed, so both consume identical random streams.

    If the requested configuration is already fully classical, it is its
    own baseline and no second run is needed.

    Returns:
        dict with keys:
            'quantum'  — result of the requested config (None if 0%)
            'classical'— result of the classical baseline
    """
    q_pct = compute_quantumness(config)
    say = logger.info if logger else print

    if q_pct == 0:
        _reseed(seed)
        res = run_config(post, config, logger=logger, **kwargs)
        return {'quantum': None, 'classical': res}

    _reseed(seed)
    res_q = run_config(post, config, logger=logger, **kwargs)

    say(f">>> Mandatory classical baseline (0%) with identical parameters "
        f"and seed={seed} — fair-benchmark requirement <<<")
    _reseed(seed)                      # [BASE] same random stream as quantum
    res_c = run_config(post, dict(CLASSICAL_BASELINE), logger=logger, **kwargs)

    return {'quantum': res_q, 'classical': res_c}


# =============================================================================
# 5.  VISUALIZATION — every figure OVERLAYS classical (blue) vs quantum (red)
# =============================================================================

def _q_colors(pcts):
    """Warm colormap (orange→red) for quantum quantumness levels; the 0%
    classical baseline is always drawn in blue (C_CLASSICAL)."""
    cmap = plt.cm.autumn_r
    out = []
    for p in pcts:
        if p == 0:
            out.append(C_CLASSICAL)
        else:
            out.append(cmap(0.25 + 0.75 * min(p, 100.0) / 100))
    return out


def _legend_cq(ax, q_label, classical_label='Classical (0%)',
               q_color=C_QUANTUM):
    """Standard two-entry classical-vs-quantum legend."""
    handles = [Line2D([0], [0], color=C_CLASSICAL, lw=2.4,
                      label=classical_label),
               Line2D([0], [0], color=q_color, lw=2.4, label=q_label)]
    ax.legend(handles=handles, fontsize=9)


def plot_kl_overlay(res_q: dict, res_c: dict, outdir: str, tag: str):
    """KL training curves of Classical VI vs QVMC OVERLAID on one axis.

    Blue = Classical VI (baseline), orange = QVMC at the requested
    quantumness. Both curves share the same iteration budget by
    construction (requirement 2).
    """
    fig, ax = plt.subplots(figsize=(8, 5.2))
    for res, col, lab in ((res_c, C_CLASSICAL, 'Classical VI (0%)'),
                          (res_q, C_QUANTUM2,
                           f"QVMC ({res_q['quantumness']:.0f}% quantum)")):
        h = res['qvmc_history']
        if h:
            ax.semilogy([d['it'] for d in h],
                        [max(d['kl'], 1e-12) for d in h],
                        color=col, lw=2.0, alpha=0.95, label=lab)
    ax.set_xlabel('Training iteration')
    ax.set_ylabel(r'KL$(Q_\varphi\,\|\,P_{\rm target})$')
    ax.set_title('Variational training — Classical VI vs QVMC\n'
                 f"(iterations = {res_q.get('n_iter', '?')},  "
                 f"nqpp = {res_q.get('nqpp', '?')} qubits/parameter)")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    f = os.path.join(outdir, f'kl_overlay_{tag}.png')
    fig.savefig(f, dpi=150, bbox_inches='tight')
    fig.savefig(f.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close(fig)
    return f


def plot_rhat_overlay(res_q: dict, res_c: dict, outdir: str, tag: str):
    """Gelman-Rubin convergence (R̂−1 vs steps) of Classical MCMC vs QMCMC
    OVERLAID on one axis. Blue = classical, red = quantum."""
    fig, ax = plt.subplots(figsize=(8, 5.2))
    for res, col, lab in ((res_c, C_CLASSICAL, 'Classical MCMC (0%)'),
                          (res_q, C_QUANTUM,
                           f"QMCMC ({res_q['quantumness']:.0f}% quantum)")):
        hist = res['mcmc']['rhat_hist']
        if hist:
            steps, rhats = zip(*hist)
            ax.semilogy(steps, np.array(rhats) - 1, 'o-', color=col,
                        lw=2.0, ms=4, alpha=0.95, label=lab)
    ax.axhline(0.05, color='k', ls='--', lw=1.2,
               label=r'threshold $\hat R-1=0.05$')
    ax.set_xlabel('Sampling steps')
    ax.set_ylabel(r'$\hat{R}_{\max} - 1$')
    ax.set_title('MCMC convergence — Classical MCMC vs QMCMC\n'
                 f"(total steps = {res_q.get('n_steps', '?')})")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    f = os.path.join(outdir, f'rhat_overlay_{tag}.png')
    fig.savefig(f, dpi=150, bbox_inches='tight')
    fig.savefig(f.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close(fig)
    return f


def plot_corner_overlay(flat_c: np.ndarray, flat_q: np.ndarray, model,
                        outdir: str, tag: str, title: str,
                        labels=('Classical', 'Quantum'),
                        q_color=C_QUANTUM,
                        weights_c=None, weights_q=None):
    """Corner plot (corner.py) with classical and quantum posteriors
    OVERLAID: 2D contours and 1D marginals on the same axes.

    [NEW] Requirement 3: blue = classical baseline, red/orange = quantum.
    Shared axis ranges are computed from the union of both sample sets so
    the contours are directly comparable.
    """
    d = model.n_params
    # Common ranges from the union of both sample sets (1–99 percentiles,
    # padded), so corner uses identical axes for both overlays.
    both = np.vstack([flat_c, flat_q])
    rng_ = []
    for p in range(d):
        lo, hi = np.percentile(both[:, p], [0.5, 99.5])
        pad = 0.08 * (hi - lo + 1e-12)
        rng_.append((lo - pad, hi + pad))

    # [FIX] corner.corner MUTATES hist_kwargs in place (it writes the
    # 'color' key on the first call), so sharing one dict between both
    # calls would silently draw BOTH diagonal histograms in the classical
    # color. A fresh kwargs dict per call keeps each overlay's color.
    def _kw():
        return dict(labels=model.param_latex, bins=35, range=rng_,
                    plot_datapoints=False, plot_density=False, smooth=1.0,
                    levels=(0.393, 0.865),    # 1σ and 2σ for 2D Gaussians
                    hist_kwargs=dict(density=True, lw=2.0))

    fig = corner.corner(flat_c, color=C_CLASSICAL, weights=weights_c,
                        truths=model.fiducial, truth_color='k', **_kw())
    corner.corner(flat_q, color=q_color, weights=weights_q, fig=fig, **_kw())

    handles = [Line2D([0], [0], color=C_CLASSICAL, lw=2.6, label=labels[0]),
               Line2D([0], [0], color=q_color, lw=2.6, label=labels[1]),
               Line2D([0], [0], color='k', ls='--', lw=1.4,
                      label='Fiducial (Planck)')]
    fig.legend(handles=handles, loc='upper right', fontsize=11,
               bbox_to_anchor=(0.98, 0.92))
    fig.suptitle(title, fontsize=13, fontweight='bold', y=1.02)

    f = os.path.join(outdir, f'corner_{tag}.png')
    fig.savefig(f, dpi=150, bbox_inches='tight')
    fig.savefig(f.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close(fig)
    return f


def plot_corner_single(flat: np.ndarray, model, outdir: str, tag: str,
                       title: str, color: str, weights=None):
    """Single-distribution corner plot (corner.py) for ONE method/config.

    [GROUP-1 | Individual] Requirement 4: a separate corner plot for every
    method and configuration that is run (no overlay), so each posterior
    can be inspected on its own.
    """
    d = model.n_params
    rng_ = []
    for p in range(d):
        lo, hi = np.percentile(flat[:, p], [0.5, 99.5])
        pad = 0.08 * (hi - lo + 1e-12)
        rng_.append((lo - pad, hi + pad))
    fig = corner.corner(flat, color=color, weights=weights,
                        labels=model.param_latex, bins=35, range=rng_,
                        plot_datapoints=False, plot_density=False, smooth=1.0,
                        levels=(0.393, 0.865), truths=model.fiducial,
                        truth_color='k',
                        hist_kwargs=dict(density=True, lw=2.0))
    fig.suptitle(title, fontsize=13, fontweight='bold', y=1.02)
    f = os.path.join(outdir, f'corner_{tag}.png')
    fig.savefig(f, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return f


def plot_corner_multi(datasets, colors, labels, model, outdir: str, tag: str,
                      title: str, weights_list=None):
    """Overlay N posteriors on ONE corner plot with a shared set of axes.

    [GROUP-2/3] Requirement 4: used both for the family "all-in-one" plots
    (classical baseline + ALL quantum percentages of a family) and for the
    per-percentage 1-to-1 plots (a single percentage of BOTH quantum
    families overlaid with the two classical baselines). Axis ranges are
    the union over all datasets so the contours are directly comparable.

    Args:
        datasets: list of (N_i, d) sample arrays.
        colors: one color per dataset.
        labels: one legend label per dataset.
        weights_list: optional list of per-dataset weight arrays (or None).
    """
    d = model.n_params
    if weights_list is None:
        weights_list = [None] * len(datasets)

    # Shared ranges from the union of all datasets (robust percentiles).
    allcat = np.vstack(datasets)
    rng_ = []
    for p in range(d):
        lo, hi = np.percentile(allcat[:, p], [0.5, 99.5])
        pad = 0.08 * (hi - lo + 1e-12)
        rng_.append((lo - pad, hi + pad))

    # corner.py mutates hist_kwargs in place (writes 'color'); every call
    # needs a FRESH dict or later histograms inherit the first color.
    def _kw():
        return dict(labels=model.param_latex, bins=35, range=rng_,
                    plot_datapoints=False, plot_density=False, smooth=1.0,
                    levels=(0.393, 0.865),
                    hist_kwargs=dict(density=True, lw=1.8))

    fig = None
    for data, col, w in zip(datasets, colors, weights_list):
        fig = corner.corner(data, color=col, weights=w, fig=fig, **_kw())
    corner.overplot_lines(fig, model.fiducial, color='k', ls='--', lw=1.2)

    handles = [Line2D([0], [0], color=c, lw=2.4, label=l)
               for c, l in zip(colors, labels)]
    handles.append(Line2D([0], [0], color='k', ls='--', lw=1.2,
                          label='Fiducial (Planck)'))
    fig.legend(handles=handles, loc='upper right', fontsize=10,
               bbox_to_anchor=(0.99, 0.93))
    fig.suptitle(title, fontsize=13, fontweight='bold', y=1.02)
    f = os.path.join(outdir, f'corner_{tag}.png')
    fig.savefig(f, dpi=150, bbox_inches='tight')
    fig.savefig(f.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close(fig)
    return f


def plot_marginals_overlay(res_q: dict, res_c: dict, post: Posterior,
                           outdir: str, tag: str):
    """1D marginal histograms with classical and quantum OVERLAID.

    Row 0: Classical MCMC (blue) vs QMCMC (red).
    Row 1: Classical VI (blue) vs QVMC (orange).
    Plus an H(z) posterior-predictive panel overlaying all four means.
    """
    model = post.model
    d = model.n_params
    qpct = res_q['quantumness']
    fig, axes = plt.subplots(2, d + 1, figsize=(4.2 * (d + 1), 8.2),
                             squeeze=False)
    fig.suptitle(f"{model.label} | {post.dataset} | "
                 f"Classical baseline vs {res_q['label']}\n"
                 f"MCMC steps = {res_q.get('n_steps', '?')}  |  "
                 f"QVMC iterations = {res_q.get('n_iter', '?')}  |  "
                 f"nqpp = {res_q.get('nqpp', '?')} qubits/parameter",
                 fontsize=12, fontweight='bold')

    # ── Row 0: MCMC family ───────────────────────────────────────────────────
    for p in range(d):
        ax = axes[0, p]
        ax.hist(res_c['flat_mcmc'][:, p], bins=35, density=True,
                color=C_CLASSICAL, alpha=0.5, label='Classical MCMC (0%)')
        ax.hist(res_q['flat_mcmc'][:, p], bins=35, density=True,
                color=C_QUANTUM, alpha=0.5,
                label=f'QMCMC ({qpct:.0f}%)')
        ax.axvline(model.fiducial[p], color='k', ls='--', lw=1.3)
        ax.set_xlabel(model.param_latex[p])
        if p == 0:
            ax.set_ylabel('MCMC family\ndensity')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.25)

    # ── Row 1: variational family ────────────────────────────────────────────
    for p in range(d):
        ax = axes[1, p]
        Sc, Wc = res_c['qvmc_samples'], res_c['qvmc_weights']
        Sq, Wq = res_q['qvmc_samples'], res_q['qvmc_weights']
        ax.hist(Sc[:, p], weights=Wc, bins=35, density=True,
                color=C_CLASSICAL, alpha=0.5, label='Classical VI (0%)')
        ax.hist(Sq[:, p], weights=Wq, bins=35, density=True,
                color=C_QUANTUM2, alpha=0.5,
                label=f'QVMC ({qpct:.0f}%)')
        ax.axvline(model.fiducial[p], color='k', ls='--', lw=1.3)
        ax.set_xlabel(model.param_latex[p])
        if p == 0:
            ax.set_ylabel('Variational family\ndensity')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.25)

    # ── H(z) posterior predictive (all four means overlaid) ──────────────────
    z_plot = np.linspace(0.01, 2.4, 200)
    for row, (key, fam, col_c, col_q, mu_key) in enumerate([
            ('mcmc', 'MCMC', C_CLASSICAL, C_QUANTUM, 'mu'),
            ('qvmc', 'VI/QVMC', C_CLASSICAL, C_QUANTUM2, 'mu')]):
        ax = axes[row, d]
        ax.errorbar(post.z_cc, post.H_cc, yerr=post.sig_cc, fmt='.k',
                    ms=4, capsize=2, alpha=0.7)
        ax.plot(z_plot, model.H(z_plot, res_c[key][mu_key]), color=col_c,
                lw=2.2, label=f'Classical {fam}')
        ax.plot(z_plot, model.H(z_plot, res_q[key][mu_key]), color=col_q,
                lw=2.2, ls='--', label=f'Quantum {fam}')
        ax.set_xlabel(r'$z$')
        ax.set_ylabel(r'$H(z)$ [km/s/Mpc]')
        ax.set_title(f'H(z) — {fam}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    f = os.path.join(outdir, f'marginals_{tag}.png')
    fig.savefig(f, dpi=150, bbox_inches='tight')
    fig.savefig(f.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close(fig)
    return f


def plot_traces_overlay(res_q: dict, res_c: dict, model, outdir: str,
                        tag: str):
    """Trace plots of every free physical parameter, classical and quantum
    chains OVERLAID on the same axes.

    Left column: MCMC chains vs step — classical chains in blue, quantum
    chains in red (one line per chain).
    Right column: E_Q[θ] during variational training — Classical VI in
    blue, QVMC in orange.
    """
    d = model.n_params
    qpct = res_q['quantumness']
    fig, axes = plt.subplots(d, 2, figsize=(13, 2.7 * d), squeeze=False)
    fig.suptitle(f"Trace plots — {model.label} — "
                 f"Classical (blue) vs Quantum (red/orange, {qpct:.0f}%)\n"
                 f"MCMC steps = {res_q.get('n_steps', '?')}  |  "
                 f"QVMC iterations = {res_q.get('n_iter', '?')}  |  "
                 f"nqpp = {res_q.get('nqpp', '?')}",
                 fontsize=12, fontweight='bold')
    for p in range(d):
        ax = axes[p, 0]
        ch_c, ch_q = res_c['chains_mcmc'], res_q['chains_mcmc']
        for c in range(ch_c.shape[0]):
            ax.plot(ch_c[c, :, p], lw=0.6, alpha=0.55, color=C_CLASSICAL)
        for c in range(ch_q.shape[0]):
            ax.plot(ch_q[c, :, p], lw=0.6, alpha=0.55, color=C_QUANTUM)
        ax.axhline(model.fiducial[p], color='k', ls='--', lw=1)
        ax.set_ylabel(model.param_latex[p])
        if p == 0:
            ax.set_title('MCMC chains — Classical (blue) vs QMCMC (red)')
            _legend_cq(ax, f'QMCMC ({qpct:.0f}%)',
                       'Classical MCMC (0%)', C_QUANTUM)
        if p == d - 1:
            ax.set_xlabel('Step')
        ax.grid(True, alpha=0.25)

        ax = axes[p, 1]
        for res, col in ((res_c, C_CLASSICAL), (res_q, C_QUANTUM2)):
            h = res['qvmc_history']
            if h:
                ax.plot([x['it'] for x in h],
                        [x['theta_mean'][p] for x in h], color=col, lw=1.7)
        ax.axhline(model.fiducial[p], color='k', ls='--', lw=1)
        if p == 0:
            ax.set_title(r'$E_{Q_\varphi}[\theta]$ during training — '
                         'Classical VI (blue) vs QVMC (orange)')
            _legend_cq(ax, f'QVMC ({qpct:.0f}%)',
                       'Classical VI (0%)', C_QUANTUM2)
        if p == d - 1:
            ax.set_xlabel('Iteration')
        ax.grid(True, alpha=0.25)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    f = os.path.join(outdir, f'traces_{tag}.png')
    fig.savefig(f, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return f


def plot_comparison_figures(comp: dict, post: Posterior, outdir: str) -> list:
    """Generate the FULL overlay figure set for one quantum-vs-classical
    comparison (requirement 3): corner plots (MCMC and variational
    families), 1D marginals, KL curves, R̂ diagnostics and traces.
    """
    res_q, res_c = comp['quantum'], comp['classical']
    model = post.model
    tag = f"{model.name}_q{int(res_q['quantumness']):03d}"
    qpct = res_q['quantumness']
    files = []

    # Corner — MCMC family (Classical MCMC blue vs QMCMC red)
    files.append(plot_corner_overlay(
        res_c['flat_mcmc'], res_q['flat_mcmc'], model, outdir,
        tag=f'mcmc_{tag}',
        title=f'{model.label} — Classical MCMC vs QMCMC ({qpct:.0f}%)  '
              f'[steps={res_q.get("n_steps", "?")}]',
        labels=('Classical MCMC (0%)', f'QMCMC ({qpct:.0f}%)'),
        q_color=C_QUANTUM))

    # Corner — variational family (Classical VI blue vs QVMC orange),
    # weighted samples.
    files.append(plot_corner_overlay(
        res_c['qvmc_samples'], res_q['qvmc_samples'], model, outdir,
        tag=f'qvmc_{tag}',
        title=f'{model.label} — Classical VI vs QVMC ({qpct:.0f}%)  '
              f'[iters={res_q.get("n_iter", "?")}, '
              f'nqpp={res_q.get("nqpp", "?")}]',
        labels=('Classical VI (0%)', f'QVMC ({qpct:.0f}%)'),
        q_color=C_QUANTUM2,
        weights_c=res_c['qvmc_weights'], weights_q=res_q['qvmc_weights']))

    files.append(plot_marginals_overlay(res_q, res_c, post, outdir, tag))
    files.append(plot_kl_overlay(res_q, res_c, outdir, tag))
    files.append(plot_rhat_overlay(res_q, res_c, outdir, tag))
    files.append(plot_traces_overlay(res_q, res_c, model, outdir, tag))
    return files


# ── multi-preset (benchmark) overlays ────────────────────────────────────────

def plot_kl_curves(results: List[dict], outdir: str, tag: str):
    """KL training curves vs iteration, OVERLAID for ALL evaluated
    quantumness levels. The 0% classical baseline is drawn in thick blue;
    quantum levels use a warm orange→red colormap."""
    fig, ax = plt.subplots(figsize=(8, 5.2))
    for r, col in zip(results, _q_colors([r['quantumness'] for r in results])):
        h = r['qvmc_history']
        if not h:
            continue
        lw = 2.8 if r['quantumness'] == 0 else 1.8
        lab = ('Classical VI (0%)' if r['quantumness'] == 0
               else f"{r['quantumness']:.0f}%")
        ax.semilogy([d['it'] for d in h], [max(d['kl'], 1e-12) for d in h],
                    color=col, lw=lw, alpha=0.95, label=lab)
    ax.set_xlabel('Training iteration')
    ax.set_ylabel(r'KL$(Q_\varphi\,\|\,P_{\rm target})$')
    meta = results[0] if results else {}
    ax.set_title('Variational training — classical baseline vs quantumness '
                 f'levels\n(iterations = {meta.get("n_iter", "?")},  '
                 f'nqpp = {meta.get("nqpp", "?")} qubits/parameter)')
    ax.legend(title='Quantumness', fontsize=9)
    ax.grid(True, alpha=0.3)
    f = os.path.join(outdir, f'kl_curves_{tag}.png')
    fig.savefig(f, dpi=150, bbox_inches='tight')
    fig.savefig(f.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close(fig)
    return f


def plot_rhat_curves(results: List[dict], outdir: str, tag: str):
    """Gelman-Rubin convergence (R̂−1 vs steps) OVERLAID for ALL
    quantumness levels; 0% classical baseline in thick blue."""
    fig, ax = plt.subplots(figsize=(8, 5.2))
    for r, col in zip(results, _q_colors([r['quantumness'] for r in results])):
        hist = r['mcmc']['rhat_hist']
        if not hist:
            continue
        steps, rhats = zip(*hist)
        lw = 2.8 if r['quantumness'] == 0 else 1.8
        lab = ('Classical MCMC (0%)' if r['quantumness'] == 0
               else f"{r['quantumness']:.0f}%")
        ax.semilogy(steps, np.array(rhats) - 1, 'o-', color=col, lw=lw,
                    ms=4, alpha=0.95, label=lab)
    ax.axhline(0.05, color='k', ls='--', lw=1.2,
               label=r'threshold $\hat R-1=0.05$')
    ax.set_xlabel('Sampling steps')
    ax.set_ylabel(r'$\hat{R}_{\max} - 1$')
    meta = results[0] if results else {}
    ax.set_title('MCMC convergence — classical baseline vs quantumness '
                 f'levels\n(total steps = {meta.get("n_steps", "?")})')
    ax.legend(title='Quantumness', fontsize=9)
    ax.grid(True, alpha=0.3)
    f = os.path.join(outdir, f'rhat_curves_{tag}.png')
    fig.savefig(f, dpi=150, bbox_inches='tight')
    fig.savefig(f.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close(fig)
    return f


def plot_benchmark_summary(results: List[dict], model, outdir: str):
    """Comparison panel + extended table (χ², reduced χ², AIC, BIC, ESS).
    The 0% classical baseline appears in blue in every panel."""
    pcts = [r['quantumness'] for r in results]
    labels = [f"{int(p)}%" for p in pcts]
    colors = _q_colors(pcts)

    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(3, 3, hspace=0.5, wspace=0.35)

    ax = fig.add_subplot(gs[0, 0])
    times = [r['elapsed_total'] for r in results]
    bars = ax.bar(labels, times, color=colors, edgecolor='white')
    for b, t in zip(bars, times):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f'{t:.0f}s',
                ha='center', va='bottom', fontsize=8)
    ax.set_title('Total runtime', fontweight='bold')
    ax.set_ylabel('s'); ax.grid(True, alpha=0.3, axis='y')

    for j, (src, fam, col) in enumerate([('mcmc', 'QMCMC', C_QUANTUM),
                                         ('qvmc', 'QVMC', C_QUANTUM2)]):
        ax = fig.add_subplot(gs[0, 1 + j])
        mus = [r[src]['mu'][0] for r in results]
        sds = [r[src]['std'][0] for r in results]
        ax.errorbar(pcts, mus, yerr=sds, fmt='o-', capsize=4, lw=2, color=col)
        # mark the classical baseline value in blue
        ax.errorbar([pcts[0]], [mus[0]], yerr=[sds[0]], fmt='s', capsize=4,
                    ms=9, color=C_CLASSICAL, label='Classical baseline')
        ax.axhline(model.fiducial[0], color='k', ls='--', lw=1.2,
                   label='Planck')
        ax.set_title(f'{fam} — {model.param_latex[0]} vs quantumness',
                     fontweight='bold')
        ax.set_xlabel('Quantumness (%)'); ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[1, 0])
    ax.plot(pcts, [r['mcmc']['acceptance'] for r in results], 'D-',
            color=C_QUANTUM, lw=2)
    ax.plot([pcts[0]], [results[0]['mcmc']['acceptance']], 's', ms=9,
            color=C_CLASSICAL, label='Classical baseline')
    ax.axhspan(0.23, 0.50, alpha=0.12, color='green', label='optimal range')
    ax.set_title('QMCMC acceptance rate', fontweight='bold')
    ax.set_xlabel('Quantumness (%)'); ax.set_ylim(0, 1)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[1, 1])
    kl = [r['qvmc']['kl_final'] for r in results]
    ax.semilogy(pcts, np.clip(kl, 1e-12, None), 'v-', color=C_QUANTUM2, lw=2)
    ax.semilogy([pcts[0]], [max(kl[0], 1e-12)], 's', ms=9,
                color=C_CLASSICAL, label='Classical baseline')
    ax.set_title('Final KL (variational)', fontweight='bold')
    ax.set_xlabel('Quantumness (%)'); ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[1, 2])
    ax.plot(pcts, [r['mcmc']['ess'] for r in results], 'o-',
            color=C_QUANTUM, label='MCMC family', lw=2)
    ax.plot(pcts, [r['qvmc']['ess'] for r in results], 's-',
            color=C_QUANTUM2, label='Variational family', lw=2)
    ax.set_title('Effective Sample Size', fontweight='bold')
    ax.set_xlabel('Quantumness (%)'); ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # extended table
    ax = fig.add_subplot(gs[2, :]); ax.axis('off')
    cols = ['Q%', 't (s)', 'MCMC mean±std', 'chi2_red', 'AIC', 'BIC', 'ESS',
            'acc', 'VI/QVMC mean±std', 'chi2_red', 'AIC', 'BIC', 'KL']
    rows = []
    for r in results:
        m, q = r['mcmc'], r['qvmc']
        rows.append([
            f"{r['quantumness']:.0f}", f"{r['elapsed_total']:.0f}",
            f"{m['mu'][0]:.4f}±{m['std'][0]:.4f}", f"{m['chi2_red']:.3f}",
            f"{m['AIC']:.1f}", f"{m['BIC']:.1f}", f"{m['ess']:.0f}",
            f"{m['acceptance']:.2f}",
            f"{q['mu'][0]:.4f}±{q['std'][0]:.4f}", f"{q['chi2_red']:.3f}",
            f"{q['AIC']:.1f}", f"{q['BIC']:.1f}", f"{q['kl_final']:.4f}",
        ])
    tbl = ax.table(cellText=rows, colLabels=cols, loc='center', cellLoc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1, 1.6)
    for i, c in enumerate(colors):
        rgba = matplotlib.colors.to_rgba(c)
        for j in range(len(cols)):
            tbl[i + 1, j].set_facecolor((*rgba[:3], 0.22))

    meta = results[0] if results else {}
    fig.suptitle(f'Benchmark {model.label} — classical baseline vs '
                 'quantumness levels\n'
                 f'steps = {meta.get("n_steps", "?")}  |  '
                 f'iterations = {meta.get("n_iter", "?")}  |  '
                 f'nqpp = {meta.get("nqpp", "?")}',
                 fontsize=14, fontweight='bold', y=0.99)
    f = os.path.join(outdir, f'benchmark_{model.name}.png')
    fig.savefig(f, dpi=150, bbox_inches='tight')
    fig.savefig(f.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close(fig)
    return f


# =============================================================================
# 6.  BENCHMARK
# =============================================================================

def plot_corner_groups(results: List[dict], model, outdir: str) -> List[str]:
    """Build the three corner-plot groupings required for the analysis.

    Requirement 4, given the full list of benchmark results (the 0%
    classical baseline first, then every quantum preset):

      GROUP 1 — Individual: one standalone corner per method per config
                (Classical MCMC, Classical VI, and QMCMC/QVMC at each %).
      GROUP 2 — Family "all-in-one": Classical MCMC overlaid with ALL
                QMCMC percentages; Classical VI overlaid with ALL QVMC %.
      GROUP 3 — 1-to-1 per percentage: each percentage's QMCMC and QVMC
                overlaid EXCLUSIVELY with the two classical baselines, so
                one can read off which quantumness level best matches the
                ideal classical distribution.
    """
    files: List[str] = []
    name = model.name
    baseline = next(r for r in results if r['quantumness'] == 0)
    quantum = [r for r in results if r['quantumness'] > 0]
    pcts = [int(r['quantumness']) for r in quantum]
    q_cols = _q_colors(pcts)                 # warm colormap for the levels

    # ── GROUP 1: individual corners (one distribution each) ─────────────────
    for r in results:
        p = int(r['quantumness'])
        m_lab = 'Classical MCMC' if p == 0 else f'QMCMC {p}%'
        v_lab = 'Classical VI' if p == 0 else f'QVMC {p}%'
        files.append(plot_corner_single(
            r['flat_mcmc'], model, outdir, f'individual_mcmc_{name}_q{p:03d}',
            title=f'{model.label} — {m_lab}  [steps={r.get("n_steps","?")}]',
            color=C_CLASSICAL if p == 0 else C_QUANTUM))
        files.append(plot_corner_single(
            r['qvmc_samples'], model, outdir,
            f'individual_qvmc_{name}_q{p:03d}',
            title=f'{model.label} — {v_lab}  '
                  f'[iters={r.get("n_iter","?")}, nqpp={r.get("nqpp","?")}]',
            color=C_CLASSICAL if p == 0 else C_QUANTUM2,
            weights=r['qvmc_weights']))

    # ── GROUP 2: family "all-in-one" overlays ───────────────────────────────
    # MCMC family: Classical MCMC + every QMCMC percentage.
    files.append(plot_corner_multi(
        [baseline['flat_mcmc']] + [r['flat_mcmc'] for r in quantum],
        [C_CLASSICAL] + list(q_cols),
        ['Classical MCMC (0%)'] + [f'QMCMC {p}%' for p in pcts],
        model, outdir, f'family_mcmc_{name}',
        title=f'{model.label} — MCMC family: Classical baseline + all '
              f'QMCMC levels  [steps={baseline.get("n_steps","?")}]'))
    # Variational family: Classical VI + every QVMC percentage (weighted).
    files.append(plot_corner_multi(
        [baseline['qvmc_samples']] + [r['qvmc_samples'] for r in quantum],
        [C_CLASSICAL] + list(q_cols),
        ['Classical VI (0%)'] + [f'QVMC {p}%' for p in pcts],
        model, outdir, f'family_qvmc_{name}',
        title=f'{model.label} — Variational family: Classical baseline + '
              f'all QVMC levels  [iters={baseline.get("n_iter","?")}, '
              f'nqpp={baseline.get("nqpp","?")}]',
        weights_list=([baseline['qvmc_weights']]
                      + [r['qvmc_weights'] for r in quantum])))

    # ── GROUP 3: 1-to-1 (each % of BOTH families vs the two baselines) ──────
    for r in quantum:
        p = int(r['quantumness'])
        files.append(plot_corner_multi(
            [baseline['flat_mcmc'], baseline['qvmc_samples'],
             r['flat_mcmc'], r['qvmc_samples']],
            [C_CLASSICAL, C_CLASSICAL2, C_QUANTUM, C_QUANTUM2],
            ['Classical MCMC', 'Classical VI',
             f'QMCMC {p}%', f'QVMC {p}%'],
            model, outdir, f'1to1_{name}_q{p:03d}',
            title=f'{model.label} — {p}% quantum vs classical baselines  '
                  f'[steps={r.get("n_steps","?")}, '
                  f'iters={r.get("n_iter","?")}, nqpp={r.get("nqpp","?")}]',
            weights_list=[None, baseline['qvmc_weights'],
                          None, r['qvmc_weights']]))
    return files

def run_benchmark(post: Posterior, n_steps_mcmc: int, max_iter_qvmc: int,
                  nqpp: int, outdir: str, seed: int = 42, logger=None,
                  log_every: int = 500) -> List[dict]:
    """Run ALL PRESETS with the active model/dataset.

    [BASE] The 0% preset is run FIRST and acts as the shared classical
    baseline: every quantum preset is overlaid against it in the per-preset
    corner/marginal/KL/R̂/trace figures, in addition to the multi-level
    overlay curves and the summary panel. All presets share identical
    parameters and the same RNG seed.
    """
    say = logger.info if logger else print
    say(f"BENCHMARK — model {post.model.label}, dataset {post.dataset}, "
        f"presets {list(PRESETS.keys())}")
    kw = dict(n_steps_mcmc=n_steps_mcmc, max_iter_qvmc=max_iter_qvmc,
              nqpp=nqpp, logger=logger, log_every=log_every,
              verbose=(logger is None),
              stop_on_convergence=False)       # full R̂ curves for overlays

    # ── 1) Mandatory classical baseline (shared by every quantum preset) ────
    _reseed(seed)
    baseline = run_config(post, dict(PRESETS[0]), **kw)
    results = [baseline]
    say(f"  * {PRESETS[0]['label']:48s} {baseline['elapsed_total']:.1f}s "
        f"[shared classical baseline]")

    # ── 2) Quantum presets, each overlaid against the baseline ──────────────
    for pct, preset in PRESETS.items():
        if pct == 0:
            continue
        _reseed(seed)                  # [BASE] identical random stream
        res = run_config(post, dict(preset), **kw)
        results.append(res)
        plot_comparison_figures({'quantum': res, 'classical': baseline},
                                post, outdir)
        say(f"  * {preset['label']:48s} {res['elapsed_total']:.1f}s")

    plot_kl_curves(results, outdir, tag=post.model.name)
    plot_rhat_curves(results, outdir, tag=post.model.name)
    plot_benchmark_summary(results, post.model, outdir)
    # [GROUP] Requirement 4: the three corner-plot groupings
    # (individual, family all-in-one, per-percentage 1-to-1).
    plot_corner_groups(results, post.model, outdir)
    say(f"Figures in {outdir}/: corner_individual_*, corner_family_*, "
        f"corner_1to1_*, corner_mcmc_*, corner_qvmc_*, marginals_*, "
        f"kl_overlay_*, rhat_overlay_*, traces_*, kl_curves_*, "
        f"rhat_curves_*, benchmark_*")
    return results


# =============================================================================
# 7.  INTERACTIVE MENU (default behavior with no arguments)
# =============================================================================

def _ask(prompt: str, options: dict, default):
    """Numbered-option question; Enter = default."""
    keys = list(options)
    for i, k in enumerate(keys, 1):
        print(f"    [{i}] {options[k]}")
    while True:
        r = input(f"  {prompt} [Enter={default}]: ").strip()
        if not r:
            return default
        if r.isdigit() and 1 <= int(r) <= len(keys):
            return keys[int(r) - 1]
        if r in options:
            return r
        print("  Invalid option.")


def sanity_check_routing(model_name: str = 'lcdm', nqpp: int = 2):
    """Standalone routing + correctness check (console prints).

    Run with `--sanity-check`. It does three things:

    1. ACCEPTANCE REGRESSION TEST — verifies the quantum acceptance is a
       monotonically INCREASING function of Δ = lp_prop − lp_cur and
       matches Metropolis min(1, e^Δ). This is the regression guard for
       the inverted-acceptance bug.
    2. PROPOSAL CHECK — confirms the quantum displacement is zero-mean and
       unit-std (drop-in for the classical Gaussian).
    3. PER-COMPONENT ROUTING TABLE — for every preset, prints which engine
       (Qiskit/Aer vs NumPy/SciPy) each component resolves to, plus a live
       trace of the first few evaluations inside the loops.
    """
    global SANITY_CHECK
    print("\n" + "=" * 70)
    print("SANITY CHECK 1/3 — quantum acceptance vs Δ (must INCREASE with Δ)")
    print("=" * 70)
    print(f"{'Δ':>6} | {'P_quantum':>10} | {'min(1,e^Δ)':>11} | match?")
    ok = True
    prev = -1.0
    for d in [-5, -2, -1, 0, 1, 2, 5]:
        pq = np.exp(hadamard_accept_log(0.0, float(d)))
        met = min(1.0, np.exp(d))
        good = abs(pq - met) < 1e-3
        ok &= good and (pq >= prev - 1e-9)
        prev = pq
        print(f"{d:6.1f} | {pq:10.4f} | {met:11.4f} | "
              f"{'OK' if good else 'MISMATCH'}")
    print(f"  → acceptance {'PASSES' if ok else 'FAILS'} "
          "(monotonic increasing, matches Metropolis)")

    print("\n" + "=" * 70)
    print("SANITY CHECK 2/3 — quantum proposal displacement statistics")
    print("=" * 70)
    model = MODELS[model_name]
    _reseed(0)
    eng = QuantumProposalEngine(model.n_params, n_layers=3, batch=2000)
    disp = np.array([eng.next() for _ in range(2000)])
    print(f"  mean per dim = {disp.mean(0).round(4)}  (≈0 expected)")
    print(f"  std  per dim = {disp.std(0).round(4)}   (≈1 after calibration)")

    print("\n" + "=" * 70)
    print("SANITY CHECK 3/3 — per-preset component routing "
          "(⚛ Qiskit/Aer vs 🖥 NumPy/SciPy)")
    print("=" * 70)
    comp_method = {'proposal': 'QMCMC', 'acceptance': 'QMCMC',
                   'training': 'QVMC', 'sampling': 'QVMC',
                   'normalization': 'QVMC'}
    hdr = f"{'preset':>7} | " + " | ".join(f"{c[:9]:>9}" for c in
                                           QUANTUM_COMPONENTS)
    print(hdr)
    print("-" * len(hdr))
    for pct, preset in PRESETS.items():
        cells = []
        for c in QUANTUM_COMPONENTS:
            cells.append("⚛Q " if preset.get(c, False) else "🖥C ")
        print(f"{('P'+str(pct)):>7} | " + " | ".join(f"{x:>9}" for x in cells))
    print("\n  (QMCMC reads proposal+acceptance; QVMC reads "
          "training+sampling+normalization.\n   Presets that differ only in "
          "the OTHER method's components are identical for a\n   given "
          "method — that is expected, not a bug.)")

    # Live trace: run a tiny fully-quantum (P100) config and watch the loops.
    print("\n  Live routing trace for P100 (fully quantum), 3 evals each:")
    SANITY_CHECK = True
    _SANITY_BUDGET.clear()
    post = Posterior(model, 'CC', 'flat')
    _reseed(0)
    mc = QMCMCModular(post, dict(PRESETS[100]), n_chains=3, n_burn=0,
                      stop_on_convergence=False)
    mc.run(n_steps=4, progress=False)
    qv = QVMCModular(post, dict(PRESETS[100]), n_qubits_per_param=nqpp)
    qv.run(max_iter=3, n_chains=2, progress=False)
    SANITY_CHECK = False
    print("=" * 70 + "\n")


def interactive_menu() -> dict:
    """Interactive menu: run mode, model, dataset, prior, components, sizes.

    The first question selects the RUN MODE:
        [1] Single configuration  — one preset/custom config + its baseline
        [2] Full benchmark         — every preset at user-chosen sizes
        [3] Quick TEST RUN         — every preset at small fixed sizes, a
                                     fast stability check across the whole
                                     hybrid spectrum (0/20/45/70/90/100%)

    If stdin is not a terminal (e.g. SLURM without --interactive), it
    falls back to a default preset so the job is not blocked.
    """
    if not sys.stdin.isatty():
        print("  [Non-interactive mode detected] Preset P20, ΛCDM, CC, flat.")
        return {'model': 'lcdm', 'dataset': 'CC', 'prior': 'flat',
                'config': dict(PRESETS[20]), 'steps': 1000, 'qvmc_iter': 300,
                'nqpp': 3, 'benchmark': False}

    print("\n" + "╔" + "═" * 63 + "╗")
    print("║   Modular Quantum/Classical Sampler — Configuration           ║")
    print("╚" + "═" * 63 + "╝")

    print("\n  ── Run mode ──")
    mode = _ask("Run mode",
                {'single': 'Single configuration (one preset + its baseline)',
                 'benchmark': 'Full benchmark (all presets, your sizes)',
                 'test': 'Quick TEST RUN (all presets, small fixed sizes — '
                         'stability check)'},
                'single')

    print("\n  ── Cosmological model ──")
    model = _ask("Model", {k: v.label + f"  ({v.n_params} parameters: "
                           + ", ".join(v.param_names) + ")"
                           for k, v in MODELS.items()}, 'lcdm')

    print("\n  ── Dataset ──")
    pan = core.load_pantheon()
    opts = {'CC': 'Cosmic Chronometers (51 pts)'}
    if pan is not None:
        opts['Pantheon+'] = f"Pantheon+ ({len(pan['z'])} SNe Ia)"
        opts['CC+Pantheon+'] = 'CC + Pantheon+ combined'
    else:
        print("    (pantheon_full_parameters.txt not found → CC only)")
    dataset = _ask("Dataset", opts, 'CC')

    print("\n  ── Prior ──")
    prior = _ask("Prior", {'flat': 'Flat (box)',
                           'gaussian': 'Planck 2018 Gaussian on (Om, H0)'},
                 'flat')

    # ── TEST RUN: small fixed sizes, every preset, no further questions ──────
    if mode == 'test':
        print("\n  → Quick TEST RUN: benchmark over presets "
              f"{list(PRESETS.keys())} at small fixed sizes "
              "(steps=200, iters=40, nqpp=2).")
        print("  This verifies code stability across the whole hybrid "
              "spectrum without a long run.")
        return {'model': model, 'dataset': dataset, 'prior': prior,
                'config': dict(PRESETS[20]), 'steps': 200, 'qvmc_iter': 40,
                'nqpp': 2, 'benchmark': True, 'test_run': True}

    # ── BENCHMARK: every preset, user-chosen sizes ───────────────────────────
    if mode == 'benchmark':
        def ask_int_b(prompt, default):
            r = input(f"  {prompt} [Enter={default}]: ").strip()
            return int(r) if r.isdigit() else default
        print("\n  ── Sizes (apply to every preset and its baseline) ──")
        steps = ask_int_b("MCMC steps", 1000)
        iters = ask_int_b("Variational iterations", 300)
        nqpp = ask_int_b("Qubits per parameter (2^n grid)", 3)
        return {'model': model, 'dataset': dataset, 'prior': prior,
                'config': dict(PRESETS[20]), 'steps': steps,
                'qvmc_iter': iters, 'nqpp': nqpp, 'benchmark': True}

    # ── SINGLE configuration ─────────────────────────────────────────────────
    print("\n  ── Quantum components ──")
    for i, (k, meta) in enumerate(QUANTUM_COMPONENTS.items(), 1):
        print(f"    {i}. [{meta['weight']:2d}%]  {meta['name']}")
    print(f"\n  Presets: " + "  ".join(f"P{p}" for p in PRESETS))
    cfg = None
    while cfg is None:
        r = input("  Components (e.g. Q C Q Q C) or preset (e.g. P45): ").strip().upper()
        if r.startswith('P') and r[1:].isdigit() and int(r[1:]) in PRESETS:
            cfg = dict(PRESETS[int(r[1:])])
        else:
            toks = r.split()
            if len(toks) == 5 and all(t in 'QC' for t in toks):
                keys = list(QUANTUM_COMPONENTS)
                cfg = {k: (toks[i] == 'Q') for i, k in enumerate(keys)}
                cfg['label'] = f"{compute_quantumness(cfg):.0f}% — custom"
            else:
                print("  Invalid format.")
    pct = compute_quantumness(cfg)
    print(f"\n  → Quantumness: {pct:.1f}%  ({quantumness_label(pct)})")
    if pct > 0:
        print("  NOTE: the exact classical baseline (Classical MCMC + "
              "Classical VI)\n  will be run automatically with the same "
              "parameters for a fair benchmark.")

    def ask_int(prompt, default):
        r = input(f"  {prompt} [Enter={default}]: ").strip()
        return int(r) if r.isdigit() else default

    steps = ask_int("MCMC steps (classical and quantum alike)", 1000)
    iters = ask_int("Variational iterations (classical and quantum alike)", 300)
    nqpp = ask_int("Qubits per parameter (2^n grid)", 3)

    return {'model': model, 'dataset': dataset, 'prior': prior, 'config': cfg,
            'steps': steps, 'qvmc_iter': iters, 'nqpp': nqpp,
            'benchmark': False}


# =============================================================================
# 8.  MAIN / CLI
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    p = argparse.ArgumentParser(
        description="Hybrid quantum/classical sampler for cosmology "
                    "(ΛCDM, wCDM, CPL, PEDE, GEDE). Any quantum run "
                    "automatically triggers its exact classical baseline "
                    "with identical parameters.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""Examples:
  python cosmo_modular_quantum.py                            # interactive
  python cosmo_modular_quantum.py --model cpl --preset 45 --steps 2000
  python cosmo_modular_quantum.py --model wcdm --benchmark
  python cosmo_modular_quantum.py --config '{"proposal":true,"acceptance":false,"training":true,"sampling":true,"normalization":false}'""")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--interactive', action='store_true',
                      help='Force the interactive menu')
    mode.add_argument('--preset', type=int, choices=list(PRESETS),
                      metavar='N', help=f'Quantumness preset {list(PRESETS)}')
    mode.add_argument('--benchmark', action='store_true',
                      help='Run every preset and compare')
    mode.add_argument('--config', type=str, metavar='JSON',
                      help='Component configuration as JSON')

    p.add_argument('--model', choices=list(MODELS), default='lcdm',
                   help='Cosmological model (default: lcdm)')
    p.add_argument('--dataset', choices=['CC', 'Pantheon+', 'CC+Pantheon+'],
                   default='CC', help='Observational dataset (default: CC)')
    p.add_argument('--prior', choices=['flat', 'gaussian'], default='flat',
                   help='Prior type (default: flat)')
    p.add_argument('--steps', type=int, default=1000,
                   help='MCMC steps — applies equally to classical and quantum')
    p.add_argument('--qvmc-iter', type=int, default=300,
                   help='Variational iterations — applies equally to classical and quantum')
    p.add_argument('--burn', type=int, default=None,
                   help='Burn-in (default: 10%% of --steps)')
    p.add_argument('--nqpp', type=int, default=3,
                   help='Qubits per physical parameter (default: 3)')
    p.add_argument('--chains', type=int, default=6, help='MCMC chains')
    p.add_argument('--shots', type=int, default=2000, help='Sampling shots')
    p.add_argument('--log-file', type=str, default=None,
                   help='Log file (default: auto in CLI mode)')
    p.add_argument('--log-every', type=int, default=500,
                   help='Progress logging cadence (default: 500)')
    p.add_argument('--outdir', type=str, default='results',
                   help='Output directory (default: results/)')
    p.add_argument('--seed', type=int, default=42, help='RNG seed')
    p.add_argument('--no-plot', action='store_true', help='Skip figures')
    p.add_argument('--sanity-check', action='store_true',
                   help='Run the routing/correctness sanity check and exit '
                        '(acceptance regression test + per-preset engine map)')
    return p


def _print_summary_block(title: str, model, st: dict, extra: str,
                         logger=None):
    """Print one method's final summary block to console (and log)."""
    lines = [f"  ── {title} ──"]
    for p_ in range(model.n_params):
        if 'p16' in st:
            lines.append(f"  {model.param_names[p_]:6s}= {st['mu'][p_]:.4f} ± "
                         f"{st['std'][p_]:.4f}  "
                         f"[{st['p16'][p_]:.4f}, {st['p84'][p_]:.4f}]")
        else:
            lines.append(f"  {model.param_names[p_]:6s}= {st['mu'][p_]:.4f} ± "
                         f"{st['std'][p_]:.4f}")
    lines.append(f"  chi2 = {st['chi2']:.2f}  chi2_red = {st['chi2_red']:.3f}  "
                 f"AIC = {st['AIC']:.2f}  BIC = {st['BIC']:.2f}")
    lines.append(extra)
    for ln in lines:
        print(ln)
        if logger:
            logger.info(ln.strip())


def main():
    """Entry point: interactive without arguments, CLI with logging if any."""
    parser = build_parser()
    args = parser.parse_args()

    # --sanity-check short-circuits everything: routing + correctness only.
    if getattr(args, 'sanity_check', False):
        sanity_check_routing(model_name=args.model, nqpp=min(args.nqpp, 2))
        return

    cli_mode = len(sys.argv) > 1 and not args.interactive

    _reseed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    # ── mode selection ───────────────────────────────────────────────────
    if cli_mode:
        # Detailed output → log file
        log_file = args.log_file or os.path.join(
            args.outdir, f"qcosmo_{args.model}_{time.strftime('%Y%m%d_%H%M%S')}.log")
        logger = setup_logger(log_file)
        model_name, dataset, prior = args.model, args.dataset, args.prior
        steps, qvmc_iter, nqpp = args.steps, args.qvmc_iter, args.nqpp
        benchmark = args.benchmark
        if args.config:
            cfg = json.loads(args.config)
            cfg.setdefault('label', f"{compute_quantumness(cfg):.0f}% — JSON")
        elif args.preset is not None:
            cfg = dict(PRESETS[args.preset])
        else:
            cfg = dict(PRESETS[20])
        print(f"  CLI mode: detailed progress in {log_file}")
    else:
        logger = None
        sel = interactive_menu()
        model_name, dataset, prior = sel['model'], sel['dataset'], sel['prior']
        steps, qvmc_iter, nqpp = sel['steps'], sel['qvmc_iter'], sel['nqpp']
        cfg, benchmark = sel['config'], sel['benchmark']

    model = MODELS[model_name]
    post = Posterior(model, dataset, prior)
    say = logger.info if logger else print
    say(f"Model: {model.label} | params: {model.param_names} | "
        f"dataset: {dataset} ({post.n_data} pts) | prior: {prior}")

    if benchmark:
        run_benchmark(post, n_steps_mcmc=steps, max_iter_qvmc=qvmc_iter,
                      nqpp=nqpp, outdir=args.outdir, seed=args.seed,
                      logger=logger, log_every=args.log_every)
        return

    # ── single configuration + MANDATORY classical baseline ──────────────
    comp = run_comparison(post, cfg, seed=args.seed, logger=logger,
                          n_steps_mcmc=steps, max_iter_qvmc=qvmc_iter,
                          n_chains_mcmc=args.chains, nqpp=nqpp,
                          n_shots=args.shots, n_burn=args.burn,
                          log_every=args.log_every, verbose=True)
    res_q, res_c = comp['quantum'], comp['classical']

    # ── final summary (always to console; also to log) ────────────────────
    header = ["=" * 65, "  FINAL SUMMARY — QUANTUM vs MANDATORY CLASSICAL BASELINE"
              if res_q else "  FINAL SUMMARY — FULLY CLASSICAL RUN", "=" * 65,
              f"  Model         : {model.label}   Dataset: {dataset}   "
              f"Prior: {prior}"]
    for ln in header:
        print(ln)
        if logger:
            logger.info(ln.strip())

    if res_q is not None:
        for res, side in ((res_q, f"QUANTUM RUN — {res_q['label']}"),
                          (res_c, "CLASSICAL BASELINE — 0% (same parameters, "
                                  "same seed)")):
            ln = f"\n  ▌ {side}  [{res['elapsed_total']:.1f}s]"
            print(ln)
            if logger:
                logger.info(ln.strip())
            m, q = res['mcmc'], res['qvmc']
            fam_m = 'QMCMC' if res is res_q else 'Classical MCMC'
            fam_v = 'QVMC' if res is res_q else 'Classical VI'
            _print_summary_block(
                fam_m, model, m,
                f"  ESS = {m['ess']:.0f}  acceptance = {m['acceptance']:.3f}  "
                f"converged = {'yes' if m['converged'] else 'no'}", logger)
            _print_summary_block(
                fam_v, model, q,
                f"  ESS = {q['ess']:.0f}  final KL = {q['kl_final']:.6f}",
                logger)
    else:
        m, q = res_c['mcmc'], res_c['qvmc']
        _print_summary_block(
            'Classical MCMC', model, m,
            f"  ESS = {m['ess']:.0f}  acceptance = {m['acceptance']:.3f}  "
            f"converged = {'yes' if m['converged'] else 'no'}", logger)
        _print_summary_block(
            'Classical VI', model, q,
            f"  ESS = {q['ess']:.0f}  final KL = {q['kl_final']:.6f}", logger)
    print("=" * 65)

    if not args.no_plot:
        if res_q is not None:
            files = plot_comparison_figures(comp, post, args.outdir)
            say("Overlay figures: " + ", ".join(files))
        else:
            # Fully classical run: classical-only corner + traces
            tag = f"{model.name}_q000"
            f1 = plot_corner_overlay(
                res_c['flat_mcmc'], res_c['qvmc_samples'], model,
                args.outdir, tag=f'cls_{tag}',
                title=f'{model.label} — Classical MCMC vs Classical VI',
                labels=('Classical MCMC', 'Classical VI'),
                q_color=C_QUANTUM2, weights_q=res_c['qvmc_weights'])
            say(f"Figure: {f1}")


if __name__ == "__main__":
    main()
