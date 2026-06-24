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
import csv
import json
import os
import sys
import time
import warnings
from typing import List, Optional

import numpy as np
import matplotlib
# [FIX] Do NOT force the 'Agg' backend at import time — doing so disabled the
# live GUI of any module that imports this one (e.g. cosmo_genetic_optimizers).
# 'Agg' is now selected ONLY in CLI/batch mode, inside main(), via
# set_headless_backend(). Interactive importers keep their GUI backend.
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from scipy.optimize import minimize
from tqdm import tqdm

import corner  # [NEW] corner.py for overlaid 2D contour + 1D marginal plots

# [A5] Do NOT silence ALL warnings: a blanket filter hides exactly the
# RuntimeWarnings (overflow/invalid in log/exp/sqrt) that would flag an
# unphysical E²<0 or a log(0) in the likelihood. We silence only the few
# known-benign, library-emitted categories and let numerical warnings through.
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")


def set_headless_backend():
    """Switch Matplotlib to the non-interactive 'Agg' backend (CLI/HPC mode)."""
    matplotlib.use('Agg', force=True)


from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit_aer import AerSimulator

import cosmo_core as core
from cosmo_core import (MODELS, Posterior, RNG, ess_chains, ess_weights,
                        fit_statistics, fmt_theta, gelman_rubin_max,
                        gpu_available, make_run_dir, make_simulator,
                        resolve_device, setup_logger)

# Module-level GPU preference. main() sets this from --gpu (CLI) or the
# interactive menu; every AerSimulator in this module is built through
# `_sim(...)` which reads it, so the CPU/GPU choice is consistent everywhere.
USE_GPU = False


def _sim(method: str = 'statevector', **kwargs):
    """Project-wide AerSimulator factory honoring the module GPU preference."""
    return make_simulator(method=method, prefer_gpu=USE_GPU, **kwargs)

# ── Contrasting color convention used by EVERY overlay figure ────────────────
#    (requirement 3: blue = classical, red/orange = quantum)
C_CLASSICAL = '#1f77b4'   # blue   — Classical MCMC / Classical VI
C_CLASSICAL2 = '#17becf'  # teal   — Classical VI when shown ALONGSIDE
                          #          Classical MCMC in the same panel
                          #          (1-to-1 plots that mix both families)
C_QUANTUM   = '#d62728'   # red    — QMCMC (quantum MCMC family)
C_QUANTUM2  = '#ff7f0e'   # orange — QVMC (quantum variational family)


def _save_fig(fig, png_path, close=True):
    """Save a figure as PNG+PDF, creating the output directory if needed.

    [FIX] Centralizes figure saving so the destination folder is guaranteed to
    exist before writing — this prevents the FileNotFoundError that occurred
    when a figure was saved before the run directory had been created.
    """
    d = os.path.dirname(png_path)
    if d:
        os.makedirs(d, exist_ok=True)
    fig.savefig(png_path, dpi=150, bbox_inches='tight')
    fig.savefig(png_path.replace('.png', '.pdf'), bbox_inches='tight')
    if close:
        plt.close(fig)
    return png_path


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
# 0.  CLASSICAL-QUANTUM ABLATION FRAMEWORK
# =============================================================================
#
#  WHAT THIS IS (and is NOT).
#  --------------------------
#  The "quantumness %" reported throughout the project is NOT a measure of
#  quantum computational resource (it does not count entangling depth, magic,
#  non-Clifford gates or quantum volume). It is the index of a CLASSICAL-
#  QUANTUM ABLATION: the pipeline is decomposed into well-delimited
#  substitution points (proposal, acceptance, sampling, training,
#  normalization), each of which can be run classically or replaced by its
#  quantum implementation. The index measures HOW MANY of those substitution
#  points are currently switched to quantum — nothing more. This is the
#  standard ablation methodology (isolate the effect of one module at a time),
#  applied to classical→quantum substitution.
#
#  UNIFORM WEIGHTING (option A).
#  ----------------------------
#  Every substitution point counts EQUALLY. The previous version assigned
#  hand-picked weights (proposal 20, acceptance 25, ...) meant to reflect
#  "how much quantum structure each injects" — a subjective quantity with no
#  measurable definition. Under the ablation reading the only defensible index
#  is the plain fraction of active substitution points, so the global index is
#
#      ablation_index = 100 · (#components switched to quantum) / (#components).
#
#  The historical weighted numbers are still computed by
#  `legacy_weighted_index` and may be reported as a clearly-labelled,
#  heuristic secondary index for continuity with older CSVs/figures.
#
#  FAITHFUL vs ALGORITHMIC cells (the key classification).
#  ------------------------------------------------------
#  Each substitution point is tagged by the kind of ablation it produces:
#
#    * FAITHFUL  — the quantum implementation encodes the EXACT SAME rule as
#                  the classical one, so flipping it MUST leave the result
#                  statistically unchanged. These are the NULL CELLS of the
#                  ablation: their job is to verify that the substitution is
#                  faithful (introduces no bias). A faithful cell that changed
#                  the answer would be the bug; a faithful cell that does NOT
#                  change it is the *result* (the quantum step reproduces the
#                  classical one). Example: the Metropolis acceptance encoded
#                  as an RY amplitude — it computes min(1, e^Δ) either way.
#    * ALGORITHMIC — the quantum implementation is a genuinely different
#                  algorithm, expected to change the result. These are the
#                  TREATMENT CELLS. Example: the QVMC quantum training
#                  (parameter-shift) reaches a different KL minimum than
#                  classical COBYLA.
#
#  This tag turns the project's central narrative ("some rungs coincide by
#  design, others differ") into first-class metadata instead of prose, and
#  makes every FAITHFUL cell a falsifiable correctness test (see tests/).

FAITHFUL = 'faithful'        # null-cell: quantum must reproduce classical
ALGORITHMIC = 'algorithmic'  # treatment-cell: quantum is a distinct algorithm

QUANTUM_COMPONENTS = {
    'proposal':      {'kind': ALGORITHMIC,
                      'name': 'QMCMC proposal (statevector circuit, Sarracino)'},
    'acceptance':    {'kind': FAITHFUL,
                      'name': 'MH acceptance (RY amplitude encoding of Metropolis)'},
    'training':      {'kind': ALGORITHMIC,
                      'name': 'QVMC training (parameter-shift gradient)'},
    'sampling':      {'kind': FAITHFUL,
                      'name': 'QVMC sampling (shots from the trained state)'},
    'normalization': {'kind': FAITHFUL,
                      'name': 'Normalization (exact renormalization)'},
}

#: Legacy hand-picked weights (NO measurable definition). Retained ONLY so
#: `legacy_weighted_index` can reproduce the old numbers for continuity.
_LEGACY_WEIGHTS = {'proposal': 20, 'acceptance': 25, 'training': 20,
                   'sampling': 25, 'normalization': 10}

PRESETS = {
    0:   dict(proposal=False, acceptance=False, training=False, sampling=False,
              normalization=False, label='0/5 quantum — Fully Classical'),
    20:  dict(proposal=True, acceptance=False, training=False, sampling=False,
              normalization=False, label='1/5 quantum — Quantum proposal only (Sarracino)'),
    45:  dict(proposal=True, acceptance=False, training=False, sampling=True,
              normalization=False, label='2/5 quantum — Proposal + Quantum sampling'),
    70:  dict(proposal=True, acceptance=True, training=False, sampling=True,
              normalization=False, label='3/5 quantum — + Quantum acceptance'),
    90:  dict(proposal=True, acceptance=True, training=True, sampling=True,
              normalization=False, label='4/5 quantum — + Quantum training (no QAE)'),
    100: dict(proposal=True, acceptance=True, training=True, sampling=True,
              normalization=True, label='5/5 quantum — Fully Quantum'),
}

#: [BASE] The exact classical counterpart used as the mandatory baseline:
#: same code path with every substitution point switched off.
CLASSICAL_BASELINE = dict(PRESETS[0])


def compute_quantumness(config: dict) -> float:
    """Global ablation index: 100 · (#quantum components)/(#components).

    Uniform weighting (option A): every substitution point counts equally.
    This is the fraction of substitution points currently switched to
    quantum — an ablation index, NOT a quantum-resource metric.
    """
    n_total = len(QUANTUM_COMPONENTS)
    n_quantum = sum(1 for k in QUANTUM_COMPONENTS if config.get(k, False))
    return round(100.0 * n_quantum / n_total, 1)


def legacy_weighted_index(config: dict) -> float:
    """Historical hand-weighted index (heuristic; for CSV continuity only).

    Reproduces the pre-ablation "quantumness %" that used subjective
    per-component weights. Reported, if at all, as a clearly-labelled
    secondary number — never as the primary index.
    """
    total = sum(_LEGACY_WEIGHTS.values())
    earned = sum(_LEGACY_WEIGHTS[k] for k in _LEGACY_WEIGHTS
                 if config.get(k, False))
    return round(100.0 * earned / total, 1)


def component_kinds(config: dict) -> dict:
    """Map each ACTIVE quantum component to its FAITHFUL/ALGORITHMIC kind.

    Lets figures, tables and tests state, per configuration, which active
    substitutions are null cells (must reproduce classical) and which are
    treatment cells (expected to differ).
    """
    return {k: QUANTUM_COMPONENTS[k]['kind']
            for k in QUANTUM_COMPONENTS if config.get(k, False)}


def quantumness_label(pct: float) -> str:
    """Human-readable label for a global ablation level (#quantum / 5)."""
    n = int(round(pct / 100.0 * len(QUANTUM_COMPONENTS)))
    if n == 0:
        return "Fully Classical (0/5 quantum)"
    if n >= len(QUANTUM_COMPONENTS):
        return "Fully Quantum (5/5 quantum)"
    return f"Hybrid ({n}/{len(QUANTUM_COMPONENTS)} quantum)"


# ── Per-method ablation axes (uniform weighting) ─────────────────────────────
# The global index bundles two INDEPENDENT samplers, so it is not monotonic
# for either one. These per-method indices fix that: each counts only the
# substitution points the corresponding sampler actually reads, and each
# ladder rung flips exactly ONE point, so the axis is monotonic and every step
# has a well-defined meaning. Under uniform weighting the per-method index is
# simply (#active quantum points for that sampler)/(#points for that sampler):
#
#   QMCMC reads : proposal, acceptance            -> rungs 0, 1/2 (50%), 2/2 (100%)
#   QVMC  reads : sampling, training, normalization -> 0, 1/3 (33%), 2/3 (67%), 3/3 (100%)
#
# NOTE. These replace the old weighted per-method numbers (QMCMC 44/100,
# QVMC 46/82/100), which were artifacts of the discarded subjective weights.
# Ladder ORDER is unchanged (proposal→acceptance; sampling→training→norm), so
# the rungs still line up one-to-one with the presets — only the axis labels
# change from weighted percentages to honest equal-spaced fractions.
_QMCMC_ORDER = ['proposal', 'acceptance']
_QVMC_ORDER = ['sampling', 'training', 'normalization']


def quantumness_qmcmc(config: dict) -> float:
    """QMCMC-only ablation index: (#active of proposal, acceptance)/2 · 100."""
    n = sum(1 for c in _QMCMC_ORDER if config.get(c, False))
    return round(100.0 * n / len(_QMCMC_ORDER), 1)


def quantumness_qvmc(config: dict) -> float:
    """QVMC-only ablation index: (#active of sampling, training, norm)/3 · 100."""
    n = sum(1 for c in _QVMC_ORDER if config.get(c, False))
    return round(100.0 * n / len(_QVMC_ORDER), 1)


def qmcmc_ladder() -> List[dict]:
    """Monotonic QMCMC configs: classical → +proposal → +acceptance."""
    rungs, cur = [dict()], dict()
    for comp in _QMCMC_ORDER:
        cur = dict(cur); cur[comp] = True
        rungs.append(cur)
    return rungs


def qvmc_ladder() -> List[dict]:
    """Monotonic QVMC configs: classical → +sampling → +training → +norm."""
    rungs, cur = [dict()], dict()
    for comp in _QVMC_ORDER:
        cur = dict(cur); cur[comp] = True
        rungs.append(cur)
    return rungs


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
        self.sim = _sim('statevector')
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

    [DESIGN — Metropolis, by request] The quantum acceptance encodes the
    SAME Metropolis rule A = min(1, e^Δ), Δ = lp_prop − lp_cur, that the
    classical baseline uses. The ONLY difference is that A is read off a
    state amplitude rather than compared to a uniform in NumPy. This is
    deliberate: the goal is to show the quantum method *replicates* the
    classical one, so the quantum acceptance must reproduce the classical
    result exactly (given the same proposal and random stream, the QMCMC
    chain is then identical with the acceptance switch on or off). In other
    words, the `acceptance` component is a faithful quantum reproduction,
    not a different sampler — that equivalence IS the result.

    A single-qubit RY(θ) with θ = 2·arccos(√A) prepares
    cos(θ/2)|0⟩ + sin(θ/2)|1⟩, so P(|0⟩) = cos²(θ/2) = A exactly. The
    amplitude is read from the Aer statevector (genuinely "quantum"), and
    the circuit + simulator are cached.

    [HISTORY] The first version read P(ancilla=0) of a CRY/Hadamard-test
    circuit that DECREASED with Δ (it accepted worse moves) — an inverted
    bug now guarded by `sanity_check_routing`'s monotonicity assertion. A
    later variant used the Barker rule σ(Δ); we switched to Metropolis so
    the quantum and classical acceptances coincide (replication goal). To
    instead make the acceptance a *distinct* kernel, set A = σ(Δ) here
    (Barker) — it shares the same stationary distribution but mixes
    differently.
    """
    if not np.isfinite(lp_prop):
        return -np.inf
    delta = float(np.clip(lp_prop - lp_cur, -700, 700))
    # Metropolis acceptance A = min(1, e^Δ) — identical to the classical
    # baseline so the quantum acceptance reproduces it exactly.
    A = min(1.0, float(np.exp(delta)))
    A = min(max(A, 1e-12), 1.0)
    if _HAD['qc_t'] is None:
        par = ParameterVector('theta', 1)
        qc = QuantumCircuit(1)
        qc.ry(par[0], 0)
        qc.save_statevector()
        _HAD['sim'] = _sim('statevector')
        _HAD['qc_t'] = transpile(qc, _HAD['sim'])
        _HAD['par'] = par
    theta = 2.0 * np.arccos(np.sqrt(A))
    bound = _HAD['qc_t'].assign_parameters({_HAD['par'][0]: theta})
    sv = np.asarray(_HAD['sim'].run(bound).result().get_statevector())
    prob_zero = float(np.abs(sv[0])**2)   # P(|0>) = A = min(1,e^Δ)
    return float(np.log(prob_zero + 1e-12))


def hadamard_accept_log_batch(lp_cur: np.ndarray,
                              lp_prop: np.ndarray) -> np.ndarray:
    """Vectorized quantum acceptance for a BATCH of chains in ONE Aer job.

    [A2] The acceptance is a FAITHFUL (null) cell: the single-qubit RY circuit
    reproduces the Metropolis A = min(1, e^Δ) EXACTLY (P(|0⟩) = A). The earlier
    code ran one Aer job per chain inside a Python loop, breaking the
    vectorization used everywhere else in the kernel. Here all M shifted
    single-qubit circuits travel in ONE Aer call (one circuit per chain, same
    transpiled template), then we read P(|0⟩) from each statevector. The
    returned log-acceptances are identical to the per-chain version; only the
    job overhead is removed.

    Returns:
        log A for each chain, shape (M,). Out-of-box proposals (non-finite
        lp_prop) return -inf (rejected).
    """
    lp_cur = np.asarray(lp_cur, dtype=float)
    lp_prop = np.asarray(lp_prop, dtype=float)
    M = len(lp_prop)
    out = np.full(M, -np.inf)
    finite = np.isfinite(lp_prop)
    if not np.any(finite):
        return out
    delta = np.clip(lp_prop[finite] - lp_cur[finite], -700, 700)
    A = np.minimum(1.0, np.exp(delta))
    A = np.clip(A, 1e-12, 1.0)
    thetas = 2.0 * np.arccos(np.sqrt(A))           # RY angle per chain
    if _HAD['qc_t'] is None:
        par = ParameterVector('theta', 1)
        qc = QuantumCircuit(1)
        qc.ry(par[0], 0)
        qc.save_statevector()
        _HAD['sim'] = _sim('statevector')
        _HAD['qc_t'] = transpile(qc, _HAD['sim'])
        _HAD['par'] = par
    circs = [_HAD['qc_t'].assign_parameters({_HAD['par'][0]: float(t)})
             for t in thetas]
    res = _HAD['sim'].run(circs).result()          # ONE job, all chains
    logs = np.empty(len(thetas))
    for k in range(len(thetas)):
        sv = np.asarray(res.get_statevector(k))
        logs[k] = np.log(float(np.abs(sv[0]) ** 2) + 1e-12)
    out[finite] = logs
    return out


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
        Quantum acceptance: ALL chains evaluated in ONE batched Aer job
        (see hadamard_accept_log_batch) — no per-chain loop.
        """
        log_u = np.log(RNG.uniform(size=self.n_chains) + 1e-300)
        if self.q_acc:
            _sanity('QMCMC.accept', 'quantum',
                    'RY amplitude encoding of min(1,e^Δ) (Metropolis), '
                    'all chains in one Aer job')
            log_A = hadamard_accept_log_batch(lp_cur, lp_prop)
            return log_u < log_A
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
    sim = _sim('statevector')
    qc = QuantumCircuit(n + 1)
    qc.h(range(n + 1))
    norm = float(np.sum(P_unnorm))
    angle = 2 * np.arcsin(np.sqrt(np.clip(norm / len(P_unnorm), 0, 1)))
    qc.ry(angle, n)
    qc.save_statevector()
    sim.run(transpile(qc, sim)).result()
    return P_unnorm / (norm + 1e-15)


def estimate_grid_window(post: Posterior, sigma_mult: float = 4.0,
                         n_steps: int = 400, n_chains: int = 4) -> List[tuple]:
    """[B3] Re-exported from cosmo_core so the adaptive grid is shared by the
    simulator and QPU pipelines (single source of truth). See
    `cosmo_core.estimate_grid_window` for the full docstring."""
    return core.estimate_grid_window(post, sigma_mult=sigma_mult,
                                     n_steps=n_steps, n_chains=n_chains)


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
                 n_shots: int = 2000, lr_train: float = 0.05,
                 grid_window=None, adaptive_grid: bool = True,
                 grid_sigma: float = None):
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
        # [ADAPTIVE GRID — option b] Center+zoom the discrete grid on the
        # posterior so a few qubits resolve its width (see
        # estimate_grid_window). The window half-width in σ is chosen FROM
        # the grid size: with only 2^nqpp points, a ±4σ window leaves the
        # spacing ~1σ (a spike); we instead size the window so ~3 cells fall
        # within ±1σ, i.e. σ_mult ≈ (n_grid−1)/6, clipped to [2, 5]. So a
        # small grid zooms in tightly (resolving a coarse peak) and a larger
        # grid widens out (resolving smooth Gaussian tails). A precomputed
        # `grid_window` is reused across ladder rungs so the pre-fit runs
        # once and every QVMC / classical-VI instance shares the SAME grid.
        if grid_sigma is None:
            grid_sigma = float(np.clip((self.n_grid - 1) / 6.0, 2.0, 5.0))
        self.grid_sigma = grid_sigma
        if grid_window is not None:
            self.grid_window = list(grid_window)
        elif adaptive_grid:
            self.grid_window = estimate_grid_window(post, sigma_mult=grid_sigma)
        else:
            self.grid_window = list(self.model.sample_box)
        self.grids = [np.linspace(lo, hi, self.n_grid)
                      for lo, hi in self.grid_window]
        self.sim = _sim('statevector')
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
            # [FIX v4 — high-quantumness divergence] On the ladder, the rungs
            # that turn quantum *training* ON (82% and 100%) used to DIVERGE:
            # KL fell to a minimum near iter ~150 and then CREPT BACK UP to
            # ~2.2, collapsing the distribution and crashing the ESS. The
            # earlier tuning (lr0=0.05, decay=0.02) was calibrated for nqpp=3
            # (~42 ansatz angles); with nqpp=6 the ansatz has many more angles
            # and a larger-norm gradient, so a fixed lr overshoots the
            # shrinking gradient near the optimum and the KL rebounds.
            #
            # Three reinforcing fixes, all evidence-based:
            #   (1) GRADIENT NORMALIZATION (clip to unit norm above a cap): one
            #       step can no longer explode just because there are more
            #       angles — decouples the step size from nqpp.
            #   (2) lr DECAY SCALED BY #angles: gamma grows with n_p so larger
            #       ansätze cool down faster (the regime that used to diverge).
            #   (3) BEST-SO-FAR: we keep the φ with the lowest KL ever seen and
            #       return THAT, not the last iterate. Even if the tail wobbles,
            #       the returned model is the true minimum — this alone removes
            #       the "creep-up" pathology from the reported result.
            lr0 = self.lr_train
            decay = 0.02 * max(1.0, n_p / 42.0)        # (2) scale with #angles
            grad_cap = 1.0                              # (1) max gradient norm
            best_kl, best_phi = np.inf, phi.copy()      # (3) best-so-far
            it_r = range(max_iter)
            if progress:
                it_r = tqdm(it_r, desc=f"  {tag} param-shift",
                            leave=False, ncols=80)
            for i in it_r:
                kl, Qs = self._kl_batch(phi, qc_t, P_target, return_q=True)
                kl0 = float(kl[0])
                record(i, kl0, Qs[0])
                if kl0 < best_kl:                       # (3) track the best
                    best_kl, best_phi = kl0, phi.copy()
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
                # (1) clip the gradient norm so the step cannot blow up
                gnorm = float(np.linalg.norm(grad))
                if gnorm > grad_cap:
                    grad = grad * (grad_cap / gnorm)
                phi = phi - (lr0 / (1.0 + decay * i)) * grad
            # (3) return the best iterate, not the last — kills the creep-up
            phi_opt = best_phi
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
    _save_fig(fig, f)
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
    _save_fig(fig, f)
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
    _save_fig(fig, f)
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
    _save_fig(fig, f)
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
    _save_fig(fig, f)
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
    os.makedirs(os.path.dirname(f) or '.', exist_ok=True)
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

# =============================================================================
# 6.  BENCHMARK
# =============================================================================

# =============================================================================
# 6b. PER-METHOD QUANTUMNESS LADDERS (Option A)
# =============================================================================

def run_quantumness_ladder(post: Posterior, n_steps_mcmc: int,
                           max_iter_qvmc: int, nqpp: int, outdir: str,
                           seed: int = 42, logger=None, log_every: int = 500,
                           n_chains_mcmc: int = 6, n_chains_qvmc: int = 3,
                           n_shots: int = 2000,
                           csv_paths: Optional[list] = None,
                           dataset_label: str = "", prior_type: str = "") -> dict:
    """Run TWO independent, monotonic per-method quantumness ladders.

    This is the canonical benchmark: it sweeps each sampler along ITS OWN
    monotonic axis, adding one quantum component at a time:

        QMCMC:  classical → +proposal → +acceptance
        QVMC :  classical → +sampling → +training → +normalization

    Each rung re-seeds identically (fair comparison) and the rung at 0 % is
    the classical baseline of that method. Because we keep the Metropolis
    acceptance, the quantum acceptance rung reproduces the classical result
    exactly — which is the point: it demonstrates the quantum method
    *replicates* the classical one component by component.
    """
    say = logger.info if logger else print
    model = post.model
    n_burn = max(50, int(0.1 * n_steps_mcmc))
    say(f"PER-METHOD LADDERS — model {model.label}, dataset {post.dataset}")

    # ── QMCMC ladder ────────────────────────────────────────────────────────
    qmcmc_runs = []
    for cfg in qmcmc_ladder():
        pct = quantumness_qmcmc(cfg)
        _reseed(seed)
        tag = 'C-MCMC' if pct == 0 else f'QMCMC{pct:.0f}'
        mc = QMCMCModular(post, cfg, n_chains=n_chains_mcmc, n_burn=n_burn,
                          stop_on_convergence=False)
        _t0 = time.time()
        r = mc.run(n_steps=n_steps_mcmc, logger=logger, log_every=log_every,
                   progress=(logger is None), tag=tag)
        fs = fit_statistics(post, r['flat'].mean(0))
        qmcmc_runs.append({'pct': pct, 'cfg': cfg, 'flat': r['flat'],
                           'chains': r['chains'], 'rhat_hist': r['rhat_hist'],
                           'acceptance': r['acceptance'], 'ess': r['ess'],
                           'elapsed': time.time() - _t0,
                           'mu': r['flat'].mean(0), 'std': r['flat'].std(0),
                           **fs})
        say(f"  QMCMC {pct:5.1f}%  mean={fmt_theta(model, r['flat'].mean(0))}"
            f"  acc={r['acceptance']:.3f}")

    # ── QVMC ladder ───────────────────────────────────────────────────────────
    # [ADAPTIVE GRID — option b] Compute the zoomed grid window ONCE (a quick
    # classical pre-fit) and share it across every rung AND the classical-VI
    # baseline, so all QVMC distributions live on the exact same grid.
    _reseed(seed)
    shared_window = estimate_grid_window(post)
    say(f"  Adaptive QVMC grid window (shared): "
        + ", ".join(f"{model.param_names[i]}∈[{lo:.4f},{hi:.4f}]"
                    for i, (lo, hi) in enumerate(shared_window)))
    qvmc_runs = []
    for cfg in qvmc_ladder():
        pct = quantumness_qvmc(cfg)
        _reseed(seed)
        tag = 'C-VI' if pct == 0 else f'QVMC{pct:.0f}'
        qv = QVMCModular(post, cfg, n_qubits_per_param=nqpp, n_shots=n_shots,
                         grid_window=shared_window)
        _t0 = time.time()
        res = qv.run(max_iter=max_iter_qvmc, n_chains=n_chains_qvmc,
                     logger=logger, log_every=log_every,
                     progress=(logger is None), tag=tag)
        fs = fit_statistics(post, res['mu'])
        qvmc_runs.append({'pct': pct, 'cfg': cfg, 'S': res['S'],
                          'W': res['W'], 'history': res['history'],
                          'mu': res['mu'], 'std': res['sd'],
                          'kl_final': res['kl_final'], 'ess': res['ess'],
                          'elapsed': time.time() - _t0, **fs})
        say(f"  QVMC  {pct:5.1f}%  mean={fmt_theta(model, res['mu'])}"
            f"  KL={res['kl_final']:.4f}")

    meta = dict(n_steps=n_steps_mcmc, n_iter=max_iter_qvmc, nqpp=nqpp)
    plot_method_ladders(qmcmc_runs, qvmc_runs, model, outdir, meta)
    say(f"Ladder figures in {outdir}/: ladder_qmcmc_*, "
        f"ladder_qvmc_*, ladder_rhat_qmcmc_*, ladder_kl_qvmc_*, "
        f"ladder_1to1_*, ladder_summary_*, ladder_trends_*")

    # [FIX] Write the benchmark to CSV too. Previously only single-config runs
    # wrote resultados_config.csv; the --benchmark path returned before the
    # writer, so the ladder produced PNG tables but no CSV. Here we emit ONE row
    # per rung of each ladder (every parameter included) to BOTH the per-run
    # (named columns) and the cumulative (generic columns) files.
    if csv_paths:
        run_csv, cumulative_csv = csv_paths[0], (
            csv_paths[1] if len(csv_paths) > 1 else "")
        side_rows = []
        for r in qmcmc_runs:
            label = ("Classical MCMC" if r['pct'] == 0
                     else f"QMCMC {r['pct']:.0f}%")
            side_rows.append((r, label, True, "—" if r['pct'] == 0 else nqpp))
        for r in qvmc_runs:
            label = ("Classical VI" if r['pct'] == 0
                     else f"QVMC {r['pct']:.0f}%")
            side_rows.append((r, label, False, "—" if r['pct'] == 0 else nqpp))
        write_run_and_cumulative(side_rows, model, run_csv, cumulative_csv,
                                 dataset_label, prior_type)
        say(f"Benchmark results written to {run_csv} "
            f"(+ cumulative resultados_config.csv)")

    return {'qmcmc': qmcmc_runs, 'qvmc': qvmc_runs}


def plot_method_ladders(qmcmc_runs, qvmc_runs, model, outdir, meta):
    """All per-method ladder figures (family overlay, diagnostics, 1-to-1)."""
    name = model.name
    steps, iters, nqpp = meta['n_steps'], meta['n_iter'], meta['nqpp']

    # QVMC lives on a DISCRETE 2^nqpp grid per parameter, so its raw samples
    # land on a few fixed values and a corner plot shows spikes, never a
    # smooth (Gaussian-like) blob. For VISUALIZATION only we add uniform
    # jitter of ±half a grid cell, which spreads each grid point across the
    # cell it represents and reveals the continuous distribution the grid is
    # approximating. (Statistics/means/KL are always computed on the exact
    # un-jittered samples; this never touches the numbers, only the picture.)
    n_grid = 2 ** nqpp
    cell = np.array([(hi - lo) / (n_grid - 1)
                     for lo, hi in model.sample_box])

    def _jitter(S):
        return S + RNG.uniform(-0.5, 0.5, size=S.shape) * cell

    # ── QMCMC family corner: classical + every QMCMC rung ────────────────────
    q_cols = _q_colors([r['pct'] for r in qmcmc_runs])
    plot_corner_multi(
        [r['flat'] for r in qmcmc_runs], q_cols,
        [f"QMCMC {r['pct']:.0f}%" + (" (classical)" if r['pct'] == 0 else "")
         for r in qmcmc_runs],
        model, outdir, f'ladder_qmcmc_{name}',
        title=f"{model.label} — QMCMC quantumness ladder  [steps={steps}]")

    # ── QVMC family corner: classical + every QVMC rung (weighted) ───────────
    v_cols = _q_colors([r['pct'] for r in qvmc_runs])
    plot_corner_multi(
        [_jitter(r['S']) for r in qvmc_runs], v_cols,
        [f"QVMC {r['pct']:.0f}%" + (" (classical)" if r['pct'] == 0 else "")
         for r in qvmc_runs],
        model, outdir, f'ladder_qvmc_{name}',
        title=f"{model.label} — QVMC quantumness ladder  "
              f"[iters={iters}, nqpp={nqpp}]  (cell-jittered for display)",
        weights_list=[r['W'] for r in qvmc_runs])

    # ── R̂ overlay along the QMCMC ladder ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5.2))
    for r, col in zip(qmcmc_runs, q_cols):
        if not r['rhat_hist']:
            continue
        s, rh = zip(*r['rhat_hist'])
        lw = 2.8 if r['pct'] == 0 else 1.9
        ax.semilogy(s, np.array(rh) - 1, 'o-', color=col, lw=lw, ms=4,
                    label=f"{r['pct']:.0f}%")
    ax.axhline(0.05, color='k', ls='--', lw=1.2, label=r'$\hat R-1=0.05$')
    ax.set_xlabel('Sampling steps'); ax.set_ylabel(r'$\hat{R}_{\max}-1$')
    ax.set_title(f'QMCMC convergence along the quantumness ladder\n'
                 f'(total steps = {steps})')
    ax.legend(title='QMCMC %', fontsize=9); ax.grid(True, alpha=0.3)
    f = os.path.join(outdir, f'ladder_rhat_qmcmc_{name}.png')
    _save_fig(fig, f)

    # ── KL overlay along the QVMC ladder ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5.2))
    for r, col in zip(qvmc_runs, v_cols):
        h = r['history']
        if not h:
            continue
        lw = 2.8 if r['pct'] == 0 else 1.9
        ax.semilogy([d['it'] for d in h], [max(d['kl'], 1e-12) for d in h],
                    color=col, lw=lw, label=f"{r['pct']:.0f}%")
    ax.set_xlabel('Training iteration')
    ax.set_ylabel(r'KL$(Q_\varphi\,\|\,P_{\rm target})$')
    ax.set_title(f'QVMC training along the quantumness ladder\n'
                 f'(iterations = {iters}, nqpp = {nqpp})')
    ax.legend(title='QVMC %', fontsize=9); ax.grid(True, alpha=0.3)
    f = os.path.join(outdir, f'ladder_kl_qvmc_{name}.png')
    _save_fig(fig, f)

    # ── 1-to-1: each quantum rung vs its classical baseline ──────────────────
    base_m = qmcmc_runs[0]
    for r in qmcmc_runs[1:]:
        plot_corner_overlay(
            base_m['flat'], r['flat'], model, outdir,
            f"ladder_1to1_qmcmc_{name}_q{int(r['pct']):03d}",
            title=f"{model.label} — QMCMC {r['pct']:.0f}% vs classical  "
                  f"[steps={steps}]",
            labels=('Classical MCMC (0%)', f"QMCMC {r['pct']:.0f}%"),
            q_color=C_QUANTUM)
    base_v = qvmc_runs[0]
    for r in qvmc_runs[1:]:
        plot_corner_overlay(
            _jitter(base_v['S']), _jitter(r['S']), model, outdir,
            f"ladder_1to1_qvmc_{name}_q{int(r['pct']):03d}",
            title=f"{model.label} — QVMC {r['pct']:.0f}% vs classical  "
                  f"[iters={iters}, nqpp={nqpp}]  (cell-jittered)",
            labels=('Classical VI (0%)', f"QVMC {r['pct']:.0f}%"),
            q_color=C_QUANTUM2,
            weights_c=base_v['W'], weights_q=r['W'])

    # ── trend panels: how everything changes with quantumness ───────────────
    # (the comparison plots requested: H0 and Om vs quantumness, runtime,
    #  goodness-of-fit, ESS, acceptance/KL — one canonical per-method scale.)
    mq = sorted(qmcmc_runs, key=lambda r: r['pct'])
    vq = sorted(qvmc_runs, key=lambda r: r['pct'])
    mp = [r['pct'] for r in mq]
    vp = [r['pct'] for r in vq]

    # [FIX] Adaptive grid: one panel per model parameter PLUS six fixed
    # diagnostic panels (runtime, chi2_red, AIC, BIC, ESS, acceptance/KL). For
    # ΛCDM (2 params) this is 8 panels = the original 2x4 layout; wCDM (3) gives
    # 9, CPL (4) gives 10, laid out in 4 columns.
    npar = model.n_params
    n_panels = npar + 6
    ncols = 4
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4.5 * ncols, 4.0 * nrows))
    axes = np.atleast_2d(axes)
    flat_axes = axes.flatten()

    def _trend(ax, getter, ylabel, title, err=None, fid=None):
        ax.plot(mp, [getter(r) for r in mq], 'o-', color=C_QUANTUM, lw=2,
                ms=6, label='QMCMC')
        ax.plot(vp, [getter(r) for r in vq], 's-', color=C_QUANTUM2, lw=2,
                ms=6, label='QVMC')
        if err is not None:
            ax.errorbar(mp, [getter(r) for r in mq],
                        yerr=[err(r) for r in mq], fmt='none',
                        ecolor=C_QUANTUM, alpha=0.4, capsize=3)
            ax.errorbar(vp, [getter(r) for r in vq],
                        yerr=[err(r) for r in vq], fmt='none',
                        ecolor=C_QUANTUM2, alpha=0.4, capsize=3)
        if fid is not None:
            ax.axhline(fid, color='k', ls='--', lw=1, label='Fiducial')
        ax.set_xlabel('Quantumness %  (per method)')
        ax.set_ylabel(ylabel); ax.set_title(title); ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    # One panel per model parameter (closure binds i correctly via default arg).
    for i in range(npar):
        pname = model.param_names[i]
        _trend(flat_axes[i],
               (lambda r, i=i: r['mu'][i]), pname,
               f'{pname} vs quantumness',
               err=(lambda r, i=i: r['std'][i]),
               fid=model.fiducial[i])

    # Fixed diagnostic panels after the parameter panels.
    j = npar
    _trend(flat_axes[j], lambda r: r['elapsed'], 'runtime [s]',
           'Runtime vs quantumness'); j += 1
    _trend(flat_axes[j], lambda r: r['chi2_red'], r'$\chi^2_\nu$',
           'Reduced chi2 vs quantumness', fid=1.0); j += 1
    _trend(flat_axes[j], lambda r: r['AIC'], 'AIC', 'AIC vs quantumness'); j += 1
    _trend(flat_axes[j], lambda r: r['BIC'], 'BIC', 'BIC vs quantumness'); j += 1
    _trend(flat_axes[j], lambda r: r['ess'], 'ESS',
           'Effective sample size vs quantumness'); j += 1
    # acceptance is QMCMC-only; KL is QVMC-only → twin axes
    ax = flat_axes[j]
    ax.plot(mp, [r['acceptance'] for r in mq], 'o-', color=C_QUANTUM,
            lw=2, ms=6, label='QMCMC acceptance')
    ax.set_xlabel('Quantumness %  (per method)')
    ax.set_ylabel('MCMC acceptance', color=C_QUANTUM)
    ax.tick_params(axis='y', labelcolor=C_QUANTUM)
    ax2 = ax.twinx()
    ax2.semilogy(vp, [max(r['kl_final'], 1e-6) for r in vq], 's-',
                 color=C_QUANTUM2, lw=2, ms=6, label='QVMC final KL')
    ax2.set_ylabel('QVMC final KL', color=C_QUANTUM2)
    ax2.tick_params(axis='y', labelcolor=C_QUANTUM2)
    ax.set_title('Acceptance (QMCMC) & KL (QVMC)'); ax.grid(alpha=0.3)
    j += 1

    # Hide any unused panels in the grid.
    for k in range(j, len(flat_axes)):
        flat_axes[k].axis('off')

    fig.suptitle(f'{model.label} — trends along the per-method quantumness '
                 f'ladders\nsteps={steps} | iters={iters} | nqpp={nqpp}  '
                 '(QMCMC red, QVMC orange)',
                 fontsize=14, fontweight='bold', y=1.0)
    fig.tight_layout()
    f = os.path.join(outdir, f'ladder_trends_{name}.png')
    _save_fig(fig, f)

    # ── summary table (per-method %) ─────────────────────────────────────────
    # [FIX] Generalized to ALL model parameters (not just the first two), so
    # extended models (wCDM: w; CPL: w0, wa; GEDE: Delta) report every free
    # parameter. One "value±std" column per parameter, built from param_names.
    npar = model.n_params
    pnames = model.param_names
    # Wider figure when there are more parameters so the table stays readable.
    fig = plt.figure(figsize=(13 + 2.0 * max(0, npar - 2), 4.5))
    ax = fig.add_subplot(111); ax.axis('off')

    def _param_cells(r):
        # "value±std" for every parameter, with sensible precision per column.
        cells = []
        for i in range(npar):
            prec = 4 if i == 0 else (2 if pnames[i] == 'H0' else 4)
            cells.append(f"{r['mu'][i]:.{prec}f}±{r['std'][i]:.{prec}f}")
        return cells

    rows = []
    for r in qmcmc_runs:
        rows.append([f"QMCMC {r['pct']:.0f}%", *_param_cells(r),
                     f"{r['chi2_red']:.3f}", f"{r['AIC']:.1f}",
                     f"{r['elapsed']:.1f}s",
                     f"{r['acceptance']:.3f}", f"{r['ess']:.0f}", "—"])
    for r in qvmc_runs:
        rows.append([f"QVMC {r['pct']:.0f}%", *_param_cells(r),
                     f"{r['chi2_red']:.3f}", f"{r['AIC']:.1f}",
                     f"{r['elapsed']:.1f}s",
                     "—", f"{r['ess']:.0f}", f"{r['kl_final']:.4f}"])
    cols = ['method/level', *pnames, 'chi2_red', 'AIC', 'time',
            'acc', 'ESS', 'KL']
    tbl = ax.table(cellText=rows, colLabels=cols, loc='center',
                   cellLoc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 1.6)
    fig.suptitle(f'{model.label} — per-method quantumness ladders\n'
                 f'steps={steps} | iters={iters} | nqpp={nqpp}',
                 fontsize=13, fontweight='bold')
    f = os.path.join(outdir, f'ladder_summary_{name}.png')
    _save_fig(fig, f)


# =============================================================================
# 7.  INTERACTIVE MENU (default behavior with no arguments)
# =============================================================================

def csv_fields_for_model(model) -> list:
    """The CSV column schema for a given model: one mean/std pair per parameter.

    Centralized so every writer (single-config and benchmark ladder) produces
    the SAME header, with a `<param>_mean` / `<param>_std` pair for every free
    parameter — H0, Om AND the extra ones (wCDM: w; CPL: w0, wa; GEDE: Delta).

    Used for the PER-RUN CSV (one model per run → named, fully readable
    columns). The cumulative cross-run file uses `csv_fields_generic()` instead,
    because there different models with different parameters share one file.
    """
    fields = ['Method']
    for p in model.param_names:
        fields += [f'{p}_mean', f'{p}_std']
    fields += ['Time_s', 'nqpp', 'chi2', 'n_data', 'chi2_red', 'AIC', 'BIC',
               'acceptance', 'final_KL', 'ESS', 'dataset', 'prior']
    return fields


#: Max free parameters any model in MODELS has (CPL has 4: Om,H0,w0,wa).
_MAX_PARAMS = max(m.n_params for m in MODELS.values())


def csv_fields_generic() -> list:
    """Fixed, model-agnostic schema for the CUMULATIVE cross-run CSV.

    [FIX] A cumulative file can hold rows from DIFFERENT models (wCDM has w,
    CPL has w0+wa, …). Named per-parameter columns then misalign when a 4-param
    row is appended under a 3-param header. To keep one tidy cumulative table,
    this schema is fixed: it records the model, its parameter names, and up to
    `_MAX_PARAMS` generic value/std slots, so any model fits the same columns.
    """
    fields = ['Method', 'model', 'params']
    for i in range(1, _MAX_PARAMS + 1):
        fields += [f'p{i}_mean', f'p{i}_std']
    fields += ['Time_s', 'nqpp', 'chi2', 'n_data', 'chi2_red', 'AIC', 'BIC',
               'acceptance', 'final_KL', 'ESS', 'dataset', 'prior']
    return fields


def csv_row_generic(side: dict, model, method_label: str, is_mcmc: bool,
                    nqpp, dataset_label: str, prior_type: str) -> dict:
    """Build a row under the fixed generic schema (cumulative CSV).

    The parameter values go into p1..pN slots and `params` names them, so the
    row is self-describing regardless of model.
    """
    mu, sd = side['mu'], side['std']
    row = {'Method': method_label, 'model': model.name,
           'params': "|".join(model.param_names)}
    for i in range(_MAX_PARAMS):
        if i < len(mu):
            row[f'p{i+1}_mean'] = f"{mu[i]:.6f}"
            row[f'p{i+1}_std'] = f"{sd[i]:.6f}"
        else:
            row[f'p{i+1}_mean'] = ""
            row[f'p{i+1}_std'] = ""
    row['Time_s'] = f"{side.get('elapsed', float('nan')):.1f}"
    row['nqpp'] = str(nqpp)
    row['chi2'] = f"{side['chi2']:.4f}"
    row['n_data'] = str(side['n_data'])
    row['chi2_red'] = f"{side['chi2_red']:.4f}"
    row['AIC'] = f"{side['AIC']:.4f}"
    row['BIC'] = f"{side['BIC']:.4f}"
    row['acceptance'] = (f"{side['acceptance']:.4f}"
                         if is_mcmc and 'acceptance' in side else "")
    row['final_KL'] = (f"{side['kl_final']:.6f}"
                       if (not is_mcmc) and 'kl_final' in side else "")
    row['ESS'] = f"{side.get('ess', float('nan')):.1f}"
    row['dataset'] = dataset_label
    row['prior'] = prior_type
    return row


def csv_row_for_side(side: dict, model, method_label: str, is_mcmc: bool,
                     nqpp, dataset_label: str, prior_type: str) -> dict:
    """Build one CSV row from a result dict, with EVERY model parameter.

    `side` must carry 'mu' and 'std' arrays of length model.n_params plus the
    scalar diagnostics (chi2, chi2_red, AIC, BIC, ess, and acceptance OR
    kl_final depending on the family).
    """
    pnames = model.param_names
    mu, sd = side['mu'], side['std']
    row = {'Method': method_label}
    for i, p in enumerate(pnames):
        row[f'{p}_mean'] = f"{mu[i]:.6f}"
        row[f'{p}_std'] = f"{sd[i]:.6f}"
    row['Time_s'] = f"{side.get('elapsed', float('nan')):.1f}"
    row['nqpp'] = str(nqpp)
    row['chi2'] = f"{side['chi2']:.4f}"
    row['n_data'] = str(side['n_data'])
    row['chi2_red'] = f"{side['chi2_red']:.4f}"
    row['AIC'] = f"{side['AIC']:.4f}"
    row['BIC'] = f"{side['BIC']:.4f}"
    row['acceptance'] = (f"{side['acceptance']:.4f}"
                         if is_mcmc and 'acceptance' in side else "")
    row['final_KL'] = (f"{side['kl_final']:.6f}"
                       if (not is_mcmc) and 'kl_final' in side else "")
    row['ESS'] = f"{side.get('ess', float('nan')):.1f}"
    row['dataset'] = dataset_label
    row['prior'] = prior_type
    return row


def _write_csv_rows(rows: list, fields: list, csv_path: str) -> None:
    """Append already-built row dicts to a CSV, writing the header if new.

    [ROBUSTNESS — HPC] Guarded by an advisory file lock (fcntl, POSIX) so that
    concurrent SLURM array jobs appending to the SAME cumulative CSV cannot
    interleave/corrupt each other's rows or write duplicate headers. Best
    effort: on platforms without fcntl it degrades to a plain append.
    """
    lock_fh = None
    try:
        import fcntl
        lock_fh = open(csv_path + ".lock", 'w')
        fcntl.flock(lock_fh, fcntl.LOCK_EX)
    except Exception:
        lock_fh = None
    try:
        write_header = (not os.path.exists(csv_path)
                        or os.path.getsize(csv_path) == 0)
        with open(csv_path, 'a', newline='') as fh:
            wtr = csv.DictWriter(fh, fieldnames=fields, extrasaction='ignore')
            if write_header:
                wtr.writeheader()
            for r in rows:
                wtr.writerow(r)
            fh.flush()
            os.fsync(fh.fileno())
    finally:
        if lock_fh is not None:
            try:
                import fcntl
                fcntl.flock(lock_fh, fcntl.LOCK_UN)
                lock_fh.close()
            except Exception:
                pass


def write_run_and_cumulative(side_rows: list, model, run_csv: str,
                             cumulative_csv: str, dataset_label: str,
                             prior_type: str) -> None:
    """Write the SAME results to two files with two schemas.

    * run_csv: per-run file inside the run folder — NAMED per-parameter columns
      (one model per run, maximally readable).
    * cumulative_csv: cross-run file in the CWD — GENERIC schema, so rows from
      different models (wCDM, CPL, GEDE) line up under one fixed header.

    Args:
        side_rows: list of (side_dict, method_label, is_mcmc, nqpp) tuples.
        model: the CosmoModel.
        run_csv, cumulative_csv: destinations.
        dataset_label, prior_type: provenance columns.
    """
    named_fields = csv_fields_for_model(model)
    named_rows = [csv_row_for_side(s, model, lbl, mc, nq,
                                   dataset_label, prior_type)
                  for (s, lbl, mc, nq) in side_rows]
    _write_csv_rows(named_rows, named_fields, run_csv)

    if cumulative_csv:
        gen_fields = csv_fields_generic()
        gen_rows = [csv_row_generic(s, model, lbl, mc, nq,
                                    dataset_label, prior_type)
                    for (s, lbl, mc, nq) in side_rows]
        _write_csv_rows(gen_rows, gen_fields, cumulative_csv)


def append_results_csv(result_side: dict, model, dataset_label: str,
                       prior_type: str, family_mcmc: str, family_vi: str,
                       nqpp, csv_path: str) -> None:
    """Append the MCMC and VI rows of one single-config run to a PER-RUN CSV.

    [FIX] The modular script previously wrote NO CSV at all — only PNG tables —
    which is why `resultados_config.csv` stayed empty when running this module.
    Reports EVERY model parameter (wCDM: w; CPL: w0, wa; GEDE: Delta), one
    `<param>_mean` / `<param>_std` pair each. Two rows per call (MCMC + VI).

    This writes the NAMED per-run schema only; the cumulative file is handled
    by `write_run_and_cumulative` from main().
    """
    fields = csv_fields_for_model(model)
    rows = [
        csv_row_for_side(result_side['mcmc'], model, family_mcmc, True,
                         nqpp, dataset_label, prior_type),
        csv_row_for_side(result_side['qvmc'], model, family_vi, False,
                         nqpp, dataset_label, prior_type),
    ]
    _write_csv_rows(rows, fields, csv_path)


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
          "(monotonic increasing, matches Metropolis min(1,e^Δ))")

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
    post = Posterior(model, 'CC+BAO', 'flat')
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
        [2] Benchmark              — the per-method quantumness ladders
                                     (QMCMC and QVMC swept along their own
                                     axes), at user-chosen sizes. This is
                                     the canonical quantumness scale.
        [3] Quick TEST RUN         — the same ladders at small fixed sizes,
                                     a fast end-to-end stability check.

    If stdin is not a terminal (e.g. SLURM without --interactive), it
    falls back to a default preset so the job is not blocked.
    """
    if not sys.stdin.isatty():
        print("  [Non-interactive mode detected] Preset P20, ΛCDM, CC, flat.")
        return {'model': 'lcdm', 'dataset': 'CC+BAO', 'prior': 'flat',
                'config': dict(PRESETS[20]), 'steps': 1000, 'qvmc_iter': 300,
                'nqpp': 3, 'benchmark': False}

    print("\n" + "╔" + "═" * 63 + "╗")
    print("║   Modular Quantum/Classical Sampler — Configuration           ║")
    print("╚" + "═" * 63 + "╝")

    print("\n  ── Run mode ──")
    mode = _ask("Run mode",
                {'single': 'Single configuration (one preset + its baseline)',
                 'benchmark': 'Benchmark — per-method quantumness ladders '
                              '(QMCMC and QVMC, the canonical scale)',
                 'test': 'Quick TEST RUN (the ladders at small fixed sizes — '
                         'fast stability check)'},
                'single')

    print("\n  ── Cosmological model ──")
    model = _ask("Model", {k: v.label + f"  ({v.n_params} parameters: "
                           + ", ".join(v.param_names) + ")"
                           for k, v in MODELS.items()}, 'lcdm')

    print("\n  ── Dataset ──")
    pan = core.load_pantheon()
    panp = core.load_pantheon_plus()
    opts = {'CC+BAO': 'CC+BAO H(z) measurements'}
    if pan is not None:
        opts['Pantheon'] = f"Pantheon 2018 ({len(pan['z'])} SNe, diagonal)"
        opts['CC+BAO+Pantheon'] = 'CC+BAO + Pantheon 2018 combined'
    if panp is not None:
        opts['Pantheon+'] = (f"Pantheon+ 2022 ({len(panp['z'])} SNe, "
                             f"full covariance)")
        opts['CC+BAO+Pantheon+'] = 'CC+BAO + Pantheon+ 2022 combined'
    if pan is None and panp is None:
        print("    (no SNe files found → CC+BAO only)")
    dataset = _ask("Dataset", opts, 'CC+BAO')

    print("\n  ── Prior ──")
    prior = _ask("Prior", {'flat': 'Flat (box)',
                           'gaussian': 'Planck 2018 Gaussian on (Om, H0)'},
                 'flat')

    # ── Hardware / profiling (common to all run modes) ───────────────────────
    print("\n  ── Compute hardware ──")
    gpu_here = gpu_available()
    if gpu_here:
        use_gpu = _ask("Simulation device",
                       {'cpu': 'CPU', 'gpu': 'GPU (Aer on CUDA — detected)'},
                       'gpu') == 'gpu'
    else:
        print("    (no Aer GPU device detected → CPU; install qiskit-aer-gpu "
              "on a CUDA node to enable)")
        use_gpu = False
    profile = _ask("Profile memory / GPU-hours and save a usage figure?",
                   {'no': 'No', 'yes': 'Yes'}, 'no') == 'yes'
    _hw = {'use_gpu': use_gpu, 'profile': profile}

    # ── TEST RUN: the per-method ladders at small fixed sizes ────────────────
    if mode == 'test':
        print("\n  → Quick TEST RUN: per-method quantumness ladders at small "
              "fixed sizes (steps=200, iters=40, nqpp=2).")
        print("  QMCMC: classical→+proposal→+acceptance   |   "
              "QVMC: classical→+sampling→+training→+norm.")
        print("  Fast end-to-end stability check across both quantum axes.")
        return {'model': model, 'dataset': dataset, 'prior': prior,
                'config': dict(PRESETS[0]), 'steps': 200, 'qvmc_iter': 40,
                'nqpp': 2, 'benchmark': True, 'test_run': True, **_hw}

    # ── BENCHMARK: per-method quantumness ladders, user-chosen sizes ─────────
    if mode == 'benchmark':
        def ask_int_b(prompt, default):
            r = input(f"  {prompt} [Enter={default}]: ").strip()
            return int(r) if r.isdigit() else default
        print("\n  → Benchmark = per-method quantumness ladders.")
        print("  QMCMC: classical→+proposal→+acceptance   |   "
              "QVMC: classical→+sampling→+training→+norm.")
        print("\n  ── Sizes (apply to every rung) ──")
        steps = ask_int_b("MCMC steps", 1000)
        iters = ask_int_b("Variational iterations", 300)
        nqpp = ask_int_b("Qubits per parameter (2^n grid)", 3)
        return {'model': model, 'dataset': dataset, 'prior': prior,
                'config': dict(PRESETS[0]), 'steps': steps,
                'qvmc_iter': iters, 'nqpp': nqpp, 'benchmark': True, **_hw}

    # ── SINGLE configuration ─────────────────────────────────────────────────
    print("\n  ── Quantum substitution points (ablation) ──")
    for i, (k, meta) in enumerate(QUANTUM_COMPONENTS.items(), 1):
        tag = 'null/faithful' if meta['kind'] == FAITHFUL else 'treatment'
        print(f"    {i}. [{tag:>13s}]  {meta['name']}")
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
            'benchmark': False, **_hw}


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
                      help='Run the full quantumness benchmark: the two '
                           'per-method ladders (QMCMC: proposal->acceptance; '
                           'QVMC: sampling->training->normalization). This is '
                           'the canonical quantumness scale.')
    mode.add_argument('--sweep-all', action='store_true',
                      help='HPC batch mode: run the full quantumness benchmark '
                           '(QMCMC + QVMC ladders) for EVERY model in one go, '
                           'into a single master folder with one subfolder per '
                           'model. Designed to launch once on a supercomputer '
                           'and collect all results for comparison. Combine '
                           'with --sweep-models / --dataset / --steps / etc.')
    mode.add_argument('--config', type=str, metavar='JSON',
                      help='Component configuration as JSON')

    p.add_argument('--sweep-models', nargs='+', choices=list(MODELS),
                   default=None, metavar='MODEL',
                   help='Restrict --sweep-all to these models '
                        f'(default: all of {list(MODELS)})')

    p.add_argument('--model', choices=list(MODELS), default='lcdm',
                   help='Cosmological model (default: lcdm)')
    p.add_argument('--dataset',
                   choices=['CC+BAO', 'Pantheon', 'Pantheon+',
                            'CC+BAO+Pantheon', 'CC+BAO+Pantheon+',
                            'CC', 'CC+Pantheon+'],   # last two: legacy aliases
                   default='CC+BAO',
                   help='Observational dataset (default: CC+BAO). '
                        'Pantheon = 2018 diagonal; Pantheon+ = 2022 full '
                        'covariance. CC / CC+Pantheon+ are accepted as legacy '
                        'aliases of CC+BAO / CC+BAO+Pantheon.')
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
    p.add_argument('--no-csv', action='store_true',
                   help='Skip writing resultados_config.csv')
    p.add_argument('--gpu', action='store_true',
                   help='Use the GPU for Aer simulation if available '
                        '(qiskit-aer-gpu + CUDA). Falls back to CPU otherwise.')
    p.add_argument('--profile', action='store_true',
                   help='Profile peak CPU/GPU memory, wall time and GPU-hours, '
                        'and save a resource_usage_*.png figure.')
    p.add_argument('--max-qubits', type=int, default=18, metavar='N',
                   help='Memory safety cap on the total statevector qubits '
                        '(nqpp*d). Default 18 (~3.5 GB, safe on a laptop). '
                        'Raise it on a supercomputer with more RAM, e.g. '
                        '--max-qubits 22 (~56 GB) or 24 (~224 GB). Each +1 '
                        'qubit roughly quadruples auxiliary memory.')
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


def run_sweep_all(models, dataset, prior, steps, qvmc_iter, nqpp, chains,
                  shots, seed, master_dir, logger, log_every,
                  no_csv=False, no_plot=False):
    """Run the full quantumness benchmark for EVERY requested model in one go.

    This is the HPC "launch once, get everything" mode. For each model it runs
    the canonical per-method ladders (QMCMC: classical -> +proposal ->
    +acceptance; QVMC: classical -> +sampling -> +training -> +normalization),
    writes that model's figures + per-model CSV into its OWN subfolder of the
    master run directory, and appends every row to ONE cumulative CSV at the
    master level so all models sit in a single comparison table.

    Robustness: each model runs inside a try/except, so if one model fails
    (e.g. a numerical edge case) the sweep logs the error and CONTINUES with the
    remaining models instead of aborting the whole batch — essential when a job
    has been queued for hours on a cluster.

    Args:
        models: list of model keys to sweep (subset of MODELS).
        dataset, prior: shared observational setup for all models.
        steps, qvmc_iter, nqpp, chains, shots, seed: shared hyper-parameters.
        master_dir: the master run folder; per-model subfolders are created in it.
        logger, log_every: logging target and cadence.
        no_csv, no_plot: pass-throughs to skip CSV / figures.

    Returns:
        dict mapping model key -> 'ok' or the error string, for the summary.
    """
    say = logger.info if logger else print
    cumulative_master = os.path.join(master_dir, "resultados_TODOS_los_modelos.csv")
    status = {}
    t_start = time.time()

    say("=" * 70)
    say(f"SWEEP-ALL — {len(models)} model(s): {', '.join(models)}")
    say(f"  dataset={dataset} | prior={prior} | steps={steps} "
        f"| qvmc_iter={qvmc_iter} | nqpp={nqpp}")
    say(f"  master folder: {master_dir}/")
    say("=" * 70)

    for i, model_name in enumerate(models, 1):
        say("")
        say(f"[{i}/{len(models)}] ===== MODEL: {model_name} "
            f"({MODELS[model_name].label}) =====")
        # Per-model subfolder INSIDE the master folder, so each model's figures
        # land together and nothing overwrites across models.
        model_dir = os.path.join(master_dir, f"model_{model_name}")
        os.makedirs(model_dir, exist_ok=True)
        try:
            _reseed(seed)
            post = Posterior(MODELS[model_name], dataset, prior)
            # Two CSV destinations: the per-model file in its subfolder, plus
            # the single cumulative master file shared by every model.
            csv_paths = None
            if not no_csv:
                csv_paths = [os.path.join(model_dir, "resultados_config.csv"),
                             cumulative_master]
            run_quantumness_ladder(
                post, n_steps_mcmc=steps, max_iter_qvmc=qvmc_iter, nqpp=nqpp,
                outdir=model_dir, seed=seed, logger=logger,
                log_every=log_every, n_chains_mcmc=chains, n_shots=shots,
                csv_paths=csv_paths, dataset_label=dataset, prior_type=prior)
            status[model_name] = 'ok'
            say(f"[{i}/{len(models)}] {model_name}: DONE -> {model_dir}/")
        except Exception as exc:                      # keep the batch alive
            status[model_name] = f"FAILED: {exc}"
            say(f"[{i}/{len(models)}] {model_name}: FAILED — {exc}")
            import traceback
            (logger.error if logger else print)(traceback.format_exc())

    elapsed = time.time() - t_start
    say("")
    say("=" * 70)
    say(f"SWEEP-ALL finished in {elapsed/60:.1f} min")
    for m in models:
        say(f"  {m:8s} : {status[m]}")
    if not no_csv:
        say(f"  Combined table: {cumulative_master}")
    say("=" * 70)
    return status


def _validate_args(args) -> None:
    """Reject out-of-range numeric arguments with a clear message.

    [ROBUSTNESS] argparse validates choices/types but not the *range* of
    integers, so values like --steps -5, --nqpp 0 or a huge --nqpp used to
    crash deep inside NumPy/Qiskit with cryptic errors. On an HPC queue a typo
    like that wastes a long job, so we fail fast here with an actionable note.

    The qubit ceiling guards the EXPONENTIAL statevector cost: the grid is
    2^(nqpp*d) states and the likelihood builds (2^(nqpp*d), N_data)
    intermediates, so memory grows ~4x per added qubit. The cap is
    --max-qubits (default 18 ≈ 3.5 GB, laptop-safe; raise on a supercomputer).
    """
    errs = []
    pos = {'steps': 'steps', 'qvmc_iter': 'qvmc-iter', 'nqpp': 'nqpp',
           'chains': 'chains', 'shots': 'shots'}
    for attr, flag in pos.items():
        v = getattr(args, attr, None)
        if v is not None and v < 1:
            errs.append(f"--{flag} must be >= 1 (got {v})")
    if getattr(args, 'seed', 0) < 0:
        errs.append(f"--seed must be >= 0 (got {args.seed})")

    max_q = getattr(args, 'max_qubits', 18)
    nqpp = getattr(args, 'nqpp', None)
    if nqpp is not None and nqpp >= 1:
        models = (args.sweep_models if getattr(args, 'sweep_all', False)
                  and args.sweep_models else
                  (list(MODELS) if getattr(args, 'sweep_all', False)
                   else [args.model]))
        max_d = max(MODELS[m].n_params for m in models)
        total_q = nqpp * max_d
        if total_q > max_q:
            approx_gb = 2 ** total_q * 1660 * 8 / 1e9
            errs.append(
                f"--nqpp {nqpp} with a {max_d}-parameter model needs a "
                f"2^{total_q}-state grid (~{approx_gb:.1f} GB worst case), "
                f"above the --max-qubits {max_q} cap. Either lower nqpp to "
                f"<= {max_q // max_d}, or raise --max-qubits if your machine "
                f"has the RAM.")
    if errs:
        sys.stderr.write("Argument error(s):\n  " + "\n  ".join(errs) + "\n")
        sys.exit(2)


def main():
    """Entry point: interactive without arguments, CLI with logging if any."""
    parser = build_parser()
    args = parser.parse_args()
    if len(sys.argv) > 1:
        _validate_args(args)

    # --sanity-check short-circuits everything: routing + correctness only.
    if getattr(args, 'sanity_check', False):
        sanity_check_routing(model_name=args.model, nqpp=min(args.nqpp, 2))
        return

    cli_mode = len(sys.argv) > 1 and not args.interactive

    # [FIX] In CLI/batch mode switch to the headless 'Agg' backend explicitly
    # (it is no longer forced at import time). Interactive runs keep their GUI.
    if cli_mode:
        set_headless_backend()

    # [GPU] Resolve the simulation device once and publish it module-wide so
    # every AerSimulator (built via _sim) uses it. --gpu requests the GPU; if
    # none is available we fall back to CPU and say so.
    global USE_GPU
    want_gpu = bool(getattr(args, 'gpu', False))
    USE_GPU = want_gpu
    do_profile = bool(getattr(args, 'profile', False))

    _reseed(args.seed)

    # ── SWEEP-ALL: run the benchmark for every model in one master folder ──
    if getattr(args, 'sweep_all', False):
        # [FIX] Force the headless backend here: --sweep-all returns before the
        # normal cli_mode branch that would otherwise call this, and on an HPC
        # compute node (no display) Matplotlib must not try to open a GUI.
        set_headless_backend()
        device = resolve_device(USE_GPU)
        sweep_models = args.sweep_models or list(MODELS)
        # One master folder for the whole sweep; per-model subfolders inside.
        if args.outdir == 'results':
            master_dir = make_run_dir('results', tag='sweep_all')
        else:
            master_dir = args.outdir
            os.makedirs(master_dir, exist_ok=True)
        log_file = args.log_file or os.path.join(
            master_dir, f"sweep_all_{time.strftime('%Y%m%d_%H%M%S')}.log")
        logger = setup_logger(log_file)
        print(f"  SWEEP-ALL mode: master folder {master_dir}/  | log {log_file}")
        logger.info("Simulation device: %s%s", device,
                    f"  | GPU available: {gpu_available()}" if USE_GPU else "")
        if USE_GPU and device == 'CPU':
            logger.info("  ⚠  --gpu requested but no Aer GPU device available; "
                        "running on CPU.")
        profiler = None
        if do_profile:
            import cosmo_profiling as _prof
            profiler = _prof.ResourceProfiler(
                tag=f"sweep_{'gpu' if device == 'GPU' else 'cpu'}",
                device=device, interval=0.5)
            profiler.start()
        run_sweep_all(
            sweep_models, args.dataset, args.prior, args.steps, args.qvmc_iter,
            args.nqpp, args.chains, args.shots, args.seed, master_dir, logger,
            args.log_every, no_csv=args.no_csv, no_plot=args.no_plot)
        _finish_profile(profiler, master_dir, logger.info,
                        f"sweep-all | {len(sweep_models)} models | "
                        f"steps={args.steps} iters={args.qvmc_iter}")
        return

    # ── mode selection ───────────────────────────────────────────────────
    if cli_mode:
        model_name, dataset, prior = args.model, args.dataset, args.prior
        steps, qvmc_iter, nqpp = args.steps, args.qvmc_iter, args.nqpp
        benchmark = args.benchmark
        if args.config:
            try:
                cfg = json.loads(args.config)
            except json.JSONDecodeError as e:
                sys.stderr.write(f"--config is not valid JSON: {e}\n"
                                 f"  Example: --config '{{\"proposal\": true, "
                                 f"\"acceptance\": false}}'\n")
                sys.exit(2)
            if not isinstance(cfg, dict):
                sys.stderr.write("--config must be a JSON object (dict), "
                                 f"got {type(cfg).__name__}\n")
                sys.exit(2)
            cfg.setdefault('label', f"{compute_quantumness(cfg):.0f}% — JSON")
        elif args.preset is not None:
            cfg = dict(PRESETS[args.preset])
        else:
            cfg = dict(PRESETS[20])
    else:
        sel = interactive_menu()
        model_name, dataset, prior = sel['model'], sel['dataset'], sel['prior']
        steps, qvmc_iter, nqpp = sel['steps'], sel['qvmc_iter'], sel['nqpp']
        cfg, benchmark = sel['config'], sel['benchmark']
        # GPU / profiling can also be chosen from the menu.
        USE_GPU = sel.get('use_gpu', want_gpu)
        do_profile = sel.get('profile', do_profile)

    # [FIX] Create the timestamped run directory AFTER the model is known.
    # Previously it was built from args.model (the CLI default 'lcdm') before
    # the interactive menu ran, so an interactive wCDM run wrote into a folder
    # named '..._lcdm' — and in some paths the folder was not yet created when
    # the first figure was saved, raising FileNotFoundError. Now the folder is
    # created once, here, using the REAL model name, and always exists before
    # any output is written.
    if args.outdir == 'results':
        args.outdir = make_run_dir('results', tag=model_name)
    else:
        os.makedirs(args.outdir, exist_ok=True)

    # Logger is attached now that the output directory exists.
    if cli_mode:
        log_file = args.log_file or os.path.join(
            args.outdir, f"qcosmo_{model_name}_{time.strftime('%Y%m%d_%H%M%S')}.log")
        logger = setup_logger(log_file)
        print(f"  CLI mode: results in {args.outdir}/")
    else:
        logger = None

    model = MODELS[model_name]
    post = Posterior(model, dataset, prior)
    say = logger.info if logger else print
    say(f"Model: {model.label} | params: {model.param_names} | "
        f"dataset: {dataset} ({post.n_data} pts) | prior: {prior}")

    # [GPU] Report the device actually in use (after fallback resolution).
    device = resolve_device(USE_GPU)
    if USE_GPU and device == 'CPU':
        say("  ⚠  --gpu requested but no Aer GPU device is available "
            "(need qiskit-aer-gpu + CUDA). Running on CPU.")
    say(f"  Simulation device: {device}"
        + (f"  | GPU available: {gpu_available()}" if USE_GPU else ""))

    # [PROFILE] Optionally wrap the whole run in the resource profiler.
    profiler = None
    if do_profile:
        import cosmo_profiling as _prof
        profiler = _prof.ResourceProfiler(
            tag=f"{model_name}_{'gpu' if device == 'GPU' else 'cpu'}",
            device=device, interval=0.25)
        profiler.start()

    # The benchmark IS the per-method quantumness ladder (the single,
    # canonical quantumness scale). Test Run routes here too, at small sizes.
    if benchmark:
        csv_paths = None
        if not args.no_csv:
            csv_paths = [os.path.join(args.outdir, "resultados_config.csv"),
                         "resultados_config.csv"]
        run_quantumness_ladder(post, n_steps_mcmc=steps,
                               max_iter_qvmc=qvmc_iter, nqpp=nqpp,
                               outdir=args.outdir, seed=args.seed,
                               logger=logger, log_every=args.log_every,
                               csv_paths=csv_paths, dataset_label=dataset,
                               prior_type=prior)
        _finish_profile(profiler, args.outdir, say,
                        f"{model.label} benchmark | steps={steps} "
                        f"iters={qvmc_iter} nqpp={nqpp}")
        return

    # ── single configuration + MANDATORY classical baseline ──────────────
    comp = run_comparison(post, cfg, seed=args.seed, logger=logger,
                          n_steps_mcmc=steps, max_iter_qvmc=qvmc_iter,
                          n_chains_mcmc=args.chains, nqpp=nqpp,
                          n_shots=args.shots, n_burn=args.burn,
                          log_every=args.log_every, verbose=True)
    res_q, res_c = comp['quantum'], comp['classical']

    # ── write results to CSV (per-run copy + cumulative table) ────────────
    # [FIX] This is what was missing: the modular module now writes
    # resultados_config.csv (all parameters, all methods), both inside the run
    # folder and as a cumulative file in the working directory.
    if not args.no_csv:
        run_csv = os.path.join(args.outdir, "resultados_config.csv")
        nqpp_tag = nqpp if res_q is not None else "—"
        side_rows = []
        if res_q is not None:
            side_rows.append((res_q['mcmc'],
                              f"QMCMC {quantumness_qmcmc(cfg):.0f}%", True,
                              nqpp_tag))
            side_rows.append((res_q['qvmc'],
                              f"QVMC {quantumness_qvmc(cfg):.0f}%", False,
                              nqpp_tag))
        side_rows.append((res_c['mcmc'], "Classical MCMC", True, "—"))
        side_rows.append((res_c['qvmc'], "Classical VI", False, "—"))
        write_run_and_cumulative(side_rows, model, run_csv,
                                 "resultados_config.csv", dataset, prior)
        say(f"Results written to {run_csv} (+ cumulative resultados_config.csv)")

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

    _finish_profile(profiler, args.outdir, say,
                    f"{model.label} | {dataset} | {cfg.get('label', '')}")


def _finish_profile(profiler, outdir, say, title_extra=''):
    """Stop a ResourceProfiler (if any), log the summary and save the figure."""
    if profiler is None:
        return
    import cosmo_profiling as _prof
    result = profiler.stop()
    say(_prof.summarize(result))
    path = _prof.ResourceProfiler.plot(result, outdir, title_extra=title_extra)
    if path:
        say(f"  Resource-usage figure: {path}")
    # Also drop a small JSON next to it for the provenance record.
    try:
        import json as _json
        with open(os.path.join(outdir, f"profile_{result.tag}.json"), 'w') as fh:
            _json.dump(result.as_row(), fh, indent=2)
    except Exception:
        pass


if __name__ == "__main__":
    main()
