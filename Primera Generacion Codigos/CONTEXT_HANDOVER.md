# CONTEXT HANDOVER DOCUMENT — Quantum Algorithms for Cosmology

> **Purpose of this document.** This is the initial context for a new chat
> instance continuing work on a hybrid classical/quantum Bayesian inference
> codebase for cosmological parameter estimation (undergraduate thesis,
> Engineering Physics, IBERO). It captures the project goal, the physics,
> the critical mathematical decisions and bug fixes, the software
> architecture, the visualization logic, and the **final source code** of
> all scripts. Read sections 1–5 to understand the state; section 6 is the
> canonical code to load into the new project.
>
> **Working style to preserve:** the user (Andrés) works fluidly in Spanish
> and English, asks "why" before "how", expects rigorous physical and
> statistical justification, prefers complete runnable files, and runs the
> code himself reporting back errors/plots. Push back with evidence when a
> requested change is suboptimal (this has happened and was valued).

---

## 1. PROJECT SUMMARY

**Main goal.** Estimate cosmological parameters (Ωm, H0, and model-specific
extras such as w, w0, wa, Δ) by fitting late-time expansion data with
Bayesian inference, and **compare classical vs quantum sampling algorithms
head-to-head** to show whether the quantum methods reproduce the trusted
classical results.

**Data.**
- **Cosmic Chronometers (CC):** 51 H(z) points (embedded in code, with a
  file fallback).
- **Pantheon+ SNe Ia:** 1048 supernovae, with **analytic marginalization
  over the absolute magnitude M_abs** following Goliath et al. (2001)
  (χ²_eff = A − B²/C). File `pantheon_full_parameters.txt`
  (`name zcmb zhel dz mb dmb`).
- Datasets selectable as `CC`, `Pantheon+`, or `CC+Pantheon+`.

**Priors.** `flat` (box) or `gaussian` (Planck 2018: H0 = 67.66 ± 0.42,
Ωm = 0.3111 ± 0.0056, applied only to (Ωm, H0); flat on extras).

**The four inference algorithms.**
| Algorithm | Family | What it does |
|---|---|---|
| **Classical MCMC** | sampling | Hand-written vectorized Metropolis-Hastings (NOT emcee). |
| **QMCMC** | sampling | Same MH loop, but the proposal step (and optionally the accept/reject) is produced by a quantum circuit (Sarracino et al. 2025 proposal). |
| **Classical VI** | variational | Discretized posterior on a grid, |ψ(φ)|² model, KL(Q‖P) minimized with COBYLA. |
| **QVMC** | variational | Same grid/KL objective, but the variational state and/or its training/sampling/normalization run on a quantum circuit. |

The quantum and classical members of each family are **the same code with
components toggled** classical↔quantum (see §4).

---

## 2. PHYSICS MODELS — STATE

### Physics ↔ sampling separation
All physics lives in **`cosmo_core.py`**, the single shared module. The
samplers never touch cosmology directly — they only call a `Posterior`
object. Key elements:

- **`CosmoModel`** (dataclass): `name`, `label`, `param_names` (convention
  `[Om, H0, ...extras]`), `param_latex`, `bounds`, `sample_box`,
  `fiducial`, and the Friedmann function `E2(z, θ)`. A registry dict
  `MODELS` holds all models. **Adding a model (e.g. Variable Curvature)
  = adding one dict entry**; no sampler changes are needed.
- **`Posterior(model, dataset, prior_type, ...)`**: the ONLY contact point
  between physics and samplers. Provides `log_prob(θ)` and the vectorized
  `log_prob_batch(thetas)` (broadcasting over a batch, ~4000 evals
  CC+Pantheon+ in ≈1 s) which both samplers use.
- **Model-agnostic Pantheon+ [CRITICAL FIX].** The χ²_Pantheon used to rely
  on a precomputed lookup grid in Ωm (valid only for flat ΛCDM). It now
  integrates the luminosity distance with a **vectorized
  `cumulative_trapezoid` over a fine z-grid**, valid for ANY E²(z; θ) —
  this is what enabled wCDM/CPL/etc. Verified bit-exact vs a loop.
- **Statistics:** FFT autocorrelation τ (O(N log N)), ESS (chains +
  Kish weights), Gelman-Rubin (max over parameters), `fit_statistics`
  (χ², χ²_red, AIC, BIC with Nelder-Mead refinement), dual logger.

### Implemented and functional models (in `MODELS`)
| key | model | params | notes |
|---|---|---|---|
| `lcdm` | Flat ΛCDM | Om, H0 | baseline, fully validated |
| `wcdm` | wCDM | Om, H0, w | w ∈ (−2, −0.3); reduces to ΛCDM at w=−1 (verified) |
| `cpl` | CPL (w0–wa) | Om, H0, w0, wa | f_DE=(1+z)^{3(1+w0+wa)}·exp(−3wa·z/(1+z)) |
| `pede` | PEDE | Om, H0 | f_DE = 1 − tanh(log10(1+z)) |
| `gede` | GEDE | Om, H0, Δ | uses z_t from matter–DE equality |

`lcdm`, `wcdm`, `cpl` are the primary validated targets; `pede`, `gede`
are implemented and selectable. (A Variable-Curvature "VC" model is a
planned future addition, deferred until the ΛCDM machinery is fully solid —
the original VC thesis had sign-convention errors in the
energy-momentum tensor, a non-standard inflated likelihood, and a
physically inconsistent positive Ok1; the revised approach prefers a
perturbative curvature treatment guaranteeing E²(0)=1.)

---

## 3. MATHEMATICAL DECISIONS AND CRITICAL FIXES  ⚠️ MOST IMPORTANT

### 3.1 QMCMC quantum-acceptance bug (inverted → fixed to Metropolis)
**Symptom.** Chains using the quantum acceptance (QMCMC rungs with the
acceptance component ON) drifted toward the prior-box edges
(Ωm ≈ 0.37, H0 ≈ 77 instead of the data optimum ≈ 0.26, 70).

**Root cause.** The old `hadamard_accept_log` built a CRY/Hadamard-test
circuit and read `P(ancilla=0)`. That quantity turned out to be a
**monotonically DECREASING** function of Δ = lp_prop − lp_cur: it accepted
*worse* proposals with high probability and rejected *better* ones. A
regression table made it obvious (Δ=+5 gave P≈0.54, Δ=−5 gave P≈0.998).

**Fix.** Encode the standard **Metropolis** acceptance A = min(1, e^Δ) as
the |0⟩-amplitude of a single-qubit RY rotation: θ = 2·arccos(√A) gives
P(|0⟩) = cos²(θ/2) = A exactly, read from the Aer statevector (still
genuinely "quantum", but now correct). Verified monotonic increasing and
matching Metropolis. Post-fix, every quantumness level agrees with the
classical baseline (Ωm ≈ 0.26).

**Why Metropolis and not Barker.** Kept Metropolis deliberately so the
quantum acceptance computes the *same number* as the classical baseline —
the goal is to demonstrate the quantum method **replicates** the classical
one. A consequence: the quantum-acceptance step is numerically identical to
classical acceptance, so for QMCMC only the **proposal** changes the
statistics (the acceptance is a faithful reproduction). The QPU script's
`metropolis_log_accept` was also aligned to Metropolis (`min(0, Δ)`) for
cross-pipeline consistency (it previously used Barker log σ(Δ)).

**Proposal calibration.** The quantum displacement re[:d]·sign(im[:d]) is
zero-mean (so symmetric-proposal Metropolis stays valid) but had std ≈ 0.35
— ~3× smaller than the classical N(0,1) the step scale was tuned for,
pushing acceptance to ≈ 0.80 (slow mixing). Each block is now normalized to
**unit std** → acceptance ≈ 0.5.

### 3.2 QVMC optimizer decision (SGD + lr decay, NOT Adam, NOT COBYLA)
**Observed.** The "stalled KL" symptom was the **classical COBYLA** baseline
(42 ansatz angles at nqpp=3, gradient-free → barely moves in the iteration
budget), NOT the quantum trainer. Quantum **parameter-shift** drives KL from
~8 down to ~0.34.

**Evidence-based optimizer choice** (benchmarked on lcdm, nqpp=3, 42 angles):
| Optimizer | min KL | tail behavior |
|---|---|---|
| fixed-lr SGD | 0.340 | **creeps back up** (+0.019) near the minimum |
| Adam (lr 0.05) | 0.603 | very stable (+0.0004) but **plateaus high** |
| **SGD + lr decay** | **0.344** | **flat tail (+0.007) — best of both** |

**Decision.** Adam — although the natural suggestion — settles into a wider,
higher-KL basin on this landscape, so it was **rejected**. The trainer uses
**parameter-shift SGD with a 1/(1+γ·i) learning-rate decay** (γ=0.02): it
keeps the low minimum of plain SGD while removing the late-iteration
creep-up. `lr_train` is exposed for tuning. No severe barren plateau at
nqpp ≤ 3 (6 qubits); SPSA would be preferable at larger grids. The
parameter-shift gradient (2·n_φ shifted circuits) is evaluated as ONE
batched Aer job per iteration.

### 3.3 QVMC adaptive grid (Option B — resolving a smooth posterior)
**Problem.** QVMC represents the posterior as a PMF on a discrete 2^nqpp
grid. Spanning the full wide `sample_box`, the cosmological posterior
(σ_Ωm ≈ 0.018) is *narrower than one grid cell*, so it collapses to a spike
and can never look Gaussian — regardless of iterations.

**Fix.** `estimate_grid_window` runs a fast classical pre-fit (short
vectorized Metropolis) to find the mode and per-parameter σ, then builds the
grid on a **zoomed window** [mode − k·σ, mode + k·σ] clipped to physical
bounds. The half-width **k scales with the grid size**:
k ≈ (2^nqpp − 1)/6, clipped to [2, 5] — a small grid zooms in tightly, a
larger grid widens to resolve the tails. The window is computed **once and
shared** by every QVMC rung and the classical-VI baseline (fair comparison).
Effect (occupied cells): nqpp 3/4/5 → ~13/40/49 (was ~3 with the full box),
and the discretized target becomes a clean bell curve. Any residual
lumpiness in the QVMC *samples* is then ansatz/training (more iters, more
layers), not the grid.

### 3.4 Other correctness/performance fixes (already applied)
- **Vectorized QMCMC:** the kernel scores all chains per step with a single
  `log_prob_batch` call (the dominant cost) instead of one `post(θ)` per
  chain. 6 chains × 2000 steps in ~0.1 s. Quantum-acceptance path keeps a
  short per-chain loop (the circuit is per-pair).
- **Custom MH, not emcee — deliberate:** owning every line of the transition
  kernel is required to swap individual components classical↔quantum and to
  guarantee the classical baseline and quantum run share the exact same
  transition structure, step scale and RNG stream. emcee's affine-invariant
  move is a black box that cannot host a quantum proposal/acceptance.
- KL renormalization (no negative KL), FFT autocorrelation, transpile-once
  for circuits, parameter-shift as a single batched job.

---

## 4. SOFTWARE ARCHITECTURE AND CLI

### Three scripts + shared core
1. **`cosmo_core.py`** — shared physics/data/statistics (see §2).
2. **`cosmo_modular_quantum.py`** — the Aer **simulator** pipeline:
   interactive menu (default, no args) + non-interactive CLI; the
   quantumness benchmark; all plotting; the sanity-check harness.
3. **`qpu_cosmo_samplers.py`** — **real IBM Quantum hardware** via
   `qiskit-ibm-runtime` (SamplerV2 + Batch/Session). **Quantum-only: runs
   NO classical method** (QPU time is scarce; classical baselines belong in
   the simulator). Uses SPSA for QVMC-QPU (2 evals/iter = 1 job, vs
   2·n_φ for parameter-shift), ⟨Z_q⟩ proposals from measured counts,
   Metropolis acceptance on CPU (sequential), Dynamical Decoupling XY4 +
   Pauli twirling, a `TimingEstimator` (queue-dominated), `--dry-run`
   (validates without an IBM account), `--max-jobs` budget guard.

### Quantumness components and the canonical "ladder" scale
Five switchable components, grouped by which sampler reads them:

| Sampler | Components (weights) |
|---|---|
| **QMCMC** | proposal (20), acceptance (25) |
| **QVMC** | sampling (25), training (20), normalization (10) |

**The single canonical scale = per-method ladders.** Because the two
samplers read different components, a single global % was confusing (raising
it often flipped a switch a given sampler ignores → identical output). The
**benchmark now sweeps each sampler along ITS OWN monotonic axis**, each %
counting only that sampler's component weights:

```
QMCMC ladder:  0%  →  44% (+proposal)   →  100% (+acceptance)
QVMC  ladder:  0%  →  46% (+sampling)    →  82% (+training)  →  100% (+normalization)
```

**Expected coincidences (the replication result, by design):**
- QMCMC 44% ≡ 100%  (quantum Metropolis acceptance reproduces classical)
- QVMC 0% ≈ 46%      (quantum sampling = same trained state, only shot noise)
- QVMC 82% ≡ 100%    (quantum normalization is a faithful renorm)
The steps that genuinely change the answer are the **proposal** (QMCMC) and
the **training** (QVMC).

`compute_quantumness` (global, for single-config labels) and the per-method
`quantumness_qmcmc` / `quantumness_qvmc` both exist; the benchmark uses the
per-method ones. Global presets `PRESETS = {0,20,45,70,90,100}` remain only
as a convenient shorthand for single custom configs.

### CLI (`argparse`) and cluster logging
- **Run modes (mutually exclusive):** `--interactive` (or no args → menu),
  `--preset N`, `--benchmark` (= the per-method ladders, the canonical
  scale), `--config JSON` (custom component dict). `--ladder` was REMOVED
  (folded into `--benchmark`). `--sanity-check` runs the routing/correctness
  self-check and exits.
- **Interactive menu** (default): asks run mode (Single / Benchmark /
  Quick Test Run), then model, dataset (auto-detects Pantheon+), prior, and
  sizes. **Test Run** = the ladders at small fixed sizes (steps 200, iters
  40, nqpp 2). A `sys.stdin.isatty()` guard falls back to a default preset
  for non-interactive (SLURM) launches.
- **Other args:** `--model`, `--dataset`, `--prior`, `--steps`,
  `--qvmc-iter`, `--nqpp`, `--chains`, `--seed`, `--outdir`, `--log-file`,
  `--log-every`, `--no-plot`.
- **Logging for clusters:** a dual logger — a DEBUG file log
  (`resultados/qcosmo_<model>_<timestamp>.log` by default) plus a minimal
  console stream. Progress is logged every `--log-every` steps (acceptance,
  R̂, KL, parameter means). Matplotlib uses the Agg backend in CLI mode.
  The QPU script writes incremental JSON results and a timing projection.

### Sanity-check harness (`--sanity-check`)
Prints (1) an **acceptance regression test** (must be monotonic increasing
and match Metropolis min(1,e^Δ) — the guard against the inverted-acceptance
bug), (2) the quantum-proposal statistics (zero-mean, unit-std), and (3) a
**per-preset engine map** (⚛ Qiskit/Aer vs 🖥 NumPy/SciPy for every
component) plus a live routing trace inside the hot loops.

---

## 5. VISUALIZATION LOGIC

All figures use **corner.py** for corner plots and a fixed contrasting-colour
convention: **classical = blue (#1f77b4)**, **QMCMC = red (#d62728)**,
**QVMC = orange (#ff7f0e)**, and **classical-VI = teal (#17becf)** when shown
alongside classical-MCMC in the same panel. Planck fiducials are dashed black
lines. Every QVMC figure prints `nqpp`; QMCMC/QVMC figures print
steps/iterations in the title.

### Benchmark (per-method ladder) figures
- **`corner_ladder_qmcmc_*` / `corner_ladder_qvmc_*`** — family overlay: the
  classical 0% baseline + every quantum rung of that method overlaid on one
  corner (warm colormap over the rungs). The QVMC corner is "cell-jittered
  for display" since it is a discrete-grid PMF.
- **`corner_ladder_1to1_*_q{pct}`** — one per rung: that rung overlaid with
  the classical baseline(s) so you can read which quantumness level best
  matches the classical distribution.
- **`ladder_rhat_qmcmc_*`** — Gelman-Rubin R̂−1 vs steps, all QMCMC rungs.
- **`ladder_kl_qvmc_*`** — KL vs iteration, all QVMC rungs.
- **`ladder_trends_*`** — 2×4 panel: Ωm, H0 (with error bars), runtime,
  reduced χ², AIC, BIC, ESS, and acceptance(QMCMC)/KL(QVMC) — all **vs
  quantumness %**. This is the "how parameters and time change with
  quantumness" summary.
- **`ladder_summary_*`** — table of all final numbers per rung.

### Single-configuration figures (one config + its forced classical baseline)
- **`corner_mcmc_*` / `corner_qvmc_*`** — classical vs quantum overlaid (2D
  contours + 1D marginals, shared ranges).
- **`marginals_*`** — 1D marginals per parameter, both families, plus H(z)
  predictive.
- **`kl_overlay_*`** (Classical VI vs QVMC), **`rhat_overlay_*`**
  (Classical MCMC vs QMCMC), **`traces_*`** (parameter traces).

When a quantum component is a faithful reproduction, its overlaid corner sits
exactly on the classical one — that visual identity IS the replication
evidence, not a bug.

---

## 6. FINAL SOURCE CODE

The canonical, current source of all files follows in separate code blocks.
These are also attached as individual files for direct upload to the new
project. Entry points: run `python cosmo_modular_quantum.py` (menu) or
`--benchmark` / `--sanity-check`; `python qpu_cosmo_samplers.py --dry-run`
for the hardware path. `requirements.txt`: numpy, scipy, matplotlib, corner,
qiskit, qiskit-aer, qiskit-ibm-runtime, tqdm.


### 6.1 `requirements.txt`

```text
numpy>=1.24
scipy>=1.10
matplotlib>=3.7
corner>=2.2               # overlaid corner plots (2D contours + 1D marginals)
qiskit>=1.2
qiskit-aer>=0.15          # cosmo_modular_quantum.py only (simulator)
qiskit-ibm-runtime>=0.30  # qpu_cosmo_samplers.py only (real hardware)
tqdm>=4.65
```

### 6.2 `cosmo_core.py`

```python
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
```

### 6.3 `cosmo_modular_quantum.py`

```python
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


# ── Per-method quantumness axes (Option A) ───────────────────────────────────
# The single global quantumness % bundles two INDEPENDENT samplers, so it is
# not monotonic for either one. These per-method scales fix that: each one
# counts only the components that the corresponding sampler actually reads,
# and each ladder rung adds exactly ONE quantum component, so the axis is
# monotonic and every step has a well-defined meaning.
#
#   QMCMC reads  : proposal (w=20), acceptance (w=25)            total 45
#   QVMC  reads  : sampling (w=25), training (w=20), norm (w=10) total 55
#
# Ladder order = the order in which components are switched on. We keep the
# historical progression (proposal→acceptance for QMCMC; sampling→training→
# normalization for QVMC) so the levels line up with the thesis presets.
_QMCMC_ORDER = ['proposal', 'acceptance']
_QVMC_ORDER = ['sampling', 'training', 'normalization']
_W = {'proposal': 20, 'acceptance': 25, 'training': 20, 'sampling': 25,
      'normalization': 10}


def quantumness_qmcmc(config: dict) -> float:
    """QMCMC-only quantumness %: active (proposal, acceptance) weights."""
    tot = sum(_W[c] for c in _QMCMC_ORDER)
    return round(100.0 * sum(_W[c] for c in _QMCMC_ORDER
                             if config.get(c, False)) / tot, 1)


def quantumness_qvmc(config: dict) -> float:
    """QVMC-only quantumness %: active (sampling, training, norm) weights."""
    tot = sum(_W[c] for c in _QVMC_ORDER)
    return round(100.0 * sum(_W[c] for c in _QVMC_ORDER
                             if config.get(c, False)) / tot, 1)


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
        _HAD['sim'] = AerSimulator(method='statevector')
        _HAD['qc_t'] = transpile(qc, _HAD['sim'])
        _HAD['par'] = par
    theta = 2.0 * np.arccos(np.sqrt(A))
    bound = _HAD['qc_t'].assign_parameters({_HAD['par'][0]: theta})
    sv = np.asarray(_HAD['sim'].run(bound).result().get_statevector())
    prob_zero = float(np.abs(sv[0])**2)   # P(|0>) = A = min(1,e^Δ)
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
                    'RY amplitude encoding of min(1,e^Δ) (Metropolis) via Aer statevector')
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


def estimate_grid_window(post: Posterior, sigma_mult: float = 4.0,
                         n_steps: int = 400, n_chains: int = 4) -> List[tuple]:
    """Quick classical pre-fit that positions the QVMC grid (adaptive grid).

    [ADAPTIVE GRID — option b] The QVMC posterior lives on a discrete grid
    of 2^nqpp points per parameter. If that grid spans the full (wide)
    sample_box, the cosmological posterior — which is far narrower than the
    grid spacing — collapses onto one or two cells and can never look like
    a smooth distribution. This helper runs a short vectorized Metropolis
    chain to locate the posterior mode and per-parameter width, then returns
    per-parameter windows [mode − k·σ, mode + k·σ] (k = `sigma_mult`),
    clipped to the model's physical bounds. Building the grid on this
    zoomed window lets a small number of qubits actually RESOLVE the
    posterior, so QVMC (and the classical VI baseline, which shares the
    grid) recover a smooth, roughly Gaussian shape.

    The pre-fit is used ONLY to place/scale the grid (an adaptive-grid
    technique, like adaptive importance sampling); it does not feed the
    QVMC result itself.
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
    mode, sd = flat.mean(0), flat.std(0) + 1e-9
    blo = np.array([b[0] for b in model.bounds])
    bhi = np.array([b[1] for b in model.bounds])
    win_lo = np.clip(mode - sigma_mult * sd, blo, bhi)
    win_hi = np.clip(mode + sigma_mult * sd, blo, bhi)
    for i in range(d):                         # guard against a degenerate window
        if win_hi[i] - win_lo[i] < 1e-6:
            win_lo[i], win_hi[i] = model.sample_box[i]
    return list(zip(win_lo.tolist(), win_hi.tolist()))


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
                           n_shots: int = 2000) -> dict:
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
    fig.savefig(f, dpi=150, bbox_inches='tight')
    fig.savefig(f.replace('.png', '.pdf'), bbox_inches='tight'); plt.close(fig)

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
    fig.savefig(f, dpi=150, bbox_inches='tight')
    fig.savefig(f.replace('.png', '.pdf'), bbox_inches='tight'); plt.close(fig)

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
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))

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

    p0, p1 = model.param_names[0], model.param_names[1]
    _trend(axes[0, 0], lambda r: r['mu'][0], p0, f'{p0} vs quantumness',
           err=lambda r: r['std'][0], fid=model.fiducial[0])
    _trend(axes[0, 1], lambda r: r['mu'][1], p1, f'{p1} vs quantumness',
           err=lambda r: r['std'][1], fid=model.fiducial[1])
    _trend(axes[0, 2], lambda r: r['elapsed'], 'runtime [s]',
           'Runtime vs quantumness')
    _trend(axes[0, 3], lambda r: r['chi2_red'], r'$\chi^2_\nu$',
           'Reduced chi2 vs quantumness', fid=1.0)
    _trend(axes[1, 0], lambda r: r['AIC'], 'AIC', 'AIC vs quantumness')
    _trend(axes[1, 1], lambda r: r['BIC'], 'BIC', 'BIC vs quantumness')
    _trend(axes[1, 2], lambda r: r['ess'], 'ESS',
           'Effective sample size vs quantumness')
    # acceptance is QMCMC-only; KL is QVMC-only → twin axes
    ax = axes[1, 3]
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

    fig.suptitle(f'{model.label} — trends along the per-method quantumness '
                 f'ladders\nsteps={steps} | iters={iters} | nqpp={nqpp}  '
                 '(QMCMC red, QVMC orange)',
                 fontsize=14, fontweight='bold', y=1.0)
    fig.tight_layout()
    f = os.path.join(outdir, f'ladder_trends_{name}.png')
    fig.savefig(f, dpi=150, bbox_inches='tight')
    fig.savefig(f.replace('.png', '.pdf'), bbox_inches='tight'); plt.close(fig)

    # ── summary table (per-method %) ─────────────────────────────────────────
    fig = plt.figure(figsize=(13, 4.5)); ax = fig.add_subplot(111); ax.axis('off')
    rows = []
    for r in qmcmc_runs:
        rows.append([f"QMCMC {r['pct']:.0f}%",
                     f"{r['mu'][0]:.4f}±{r['std'][0]:.4f}",
                     f"{r['mu'][1]:.2f}±{r['std'][1]:.2f}",
                     f"{r['chi2_red']:.3f}", f"{r['AIC']:.1f}",
                     f"{r['elapsed']:.1f}s",
                     f"{r['acceptance']:.3f}", f"{r['ess']:.0f}", "—"])
    for r in qvmc_runs:
        rows.append([f"QVMC {r['pct']:.0f}%",
                     f"{r['mu'][0]:.4f}±{r['std'][0]:.4f}",
                     f"{r['mu'][1]:.2f}±{r['std'][1]:.2f}",
                     f"{r['chi2_red']:.3f}", f"{r['AIC']:.1f}",
                     f"{r['elapsed']:.1f}s",
                     "—", f"{r['ess']:.0f}", f"{r['kl_final']:.4f}"])
    cols = ['method/level', f'{model.param_names[0]}',
            f'{model.param_names[1]}', 'chi2_red', 'AIC', 'time',
            'acc', 'ESS', 'KL']
    tbl = ax.table(cellText=rows, colLabels=cols, loc='center',
                   cellLoc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 1.6)
    fig.suptitle(f'{model.label} — per-method quantumness ladders\n'
                 f'steps={steps} | iters={iters} | nqpp={nqpp}',
                 fontsize=13, fontweight='bold')
    f = os.path.join(outdir, f'ladder_summary_{name}.png')
    fig.savefig(f, dpi=150, bbox_inches='tight'); plt.close(fig)


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
        return {'model': 'lcdm', 'dataset': 'CC', 'prior': 'flat',
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

    # ── TEST RUN: the per-method ladders at small fixed sizes ────────────────
    if mode == 'test':
        print("\n  → Quick TEST RUN: per-method quantumness ladders at small "
              "fixed sizes (steps=200, iters=40, nqpp=2).")
        print("  QMCMC: classical→+proposal→+acceptance   |   "
              "QVMC: classical→+sampling→+training→+norm.")
        print("  Fast end-to-end stability check across both quantum axes.")
        return {'model': model, 'dataset': dataset, 'prior': prior,
                'config': dict(PRESETS[0]), 'steps': 200, 'qvmc_iter': 40,
                'nqpp': 2, 'benchmark': True, 'test_run': True}

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
                      help='Run the full quantumness benchmark: the two '
                           'per-method ladders (QMCMC: proposal->acceptance; '
                           'QVMC: sampling->training->normalization). This is '
                           'the canonical quantumness scale.')
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

    # The benchmark IS the per-method quantumness ladder (the single,
    # canonical quantumness scale). Test Run routes here too, at small sizes.
    if benchmark:
        run_quantumness_ladder(post, n_steps_mcmc=steps,
                               max_iter_qvmc=qvmc_iter, nqpp=nqpp,
                               outdir=args.outdir, seed=args.seed,
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
```

### 6.4 `qpu_cosmo_samplers.py`

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qpu_cosmo_samplers.py — Cosmological inference on real IBM Quantum hardware.
================================================================================

Script EXCLUSIVELY for real QPUs via `qiskit-ibm-runtime` (SamplerV2).
It does NOT use AerSimulator anywhere: every circuit runs on the selected
physical backend (or is planned in --dry-run mode).

The physics (ΛCDM/wCDM/CPL/PEDE/GEDE models, CC and Pantheon+ likelihoods,
priors and statistics) is imported in full from `cosmo_core`, so results
are directly comparable with the simulator pipeline
(`cosmo_modular_quantum.py`).

QUANTUM-ONLY by design  [QONLY]
-------------------------------
This script is STRICTLY for dispatching the quantum algorithms (QMCMC and
QVMC) to Qiskit Runtime primitives. It deliberately runs NO classical
method: no Classical MCMC, no Classical VI. Real QPU time is scarce and
queues are long, so spending hardware sessions (or even local CPU time
inside a hardware-oriented run) on classical baselines would be wasteful
and conceptually out of place here.

Classical baselines and the full classical-vs-quantum overlay study live
in the simulator pipeline `cosmo_modular_quantum.py`, which shares the same
physics module (`cosmo_core`). The recommended workflow is therefore:

  1. Explore the whole quantumness spectrum + classical baselines on the
     simulator (`cosmo_modular_quantum.py --benchmark` / Test Run).
  2. Once a promising configuration is identified, validate it on real
     hardware here with QMCMC / QVMC only.

Figures produced here are single-method quantum diagnostics (corner plot,
KL training curve, Gelman-Rubin R̂) annotated with the run metadata
(steps / iterations / nqpp).

Key design differences vs the simulator version
-----------------------------------------------
1.  [QPU] No statevector. All quantum information comes from measured
    COUNTS. The QVMC KL is estimated over the observed support and the
    QMCMC proposal displacements come from per-qubit <Z_q>.

2.  [QPU][OPT] SPSA instead of parameter-shift for QVMC. Parameter-shift
    needs 2·n_phi evaluations per iteration (~84 circuits for 42 angles);
    SPSA always needs 2, independent of n_phi. On hardware, where each job
    costs seconds-to-minutes of queue, this cuts the per-iteration cost by
    ~40x. Both perturbed parameter sets (phi+, phi-) travel in ONE PUB
    with parameter_values of shape (2, n_phi).

3.  [QPU][OPT] QMCMC proposals pre-generated in blocks: B random angle
    sets -> ONE job with B parameter bindings -> B displacements. The
    Metropolis chain consumes them sequentially from a local queue.

4.  [QPU] The Metropolis acceptance does NOT run on the QPU by default:
    it is inherently sequential (step t depends on t-1), so on hardware it
    would cost one job + one full queue wait PER STEP. We use the
    analytically equivalent expression P(ancilla=0) = e^D/(1+e^D) that the
    Hadamard test implements (Barker rule) — identical stationary
    distribution.

5.  Error suppression via SamplerV2 options: dynamical decoupling (XY4) +
    Pauli twirling of gates and measurement.

6.  `TimingEstimator` times every real job (API overhead + queue +
    execution) and extrapolates the total cost for 100/500/1000 steps.

Usage
-----
    # Planning without spending QPU time (no IBM account needed):
    python qpu_cosmo_samplers.py --model wcdm --method qvmc --iters 30 --dry-run

    # Real run (requires a saved IBM Quantum account or --token):
    python qpu_cosmo_samplers.py --model lcdm --dataset CC --method qvmc \\
        --iters 30 --shots 4096 --least-busy --log-file qpu_run.log

References
----------
* Sarracino et al. (2025) — QMCMC proposal circuit.
* Goliath et al. (2001) — analytic M_abs marginalization (in cosmo_core).
* Spall (1998) — SPSA.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

import matplotlib
matplotlib.use('Agg')                       # headless backend (HPC-safe)
import matplotlib.pyplot as plt
import corner                               # overlaid corner plots

import cosmo_core as core
from cosmo_core import (MODELS, CosmoModel, Posterior, ess_chains,
                        fit_statistics, fmt_theta, gelman_rubin_max,
                        setup_logger)

# Qiskit (circuit construction only; NO Aer)
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

# ── Contrasting color convention shared with cosmo_modular_quantum.py ────────
C_QUANTUM   = '#d62728'   # red    — QMCMC-QPU
C_QUANTUM2  = '#ff7f0e'   # orange — QVMC-QPU

# qiskit-ibm-runtime is imported lazily: --dry-run must work on machines
# with no credentials and no network access.
_RUNTIME_IMPORTED = False


def _import_runtime():
    """Import qiskit-ibm-runtime only when real hardware is requested."""
    global _RUNTIME_IMPORTED, QiskitRuntimeService, SamplerV2, Batch, Session
    global generate_preset_pass_manager
    if _RUNTIME_IMPORTED:
        return
    from qiskit_ibm_runtime import (QiskitRuntimeService, SamplerV2, Batch,
                                    Session)  # noqa: F401
    from qiskit.transpiler.preset_passmanagers import \
        generate_preset_pass_manager  # noqa: F401
    _RUNTIME_IMPORTED = True


RNG = np.random.default_rng()


def _reseed(seed: Optional[int]):
    """Reset the global RNG (here and in cosmo_core) to a fixed seed.

    Called before each quantum method so a run is reproducible given
    --seed (chain initialization and Metropolis uniforms).
    """
    global RNG
    rng = np.random.default_rng(seed)
    RNG = rng
    core.RNG = rng


# =============================================================================
# 1.  TIMING ESTIMATOR
# =============================================================================

@dataclass
class JobRecord:
    """Record of a single job executed on the QPU."""
    n_circuits: int
    shots: int
    t_submit: float      # s — API overhead until the job is accepted
    t_queue: float       # s — queue wait (estimated)
    t_exec: float        # s — reported/measured execution time
    t_total: float       # s — total wall time (submit -> result)


class TimingEstimator:
    """Times real jobs and extrapolates the cost of long runs.

    Every primitive call is wrapped with :meth:`record`, which splits the
    wall time into (API overhead, queue, execution). If the result exposes
    QPU usage metadata it is used; otherwise execution is estimated as
    shots x circuits x t_shot with t_shot ~ 100 us (typical Eagle/Heron)
    and the remaining wall time is attributed to queue + API.
    """

    T_SHOT = 1e-4          # s per shot (Eagle/Heron order of magnitude)
    T_API_DEFAULT = 2.0    # s API overhead per job (REST + serialization)
    T_QUEUE_DEFAULT = 60.0 # s queue per job on the open plan (highly variable)

    def __init__(self):
        self.jobs: List[JobRecord] = []

    # ------------------------------------------------------------------ #
    def record(self, n_circuits: int, shots: int, t_wall: float,
               t_exec_reported: Optional[float] = None) -> JobRecord:
        """Register an executed job and decompose its wall time."""
        t_exec = (t_exec_reported if t_exec_reported is not None
                  else n_circuits * shots * self.T_SHOT)
        t_api = min(self.T_API_DEFAULT, 0.2 * t_wall)
        t_queue = max(0.0, t_wall - t_exec - t_api)
        rec = JobRecord(n_circuits, shots, t_api, t_queue, t_exec, t_wall)
        self.jobs.append(rec)
        return rec

    # ------------------------------------------------------------------ #
    @property
    def mean_wall(self) -> float:
        """Mean wall time per job (s)."""
        if not self.jobs:
            return self.T_API_DEFAULT + self.T_QUEUE_DEFAULT
        return float(np.mean([j.t_total for j in self.jobs]))

    def project(self, jobs_needed: int) -> Dict[str, float]:
        """Time projection for `jobs_needed` additional jobs."""
        if self.jobs:
            api = float(np.mean([j.t_submit for j in self.jobs]))
            queue = float(np.mean([j.t_queue for j in self.jobs]))
            execu = float(np.mean([j.t_exec for j in self.jobs]))
        else:
            api, queue, execu = (self.T_API_DEFAULT, self.T_QUEUE_DEFAULT,
                                 0.5)
        return {'api': api * jobs_needed, 'queue': queue * jobs_needed,
                'exec': execu * jobs_needed,
                'total': (api + queue + execu) * jobs_needed}

    def report(self, jobs_per_unit: float, units=(100, 500, 1000),
               unit_name: str = "iterations") -> str:
        """Estimation table for several run lengths.

        Args:
            jobs_per_unit: QPU jobs required per step/iteration
                (e.g. 1.0 for QVMC-SPSA; chains/block for QMCMC).
            units: Step/iteration counts to tabulate.
            unit_name: Unit label.
        """
        lines = [f"{'#'+unit_name:>15} | {'jobs':>6} | {'API':>9} | "
                 f"{'queue':>9} | {'QPU':>9} | {'TOTAL':>10}",
                 "-" * 72]
        for n in units:
            jobs = int(np.ceil(n * jobs_per_unit))
            p = self.project(jobs)
            lines.append(
                f"{n:>15d} | {jobs:>6d} | {_fmt_t(p['api']):>9} | "
                f"{_fmt_t(p['queue']):>9} | {_fmt_t(p['exec']):>9} | "
                f"{_fmt_t(p['total']):>10}")
        return "\n".join(lines)


def _fmt_t(seconds: float) -> str:
    """Format seconds as human-readable s/min/h."""
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds/60:.1f}min"
    return f"{seconds/3600:.1f}h"


# =============================================================================
# 2.  HARDWARE CONNECTION
# =============================================================================

class QPUConnection:
    """Encapsulates service, backend, ISA transpilation and Sampler options.

    Args:
        backend_name: Backend name (e.g. 'ibm_brisbane') or None.
        least_busy: If True, pick the least-busy operational backend.
        token: API token; if None the saved account is used.
        use_session: Session mode (iterative, sustained priority) instead
            of Batch (independent jobs, recommended for the open plan).
        shots: Default shots per circuit.
        dry_run: If True, do NOT connect to IBM; "results" are returned as
            synthetic uniform counts, only to validate the workflow and to
            feed the TimingEstimator with the default values.
    """

    def __init__(self, backend_name: Optional[str] = None,
                 least_busy: bool = True, token: Optional[str] = None,
                 use_session: bool = False, shots: int = 4096,
                 dry_run: bool = False,
                 logger: Optional[logging.Logger] = None):
        self.shots = shots
        self.dry_run = dry_run
        self.use_session = use_session
        self.log = logger or logging.getLogger("qpu")
        self.timer = TimingEstimator()
        self._context = None       # active Batch or Session
        self.backend = None
        self.sampler = None
        self.pm = None

        if dry_run:
            self.log.warning("[DRY-RUN] No IBM Quantum connection: counts "
                             "will be synthetic and timings will use the "
                             "default estimates.")
            return

        _import_runtime()
        if token:
            service = QiskitRuntimeService(channel="ibm_quantum_platform",
                                           token=token)
        else:
            service = QiskitRuntimeService()
        if backend_name:
            self.backend = service.backend(backend_name)
        elif least_busy:
            self.backend = service.least_busy(operational=True,
                                              simulator=False)
        else:
            raise ValueError("Specify --backend or --least-busy.")
        self.log.info("Backend: %s (%d qubits)", self.backend.name,
                      self.backend.num_qubits)

        # [QPU] ISA transpilation: mandatory for V2 primitives.
        self.pm = generate_preset_pass_manager(optimization_level=3,
                                               backend=self.backend)

        # Execution context: Batch groups independent jobs behind a single
        # initial queue wait; Session keeps the QPU "reserved" between jobs
        # (ideal for iterative SPSA, requires a paid plan).
        ctx_cls = Session if use_session else Batch
        self._context = ctx_cls(backend=self.backend)
        self.sampler = SamplerV2(mode=self._context)

        # [QPU] Error suppression:
        opt = self.sampler.options
        opt.dynamical_decoupling.enable = True
        opt.dynamical_decoupling.sequence_type = "XY4"
        opt.twirling.enable_gates = True
        opt.twirling.enable_measure = True
        opt.default_shots = shots
        self.log.info("Mode: %s | DD=XY4 | gate+measure twirling | "
                      "shots=%d", "Session" if use_session else "Batch",
                      shots)

    # ------------------------------------------------------------------ #
    def transpile_isa(self, qc: QuantumCircuit) -> QuantumCircuit:
        """Transpile to the backend ISA (once per template)."""
        if self.dry_run:
            return qc
        return self.pm.run(qc)

    # ------------------------------------------------------------------ #
    def run_pub(self, isa_circuit: QuantumCircuit,
                parameter_values: np.ndarray,
                shots: Optional[int] = None) -> List[Dict[str, int]]:
        """Run ONE job with one PUB (circuit, B bindings) -> list of counts.

        [QPU][OPT] Single point of contact with the hardware: ALL batching
        goes through here, so every call = exactly one job measurable by
        the TimingEstimator.

        Args:
            isa_circuit: Circuit already transpiled to ISA.
            parameter_values: Array (B, n_phi) of bindings.
            shots: Shots per binding (default: self.shots).

        Returns:
            List of B count dictionaries {bitstring: frequency}.
        """
        shots = shots or self.shots
        B = int(parameter_values.shape[0])
        t0 = time.time()

        if self.dry_run:
            # Synthetic uniform counts: enough to validate shapes,
            # decoding and logging without touching the QPU.
            n_q = isa_circuit.num_qubits
            outs = []
            for _ in range(B):
                samples = RNG.integers(0, 2**min(n_q, 20), size=shots)
                cnt: Dict[str, int] = {}
                for s in samples:
                    k = format(int(s), f'0{n_q}b')
                    cnt[k] = cnt.get(k, 0) + 1
                outs.append(cnt)
            self.timer.record(B, shots, t_wall=self.timer.T_API_DEFAULT +
                              self.timer.T_QUEUE_DEFAULT +
                              B * shots * self.timer.T_SHOT)
            return outs

        pub = (isa_circuit, parameter_values)
        job = self.sampler.run([pub], shots=shots)
        result = job.result()
        t_wall = time.time() - t0

        # Usage metrics if the backend reports them
        t_exec = None
        try:
            t_exec = float(result.metadata.get('execution',
                                               {}).get('execution_spans',
                                                       None) or 0) or None
        except Exception:
            t_exec = None
        rec = self.timer.record(B, shots, t_wall, t_exec)
        self.log.debug("Job: %d circuits x %d shots | wall %.1fs "
                       "(queue~%.1fs, exec~%.1fs)", B, shots, rec.t_total,
                       rec.t_queue, rec.t_exec)

        outs = []
        for k in range(B):
            data = result[0].data
            # The classical register is named after the circuit creg
            # ('meas' with measure_all, 'c' otherwise).
            reg = getattr(data, 'meas', None) or getattr(data, 'c', None)
            outs.append(reg.get_counts(k) if B > 1 else reg.get_counts())
        return outs

    # ------------------------------------------------------------------ #
    def close(self):
        """Close the Batch/Session if open."""
        if self._context is not None:
            try:
                self._context.close()
            except Exception:
                pass


# =============================================================================
# 3.  CIRCUITS (same topology as the simulator version)
# =============================================================================

def build_proposal_circuit(n_qubits: int, n_layers: int = 3) -> QuantumCircuit:
    """Sarracino et al. (2025) proposal circuit WITH measurement.

    Same topology as in `cosmo_modular_quantum.py`, but terminated with
    `measure_all()`: on hardware there is no statevector, so the
    displacement is reconstructed from measured <Z_q> (see
    :meth:`QPUProposalEngine._counts_to_shift`).
    """
    n_params = n_layers * n_qubits * 2 + n_layers * (n_qubits - 1) + n_qubits
    phi = ParameterVector('phi', n_params)
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
    qc.measure_all()
    return qc


def build_ansatz(n_qubits: int, n_layers: int = 3) -> QuantumCircuit:
    """Hardware-efficient ansatz (RY·RZ + chained CX) WITH measurement."""
    n_p = n_layers * n_qubits * 2 + n_qubits
    phi = ParameterVector('phi', n_p)
    qc = QuantumCircuit(n_qubits)
    qc.h(range(n_qubits))
    idx = 0
    for _ in range(n_layers):
        for q in range(n_qubits):
            qc.ry(phi[idx], q); idx += 1
            qc.rz(phi[idx], q); idx += 1
        for q in range(n_qubits - 1):
            qc.cx(q, q + 1)
    for q in range(n_qubits):
        qc.ry(phi[idx], q); idx += 1
    qc.measure_all()
    return qc


# =============================================================================
# 4.  SHARED HELPERS (acceptance rule, grid encoding)
# =============================================================================

def metropolis_log_accept(lp_cur: float, lp_prop: float) -> float:
    """Log Metropolis acceptance log min(1, e^D), D = lp_prop - lp_cur.

    [QPU] The Metropolis acceptance is sequential (step t depends on t-1),
    so on hardware it would cost one job + a full queue wait PER STEP. We
    therefore evaluate it on the CPU. It uses the SAME rule as the
    simulator's quantum acceptance (`hadamard_accept_log` in
    cosmo_modular_quantum), which encodes min(1, e^D) as a state
    amplitude, so the two pipelines are directly comparable. (Earlier this
    returned the Barker rule log sigmoid(D); switched to Metropolis for
    cross-pipeline consistency.)
    """
    if not np.isfinite(lp_prop):
        return -np.inf
    delta = lp_prop - lp_cur
    return float(min(0.0, delta))   # log min(1, e^D)


class GridEncoding:
    """Discrete grid encoding for QVMC-QPU (index <-> theta on a 2^n grid).

    Same bit convention as the simulator: parameter i occupies qubits
    [i·nqpp, (i+1)·nqpp), with the lowest bit position as the MSB of the
    chunk (Qiskit little-endian bitstrings).
    """

    def __init__(self, model: CosmoModel, nqpp: int):
        self.model = model
        self.nqpp = nqpp
        self.d = model.n_params
        self.n_qubits = self.d * nqpp
        self.n_grid = 2 ** nqpp
        self.n_states = 2 ** self.n_qubits
        self.grids = [np.linspace(b[0], b[1], self.n_grid)
                      for b in model.sample_box]
        self.theta_table = self._build_theta_table()

    def _build_theta_table(self) -> np.ndarray:
        """Table (n_states, d): theta for every basis-state index."""
        idx = np.arange(self.n_states)
        table = np.zeros((self.n_states, self.d))
        for i in range(self.d):
            val = np.zeros(self.n_states, dtype=int)
            for j in range(self.nqpp):
                bit = (idx >> (i * self.nqpp + j)) & 1
                val |= bit << (self.nqpp - 1 - j)
            table[:, i] = self.grids[i][val]
        return table

    def build_target(self, post: Posterior) -> np.ndarray:
        """Target posterior P on the grid (vectorized, classical)."""
        log_p = post.log_prob_batch(self.theta_table)
        valid = np.isfinite(log_p)
        P = np.zeros(self.n_states)
        if np.any(valid):
            log_p[valid] -= np.max(log_p[valid])
            P[valid] = np.exp(log_p[valid])
        return P / P.sum()


def kl_from_counts(counts: Dict[str, int], P: np.ndarray) -> float:
    """KL(Q_hat || P_smoothed) over the observed support of the counts.

    With finite shots Q_hat only has support on the measured bitstrings.
    KL = sum_{x: Q_hat(x)>0} Q_hat log(Q_hat / P_s), with P_s = P smoothed
    (+eps, renormalized) to avoid log(0). Biased low w.r.t. the exact KL,
    but monotonically correlated — valid as a cost function.
    """
    tot = sum(counts.values())
    eps = 1e-10
    P_s = (P + eps)
    P_s = P_s / P_s.sum()
    kl = 0.0
    for bits, c in counts.items():
        q = c / tot
        i = int(bits.replace(" ", ""), 2)
        kl += q * np.log(q / P_s[i])
    return float(max(kl, 0.0))


def counts_theta_mean(counts: Dict[str, int],
                      theta_table: np.ndarray) -> np.ndarray:
    """E_Q[theta] from measured (or synthetic-multinomial) counts."""
    tot = sum(counts.values())
    tm = np.zeros(theta_table.shape[1])
    for bits, c in counts.items():
        tm += (c / tot) * theta_table[int(bits.replace(" ", ""), 2)]
    return tm


def spsa_gains(k: int, a0: float, c0: float) -> Tuple[float, float]:
    """Standard SPSA gain schedule (Spall 1998): a_k, c_k at iteration k."""
    return a0 / (k + 1) ** 0.602, c0 / (k + 1) ** 0.101


# =============================================================================
# 5.  QMCMC ON QPU  +  MANDATORY CLASSICAL MCMC BASELINE
# =============================================================================

class QPUProposalEngine:
    """Block-batched quantum proposals measured on real hardware.

    [QPU][OPT] B random angle sets -> ONE job (PUB with parameter_values
    of shape (B, n_phi)) -> B count distributions. Each distribution is
    converted to a d-dimensional displacement via
    <Z_q> = 1 - 2·P(q=1) in [-1, 1] per qubit, replacing the real part of
    the statevector used in simulation (inaccessible on hardware).

    Args:
        conn: Active QPU connection.
        n_phys: Dimension of the physical parameter vector theta.
        n_layers: Proposal-circuit layers.
        block: Proposals per hardware job.
        shots_per_proposal: Shots used to estimate each <Z_q>.
    """

    def __init__(self, conn: QPUConnection, n_phys: int, n_layers: int = 3,
                 block: int = 64, shots_per_proposal: int = 128):
        self.conn = conn
        self.d = n_phys
        self.n_qubits = max(2, n_phys)
        self.block = block
        self.shots = shots_per_proposal
        qc = build_proposal_circuit(self.n_qubits, n_layers)
        self.n_phi = qc.num_parameters
        self.isa = conn.transpile_isa(qc)   # [OPT] ISA transpiled once
        self._queue: List[np.ndarray] = []

    def _counts_to_shift(self, counts: Dict[str, int]) -> np.ndarray:
        """Per-qubit <Z_q> from counts -> displacement in [-1, 1]^d."""
        tot = sum(counts.values())
        p1 = np.zeros(self.n_qubits)
        for bits, c in counts.items():
            b = bits.replace(" ", "")[::-1]    # Qiskit little-endian
            for q in range(self.n_qubits):
                if b[q] == '1':
                    p1[q] += c
        z = 1.0 - 2.0 * p1 / tot
        return z[:self.d].copy()

    def _refill(self):
        """Fill the queue: one job -> `block` displacements."""
        phis = RNG.uniform(0, 2 * np.pi, size=(self.block, self.n_phi))
        all_counts = self.conn.run_pub(self.isa, phis, shots=self.shots)
        for cnt in all_counts:
            self._queue.append(self._counts_to_shift(cnt))

    def next(self) -> np.ndarray:
        """Next displacement; refills in blocks when the queue is empty."""
        if not self._queue:
            self._refill()
        return self._queue.pop()


class MCMC_QPU:
    """Metropolis-Hastings (QMCMC-QPU) whose proposals come from the QPU.

    The proposal displacements are produced by :class:`QPUProposalEngine`,
    i.e. measured on real hardware (per-qubit <Z_q> from counts). The
    acceptance uses the Barker rule, analytically equivalent to the
    Hadamard test (see :func:`metropolis_log_accept`); it runs on the CPU
    because the chain is inherently sequential and a per-step QPU job would
    waste a full queue wait on every step.

    Args:
        post: cosmo_core posterior.
        engine: Quantum proposal engine (:class:`QPUProposalEngine`).
        n_chains: Parallel chains sharing the QPU proposal queue.
        step_frac: Step scale as a fraction of the sample box.
        rhat_every: Gelman-Rubin recording cadence.
        log_every: Detailed-logging cadence.
        tag: Label used in log lines.
    """

    def __init__(self, post: Posterior, engine, n_chains: int = 4,
                 step_frac: float = 0.06, rhat_every: int = 25,
                 log_every: int = 500, tag: str = 'QMCMC-QPU',
                 logger: Optional[logging.Logger] = None):
        self.post = post
        self.model = post.model
        self.engine = engine
        self.n_chains = n_chains
        widths = np.array([hi - lo for lo, hi in self.model.sample_box])
        self.step = step_frac * widths
        self.rhat_every = rhat_every
        self.log_every = log_every
        self.tag = tag
        self.log = logger or logging.getLogger("qpu")

    # ------------------------------------------------------------------ #
    def run(self, n_steps: int, n_burn: Optional[int] = None) -> dict:
        """Run the chains. Returns a dict with chains/flat/statistics."""
        d = self.model.n_params
        n_burn = n_burn if n_burn is not None else n_steps // 10
        lo = np.array([b[0] for b in self.model.sample_box])
        hi = np.array([b[1] for b in self.model.sample_box])
        cur = lo + (hi - lo) * RNG.uniform(0.3, 0.7, size=(self.n_chains, d))
        lp = np.array([self.post.log_prob(t) for t in cur])

        chains = np.zeros((self.n_chains, n_steps, d))
        n_acc = 0
        rhat_hist: List[Tuple[int, float]] = []
        t0 = time.time()

        for s in range(n_steps):
            for c in range(self.n_chains):
                shift = self.engine.next()      # [QPU] or [BASE] classical
                prop = cur[c] + self.step * shift
                lp_prop = self.post.log_prob(prop)
                if np.log(RNG.uniform() + 1e-300) < metropolis_log_accept(
                        lp[c], lp_prop):
                    cur[c], lp[c] = prop, lp_prop
                    n_acc += 1
                chains[c, s] = cur[c]

            if (s + 1) % self.rhat_every == 0 and s > 20:
                r = gelman_rubin_max(chains[:, max(0, s // 2):s + 1, :])
                rhat_hist.append((s + 1, r))

            if (s + 1) % self.log_every == 0:
                acc = n_acc / ((s + 1) * self.n_chains)
                tm = chains[:, max(0, s // 2):s + 1, :].reshape(-1, d).mean(0)
                self.log.info("[%s] step %d/%d | accept=%.3f | "
                              "Rhat_max=%.4f | theta_mean=%s | jobs=%d",
                              self.tag, s + 1, n_steps, acc,
                              rhat_hist[-1][1] if rhat_hist else np.nan,
                              fmt_theta(self.model, tm),
                              len(self.engine.conn.timer.jobs)
                              if hasattr(self.engine, 'conn') else 0)

        flat = chains[:, n_burn:, :].reshape(-1, d)
        return {
            'chains': chains, 'flat': flat,
            'acceptance': n_acc / (n_steps * self.n_chains),
            'elapsed': time.time() - t0,
            'rhat_hist': rhat_hist,
            'ess': ess_chains(chains[:, n_burn:, :]),
        }


# =============================================================================
# 6.  QVMC ON QPU (SPSA)  +  MANDATORY CLASSICAL VI BASELINE
# =============================================================================

class QVMC_QPU:
    """Variational Quantum Monte Carlo on real hardware with SPSA.

    Encodes the discretized posterior (grid of 2^{nqpp} points per
    parameter) into |psi(phi)|^2 and minimizes KL(Q||P) with Q estimated
    from measured counts.

    [QPU][OPT] SPSA optimization (Spall 1998): the full gradient is
    estimated from ONLY 2 cost evaluations per iteration (phi + c·Delta
    and phi - c·Delta, with Delta in {-1, +1}^{n_phi} random), packed into
    ONE PUB of shape (2, n_phi) -> 1 job per iteration, independent of the
    number of ansatz parameters. Parameter-shift would have cost
    2·n_phi ~ 84 evaluations/iteration.

    Args:
        post: cosmo_core posterior.
        conn: QPU connection.
        n_qubits_per_param: Qubits per parameter (grid of 2^n points).
        n_layers: Ansatz layers.
        a0, c0: Initial SPSA gains (learning rate and perturbation).
        log_every: Logging cadence.
    """

    def __init__(self, post: Posterior, conn: QPUConnection,
                 n_qubits_per_param: int = 3, n_layers: int = 2,
                 a0: float = 0.15, c0: float = 0.1, log_every: int = 500,
                 logger: Optional[logging.Logger] = None):
        self.post = post
        self.model = post.model
        self.conn = conn
        self.enc = GridEncoding(self.model, n_qubits_per_param)
        self.P = self.enc.build_target(post)
        self.a0, self.c0 = a0, c0
        self.log_every = log_every
        self.log = logger or logging.getLogger("qpu")

        qc = build_ansatz(self.enc.n_qubits, n_layers)
        self.n_phi = qc.num_parameters
        self.isa = conn.transpile_isa(qc)   # [OPT] ISA transpiled once
        self.history: List[dict] = []

    # ── SPSA training ───────────────────────────────────────────────────── #
    def train(self, n_iters: int) -> np.ndarray:
        """Optimize phi with SPSA: 1 hardware job per iteration."""
        phi = RNG.uniform(0, 2 * np.pi, self.n_phi)
        t0 = time.time()
        for k in range(n_iters):
            ak, ck = spsa_gains(k, self.a0, self.c0)
            delta = RNG.choice([-1.0, 1.0], size=self.n_phi)
            pv = np.vstack([phi + ck * delta, phi - ck * delta])
            c_plus, c_minus = self.conn.run_pub(self.isa, pv)   # [QPU] 1 job
            f_plus = kl_from_counts(c_plus, self.P)
            f_minus = kl_from_counts(c_minus, self.P)
            ghat = (f_plus - f_minus) / (2 * ck) * delta
            phi = phi - ak * ghat

            kl_mid = 0.5 * (f_plus + f_minus)
            # theta_mean estimated from the "+" side (avoids an extra job
            # used only for logging)
            tm = counts_theta_mean(c_plus, self.enc.theta_table)
            self.history.append({'it': k + 1, 'kl': kl_mid,
                                 'theta_mean': tm})
            if (k + 1) % self.log_every == 0 or k == n_iters - 1:
                self.log.info("[QVMC-QPU] iter %d/%d | KL~%.4f | "
                              "theta_mean=%s | jobs=%d | %.0fs", k + 1,
                              n_iters, kl_mid, fmt_theta(self.model, tm),
                              len(self.conn.timer.jobs), time.time() - t0)
        self.phi_opt = phi
        return phi

    # ── final sampling ──────────────────────────────────────────────────── #
    def sample(self, n_samples: int = 4000,
               shots_per_job: int = 4096) -> np.ndarray:
        """Sample theta from the optimized circuit by measuring on the QPU."""
        samples: List[np.ndarray] = []
        pv = self.phi_opt[None, :]
        while len(samples) < n_samples:
            counts = self.conn.run_pub(self.isa, pv,
                                       shots=shots_per_job)[0]
            for bits, c in counts.items():
                th = self.enc.theta_table[int(bits.replace(" ", ""), 2)]
                samples.extend([th] * c)
        return np.array(samples[:n_samples])


# =============================================================================
# 7.  QUANTUM-ONLY FIGURES (single method, annotated with run metadata)
# =============================================================================
#
# [QONLY] This script runs no classical method, so the figures show the
# quantum result alone. Each title/legend embeds the run metadata required
# for the analysis (requirement 2): MCMC steps for QMCMC; SPSA iterations
# AND nqpp (qubits per parameter) for QVMC.

def plot_corner_quantum(flat: np.ndarray, model: CosmoModel, outdir: str,
                        tag: str, title: str, label: str,
                        color: str = C_QUANTUM,
                        weights: Optional[np.ndarray] = None) -> str:
    """Single-distribution corner plot (corner.py) of a quantum posterior.

    2D contours (1sigma/2sigma) + 1D marginals, with Planck fiducials as
    dashed black lines. `title` carries the run metadata.
    """
    rng = [(flat[:, i].min(), flat[:, i].max())
           for i in range(model.n_params)]
    fig = corner.corner(
        flat, color=color, weights=weights, labels=list(model.param_latex),
        range=rng, plot_datapoints=False, plot_density=False,
        levels=(0.393, 0.865), smooth=1.0, bins=30,
        hist_kwargs={'density': True, 'lw': 2})
    corner.overplot_lines(fig, model.fiducial, color='k', ls='--', lw=1)
    handles = [plt.Line2D([], [], color=color, lw=2, label=label),
               plt.Line2D([], [], color='k', ls='--', lw=1,
                          label='Fiducial (Planck)')]
    fig.legend(handles=handles, loc='upper right',
               bbox_to_anchor=(0.98, 0.88), fontsize=11, frameon=True)
    fig.suptitle(title, fontsize=13, fontweight='bold', y=1.02)
    f = os.path.join(outdir, f'corner_{tag}.png')
    fig.savefig(f, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return f


def plot_kl_quantum(hist: List[dict], outdir: str, tag: str,
                    n_iters: int, nqpp: int) -> str:
    """QVMC-QPU KL training curve. Title embeds SPSA iterations and nqpp."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy([h['it'] for h in hist], [max(h['kl'], 1e-12) for h in hist],
                color=C_QUANTUM2, lw=2.2, label='QVMC-QPU (hardware)')
    ax.set_xlabel('SPSA iteration')
    ax.set_ylabel(r'KL$(\hat{Q}\,\|\,P_{\rm target})$  (shot-estimated)')
    ax.set_title(f'QVMC-QPU variational training\n'
                 f'(SPSA iterations = {n_iters},  nqpp = {nqpp} '
                 f'qubits/parameter)')
    ax.grid(alpha=0.3)
    ax.legend()
    f = os.path.join(outdir, f'kl_quantum_{tag}.png')
    fig.savefig(f, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return f


def plot_rhat_quantum(rh: List[Tuple[int, float]], outdir: str, tag: str,
                      n_steps: int, n_chains: int) -> str:
    """QMCMC-QPU Gelman-Rubin curve. Title embeds total steps and chains."""
    fig, ax = plt.subplots(figsize=(8, 5))
    if rh:
        ax.semilogy([p[0] for p in rh], [max(p[1] - 1.0, 1e-6) for p in rh],
                    color=C_QUANTUM, lw=2.2, marker='o', ms=4,
                    label='QMCMC-QPU (hardware)')
    ax.axhline(0.05, color='gray', ls='--', lw=1,
               label=r'$\hat{R} = 1.05$ threshold')
    ax.set_xlabel('MCMC step')
    ax.set_ylabel(r'$\hat{R}_{\max} - 1$')
    ax.set_title(f'QMCMC-QPU convergence\n'
                 f'(total steps = {n_steps},  chains = {n_chains})')
    ax.grid(alpha=0.3)
    ax.legend()
    f = os.path.join(outdir, f'rhat_quantum_{tag}.png')
    fig.savefig(f, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return f


# =============================================================================
# 8.  CLI
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    p = argparse.ArgumentParser(
        description="Cosmological inference on real IBM Quantum hardware "
                    "(SamplerV2). QUANTUM-ONLY: runs QMCMC and/or QVMC on "
                    "the QPU and runs NO classical method. Use --dry-run to "
                    "plan without an account. For classical baselines and "
                    "the full classical-vs-quantum overlay study, use "
                    "cosmo_modular_quantum.py.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--model', choices=list(MODELS), default='lcdm',
                   help='Cosmological model')
    p.add_argument('--dataset', choices=['CC', 'Pantheon+', 'CC+Pantheon+'],
                   default='CC', help='Observational dataset')
    p.add_argument('--prior', choices=['flat', 'gaussian'], default='flat',
                   help='Prior (gaussian = Planck 2018 on Om, H0)')
    p.add_argument('--method', choices=['qvmc', 'qmcmc', 'both'],
                   default='qvmc', help='QPU inference method')
    p.add_argument('--steps', type=int, default=200,
                   help='QMCMC steps (annotated on the R-hat figure)')
    p.add_argument('--iters', type=int, default=30,
                   help='QVMC SPSA iterations (= 1 QPU job each)')
    p.add_argument('--burn', type=int, default=None,
                   help='MCMC burn-in (default: 10%% of steps)')
    p.add_argument('--chains', type=int, default=4, help='MCMC chains')
    p.add_argument('--block', type=int, default=64,
                   help='QMCMC proposals per hardware job')
    p.add_argument('--nqpp', type=int, default=3,
                   help='Qubits per physical parameter (QVMC grid)')
    p.add_argument('--layers', type=int, default=2,
                   help='Ansatz / proposal-circuit layers')
    p.add_argument('--shots', type=int, default=4096,
                   help='Shots per circuit (KL estimation and sampling)')
    p.add_argument('--samples', type=int, default=4000,
                   help='Final posterior samples per method')
    # Hardware
    p.add_argument('--backend', type=str, default=None,
                   help="Specific backend, e.g. 'ibm_brisbane'")
    p.add_argument('--least-busy', action='store_true', default=True,
                   help='Pick the least-busy operational backend')
    p.add_argument('--token', type=str, default=None,
                   help='IBM Quantum token (if no saved account)')
    p.add_argument('--session', action='store_true',
                   help='Use Session instead of Batch (paid plans)')
    p.add_argument('--dry-run', action='store_true',
                   help='Plan and validate WITHOUT connecting to IBM')
    p.add_argument('--max-jobs', type=int, default=200,
                   help='Abort if the run would require more jobs than this')
    # Output
    p.add_argument('--log-file', type=str, default=None,
                   help='Log file (default: results/qpu_<ts>.log)')
    p.add_argument('--log-every', type=int, default=500,
                   help='Progress-logging cadence')
    p.add_argument('--outdir', type=str, default='results')
    p.add_argument('--seed', type=int, default=42,
                   help='RNG seed for the QPU run')
    p.add_argument('--no-plot', action='store_true',
                   help='Skip figure generation')
    return p


def estimate_jobs(args) -> int:
    """QPU jobs the requested run will need (for the --max-jobs guard).

    [QONLY] Only the quantum methods consume hardware; this script runs no
    classical code at all.
    """
    jobs = 0
    if args.method in ('qvmc', 'both'):
        jobs += args.iters                                # 1 SPSA job/iter
        jobs += int(np.ceil(args.samples / args.shots))   # final sampling
    if args.method in ('qmcmc', 'both'):
        proposals = args.steps * args.chains
        jobs += int(np.ceil(proposals / args.block))
    return jobs


def _summary(log, side: str, model: CosmoModel, mean: np.ndarray,
             std: np.ndarray, stats: dict, extra: str):
    """Log one method's summary block (mean ± std, chi2/AIC/BIC, extras)."""
    log.info("  | %s", side)
    for n, m, s in zip(model.param_names, mean, std):
        log.info("  |   %-5s = %.4f +/- %.4f", n, m, s)
    log.info("  |   chi2=%.2f  chi2_red=%.3f  AIC=%.2f  BIC=%.2f  %s",
             stats['chi2'], stats['chi2_red'], stats['AIC'], stats['BIC'],
             extra)


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point."""
    args = build_parser().parse_args(argv)

    os.makedirs(args.outdir, exist_ok=True)
    ts = time.strftime('%Y%m%d_%H%M%S')
    log_file = args.log_file or os.path.join(args.outdir, f"qpu_{ts}.log")
    log = setup_logger(log_file=log_file, name="qpu")
    # On QPU runs, progress matters on the console too (long wall times):
    for h in log.handlers:
        if isinstance(h, logging.StreamHandler) and not isinstance(
                h, logging.FileHandler):
            h.setLevel(logging.INFO)

    log.info("=" * 70)
    log.info("QPU COSMO SAMPLERS — model=%s dataset=%s prior=%s method=%s "
             "seed=%s", args.model, args.dataset, args.prior, args.method,
             args.seed)

    # Job budget BEFORE touching the hardware
    n_jobs = estimate_jobs(args)
    log.info("Estimated QPU jobs for this run: %d (quantum-only; no "
             "classical code runs here)", n_jobs)
    if n_jobs > args.max_jobs:
        log.error("The run would require %d jobs > --max-jobs=%d. "
                  "Reduce --steps/--iters or raise --block/--max-jobs.",
                  n_jobs, args.max_jobs)
        return 1

    model = MODELS[args.model]
    post = Posterior(model, dataset=args.dataset, prior_type=args.prior)
    log.info("Posterior: %d parameters, %d data points",
             model.n_params, post.n_data)

    conn = QPUConnection(backend_name=args.backend,
                         least_busy=args.least_busy, token=args.token,
                         use_session=args.session, shots=args.shots,
                         dry_run=args.dry_run, logger=log)
    try:
        results: Dict[str, dict] = {}
        figures: List[str] = []

        # ================= QMCMC (MCMC family) ========================== #
        if args.method in ('qmcmc', 'both'):
            log.info("-" * 70)
            log.info("QMCMC-QPU: %d steps x %d chains (blocks of %d)",
                     args.steps, args.chains, args.block)
            _reseed(args.seed)
            qm = MCMC_QPU(post, QPUProposalEngine(conn, model.n_params,
                                                  n_layers=args.layers,
                                                  block=args.block),
                          n_chains=args.chains, log_every=args.log_every,
                          tag='QMCMC-QPU', logger=log)
            r_q = qm.run(args.steps, args.burn)
            st_q = fit_statistics(post, r_q['flat'].mean(0))

            log.info("QMCMC-QPU — %d steps, %d chains:",
                     args.steps, args.chains)
            _summary(log, "QMCMC-QPU (hardware)", model,
                     r_q['flat'].mean(0), r_q['flat'].std(0), st_q,
                     f"accept={r_q['acceptance']:.3f} ESS={r_q['ess']:.0f} "
                     f"jobs={len(conn.timer.jobs)}")

            if not args.no_plot:
                tag = f"mcmc_{model.name}"
                # [QONLY] single-method figures, metadata in the titles
                figures.append(plot_corner_quantum(
                    r_q['flat'], model, args.outdir, tag,
                    title=(f"{model.label} — QMCMC-QPU\n"
                           f"(steps = {args.steps}, chains = {args.chains})"),
                    label='QMCMC-QPU (hardware)', color=C_QUANTUM))
                figures.append(plot_rhat_quantum(
                    r_q['rhat_hist'], args.outdir, model.name,
                    n_steps=args.steps, n_chains=args.chains))

            st = dict(st_q)
            st['theta_best'] = np.asarray(st['theta_best']).tolist()
            results['qmcmc_qpu'] = {
                'theta_mean': r_q['flat'].mean(0).tolist(),
                'theta_std': r_q['flat'].std(0).tolist(),
                **st, 'acceptance': r_q['acceptance'], 'ess': r_q['ess'],
                'n_steps': args.steps, 'n_chains': args.chains}

        # ================= QVMC (variational family) ==================== #
        if args.method in ('qvmc', 'both'):
            log.info("-" * 70)
            log.info("QVMC-QPU: %d SPSA iterations (1 job each), nqpp=%d "
                     "-> %d qubits", args.iters, args.nqpp,
                     model.n_params * args.nqpp)
            _reseed(args.seed)
            qv = QVMC_QPU(post, conn, n_qubits_per_param=args.nqpp,
                          n_layers=args.layers, log_every=args.log_every,
                          logger=log)
            qv.train(args.iters)
            S_q = qv.sample(args.samples, shots_per_job=args.shots)
            st_q = fit_statistics(post, S_q.mean(0))

            kl_q = qv.history[-1]['kl'] if qv.history else np.nan
            log.info("QVMC-QPU — %d iterations, nqpp=%d, %d shots:",
                     args.iters, args.nqpp, args.shots)
            _summary(log, "QVMC-QPU (hardware)", model, S_q.mean(0),
                     S_q.std(0), st_q, f"final KL~{kl_q:.4f}")

            if not args.no_plot:
                tag = f"qvmc_{model.name}"
                figures.append(plot_corner_quantum(
                    S_q, model, args.outdir, tag,
                    title=(f"{model.label} — QVMC-QPU\n"
                           f"(iterations = {args.iters}, "
                           f"nqpp = {args.nqpp} qubits/parameter)"),
                    label='QVMC-QPU (hardware)', color=C_QUANTUM2))
                figures.append(plot_kl_quantum(
                    qv.history, args.outdir, model.name,
                    n_iters=args.iters, nqpp=args.nqpp))

            st = dict(st_q)
            st['theta_best'] = np.asarray(st['theta_best']).tolist()
            results['qvmc_qpu'] = {
                'theta_mean': S_q.mean(0).tolist(),
                'theta_std': S_q.std(0).tolist(), **st, 'kl_final': kl_q,
                'nqpp': args.nqpp, 'n_iters': args.iters,
                'history': [{'it': h['it'], 'kl': h['kl']}
                            for h in qv.history]}

        # ── Timing report (measured + projection 100/500/1000) ─────────── #
        log.info("=" * 70)
        log.info("MEASURED TIMINGS: %d jobs, mean wall %.1fs/job",
                 len(conn.timer.jobs), conn.timer.mean_wall)
        if args.method in ('qvmc', 'both'):
            log.info("QVMC-QPU projection (1 job per SPSA iteration):\n%s",
                     conn.timer.report(1.0, unit_name="iterations"))
        if args.method in ('qmcmc', 'both'):
            jpu = args.chains / args.block
            log.info("QMCMC-QPU projection (%.3f jobs/step with %d chains, "
                     "block %d):\n%s", jpu, args.chains, args.block,
                     conn.timer.report(jpu, unit_name="steps"))

        if figures:
            log.info("Quantum figures: %s", ", ".join(figures))
        out_json = os.path.join(args.outdir, f"qpu_results_{ts}.json")
        with open(out_json, 'w') as f:
            json.dump({'args': vars(args), 'results': results}, f, indent=2,
                      default=float)
        log.info("Results -> %s | Log -> %s", out_json, log_file)
        return 0
    finally:
        conn.close()


if __name__ == '__main__':
    sys.exit(main())
```
