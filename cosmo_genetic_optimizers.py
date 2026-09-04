# =============================================================================
#  cosmo_genetic_optimizers.py — Classical & Quantum Genetic Algorithms (CGA/QGA)
# =============================================================================
#
#  PHASE 2 of the project: GLOBAL OPTIMIZATION FOR THE MAP.
#
#  This module adds two GLOBAL optimizers that locate the Maximum A Posteriori
#  (MAP) of the cosmological posterior — to be used BEFORE or IN PARALLEL with
#  the MCMC/VI samplers of `cosmo_modular_quantum.py`:
#
#    * CGA — Classical Genetic Algorithm, written from scratch (NO black-box
#            library such as DEAP/PyGAD), fully vectorized with NumPy.
#    * QGA — Quantum Genetic Algorithm built with Qiskit, with a MODULAR
#            "quantumness" system: quantum initialization, quantum mutation
#            and quantum crossover can each be toggled on/off independently.
#
#  [ARCH] STRICT REUSE OF THE EXISTING PHYSICS.
#         The fitness is NOTHING but the cosmological posterior already
#         structured in `cosmo_core.py`. We never re-derive a χ²: we call the
#         SAME `Posterior.log_prob` / `Posterior.log_prob_batch` (CC + Pantheon+
#         likelihoods, analytic M_abs marginalization, box/Gaussian priors) and
#         the SAME `fit_statistics` (χ², χ²_red, AIC, BIC). The genetic fitness
#         is  f(θ) = log P(θ | data) = −½ χ²(θ) + log prior(θ),  so MAXIMIZING
#         the genetic fitness ≡ MINIMIZING χ² under the prior ≡ finding the MAP.
#         Adding a new model (VC, …) requires ZERO changes here: it inherits
#         straight from `cosmo_core.MODELS`.
#
#  [QUANT] The "quantumness" of the QGA reuses the same philosophy as the
#         sampler module: a weighted 0–100 % score over independently
#         switchable quantum components, so a 0 % QGA is bit-compatible with
#         the CGA code path and serves as the mandatory classical baseline.
#
#  [GUI]  LIVE VISUALIZATION (interactive mode ONLY): a two-panel Matplotlib
#         window updated every generation —
#           * Phase-space panel  : population scatter converging to the MAP
#                                   in (Ωm, H0) [or the first two parameters].
#           * Fitness panel      : best χ² and mean χ² vs generation.
#         with dynamic text (generation, best χ², current physical values).
#
#  [PLOT] After evolution, the result plugs into the EXISTING visualization
#         pipeline: fitness-weighted corner plot of the final population, and
#         an OVERLAY option that superimposes the genetic MAP + final spread
#         on top of the MCMC/VI corner plots (reusing `plot_corner_multi`).
#
#  [CSV]  The MAP and its final χ² are appended to `resultados_config.csv`
#         under "Method" = "CGA" / "QGA (q=NN%)", matching the schema written
#         by `lcdm_quantum_samplers_personal.py` so all runs share one table.
#
#  [CLI]  argparse extended with the genetic methods and hyperparameters
#           --methods cga qga    --generations N    --population-size N    …
#         CRITICAL RULE: when launched from the CLI with arguments
#         (batch / HPC mode) the live animation is DISABLED automatically and
#         the generational progress is written to the LOG every N generations.
#
#  Usage:
#    python cosmo_genetic_optimizers.py                       # interactive menu
#    python cosmo_genetic_optimizers.py --methods cga --model lcdm --generations 80
#    python cosmo_genetic_optimizers.py --methods cga qga --dataset CC+Pantheon+ \
#           --population-size 200 --generations 120 --qga-preset 67
# =============================================================================

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import warnings
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

# [FIX] Matplotlib backend handling — this is what made the live GUI fail.
#   * At import time we DO NOT force any backend.
#   * `cosmo_modular_quantum` no longer forces 'Agg' on import either, so simply
#     importing it (done lazily, see below) can no longer disable our GUI.
#   * CLI/batch mode calls set_headless_backend() ('Agg'); interactive mode
#     calls ensure_interactive_backend() to guarantee a real GUI window.
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# [A5] Scope warning suppression to benign library categories only; let
# numerical RuntimeWarnings (overflow/invalid) surface — they signal real
# physics/likelihood problems.
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# ── Physics & shared infrastructure: imported, never re-implemented ───────────
import cosmo_core as core
import cosmo_noise as cnoise
from cosmo_core import (MODELS, Posterior, fit_statistics, fmt_theta,
                        ess_weights, gpu_available, make_run_dir,
                        make_simulator, resolve_device, setup_logger)

# Module-level GPU preference, set by main() from --gpu or the menu. The QGA's
# AerSimulator is built through make_simulator honoring this flag.
USE_GPU = False

# [ANIM] Formatos de la animacion del genetico, fijados por main() desde --anim.
# Mismo patron que USE_GPU: un global que toda ruta de figuras consulta, para no
# arrastrar el parametro por cinco firmas.
#   ()            -> no animar
#   ('gif',)      -> GIF (por defecto: no necesita ffmpeg y se pega en slides)
#   ('gif','mp4') -> ambos
ANIM_FORMATS: Sequence[str] = ('gif',)

# [B-OVERLAP] Disposicion de la animacion: 'overlay' o 'facets'.
ANIM_LAYOUT: str = 'overlay'

# =============================================================================
#  [NOISE] Segundo eje de ablacion — nivel de ruido del modulo.
# =============================================================================
#
#  El QGA es el metodo mas facil de llevar al eje de ruido, y no por casualidad:
#  ya opera al 100% por MEDICION. Sus tres operadores cuanticos leen
#  `result.get_memory(...)` — cadenas de bits medidas — y no tocan una sola
#  amplitud. Por eso aqui no hay ninguna conversion de lectura que hacer: basta
#  con entregarle a Aer el modelo de ruido y el propio simulador aplica el canal
#  de compuertas y el de lectura donde corresponde.
#
#  Consecuencia para la interpretacion de la escalera: el error de lectura
#  voltea bits medidos, que es casi indistinguible de mutacion adicional. Es la
#  razon estructural por la que cabe esperar que el QGA sea el metodo mas
#  robusto del eje — pero es una hipotesis a medir, no un resultado.
NOISE: "cnoise.NoiseSpec" = cnoise.NoiseSpec.from_level('none')


def set_noise(spec: "cnoise.NoiseSpec") -> None:
    """Fija el peldano de ruido del modulo.

    Debe llamarse ANTES de construir cualquier QGA: el simulador se crea en
    `__init__` y los circuitos se transpilan una sola vez contra el.

    Args:
        spec: peldano del eje, de `cosmo_noise.NoiseSpec`.
    """
    global NOISE
    NOISE = spec


def _qga_sim():
    """AerSimulator del peldano de ruido actual, para la ruta por medicion.

    `counts_route=True` porque los circuitos del QGA MIDEN: Aer aplica el
    canal de lectura por si mismo y no debe volver a aplicarse en cerrado.
    """
    return make_simulator(prefer_gpu=USE_GPU,
                          **NOISE.simulator_kwargs(counts_route=True))

# ── Color convention, defined LOCALLY so the genetic module and its live GUI
#    never depend on importing the (heavier) sampler module. The values match
#    cosmo_modular_quantum exactly, so overlays stay visually consistent. The
#    sampler module itself is imported LAZILY (only inside the corner-overlay
#    helpers) via `_cmq()`, so a missing Qiskit/corner install cannot block the
#    CGA or the live GUI.
C_CLASSICAL = '#1f77b4'   # blue   — Classical MCMC / Classical VI
C_CLASSICAL2 = '#17becf'  # teal
C_QUANTUM   = '#d62728'   # red    — QMCMC
C_QUANTUM2  = '#ff7f0e'   # orange — QVMC
C_GENETIC  = '#2ca02c'    # green  — CGA (classical genetic)
C_GENETIC2 = '#9467bd'    # purple — QGA (quantum genetic)

# [B-OVERLAP] Estilos de linea para series que se solapan. Nombrados (no
# tuplas) porque las colecciones de contorno de matplotlib exigen nombres.
_LS_CYCLE = ['solid', 'dashed', 'dashdot', 'dotted', (0, (3, 1, 1, 1))]


def _cmq():
    """Lazily import cosmo_modular_quantum (only needed for corner overlays).

    Importing it eagerly pulled in Qiskit/corner and—historically—forced the
    'Agg' backend, silently killing the live window. Importing it on demand,
    after the GUI backend is chosen, removes that coupling entirely.
    """
    import cosmo_modular_quantum as cmq
    return cmq


def set_headless_backend():
    """Force the non-interactive 'Agg' backend (CLI/batch/HPC mode)."""
    matplotlib.use('Agg', force=True)


def ensure_interactive_backend() -> bool:
    """Try to guarantee a WORKING interactive Matplotlib backend for the GUI.

    The subtlety this handles: `matplotlib.use('TkAgg', force=True)` does NOT
    raise even when tkinter is missing — the failure only surfaces later when a
    window is actually created. So here we don't trust `use()`; for each
    candidate backend we actually try to BUILD and close a throwaway figure,
    and only accept the backend if that round-trip succeeds.

    Returns True if a usable interactive backend is active, False otherwise
    (headless / no GUI toolkit installed), in which case the caller skips the
    live window and falls back to static figures.
    """
    backend = matplotlib.get_backend().lower()

    def _works() -> bool:
        # A real smoke test: open a 1x1 figure and close it. If the toolkit is
        # missing or there is no display, this raises and we move on.
        try:
            fig = plt.figure()
            plt.close(fig)
            return True
        except Exception:
            return False

    # If we are already on something other than Agg and it actually works, keep
    # it. (A non-interactive backend like 'pdf' would pass _works() but isn't
    # interactive, so we also require the backend name to be a known GUI one.)
    interactive_names = ('qtagg', 'tkagg', 'macosx', 'qt5agg', 'qt6agg',
                         'gtk3agg', 'gtk4agg', 'wxagg', 'nbagg')
    if backend in interactive_names and _works():
        return True

    for cand in ('QtAgg', 'TkAgg', 'MacOSX', 'Qt5Agg', 'GTK3Agg', 'WXAgg'):
        try:
            matplotlib.use(cand, force=True)
            import importlib
            importlib.reload(plt)
            if _works():
                return True
        except Exception:
            continue

    # Nothing usable: restore Agg so static figures still save cleanly.
    try:
        matplotlib.use('Agg', force=True)
        import importlib
        importlib.reload(plt)
    except Exception:
        pass
    return False


def diagnose_gui_backend() -> str:
    """Return a short, actionable hint about why no GUI backend was found.

    Tailored to the most common case (WSL/Ubuntu): a missing `python3-tk`
    system package and/or an unset DISPLAY. Used only to print guidance when
    the live GUI cannot start, so the user can fix their environment.
    """
    lines = []
    has_display = bool(os.environ.get('DISPLAY') or
                       os.environ.get('WAYLAND_DISPLAY'))
    try:
        import tkinter  # noqa: F401
        has_tk = True
    except Exception:
        has_tk = False
    has_qt = False
    for q in ('PyQt5', 'PyQt6', 'PySide6'):
        try:
            __import__(q)
            has_qt = True
            break
        except Exception:
            pass

    if not has_tk and not has_qt:
        lines.append("    • No GUI toolkit found. Install one (pick ONE):")
        lines.append("        sudo apt install python3-tk      # Tk backend "
                     "(simplest on WSL/Ubuntu)")
        lines.append("        pip install PyQt5                # Qt backend "
                     "(alternative)")
    if not has_display:
        lines.append("    • DISPLAY is not set. On WSL2 with Windows 11, WSLg "
                     "provides this automatically — make sure WSL is updated "
                     "(`wsl --update` in Windows PowerShell) and restart the "
                     "terminal.")
        lines.append("      On Windows 10 / WSL1, run an X server (VcXsrv) and "
                     "`export DISPLAY=:0` (or your server's address).")
    if not lines:
        lines.append("    • A GUI toolkit and DISPLAY are present, but the "
                     "backend still failed to open a window. Try forcing one: "
                     "`export MPLBACKEND=TkAgg`.")
    return "\n".join(lines)


# =============================================================================
# 0.  QGA QUANTUM COMPONENTS AND QUANTUMNESS SCORE
# =============================================================================
#
#  Same design as cosmo_modular_quantum.QUANTUM_COMPONENTS, but for the THREE
#  genetic operators that can be made quantum. Each one carries a weight; the
#  quantumness % is the weighted fraction of active quantum operators.
#
#  Why these weights? They reflect how much genuinely-quantum structure each
#  operator injects:
#    * initialization (25): a Hadamard layer prepares a uniform superposition
#      over the encoded gene grid → the initial population is sampled from a
#      true quantum measurement instead of a classical PRNG.
#    * mutation (35): parametrized RY rotations rotate each gene-qubit toward
#      |0⟩/|1⟩, i.e. a coherent, amplitude-level mutation; it is the operator
#      most often run on hardware, hence the largest weight.
#    * crossover (40): [describes the PRE-C1-FIX operator] entangling CX plus
#      a controlled interference layer mixes parental information through
#      genuine entanglement — the most "quantum" of the three.
#
#  A QGA with all three OFF (0 %) reduces EXACTLY to the CGA operators, which
#  is the mandatory classical baseline (verified in the self-test).

# =============================================================================
# 0.  QGA CLASSICAL-QUANTUM ABLATION (uniform weighting)
# =============================================================================
#
#  Same ablation philosophy as cosmo_modular_quantum: the QGA is the CGA with
#  one or more genetic operators (init / mutation / crossover) substituted by a
#  quantum implementation. The index counts HOW MANY operators are quantum,
#  with UNIFORM weighting (option A) — every operator counts equally, because
#  "how much quantum structure each injects" has no measurable definition. The
#  three operators are ALGORITHMIC substitutions (a quantum-sampled population,
#  a coherent RY mutation and an entangling crossover are genuinely different
#  operators from their classical counterparts, expected to change the
#  trajectory), while the 0% configuration reproduces the CGA exactly and is
#  the FAITHFUL baseline cell.

FAITHFUL = 'faithful'
ALGORITHMIC = 'algorithmic'

QGA_COMPONENTS: Dict[str, dict] = {
    'q_init':      {'kind': ALGORITHMIC,
                    'name': 'Quantum initialization (superposition population)'},
    'q_mutation':  {'kind': ALGORITHMIC,
                    'name': 'Quantum mutation (parametrized RY rotations)'},
    'q_crossover': {'kind': ALGORITHMIC,
                    'name': 'Quantum crossover (partial-SWAP interference, '
                            'consensus-preserving)'},
}

#: Legacy hand-picked weights (no measurable definition), kept only so
#: `legacy_qga_weighted_index` can reproduce old CSV numbers.
_LEGACY_QGA_WEIGHTS = {'q_init': 25, 'q_mutation': 35, 'q_crossover': 40}

#: Convenience presets spanning the ablation ladder. Keys are the UNIFORM
#: index (#quantum operators / 3 · 100, rounded), recomputed by
#: `compute_qga_quantumness`.
QGA_PRESETS: Dict[int, dict] = {
    0:   dict(q_init=False, q_mutation=False, q_crossover=False,
              label='0/3 quantum — Classical operators (CGA baseline)'),
    33:  dict(q_init=True,  q_mutation=False, q_crossover=False,
              label='1/3 quantum — Quantum initialization only'),
    67:  dict(q_init=True,  q_mutation=True,  q_crossover=False,
              label='2/3 quantum — Quantum init + mutation'),
    100: dict(q_init=True,  q_mutation=True,  q_crossover=True,
              label='3/3 quantum — Fully quantum operators'),
}


def compute_qga_quantumness(config: dict) -> float:
    """QGA ablation index: 100 · (#quantum operators)/3 (uniform weighting)."""
    n_total = len(QGA_COMPONENTS)
    n_quantum = sum(1 for k in QGA_COMPONENTS if config.get(k, False))
    return round(100.0 * n_quantum / n_total, 1)


def legacy_qga_weighted_index(config: dict) -> float:
    """Historical hand-weighted QGA index (heuristic; CSV continuity only)."""
    total = sum(_LEGACY_QGA_WEIGHTS.values())
    earned = sum(_LEGACY_QGA_WEIGHTS[k] for k in _LEGACY_QGA_WEIGHTS
                 if config.get(k, False))
    return round(100.0 * earned / total, 1)


# =============================================================================
# 1.  SHARED GENETIC INFRASTRUCTURE  (encoding + fitness + classical operators)
# =============================================================================
#
#  Both CGA and QGA share:
#    * a real-valued chromosome θ ∈ R^d living in the model's `bounds` box,
#    * the SAME fitness  f(θ) = log P(θ | data)  (the existing posterior),
#    * the SAME selection / elitism logic.
#  They differ ONLY in the init / mutation / crossover operators, which the QGA
#  may route through Qiskit. This keeps the comparison strictly controlled.


@dataclass
class GAConfig:
    """Hyper-parameters shared by CGA and QGA.

    Attributes:
        pop_size: Number of individuals per generation.
        n_generations: Number of generations to evolve.
        crossover_rate: Probability that a mating pair recombines.
        mutation_rate: Per-gene probability of mutation.
        mutation_scale: Std of the (classical) Gaussian mutation, as a
            FRACTION of each parameter's box width. Quantum mutation uses the
            same scale to set its rotation amplitude, for a fair comparison.
        elite_frac: Fraction of top individuals copied verbatim to the next
            generation (elitism guarantees the best χ² never worsens).
        tournament_k: Tournament size for parent selection.
        seed: RNG seed (shared by CGA and the QGA's classical parts).
    """
    pop_size: int = 120
    n_generations: int = 80
    crossover_rate: float = 0.9
    mutation_rate: float = 0.20
    mutation_scale: float = 0.12
    elite_frac: float = 0.08
    tournament_k: int = 3
    seed: int = 42


class GeneticBase:
    """Common machinery for the genetic optimizers.

    The class owns the population (a (pop_size, d) float array in the box),
    the fitness evaluation (delegated to the cosmological posterior) and the
    classical selection / crossover / mutation operators. Subclasses (CGA,
    QGA) override the three operators they wish to make quantum.

    Args:
        post: The cosmological `Posterior` (physics + data + prior).
        ga: A `GAConfig` with the hyper-parameters.
        rng: A NumPy Generator (so CGA and QGA can share an identical stream).
    """

    def __init__(self, post: Posterior, ga: GAConfig,
                 rng: Optional[np.random.Generator] = None):
        self.post = post
        self.model = post.model
        self.d = self.model.n_params
        self.ga = ga
        self.rng = rng if rng is not None else np.random.default_rng(ga.seed)

        # Optimization is done over the FULL prior box (global search), but the
        # initial population is seeded inside the narrower `sample_box` so the
        # search starts in the physically sensible region (and converges fast),
        # exactly mirroring how the samplers initialize their chains.
        self.lo = np.array([b[0] for b in self.model.bounds], dtype=float)
        self.hi = np.array([b[1] for b in self.model.bounds], dtype=float)
        self.slo = np.array([b[0] for b in self.model.sample_box], dtype=float)
        self.shi = np.array([b[1] for b in self.model.sample_box], dtype=float)
        self.width = self.hi - self.lo

        self.n_elite = max(1, int(round(ga.elite_frac * ga.pop_size)))

    # ── fitness = the existing cosmological log-posterior ────────────────────
    def fitness(self, pop: np.ndarray) -> np.ndarray:
        """Vectorized fitness f(θ)=log P(θ|data) for a (P,d) population.

        This is the ONLY contact with the physics, and it is the EXACT same
        entry point the samplers use: `Posterior.log_prob_batch` (CC + Pantheon+
        likelihoods, prior, box mask). Out-of-box individuals get −inf and are
        naturally purged by selection.
        """
        return self.post.log_prob_batch(pop)

    @staticmethod
    def chi2_from_logp(logp: np.ndarray) -> np.ndarray:
        """χ²-like score for display only: −2·logP (flat prior ⇒ exact χ²).

        With a Gaussian prior this is χ² + prior penalty; we still label the
        fitness panel "χ²" because under the default flat prior it is the
        genuine χ², and the offset is irrelevant for monitoring convergence.
        """
        return -2.0 * logp

    # ── population initialization (classical; QGA overrides) ─────────────────
    def init_population(self) -> np.ndarray:
        """Uniform random population inside the narrow `sample_box`."""
        u = self.rng.uniform(0.0, 1.0, size=(self.ga.pop_size, self.d))
        return self.slo + u * (self.shi - self.slo)

    # ── selection (shared) ───────────────────────────────────────────────────
    def tournament_select(self, pop: np.ndarray, fit: np.ndarray,
                          n: int) -> np.ndarray:
        """Vectorized tournament selection returning `n` parents.

        Draws (n, k) random contenders and keeps, per row, the one with the
        highest fitness. Fully vectorized — no Python loop over individuals.
        """
        k = self.ga.tournament_k
        idx = self.rng.integers(0, len(pop), size=(n, k))
        cand_fit = fit[idx]                              # (n, k)
        winners = idx[np.arange(n), np.argmax(cand_fit, axis=1)]
        return pop[winners]

    # ── classical crossover (CGA; QGA may override) ──────────────────────────
    def crossover_classical(self, parents_a: np.ndarray,
                            parents_b: np.ndarray) -> np.ndarray:
        """Whole-arithmetic (blend) crossover, vectorized.

        For each mating pair, with probability `crossover_rate`, the child is
        a random convex blend  α·a + (1−α)·b  (BLX-style, α∼U(0,1) per gene).
        Otherwise the child copies parent A. Operates on the whole batch at
        once.
        """
        P, d = parents_a.shape
        do = self.rng.uniform(0, 1, size=P) < self.ga.crossover_rate
        alpha = self.rng.uniform(0.0, 1.0, size=(P, d))
        blended = alpha * parents_a + (1.0 - alpha) * parents_b
        child = np.where(do[:, None], blended, parents_a)
        return child

    # ── classical mutation (CGA; QGA may override) ───────────────────────────
    def mutate_classical(self, pop: np.ndarray) -> np.ndarray:
        """Per-gene Gaussian mutation scaled by each box width, vectorized."""
        P, d = pop.shape
        mask = self.rng.uniform(0, 1, size=(P, d)) < self.ga.mutation_rate
        sigma = self.ga.mutation_scale * self.width            # (d,)
        noise = self.rng.normal(0.0, 1.0, size=(P, d)) * sigma[None, :]
        out = pop + np.where(mask, noise, 0.0)
        return self._clip(out)

    def _clip(self, pop: np.ndarray) -> np.ndarray:
        """Reflect/clip individuals back into the hard prior box."""
        return np.clip(pop, self.lo, self.hi)


# =============================================================================
# 2.  THE EVOLUTION LOOP  (shared by CGA and QGA)
# =============================================================================


@dataclass
class GAResult:
    """Outcome of a genetic optimization, ready for the existing pipeline.

    Attributes:
        method: 'CGA' or 'QGA'.
        quantumness: 0–100 % (0 for CGA).
        theta_map: Best individual found (the MAP estimate).
        chi2_map: χ² at the MAP (from `fit_statistics`, locally refined).
        stats: Full `fit_statistics` dict at the MAP (χ², χ²_red, AIC, BIC…).
        final_pop: (P, d) last-generation population.
        final_fit: (P,) log-posterior fitness of the last population.
        final_weights: (P,) normalized fitness weights (for weighted corner).
        history: per-generation dict list (gen, best_chi2, mean_chi2, theta_best).
        elapsed: wall-clock seconds.
        config: the quantum-component config (QGA) or {} (CGA).
        label: human-readable label used in legends and the CSV.
        pop_history: [ANIM] poblacion completa por generacion, para animar la
            convergencia. Vacia si se corrio con record_population=False.
        fit_history: [ANIM] log-posterior por individuo y generacion.
    """
    method: str
    quantumness: float
    theta_map: np.ndarray
    chi2_map: float
    stats: dict
    final_pop: np.ndarray
    final_fit: np.ndarray
    final_weights: np.ndarray
    history: List[dict]
    elapsed: float
    config: dict = field(default_factory=dict)
    label: str = ''
    pop_history: List[np.ndarray] = field(default_factory=list)
    fit_history: List[np.ndarray] = field(default_factory=list)


def _fitness_weights(fit: np.ndarray) -> np.ndarray:
    """Turn log-posterior fitness into normalized, finite weights.

    Softmax-style:  w_i ∝ exp(f_i − max f).  −inf individuals get weight 0.
    These weights drive the fitness-weighted corner plot of the final
    population, so denser (better-fitting) regions dominate the contours.
    """
    f = np.asarray(fit, dtype=float)
    good = np.isfinite(f)
    w = np.zeros_like(f)
    if not np.any(good):
        return w
    fg = f[good]
    w[good] = np.exp(fg - fg.max())
    s = w.sum()
    return w / s if s > 0 else w


class GeneticEvolver(GeneticBase):
    """Adds the generation loop, live-GUI hook and logging to `GeneticBase`.

    The concrete CGA / QGA subclasses only provide the three operators
    (`do_init`, `do_crossover`, `do_mutate`); everything else — selection,
    elitism, bookkeeping, the live window and the headless logging — lives
    here and is identical for both, guaranteeing a controlled comparison.
    """

    method_name = 'GA'
    quantumness = 0.0
    config: dict = {}

    # Subclasses override these three to switch an operator to quantum.
    def do_init(self) -> np.ndarray:
        return self.init_population()

    def do_crossover(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return self.crossover_classical(a, b)

    def do_mutate(self, pop: np.ndarray) -> np.ndarray:
        return self.mutate_classical(pop)

    # ── the main evolutionary loop ───────────────────────────────────────────
    def evolve(self, live: bool = False, logger=None,
               record_population: bool = True,
               log_every: int = 10, gui=None,
               outdir: Optional[str] = None) -> GAResult:
        """Run the genetic optimization.

        Args:
            live: If True (interactive mode only), update a `LiveGA` window
                every generation. MUST be False in CLI/batch/HPC mode.
            logger: Optional logging.Logger; in headless mode the generational
                metrics are written here every `log_every` generations.
            log_every: Logging / console cadence in generations.
            gui: An already-constructed `LiveGA` (or None to build one when
                `live` is True).
            outdir: If given (interactive mode), a snapshot of the final live
                window is saved here as live_<method>_q<NN>.png.

        Returns:
            A `GAResult` with the MAP, the final population and the history.
        """
        t0 = time.time()
        ga = self.ga
        say = (logger.info if logger else print)
        say(f"[{self.method_name}] start | pop={ga.pop_size} "
            f"gens={ga.n_generations} | model={self.model.label} "
            f"| quantumness={self.quantumness:.0f}%")

        pop = self.do_init()
        fit = self.fitness(pop)
        history: List[dict] = []
        pop_history: List[np.ndarray] = []
        fit_history: List[np.ndarray] = []

        if live and gui is None:
            gui = LiveGA(self.model, self.method_name, self.quantumness)

        for gen in range(ga.n_generations):
            # 1) elitism — carry the best `n_elite` individuals unchanged
            order = np.argsort(fit)[::-1]
            elite = pop[order[:self.n_elite]].copy()

            # 2) parent selection (tournament), then crossover + mutation
            n_children = ga.pop_size - self.n_elite
            pa = self.tournament_select(pop, fit, n_children)
            pb = self.tournament_select(pop, fit, n_children)
            children = self.do_crossover(pa, pb)
            children = self.do_mutate(children)

            # 3) new generation = elite + children, re-evaluate fitness
            pop = np.vstack([elite, children])
            fit = self.fitness(pop)

            # 4) bookkeeping
            chi2 = self.chi2_from_logp(fit)
            finite = np.isfinite(chi2)
            best_i = int(np.argmax(fit))
            best_chi2 = float(chi2[best_i])
            mean_chi2 = float(np.mean(chi2[finite])) if np.any(finite) else np.nan
            theta_best = pop[best_i].copy()
            history.append(dict(gen=gen, best_chi2=best_chi2,
                                mean_chi2=mean_chi2,
                                theta_best=theta_best.tolist()))
            # [ANIM] Instantanea de la poblacion para la animacion. Es lo unico
            # que permite MOSTRAR la convergencia en vez de resumirla: el
            # historial solo guardaba el mejor individuo, y con eso no se ve
            # que la nube se contrae, que es la mitad del fenomeno.
            # Coste: P*d floats por generacion (200x4x120 ~ 0.8 MB), asi que se
            # graba siempre y se submuestrea al animar, no al grabar.
            if record_population:
                pop_history.append(pop.copy())
                fit_history.append(fit.copy())

            # 5) live GUI (interactive) OR periodic logging (headless)
            if live and gui is not None:
                gui.update(gen, pop, fit, theta_best, best_chi2, mean_chi2)
            if (gen % log_every == 0) or (gen == ga.n_generations - 1):
                msg = (f"[{self.method_name}] gen {gen:4d}/"
                       f"{ga.n_generations} | best χ²={best_chi2:8.3f} | "
                       f"mean χ²={mean_chi2:8.3f} | "
                       f"{fmt_theta(self.model, theta_best)}")
                if logger:
                    logger.info(msg)
                elif not live:
                    print(msg)

        # ── finalize: refine the MAP with the SAME fit_statistics as samplers
        best_i = int(np.argmax(fit))
        theta_map = pop[best_i].copy()
        stats = fit_statistics(self.post, theta_map, refine=True)
        theta_map = np.asarray(stats['theta_best'], dtype=float)
        weights = _fitness_weights(fit)
        elapsed = time.time() - t0

        say(f"[{self.method_name}] done in {elapsed:.1f}s | "
            f"MAP: {fmt_theta(self.model, theta_map)} | "
            f"χ²={stats['chi2']:.3f}  χ²_red={stats['chi2_red']:.3f}  "
            f"AIC={stats['AIC']:.2f}  BIC={stats['BIC']:.2f}")

        if live and gui is not None:
            snap = None
            if outdir is not None:
                snap = os.path.join(
                    outdir,
                    f"live_{self.method_name.lower()}_"
                    f"q{int(self.quantumness):03d}.png")
            gui.finalize(theta_map, stats['chi2'], save_path=snap)
            if snap is not None:
                say(f"[{self.method_name}] live snapshot: {snap}")

        return GAResult(
            method=self.method_name, quantumness=self.quantumness,
            theta_map=theta_map, chi2_map=float(stats['chi2']), stats=stats,
            final_pop=pop, final_fit=fit, final_weights=weights,
            history=history, elapsed=elapsed, config=dict(self.config),
            label=self._label(),
            pop_history=pop_history, fit_history=fit_history)

    def _label(self) -> str:
        """Legend/CSV label for this optimizer."""
        if self.method_name == 'CGA':
            return 'CGA'
        return f"QGA (q={self.quantumness:.0f}%)"


# =============================================================================
# 2b.  CGA — Classical Genetic Algorithm
# =============================================================================

class CGA(GeneticEvolver):
    """Classical Genetic Algorithm (all operators classical, NumPy-vectorized).

    Pure baseline: uniform-box init, tournament selection, blend crossover,
    Gaussian mutation, elitism. No Qiskit. This is also the 0 % rung of the
    QGA quantumness ladder, so QGA(0 %) must reproduce CGA exactly (checked
    in the self-test).
    """
    method_name = 'CGA'
    quantumness = 0.0
    config: dict = {}


# =============================================================================
# 3.  QGA — Quantum Genetic Algorithm (Qiskit), modular quantumness
# =============================================================================
#
#  ENCODING.  Each real parameter θ_j ∈ [lo_j, hi_j] is mapped to an integer
#  in {0, …, 2^n_bits − 1} (a fixed-point grid over its box) and stored in
#  `n_bits` qubits. An individual therefore occupies d·n_bits qubits. Decoding
#  inverts the map to a real value at the center of its grid cell.
#
#  The three quantum operators act ON THIS QUBIT REGISTER:
#    * q_init      : H^⊗(d·n_bits) → measure → uniform random integers, i.e. a
#                    genuine quantum-sampled initial population.
#    * q_mutation  : per gene-qubit, RY(φ) with a small amplitude set by
#                    `mutation_scale`; a qubit initialized to |b⟩ is rotated so
#                    that measuring it flips with probability sin²(φ/2). This is
#                    a coherent, amplitude-level bit-flip mutation.
#    * q_crossover : load two parents in two n_bits registers, apply a partial
#                    SWAP (SWAP^α, α = 1/2) between homologous gene-qubits and
#                    measure register A as the child. Consensus bits (a = b)
#                    pass through UNCHANGED; disagreeing bits are inherited
#                    from either parent with probability cos²(πα/2) via genuine
#                    two-qubit interference — a quantum-sampled per-bit uniform
#                    crossover. [C1 FIX: the earlier CX(B→A)+CRY layer produced
#                    child ≈ a XOR b, mapping (1,1)→0 with prob 0.85 and
#                    destroying consensus bits in converged populations.]
#
#  All circuits are TRANSPILED ONCE at construction (handover lesson: per-step
#  transpilation was the main QMCMC bottleneck). Operators are evaluated in
#  BATCHED Aer jobs (one job per generation per operator, not one per
#  individual), mirroring the batched-submission pattern of the QMCMC engine.
#
#  Qiskit is imported lazily so the CGA remains usable on machines without it.

_QISKIT_OK = True
try:
    from qiskit import QuantumCircuit, transpile
    from qiskit.circuit import ParameterVector
    from qiskit.circuit.library import UnitaryGate
    from qiskit_aer import AerSimulator
except Exception:                                       # pragma: no cover
    _QISKIT_OK = False


class QGA(GeneticEvolver):
    """Quantum Genetic Algorithm with independently switchable quantum operators.

    Args:
        post: cosmological `Posterior`.
        ga: `GAConfig` hyper-parameters.
        config: dict with booleans q_init / q_mutation / q_crossover.
        n_bits: qubits per parameter (grid resolution = 2^n_bits per axis).
        rng: shared NumPy Generator.
        shots: Aer shots per circuit (one circuit per individual per operator).

    Any operator left False falls back to the CGA classical implementation, so
    every quantumness level is a controlled, one-component-at-a-time change.
    """

    method_name = 'QGA'

    def __init__(self, post: Posterior, ga: GAConfig, config: dict,
                 n_bits: int = 6, rng: Optional[np.random.Generator] = None,
                 shots: int = 1, crossover_alpha: float = 0.5):
        super().__init__(post, ga, rng)
        if not _QISKIT_OK:
            raise RuntimeError(
                "QGA requires Qiskit + qiskit-aer. Install them, or run with "
                "--methods cga for the classical optimizer.")
        self.config = dict(config)
        self.quantumness = compute_qga_quantumness(self.config)
        self.n_bits = int(n_bits)
        self.shots = int(shots)
        # [C1 FIX] Partial-swap exponent of the quantum crossover: with SWAP^α
        # a disagreeing bit is inherited from parent A with prob cos²(πα/2)
        # (α = 0.5 → uniform 50/50; α → 0 keeps A; α → 1 takes B). Consensus
        # bits are ALWAYS preserved regardless of α.
        self.crossover_alpha = float(np.clip(crossover_alpha, 0.0, 1.0))
        self._levels = 2 ** self.n_bits                  # grid points per axis
        # [NOISE] Sin ruido esto es exactamente `make_simulator('statevector',
        # prefer_gpu=USE_GPU)`, la construccion previa al eje. Con ruido pasa a
        # matriz de densidad con el modelo completo; el QGA no necesita ningun
        # otro cambio porque ya lee por medicion.
        self.sim = _qga_sim()
        self.noise_label = NOISE.label
        # [M3 FIX] Aer measurement seeds are now DERIVED from the run seed
        # (GAConfig.seed) plus a per-call counter. Previously the quantum
        # operators ran with Aer's own random seed, so QGA runs were NOT
        # reproducible across executions even with a fixed ga.seed — the
        # classical parts repeated, the quantum measurements did not.
        self._aer_call = 0

        # ── pre-build & transpile the per-operator circuit templates ONCE ────
        self._build_circuits()

    def _sim_seed(self) -> int:
        """Deterministic per-call simulator seed: f(ga.seed, call counter)."""
        self._aer_call += 1
        return (int(self.ga.seed) * 100003 + self._aer_call) % (2**31 - 1)

    # ── fixed-point encode/decode between θ and integer gene grids ───────────
    def _encode(self, pop: np.ndarray) -> np.ndarray:
        """Map a (P,d) real population to (P,d) integer genes in [0, 2^n−1]."""
        frac = (pop - self.lo) / np.where(self.width > 0, self.width, 1.0)
        frac = np.clip(frac, 0.0, 1.0 - 1e-12)
        return np.floor(frac * self._levels).astype(int)

    def _decode(self, genes: np.ndarray) -> np.ndarray:
        """Map (P,d) integer genes back to real θ at each grid-cell center."""
        frac = (genes.astype(float) + 0.5) / self._levels
        return self.lo + frac * self.width

    @staticmethod
    def _int_to_bits(vals: np.ndarray, n_bits: int) -> np.ndarray:
        """Vectorized integer → (…, n_bits) bit array (MSB first)."""
        shifts = np.arange(n_bits - 1, -1, -1)
        return ((vals[..., None] >> shifts) & 1).astype(int)

    @staticmethod
    def _bits_to_int(bits: np.ndarray) -> np.ndarray:
        """Vectorized (…, n_bits) bit array → integer (MSB first)."""
        n_bits = bits.shape[-1]
        weights = (1 << np.arange(n_bits - 1, -1, -1))
        return np.sum(bits * weights, axis=-1).astype(int)

    # ── circuit templates (built & transpiled ONCE) ─────────────────────────
    def _build_circuits(self):
        """Construct and transpile the init / mutation / crossover templates.

        [A1] CRITICAL: the templates are PARAMETRIZED so that the per-
        individual gene bits are carried by BOUND ANGLES, not by structural
        X gates. That lets every individual reuse the SAME transpiled circuit
        via `assign_parameters` — the hot loop never calls `transpile` again.
        The previous version built one fresh circuit per (individual, gene)
        with X-gate state-prep and transpiled the WHOLE list every generation
        (e.g. pop=200, d=4 → 800 transpilations/generation), which dominated
        the QGA runtime and broke the project's own "transpile once" rule.
        """
        nb = self.n_bits

        # (a) Quantum initialization: d·nb qubits in uniform superposition.
        n_init = self.d * nb
        qc_init = QuantumCircuit(n_init)
        qc_init.h(range(n_init))
        qc_init.measure_all()
        self._qc_init = transpile(qc_init, self.sim)

        # (b) Quantum mutation template (nb qubits, nb angle parameters).
        #     State-prep is folded into the angle: a qubit representing bit b
        #     with per-qubit flip probability p must measure 1 with probability
        #     P1 = b·(1−p) + (1−b)·p, achieved by RY(2·arcsin(√P1))|0⟩. So the
        #     gene bits enter ONLY through the bound angles — no X gates, one
        #     transpiled template for all individuals.
        phi_m = ParameterVector('m', nb)
        qc_mut = QuantumCircuit(nb)
        for q in range(nb):
            qc_mut.ry(phi_m[q], q)
        qc_mut.measure_all()
        self._qc_mut_t = transpile(qc_mut, self.sim)     # transpiled ONCE
        self._phi_m = phi_m

        # (c) Quantum crossover template (2·nb qubits). Parent bits are loaded
        #     by per-qubit prep angles (RY(π)|0⟩ = |1⟩, RY(0)|0⟩ = |0⟩), then a
        #     per-qubit PARTIAL SWAP (SWAP^α) mixes the homologous gene-qubits
        #     of the two registers. Measuring register A yields the child.
        #
        #     [C1 FIX — consensus-preserving crossover] The previous layer was
        #     CX(B→A) + CRY(φ, B→A), whose measured child bit was ≈ a XOR b:
        #     verified truth table P(child=1) = {(0,0)→0, (0,1)→0.85,
        #     (1,0)→1, (1,1)→0.146}. Mapping (1,1)→0 destroys exactly the
        #     consensus bits a converged population agrees on — the opposite of
        #     what a crossover must do. SWAP^α acts as identity on the |aa⟩
        #     subspace and rotates only within span{|01⟩, |10⟩}, so the truth
        #     table becomes:
        #         (0,0) → 0 always,   (1,1) → 1 always   (consensus preserved)
        #         (a≠b) → child = a with prob cos²(πα/2), = b otherwise,
        #     with genuine two-qubit interference on the disagreeing bits
        #     (the SWAP^α amplitudes (1±e^{iπα})/2 are complex — this is not a
        #     classical coin flip on amplitudes). α = 1/2 gives the quantum
        #     analogue of a per-bit UNIFORM crossover (50/50 inheritance);
        #     α → 0 keeps parent A, α → 1 swaps in parent B. The operator
        #     remains an ALGORITHMIC substitution: the classical baseline is a
        #     real-valued blend crossover, this is a bitwise quantum-sampled
        #     uniform crossover.
        #
        #     The fixed SWAP^α unitary lives inside the ONE transpiled
        #     template; parent pairs are still pure parameter bindings on the
        #     prep angles (transpile-once pattern [A1] unchanged).
        prep_a = ParameterVector('pa', nb)
        prep_b = ParameterVector('pb', nb)
        alpha = self.crossover_alpha
        e = np.exp(1j * np.pi * alpha)
        sqrt_swap = np.array([[1, 0, 0, 0],
                              [0, (1 + e) / 2, (1 - e) / 2, 0],
                              [0, (1 - e) / 2, (1 + e) / 2, 0],
                              [0, 0, 0, 1]], dtype=complex)
        pswap = UnitaryGate(sqrt_swap, label=f'SWAP^{alpha:g}')
        qc_cx = QuantumCircuit(2 * nb)
        for q in range(nb):
            qc_cx.ry(prep_a[q], q)                 # load parent A bit on qubit q
            qc_cx.ry(prep_b[q], nb + q)            # load parent B bit
        for q in range(nb):
            qc_cx.append(pswap, [q, nb + q])       # partial swap A[q] <-> B[q]
        qc_cx.measure_all()
        self._qc_cx_t = transpile(qc_cx, self.sim)       # transpiled ONCE
        self._prep_a, self._prep_b = prep_a, prep_b

    # ── operator 1: quantum initialization ──────────────────────────────────
    def do_init(self) -> np.ndarray:
        """Quantum-sampled initial population (or classical if disabled)."""
        if not self.config.get('q_init', False):
            return self.init_population()

        nb, P = self.n_bits, self.ga.pop_size
        # One shot per individual gives P independent measurement bitstrings.
        circ = self._qc_init
        res = self.sim.run([circ], shots=P, memory=True,
                           seed_simulator=self._sim_seed()).result()  # [M3]
        mem = res.get_memory(0)                # list of P bitstrings
        # Qiskit bitstrings are little-endian over the full register; parse
        # each into d genes of nb bits.
        genes = np.empty((P, self.d), dtype=int)
        for i, bs in enumerate(mem):
            bs = bs.replace(' ', '')
            bits = np.array([int(c) for c in bs[::-1]], dtype=int)  # qubit order
            for j in range(self.d):
                seg = bits[j * nb:(j + 1) * nb]
                genes[i, j] = self._bits_to_int(seg[::-1])          # MSB first
        # Quantum init is uniform over the FULL box (true superposition), then
        # nudged into the sample_box region by rejecting nothing — the global
        # search will pull good individuals in within a few generations.
        return self._decode(genes)

    # ── operator 2: quantum mutation ─────────────────────────────────────────
    def do_mutate(self, pop: np.ndarray) -> np.ndarray:
        """Coherent RY mutation on gene-qubits (or classical if disabled)."""
        if not self.config.get('q_mutation', False):
            return self.mutate_classical(pop)

        nb = self.n_bits
        genes = self._encode(pop)              # (P, d)
        P = len(pop)

        # [H6 FIX — same gating, first-moment-calibrated strength] The old
        # code (a) rotated EVERY gene of EVERY individual, ignoring
        # `mutation_rate` (the classical operator mutates a gene with prob
        # mutation_rate only), and (b) set p_flip = mutation_scale directly,
        # so the expected displacement grew with n_bits and an MSB flip
        # jumped half the box — the docstring's "comparable in strength" did
        # not hold, and the mutation rung of the ladder changed operator AND
        # effective intensity at once.
        #
        # Now: (a) genes are gated by `mutation_rate` with the SAME shared
        # classical rng stream the CGA uses (only gated genes run circuits —
        # also ~5x fewer circuits per generation at rate 0.2); (b) p_flip is
        # calibrated so the FIRST MOMENT matches the classical Gaussian
        # kick: for independent per-bit flips with prob p, E|Δθ| ≈
        # p·width·(1 − 2^{−nb}), while the classical operator gives
        # E|Δθ| = σ·√(2/π) with σ = mutation_scale·width. Hence
        #     p_flip = mutation_scale·√(2/π) / (1 − 2^{−nb}).
        # The DISTRIBUTION still differs (bit flips are heavy-tailed — that
        # is the ALGORITHMIC treatment being tested); only the mean absolute
        # step is matched so the rung isolates the operator, not its gain.
        mask = self.rng.uniform(0, 1, size=(P, self.d)) < self.ga.mutation_rate
        if not np.any(mask):
            return self._clip(pop.copy())
        p_flip = self.ga.mutation_scale * np.sqrt(2.0 / np.pi)
        p_flip = float(np.clip(p_flip / (1.0 - 2.0**(-nb)), 1e-3, 0.5))

        # [A1] Build the (B, nb) angle-binding array for the GATED genes
        # only. For each (individual, gene) the nb angles encode the gene
        # bits with flip probability p_flip: a bit b maps to RY angle
        # 2·arcsin(√(b·(1−p)+(1−b)·p)). No X gates, no per-call
        # transpilation — we bind on the ONE pre-transpiled template and run
        # all bindings in a single Aer job.
        sel = np.argwhere(mask)                             # (B, 2) -> (i, j)
        gene_flat = genes[sel[:, 0], sel[:, 1]]             # (B,)
        bits = self._int_to_bits(gene_flat, nb)             # (B, nb), MSB-first
        # qubit q holds bit (nb-1-q) under the MSB-first/qubit-index convention
        bits_q = bits[:, ::-1]                              # (B, nb) per qubit
        P1 = bits_q * (1.0 - p_flip) + (1 - bits_q) * p_flip
        angles = 2.0 * np.arcsin(np.sqrt(P1))               # (B, nb)

        tmpl = self._qc_mut_t
        param_order = list(self._phi_m)
        circuits = [tmpl.assign_parameters(
                        {param_order[q]: float(angles[k, q])
                         for q in range(nb)})
                    for k in range(angles.shape[0])]
        res = self.sim.run(circuits, shots=1, memory=True,
                           seed_simulator=self._sim_seed()).result()  # [M3]

        new_genes = genes.copy()
        for k in range(gene_flat.shape[0]):
            i, j = sel[k]                                          # [H6] gated
            bs = res.get_memory(k)[0].replace(' ', '')
            mb = np.array([int(c) for c in bs[::-1]], dtype=int)   # qubit order
            new_genes[i, j] = self._bits_to_int(mb[::-1])          # MSB first
        return self._clip(self._decode(new_genes))

    # ── operator 3: quantum crossover ────────────────────────────────────────
    def do_crossover(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Entangling crossover of parent pairs (or classical if disabled)."""
        if not self.config.get('q_crossover', False):
            return self.crossover_classical(a, b)

        nb = self.n_bits
        ga_, gb = self._encode(a), self._encode(b)     # (P, d) each
        P = len(a)
        do = self.rng.uniform(0, 1, size=P) < self.ga.crossover_rate

        # [C1 FIX] The per-qubit mixing is the fixed SWAP^α unitary baked into
        # the transpiled template (see _build_circuits): consensus bits pass
        # through unchanged, disagreeing bits are inherited from A/B with
        # probability cos²(πα/2) / sin²(πα/2) via genuine two-qubit
        # interference. No per-pair interference angle is needed anymore.

        # [A1] Parent bits are loaded by PREP ANGLES (RY(π)→|1⟩, RY(0)→|0⟩)
        # bound on the ONE pre-transpiled template, instead of structural X
        # gates that forced a per-pair retranspile. Collect the (i,j) pairs
        # that recombine, build their bindings, run in a single Aer job.
        idx_pairs = [(i, j) for i in range(P) if do[i] for j in range(self.d)]
        child = a.copy()                                # default: parent A
        if idx_pairs:
            pa, pb = self._prep_a, self._prep_b
            tmpl = self._qc_cx_t
            circuits = []
            for (i, j) in idx_pairs:
                bits_a = self._int_to_bits(np.array(ga_[i, j]), nb)[::-1]  # per qubit
                bits_b = self._int_to_bits(np.array(gb[i, j]), nb)[::-1]
                binding = {}
                for q in range(nb):
                    binding[pa[q]] = float(np.pi) if bits_a[q] == 1 else 0.0
                    binding[pb[q]] = float(np.pi) if bits_b[q] == 1 else 0.0
                circuits.append(tmpl.assign_parameters(binding))
            res = self.sim.run(circuits, shots=1, memory=True,
                               seed_simulator=self._sim_seed()).result()  # [M3]
            new_genes = ga_.copy()
            for k, (i, j) in enumerate(idx_pairs):
                bs = res.get_memory(k)[0].replace(' ', '')
                mb = np.array([int(c) for c in bs[::-1]], dtype=int)
                segA = mb[:nb]                          # register A = qubits 0..nb-1
                new_genes[i, j] = self._bits_to_int(segA[::-1])
            child = self._decode(new_genes)
        return self._clip(child)


# =============================================================================
# 4.  LIVE GUI  (interactive mode ONLY — never created in batch/HPC mode)
# =============================================================================


class LiveGA:
    """Real-time two-panel Matplotlib window for the genetic evolution.

    LEFT  — Phase-space panel: a scatter of the whole population in the first
            two parameters (Ωm, H0), updated every generation, with the best
            individual and the running MAP highlighted; the population is seen
            converging toward the optimum.
    RIGHT — Fitness panel: best χ² and mean χ² of the population vs generation.

    A dynamic suptitle reports the generation, the best χ² and the current
    physical values. The window uses Matplotlib's interactive mode
    (`plt.ion()`), so it refreshes in place without blocking the evolution.

    IMPORTANT: this class is only instantiated when `live=True`, which the CLI
    layer forces to False whenever the script is run with arguments (batch/HPC).
    """

    def __init__(self, model, method_name: str, quantumness: float,
                 pause: float = 0.05):
        self.model = model
        self.method = method_name
        self.qpct = quantumness
        self.pause = float(pause)
        self.closed = False
        self.p0_name = model.param_latex[0]
        self.p1_name = model.param_latex[1] if model.n_params > 1 else 'index'
        self.gens: List[int] = []
        self.best_hist: List[float] = []
        self.mean_hist: List[float] = []

        plt.ion()                                  # interactive mode ON
        self.fig, (self.ax_ps, self.ax_fit) = plt.subplots(
            1, 2, figsize=(13, 5.4))
        # Detect the user closing the window so we can stop refreshing it.
        self.fig.canvas.mpl_connect('close_event', self._on_close)
        color = C_GENETIC if quantumness == 0 else C_GENETIC2
        self._color = color

        # ── phase-space panel ────────────────────────────────────────────────
        self.ax_ps.set_xlabel(self.p0_name, fontsize=12)
        self.ax_ps.set_ylabel(self.p1_name, fontsize=12)
        self.ax_ps.set_title('Phase space — population convergence',
                             fontsize=11)
        self.ax_ps.grid(True, alpha=0.3)
        b = model.bounds
        self.ax_ps.set_xlim(b[0]); self.ax_ps.set_ylim(b[1])
        # Planck fiducial cross-hair for reference
        self.ax_ps.axvline(model.fiducial[0], color='k', ls='--', lw=1,
                           alpha=0.5)
        self.ax_ps.axhline(model.fiducial[1], color='k', ls='--', lw=1,
                           alpha=0.5)
        # Population colored by fitness (viridis): brighter = better fit, so the
        # convergence toward the high-fitness MAP is visually obvious.
        self._scat = self.ax_ps.scatter([], [], s=26, c=[], cmap='viridis',
                                        alpha=0.7, edgecolors='none',
                                        label='Population')
        self._best_pt, = self.ax_ps.plot([], [], '*', ms=20, color='gold',
                                        mec='k', mew=1.2, label='Best (MAP)',
                                        zorder=5)
        self.ax_ps.legend(loc='upper right', fontsize=9)

        # ── fitness panel ────────────────────────────────────────────────────
        self.ax_fit.set_xlabel('Generation', fontsize=12)
        self.ax_fit.set_ylabel(r'$\chi^2$', fontsize=12)
        self.ax_fit.set_title('Fitness evolution', fontsize=11)
        self.ax_fit.grid(True, alpha=0.3)
        self._best_line, = self.ax_fit.plot([], [], '-', color=color, lw=2,
                                           label=r'best $\chi^2$')
        self._mean_line, = self.ax_fit.plot([], [], '--', color='gray', lw=1.5,
                                           label=r'mean $\chi^2$')
        self.ax_fit.legend(loc='upper right', fontsize=9)

        self._suptitle = self.fig.suptitle('', fontsize=12, fontweight='bold')
        self.fig.tight_layout(rect=[0, 0, 1, 0.94])
        # First draw + flush so the window actually appears before gen 0.
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.show(block=False)
        plt.pause(self.pause)

    def _on_close(self, _event):
        """Mark the window as closed (user clicked the X)."""
        self.closed = True

    def update(self, gen: int, pop: np.ndarray, fit: np.ndarray,
               theta_best: np.ndarray, best_chi2: float, mean_chi2: float):
        """Refresh both panels with the current generation (no-op if closed)."""
        if self.closed:
            return
        finite = np.isfinite(fit)
        P = pop[finite]
        c = fit[finite]
        if P.shape[1] >= 2:
            offs = P[:, :2]
        else:
            offs = np.c_[P[:, 0], np.zeros(len(P))]
        self._scat.set_offsets(offs)
        # Color by fitness; guard against an all-equal array (constant color).
        if len(c) and np.ptp(c) > 0:
            self._scat.set_array(c)
            self._scat.set_clim(np.min(c), np.max(c))
        self._best_pt.set_data([theta_best[0]],
                               [theta_best[1] if self.model.n_params > 1 else 0])

        self.gens.append(gen)
        self.best_hist.append(best_chi2)
        self.mean_hist.append(mean_chi2)
        self._best_line.set_data(self.gens, self.best_hist)
        self._mean_line.set_data(self.gens, self.mean_hist)
        self.ax_fit.relim(); self.ax_fit.autoscale_view()

        vals = "  ".join(f"{n}={v:.4f}"
                         for n, v in zip(self.model.param_names, theta_best))
        qtag = (f"{self.method}" if self.qpct == 0
                else f"{self.method} (q={self.qpct:.0f}%)")
        self._suptitle.set_text(
            f"{qtag} — {self.model.label} | gen {gen} | "
            f"best χ²={best_chi2:.3f} | {vals}")
        try:
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
            plt.pause(self.pause)
        except Exception:
            # Backend hiccup or window gone mid-draw: stop updating gracefully.
            self.closed = True

    def finalize(self, theta_map: np.ndarray, chi2_map: float,
                 save_path: Optional[str] = None):
        """Mark the final MAP, optionally save a snapshot, keep window open."""
        if not self.closed:
            self._best_pt.set_data(
                [theta_map[0]],
                [theta_map[1] if self.model.n_params > 1 else 0])
            vals = "  ".join(f"{n}={v:.4f}"
                             for n, v in zip(self.model.param_names, theta_map))
            self.ax_ps.set_title(
                f'Phase space — converged MAP\n{vals}  χ²={chi2_map:.3f}',
                fontsize=10)
            try:
                self.fig.canvas.draw_idle()
                self.fig.canvas.flush_events()
                plt.pause(self.pause)
            except Exception:
                pass
        # A snapshot of the final live figure is always saved (even headless
        # would work, but this path runs only in interactive mode).
        if save_path is not None:
            try:
                self.fig.savefig(save_path, dpi=150, bbox_inches='tight')
            except Exception:
                pass


# =============================================================================
# 5.  RESULT INTEGRATION  (corner plots, overlay, fitness curve, CSV)
# =============================================================================
#
#  These functions plug the genetic results into the EXISTING pipeline:
#    * fitness-weighted corner of the final population,
#    * overlay of the genetic MAP + spread on the MCMC/VI corner plots
#      (reusing cosmo_modular_quantum.plot_corner_multi),
#    * the standalone fitness-vs-generation curve,
#    * a row in `resultados_config.csv` matching the shared schema.


def plot_population_corner(result: GAResult, model, outdir: str,
                           tag: Optional[str] = None) -> str:
    """Fitness-weighted corner plot of the final genetic population.

    Reuses `cosmo_modular_quantum.plot_corner_multi` so the style matches the
    sampler figures exactly. The single dataset is the last-generation
    population, weighted by the softmax fitness weights, with the MAP overplotted
    as a marker via corner's truth lines.
    """
    os.makedirs(outdir, exist_ok=True)
    tag = tag or f"{model.name}_{result.method.lower()}_q{int(result.quantumness):03d}"
    color = C_GENETIC if result.quantumness == 0 else C_GENETIC2
    finite = np.isfinite(result.final_fit)
    pop = result.final_pop[finite]
    w = result.final_weights[finite]

    f = _cmq().plot_corner_multi(
        datasets=[pop], colors=[color],
        labels=[f"{result.label} final population"],
        model=model, outdir=outdir, tag=tag,
        title=(f"{model.label} — {result.label} final population "
               f"(fitness-weighted)\nMAP: "
               + "  ".join(f"{n}={v:.4f}"
                           for n, v in zip(model.param_names, result.theta_map))
               + f"   χ²={result.chi2_map:.3f}"),
        weights_list=[w])
    return f


def plot_genetic_convergence(results, model, outdir: str,
                             tag: Optional[str] = None) -> str:
    """Convergencia y estimacion final del genetico, para TODOS los rungs.

    [PLOT-GA] Sustituye al corner plot de la poblacion final.

    Por que se cambio: un algoritmo genetico CONVERGE a un punto. Con elitismo
    y unas decenas de generaciones, la poblacion final se concentra en una o
    dos celdas de la rejilla, asi que su corner plot es literalmente un punto
    con unos pocos pixeles alrededor. Gasta una figura entera en no mostrar
    nada, y sobre todo no muestra lo unico que distingue un rung de otro: COMO
    llego ahi y DONDE aterrizo respecto a los demas.

    Un corner plot responde "que forma tiene el posterior", que es la pregunta
    correcta para MCMC/VI, donde la nube de muestras ES el resultado. El
    resultado de un GA es un punto y una trayectoria, asi que la figura
    correcta es otra:

      Fila 1 — una traza por parametro: theta_best contra generacion, un color
        por rung. Aqui se ve la velocidad de convergencia, si un rung se queda
        atascado, y si dos rungs convergen al mismo sitio por caminos
        distintos. Es la informacion que el corner destruye al mostrar solo el
        estado final.

      Fila 2 — un punto-y-bigote por parametro: MAP final con la dispersion de
        la poblacion final como barra. Compara todos los rungs en un vistazo y
        deja ver si las diferencias entre ellos son mayores o menores que la
        dispersion interna de cada uno — que es justo lo que hay que saber
        antes de afirmar que un rung difiere de otro.

    La linea punteada marca el valor fiducial de referencia del modelo.

    Args:
        results: iterable de `GAResult` (CGA y los rungs del QGA).
        model: `CosmoModel` activo.
        outdir: carpeta de salida.
        tag: sufijo del nombre de archivo.

    Returns:
        Ruta del PNG generado.
    """
    os.makedirs(outdir, exist_ok=True)
    results = [r for r in results if getattr(r, 'history', None)]
    if not results:
        return ""
    tag = tag or f"{model.name}_genetic_convergence"
    d = model.n_params
    names = model.param_names
    latex = getattr(model, 'param_latex', names)

    # Un color y un marcador por rung: la identidad no descansa solo en el
    # color (requisito de accesibilidad y de impresion en blanco y negro).
    palette = [C_GENETIC, C_GENETIC2, C_QUANTUM2, C_CLASSICAL2, C_QUANTUM]
    markers = ['o', 's', '^', 'D', 'v']
    styles = {}
    for i, r in enumerate(sorted(results, key=lambda x: x.quantumness)):
        styles[r.label] = (palette[i % len(palette)], markers[i % len(markers)])

    fig, axes = plt.subplots(2, d, figsize=(4.6 * d, 8.2), squeeze=False)
    order = sorted(results, key=lambda x: x.quantumness)

    # ── fila 1: trayectoria de theta_best ────────────────────────────────
    # [PLOT-GA] QGA(0%) reproduce al CGA bit a bit por diseno (es la celda
    # faithful del genetico), asi que sus curvas se solapan EXACTAMENTE y la
    # de abajo desaparece. Sin avisarlo parece una serie perdida, cuando en
    # realidad es el resultado. Se detecta y se anota en la leyenda, y se
    # dibuja con grosor decreciente para que la de abajo asome como halo.
    def _traj(r):
        return np.array([h['theta_best'] for h in r.history], float)

    coincide = {}
    for a in range(len(order)):
        for b in range(a + 1, len(order)):
            ta, tb = _traj(order[a]), _traj(order[b])
            if ta.shape == tb.shape and np.allclose(ta, tb, atol=0, rtol=0):
                coincide.setdefault(order[b].label, []).append(order[a].label)

    def _label(r):
        same = coincide.get(r.label)
        return f"{r.label}  ≡ {same[0]}" if same else r.label

    for j in range(d):
        ax = axes[0][j]
        todos = []
        for i, r in enumerate(order):
            col, mk = styles[r.label]
            gens = [h['gen'] for h in r.history]
            vals = [h['theta_best'][j] for h in r.history]
            todos.extend(vals)
            ax.plot(gens, vals, '-', color=col,
                    lw=max(1.4, 4.0 - 0.7 * i), label=_label(r) if j == 0 else None,
                    marker=mk, markevery=max(1, len(gens) // 8), ms=5,
                    alpha=0.95, zorder=2 + i)
        # [B-FIDSCALE] El fiducial NO puede fijar la escala del eje. En
        # lcdm/CC+BAO+Pantheon los rungs viven entre 0.2763 y 0.2775 y el
        # fiducial de Planck esta en 0.3111: dejarlo dentro de los limites
        # estiraba el eje a 0.275-0.311 y las diferencias entre rungs — que
        # son el contenido de la figura — quedaban en un pixel. Se dibuja la
        # linea igual, pero los limites los ponen las trayectorias; si el
        # fiducial cae fuera, se anota en el borde en vez de deformar el eje.
        if todos:
            lo, hi = min(todos), max(todos)
            pad = max((hi - lo) * 0.25, abs(hi) * 1e-4, 1e-12)
            ax.set_ylim(lo - pad, hi + pad)
        fid = getattr(model, 'fiducial', None)
        if fid is not None and j < len(fid):
            y0, y1 = ax.get_ylim()
            if y0 <= fid[j] <= y1:
                ax.axhline(fid[j], ls=':', color='0.45', lw=1.2, zorder=0)
            else:
                arriba = fid[j] > y1
                ax.annotate(f"fiducial = {fid[j]:.4g} "
                            f"({'arriba' if arriba else 'abajo'} del rango)",
                            xy=(0.99, 0.985 if arriba else 0.015),
                            xycoords='axes fraction', ha='right',
                            va='top' if arriba else 'bottom',
                            fontsize=8, color='0.35')
        ax.set_xlabel('Generation', fontsize=11)
        # param_latex ya viene con sus delimitadores $...$: envolverlo otra
        # vez rompe el parser de mathtext.
        ax.set_ylabel(latex[j] if j < len(latex) else names[j], fontsize=12)
        ax.grid(True, alpha=0.25, lw=0.6)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        if j == 0:
            ax.legend(fontsize=8, frameon=False)

    # ── fila 2: estimacion final, LA MISMA que reporta el CSV ───────────
    # [B-GAPLOT] Esta fila dibujaba `theta_map` con la desviacion SIN PESOS de
    # la poblacion final, mientras que la fila del CSV (`_ga_side`) reporta la
    # media y la desviacion PONDERADAS POR FITNESS. Son dos cosas distintas y
    # la diferencia no es cosmetica: en lcdm/nb6 de la campana 2026-09-03 el
    # CSV daba Om = 0.276289 +- 0.001407 y la figura pintaba la misma fila con
    # una barra de +-0.0165, doce veces mas ancha. Quien comparara tabla y
    # figura veria una contradiccion y no sabria a cual creerle.
    #
    # Se unifica en la version ponderada, que es la que va al CSV y a la tesis:
    # la poblacion final de un GA con elitismo tiene una cola de individuos
    # malos que la desviacion sin pesos cuenta igual que a los buenos, asi que
    # mide el ancho de la poblacion, no la incertidumbre de la estimacion.
    #
    # El MAP se conserva como marcador abierto encima, porque es el punto que
    # define chi2/AIC/BIC y no coincide exactamente con la media ponderada.
    #
    # ADVERTENCIA que va en el pie de figura: esta barra es dispersion de
    # convergencia del optimizador, NO un intervalo de credibilidad. Comparar
    # un rung con otro en unidades de esta barra sobreestima cualquier
    # discrepancia — para eso esta sigma del MCMC.
    ypos = np.arange(len(order))
    for j in range(d):
        ax = axes[1][j]
        extremos: List[float] = []
        for i, r in enumerate(order):
            col, mk = styles[r.label]
            finite = np.isfinite(r.final_fit)
            pop = r.final_pop[finite]
            w = np.asarray(r.final_weights, float)[finite]
            if len(pop) == 0:
                mu_j, spread = float(r.theta_map[j]), 0.0
            else:
                if not np.isfinite(w).all() or w.sum() <= 0:
                    w = np.ones(len(pop)) / len(pop)
                mu_j = float(np.average(pop[:, j], weights=w))
                spread = float(np.sqrt(
                    np.average((pop[:, j] - mu_j) ** 2, weights=w)))
            ax.errorbar(mu_j, ypos[i], xerr=spread, fmt=mk,
                        color=col, ms=9, capsize=4, lw=2, mec='white', mew=1.2)
            ax.plot(r.theta_map[j], ypos[i], marker='|', ms=13, mew=1.6,
                    color=col, alpha=0.9, zorder=5,
                    label='MAP' if (i == 0 and j == 0) else None)
            extremos.extend([mu_j - spread, mu_j + spread,
                             float(r.theta_map[j])])
        # [B-FIDSCALE] Mismo criterio que arriba: el fiducial se dibuja si cae
        # dentro, y se anota en el borde si no. Que Planck quede fuera del
        # rango del genetico ES un resultado (tension en Om con este dataset),
        # pero no puede costar la legibilidad de la comparacion entre rungs.
        if extremos:
            lo, hi = min(extremos), max(extremos)
            pad = max((hi - lo) * 0.30, abs(hi) * 1e-4, 1e-12)
            ax.set_xlim(lo - pad, hi + pad)
        fid = getattr(model, 'fiducial', None)
        if fid is not None and j < len(fid):
            x0, x1 = ax.get_xlim()
            if x0 <= fid[j] <= x1:
                ax.axvline(fid[j], ls=':', color='0.45', lw=1.2, zorder=0)
            else:
                der = fid[j] > x1
                ax.annotate(f"fiducial = {fid[j]:.4g} "
                            f"({'>' if der else '<'} rango)",
                            xy=(0.985 if der else 0.015, 0.02),
                            xycoords='axes fraction',
                            ha='right' if der else 'left', va='bottom',
                            fontsize=8, color='0.35')
        ax.set_yticks(ypos)
        ax.set_yticklabels([_label(r) for r in order] if j == 0 else [],
                           fontsize=8)
        if j > 0:
            # sin etiquetas no hacen falta las marcas: dejaban un guion suelto
            ax.tick_params(axis='y', length=0)
        ax.set_ylim(-0.6, len(order) - 0.4)
        ax.invert_yaxis()
        ax.set_xlabel(latex[j] if j < len(latex) else names[j], fontsize=12)
        ax.grid(True, axis='x', alpha=0.25, lw=0.6)
        for s in ('top', 'right', 'left'):
            ax.spines[s].set_visible(False)

    axes[0][0].set_title('', loc='left')
    fig.suptitle(
        f'{model.label} — genetic convergence and final estimate\n'
        f'top: best individual per generation   ·   dotted line: fiducial\n'
        f'bottom: fitness-weighted mean ± weighted spread (same numbers as '
        f'the CSV); tick = MAP.  This bar is optimizer convergence spread, '
        f'NOT a credible interval.',
        fontsize=11, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path = os.path.join(outdir, f"genetic_convergence_{tag}.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    fig.savefig(path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close(fig)
    return path


def _coincidence_label(result, others) -> str:
    """Etiqueta con una marca si otro rung produjo la MISMA trayectoria.

    QGA(0%) reproduce al CGA bit a bit: es la celda faithful del genetico y su
    identidad ES el resultado. Pero sus curvas se solapan exactamente y la de
    abajo desaparece, con lo que una figura honesta parece tener una serie
    perdida. Anotarlo convierte un artefacto visual confuso en la afirmacion
    que el proyecto quiere hacer.

    Args:
        result: el `GAResult` a etiquetar.
        others: los demas rungs de la misma figura.

    Returns:
        La etiqueta, con sufijo " ≡ <otro>" si coincide con uno anterior.
    """
    mine = np.array([h['theta_best'] for h in result.history], float)
    for o in others:
        if o is result or o.quantumness >= result.quantumness:
            continue
        theirs = np.array([h['theta_best'] for h in o.history], float)
        if mine.shape == theirs.shape and np.array_equal(mine, theirs):
            return f"{result.label}  ≡ {o.label}"
    return result.label


def animate_genetic_evolution(results, model, outdir: str,
                              tag: Optional[str] = None,
                              fps: int = 8, max_frames: int = 120,
                              formats: Sequence[str] = ('gif',),
                              layout: str = 'overlay') -> List[str]:
    """Anima la evolucion de la poblacion, con TODOS los rungs a la vez.

    [ANIM] Pensada para presentar. Un genetico se entiende mucho mejor viendo
    la nube contraerse que leyendo la curva de chi2 ya terminada, y poner los
    rungs uno al lado de otro en la MISMA animacion deja ver algo que ninguna
    figura estatica muestra: que unos convergen antes que otros, y si alguno
    se queda atascado en otro sitio.

    Dos paneles:

      Izquierda — espacio de fase de los dos primeros parametros. Cada rung es
        un color; los puntos son la poblacion de esa generacion, con opacidad
        proporcional a su fitness relativa, y el mejor individuo marcado con
        una estrella. La nube contrayendose ES la convergencia.

      Derecha — la curva de chi2 dibujandose en tiempo real, para poder decir
        "aqui es donde se estanca" senalando el mismo instante en los dos
        paneles.

    Args:
        results: iterable de `GAResult` CON `pop_history` (si esta vacio, el
            rung se omite de la animacion en vez de romperla).
        model: `CosmoModel` activo.
        outdir: carpeta de salida.
        tag: sufijo del nombre de archivo.
        fps: cuadros por segundo.
        max_frames: tope de cuadros; si hay mas generaciones se submuestrea de
            forma uniforme, para que un run de 500 generaciones no produzca un
            GIF de 60 MB.
        formats: 'gif' y/o 'mp4'. El mp4 necesita ffmpeg; si falta, se avisa y
            se sigue con el resto.
        layout: 'overlay' (todos los rungs en un panel) o 'facets' (un panel
            por rung). [B-OVERLAP] La superposicion es honesta —los rungs
            convergen al MISMO sitio, y verlo es el resultado— pero por eso
            mismo las nubes se tapan y solo se distingue la ultima dibujada.
            'overlay' lo mitiga con un marcador distinto por rung y orden de
            dibujo por quantumness; 'facets' lo elimina dando a cada rung su
            propio panel, a costa de perder la comparacion directa.

    Returns:
        Rutas generadas (vacia si no habia nada que animar).
    """
    import matplotlib.animation as manim

    os.makedirs(outdir, exist_ok=True)
    usable = [r for r in results if getattr(r, 'pop_history', None)]
    if not usable:
        return []
    usable = sorted(usable, key=lambda r: r.quantumness)
    tag = tag or model.name
    d = model.n_params
    if d < 2:
        return []
    latex = getattr(model, 'param_latex', model.param_names)

    n_gen = min(len(r.pop_history) for r in usable)
    step = max(1, int(np.ceil(n_gen / max_frames)))
    frames = list(range(0, n_gen, step))
    if frames[-1] != n_gen - 1:
        frames.append(n_gen - 1)

    palette = [C_GENETIC, C_GENETIC2, C_QUANTUM2, C_CLASSICAL2, C_QUANTUM]
    marks = ['o', 's', '^', 'D', 'v', 'P', 'X']
    style = {r.label: palette[i % len(palette)] for i, r in enumerate(usable)}
    # [B-OVERLAP] Un MARCADOR distinto por rung ademas del color: cuando las
    # nubes se solapan —y se solapan siempre, porque convergen al mismo sitio—
    # el color por si solo no basta para saber cual es cual.
    mark = {r.label: marks[i % len(marks)] for i, r in enumerate(usable)}
    # QGA(0%) reproduce al CGA bit a bit, asi que sus nubes se solapan y la de
    # abajo desaparece. Se anota para que no parezca una serie perdida.
    lab = {r.label: _coincidence_label(r, usable) for r in usable}

    # [B-OVERLAP] 'facets' da un panel por rung: la unica forma de garantizar
    # que TODOS se vean cuando convergen al mismo punto. 'overlay' conserva la
    # comparacion directa, que es lo que hace evidente que convergen juntos.
    facets = (layout == 'facets') and len(usable) > 1
    if facets:
        ncol = len(usable) + 1
        # Sin sharex/sharey: el panel de chi2 tiene escalas propias, y
        # desengancharlo despues depende de una API privada de matplotlib que
        # cambia entre versiones (`GrouperView.remove` no existe en 3.10+).
        # Los paneles de fase se igualan a mano mas abajo, que es explicito y
        # no depende de la version.
        # El panel de chi2 lleva doble ancho: es una serie temporal con varias
        # curvas y a 1/6 del ancho no se distingue cual es cual.
        fig, axes = plt.subplots(
            1, ncol, figsize=(3.7 * len(usable) + 6.2, 4.8),
            gridspec_kw={'width_ratios': [1] * len(usable) + [1.9]})
        panel = {r.label: axes[i] for i, r in enumerate(usable)}
        axc = axes[-1]
        axp = axes[0]
    else:
        fig, (axp, axc) = plt.subplots(1, 2, figsize=(13.5, 5.6),
                                       gridspec_kw={'width_ratios': [1.15, 1]})
        panel = {r.label: axp for r in usable}

    allpop = np.vstack([p for r in usable for p in r.pop_history])
    lo0, hi0 = np.nanpercentile(allpop[:, 0], [0.5, 99.5])
    lo1, hi1 = np.nanpercentile(allpop[:, 1], [0.5, 99.5])
    pad0, pad1 = 0.08 * (hi0 - lo0 + 1e-9), 0.08 * (hi1 - lo1 + 1e-9)
    fid = getattr(model, 'fiducial', None)
    seen_axes = []
    for r in usable:
        ax = panel[r.label]
        if ax in seen_axes:
            continue
        seen_axes.append(ax)
        ax.set_xlim(lo0 - pad0, hi0 + pad0)
        ax.set_ylim(lo1 - pad1, hi1 + pad1)
        ax.set_xlabel(latex[0] if len(latex) > 0 else model.param_names[0],
                      fontsize=12)
        if ax is seen_axes[0]:
            ax.set_ylabel(latex[1] if len(latex) > 1 else model.param_names[1],
                          fontsize=12)
        if fid is not None and len(fid) >= 2:
            ax.plot(fid[0], fid[1], '+', color='0.35', ms=14, mew=2, zorder=1)
        ax.grid(True, alpha=0.2, lw=0.6)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
    for sp in ('top', 'right'):
        axc.spines[sp].set_visible(False)

    all_chi = np.concatenate([[h['best_chi2'] for h in r.history]
                              for r in usable])
    all_chi = all_chi[np.isfinite(all_chi)]
    axc.set_xlim(0, (n_gen - 1) * 1.14)      # hueco para las etiquetas
    if len(all_chi):
        # Rango COMPLETO, no percentiles: recortar al 98% dejaba fuera el rung
        # peor, que es justo el que hay que poder ver empeorar.
        c_lo, c_hi = float(np.min(all_chi)), float(np.max(all_chi))
        pad = 0.08 * (c_hi - c_lo + 1e-9)
        axc.set_ylim(c_lo - pad, c_hi + pad)
    axc.set_xlabel('Generation', fontsize=12)
    _chi2_axis(axc, usable, plt)
    axc.grid(True, alpha=0.2, lw=0.6)

    scats, bests, lines, ends = {}, {}, {}, {}
    for r in usable:
        col = style[r.label]
        scats[r.label] = panel[r.label].scatter(
                                     [], [], s=30, color=col, alpha=0.55,
                                     marker=mark[r.label],
                                     linewidths=0.4, edgecolors='white',
                                     label=lab[r.label], zorder=3)
        bests[r.label] = panel[r.label].plot(
            [], [], '*', color=col, ms=20, mec='black', mew=0.9, zorder=6)[0]
        if facets:
            panel[r.label].set_title(lab[r.label], fontsize=10,
                                     color=col, fontweight='bold')
        # [B-OVERLAP] Estilo de linea Y grosor decreciente por rung. Cuatro de
        # las cinco curvas son GENUINAMENTE distintas (27.469 / 27.475 /
        # 27.680 / 28.079 en la corrida de referencia) pero difieren en
        # milesimas sobre un eje que abarca ~1.5, asi que a color solo se ven
        # como una. El guionado deja ver la de abajo y el grosor da orden.
        i = usable.index(r)
        lines[r.label] = axc.plot(
            [], [], color=col, lw=max(1.3, 3.4 - 0.55 * i),
            ls=_LS_CYCLE[i % len(_LS_CYCLE)],
            marker=mark[r.label], markevery=0.18, ms=5,
            label=r.label)[0]
        # Etiqueta directa con el valor: cuando dos curvas se pisan de todos
        # modos, el numero es lo unico que las separa sin ambiguedad.
        ends[r.label] = axc.annotate(
            '', xy=(0, 0), fontsize=8, color=col, fontweight='bold',
            xytext=(4, 0), textcoords='offset points',
            va='center', ha='left', annotation_clip=False)
    if not facets:
        axp.legend(fontsize=8, frameon=False, loc='upper right')
    title = fig.suptitle('', fontsize=13, fontweight='bold')

    def draw(k):
        g = frames[k]
        finals = []
        for r in usable:
            pop = r.pop_history[g]
            scats[r.label].set_offsets(pop[:, :2])
            tb = r.history[g]['theta_best']
            bests[r.label].set_data([tb[0]], [tb[1]])
            gens = [h['gen'] for h in r.history[:g + 1]]
            chi = [h['best_chi2'] for h in r.history[:g + 1]]
            lines[r.label].set_data(gens, chi)
            if chi:
                ends[r.label].set_text(f"{chi[-1]:.3f}")
                ends[r.label].xy = (gens[-1], chi[-1])
                finals.append((chi[-1], ends[r.label]))
        _spread_labels(axc, finals)
        title.set_text(f"{model.label} — genetic evolution   ·   "
                       f"generation {g + 1}/{n_gen}")
        return (list(scats.values()) + list(bests.values())
                + list(lines.values()) + list(ends.values()))

    anim = manim.FuncAnimation(fig, draw, frames=len(frames),
                               interval=1000 / max(fps, 1), blit=False)

    made: List[str] = []
    for fmt in formats:
        path = os.path.join(outdir, f"genetic_evolution_{tag}.{fmt}")
        try:
            if fmt == 'gif':
                anim.save(path, writer=manim.PillowWriter(fps=fps))
            elif fmt == 'mp4':
                anim.save(path, writer=manim.FFMpegWriter(fps=fps, bitrate=2400))
            else:
                continue
            made.append(path)
        except Exception as exc:                            # pragma: no cover
            print(f"  (no pude guardar {fmt}: {type(exc).__name__}: {exc})")
    plt.close(fig)
    return made


def plot_overlay_with_samplers(result: GAResult, sampler_sets: dict,
                               model, outdir: str,
                               tag: Optional[str] = None) -> str:
    """ALL-IN-ONE overlay: genetic population over the MCMC/VI corner plots.

    Requirement 3 (overlay): superimpose the genetic MAP and final spread on
    top of the sampler posteriors, all on shared axes via `plot_corner_multi`.

    Args:
        result: the genetic `GAResult`.
        sampler_sets: dict mapping a legend label -> (samples, color, weights)
            for each sampler family to overlay, e.g.
            {'Classical MCMC': (flat_mcmc, C_CLASSICAL, None),
             'QVMC':           (qvmc_samples, C_QUANTUM2, qvmc_weights)}.
            Pass {} to overlay nothing (population corner only).
    """
    os.makedirs(outdir, exist_ok=True)
    tag = tag or f"overlay_{model.name}_{result.method.lower()}_q{int(result.quantumness):03d}"

    datasets, colors, labels, weights = [], [], [], []
    for lbl, (S, col, w) in sampler_sets.items():
        datasets.append(np.asarray(S)); colors.append(col)
        labels.append(lbl); weights.append(w)

    # genetic population last so its contour sits on top
    finite = np.isfinite(result.final_fit)
    gcolor = C_GENETIC if result.quantumness == 0 else C_GENETIC2
    datasets.append(result.final_pop[finite])
    colors.append(gcolor)
    labels.append(f"{result.label} (MAP overlay)")
    weights.append(result.final_weights[finite])

    f = _cmq().plot_corner_multi(
        datasets=datasets, colors=colors, labels=labels, model=model,
        outdir=outdir, tag=tag,
        title=(f"{model.label} — samplers + genetic optimizer overlay\n"
               f"{result.label} MAP: "
               + "  ".join(f"{n}={v:.4f}"
                           for n, v in zip(model.param_names, result.theta_map))),
        weights_list=weights)
    return f


def _spread_labels(ax, items, min_gap_frac: float = 0.045):
    """Separa verticalmente etiquetas que se pisan, sin mover los datos.

    [B-OVERLAP] Las etiquetas directas de valor final resuelven el solapamiento
    de las CURVAS, pero cuando dos curvas acaban casi en el mismo sitio son las
    ETIQUETAS las que se pisan y el numero queda ilegible — que es peor que no
    ponerlo, porque parece un valor y no lo es.

    Empuja cada etiqueta lo justo para dejar un hueco minimo, en orden de
    valor. Solo mueve el texto: el punto al que apunta no cambia, asi que la
    figura no miente sobre donde acaba cada curva.

    Args:
        ax: eje que contiene las anotaciones.
        items: lista de (valor_y, objeto_anotacion).
        min_gap_frac: hueco minimo como fraccion del rango del eje.
    """
    if len(items) < 2:
        return
    lo, hi = ax.get_ylim()
    gap = min_gap_frac * (hi - lo)
    items = sorted(items, key=lambda t: t[0])
    placed = []
    for val, ann in items:
        y = val
        if placed and y - placed[-1] < gap:
            y = placed[-1] + gap
        placed.append(y)
        x, _ = ann.xy
        ann.set_position((4, (y - val) / (hi - lo) * ax.bbox.height))


def _chi2_axis(ax, results, plt):
    """Formatea un eje de chi2 para que NO mienta con notacion de offset.

    [B-OFFSET] Los rungs convergen a chi2 casi identicos — con CC+BAO+Pantheon
    (~1099 puntos) todos aterrizan cerca de 1100 — asi que el rango del eje es
    de decimas sobre un valor de cuatro cifras. Ante eso matplotlib etiqueta el
    eje 0.0, 0.1, 0.2... y pone un "+1.1e3" minusculo en una esquina, con lo
    que la figura APARENTA un chi2 entre 0 y 1. Un lector razonable concluye
    que esta viendo el chi2 reducido, o que el ajuste es absurdamente bueno.

    Se desactiva el offset y se anade al eje el numero de datos y el chi2
    reducido, que es lo que de verdad permite juzgar el ajuste de un vistazo:
    chi2 ~ 1100 con 1099 puntos es chi2_red ~ 1.0, es decir, un buen ajuste.

    Args:
        ax: eje de matplotlib.
        results: `GAResult` de la figura (para sacar n_data y chi2_red).
        plt: modulo pyplot.
    """
    ax.ticklabel_format(axis='y', style='plain', useOffset=False)
    ax.get_yaxis().get_major_formatter().set_useOffset(False)
    n_data, red = None, None
    for r in results:
        st = getattr(r, 'stats', None) or {}
        if st.get('n_data'):
            n_data = int(st['n_data'])
            red = st.get('chi2_red')
            break
    label = r'best $\chi^2$  (raw, not reduced)'
    if n_data:
        label += f'\n{n_data} data points'
        if red is not None and np.isfinite(red):
            label += f'   ·   best $\\chi^2_\\nu$ = {red:.3f}'
    ax.set_ylabel(label, fontsize=11)


def plot_fitness_curve(results: Sequence[GAResult], outdir: str,
                       model, tag: Optional[str] = None) -> str:
    """Best-χ² and mean-χ² vs generation for one or more genetic runs.

    Overlays several runs (e.g. CGA vs QGA at various quantumness levels) so
    convergence speed and final χ² are directly comparable.
    """
    os.makedirs(outdir, exist_ok=True)
    tag = tag or f"{model.name}_fitness"
    fig, ax = plt.subplots(figsize=(9, 5))
    # [B-OVERLAP] Un color, un estilo y un grosor por rung, mas la etiqueta
    # directa del valor final: las curvas convergen a valores que difieren en
    # milesimas y a color solo se leen como una sola.
    pal = [C_GENETIC, C_GENETIC2, C_QUANTUM2, C_CLASSICAL2, C_QUANTUM]
    order = sorted(results, key=lambda x: x.quantumness)
    end_labels = []
    for i, r in enumerate(order):
        gens = [h['gen'] for h in r.history]
        best = [h['best_chi2'] for h in r.history]
        mean = [h['mean_chi2'] for h in r.history]
        col = pal[i % len(pal)]
        ax.plot(gens, best, color=col, lw=max(1.3, 3.4 - 0.55 * i),
                ls=_LS_CYCLE[i % len(_LS_CYCLE)],
                label=f"{_coincidence_label(r, order)} — best")
        ax.plot(gens, mean, color=col, lw=1.0, alpha=0.45,
                ls=_LS_CYCLE[i % len(_LS_CYCLE)])
        if best:
            ann = ax.annotate(f"{best[-1]:.3f}", xy=(gens[-1], best[-1]),
                              xytext=(5, 0), textcoords='offset points',
                              fontsize=8, color=col, fontweight='bold',
                              va='center', annotation_clip=False)
            end_labels.append((best[-1], ann))
    ax.set_xlim(0, max(len(r.history) for r in order) * 1.12)
    ax.set_xlabel('Generation', fontsize=12)
    _chi2_axis(ax, results, plt)
    ax.figure.canvas.draw()
    _spread_labels(ax, end_labels)
    ax.set_title(f'{model.label} — genetic fitness convergence', fontsize=12,
                 fontweight='bold')
    ax.grid(True, alpha=0.3); ax.legend(fontsize=9)
    fig.tight_layout()
    f = os.path.join(outdir, f'fitness_{tag}.png')
    fig.savefig(f, dpi=150, bbox_inches='tight')
    fig.savefig(f.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close(fig)
    return f


# ── CSV export matching the shared `resultados_config.csv` schema ────────────
#: Exact field order written by lcdm_quantum_samplers_personal.py so the
#: genetic rows append cleanly to the same accumulating table.
def _ga_side(result: GAResult, post: Posterior) -> dict:
    """Pack a GAResult into the 'side' dict shape the shared CSV writers expect.

    The genetic estimate is reported on equal footing with the samplers: the
    (mu, std) are the fitness-weighted statistics of the final population, and
    chi2/chi2_red/AIC/BIC come from the refined MAP via `fit_statistics`.
    """
    finite = np.isfinite(result.final_fit)
    pop = result.final_pop[finite]
    w = result.final_weights[finite]
    if w.sum() <= 0:
        w = np.ones(len(pop)) / max(len(pop), 1)
    mu = np.average(pop, weights=w, axis=0)
    std = np.sqrt(np.average((pop - mu) ** 2, weights=w, axis=0))
    st = result.stats
    return {
        'mu': mu, 'std': std, 'elapsed': result.elapsed,
        'chi2': st['chi2'], 'n_data': st['n_data'],
        'chi2_red': st['chi2_red'], 'AIC': st['AIC'], 'BIC': st['BIC'],
        'ess': ess_weights(w),
        # genetic optimizers have neither MH acceptance nor a VI KL, so those
        # columns are left blank by the shared writer (is_mcmc=True, no keys).
    }


def append_results_csv(result: GAResult, post: Posterior,
                       dataset_label: str, prior_type: str,
                       run_csv: str = "resultados_config.csv",
                       cumulative_csv: str = "resultados_config.csv",
                       n_bits: Optional[int] = None) -> str:
    """Append the genetic MAP to the per-run AND cumulative results CSVs.

    [FIX] Now reports EVERY model parameter (wCDM: w; CPL: w0, wa; GEDE: Delta),
    not just Om/H0, by reusing the SAME schema and writers as the sampler module
    (`cosmo_modular_quantum`). This guarantees CGA/QGA rows line up exactly with
    the QMCMC/QVMC rows in the shared cumulative table.

    Two schemas, like the sampler module:
      * run_csv        — NAMED per-parameter columns (one model per run).
      * cumulative_csv — GENERIC p1..pN columns (mixed models share one file).

    "Method" is 'CGA' or 'QGA (q=NN%)'. The qubits-per-parameter (n_bits) is
    written in the nqpp column for a QGA, '—' for the classical CGA.
    """
    side = _ga_side(result, post)
    model = post.model
    nqpp_tag = (str(n_bits) if (result.method == 'QGA' and n_bits is not None)
                else "—")
    side_rows = [(side, result.label, True, nqpp_tag)]

    # NAMED per-run schema
    named_fields = _cmq().csv_fields_for_model(model)
    named_rows = [_cmq().csv_row_for_side(s, model, lbl, mc, nq,
                                       dataset_label, prior_type)
                  for (s, lbl, mc, nq) in side_rows]
    _cmq()._write_csv_rows(named_rows, named_fields, run_csv)

    # GENERIC cumulative schema
    if cumulative_csv:
        gen_fields = _cmq().csv_fields_generic()
        gen_rows = [_cmq().csv_row_generic(s, model, lbl, mc, nq,
                                        dataset_label, prior_type)
                    for (s, lbl, mc, nq) in side_rows]
        _cmq()._write_csv_rows(gen_rows, gen_fields, cumulative_csv)
    return run_csv
    return csv_path


# =============================================================================
# 6.  ORCHESTRATION  (run one/both methods, optionally overlay on samplers)
# =============================================================================


def _build_optimizer(method: str, post: Posterior, ga: GAConfig,
                     qga_config: dict, n_bits: int,
                     rng: np.random.Generator, shots: int):
    """Factory: return a CGA or QGA instance sharing the given RNG."""
    if method == 'cga':
        return CGA(post, ga, rng=rng)
    if method == 'qga':
        return QGA(post, ga, qga_config, n_bits=n_bits, rng=rng, shots=shots)
    raise ValueError(f"Unknown genetic method: {method!r}")


def run_genetic(post: Posterior, methods: Sequence[str], ga: GAConfig,
                qga_config: dict, n_bits: int, dataset_label: str,
                prior_type: str, outdir: str, live: bool,
                logger=None, log_every: int = 10, shots: int = 1,
                make_plots: bool = True, write_csv: bool = True,
                sampler_overlay: Optional[dict] = None,
                cumulative_csv: str = "resultados_config.csv"
                ) -> Dict[str, GAResult]:
    """Run the requested genetic optimizers and wire results into the pipeline.

    Args:
        post: cosmological posterior.
        methods: subset of {'cga', 'qga'}.
        ga: GA hyper-parameters.
        qga_config: quantum-component dict for the QGA.
        n_bits: qubits per parameter for the QGA encoding.
        dataset_label: e.g. 'CC', 'CC+Pantheon+' (for the CSV).
        prior_type: 'flat' or 'gaussian' (for the CSV).
        outdir: figure output directory.
        live: launch the live GUI (interactive only; forced False in CLI).
        logger: headless logging target.
        log_every: generational cadence for logging / live updates.
        shots: Aer shots per circuit (QGA).
        make_plots: produce corner + fitness figures.
        write_csv: append MAP rows to resultados_config.csv.
        sampler_overlay: optional dict for the all-in-one overlay (see
            `plot_overlay_with_samplers`). If provided, an overlay corner is
            also produced for each genetic result.

    Returns:
        dict mapping method name -> GAResult.
    """
    os.makedirs(outdir, exist_ok=True)
    say = logger.info if logger else print
    model = post.model
    results: Dict[str, GAResult] = {}

    # A fresh, seed-locked RNG per method so CGA and QGA are reproducible and
    # comparable (the seed is shared so the classical parts coincide).
    for method in methods:
        rng = np.random.default_rng(ga.seed)
        opt = _build_optimizer(method, post, ga, qga_config, n_bits, rng, shots)
        res = opt.evolve(live=live, logger=logger, log_every=log_every,
                         outdir=outdir)
        results[method] = res

        if write_csv:
            # Two destinations:
            #   1) a per-run copy inside this run's folder (NAMED columns),
            #   2) the cumulative global table in the CWD (GENERIC columns,
            #      so mixed models share one tidy file).
            run_csv = os.path.join(outdir, "resultados_config.csv")
            append_results_csv(res, post, dataset_label, prior_type,
                               run_csv=run_csv,
                               cumulative_csv=cumulative_csv,
                               n_bits=n_bits)
            say(f"[{res.method}] MAP appended to {run_csv} "
                f"(+ cumulative {cumulative_csv})")

        if make_plots:
            # [PLOT-GA] El corner de la poblacion final ya NO se genera por
            # defecto: un GA converge a un punto, asi que ese corner es un
            # punto con unos pixeles alrededor y no distingue un rung de otro.
            # Su sustituto (`plot_genetic_convergence`) se genera una sola vez
            # con TODOS los rungs juntos, mas abajo, porque la comparacion
            # entre rungs es justo lo que la figura debe mostrar.
            if sampler_overlay:
                # El overlay SI se conserva: ahi el punto del genetico sobre el
                # posterior de los samplers es informativo — dice si el MAP cae
                # dentro de la nube o fuera.
                f2 = plot_overlay_with_samplers(
                    res, sampler_overlay, model, outdir)
                say(f"[{res.method}] all-in-one overlay: {f2}")

    if make_plots and len(results) > 1:
        # Con un solo metodo no hay nada que comparar: en --sweep-all la figura
        # conjunta la genera el llamador con todos los rungs acumulados.
        f3 = plot_fitness_curve(list(results.values()), outdir, model)
        say(f"Fitness convergence figure: {f3}")
        f4 = plot_genetic_convergence(list(results.values()), model, outdir)
        if f4:
            say(f"Genetic convergence + final estimate: {f4}")
        for a in animate_genetic_evolution(list(results.values()), model,
                                           outdir, formats=ANIM_FORMATS, layout=ANIM_LAYOUT):
            say(f"Animation: {a}")

    return results


# =============================================================================
# 7.  INTERACTIVE MENU
# =============================================================================


def _ask(prompt: str, options: dict, default):
    """Minimal single-choice prompt returning the chosen value."""
    keys = list(options)
    print(f"\n{prompt}")
    for k in keys:
        mark = "  <- default" if k == default else ""
        print(f"   [{k}] {options[k]}{mark}")
    raw = input("  > ").strip()
    if raw == "":
        return default
    try:
        key = int(raw) if not isinstance(default, str) else raw
    except ValueError:
        key = raw
    return key if key in options else default


def interactive_menu() -> dict:
    """Interactive configuration (mirrors the sampler module's menu style)."""
    print("=" * 70)
    print("  GENETIC GLOBAL OPTIMIZATION — CGA / QGA  (interactive mode)")
    print("=" * 70)

    model_opts = {i: f"{k} — {MODELS[k].label}"
                  for i, k in enumerate(MODELS)}
    mi = _ask("Cosmological model:", model_opts, 0)
    model_name = list(MODELS)[mi if mi in model_opts else 0]

    # Dataset menu: offer Pantheon 2018 / Pantheon+ 2022 only if their files
    # are present (same logic as the sampler module).
    pan = core.load_pantheon()
    panp = core.load_pantheon_plus()
    ds_opts = {0: 'CC+BAO H(z)'}
    ds_keys = {0: 'CC+BAO'}
    nxt = 1
    if pan is not None:
        ds_opts[nxt] = f"Pantheon 2018 ({len(pan['z'])} SNe, diagonal)"
        ds_keys[nxt] = 'Pantheon'; nxt += 1
        ds_opts[nxt] = 'CC+BAO + Pantheon 2018'
        ds_keys[nxt] = 'CC+BAO+Pantheon'; nxt += 1
    if panp is not None:
        ds_opts[nxt] = f"Pantheon+ 2022 ({len(panp['z'])} SNe, full cov.)"
        ds_keys[nxt] = 'Pantheon+'; nxt += 1
        ds_opts[nxt] = 'CC+BAO + Pantheon+ 2022'
        ds_keys[nxt] = 'CC+BAO+Pantheon+'; nxt += 1
    di = _ask("Dataset:", ds_opts, 0)
    dataset = ds_keys.get(di, 'CC+BAO')

    pr_opts = {0: 'flat (box)', 1: 'gaussian (Planck on Ωm,H0)'}
    pi = _ask("Prior:", pr_opts, 0)
    prior = 'gaussian' if pi == 1 else 'flat'

    me_opts = {0: 'CGA only', 1: 'QGA only', 2: 'CGA + QGA'}
    mei = _ask("Method(s):", me_opts, 0)
    methods = {0: ['cga'], 1: ['qga'], 2: ['cga', 'qga']}.get(mei, ['cga'])

    qcfg = dict(QGA_PRESETS[0])
    if 'qga' in methods:
        q_opts = {p: QGA_PRESETS[p]['label'] for p in QGA_PRESETS}
        qi = _ask("QGA quantumness preset:", q_opts, 100)
        qcfg = dict(QGA_PRESETS.get(qi, QGA_PRESETS[100]))

    def ask_int(prompt, default):
        raw = input(f"\n{prompt} [{default}]: ").strip()
        try:
            return int(raw) if raw else default
        except ValueError:
            return default

    pop = ask_int("Population size", 120)
    gens = ask_int("Generations", 80)
    n_bits = ask_int("QGA qubits per parameter (n_bits)", 6) if 'qga' in methods else 6

    # Hardware / profiling (the QGA can use a GPU; the CGA is pure NumPy).
    use_gpu = False
    if 'qga' in methods and gpu_available():
        use_gpu = _ask("QGA simulation device:",
                       {0: 'CPU', 1: 'GPU (Aer on CUDA — detected)'}, 1) == 1
    profile = _ask("Profile memory / GPU-hours and save a usage figure?",
                   {0: 'No', 1: 'Yes'}, 0) == 1

    return dict(model=model_name, dataset=dataset, prior=prior,
                methods=methods, qga_config=qcfg, pop_size=pop,
                n_generations=gens, n_bits=n_bits,
                use_gpu=use_gpu, profile=profile)


# =============================================================================
# 8.  CLI  (argparse + headless rule)
# =============================================================================


def build_parser() -> argparse.ArgumentParser:
    """Extend the project's CLI with the genetic methods and hyper-parameters."""
    p = argparse.ArgumentParser(
        description="Global optimization for the cosmological MAP via "
                    "Classical (CGA) and Quantum (QGA) Genetic Algorithms. "
                    "Reuses the EXACT cosmo_core posterior (CC + Pantheon+ "
                    "likelihoods). With arguments -> headless/batch (no live "
                    "GUI, progress to the log); without arguments -> "
                    "interactive menu with the live window.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""Examples:
  python cosmo_genetic_optimizers.py
  python cosmo_genetic_optimizers.py --methods cga --model lcdm --generations 80
  python cosmo_genetic_optimizers.py --methods cga qga --dataset CC+Pantheon+ \\
         --population-size 200 --generations 120 --qga-preset 67
  python cosmo_genetic_optimizers.py --methods qga --qga-config \\
         '{"q_init":true,"q_mutation":true,"q_crossover":false}'""")

    p.add_argument('--methods', nargs='+', choices=['cga', 'qga'],
                   default=None,
                   help="Genetic method(s) to run: cga, qga, or both")
    p.add_argument('--model', choices=list(MODELS), default='lcdm',
                   help='Cosmological model (default: lcdm)')
    p.add_argument('--sweep-all', action='store_true',
                   help='HPC batch mode: run CGA + QGA across the full QGA '
                        'quantumness ladder for EVERY model, into one master '
                        'folder with a subfolder per model. Launch once on a '
                        'supercomputer to collect all genetic results.')
    p.add_argument('--sweep-models', nargs='+', choices=list(MODELS),
                   default=None, metavar='MODEL',
                   help='Restrict --sweep-all to these models '
                        f'(default: all of {list(MODELS)})')
    p.add_argument('--sweep-qga-levels', nargs='+', type=int,
                   choices=list(QGA_PRESETS), default=None, metavar='PCT',
                   help='QGA quantumness presets to sweep '
                        f'(default: all of {list(QGA_PRESETS)})')
    p.add_argument('--dataset',
                   choices=['CC+BAO', 'Pantheon', 'Pantheon+',
                            'CC+BAO+Pantheon', 'CC+BAO+Pantheon+',
                            'CC', 'CC+Pantheon+'],   # last two: legacy aliases
                   default='CC+BAO',
                   help='Observational dataset (default: CC+BAO). '
                        'Pantheon = 2018 diagonal; Pantheon+ = 2022 full '
                        'covariance. CC / CC+Pantheon+ are legacy aliases.')
    p.add_argument('--prior', choices=['flat', 'gaussian'], default='flat',
                   help='Prior type (default: flat)')

    # genetic hyper-parameters
    p.add_argument('--generations', type=int, default=80,
                   help='Number of generations (default: 80)')
    p.add_argument('--population-size', type=int, default=120,
                   help='Population size (default: 120)')
    p.add_argument('--crossover-rate', type=float, default=0.9,
                   help='Crossover probability (default: 0.9)')
    p.add_argument('--mutation-rate', type=float, default=0.20,
                   help='Per-gene mutation probability (default: 0.20)')
    p.add_argument('--mutation-scale', type=float, default=0.12,
                   help='Mutation scale as a fraction of each box width '
                        '(default: 0.12)')
    p.add_argument('--elite-frac', type=float, default=0.08,
                   help='Elitism fraction (default: 0.08)')
    p.add_argument('--tournament-k', type=int, default=3,
                   help='Tournament size (default: 3)')

    # QGA-specific
    qg = p.add_mutually_exclusive_group()
    qg.add_argument('--qga-preset', type=int, choices=list(QGA_PRESETS),
                    metavar='N',
                    help=f'QGA quantumness preset {list(QGA_PRESETS)}')
    qg.add_argument('--qga-config', type=str, metavar='JSON',
                    help='QGA quantum-component config as JSON '
                         '(keys: q_init, q_mutation, q_crossover)')
    p.add_argument('--n-bits', type=int, default=6,
                   help='QGA qubits per parameter (grid = 2^n_bits; default 6)')
    p.add_argument('--shots', type=int, default=1,
                   help='Aer shots per circuit for the QGA (default: 1)')
    p.add_argument('--max-qubits', type=int, default=26, metavar='N',
                   help='Memory safety cap on total QGA qubits (n_bits*d), '
                        'i.e. the width of the quantum-initialization '
                        'circuit. Default 26 (~1.1 GB, laptop-safe). Each +1 '
                        'qubit DOUBLES the statevector (16 B/state): '
                        '24q~=0.3GB, 26q~=1.1GB, 28q~=4.3GB, 30q~=17GB. '
                        '[FIX] The old default of 18 came from applying the '
                        'SAMPLERS memory model (grid + likelihood auxiliary '
                        'arrays, 13.3 kB/state) to the QGA, which never '
                        'evaluates the likelihood over the grid — it '
                        'over-estimated QGA memory by ~830x and needlessly '
                        'clamped n_bits for 4-parameter models.')

    # run control
    p.add_argument('--outdir', type=str, default='results',
                   help='Output directory. Default creates a timestamped '
                        'subfolder results/run_<date>_<model>/ per run so '
                        'results never mix; pass an explicit path to override.')
    p.add_argument('--seed', type=int, default=42, help='RNG seed (default 42)')
    p.add_argument('--log-file', type=str, default=None,
                   help='Log file (default: auto in CLI mode)')
    p.add_argument('--log-every', type=int, default=10,
                   help='Generational logging cadence (default: 10)')
    p.add_argument('--no-plot', action='store_true', help='Skip all figures')
    p.add_argument('--no-csv', action='store_true',
                   help='Do not append to resultados_config.csv')
    p.add_argument('--no-live', action='store_true',
                   help='Force-disable the live GUI even in interactive mode')
    # [ANIM] La animacion se guarda SIEMPRE por defecto, tambien en modo batch:
    # es material de presentacion y no cuesta casi nada (se submuestrea a 120
    # cuadros como mucho). --anim none la desactiva.
    p.add_argument('--anim', type=str, default='gif',
                   choices=('gif', 'mp4', 'both', 'none'),
                   help="animacion de la evolucion de la poblacion, con todos "
                        "los rungs de quantumness superpuestos. Por defecto "
                        "'gif' (no necesita ffmpeg y se inserta directo en "
                        "diapositivas); 'mp4' pesa menos pero exige ffmpeg; "
                        "'both' guarda los dos; 'none' no anima.")
    p.add_argument('--anim-layout', type=str, default='overlay',
                   choices=('overlay', 'facets'),
                   help="'overlay' pone todos los rungs en un panel (se ve que "
                        "convergen juntos, pero las nubes se tapan); 'facets' "
                        "da un panel a cada uno (todos visibles, sin "
                        "comparacion directa).")
    p.add_argument('--anim-fps', type=int, default=8,
                   help='cuadros por segundo de la animacion (por defecto 8)')
    # [NOISE] Segundo eje de ablacion. `--noise none` (por defecto) reproduce
    # bit a bit el comportamiento previo a este eje.
    cnoise.add_noise_cli(p)
    p.add_argument('--gpu-check', action='store_true',
                   help='imprime por que la GPU esta o no disponible y sale, sin correr nada. Util en una HPC nueva: --gpu degrada a CPU cuando Aer no expone GPU, y sin este chequeo eso solo se nota por el tiempo de pared.')
    p.add_argument('--gpu', action='store_true',
                   help='Use the GPU for the QGA Aer simulation if available '
                        '(qiskit-aer-gpu + CUDA). Falls back to CPU otherwise. '
                        'No effect on the pure-NumPy CGA.')
    p.add_argument('--profile', action='store_true',
                   help='Profile peak CPU/GPU memory, wall time and GPU-hours, '
                        'and save a resource_usage_*.png figure.')
    p.add_argument('--self-test', action='store_true',
                   help='Run a quick correctness self-test and exit '
                        '(CGA reaches the known optimum; QGA(0%%) == CGA)')
    return p


def self_test() -> int:
    """Quick correctness checks (run with --self-test).

    1. CGA on ΛCDM/CC reaches the known data optimum (Ωm≈0.257, H0≈70.7).
    2. QGA with all quantum components OFF reproduces the CGA bit-for-bit
       (the mandatory classical baseline of the quantumness ladder).
    3. Each quantum operator (init / mutation / crossover) runs and still
       converges to the same MAP within the grid resolution.
    """
    matplotlib.use('Agg')
    print("=" * 64)
    print("  SELF-TEST — cosmo_genetic_optimizers")
    print("=" * 64)
    post = Posterior(MODELS['lcdm'], dataset='CC', prior_type='flat')
    ga = GAConfig(pop_size=60, n_generations=40, seed=1)

    cga = CGA(post, ga, rng=np.random.default_rng(1))
    rc = cga.evolve(live=False, log_every=999)
    ok1 = abs(rc.theta_map[0] - 0.2574) < 0.02 and abs(rc.theta_map[1] - 70.7) < 1.5
    print(f"  [1] CGA optimum  MAP={np.round(rc.theta_map, 4)} "
          f"χ²={rc.chi2_map:.3f}   -> {'PASS' if ok1 else 'FAIL'}")

    if _QISKIT_OK:
        q0 = QGA(post, ga, QGA_PRESETS[0], n_bits=6,
                 rng=np.random.default_rng(1))
        r0 = q0.evolve(live=False, log_every=999)
        ok2 = np.allclose(rc.theta_map, r0.theta_map, atol=1e-6)
        print(f"  [2] QGA(0%) == CGA                          "
              f"-> {'PASS' if ok2 else 'FAIL'}")

        ga_s = GAConfig(pop_size=20, n_generations=10, seed=2)
        ok3 = True
        # One ALGORITHMIC operator switched on at a time (single-operator
        # ablation): each should still locate the MAP region.
        for nm, cfg in [('q_init',
                         dict(q_init=True, q_mutation=False, q_crossover=False)),
                        ('q_mutation',
                         dict(q_init=False, q_mutation=True, q_crossover=False)),
                        ('q_crossover',
                         dict(q_init=False, q_mutation=False, q_crossover=True))]:
            q = QGA(post, ga_s, cfg, n_bits=5, rng=np.random.default_rng(2))
            r = q.evolve(live=False, log_every=999)
            good = abs(r.theta_map[0] - 0.2574) < 0.03
            ok3 &= good
            print(f"  [3] quantum {nm:12s} q={r.quantumness:5.1f}%  "
                  f"MAP={np.round(r.theta_map, 3)}  "
                  f"-> {'PASS' if good else 'FAIL'}")
        all_ok = ok1 and ok2 and ok3
    else:
        print("  [2-3] Qiskit not available -> QGA tests skipped")
        all_ok = ok1

    print("=" * 64)
    print(f"  RESULT: {'ALL PASS' if all_ok else 'SOME FAILED'}")
    print("=" * 64)
    return 0 if all_ok else 1


def _resolve_qga_config(args) -> dict:
    """Resolve the QGA component config from --qga-preset / --qga-config."""
    if getattr(args, 'qga_config', None):
        cfg = json.loads(args.qga_config)
        cfg.setdefault('label',
                       f"{compute_qga_quantumness(cfg):.0f}% — JSON config")
        return cfg
    if getattr(args, 'qga_preset', None) is not None:
        return dict(QGA_PRESETS[args.qga_preset])
    return dict(QGA_PRESETS[100])              # default: fully quantum operators


def run_genetic_sweep_all(models, qga_levels, methods, dataset, prior, ga,
                          n_bits, shots, master_dir, logger, log_every,
                          no_csv=False, no_plot=False):
    """Run CGA + QGA (across the quantumness ladder) for EVERY model in one go.

    HPC "launch once, get everything" mode for the genetic optimizers. For each
    model it runs the CGA once (it has no quantumness) and the QGA at every
    requested quantumness preset, into the model's OWN subfolder of the master
    run directory, appending every MAP row to a single cumulative CSV.

    Each model runs inside try/except so one failure does not abort the batch.

    Args:
        models: model keys to sweep.
        qga_levels: list of QGA preset percentages (keys of QGA_PRESETS).
        methods: which of {'cga','qga'} to include.
        dataset, prior, ga, n_bits, shots: shared setup / hyper-parameters.
        master_dir: master folder; per-model subfolders created inside.
        logger, log_every: logging target and cadence.
        no_csv, no_plot: pass-throughs.

    Returns:
        dict mapping model key -> 'ok' or error string.
    """
    say = logger.info if logger else print
    cumulative_master = os.path.join(master_dir,
                                     "resultados_TODOS_los_modelos.csv")
    status = {}
    t_start = time.time()

    say("=" * 70)
    say(f"GENETIC SWEEP-ALL — {len(models)} model(s): {', '.join(models)}")
    say(f"  methods={methods} | QGA levels={qga_levels} | dataset={dataset} "
        f"| prior={prior}")
    say(f"  pop={ga.pop_size} gens={ga.n_generations} n_bits={n_bits}")
    say(f"  master folder: {master_dir}/")
    say("=" * 70)

    for i, model_name in enumerate(models, 1):
        say("")
        say(f"[{i}/{len(models)}] ===== MODEL: {model_name} "
            f"({MODELS[model_name].label}) =====")
        model_dir = os.path.join(master_dir, f"model_{model_name}")
        os.makedirs(model_dir, exist_ok=True)
        try:
            post = Posterior(MODELS[model_name], dataset, prior)

            # [PLOT-GA] Los resultados de TODOS los rungs se acumulan para
            # que la figura de convergencia los compare en una sola imagen.
            # Generarla dentro de cada llamada produciria una figura por rung,
            # cada una con una sola curva — que es justo lo que no sirve.
            all_res: List[GAResult] = []

            # CGA once (no quantumness).
            if 'cga' in methods:
                all_res += list(run_genetic(
                    post=post, methods=['cga'], ga=ga,
                    qga_config=dict(QGA_PRESETS[0]), n_bits=n_bits,
                    dataset_label=dataset, prior_type=prior, outdir=model_dir,
                    live=False, logger=logger, log_every=log_every,
                    shots=shots, make_plots=not no_plot, write_csv=not no_csv,
                    cumulative_csv=cumulative_master).values())

            # QGA at each requested quantumness preset.
            if 'qga' in methods:
                for pct in qga_levels:
                    all_res += list(run_genetic(
                        post=post, methods=['qga'], ga=ga,
                        qga_config=dict(QGA_PRESETS[pct]), n_bits=n_bits,
                        dataset_label=dataset, prior_type=prior,
                        outdir=model_dir, live=False, logger=logger,
                        log_every=log_every, shots=shots,
                        make_plots=not no_plot, write_csv=not no_csv,
                        cumulative_csv=cumulative_master).values())

            if not no_plot and len(all_res) > 1:
                f = plot_genetic_convergence(all_res, post.model, model_dir)
                if f:
                    say(f"  escalera completa en una figura: {f}")
                f = plot_fitness_curve(all_res, model_dir, post.model)
                say(f"  fitness de todos los rungs: {f}")
                # [ANIM] Todos los rungs en la MISMA animacion: es donde se ve
                # cual converge antes y cual se queda atascado.
                for a in animate_genetic_evolution(
                        all_res, post.model, model_dir, formats=ANIM_FORMATS, layout=ANIM_LAYOUT):
                    say(f"  animacion: {a}")
            status[model_name] = 'ok'
            say(f"[{i}/{len(models)}] {model_name}: DONE -> {model_dir}/")
        except Exception as exc:
            status[model_name] = f"FAILED: {exc}"
            say(f"[{i}/{len(models)}] {model_name}: FAILED — {exc}")
            import traceback
            (logger.error if logger else print)(traceback.format_exc())

    elapsed = time.time() - t_start
    say("")
    say("=" * 70)
    say(f"GENETIC SWEEP-ALL finished in {elapsed/60:.1f} min")
    for m in models:
        say(f"  {m:8s} : {status[m]}")
    if not no_csv:
        say(f"  Combined table: {cumulative_master}")
    say("=" * 70)
    return status


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point.

    REGLA CRÍTICA (headless rule): if the script is launched with ANY CLI
    argument we are in batch/HPC mode -> the live animation is DISABLED and the
    generational progress goes to the log file every --log-every generations.
    Without arguments we enter the interactive menu and the live GUI is shown.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    if getattr(args, 'gpu_check', False):
        from cosmo_core import gpu_diagnosis
        print(gpu_diagnosis())
        return 0

    if getattr(args, 'self_test', False):
        return self_test()

    # CLI mode = any argument present (other than the bare program name).
    raw_args = sys.argv[1:] if argv is None else argv
    cli_mode = len(raw_args) > 0

    # [ROBUSTNESS] Reject out-of-range numeric args with a clear message before
    # they crash deep inside NumPy/Qiskit (population=0, n-bits=0, etc.).
    if cli_mode:
        errs = []
        for attr, flag in [('population_size', 'population-size'),
                           ('generations', 'generations'),
                           ('n_bits', 'n-bits'), ('shots', 'shots')]:
            v = getattr(args, attr, None)
            if v is not None and v < 1:
                errs.append(f"--{flag} must be >= 1 (got {v})")
        if getattr(args, 'seed', 0) < 0:
            errs.append(f"--seed must be >= 0 (got {args.seed})")
        nb = getattr(args, 'n_bits', None)
        max_q = getattr(args, 'max_qubits', 18)
        uses_qga = (getattr(args, 'sweep_all', False)
                    or (args.methods and 'qga' in args.methods))
        if nb is not None and nb >= 1 and uses_qga:
            sweep = (args.sweep_models if getattr(args, 'sweep_all', False)
                     and args.sweep_models else
                     (list(MODELS) if getattr(args, 'sweep_all', False)
                      else [args.model]))
            max_d = max(MODELS[m].n_params for m in sweep)
            if nb * max_d > max_q:
                # 16 B/state: the quantum-init circuit is a plain statevector
                # (the QGA scores fitness on the POPULATION, not on the grid).
                approx_gb = 2 ** (nb * max_d) * 16 / 1e9
                errs.append(
                    f"--n-bits {nb} with a {max_d}-parameter model needs a "
                    f"{nb * max_d}-qubit initialization circuit "
                    f"(2^{nb * max_d} states ~ {approx_gb:.2f} GB), above "
                    f"the --max-qubits {max_q} cap. Lower n_bits to "
                    f"<= {max_q // max_d} or raise --max-qubits if your "
                    f"machine has the RAM.")
        if errs:
            sys.stderr.write("Argument error(s):\n  " + "\n  ".join(errs)
                             + "\n")
            return 2

    np.random.seed(args.seed)

    # [GPU] Publish the device choice module-wide so the QGA's simulator uses
    # it. --gpu requests the GPU; with no GPU present we fall back to CPU.
    global USE_GPU, ANIM_FORMATS, ANIM_LAYOUT
    USE_GPU = bool(getattr(args, 'gpu', False))
    do_profile = bool(getattr(args, 'profile', False))
    ANIM_FORMATS = {'gif': ('gif',), 'mp4': ('mp4',),
                    'both': ('gif', 'mp4'),
                    'none': ()}[getattr(args, 'anim', 'gif')]
    ANIM_LAYOUT = getattr(args, 'anim_layout', 'overlay')

    # [NOISE] Publicar el peldano ANTES de construir ningun QGA: el simulador
    # nace en __init__ y los circuitos se transpilan una sola vez contra el.
    _spec = cnoise.spec_from_args(args)
    set_noise(_spec)
    if not _spec.is_ideal:
        print(f"[NOISE] peldano='{_spec.label}' | metodo=density_matrix | "
              f"techo={cnoise.MAX_NOISY_QUBITS}q | "
              f"el QGA ya lee por medicion: sin conversion de lectura")

    # ── SWEEP-ALL: CGA + QGA across all models in one master folder ──
    if getattr(args, 'sweep_all', False):
        set_headless_backend()
        device = resolve_device(USE_GPU, warn=True)
        sweep_models = args.sweep_models or list(MODELS)
        qga_levels = args.sweep_qga_levels or list(QGA_PRESETS)
        methods = args.methods or ['cga', 'qga']
        if args.outdir == 'results':
            master_dir = make_run_dir('results', tag='genetic_sweep_all')
        else:
            master_dir = args.outdir
            os.makedirs(master_dir, exist_ok=True)
        ts = time.strftime('%Y%m%d_%H%M%S')
        log_file = args.log_file or os.path.join(
            master_dir, f"genetic_sweep_all_{ts}.log")
        logger = setup_logger(log_file, name="genetic")
        import logging as _logging
        for h in logger.handlers:
            if isinstance(h, _logging.StreamHandler) and not isinstance(
                    h, _logging.FileHandler):
                h.setLevel(_logging.INFO)
        print(f"  GENETIC SWEEP-ALL: master folder {master_dir}/  | "
              f"log {log_file}")
        logger.info("QGA simulation device: %s", device)
        if USE_GPU and device == 'CPU':
            logger.info("  ⚠  --gpu requested but no Aer GPU device available; "
                        "QGA running on CPU.")
        ga = GAConfig(pop_size=args.population_size,
                      n_generations=args.generations,
                      crossover_rate=args.crossover_rate,
                      mutation_rate=args.mutation_rate,
                      mutation_scale=args.mutation_scale,
                      elite_frac=args.elite_frac,
                      tournament_k=args.tournament_k, seed=args.seed)
        profiler = None
        if do_profile:
            import cosmo_profiling as _prof
            profiler = _prof.ResourceProfiler(
                tag=f"genetic_sweep_{'gpu' if device == 'GPU' else 'cpu'}",
                device=device, interval=0.5)
            profiler.start()
        run_genetic_sweep_all(
            sweep_models, qga_levels, methods, args.dataset, args.prior, ga,
            args.n_bits, args.shots, master_dir, logger, args.log_every,
            no_csv=args.no_csv, no_plot=args.no_plot)
        if profiler is not None:
            import cosmo_profiling as _prof
            result = profiler.stop()
            logger.info(_prof.summarize(result))
            _prof.ResourceProfiler.plot(
                result, master_dir,
                title_extra=f"genetic sweep | {len(sweep_models)} models")
        return 0

    if cli_mode:
        # ── headless / batch / HPC ───────────────────────────────────────────
        # [FIX] Force the non-interactive backend HERE (not at import time) so a
        # headless node never tries to open a window and the live GUI code is
        # never reached.
        set_headless_backend()
        methods = args.methods or ['cga']
        model_name, dataset, prior = args.model, args.dataset, args.prior
        qga_config = _resolve_qga_config(args)
        ga = GAConfig(pop_size=args.population_size,
                      n_generations=args.generations,
                      crossover_rate=args.crossover_rate,
                      mutation_rate=args.mutation_rate,
                      mutation_scale=args.mutation_scale,
                      elite_frac=args.elite_frac,
                      tournament_k=args.tournament_k, seed=args.seed)
        n_bits = args.n_bits
        live = False                            # <- CRITICAL: never live in CLI
    else:
        # ── interactive menu + live GUI ──────────────────────────────────────
        logger = None
        sel = interactive_menu()
        model_name, dataset, prior = sel['model'], sel['dataset'], sel['prior']
        methods = sel['methods']
        qga_config = sel['qga_config']
        ga = GAConfig(pop_size=sel['pop_size'],
                      n_generations=sel['n_generations'], seed=args.seed)
        n_bits = sel['n_bits']
        USE_GPU = sel.get('use_gpu', USE_GPU)
        do_profile = sel.get('profile', do_profile)
        live = not args.no_live
        # [FIX] Guarantee an interactive backend BEFORE any LiveGA is built.
        # If only a headless backend is available (e.g. SSH without X-forwarding)
        # we disable the live window and tell the user, but still run + save the
        # static figures.
        if live and not ensure_interactive_backend():
            print("  ⚠  No interactive Matplotlib backend available "
                  "(headless display). Live GUI disabled; static figures and "
                  "the snapshot will still be saved.")
            print("  To enable the live window, fix one of these:")
            print(diagnose_gui_backend())
            live = False
        print(f"  Interactive mode: results -> {args.outdir}/"
              if args.outdir != 'results' else "")

    # [FIX] Create the timestamped run directory AFTER the model is known
    # (the interactive menu chooses it), so an interactive wCDM run no longer
    # writes into a folder named '..._lcdm' and the folder always exists before
    # any figure/CSV is written.
    if args.outdir == 'results':
        args.outdir = make_run_dir('results', tag=model_name)
    else:
        os.makedirs(args.outdir, exist_ok=True)

    if cli_mode:
        ts = time.strftime('%Y%m%d_%H%M%S')
        log_file = args.log_file or os.path.join(
            args.outdir, f"genetic_{model_name}_{ts}.log")
        logger = setup_logger(log_file, name="genetic")
        import logging as _logging
        for h in logger.handlers:
            if isinstance(h, _logging.StreamHandler) and not isinstance(
                    h, _logging.FileHandler):
                h.setLevel(_logging.INFO)
        logger.info("CLI/batch mode: live GUI disabled; results -> %s/  | "
                    "progress -> %s", args.outdir, log_file)
    else:
        logger = None
        print(f"  Results -> {args.outdir}/")

    model = MODELS[model_name]
    post = Posterior(model, dataset, prior)
    say = logger.info if logger else print
    say(f"Model: {model.label} | params {model.param_names} | "
        f"dataset {dataset} ({post.n_data} pts) | prior {prior}")
    if 'qga' in methods:
        say(f"QGA quantumness: {compute_qga_quantumness(qga_config):.0f}%  "
            f"({qga_config.get('label', '')})")

    # [GPU] Report the device actually used by the QGA (CGA is pure NumPy).
    device = resolve_device(USE_GPU, warn=True)
    if 'qga' in methods:
        if USE_GPU and device == 'CPU':
            say("  ⚠  --gpu requested but no Aer GPU device is available "
                "(need qiskit-aer-gpu + CUDA). QGA running on CPU.")
        say(f"  QGA simulation device: {device}")

    # [PROFILE] Optionally wrap the whole optimization in the resource profiler.
    profiler = None
    if do_profile:
        import cosmo_profiling as _prof
        profiler = _prof.ResourceProfiler(
            tag=f"genetic_{model_name}_{'gpu' if device == 'GPU' else 'cpu'}",
            device=device, interval=0.25)
        profiler.start()

    run_genetic(
        post=post, methods=methods, ga=ga, qga_config=qga_config,
        n_bits=n_bits, dataset_label=dataset, prior_type=prior,
        outdir=args.outdir, live=live, logger=logger,
        log_every=args.log_every, shots=args.shots,
        make_plots=not args.no_plot, write_csv=not args.no_csv)

    if profiler is not None:
        import cosmo_profiling as _prof
        result = profiler.stop()
        say(_prof.summarize(result))
        ptitle = f"{model.label} | {'+'.join(methods).upper()} | {dataset}"
        ppath = _prof.ResourceProfiler.plot(result, args.outdir,
                                            title_extra=ptitle)
        if ppath:
            say(f"  Resource-usage figure: {ppath}")
        try:
            with open(os.path.join(args.outdir,
                                   f"profile_{result.tag}.json"), 'w') as fh:
                json.dump(result.as_row(), fh, indent=2)
        except Exception:
            pass

    if live:
        print("\nClose the live window(s) to exit.")
        plt.ioff()
        plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
