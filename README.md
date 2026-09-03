# Quantum Algorithms for Cosmology — Classical vs Quantum Bayesian Inference

A toolkit that fits cosmological models (ΛCDM, wCDM, CPL, PEDE, GEDE) to
real data (combined CC+BAO H(z) measurements, and Type Ia supernovae —
Pantheon 2018 or Pantheon+ 2022) using **classical** and **quantum**
sampling algorithms, and compares them head to head.

**Status:** post-Fase-4 (62 tests, all green). Fase 4 added the **noise axis** — a second, orthogonal ablation dimension, so results are now a 2-D matrix (quantumness x noise) instead of a ladder; see [The noise axis](#the-noise-axis-second-ablation-dimension). `--noise none` is the default and reproduces every previous result bit for bit.

**Status (Fase 3):** hardening (33 tests, all green). Convergence,
divergence-tracking (KL), gradients, and reproducibility were audited and
fixed this round — see [Diagnostics and correctness fixes](#diagnostics-and-correctness-fixes)
for the full list, and re-run any figure generated before this round before
citing it (the convergence flag and KL numbers changed definition).

This README has two parts:

* **Part 1 — Plain-language guide.** No physics or quantum background
  needed. Start here.
* **Part 2 — Technical reference.** The academic detail: math, code
  architecture, algorithms, hardware, and diagnostics.

---
---

# Part 1 — Plain-language guide

## What is this, in one paragraph?

Cosmologists have measurements of how fast the Universe has expanded over
time. From those measurements they want to estimate a few numbers — for
example how much matter the Universe contains (Ωm) and how fast it is
expanding today (H0). There is no single "correct answer" you can read off
directly; instead you explore many possible combinations and keep the ones
that fit the data well. This project does that exploration in two different
ways — the **classical** way (ordinary computer algorithms) and the
**quantum** way (algorithms that use quantum circuits) — and checks whether
they give the same answer.

## A simple analogy

Imagine you lost your keys in a dark park and you're feeling around for
them.

* **MCMC** is "take a step, check if the ground feels more key-like, and
  wander toward the better spots." Do that long enough and you map out
  where the keys probably are.
* **VI (variational inference)** is "guess a simple shape for where the
  keys are (say, a circle), then keep adjusting that shape until it best
  matches what you feel."

This project has a **quantum version of each**:

* **QMCMC** — the same wandering search, but the *direction of each step*
  is suggested by a quantum circuit.
* **QVMC** — the same shape-fitting, but the *shape itself* is produced by
  a quantum circuit.

The whole point of the thesis is to ask: **do the quantum versions land in
the same place as the classical ones?** If yes, that's a meaningful result
— it shows these quantum methods faithfully reproduce trusted classical
results.

## What does the "quantumness %" mean?

Each method is built from swappable parts. You can run each part on a
normal computer ("classical") or on a quantum circuit ("quantum"). The
**quantumness %** is an *ablation index*: it counts *how many of the parts
are currently running on the quantum circuit*, from 0% (all classical) to
100% (all quantum). Every part counts equally — it is simply the fraction of
parts switched to quantum. (It is **not** a measure of "how much quantum
power" is used; it is a bookkeeping index for a controlled
classical-vs-quantum ablation, where we swap one part at a time and watch
what happens.)

There are **two separate dials**, one per method, because the two methods
are made of different numbers of parts:

```
QMCMC dial:  0%  →  50% (quantum step direction)  →  100% (quantum accept/reject)
QVMC  dial:  0%  →  33% (quantum draw)  →  67% (quantum fitting)  →  100% (quantum normalize)
```

Turning a dial up switches one more part to quantum. That's it. (The numbers
are just even fractions: QMCMC has 2 parts, so its rungs are halves; QVMC
has 3 parts, so its rungs are thirds.)

The genetic optimizer (`cosmo_genetic_optimizers.py`) has its own **third
dial**, also in thirds — see the technical section for details.

## The second dial: how much noise

Everything above assumes a **perfect** quantum computer. Real quantum
hardware makes mistakes: gates are slightly off, and when you finally read a
qubit you sometimes read the wrong value. Every result in this project up to
this point was computed on a *noiseless* simulator — the ideal limit.

So there is now a **second dial**, independent of the quantumness one:

```
                        noise  →
                    none   readout   gates+readout   real device
    quantumness 0%    ·        ·            ·             ·
          |    33%    ·        ·            ·             ·
          ↓    67%    ·        ·            ·             ·
              100%    ·        ·            ·             ·
```

Turn it up and the answers get **worse**. That is the point, and it is not a
failure: the interesting question is not *whether* the methods degrade but
**at what rate** — which rung of the quantumness ladder survives noise, and
which one breaks first.

```bash
# One noise level
python cosmo_modular_quantum.py --benchmark --model lcdm --noise full

# The whole 2-D matrix in one launch
python cosmo_hpc_runner.py --models lcdm --noise-sweep none,readout,full
```

`--noise none` is the default and reproduces the previous, noiseless
behaviour **bit for bit**, so nothing published before this change moves.

### What each noise setting actually means

There are several noise-related flags and they are easy to confuse. In plain
language:

**`--noise none`** — no noise at all. The perfect-computer limit. This is what
every result before this feature used, and it is still the default: run the
code exactly as you always did and nothing changes.

**`--noise readout`** — *only* the mistake at the very end. The quantum
computer does its work perfectly, but when you finally look at a qubit to see
whether it says 0 or 1, sometimes you read the wrong one. Nothing else is
wrong; only the answer you write down.

**`--noise full`** — the above **plus** mistakes during the work itself. Each
operation on the qubits nudges them slightly off where they should be, and
those nudges pile up through the circuit. This is the realistic "everything
is a bit wrong" setting.

**`--noise FakeBrisbane`** (or any other device name) — instead of made-up
error rates, use the **measured** ones from a real IBM machine. Every qubit on
a real chip is a little different, so this is the least idealized option.

Then three dials that only set *how much*, and only matter for the synthetic
levels (a real device brings its own measured numbers):

**`--noise-readout-p`** (default 0.03) — how often the final reading is wrong.
0.03 means 3 times in 100. This is the big one on today's hardware.

**`--noise-gate-p1`** (default 0.001) — how often a single-qubit operation goes
wrong: 1 in 1000. These are the cheap, reliable operations.

**`--noise-gate-p2`** (default 0.01) — how often a *two*-qubit operation goes
wrong: 1 in 100. Ten times worse than single-qubit ones, which is why circuits
are designed to use as few of them as possible.

And one flag that is not about noise at all, but gets tangled with it:

**`--proposal-route`** — *how* the QMCMC reads its answer out of the circuit.
With no noise it can read the exact wave amplitudes (`amplitude`); with noise
those stop existing and it has to count measurement outcomes instead
(`counts`), which is also what real hardware does. Because those are two
different ways of reading, comparing a noiseless run against a noisy one
compares *two things at once*. That is why the HPC runner adds a
**`none-counts`** column automatically: a noiseless run that reads by counting,
so you can tell the effect of the noise apart from the effect of the change in
reading. **Do not drop that column** — without it, in our first campaign the
numbers appeared to say that readout noise *improves* the sampling, which is
false.


## The one thing that surprises everyone

When you turn a dial up, **sometimes the answer changes and sometimes it
stays exactly the same** — and that is *expected*, not a bug. We label every
part as one of two kinds:

* **Treatment parts** are genuinely new algorithms, so they change the
  answer (the quantum *step direction* in QMCMC; the quantum *fitting* in
  QVMC). These are the parts where something interesting happens.
* **Faithful parts** are quantum *re-implementations of the exact same rule*
  the classical computer uses. Those give an identical answer **on purpose**
  — that identity is the proof that "the quantum version reproduces the
  classical one." They are the control of the experiment.

So when two neighboring dial settings look identical, that's a faithful part
doing its job — the project succeeding at its goal, not failing.

## How to run it

Install once:

```bash
pip install -r requirements.txt
```

Then run it and follow the menu:

```bash
python cosmo_modular_quantum.py
```

The menu asks three things:

1. **Run mode** — pick **Quick Test Run** the first time (fast, exercises
   everything), **Benchmark** for a full comparison, or **Single
   configuration** to run one specific setup.
2. **Which model and data** — defaults (ΛCDM, cosmic chronometers) are
   fine to start.
3. **Sizes** — how long to run (bigger = more accurate, slower). Defaults
   are sensible.

It then produces a set of **pictures** in the output folder.

> **CLI mode prints less than you'd expect — on purpose.** With any
> command-line argument, the script switches to batch mode: it prints only
> where results are going, then sends every step of progress to a **log
> file** in that folder (not the terminal), so it behaves well when
> redirected on an HPC job. The terminal going quiet for a minute or two
> **does not mean it stopped** — `tail -f results_.../*.log` to watch it
> live, or just wait: it returns to the prompt when it's actually done, with
> every figure and CSV already written.

### The figures the noise axis adds

* **`noise_comparison_<model>.png`** — the one to look at first. One line per
  quantumness rung, x-axis running from no noise to the real device. The
  parameter panels show the **shift away from the ideal run measured in that
  method's own sigma**, with a grey ±1σ band: a line that stays inside the band
  moved less than the method's own uncertainty, i.e. the noise did not
  displace it detectably. The right-hand panel shows fit quality (KL, or χ²),
  where the degradation actually shows up. Written automatically at the end of
  any `--noise-sweep` run.

There is also a **global optimizer** that hunts for the single best-fit point
(the MAP) using genetic algorithms — classical (CGA) and quantum (QGA) — with
a live animation of the population converging:

```bash
python cosmo_genetic_optimizers.py
```

The genetic runs produce two things worth knowing about:

* **`genetic_evolution_<model>.gif`** — an **animation** of the population
  converging, with **every quantumness rung on the same axes**. The left panel
  is the cloud of candidate universes shrinking onto the answer (stars = the
  best individual so far); the right panel is the χ² curve drawing itself in
  time with it, so you can point at the moment a rung gets stuck. Saved by
  default on every run, batch included — it is made for showing, not for
  debugging. `--anim mp4` (needs ffmpeg), `--anim both`, or `--anim none`.
  Long runs are subsampled to at most 120 frames so the file stays small.

* **`genetic_convergence_<model>.png`** — **this replaced the population corner
  plot.** A genetic algorithm converges to a *point*, so a corner plot of its
  final population is a dot with a few pixels around it: it spends a whole
  figure showing nothing, and hides the only things that separate one rung
  from another. The replacement shows the best individual per generation (top,
  one trace per rung) and the final estimate ± population spread (bottom, all
  rungs stacked so they compare at a glance). Where two rungs coincide exactly
  — QGA at 0% reproduces the CGA bit for bit, by design — the legend marks it
  with `≡` so an overlapping curve does not look like a missing one.

Corner plots are still the right figure for MCMC/VI, where the cloud of
samples *is* the result. They are the wrong figure for an optimizer whose
result is one point.

Every run now saves its pictures, log and results table into its own
timestamped folder, `results/run_<date>_<model>/`, so different runs never
overwrite or mix.

### Run everything at once (`--sweep-all`)

For an HPC job you usually want **all models in one launch**. The
`--sweep-all` flag runs the full quantumness benchmark (the QMCMC + QVMC
ladders) for every model, into a single master folder with one subfolder
per model, plus one cumulative CSV (`resultados_TODOS_los_modelos.csv`)
collecting every model/method/quantumness row for easy comparison. If a
model fails, the sweep logs it and continues with the rest.

```bash
# Samplers: benchmark of ALL models in one go
python cosmo_modular_quantum.py --sweep-all --dataset CC+BAO+Pantheon+ \
    --steps 15000 --qvmc-iter 3000 --nqpp 6 --gpu --profile

# Genetic: CGA + QGA across all quantumness levels, all models
python cosmo_genetic_optimizers.py --sweep-all --dataset CC+BAO+Pantheon+ \
    --generations 120 --population-size 200 --n-bits 6 --gpu --profile
```

Restrict the sweep with `--sweep-models lcdm cpl` (and, for the genetic
script, `--sweep-qga-levels 0 100`).

**Before a long HPC launch**, pin the BLAS thread count *before* the
process starts, or many parallel model-processes will oversubscribe the
node's cores against each other:

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
```

## How to read the pictures

* **Corner plots** (`corner_ladder_*`) — estimated values and their
  uncertainty. Each blob/contour is one method at one dial setting. **If
  the blobs sit on top of each other, the methods agree.** Dashed lines
  mark the reference ("Planck") values.

  **The shaded bands are 1σ, 2σ and 3σ**, darkest in the middle. Read them
  as "the true value is somewhere in here": the innermost band is where it
  most likely is, the outermost is where it almost certainly is.

  One number to be careful with: in a **two-parameter** panel those bands
  hold **39.3%, 86.5% and 98.9%** of the probability — *not* 68/95/99.7%.
  Those familiar numbers are for a single parameter (the histograms along
  the diagonal). Contours drawn at 0.68/0.95 in a 2-D panel would be wider
  than 1σ and 2σ and would make two methods look like they agree better
  than they do. The code uses `1 - exp(-n²/2)`, which is the correct 2-D
  value.
* **Convergence curve** (`ladder_rhat_*`) — whether the wandering search
  has "settled down." Lower is better; below the dashed line (R̂−1 = 0.01)
  means "settled." This threshold is strict on purpose — the whole project
  compares posteriors to each other, and a half-cooked one invalidates the
  comparison.
* **Training curve** (`ladder_kl_*`) — the shape-fitting getting better
  over time. Lower is a better fit. Since the Fase-3 hardening, this number
  can no longer look artificially good by ignoring probability the model
  puts in the wrong place — a genuinely bad fit now genuinely shows a high
  KL.
* **Summary table** (`ladder_summary_*`) — the final numbers for every
  dial setting, side by side.

## The headline result (default model)

Both quantum methods land on the **same answer** as the classical ones:
Ωm ≈ 0.26 (about a quarter of the Universe is matter) and H0 ≈ 70 (the
expansion-rate number). The quantum methods reproduce the classical
results — exactly what you want to demonstrate.

## A quick sanity button

If you ever doubt whether a "quantum" run really used the quantum circuit:

```bash
python cosmo_modular_quantum.py --sanity-check
```

It prints a table of which parts ran on the quantum circuit (⚛) vs a
normal computer (🖥), plus a self-test that the accept/reject rule behaves
correctly. The genetic optimizer has the equivalent button:

```bash
python cosmo_genetic_optimizers.py --self-test
```

---
---

# Part 2 — Technical reference

## Architecture (shared core + 4 executables + orchestrator + profiler)

```
                      cosmo_hpc_runner.py
                  (parallel orchestrator: one model
                   per process, core partitioning,
                   RAM budget, convergence plots)
                            │ launches
                            ▼
   cosmo_core.py  ← PHYSICS + DATA + STATISTICS + Aer device factory
   cosmo_noise.py ← NOISE AXIS (levels, models, analytic readout, ceiling)
        ┌────────┼────────┐         (both shared by everything)
        │        │        │
cosmo_modular_  qpu_cosmo_  cosmo_genetic_      cosmo_profiling.py
quantum.py      samplers.py optimizers.py       (RAM / VRAM / GPU-hours,
(Aer simulator, (real IBM   (CGA/QGA global      used by the two simulators
 quantumness     Quantum HW, optimization for    via --profile)
 benchmark)      SamplerV2)  the MAP + live GUI)
                     ▲
        qpu_noisy_simulation.py — same pipeline, noisy local Aer
        instead of IBM. Imports qpu_cosmo_samplers and swaps ONLY
        the connection, so the hardware module stays untouched.
```

The two axes are orthogonal: `cosmo_core` decides *what* is computed and
`cosmo_noise` decides *under what noise*. Every quantum module — and the HPC
runner — reads its noise level from that one module.

All executable scripts share the SAME physics through `cosmo_core.py`, select
CPU/GPU through its `make_simulator` factory, and write their outputs into a
**timestamped run folder** `results/run_<YYYYMMDD_HHMMSS>_<model>/` (figures,
log, per-run `resultados_config.csv`, and — with `--profile` — a
`resource_usage_*.png` and `profile_*.json`), so results from different runs
never mix. Pass an explicit `--outdir` to override. A cumulative
`resultados_config.csv` is also kept in the working directory to compare
methods across runs.

### `cosmo_core.py` — shared physics module

Strict physics ↔ sampling separation:

* **`CosmoModel`** (dataclass) — name, parameters, bounds, fiducials and
  the Friedmann function `E2(z, θ)`. Registry `MODELS` holds `lcdm`,
  `wcdm`, `cpl`, `pede`, `gede`. Adding a model = one dict entry.
* **`Posterior`** — the single contact point for the samplers. Combines a
  **CC+BAO** H(z) likelihood (Cosmic Chronometers + BAO points, treated as
  diagonal H(z) measurements) with a supernova likelihood that is EITHER
  **Pantheon 2018** (1048 SNe, diagonal errors) OR **Pantheon+ 2022** (full
  covariance matrix χ² = ΔᵀC⁻¹Δ), both with analytic M_abs marginalization
  (Goliath et al. 2001; the Pantheon+ branch is its matrix generalization).
  The luminosity distance uses a vectorized `cumulative_trapezoid` over a
  fine z-grid, valid for **any** E²(z; θ).
* **`log_prob_batch`** — vectorized batch evaluation of the log-posterior
  (used by the QVMC targets and the vectorized QMCMC kernel); handles the
  diagonal and full-covariance SNe likelihoods alike.
* **Statistics** — integrated autocorrelation τ (FFT, O(N log N), with
  Sokal automatic windowing), ESS (chains via τ_max, and Kish weights),
  and **rank-normalized split-R̂** (Vehtari et al. 2021), `RHAT_THRESHOLD
  = 1.01`. This is now the ACTIVE convergence criterion everywhere — it
  drives early-stop, the `converged` flag in every CSV, and every R̂ figure
  (previously the loops used a legacy Gelman-Rubin check at an effective
  1.05 despite split-R̂ already being implemented here; that inconsistency
  is fixed — see [Diagnostics](#diagnostics-and-correctness-fixes)).
  `fit_statistics` gives χ², χ²_red, AIC, BIC with Nelder-Mead refinement.

### `cosmo_modular_quantum.py` — simulator with switchable ablation

Five quantum/classical switchable **substitution points**, grouped by which
sampler reads them and tagged by ablation kind — **faithful** (the quantum
version reproduces the classical rule exactly; a null cell) or
**algorithmic** (a genuinely different quantum algorithm; a treatment cell):

| Sampler | Substitution point | Kind |
|---|---|---|
| **QMCMC** | proposal | algorithmic |
| **QMCMC** | acceptance | faithful |
| **QVMC** | sampling | faithful |
| **QVMC** | training | algorithmic |
| **QVMC** | normalization | faithful |

The **quantumness %** is the *uniform-weight ablation index*: the fraction of
substitution points switched to quantum, every point counting equally. (An
earlier version used hand-picked subjective weights; those have no measurable
definition and are gone. `legacy_weighted_index` still reproduces the old
numbers for continuity with older CSVs.)

Classical MCMC is a **hand-written Metropolis-Hastings** (not `emcee`):
owning every line of the transition kernel is required to swap individual
components classical↔quantum and to guarantee the classical baseline and
the quantum run share the exact same transition structure, step scale and
RNG stream. The kernel is fully **vectorized in NumPy** (one
`log_prob_batch` call scores all chains per step; 6 chains × 2000 steps in
~0.1 s). The quantum acceptance, a faithful cell, is likewise evaluated for
all chains in a single Aer job.

#### The canonical scale: per-method ablation ladders

The benchmark sweeps **each sampler along its own monotonic axis**, switching
one substitution point to quantum at a time. Under uniform weighting each
per-method index is just (#quantum points)/(#points for that sampler), so the
rungs are even fractions:

```
QMCMC ladder:  0%  →  50% (+proposal)  →  100% (+acceptance)
QVMC  ladder:  0%  →  33% (+sampling)  →  67% (+training)  →  100% (+norm)
```

Run modes (CLI `--benchmark`, or the interactive menu):

* **Single configuration** — one preset/custom config + its forced
  classical baseline (overlaid corner/marginal/KL/R̂/trace figures).
* **Benchmark** (`--benchmark`) — the two per-method ladders. **This is the
  one canonical scale.**
* **Quick Test Run** (menu) — the same ladders at small fixed sizes
  (steps 200, iters 40, nqpp 2) for a fast stability check.

Benchmark figures (labelled by per-method %): `corner_ladder_qmcmc/qvmc`
(family overlay of all rungs), `ladder_rhat_qmcmc` (R̂ vs steps),
`ladder_kl_qvmc` (KL vs iteration), `corner_ladder_1to1_*` (each rung vs
its classical baseline), `ladder_summary` (table). Single-config figures:
`corner_mcmc/qvmc`, `marginals_*`, `kl_overlay_*`, `rhat_overlay_*`,
`traces_*`. Every QVMC figure prints `nqpp`; QMCMC/QVMC figures print
steps/iterations. Colours: classical blue, QMCMC red, QVMC orange.

Because **Metropolis** acceptance is kept (so quantum methods can be shown
to *replicate* the classical ones), the faithful rungs coincide with their
classical neighbour — by design:

| Ladder step | What changes | Kind | Outcome |
|---|---|---|---|
| QMCMC 0→50 % | proposal C→Q | algorithmic | **changes** (genuine quantum proposal, calibrated to unit std) |
| QMCMC 50→100 % | acceptance C→Q | faithful | **identical** — quantum Metropolis reproduces classical |
| QVMC 0→33 % | sampling C→Q | faithful | ~identical (same trained state, only shot noise) |
| QVMC 33→67 % | training C→Q | algorithmic | **changes strongly** (exact parameter-shift reaches a lower KL) |
| QVMC 67→100 % | normalization C→Q | faithful | ~identical (faithful renormalization — the circuit is illustrative; the norm applied is the classical sum) |

```bash
# Interactive (menu)
python cosmo_modular_quantum.py

# Benchmark = per-method ladders (the canonical scale)
python cosmo_modular_quantum.py --model wcdm --benchmark --steps 4000 \
    --qvmc-iter 300 --dataset CC+Pantheon+

# Single configuration (custom component string via JSON)
python cosmo_modular_quantum.py --config \
  '{"proposal":true,"acceptance":false,"training":true,"sampling":true,"normalization":false}'

# Routing + correctness self-check
python cosmo_modular_quantum.py --sanity-check
```

**A note on `--preset N`.** The single-config presets (`--preset 0/20/45/70/90/100`)
use legacy keys kept for CSV continuity — the number in the flag is **not**
always the number reported in the output. `--preset 45` reports 40%,
`--preset 70` reports 60%, `--preset 90` reports 80% (0 and 100 do match).
If you want the flag and the report to agree, use `--config` or the
per-method `--benchmark` ladders instead, where the keys are exact.

### `cosmo_genetic_optimizers.py` — global optimization (CGA / QGA)

Phase 2 of the project: **global optimizers that locate the MAP** before or
in parallel with the samplers. The fitness is NOT a new χ² — it is the SAME
`Posterior.log_prob_batch` (CC + Pantheon+) used everywhere else, so
maximizing fitness ≡ minimizing χ² under the prior ≡ finding the MAP. Adding
a model (VC, …) needs zero changes here.

* **CGA — Classical Genetic Algorithm**, written from scratch (no DEAP),
  fully NumPy-vectorized: uniform-box init, tournament selection, blend
  (BLX) crossover, Gaussian mutation, elitism.
* **QGA — Quantum Genetic Algorithm** (Qiskit), with the same
  uniform-weight *ablation index* over three independently switchable
  operators (each an **algorithmic** substitution — a genuinely different
  operator from its classical counterpart). Each parameter is encoded in
  `n_bits` qubits (grid = 2^n_bits per axis):

  | Operator | Kind | Quantum implementation |
  |---|---|---|
  | `q_init` | algorithmic | Hadamard layer → superposition → population sampled by measurement |
  | `q_mutation` | algorithmic | RY rotation per gene-qubit, **gated by `mutation_rate`** (only selected genes get a circuit — same rate the classical operator uses) with flip probability calibrated so the expected step size matches the classical Gaussian kick (`p_flip = mutation_scale·√(2/π)/(1−2⁻ⁿᵇ)`) |
  | `q_crossover` | algorithmic | **partial SWAP** (SWAP^α, α=0.5 default) between homologous parent gene-qubits, measure register A. Identity on agreeing bits (**consensus is always preserved**); disagreeing bits inherit from either parent via genuine two-qubit interference |

  The ablation index is (#quantum operators)/3 · 100, so the QGA ladder is
  `0 → 33 → 67 → 100 %`. QGA with all operators OFF (0%) reproduces the CGA
  **bit-for-bit** — the mandatory classical baseline (a faithful cell),
  verified by `--self-test`.

  > **Fase 3 note — the crossover operator was rewritten.** The original
  > `q_crossover` circuit (CX entanglement + controlled-RY interference)
  > had a verified bug: it computed child ≈ parent_A XOR parent_B, so when
  > both parents *agreed* on a bit the child lost it with ~85% probability
  > — the opposite of what a crossover should do. Elitism masked the
  > symptom (the best individual survived regardless), but the population's
  > mean fitness suffered badly. The SWAP^α operator above replaces it and
  > is covered by a dedicated regression test
  > (`tests/test_qga_crossover.py`) asserting the truth table on the real
  > transpiled circuit. **Any QGA result generated with `q_crossover=True`
  > before this fix (rung 100%, or a custom config) should be
  > re-run.**

  [A1] All operator circuits are PARAMETRIZED and transpiled **once** at
  `__init__`; each individual's gene bits enter only through bound rotation
  angles, so the per-generation hot loop binds parameters on the cached
  transpiled template and never re-transpiles. Aer measurement seeds are
  derived deterministically from the run's `--seed` (Fase 3): two QGA runs
  with the same seed now reproduce bit-for-bit, including the quantum
  operators (previously only the classical parts were reproducible).

**Live GUI** (interactive mode only): a two-panel Matplotlib window —
phase-space scatter (population colored by fitness, converging to the MAP in
Ωm–H0) and fitness curve (best χ² and mean χ² vs generation), with dynamic
text. A snapshot is saved to the run folder.

**Integration**: fitness-weighted corner of the final population, an
all-in-one overlay of the genetic MAP + spread on the MCMC/VI corners (reuses
`plot_corner_multi`), a fitness-convergence figure, and a MAP row appended to
`resultados_config.csv` under Method = `CGA` / `QGA (q=NN%)`.

**Headless rule**: launched with arguments → batch/HPC mode, the live
animation is disabled automatically and the generational metrics go to the
log every `--log-every` generations. No arguments → interactive menu + GUI.

```bash
# Interactive (menu + live GUI)
python cosmo_genetic_optimizers.py

# Batch: classical genetic on ΛCDM
python cosmo_genetic_optimizers.py --methods cga --model lcdm --generations 80

# Batch: CGA + QGA at 67% quantumness, CC+Pantheon+
python cosmo_genetic_optimizers.py --methods cga qga --dataset CC+Pantheon+ \
  --population-size 200 --generations 120 --qga-preset 67 --n-bits 6

# Custom quantum components via JSON
python cosmo_genetic_optimizers.py --methods qga --qga-config \
  '{"q_init":true,"q_mutation":true,"q_crossover":false}'

# Correctness self-test (CGA reaches optimum; QGA(0%) == CGA)
python cosmo_genetic_optimizers.py --self-test
```

### `cosmo_hpc_runner.py` — parallel orchestrator for a compute node

The launcher you actually use on a cluster. It runs the **two simulator
pipelines in parallel** — `cosmo_modular_quantum.py` (QMCMC + QVMC ladders)
and `cosmo_genetic_optimizers.py` (CGA + QGA) — **one model per process**,
without modifying a line of either script: it just invokes their existing,
tested CLI (`--sweep-all --sweep-models <model>`). It auto-detects the node's
cores and RAM, so nothing is hard-wired to a particular machine.

**What it solves.** Four things that bite on a shared node:

1. **Real parallelism.** Python's GIL serializes the MCMC and GA loops, so
   threads buy nothing; separate processes give separate GILs plus crash
   isolation (one model blowing up doesn't take the batch with it).
2. **Oversubscription.** The heavy math (NumPy/BLAS, Aer) is already
   multi-threaded in C++. Launch *W* processes that each grab all 80 cores and
   you get W×80 threads fighting over 80 cores — slower than serial. The runner
   fixes `OMP_NUM_THREADS` & friends in each child's environment **before** the
   interpreter starts (they're read at NumPy/Qiskit import time, which is why
   this can't be done from threads), enforcing J processes × T threads ≈ cores.
3. **RAM.** It estimates each task's peak memory, refuses to admit more
   concurrent tasks than the budget allows, and **clamps the grid per model**:
   if a 4-parameter model would exceed the ceiling, only that model's `nqpp` /
   `n_bits` is lowered, while the light models keep the requested value
   (`--strict-qubits` skips instead of clamping).
4. **Measurement.** It samples peak RSS of each process *tree*, prefers the
   figure `cosmo_profiling` measured inside the child when available, and
   writes `master_profile.csv` / `.json` plus a summary table. On failure it
   tails the child's log so you see the traceback without hunting for it.

It also generates the **convergence-vs-grid figures** at the end of a sweep
(`convergence_<model>.png`, `cost_<model>.png`): each parameter ±σ vs `nqpp`,
with Planck / SH0ES reference lines, and wall time + peak RAM vs `nqpp`.
Methods that don't use the grid (MCMC/QMCMC) are detected automatically and
drawn as a single horizontal line rather than a fake curve.

```bash
# Whole node, auto-detected, both pipelines, all models:
python cosmo_hpc_runner.py --dataset CC+BAO+Pantheon+ \
    --steps 15000 --qvmc-iter 3000 --nqpp 3 \
    --generations 120 --population-size 200 --n-bits 6 \
    --threads-per-worker 8

# See the plan (and every command it would run) without running anything:
python cosmo_hpc_runner.py --dry-run

# Grid-convergence study: one task per nqpp in {2..5}, samplers only
python cosmo_hpc_runner.py --nqpp-sweep 2 5 --only-samplers --models lcdm wcdm

# The QGA analogue: sweep the genetic grid size
python cosmo_hpc_runner.py --nbits-sweep 3 6 --only-genetic --models lcdm

# Regenerate the convergence figures of a finished run, without recomputing:
python cosmo_hpc_runner.py --plot-only results/hpc_<timestamp>/
```

Useful flags: `--only-samplers` / `--only-genetic`, `--models`, `--max-parallel`,
`--threads-per-worker`, `--mem-budget-gb`, `--max-task-gb`, `--strict-qubits`,
`--gpu`, `--no-profile`, `--no-plots`.

> **RAM, cores and the per-task qubit ceiling are all auto-detected — you
> never have to set a memory-related flag to use your own machine.** The node's
> cores (`os.cpu_count()`) and total RAM (`psutil`) are read at startup, and
> the per-task qubit cap for EACH pipeline is derived straight from that: how
> many qubits fit is computed from the pipeline's own memory cost (samplers:
> 13.3 kB/state, the likelihood auxiliary arrays; genetic: 16 B/state, a plain
> statevector — the QGA never builds the samplers' grid). `--max-qubits` /
> `--max-qubits-genetic` exist only as OPTIONAL overrides to be more
> conservative than your RAM allows (a shared node, or a deliberately faster/
> coarser run) — leave them unset and the ceiling is whatever your machine
> actually has. The printed plan always says which case you're in:
> `[auto, derived from detected RAM]` or `[user override --max-qubits=N]`.
> (Fixed after an audit of this file — the two caps used to default to fixed
> numbers regardless of the detected RAM, and shared the samplers' memory
> model, needlessly clamping the QGA. See *Memory limits* below.)

### `qpu_cosmo_samplers.py` — real IBM Quantum hardware

QPU-only, via `qiskit-ibm-runtime` (SamplerV2 + Batch/Session). **No
AerSimulator. Runs no classical method** — real QPU time is scarce, and
classical baselines belong in the simulator pipeline. Hardware-driven
design differences:

| Aspect | Simulator | Real QPU |
|---|---|---|
| Quantum information | exact statevector | measured counts (shots) |
| Proposal displacement | Re(amplitudes), unit-std calibrated from a fixed constant | ⟨Z_q⟩ = 1 − 2·P(q=1), **now also unit-std calibrated** from the first hardware block (Fase 3 — see Diagnostics) |
| QVMC gradient | exact parameter-shift (shift applied to the circuit probabilities + chain rule) | **SPSA (2 evals/iter, 1 job)** |
| KL | over the full 2^n grid, ε-smoothed target — same definition as the simulator | estimated on the observed support, ε-smoothed (same definition; biased low by unobserved support, declared) |
| Acceptance | Metropolis via abs(amp0)^2 | Metropolis on CPU (sequential) |
| Error suppression | — | Dynamical Decoupling XY4 + Pauli twirling (**inert on a simulated backend** — see [DD-INERTE](#qpu_noisy_simulationpy--noisy-twin-of-the-qpu-pipeline)) |

```bash
# Plan without spending QPU time (no IBM account needed):
python qpu_cosmo_samplers.py --model wcdm --method both --dry-run

# Real run with a bounded job budget:
python qpu_cosmo_samplers.py --model lcdm --method qvmc --iters 50 \
    --shots 4096 --least-busy --max-jobs 60 --log-file qpu_run.log
```

`--max-jobs` aborts BEFORE connecting if the run would exceed the budget;
`--dry-run` validates the whole workflow with synthetic counts. **Status:**
validated in dry-run; no real-hardware run has been executed yet. Before the
first real run, confirm on the actual device that (a) the calibrated
proposal displacement lands in the healthy 0.2–0.5 acceptance band and (b)
the calibration is stable across blocks (readout drift would show up as
acceptance drifting over the run).

## QPU time estimation

Wall time on hardware is **queue-dominated**, not execution-dominated.
Per-job components:

| Component | Typical (open plan) | Notes |
|---|---|---|
| API overhead | 1–3 s | REST + PUB serialization |
| Queue | 30 s – 30 min (≈60 s default) | Backend/time-dependent; Session removes it between jobs |
| QPU execution | ~100 µs/shot/circuit | e.g. 2 circuits × 4096 shots ≈ 0.8 s |

`TimingEstimator` times every real job and replaces these defaults with
measured values, printing a projection at the end. With the defaults
(queue ≈ 60 s/job, Batch, open plan):

**QVMC-QPU (SPSA = 1 job/iteration):**

| Iterations | Jobs | API | Queue | QPU | **Estimated TOTAL** |
|---:|---:|---:|---:|---:|---:|
| 100 | 100 | ~3 min | ~1.7 h | ~13 s | **~1.7 h** |
| 500 | 500 | ~17 min | ~8.3 h | ~1 min | **~8.6 h** |
| 1000 | 1000 | ~33 min | ~17 h | ~2 min | **~17 h** |

**QMCMC-QPU (proposal blocks of 64; 4 chains → 1/16 job/step):**

| Steps | Jobs | API | Queue | QPU | **Estimated TOTAL** |
|---:|---:|---:|---:|---:|---:|
| 100 | 7 | ~14 s | ~7 min | ~1 s | **~7 min** |
| 500 | 32 | ~1 min | ~32 min | ~4 s | **~33 min** |
| 1000 | 63 | ~2 min | ~1 h | ~8 s | **~1 h** |

Practical notes: QMCMC scales much better (proposal batching amortizes the
queue across 64 proposals); for QVMC, **Session** (paid plans) removes the
inter-iteration queue; keep `--iters ≤ 50` on the open plan.

<!-- eje-de-ruido-nisq -->

## The noise axis (second ablation dimension)

Until this section the framework had ONE axis, quantumness. It now has two,
and results form a matrix rather than a ladder:

```
                        noise  →
                    none   readout   full   <backend>
    quantumness 0%    ·        ·        ·        ·
          |    33%    ·        ·        ·        ·
          ↓    67%    ·        ·        ·        ·
              100%    ·        ·        ·        ·
```

`cosmo_noise.py` is the single source of truth for this axis, the way
`cosmo_core.py` is for the physics. The four rungs:

| Level | What it models |
|---|---|
| `none` | ideal limit — every result published before this change |
| `readout` | symmetric, uniform measurement-flip error only |
| `full` | depolarizing on 1- and 2-qubit gates **plus** readout |
| `<backend>` | calibrated model of a real IBM device (`FakeBrisbane`, …) |

Available as `--noise` on all three executables and on the HPC runner, which
additionally takes `--noise-sweep none,readout,full` and treats the level as
a task dimension exactly like `--nqpp-sweep`. Every result row carries a
`noise` column, written even for ideal runs so the matrix pivots without
special cases.

### Why readout error travels separately from the NoiseModel

Readout error is a **classical channel applied after measurement**: it does
not touch the state. Aer only applies it when the circuit measures. But two
of the simulator's three amplitude reads (`hadamard_accept_log_batch` and
`_kl_batch`) are probabilities obtained from `rho` **without measuring**, so
they are *blind* to it — the `readout` column of the matrix would come out
identical to the ideal column. Not a loud failure: clean, plausible, wrong
numbers.

Measured: at p = 0.01, 0.03 and 0.05, `rho[0,0]` does not move a single
digit.

The fix is not to measure but to apply the channel in closed form. On n
qubits readout error is a tensor product of 2x2 stochastic matrices, so it
acts exactly on `diag(rho)`:

```
P_noisy = (tensor_q M_q) @ P_ideal
```

That gives the **complete noise axis, exact and free of shot noise**, at the
cost of an O(n·2^n) contraction. It matters because the shot-based route
cannot resolve the bias being measured: at a 1e-3 gate error the acceptance
bias is ~1.4e-4, and separating that from sampling noise would take ~1e7
shots *per acceptance evaluation*.

`NoiseSpec.simulator_kwargs(counts_route=...)` decides which model goes to
Aer, because getting it wrong either double-counts readout error or makes it
vanish silently.

### Density matrix, not shots

Noisy simulation needs either a density matrix (`16·4^n` bytes) or shot
trajectories (`shots·2^n` in time). Measured on the project's real ansatz
(3 layers, B=2 bindings, 4096 shots, FakeBrisbane):

| qubits | density matrix | trajectories | rho size |
|---|---|---|---|
| 6 | **1.66 s** | 5.31 s | 0.1 MB |
| 10 | **3.77 s** | 22.4 s | 16 MB |
| 12 | **34.4 s** | 101 s | 256 MB |
| 13 | **216 s** | 333 s | 1.0 GB |

The density matrix **wins on time everywhere it fits** (1.5x–15x), because it
evolves once and then samples, while trajectories re-simulate the whole
circuit per shot. The advantage is largest exactly where QMCMC operates (few
qubits, many bindings: 4.2 s vs 52 s for a 64-proposal block).

**Shots do not buy qubits.** Above ~14 qubits neither route is usable — one
runs out of memory, the other out of time (CPL at nqpp=6 is ~114 h *per
job*). Hence `MAX_NOISY_QUBITS = 13`, a **time** ceiling as much as a memory
one, which is why it is an explicit constant and not derived from detected
RAM like the other two ceilings. More RAM does not move it.

| Model | d | nqpp=3 | nqpp=4 | nqpp=5 | nqpp=6 |
|---|---|---|---|---|---|
| LCDM, PEDE | 2 | 6 ok | 8 ok | 10 ok | 12 ok |
| wCDM, GEDE | 3 | 9 ok | 12 ok | 15 no | 18 no |
| CPL | 4 | 12 ok | 16 no | 20 no | 24 no |

QMCMC is absent from that table because its proposal engine uses
`n_qubits = max(2, d)` — 2 to 4 qubits, essentially free under any noise
model, for every model and every nqpp.

### The proposal changes operator under noise (and what that costs)

`_raw_block` reads `Re(psi)·sign(Im(psi))`, which is phase-sensitive and
therefore undefined for a mixed state. It is the ONLY one of the simulator's
reads that noise genuinely breaks; the other two are probabilities and
survive exactly as `rho[0,0]` and `diag(rho)`.

Under noise the proposal must be read by measurement instead
(`<Z_q> = 1 - 2·P(q=1)`, the same rule the QPU pipeline runs on hardware).
That is a **different operator**, not a noisy version of the same one, so
comparing the ideal column (amplitudes) against the noisy ones (counts)
would confound two effects. `--proposal-route counts` is therefore available
**in the ideal rung too**: the noise axis is measured inside one route, and
the route change stays a separate, controlled comparison.

### Results that came out of building this

**The unit-std calibration absorbs 100% of uniform readout error.** With a
symmetric flip probability p, `<Z_q>' = (1-2p)·<Z_q>` — a scalar rescaling —
and the calibration divides it straight out. Verified to 8e-15 **even at
p = 0.20**. Gate noise is not absorbed (0.109 for `full`). So in the proposal
row, the `readout` column comes out **identical** to the ideal-counts column.
That is a provable invariance, not a regression, and it is pinned by a
regression test so it is never mistaken for one.

**Readout error dominates the rejection tail of the acceptance.** An exact
acceptance of A = 0.00248 becomes 0.0323 at p = 0.03: readout puts a floor of
~p on the acceptance, so moves that should almost always be rejected are
accepted **13x too often**. This is far larger than the gate-noise effect and
runs against the intuition that noise degrades smoothly.

**The gate-noise bias on the acceptance is linear and directional.**
Depolarizing drives rho toward I/2, so the bias is `lambda·(1/2 - A)`: the
acceptance is pulled **toward 1/2**, over-accepting bad moves and
under-accepting good ones. Relative bias scales linearly with the gate error
rate (2.60% at 1e-3, 25.98% at 1e-2).

**The QVMC normalization component is immune to this axis by construction.**
`quantum_amplitude_normalization` runs its circuit but discards the result —
the returned value is the exact classical sum. The QVMC 100% rung therefore
cannot degrade because of it, and any degradation seen between 67% and 100%
comes from somewhere else.

### `qpu_noisy_simulation.py` — noisy twin of the QPU pipeline

Runs the hardware pipeline against a noisy local simulator instead of IBM.
It does **not** copy `qpu_cosmo_samplers.py` — it imports it and replaces one
piece, the connection. A copied file would diverge from the original at the
first fix applied to one and not the other, silently.

Its value is twofold: it is the **first end-to-end execution** of
`qpu_cosmo_samplers.py` (previously validated only in `--dry-run`, i.e. with
synthetic uniform counts that exercise shapes, not physics), and it gives the
**shot-noise** version of the axis, whereas the `cosmo_modular_quantum` axis
computes exact probabilities to isolate decoherence from sampling.

```bash
python qpu_noisy_simulation.py --model lcdm --method qmcmc --noise full
python qpu_noisy_simulation.py --model wcdm --method qvmc --noise FakeBrisbane
```

> **[DD-INERTE] The error suppression is inert on a simulated backend.**
> `QPUConnection` enables dynamical decoupling XY4 and Pauli twirling. On real
> hardware those act. In qiskit-ibm-runtime's local testing mode they are
> **discarded**, announced only by a `UserWarning` that is lost in a long run's
> logs. So a simulated noisy run measures *noise without error suppression*
> while hardware measures *noise with it* — not the same experiment. It is a
> defensible reading (a **lower bound** on achievable hardware quality) but it
> must be recorded, not inferred. `qpu_noisy_simulation.py` therefore avoids
> `SamplerV2` entirely and states this in the run log.

### Bugs this axis surfaced

Three of them, all silent — they returned clean, plausible numbers.

* **`[N1]`** — `AerSimulator(method='statevector', noise_model=...)` raises
  nothing. It runs, reports `COMPLETED`, and returns the **noiseless**
  result. An entire noisy campaign would have come back ideal with no signal
  at all. `make_simulator` now rejects the pairing.
* **`[B-RO]`** — an `add_all_qubit_readout_error` appears in `to_dict()`
  **without** a `gate_qubits` key. Reading that absence as `[[0]]` degrades
  the channel to a single qubit; the symptom matched exactly at n=1 and
  diverged only as n and p grew.
* **`[B-RECON]`** — rebuilding a readout-free model with
  `NoiseModel.from_dict()` is **lossy** for calibrated backends: 6.2e-4 of
  drift in rho against FakeBrisbane at 3 qubits, while synthetic rungs showed
  0.0. The reconstruction was removed entirely (it was also using an API
  deprecated since qiskit-aer 0.15).

`noise_feasibility_probe.py` reproduces the measurements behind this section.


## Diagnostics and correctness fixes

`--sanity-check` prints an acceptance regression test, the proposal
statistics, and a per-preset engine map (Qiskit/Aer vs NumPy/SciPy) with a
live routing trace.

### Fase 3 hardening (convergence, KL, gradients, reproducibility)

The most recent audit round (adversarial code review, all fixes verified
with executed numerical checks, not just read) closed six specific gaps.
**Any figure, CSV row, or convergence flag produced before this round is not
directly comparable to a new one** — regenerate before citing.

* **QGA crossover bug (the significant one).** See the operator table above
  — `q_crossover` computed the XOR of the parents' bits, destroying
  consensus. Fixed with a consensus-preserving partial-SWAP. Affects rung
  100% and any custom config with `q_crossover=True`; re-run those.
* **Convergence criterion wasn't wired up.** `split_rhat` /
  `RHAT_THRESHOLD = 1.01` were implemented and tested but never called from
  the sampling loops, which used a legacy Gelman-Rubin check at an
  effective 1.05 instead. Now split-R̂ drives early-stop, the `converged`
  flag, and every R̂ figure, in both the simulator and the QPU pipeline.
  **CSVs with `converged=True` from before this fix used the looser
  criterion — don't compare that flag across the change.**
* **Comparisons could silently drift.** The forced classical baseline could
  stop at a different chain length than its quantum counterpart (early-stop
  triggering independently on each side), and the VI part of the
  comparison could start from a different initial state on each side
  (inherited RNG state from whatever the preceding sampler consumed).
  Both are now fixed: comparisons force equal chain lengths, and every
  sampler is explicitly re-seeded right before it starts, so a same-seed
  comparison shares its VI initialization bit-for-bit.
* **The KL objective could ignore leaked probability mass.** The training
  target was previously masked and renormalized to the support where the
  reference posterior is non-negligible — so a trained state that put most
  of its probability *outside* that support could still report a KL near
  zero. It's now computed against an ε-smoothed target over the *full*
  support (matching the definition the QPU pipeline already used), so
  leaked mass is penalized and the simulator and QPU KL values are directly
  comparable.
* **The QVMC training gradient was a biased finite difference, not exact
  parameter-shift.** Parameter-shift is only exact on quantities that are
  themselves circuit expectation values; applying it directly to the KL
  (which is not one) had a measured ~10% relative error. It's now applied
  to the circuit *probabilities* (where it is exact) with the KL gradient
  assembled by the chain rule — same circuit cost, gradient exact to
  numerical precision.
* **Quantum mutation ignored `mutation_rate` and its strength wasn't
  calibrated.** See the operator table — fixed by gating on the rate and
  calibrating the flip probability to match the classical operator's
  expected step size, so the mutation rung of the ladder isolates the
  *operator*, not an uncontrolled change in intensity.
* **Reproducibility gaps.** QGA's quantum operators now derive their Aer
  seed from the run seed (previously irreproducible even with a fixed
  `--seed`); the simulator's quantum-proposal engine calibrates its
  zero-mean/unit-std normalization once from a dedicated block instead of
  per-block (previously each block was normalized against itself, making
  displacements within a block weakly correlated); the same fix was applied
  to the QPU proposal engine, which previously had no calibration at all
  (raw ⟨Z_q⟩ has std ≈ 0.05–0.12 on 128 shots — roughly 10× smaller than
  the scale `step_frac` is tuned for, which on real hardware would have
  driven acceptance to ≈1 with barely-moving chains).

### Orchestrator audit (`cosmo_hpc_runner.py`)

Reviewed after the Fase 3 round. One real defect, fixed and verified:

* **The QGA memory estimate used the samplers' formula.** Both the
  orchestrator and `cosmo_genetic_optimizers.py`'s own `--max-qubits`
  validation costed the QGA at 2^(n_bits·d) × 13.3 kB/state — the samplers
  model, where the likelihood really does build `(n_states, N_data)` auxiliary
  arrays over the grid. The QGA does no such thing (fitness is scored on the
  population), so its true cost is one plain statevector at 16 B/state:
  an **~830x over-estimate**. The visible consequence was the clamp lowering
  CPL from `n_bits=6` to `4` — a 64-level grid per axis down to 16 — to protect
  ~268 MB it thought were 223 GB. Nothing crashed and no result was wrong; the
  cost was resolution and parallelism (each genetic task also reserved 3.5 GB
  of phantom budget, admitting fewer concurrent tasks than the node could take).
  Fixed with per-pipeline constants and a per-process baseline (~250 MB) the
  estimate had been ignoring entirely. Verified end-to-end: CPL at 24 q now
  runs, estimated 518 MB vs 510 MB measured.

* **The qubit ceiling was a fixed number, not actually tied to the RAM the
  code already detects.** A same-day follow-up: `psutil` was already reading
  the node's RAM, but `--max-qubits` / `--max-qubits-genetic` were fixed
  defaults (18 / 26) layered on top, so using your own detected RAM required
  manually raising a flag. Both now default to `None` ("auto"): with nothing
  set, the ceiling is derived purely from detected RAM per pipeline (~20 q
  samplers / ~29 q genetic at a 14 GB budget); the flags remain as optional
  overrides to be more conservative. The printed plan states which case
  applies. Verified sane on both ends (14 GB laptop -> 20/29 q; ~106 GB HPC
  node -> 22/~34 q — no runaway multi-day plan from leaving it on auto).

### Fase 2 audit (original hardening — kept for the record)

* **Apparent "identical results across quantumness".** Not a routing bug:
  each sampler only reads its own components, the `sampling` toggle draws
  from the *same trained state* (shot noise only), identical re-seeding
  makes same-code-path rungs bit-identical, and the Metropolis acceptance
  is a faithful reproduction. The per-method ladders make the axis
  monotonic; the remaining coincidences are the *replication* result.

* **Inverted quantum acceptance (the original convergence bug).** The old
  Hadamard-test readout *decreased* with Δ = lp_prop − lp_cur (accepted
  worse moves), so quantum-acceptance chains drifted to the box edges
  (Ωm ≈ 0.37, H0 ≈ 77). It now encodes Metropolis min(1, e^Δ) as the |0⟩
  amplitude of an RY rotation (verified monotonic, matching Metropolis).
  Post-fix every quantumness level agrees with the classical baseline.

* **Proposal calibration.** The quantum displacement is zero-mean but had
  std ≈ 0.35 (~3× smaller than the classical N(0,1)), pushing acceptance
  to ≈ 0.80 (slow mixing). Each block is normalized to unit std →
  acceptance ≈ 0.5. (Fase 3 replaced the per-block normalization with a
  once-off calibration — see above.)

* **QVMC optimizer & the high-quantumness divergence.** Two distinct
  issues were untangled here. (1) The original "stalled-KL" symptom was the
  *classical* COBYLA baseline, not the quantum trainer. (2) At high
  quantumness (the rungs that switch quantum *training* on), the KL would
  fall to a minimum near iteration ~150 and then **creep back up** to ~2,
  collapsing the distribution and crashing the ESS — the tuning
  (lr0=0.05, decay=0.02) had been calibrated for nqpp=3 (~42 ansatz
  angles), but at nqpp=6 the ansatz has many more angles and a larger
  gradient, so a fixed step overshoots near the optimum. Fixed with three
  reinforcing measures: **(a) gradient-norm clipping** (one step can't
  explode just because there are more angles, decoupling the step from
  nqpp), **(b) learning-rate decay scaled by the number of angles** (larger
  ansätze cool faster), and **(c) best-so-far selection** (the returned φ is
  the lowest-KL iterate ever seen, not the last — so even a wobbly tail
  reports the true minimum). The creep-up is gone; re-verified after the
  Fase 3 exact-gradient fix (which changes the effective step scale) with
  zero KL upticks across 150 iterations on the reference model. Note that
  the absolute KL floor depends on grid resolution (see adaptive grid
  below): with a coarse grid both classical and quantum plateau at a higher
  KL, which is a *resolution* limit, not an optimizer failure.

* **QVMC adaptive grid (resolving a smooth posterior).** QVMC represents
  the posterior as a probability mass function on a discrete 2^nqpp grid.
  Spanning the full (wide) `sample_box`, the cosmological posterior is far
  narrower than the grid spacing, so it collapses onto ~1–3 cells and can
  never look smooth — independent of the number of iterations. The grid is
  now **adaptive**: a fast classical pre-fit (`estimate_grid_window`)
  locates the posterior mode and width, and the grid is centered on a
  zoomed window [mode − k·σ, mode + k·σ] with the half-width k scaled to
  the grid size (k ≈ (2^nqpp − 1)/6, clipped to [2, 5]) so a small grid
  zooms in tightly and a larger one widens out. The window is computed once
  and shared by every QVMC rung and the classical-VI baseline (fair
  comparison). Effect at nqpp = 3/4/5: occupied cells go from ~3 (full box)
  to ~13/40/49, and the discretized target becomes a clean bell curve. Any
  residual lumpiness in the QVMC *samples* is then the variational
  ansatz/training (more iterations, more layers), not the grid.

### Hardening from adversarial testing

A round of deliberate break-it testing (feeding deliberately bad inputs and
edge cases) surfaced and fixed several robustness gaps that would only bite
on an HPC queue:

* **Input validation.** Out-of-range numeric args (negative `--steps`,
  `--nqpp 0`, `--chains 0`, `--seed -1`, `population-size 0`, `n-bits 0`,
  malformed `--config` JSON) used to crash deep inside NumPy/Qiskit with
  cryptic messages. They now fail fast with an actionable note before any
  work starts.
* **Exponential-memory guard.** A too-large `nqpp` (e.g. nqpp=6 with CPL =
  2^24 states) would attempt a multi-GB/TB allocation and get OOM-killed.
  The `--max-qubits` cap (default 18) refuses it with the per-model limit;
  raise the cap explicitly on a bigger machine (see *Memory limits* below).
* **Corrupt data rows.** Data files with NaN/inf or non-positive sigma used
  to load silently and poison every χ². They are now dropped with a warning.
* **Concurrent CSV writes.** Multiple SLURM array jobs appending to the same
  cumulative CSV could interleave and corrupt it. Writes are now guarded by a
  POSIX file lock (verified with real multiprocessing); duplicate headers and
  torn rows are gone.
* **Pantheon+ error clarity.** "Files present but unloadable" (bad/singular
  covariance) is now reported distinctly from "files missing" — different
  problems, different fixes.

## Installation and data

```bash
pip install -r requirements.txt
```

### Datasets

The available datasets (pass via `--dataset`, or pick in the menu):

| Key | What it is | Files needed |
|-----|-----------|--------------|
| `CC+BAO` | Combined Cosmic Chronometers + BAO H(z) measurements (diagonal) | embedded, or `cosmic_chronometers.txt` |
| `Pantheon` | Pantheon 2018, 1048 SNe Ia, **diagonal** errors | `pantheon_full_parameters.txt` (`name zcmb zhel dz mb dmb`) |
| `Pantheon+` | Pantheon+ 2022, **full covariance matrix** χ²=ΔᵀC⁻¹Δ | `Pantheon+SH0ES.dat` **and** `Pantheon+SH0ES_STAT+SYS.cov` |
| `CC+BAO+Pantheon` | the two above combined | both sets |
| `CC+BAO+Pantheon+` | CC+BAO with the full-covariance Pantheon+ | CC + Pantheon+ files |

Legacy aliases `CC` → `CC+BAO` and `CC+Pantheon+` → `CC+BAO+Pantheon` are
still accepted so old commands and CSVs keep working.

The difference between **Pantheon** and **Pantheon+** is statistical, not
just cosmetic: Pantheon+ ships a full N×N covariance matrix (correlated
systematics), so its χ² is the quadratic form ΔᵀC⁻¹Δ rather than a sum of
independent terms. The covariance code is ready and waiting for the
`.dat` + `.cov` files; if they are not present, the `Pantheon+` options
simply do not appear in the menu.

* **IBM Quantum**: save your account once with
  `QiskitRuntimeService.save_account(channel="ibm_quantum_platform",
  token="...")` or pass `--token`.
* **Live GUI** (`cosmo_genetic_optimizers.py` interactive mode): needs an
  interactive Matplotlib backend (Tk or Qt). On WSL/Ubuntu install
  `sudo apt install python3-tk`; over SSH it requires X-forwarding. If no GUI
  backend is found the script falls back to saving static figures (no crash),
  and in batch/HPC mode (any CLI argument) the live window is disabled by
  design.

### Memory limits: how high can `nqpp` (and `n_bits`) go?

> **The samplers and the QGA have DIFFERENT memory models** — an audit of
> `cosmo_hpc_runner.py` found that the same formula was being applied to both,
> over-estimating the QGA by ~830x and needlessly clamping `n_bits` for
> 4-parameter models (CPL was silently dropped from a 64-level grid per axis to
> 16 to "save" memory it was never going to use). The table below is the
> **samplers** model (QVMC/VI); the QGA model follows it.

#### Samplers (QVMC / classical VI): `nqpp`

The variational methods discretize the posterior on a statevector
grid of **2^(nqpp·d)** states, where `d` is the number of model parameters.
Both the time and the memory grow **exponentially** in `nqpp·d`: the grid
itself, plus the auxiliary arrays the likelihood builds over it, roughly
**quadruple with each extra qubit**. The scripts enforce a safety cap,
`--max-qubits` (default **18**), and refuse a run that would exceed it with a
clear message — so a typo can't silently trigger a 200 GB allocation on a
shared node.

The cap is on the **total** qubits `nqpp·d`, so the per-model `nqpp` limit
depends on `d`. Worst-case auxiliary memory (combined dataset with ~1048 SNe):

| Model | d | `nqpp` ≤ 18 q (laptop, default) | ≤ 22 q (workstation ~64 GB) | ≤ 24 q (HPC node ~256 GB) |
|-------|---|------|------|------|
| ΛCDM, PEDE | 2 | **9** | 11 | 12 |
| wCDM, GEDE | 3 | **6** | 7 | 8 |
| CPL | 4 | **4** | 5 | 6 |

Approximate worst-case memory by total qubits: 12 q ≈ 54 MB · 16 q ≈ 0.9 GB ·
18 q ≈ 3.5 GB · 20 q ≈ 14 GB · 22 q ≈ 56 GB · 24 q ≈ 224 GB.

> **QVMC training pays an extra multiplier on top of the grid.** The above
> is the cost of the target/state vector itself. When `training` is quantum
> (the ladder's 67%/100% rungs), the gradient step batches **2·n_φ**
> statevectors in one Aer job (n_φ = ansatz parameter count), so peak
> memory during training is roughly `2·n_φ ×` the single-statevector cost
> in the table above — for CPL (d=4) at `nqpp=6` this is already ~90 GB,
> outside laptop and workstation range even though the *sampling*-only
> cost at that resolution might look affordable. If a training rung OOMs
> while sampling-only rungs at the same `nqpp` don't, this is why — lower
> `nqpp` for that model rather than raising `--max-qubits` further.

> **A noisy run has a third, much lower ceiling.** With `--noise` on, simulation switches to a density matrix: `16·4^n` bytes *added to* the grid cost above, not replacing it (1.0 GB of rho at 13 qubits, 4.0 GB at 14). The binding limit is TIME rather than memory — 216 s per job at 13 qubits, ~14 min at 14 — and the shot-trajectory alternative, which would fit in RAM, is slower still. `cosmo_noise.MAX_NOISY_QUBITS = 13` is therefore a hard constant that more RAM does not raise. The HPC runner applies it automatically and clamps heavy models per model, the same way the other two ceilings do.

To go above the default cap on a bigger machine, raise it explicitly:

```bash
# CPL at nqpp=6 (24 qubits, ~224 GB) on an HPC node
python cosmo_modular_quantum.py --benchmark --model cpl --nqpp 6 \
    --max-qubits 24 --dataset CC+BAO+Pantheon+ --gpu --profile
```

#### Genetic (QGA): `n_bits`

The QGA is **much cheaper** than the equivalent `nqpp`, because it never
evaluates the likelihood over the grid: fitness is scored on the *population*
(`pop_size × d`), not on 2^(n_bits·d) points. Only the quantum initialization
is `d·n_bits` qubits wide; mutation uses `n_bits` and crossover `2·n_bits`,
both independent of `d`. So the cost is one plain statevector — **16 bytes per
state** (one complex128 amplitude), not the 13.3 kB/state of the samplers.

Measured (peak RSS, Aer statevector, `H^n + measure_all`): 20 q → 17.5 MB ·
22 q → 65.6 MB · 24 q → 258 MB, i.e. it converges to the 16 B/state theory.
Add ~250 MB of fixed per-process cost (interpreter + numpy/qiskit/matplotlib).

| Total qubits `n_bits·d` | 20 | 24 | 26 | 28 | 30 |
|---|---|---|---|---|---|
| Statevector | 17 MB | 268 MB | 1.1 GB | 4.3 GB | 17 GB |

When run through the orchestrator, the cap is **auto-derived from your
detected RAM** (see above) — no flag needed. At a typical 14 GB laptop budget
this alone gives ~29 q (comfortably above CPL's `n_bits=6` = 24 q, so every
model runs at the same resolution); on Nicte-Ha's ~106 GB it is ~34 q. Running
`cosmo_genetic_optimizers.py` directly (not through the orchestrator) uses its
own fixed default of **26 q** (≈1.1 GB), since a standalone script has no
"aggregate node" to detect. `--max-qubits` (standalone) /
`--max-qubits-genetic` (orchestrator) override either default if you want to
be more conservative.

**On real quantum hardware** (`qpu_cosmo_samplers.py`) this RAM limit does
**not** apply — a QPU never materializes the statevector in memory, it holds
the qubits physically. There the constraints are different (qubit count,
circuit depth, and noise), not classical memory. The `--max-qubits` cap is
purely a guard for the **classical statevector simulation** used everywhere
else.

### Qiskit version compatibility (important)

Qiskit and Qiskit-Aer must come from the **same generation**. Mixing them
raises errors such as `ImportError: cannot import name 'convert_to_target'
from 'qiskit.providers'` (Aer built against a different Qiskit). The
**verified-good combination** used to produce the reference results (and
pinned in `requirements.txt`) is the Qiskit 2.x line:

```bash
pip install "qiskit==2.4.2" "qiskit-aer==0.17.2" "numpy==1.26.4"
```

The older 1.x line also works if Aer matches it (`qiskit==1.0.2` with
`qiskit-aer==0.14.2`, `numpy<2`). `numpy<2` is required either way (Aer and
cuQuantum need it). See `requirements.txt` for the full pinned set and
`requirements-gpu.txt` for the CUDA extras.

### GPU acceleration and resource profiling

Both samplers and the genetic module accept `--gpu` (use an Aer GPU device if
present; otherwise fall back to CPU) and `--profile` (record peak host RAM,
GPU VRAM, CPU/GPU utilization, wall time and GPU-hours, and save a
`resource_usage_*.png` figure plus a `profile_*.json` next to the run).

```bash
# CPU run with profiling
python cosmo_modular_quantum.py --model lcdm --preset 45 --profile

# On a CUDA node: install the cuQuantum extras, then add --gpu
pip install -r requirements.txt -r requirements-gpu.txt
python cosmo_modular_quantum.py --benchmark --model cpl --gpu --profile
```

**GPU backend note.** On the development machine the Aer GPU statevector path
is provided by NVIDIA **cuQuantum / cuStateVec** (`cuquantum-cu12`,
`custatevec-cu12`), against which `qiskit-aer==0.17.2` is built — **not** by a
separate `qiskit-aer-gpu` wheel. `make_simulator` requests cuStateVec when
available and is otherwise backend-agnostic: `AerSimulator().
available_devices()` reports `'GPU'` whenever a usable CUDA device is present,
so `--gpu` engages it and `--profile`'s NVML/`nvidia-smi` sampling records the
VRAM and GPU-hours. On CPU-only nodes (e.g. Nicte-Ha) omit the GPU extras and
everything runs on CPU automatically.

The device is auto-detected via `AerSimulator().available_devices()`; no
source change is needed. The `cosmo_profiling.py` module is standalone and
also usable on its own. GPU enablement and the experiment-magnitude estimates
are documented in detail in the technical habilitation dossier.

### Output layout

Every run writes to a timestamped folder
`results/run_<YYYYMMDD_HHMMSS>_<model>/` containing the figures, the log, a
per-run `resultados_config.csv`, and (with `--profile`) the resource figure
and JSON. A cumulative `resultados_config.csv` in the working directory
collects all runs across models for cross-comparison. Pass an explicit
`--outdir` to override the folder.

## Adding a new model (e.g. Variable Curvature)

```python
# in cosmo_core.py
def _E2_vc(z, th):
    Om, H0, Ok1 = th[0], th[1], th[2]
    ...  # your E²(z)

MODELS['vc'] = CosmoModel(
    name='vc', label='Variable Curvature',
    param_names=['Om', 'H0', 'Ok1'],
    param_latex=[r'\Omega_m', r'H_0', r'\Omega_{k,1}'],
    bounds=[(0.05, 0.7), (50, 90), (-0.3, 0.3)],
    sample_box=[(0.1, 0.6), (60, 80), (-0.2, 0.2)],
    fiducial=[0.31, 67.7, 0.0], E2=_E2_vc)
```

Both samplers recognize it via `--model vc`. The simulator runs the full
ladder + classical baseline; the QPU dispatches it quantum-only.

## References

* Sarracino et al. (2025) — QMCMC proposal circuit.
* Goliath et al. (2001) — analytic M_abs marginalization.
* Brout et al. (2022) — Pantheon+ data and covariance.
* Li & Shafieloo (2019, 2020) — PEDE / GEDE dark-energy models.
* Planck Collaboration (2018) — Gaussian priors (Ωm, H0).
* Spall (1998) — SPSA.
* Gelman & Rubin (1992) — original R̂ diagnostic.
* Vehtari, Gelman, Simpson, Carpenter & Bürkner (2021) — rank-normalized
  split-R̂ and the 1.01 convergence threshold (the active criterion since
  the Fase 3 hardening).
* Sokal (1996) — integrated autocorrelation time and automatic windowing.
* Foreman-Mackey et al. (2013, 2016) — emcee autocorrelation, corner.py.

## Reproducibility

This project follows the modern reproducibility checklist for computational
physics:

* **Pinned environment.** `requirements.txt` is pinned to the verified
  working environment (Qiskit 2.4.2 / Aer 0.17.2 / numpy 1.26.4); CUDA extras
  live in `requirements-gpu.txt`.
* **Tests + CI.** `tests/` holds 62 tests (33 through Fase 3, plus 29 added in Fase 4 for the noise axis — the analytic readout map against Aer counts, the `[N1]` guard, the ideal rung's bit-for-bit reproducibility, the proposal's readout invariance, and the runner's third memory model): a pure-NumPy correctness floor
  (physics, statistics, the ablation framework, the QPU helpers) that runs
  without Qiskit, plus dedicated Qiskit-Aer regression tests — added in the
  Fase 3 round — that verify the QGA crossover truth table, the exact
  training gradient, the KL leakage penalty, and the calibration/
  reproducibility of both proposal engines on real circuits. The Qiskit-Aer
  tests skip automatically if Qiskit/Aer are absent, so make sure the CI
  environment installs them if you want CI to catch a regression on that
  side too. `pytest` runs the full suite; `.github/workflows/tests.yml` runs
  it on every push. Faithful (null) ablation cells are encoded as falsifiable
  tests.
* **Data provenance.** `data_manifest.py` records SHA256 checksums and the
  source of every dataset; run `python data_manifest.py --generate` after
  placing the data files, and `--verify` to check integrity. The data files
  themselves are not redistributed — download them from the official releases
  listed in the manifest.
* **License & citation.** `LICENSE` (MIT) and `CITATION.cff` (add your ORCID
  and the archived DOI once you mint a release on Zenodo).
* **Determinism.** Runs are seeded (`--seed`, default 42). QGA's quantum
  operators and comparison-vs-baseline runs are now re-seeded explicitly at
  the point each stochastic component starts (Fase 3), so a fixed seed
  reproduces the quantum parts too, not just the classical ones. Note that
  float-reduction order under multi-threaded BLAS can make bit-for-bit
  identity across machines fragile; the "QGA(0%) == CGA" claim is verified at
  fixed thread count and is otherwise statistically (not bitwise) identical.
