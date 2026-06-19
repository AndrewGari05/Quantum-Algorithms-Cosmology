# Quantum Algorithms for Cosmology — Classical vs Quantum Bayesian Inference

A toolkit that fits cosmological models (ΛCDM, wCDM, CPL, PEDE, GEDE) to
real data (combined CC+BAO H(z) measurements, and Type Ia supernovae —
Pantheon 2018 or Pantheon+ 2022) using **classical** and **quantum**
sampling algorithms, and compares them head to head.

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

## What does "quantumness %" mean?

Each method is built from swappable parts. You can run each part on a
normal computer ("classical") or on a quantum circuit ("quantum"). The
**quantumness %** is simply *how many of the parts are running on the
quantum circuit*, from 0% (all classical) to 100% (all quantum).

There are **two separate dials**, one per method, because the two methods
are made of different parts:

```
QMCMC dial:  0%  →  44% (quantum step direction)  →  100% (quantum accept/reject)
QVMC  dial:  0%  →  46% (quantum draw)  →  82% (quantum fitting)  →  100% (quantum normalize)
```

Turning a dial up adds one more quantum part. That's it.

## The one thing that surprises everyone

When you turn a dial up, **sometimes the answer changes and sometimes it
stays exactly the same** — and that is *expected*, not a bug:

* Some quantum parts are genuinely new algorithms, so they change the
  answer (the quantum *step direction* in QMCMC; the quantum *fitting* in
  QVMC).
* Other quantum parts are quantum *re-implementations of the exact same
  rule* the classical computer uses. Those give an identical answer **on
  purpose** — that identity is the proof that "the quantum version
  reproduces the classical one."

So when two neighboring dial settings look identical, that's the project
succeeding at its goal, not failing.

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

There is also a **global optimizer** that hunts for the single best-fit point
(the MAP) using genetic algorithms — classical (CGA) and quantum (QGA) — with
a live animation of the population converging:

```bash
python cosmo_genetic_optimizers.py
```

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

## How to read the pictures

* **Corner plots** (`corner_ladder_*`) — estimated values and their
  uncertainty. Each blob/contour is one method at one dial setting. **If
  the blobs sit on top of each other, the methods agree.** Dashed lines
  mark the reference ("Planck") values.
* **Convergence curve** (`ladder_rhat_*`) — whether the wandering search
  has "settled down." Lower is better; below the dashed line means
  "settled."
* **Training curve** (`ladder_kl_*`) — the shape-fitting getting better
  over time. Lower is a better fit.
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
correctly.

---
---

# Part 2 — Technical reference

## Architecture (shared core + 4 executables + profiler)

```
            cosmo_core.py   ← PHYSICS + DATA + STATISTICS + Aer device factory
        ┌────────┼────────┐         (shared by everything)
        │        │        │
cosmo_modular_  qpu_cosmo_  cosmo_genetic_      cosmo_profiling.py
quantum.py      samplers.py optimizers.py       (RAM / VRAM / GPU-hours,
(Aer simulator, (real IBM   (CGA/QGA global      used by the two simulators
 quantumness     Quantum HW, optimization for    via --profile)
 benchmark)      SamplerV2)  the MAP + live GUI)
```

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
* **Statistics** — autocorrelation τ (FFT, O(N log N)), ESS (chains and
  Kish weights), Gelman-Rubin (max over parameters), `fit_statistics`
  (χ², χ²_red, AIC, BIC with Nelder-Mead refinement).

### `cosmo_modular_quantum.py` — simulator with switchable quantumness

Five quantum/classical switchable components, grouped by which sampler
reads them:

| Sampler | Components (weights) |
|---|---|
| **QMCMC** | proposal (20), acceptance (25) |
| **QVMC** | sampling (25), training (20), normalization (10) |

Classical MCMC is a **hand-written Metropolis-Hastings** (not `emcee`):
owning every line of the transition kernel is required to swap individual
components classical↔quantum and to guarantee the classical baseline and
the quantum run share the exact same transition structure, step scale and
RNG stream. The kernel is fully **vectorized in NumPy** (one
`log_prob_batch` call scores all chains per step; 6 chains × 2000 steps in
~0.1 s).

#### The canonical quantumness scale: per-method ladders

The benchmark sweeps **each sampler along its own monotonic axis**, adding
one quantum component at a time. Each per-method % counts only that
sampler's component weights (QMCMC total 45, QVMC total 55):

```
QMCMC ladder:  0%  →  44% (+proposal)  →  100% (+acceptance)
QVMC  ladder:  0%  →  46% (+sampling)  →  82% (+training)  →  100% (+norm)
```

Run modes (CLI `--benchmark`, or the interactive menu):

* **Single configuration** — one preset/custom config + its forced
  classical baseline (overlaid corner/marginal/KL/R̂/trace figures).
* **Benchmark** (`--benchmark`) — the two per-method ladders. **This is the
  one canonical quantumness scale.**
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
to *replicate* the classical ones), some adjacent rungs coincide — by
design:

| Ladder step | What changes | Outcome |
|---|---|---|
| QMCMC 0→44 % | proposal C→Q | **changes** (genuine quantum proposal) |
| QMCMC 44→100 % | acceptance C→Q | **identical** — quantum Metropolis reproduces classical |
| QVMC 0→46 % | sampling C→Q | ~identical (same trained state, only shot noise) |
| QVMC 46→82 % | training C→Q | **changes strongly** (param-shift reaches a far lower KL) |
| QVMC 82→100 % | normalization C→Q | ~identical (faithful renormalization) |

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

### `cosmo_genetic_optimizers.py` — global optimization (CGA / QGA)

Phase 2 of the project: **global optimizers that locate the MAP** before or
in parallel with the samplers. The fitness is NOT a new χ² — it is the SAME
`Posterior.log_prob_batch` (CC + Pantheon+) used everywhere else, so
maximizing fitness ≡ minimizing χ² under the prior ≡ finding the MAP. Adding
a model (VC, …) needs zero changes here.

* **CGA — Classical Genetic Algorithm**, written from scratch (no DEAP),
  fully NumPy-vectorized: uniform-box init, tournament selection, blend
  (BLX) crossover, Gaussian mutation, elitism.
* **QGA — Quantum Genetic Algorithm** (Qiskit), with a modular *quantumness*
  score over three independently switchable operators. Each parameter is
  encoded in `n_bits` qubits (grid = 2^n_bits per axis):

  | Operator | Weight | Quantum implementation |
  |---|---|---|
  | `q_init` | 25 | Hadamard layer → superposition → population sampled by measurement |
  | `q_mutation` | 35 | parametrized RY rotation per gene-qubit (amplitude tied to `mutation_scale`) |
  | `q_crossover` | 40 | CX entanglement between homologous parent qubits + controlled-RY interference |

  QGA with all operators OFF (0%) reproduces the CGA **bit-for-bit** — the
  mandatory classical baseline. Circuits are transpiled once at `__init__`
  and evaluated in batched Aer jobs.

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

# Batch: CGA + QGA at 60% quantumness, CC+Pantheon+
python cosmo_genetic_optimizers.py --methods cga qga --dataset CC+Pantheon+ \
  --population-size 200 --generations 120 --qga-preset 60 --n-bits 6

# Custom quantum components via JSON
python cosmo_genetic_optimizers.py --methods qga --qga-config \
  '{"q_init":true,"q_mutation":true,"q_crossover":false}'

# Correctness self-test (CGA reaches optimum; QGA(0%) == CGA)
python cosmo_genetic_optimizers.py --self-test
```

### `qpu_cosmo_samplers.py` — real IBM Quantum hardware


QPU-only, via `qiskit-ibm-runtime` (SamplerV2 + Batch/Session). **No
AerSimulator. Runs no classical method** — real QPU time is scarce, and
classical baselines belong in the simulator pipeline. Hardware-driven
design differences:

| Aspect | Simulator | Real QPU |
|---|---|---|
| Quantum information | exact statevector | measured counts (shots) |
| Proposal displacement | Re(amplitudes), unit-std calibrated | ⟨Z_q⟩ = 1 − 2·P(q=1) |
| QVMC gradient | parameter-shift | **SPSA (2 evals/iter, 1 job)** |
| KL | exact over 2^n states | estimated on the observed support |
| Acceptance | Metropolis via abs(amp0)^2 | Metropolis on CPU (sequential) |
| Error suppression | — | Dynamical Decoupling XY4 + Pauli twirling |

```bash
# Plan without spending QPU time (no IBM account needed):
python qpu_cosmo_samplers.py --model wcdm --method both --dry-run

# Real run with a bounded job budget:
python qpu_cosmo_samplers.py --model lcdm --method qvmc --iters 50 \
    --shots 4096 --least-busy --max-jobs 60 --log-file qpu_run.log
```

`--max-jobs` aborts BEFORE connecting if the run would exceed the budget;
`--dry-run` validates the whole workflow with synthetic counts.

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

## Diagnostics and correctness fixes

`--sanity-check` prints an acceptance regression test, the proposal
statistics, and a per-preset engine map (Qiskit/Aer vs NumPy/SciPy) with a
live routing trace. Audited issues:

* **Apparent "identical results across quantumness".** Not a routing bug:
  each sampler only reads its own components, the `sampling` toggle draws
  from the *same trained state* (shot noise only), identical re-seeding
  makes same-code-path rungs bit-identical, and the Metropolis acceptance
  is a faithful reproduction. The per-method ladders make the axis
  monotonic; the remaining coincidences are the *replication* result.

* **Inverted quantum acceptance (the real convergence bug).** The old
  Hadamard-test readout *decreased* with Δ = lp_prop − lp_cur (accepted
  worse moves), so quantum-acceptance chains drifted to the box edges
  (Ωm ≈ 0.37, H0 ≈ 77). It now encodes Metropolis min(1, e^Δ) as the |0⟩
  amplitude of an RY rotation (verified monotonic, matching Metropolis).
  Post-fix every quantumness level agrees with the classical baseline.

* **Proposal calibration.** The quantum displacement is zero-mean but had
  std ≈ 0.35 (~3× smaller than the classical N(0,1)), pushing acceptance
  to ≈ 0.80 (slow mixing). Each block is normalized to unit std →
  acceptance ≈ 0.5.

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
  reports the true minimum). The creep-up is gone. Note that the absolute
  KL floor depends on grid resolution (see adaptive grid below): with a
  coarse grid both classical and quantum plateau at a higher KL, which is a
  *resolution* limit, not an optimizer failure.

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
  raise the cap explicitly on a bigger machine (see *Memory limits* above).
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

### Memory limits: how high can `nqpp` go?

The quantum methods (QVMC, QGA) discretize the posterior on a statevector
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

To go above the default cap on a bigger machine, raise it explicitly:

```bash
# CPL at nqpp=6 (24 qubits, ~224 GB) on an HPC node
python cosmo_modular_quantum.py --benchmark --model cpl --nqpp 6 \
    --max-qubits 24 --dataset CC+BAO+Pantheon+ --gpu --profile
```

**On real quantum hardware** (`qpu_cosmo_samplers.py`) this RAM limit does
**not** apply — a QPU never materializes the statevector in memory, it holds
the qubits physically. There the constraints are different (qubit count,
circuit depth, and noise), not classical memory. The `--max-qubits` cap is
purely a guard for the **classical statevector simulation** used everywhere
else.

### Qiskit version compatibility (important)

Qiskit and Qiskit-Aer must come from the **same generation**. Mixing them
raises errors such as `ImportError: cannot import name 'convert_to_target'
from 'qiskit.providers'` (Aer built against a different Qiskit). A known-good
CPU combination is:

```bash
pip install "qiskit==1.0.2" "qiskit-aer==0.14.2" "numpy<2"
```

The current Qiskit 2.x line also works as long as Aer matches it
(`qiskit==2.4.x` with `qiskit-aer==0.17.x`). Do **not** install
`qiskit-aer-gpu` on a machine without a CUDA GPU: it pins `qiskit>=1.1.0` and
will fight the rest of the stack. The GPU build is only for the cluster /
local RTX (see below).

### GPU acceleration and resource profiling

Both samplers and the genetic module accept `--gpu` (use an Aer GPU device if
present; otherwise fall back to CPU) and `--profile` (record peak host RAM,
GPU VRAM, CPU/GPU utilization, wall time and GPU-hours, and save a
`resource_usage_*.png` figure plus a `profile_*.json` next to the run).

```bash
# CPU run with profiling
python cosmo_modular_quantum.py --model lcdm --preset 45 --profile

# On a CUDA node: swap in the GPU build of Aer first, then add --gpu
pip uninstall qiskit-aer && pip install qiskit-aer-gpu
python cosmo_modular_quantum.py --benchmark --model cpl --gpu --profile
```

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
* Planck Collaboration (2018) — Gaussian priors (Ωm, H0).
* Spall (1998) — SPSA.
* Gelman & Rubin (1992) — R̂ diagnostic.
* Foreman-Mackey (2016) — corner.py.
