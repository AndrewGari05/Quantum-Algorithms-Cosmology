# Quantum Algorithms for Cosmology — Classical vs Quantum Bayesian Inference

Hybrid quantum/classical pipeline to estimate cosmological parameters
(Ωm, H0, w, w0, wa, Δ) from Cosmic Chronometers (CC) and Pantheon+
supernovae, with QMCMC (Sarracino et al. 2025) and QVMC methods over five
models: **ΛCDM, wCDM, CPL, PEDE and GEDE**.

---

## Architecture (3 + 1 files)

```
cosmo_core.py               ← PHYSICS + DATA + STATISTICS (shared)
        ▲                ▲
        │                │
cosmo_modular_quantum.py   qpu_cosmo_samplers.py
(Aer simulator,            (real IBM Quantum hardware,
 quantumness benchmark)     SamplerV2, no Aer)
```

### 1. `cosmo_core.py` — shared physics module

Strict physics ↔ sampling separation. Contains:

* **`CosmoModel`** (dataclass): name, parameters, bounds, fiducials and
  the Friedmann function `E2(z, θ)`. The `MODELS` registry exposes
  `lcdm`, `wcdm`, `cpl`, `pede`, `gede`. **Injecting a new model (e.g.
  VC) = adding one entry to the dictionary**; no sampler needs changes.
* **`Posterior`**: single point of contact for the samplers. CC
  likelihood (51 points) and Pantheon+ (1048 SNe, analytic M_abs
  marginalization, Goliath et al. 2001). The luminosity distance is
  integrated with a vectorized `cumulative_trapezoid` over a fine z grid,
  **valid for any E²(z; θ)** — this replaces the previous Ωm lookup grid,
  which was only correct for flat ΛCDM.
* **`log_prob_batch`**: vectorized batch evaluation of θ
  (~4 000 CC+Pantheon+ evaluations in ≈1 s) used by the QVMC targets.
* **Statistics**: autocorrelation time τ (FFT, O(N log N)), ESS by chains
  and by weights (Kish), Gelman-Rubin (maximum over parameters),
  `fit_statistics` (χ², reduced χ², AIC, BIC with Nelder-Mead
  refinement).
* **`setup_logger`**: dual logging (detailed file + minimal console).

### 2. `cosmo_modular_quantum.py` — simulator with switchable quantumness

Five quantum/classical switchable components (proposal 20 %, acceptance
25 %, training 20 %, sampling 25 %, normalization 10 %) with presets
0/20/45/70/90/100 %. The 0 % preset IS the fully classical pipeline
(Classical MCMC + Classical VI on the same grid).

**Overlay figures.** Every figure overlays classical (blue `#1f77b4`)
vs quantum (red `#d62728` for the MCMC family, orange `#ff7f0e` for the
variational family) on the same axes, with explicit legends stating the
method and quantumness level. **Every title also embeds the run
metadata** required for the analysis: total MCMC `steps` on QMCMC
figures, and SPSA `iterations` **and `nqpp`** (qubits per parameter) on
every QVMC figure.

| Figure | Content |
|---|---|
| `corner_mcmc_*` / `corner_qvmc_*` | corner.py 2D contours (1σ/2σ) + 1D marginals, classical vs quantum, shared ranges, Planck fiducials |
| `marginals_*` | 1D marginal histograms per parameter, both methods overlaid |
| `kl_overlay_*` | KL vs iteration: Classical VI (blue) vs QVMC (orange) |
| `rhat_overlay_*` | Gelman-Rubin R̂−1 vs steps: Classical MCMC (blue) vs QMCMC (red) |
| `traces_*` | parameter traces, classical chains vs quantum chains |
| `kl_curves_*` / `rhat_curves_*` | benchmark mode: ALL quantumness levels overlaid (thick blue baseline + orange→red colormap) |
| `benchmark_*` | summary panel + extended table (χ², χ²_red, AIC, BIC, ESS, acceptance, KL) |

**Corner-plot groupings (benchmark / Test Run).** In addition to the
per-preset overlays above, a benchmark produces three corner-plot
families so the posteriors can be read at three levels of aggregation:

| Grouping | Files | Content |
|---|---|---|
| Individual | `corner_individual_{mcmc,qvmc}_*_q{pct}` | one standalone corner per method per configuration (each posterior alone) |
| Family "all-in-one" | `corner_family_mcmc_*`, `corner_family_qvmc_*` | Classical MCMC overlaid with **all** QMCMC levels; Classical VI overlaid with **all** QVMC levels (warm colormap over quantumness) |
| 1-to-1 vs baselines | `corner_1to1_*_q{pct}` | one plot per percentage: that level's QMCMC and QVMC overlaid **only** with the two classical baselines (blue = Classical MCMC, teal = Classical VI, red = QMCMC, orange = QVMC), to read off which quantumness level best matches the ideal classical distribution |

**Run modes (interactive menu).** With no arguments the script opens an
interactive menu whose first question is the run mode:

1. **Single configuration** — one preset (or custom Q/C component string)
   plus its mandatory classical baseline.
2. **Full benchmark** — every preset at user-chosen sizes.
3. **Quick Test Run** — every preset (0/20/45/70/90/100 %) at small fixed
   sizes (steps 200, iters 40, nqpp 2): a fast end-to-end stability check
   across the whole hybrid spectrum, producing all overlays and corner
   groupings in a couple of minutes.

**Classical MCMC is a hand-written Metropolis-Hastings (not `emcee`).**
This is deliberate. The project's purpose is to swap *individual*
algorithmic components (proposal, acceptance, …) between classical and
quantum, which requires owning every line of the transition kernel.
`emcee`'s affine-invariant ensemble move is a fixed black box that cannot
host a quantum proposal or a Hadamard-test acceptance, and its internal
bookkeeping would prevent a like-for-like classical-vs-quantum
comparison. Owning the loop also guarantees the classical baseline and
the quantum run share the *exact* same transition structure, step scale
and RNG stream. The kernel is fully **vectorized in NumPy**: each step
proposes for all chains at once and scores them with a single
`log_prob_batch` call (the dominant cost), so the classical baseline runs
6 chains × 2000 steps in ~0.1 s. The quantum-acceptance path keeps a
short per-chain loop only because the Hadamard test is a circuit
evaluated per pair.

**Mandatory classical baseline.** Whenever a configuration with any
quantum component runs, the exact classical counterpart (the 0 % preset)
runs automatically with **identical parameters** (steps, iterations,
chains, burn-in, grid, shots) and the **same RNG seed**, so both consume
the same random streams. In `--benchmark` mode the 0 % baseline runs
first and is shared by every quantum preset.

```bash
# Interactive (default with no arguments)
python cosmo_modular_quantum.py

# CLI: wCDM at 70% quantumness — the 0% classical baseline runs
# automatically with the same parameters and seed
python cosmo_modular_quantum.py --model wcdm --dataset CC+Pantheon+ \
    --preset 70 --steps 4000 --qvmc-iter 300 --log-every 500

# Full quantumness benchmark for CPL (shared classical baseline)
python cosmo_modular_quantum.py --model cpl --benchmark --steps 3000
```

### 3. `qpu_cosmo_samplers.py` — real IBM Quantum hardware

QPU-only, via `qiskit-ibm-runtime` (SamplerV2 + Batch/Session).
**No AerSimulator.** Hardware-driven design differences:

| Aspect | Simulator | Real QPU |
|---|---|---|
| Quantum information | exact statevector | measured counts (shots) |
| Proposal displacement | Re(amplitudes) | ⟨Z_q⟩ = 1 − 2·P(q=1) |
| QVMC gradient | parameter-shift (2·n_φ ≈ 84 evals/iter) | **SPSA (2 evals/iter, 1 job)** |
| KL | exact over 2^n states | estimated on the observed support |
| Metropolis acceptance | simulated Hadamard test | equivalent classical Barker rule¹ |
| Error suppression | — | Dynamical Decoupling XY4 + Pauli twirling |

¹ Acceptance is sequential (step t depends on t−1): on a QPU it would
cost one job + a full queue wait **per step**. The Barker rule
P = e^Δ/(1+e^Δ) is exactly P(ancilla=0) of the Hadamard test, so the
stationary distribution is identical.

**Quantum-only by design.** This script is strictly for dispatching the
quantum algorithms (QMCMC, QVMC) to Qiskit Runtime primitives. It runs
**no classical method at all** — no Classical MCMC, no Classical VI.
Real QPU time is scarce and queues are long, so spending hardware
sessions (or even local CPU time inside a hardware-oriented run) on
classical baselines would be wasteful and conceptually out of place
here. Classical baselines and the full classical-vs-quantum overlay
study live in `cosmo_modular_quantum.py`, which shares the same physics
module. Recommended workflow:

1. Explore the whole quantumness spectrum + classical baselines on the
   simulator (`cosmo_modular_quantum.py --benchmark` / Test Run).
2. Validate a chosen configuration on real hardware here (QMCMC / QVMC).

Figures produced here are **single-method quantum diagnostics**
(`corner_*`, `kl_quantum_*`, `rhat_quantum_*`), each annotated with the
run metadata: steps and chains for QMCMC; SPSA iterations and `nqpp` for
QVMC.

```bash
# Planning without spending QPU time (no IBM account needed):
python qpu_cosmo_samplers.py --model wcdm --method both --dry-run

# Real run with a bounded job budget:
python qpu_cosmo_samplers.py --model lcdm --method qvmc --iters 50 \
    --shots 4096 --least-busy --max-jobs 60 --log-file qpu_run.log
```

`--max-jobs` aborts BEFORE connecting if the run would exceed the
budget, and `--dry-run` validates the whole workflow (shapes, decoding,
figures, logging, output JSON) with synthetic counts.

---

## QPU time estimation

Wall time on hardware is **queue-dominated**, not execution-dominated.
Per-job components:

| Component | Typical (open plan) | Notes |
|---|---|---|
| API overhead | 1–3 s | REST + PUB serialization |
| Queue | 30 s – 30 min (≈60 s used as default) | Backend- and time-dependent; Session removes it between jobs |
| QPU execution | ~100 µs/shot/circuit | e.g. 2 circuits × 4096 shots ≈ 0.8 s |

The `TimingEstimator` class times every real job and replaces these
defaults with measured values; at the end of each run it prints the
projection. With the defaults (queue ≈ 60 s/job, Batch, open plan):

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

Practical conclusions:

1. **QMCMC scales much better on the QPU** thanks to proposal batching
   (each block amortizes the queue across 64 proposals).
2. For QVMC, **Session** (paid plans) removes the queue between SPSA
   iterations and cuts 1000 iterations from ~17 h to ~1 h.
3. Keep `--iters ≤ 50` on the open plan; SPSA convergence with a good
   initial point (e.g. φ pre-trained on the simulator) is usually enough.

---

## Diagnostics and correctness fixes

Two anomalies were audited and resolved; both are guarded by the
`--sanity-check` harness (`python cosmo_modular_quantum.py --sanity-check`),
which prints an acceptance regression test, the proposal statistics, and a
per-preset engine map (Qiskit/Aer vs NumPy/SciPy) plus a live routing
trace.

**"Results look identical across quantumness levels" — mostly expected,
not a routing bug.** The routing (`config.get(...)`) is correct: the
sanity map confirms each component resolves to the intended engine. The
apparent identity has three real causes: (1) each method only reads its
own components — QMCMC uses `proposal`+`acceptance`, QVMC uses
`training`+`sampling`+`normalization` — so presets differing only in the
*other* method's components are identical for a given method; (2) the
`sampling` toggle draws from the *same trained* state |ψ(φ\*)|², so it only
adds shot noise (the QVMC posterior barely moves); (3) the benchmark
re-seeds every preset identically for a fair comparison, making
same-code-path presets bit-identical. After the acceptance fix below, the
quantum acceptance reproduces classical Metropolis exactly, so for QMCMC
**only the proposal** changes the statistics and for QVMC **only the
training** does. That is the correct, interpretable behavior — the
quantumness axis bundles two independent methods.

**Inverted quantum acceptance (the real bug behind non-convergence).** The
old `hadamard_accept_log` built a CRY/Hadamard-test circuit and read
P(ancilla=0); that quantity *decreased* with Δ = lp_prop − lp_cur, i.e. it
accepted worse moves and rejected better ones. Chains using the quantum
acceptance (presets ≥ 70 %) drifted toward the box edges (Ωm ≈ 0.37,
H0 ≈ 77 instead of ≈ 0.26, 70). It now encodes the standard Metropolis
acceptance min(1, e^Δ) as the |0⟩ amplitude of a single-qubit RY rotation
(still read from the Aer statevector), verified monotonic and matching
Metropolis. Post-fix, every quantumness level agrees with the classical
baseline.

**Proposal calibration.** The quantum displacement re[:d]·sign(im[:d]) is
zero-mean (so symmetric-proposal Metropolis stays valid) but had std ≈ 0.35
— ~3× smaller than the classical N(0,1) the step scale was tuned for,
pushing acceptance to ≈ 0.80 (too high → slow mixing). Each block is now
normalized to unit std, bringing acceptance back to ≈ 0.5.

**QVMC convergence — quantum side already converges; the optimizer matters.**
The stalled-KL symptom was the *classical* COBYLA baseline (42 angles,
gradient-free), not the quantum trainer. Quantum parameter-shift drives KL
from ~8 to ~0.34. With a fixed learning rate the KL crept back up near the
minimum; we benchmarked fixed-lr SGD vs Adam vs lr-decay SGD on this exact
landscape and adopted **parameter-shift SGD with a 1/(1+γ·i) learning-rate
decay** (min KL ≈ 0.34 with a flat tail). Adam, although the natural
suggestion, settled into a higher-KL basin here, so it was *not* adopted —
the choice is evidence-based, and `lr_train` is exposed for tuning. No
severe barren plateau was observed at nqpp ≤ 3 (6 qubits); it may reappear
at larger grids, where SPSA would become preferable.

## Installation and data



```bash
pip install -r requirements.txt
```

* **CC**: 51 points embedded in `cosmo_core.py` (or
  `cosmic_chronometers.txt` if present).
* **Pantheon+**: place `pantheon_full_parameters.txt` (format
  `name zcmb zhel dz mb dmb`) next to the scripts.
* **IBM Quantum**: save your account once with
  `QiskitRuntimeService.save_account(channel="ibm_quantum_platform",
  token="...")` or pass `--token`.

## Adding a new model (e.g. Variable Curvature)

```python
# in cosmo_core.py
def _E2_vc(z, th):
    Om, H0, Ok1 = th[0], th[1], th[2]
    ...  # your E²(z)

MODELS['vc'] = CosmoModel(
    name='vc', label='Variable Curvature',
    param_names=['Om', 'H0', 'Ok1'],
    param_latex=[r'$\Omega_m$', r'$H_0$ [km/s/Mpc]', r'$\Omega_{k,1}$'],
    bounds=[(0.05, 0.7), (50, 90), (-0.3, 0.3)],
    sample_box=[(0.1, 0.6), (60, 80), (-0.2, 0.2)],
    fiducial=[0.31, 67.7, 0.0], E2=_E2_vc)
```

Every sampler (simulator and QPU) recognizes it automatically via
`--model vc`. On the simulator it gets the mandatory classical baseline
and the full overlay/corner figure set; on the QPU it is dispatched as a
quantum-only run.

## References

* Sarracino, A., et al. (2025) — QMCMC proposal circuit.
* Goliath, M., et al. (2001) — analytic M_abs marginalization.
* Planck Collaboration (2018) — Gaussian priors (Ωm, H0).
* Spall, J. C. (1998) — SPSA.
* Gelman, A. & Rubin, D. B. (1992) — R̂ diagnostic.
* Foreman-Mackey, D. (2016) — corner.py.
