# Errata for the thesis code (`v0.8.1-thesis` → `thesis-errata`)

This file lists every defect found in the adversarial review of 27 September
2026 that affects the thesis code, what was done about it, and whether it
changes numbers that were already computed.

* **Legacy baseline:** tag `v0.8.1-thesis` (= `c2f2004`), the code of the
  running campaign. It is not the source of the defense numbers.
* **Defense numbers:** reproduced only from tag `v0.8.2-thesis-defense`, created
  on this branch when the corrected numbers are final and approved. Until
  that tag exists, no number in this branch is final.
* Each fix is one commit on branch `thesis-errata`. Commits that change
  results say so in their message (`Changes-results: <ID>, <outputs>`) and
  regenerate `tests/reference/` in the same commit; every other commit
  reproduces the reference set bit for bit.
* Every fix has a regression test in `tests/test_errata.py` that fails on
  `v0.8.1-thesis` and passes after the fix.
* File names below refer to branch `thesis-errata`. On the default branch the
  campaign tools are `python -m thesis legacy {refit,grid-floor,mc-error}`, and
  every fix is also part of the `qablate` library (see CHANGELOG.md).
* Since errata(11) the code is in English (numbers bit-identical). Renamed
  scripts: `comparar_algoritmos.py` → `compare_algorithms.py`,
  `comparar_semillas.py` → `compare_seeds.py`, `graficas_ruido.py` →
  `noise_plots.py`, `triage_campana.py` → `triage_campaign.py`,
  `rehacer_todo.sh` → `rebuild_all.sh`; flags `--salida` → `--out`,
  `--peldano` → `--rung`. New runs write `results_config.csv` and
  `results_all_models.csv`; every reader also accepts the legacy names
  (`resultados_config.csv`, `resultados_TODOS_los_modelos.csv`) of campaigns
  already on disk. Legacy names quoted below describe those older files.

## 1. Fixes that change results

| ID | Defect | Fix | Outputs that change | Before → after |
|---|---|---|---|---|
| QM-1 | The quantum proposal subtracted the empirical mean of its calibration block. The raw displacement is exactly symmetric, so this added a constant drift to every step; with a plain Metropolis acceptance the chain sampled a shifted distribution. | Calibrate the scale only (RMS) and give each displacement a random sign, which makes q(δ) = q(−δ) exact on any backend. | QMCMC 50 % and 100 % rows. Negligible (< 0.02σ) for ΛCDM, PEDE, wCDM, GEDE on the ideal amplitude route at seed 42; up to ~0.15σ for CPL and for every counts-route / noisy rung. | see §4 |
| QPU-2 | Same drift in the hardware engine, calibrated on a single 64-draw block (up to 0.3σ in a dry run). | Same fix. | QMCMC-QPU runs only. | — |
| HPC-3 | FakeBrisbane: circuits were transpiled against a bare simulator, so two-qubit gates landed on logical pairs (0,1), (1,2)… that carry no ECR error in the device noise model. The "real backend" column had no two-qubit noise. | Place the circuit on a connected path of physical qubits and route it against the device target; Aer `save_*` instructions are re-attached to the final physical positions so outputs stay in logical order; readout matrices are taken from the physical qubits. | Every `fake_brisbane` cell (samplers quantum rungs, QGA, noisy QPU twin). Example: 6-qubit ring ansatz, total-variation distance to the ideal output 0.009 → 0.185. | see §4 |
| CO-3/4 | `fit_statistics` discarded a best fit on the prior boundary (strict prior test) and L-BFGS-B stopped early on badly scaled parameters. | Box-normalized L-BFGS-B + bounded Nelder-Mead polish; boundary optima moved 1e-9 of the box width inside. Never worse than the start. | `chi2`, `chi2_red`, `AIC`, `BIC` (mostly CPL and GEDE) and the model-selection table. | see §4 |
| HPC-2 | `--noise-readout-p`, `--noise-gate-p1`, `--noise-gate-p2` were parsed by the runner and never forwarded, so every child ran with the defaults. | Forwarded when not default. | Every run launched through the runner with non-default channel strengths (for example a readout-p curve: all its points were the default p = 0.03). | identical runs → the requested p |
| QPU-5 | The noisy QPU twin fixed `seed_simulator`, so repeated jobs returned identical counts (committed together with HPC-3 in errata(4)). | One seed per job, derived from `--seed` and a job counter. | Noisy QPU-twin runs (`qpu_noisy_simulation.py`) only. | repeated jobs no longer replay the same shot noise |
| QPU-12 | The hardware QVMC (`qpu_cosmo_samplers.py`) did not run the thesis circuit: no ring-closing CX, 2 layers instead of 3 (`--layers` default, also used by the QMCMC-QPU proposal circuit), and φ₀ drawn uniformly in [0, 2π) instead of 0.1·N(0, 1). Hardware and simulator numbers were not comparable. | Same ansatz (operator-equivalence test), 3 layers, same initialization. For timing comparisons use the v1 protocol `python -m thesis hardware`, which runs one compiled circuit on the ideal simulator, the noisy twin and the device. | Any QVMC-QPU / QMCMC-QPU run made with the defaults. | not comparable → comparable |

## 2. Fixes that do not change results

| ID | Defect | Fix |
|---|---|---|
| HPC-1 | A model that failed inside `--sweep-all` exited 0, so the runner recorded the task as OK. | Children exit 3; the runner also checks the log for `FAILED —`. |
| QPU-6 | Analysis tools read both the per-model and the per-task cumulative CSV: every task counted twice. | Shared reader `campaign_io.py`. |
| HPC-8 / QPU-10 | Task folders without grid/noise tags were skipped silently. | Tags optional; grid falls back to the `nqpp` column, noise to `none`. |
| QPU-7 | Seed comparison divided by per-seed spreads instead of a paired test (a t = 20.8 offset was reported as 0.54 "sigmas"). | Paired t-test with p-value. |
| QPU-8 | Rows were paired across campaigns, datasets, priors and seeds. | Cell key includes all of them; model selection refuses mixed datasets. |
| QPU-9 | Genetic rows were given a "shift in sigmas" computed from the population spread (223σ artefacts). | Metric not computed for genetic rows. |
| QPU-11 | Triage reported crashed tasks as still running. | Tracebacks and model failures reported as FAILED. |
| QV-4 | `--benchmark` ignored `--chains` and `--shots`. | Forwarded. |
| QV-5 | Appending to a CSV with a different header wrote misaligned columns. | Old file moved aside, fresh header written. |
| QM-4 | QMCMC ladder rows reported the first seed of the process. | Each row carries its own seed. |
| HPC-17 | `NoiseSpec.metadata()` and `repr()` raised `AttributeError`. | Fixed. |
| — | Provenance | New columns `noise_params`, `rhat`, `mc_converged` (R̂ < 1.01 and ESS ≥ 400), `code_version`. |
| — | Partial re-runs | `--rungs` / `--qga-levels` run only the rungs a fix affects; each rung re-seeds, so its rows equal those of the full ladder. |

## 3. Re-statements without re-running

* **GA-1 — ΔQGA vs CGA is a grid-resolution floor, not a quantum penalty.**
  QGA 67 % and 100 % return grid-cell centres; the CGA is continuous. Their
  `chi2_grid` can never go below the best cell-centre χ² of the grid, which is
  0.93 above the continuous minimum at n_bits = 4, 0.21 at 5 and 0.04 at 6
  (ΛCDM, CC+BAO). Report ΔQGA against that floor (`errata_tools.py grid-floor`).
* **QM-3 — Monte Carlo error.** σ differences between QMCMC and MCMC rows are
  only meaningful when both rows have converged (`mc_converged`); at
  campaign-length runs with ESS ≈ 100 they are Monte Carlo noise.
* **QM-2 — noisy acceptance.** With readout/gate noise the amplitude-encoded
  acceptance returns P0 = (1 − 2p)·A + p, which violates detailed balance
  (every proposal is accepted at least ~3 % of the time). Noisy QMCMC-100 %
  samples a different, heavier-tailed stationary distribution, not a noisier
  estimate of the same one.
* **QV-1 — QVMC under-dispersion.** σ(QVMC)/σ(MCMC) ≈ 0.66 comes from the
  reverse KL objective combined with an ansatz that ends close to a product
  state (trained correlation −0.11 vs −0.77 in the target); the mean-field
  floor for ΛCDM on CC+BAO+Pantheon is 0.637. It is not a grid artefact.
* **QPU-4.** χ²/AIC/BIC reported for hardware runs are the classical MAP
  polish, identical whatever the hardware returned; quote them only as such.

## 4. Re-runs and before/after tables

### 4.1 What has to be re-run (on the workstation, after the running campaign ends)

Use the settings of the original campaign (read them from the `PLAN` block at
the top of its log; below, the values of `hpc_20260907_130425`). Every
command writes a new, self-contained campaign folder that includes the
classical rung it is compared with (`C-MCMC`, `CGA`), so the corrected
comparison never pairs rows across campaigns.

```bash
COMMON="--dataset CC+BAO+Pantheon --steps 20000 --qvmc-iter 15000 --chains 8 --shots 4096 --seed 42"

# QM-1: QMCMC rungs, ideal cells nqpp 3..9 (amplitude route and the counts control)
python cosmo_hpc_runner.py --only-samplers $COMMON --nqpp-sweep 3 9 --noise-sweep none \
    --rungs C-MCMC QMCMC50 QMCMC100 --outdir results/errata_qm1_ideal
python cosmo_hpc_runner.py --only-samplers $COMMON --nqpp-sweep 3 9 --noise-sweep none \
    --proposal-route counts --rungs C-MCMC QMCMC50 QMCMC100 --outdir results/errata_qm1_counts

# QM-1: QMCMC rungs, noisy cells nqpp 3..5 (readout and full; FakeBrisbane below)
python cosmo_hpc_runner.py --only-samplers $COMMON --nqpp-sweep 3 5 --noise-sweep readout,full \
    --rungs C-MCMC QMCMC50 QMCMC100 --outdir results/errata_qm1_noisy

# HPC-3 (+QM-1): every FakeBrisbane cell, all sampler rungs and the genetic ladder
python cosmo_hpc_runner.py --only-samplers $COMMON --nqpp-sweep 3 5 --noise FakeBrisbane \
    --outdir results/errata_fakebrisbane
python cosmo_hpc_runner.py --only-genetic --dataset CC+BAO+Pantheon --seed 42 \
    --nbits-sweep 4 6 --noise FakeBrisbane --outdir results/errata_fakebrisbane_genetic
```

The qubit ceilings of the runner are derived from the machine's RAM. Before
launching, add the original campaign's `--max-qubits` / `--max-qubits-genetic`
to `COMMON` (from its `PLAN` block), run each command with `--dry-run`, and
check that the planned cells match the original ones; on a small machine
the FakeBrisbane commands skip or trim the largest cells.

The QMCMC rows do not depend on `nqpp` (the proposal uses max(2, d) qubits),
so the `nqpp` sweeps above only restore one row per reported cell; running a
single `nqpp` is enough if only the fidelity table is needed.

No re-run is needed for CO-3/4 (`errata_tools.py refit <campaign>`), GA-1
(`errata_tools.py grid-floor ...`) or QM-3 (`errata_tools.py mc-error <campaign>`);
they recompute from the stored CSVs.

### 4.2 QM-1: QMCMC 50 % minus Classical MCMC, before and after

CC+BAO+Pantheon, seed 42, 20 000 steps x 8 chains, ideal simulator. Shift of
the QMCMC 50 % mean from the Classical MCMC mean, in units of the Classical
MCMC sigma. `MC SE` is the Monte Carlo standard error of that shift,
sqrt(1/ESS_C + 1/ESS_Q); shifts well below 3 MC SE are consistent with zero.
Script: `tests/reference/before_after_qm1.py`.

| model | route | parameter | before (σ) | after (σ) | MC SE |
|---|---|---|---|---|---|
| lcdm | amplitude | Om | -0.005 | +0.015 | 0.017 |
| lcdm | amplitude | H0 | +0.010 | -0.012 | 0.017 |
| lcdm | counts | Om | +0.074 | +0.023 | 0.017 |
| lcdm | counts | H0 | -0.049 | -0.028 | 0.017 |
| pede | amplitude | Om | +0.003 | +0.025 | 0.017 |
| pede | amplitude | H0 | +0.007 | -0.019 | 0.017 |
| pede | counts | Om | +0.088 | +0.035 | 0.017 |
| pede | counts | H0 | -0.060 | -0.035 | 0.017 |
| wcdm | amplitude | Om | -0.013 | -0.021 | 0.029 |
| wcdm | amplitude | H0 | +0.033 | +0.022 | 0.029 |
| wcdm | amplitude | w | -0.007 | +0.006 | 0.029 |
| wcdm | counts | Om | -0.016 | -0.028 | 0.027 |
| wcdm | counts | H0 | +0.077 | +0.034 | 0.027 |
| wcdm | counts | w | -0.030 | +0.007 | 0.027 |
| gede | amplitude | Om | -0.020 | -0.020 | 0.027 |
| gede | amplitude | H0 | +0.022 | +0.023 | 0.027 |
| gede | amplitude | Delta | -0.005 | -0.005 | 0.027 |
| gede | counts | Om | -0.029 | -0.011 | 0.025 |
| gede | counts | H0 | +0.103 | +0.026 | 0.025 |
| gede | counts | Delta | +0.042 | +0.007 | 0.025 |
| cpl | amplitude | Om | -0.206 | -0.079 | 0.056 |
| cpl | amplitude | H0 | +0.030 | +0.033 | 0.056 |
| cpl | amplitude | w0 | -0.014 | +0.035 | 0.056 |
| cpl | amplitude | wa | +0.215 | +0.052 | 0.056 |
| cpl | counts | Om | -0.159 | +0.029 | 0.056 |
| cpl | counts | H0 | +0.083 | +0.011 | 0.056 |
| cpl | counts | w0 | +0.040 | -0.011 | 0.056 |
| cpl | counts | wa | +0.106 | -0.023 | 0.056 |

Reading: before the fix the counts-route shifts reach 4-6 MC SE for ΛCDM,
PEDE and GEDE (H0 +0.10σ for GEDE) and the CPL amplitude-route shifts reach
about 4 MC SE (Ωm −0.21σ, wa +0.22σ), in line with the drift predicted from
the calibration offsets. After the fix every shift is within about 2 MC SE of
zero. The ideal amplitude-route cells of ΛCDM, PEDE, wCDM and GEDE were
already within Monte Carlo error before the fix. (MC SE treats the two chains
as independent; they share the seed, so it is approximate.)

## 5. Terminology

The thesis text, slides and code comments should use these names.

| Old name | Accurate name | Why |
|---|---|---|
| "Hadamard-test acceptance" | amplitude-encoded Metropolis acceptance | A = min(1, e^Δ) is computed classically, loaded into one RY angle and read back; no Hadamard test is performed. In the ideal limit it equals classical Metropolis by construction. |
| "quantum proposal" | random-circuit proposal | Displacements come from measuring a randomly parameterized circuit; they do not depend on θ. The statevector ("amplitude") route is not realizable on hardware; the counts route is. |
| QGA "quantum operators", "genuine two-qubit interference, not a classical coin flip" | circuit-sampled QGA operators | Every QGA circuit starts in a basis state and is measured in that basis; the output bits have exactly the distribution of classical coin flips. |
| "quantum amplitude normalization" | normalization rung (classical value) | The circuit is run but its output is not used; the exact classical normalization is returned. |
| "FAITHFUL cell confirms fidelity" | FAITHFUL cell = plumbing check | Bit-identity of QMCMC 50 %/100 % and QVMC 67 %/100 % holds by construction; it verifies the implementation, not a physical property. |
| "Jeffreys scale" (ΔBIC) | Kass & Raftery (1995) scale | The 2/6/10 thresholds are Kass & Raftery's. |

## 6. Known data caveats (not bugs; documented choices)

* The 51-point H(z) table (Magaña et al. 2018 compilation: 31 CC + 20 BAO)
  treats overlapping BAO measurements as independent. With only
  non-overlapping BAO, σ(H0) grows by ~50 %.
* Pantheon 2018 is used with statistical errors only (no `sys_full_long`
  covariance). With systematics, σ(Ωm) goes from 0.0104 to 0.0143 (Fisher).
* z_hel = z_cmb in the Pantheon file (≤ 6 mmag effect); the Moresco CC
  systematic covariance is not included.
* GEDE uses the ΛCDM transition redshift, not the Δ-dependent z_t of
  Li & Shafieloo (2020) (≤ 1.8 % in H(z) inside the prior).

These affect the cosmology quoted, not the fidelity comparison: classical and
quantum methods share the same likelihood.
