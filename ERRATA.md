# Errata for the thesis code (`v0.8.1-thesis` → `thesis-errata`)

This file lists every defect found in the adversarial review of 27 September
2026 that affects the thesis code, what was done about it, and whether it
changes numbers that were already computed.

* **Frozen code of the running campaign:** tag `v0.8.1-thesis` (= `c2f2004`).
* **Defense numbers:** reproduced only from tag `v0.8.2-thesis-defense`
  (created when the corrected numbers are final).
* Each fix is one commit on branch `thesis-errata`. Commits that change
  results say so in their message (`Changes-results: <ID>, <outputs>`) and
  regenerate `tests/reference/` in the same commit; every other commit
  reproduces the reference set bit for bit.
* Every fix has a regression test in `tests/test_errata.py` that fails on
  `v0.8.1-thesis` and passes after the fix.
* File names below refer to branch `thesis-errata`. On the default branch the
  campaign tools are `python -m thesis legacy {refit,grid-floor,mc-error}`, and
  every fix is also part of the `qablate` library (see CHANGELOG.md).

## 1. Fixes that change results

| ID | Defect | Fix | Outputs that change | Before → after |
|---|---|---|---|---|
| QM-1 | The quantum proposal subtracted the empirical mean of its calibration block. The raw displacement is exactly symmetric, so this added a constant drift to every step; with a plain Metropolis acceptance the chain sampled a shifted distribution. | Calibrate the scale only (RMS) and give each displacement a random sign, which makes q(δ) = q(−δ) exact on any backend. | QMCMC 50 % and 100 % rows. Negligible (< 0.02σ) for ΛCDM, PEDE, wCDM, GEDE on the ideal amplitude route at seed 42; up to ~0.15σ for CPL and for every counts-route / noisy rung. | see §4 |
| QPU-2 | Same drift in the hardware engine, calibrated on a single 64-draw block (up to 0.3σ in a dry run). | Same fix. | QMCMC-QPU runs only. | — |
| HPC-3 | FakeBrisbane: circuits were transpiled against a bare simulator, so two-qubit gates landed on logical pairs (0,1), (1,2)… that carry no ECR error in the device noise model. The "real backend" column had no two-qubit noise. | Place the circuit on a connected path of physical qubits and route it against the device target; Aer `save_*` instructions are re-attached to the final physical positions so outputs stay in logical order; readout matrices are taken from the physical qubits. | Every `fake_brisbane` cell (samplers quantum rungs, QGA, noisy QPU twin). Example: 6-qubit ring ansatz, total-variation distance to the ideal output 0.009 → 0.185. | see §4 |
| CO-3/4 | `fit_statistics` discarded a best fit on the prior boundary (strict prior test) and L-BFGS-B stopped early on badly scaled parameters. | Box-normalized L-BFGS-B + bounded Nelder-Mead polish; boundary optima moved 1e-9 of the box width inside. Never worse than the start. | `chi2`, `chi2_red`, `AIC`, `BIC` (mostly CPL and GEDE) and the model-selection table. | see §4 |

## 2. Fixes that do not change results

| ID | Defect | Fix |
|---|---|---|
| HPC-1 | A model that failed inside `--sweep-all` exited 0, so the runner recorded the task as OK. | Children exit 3; the runner also checks the log for `FAILED —`. |
| HPC-2 | `--noise-readout-p`, `--noise-gate-p1`, `--noise-gate-p2` were parsed by the runner and never forwarded. | Forwarded when not default. |
| QPU-5 | The noisy QPU twin fixed `seed_simulator`, so repeated jobs returned identical counts. | One seed per job, derived from `--seed` and a job counter. |
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

## 4. Before/after tables

To be filled from the reference re-runs (see `errata_tools.py`).

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
