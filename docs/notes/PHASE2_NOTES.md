# Phase 2 — change log and verification status

This file records every code change made in Phase 2, what was verified, and
what still requires validation on real hardware. It exists so a reviewer (or
future-you) can see exactly how much trust each fix carries.

## Environment reconciliation

The pinned `requirements.txt` was corrected to the ACTUAL working environment
(`pip freeze` of the `cosmologia` conda env), which differs from what the
original README claimed:

| Package | README claimed | Actual (verified) |
|---|---|---|
| qiskit | 1.0.2 | **2.4.2** |
| qiskit-aer | 0.14.2 | **0.17.2** |
| numpy | <2 | **1.26.4** |
| GPU backend | qiskit-aer-gpu | **cuQuantum / cuStateVec** (cuquantum-cu12 26.3.2, custatevec-cu12 1.13.1) |
| qiskit-ibm-runtime | (referenced) | **not installed** |
| pynvml | (referenced) | **not installed** (nvidia-smi fallback used) |

Consequences:
- The QPU path (`qpu_cosmo_samplers.py`) has only ever run in `--dry-run` on
  this machine, since `qiskit-ibm-runtime` is absent. The QPU fixes below are
  therefore **not hardware-tested**; they are written defensively and must be
  validated on the first real-hardware run.
- The GPU path uses cuQuantum, not a `qiskit-aer-gpu` wheel. Whether `--gpu`
  actually engages the RTX 4070 must be confirmed on the dev machine; the
  device probe is backend-agnostic but untested here (no CUDA in the dev
  sandbox).

## Fixes — verification status

### Verified in a pure-NumPy/SciPy sandbox (tests pass)

| ID | Fix | Test |
|---|---|---|
| B5 | Sokal-windowed integrated autocorrelation time τ | test_autocorr_* |
| S2/S4 | rank-normalized split-R̂ + 1.01 threshold | test_split_rhat_* |
| S3 | ESS formula documented + routed through `autocorr_time_max` | test_ess_* |
| B6 | uniform-weight ablation index + FAITHFUL/ALGORITHMIC kinds | test_ablation.py |
| P1 | PEDE/GEDE f_DE(0)=1 (and Δ=0 → ΛCDM at z≠0) | test_emergent_de_*, test_gede_* |
| B4 | combined-dataset M_abs interaction (shift-invariance) | test_mabs_marginalization_* |
| B3/S1 | adaptive grid window moved to cosmo_core, median-centred | verified narrower-than-box at runtime |
| B2 | QPU execution-time extraction (was always falling back) | test_qpu_helpers::test_b2_* |
| B1 | Metropolis rule (text/docstrings unified with code) | test_qpu_helpers::test_b1_* |
| A2 | batched quantum acceptance == scalar Metropolis | verified numerically (FAITHFUL cell) |

### Written, logic-checked, but NOT executed on Qiskit/hardware

| ID | Fix | What was checked | What still needs a real run |
|---|---|---|---|
| A1 | QGA: transpile templates ONCE; gene bits via bound angles, no per-individual transpile | bit↔qubit-order round-trip; bit-as-angle encoding math | run a QGA generation on Aer and confirm identical MAPs + the expected speedup |
| B3 | QPU GridEncoding now uses the adaptive window | window-construction logic | a real QVMC-QPU run to confirm the KL is now comparable to the simulator |
| B2 | execution-span parsing | all metadata shapes via mocks | confirm against a genuine SamplerV2 result on qiskit-ibm-runtime ≥0.30 |
| A4 | GPU blocking made circuit-size aware; cuStateVec opt-in | option-dict construction | confirm `--gpu` engages the RTX and tiny circuits no longer pay blocking overhead |

### Documentation-only (no behavioural change)

- P2: radiation/neutrino approximation declared at `OMEGA_R0`.
- P3: flat-universe assumption documented in `_mu_theory`, with the curved
  generalization spelled out for any future non-flat model.
- A5: `filterwarnings` scoped to benign categories so numerical warnings show.

## Repository scaffolding added

- `requirements.txt` (verified pins) + `requirements-gpu.txt` (cuQuantum).
- `LICENSE` (MIT — confirm author preference).
- `CITATION.cff` (add ORCID when available).
- `tests/` (pytest): test_core.py, test_ablation.py, test_qpu_helpers.py,
  conftest.py (auto-skips Qiskit tests when absent).
- `pytest.ini`, `.github/workflows/tests.yml` (CI on the no-Qiskit floor).
- `data_manifest.py` (SHA256 provenance/verification for the data files).

## Index numbers that CHANGED (expected, from B6 uniform weighting)

The per-method ladder labels changed from the old weighted percentages to
honest equal-spaced fractions. Anything (README, .tex, slides, prior CSVs)
that cites the old numbers must be updated:

| Ladder | Old (weighted) | New (uniform) |
|---|---|---|
| QMCMC | 0 / 44 / 100 | **0 / 50 / 100** |
| QVMC | 0 / 46 / 82 / 100 | **0 / 33.3 / 66.7 / 100** |
| QGA | 0 / 25 / 60 / 75 / 100 | **0 / 33.3 / 66.7 / 100** |

`legacy_weighted_index` / `legacy_qga_weighted_index` still reproduce the old
numbers for CSV continuity.

## Still TODO in Phase 2b

- README rewrite (the ablation vocabulary + corrected versions/ladders).
- The full LaTeX bridge document per the approved index.
