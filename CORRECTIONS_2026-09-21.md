# Corrections of 20–21 September 2026

Nine markers. **Only one touches scientific numbers** (`[B-EPS15]`); the
other eight are plotting, command line or new analysis, and are safe for a
campaign in progress.

Each marker is in the code with a comment that explains the why, not just the
what. To find them: `grep -rn "B-<NAME>" *.py`.

---

## The ones that change numbers

### `[B-EPS15]` — `cosmo_modular_quantum.py`

`quantum_amplitude_normalization` returned `P_unnorm / (norm + 1e-15)`. That
epsilon introduced a difference of 0 to 2 ULP that broke the **FAITHFUL**
cell: the QVMC 67 % rung no longer matched the 100 % one exactly. The
difference is amplified to ~1e-5 in the parameters.

It is now `P_unnorm / norm`. The guard is not needed because `build_target`
already raises `[B-EMPTY]` if `total <= 0`.

> **It changes the QVMC 100 % numbers in the fifth decimal.** A campaign run
> before this correction is not comparable cell by cell with a run after it.

---

## The ones that fix the plotting

### `[B-KLLOGX]` — `cosmo_modular_quantum.py`
Logarithmic x axis in `ladder_kl_qvmc_*`.

The rungs **do not spend the same number of iterations**: the budget is
equalized by *circuits*, not by iterations, and scipy's classical optimizer
counts function evaluations. At `nqpp=3` the 33 % rung records ~1.2e5 against
1.5e4 for the quantum ones; at `nqpp=9` the difference reaches a factor of
**60**. With a linear axis the longest rung ate the whole scale and the other
three were squashed against the left margin, which is exactly where the
interesting decrease happens.

Not a single point is clipped. **That budget difference must be explained in
the paper, not hidden.**

### `[B-NBITS2]` — `cosmo_hpc_runner.py`
The `cost_<model>` figure did not apply the family filter that
`convergence_*` had applied since `[B-NBITS]`. Result: the genetic curves —
whose x axis is `n_bits` — were drawn on an axis labeled `nqpp` with ticks
placed at the other family's values, so their points fell to the right of the
last tick, unlabeled.

Regression test in `tests/test_cost_figure.py`; verified to fail without the
fix.

### `[B-PDF]` — `noise_plots.py`, `compare_algorithms.py`, `cosmo_hpc_runner.py`
All figures are now saved as PNG **and** PDF. The PDF is vector and is the
one the paper needs. Both are saved from the *same* figure object, so they
cannot disagree.

---

## The ones that fix provenance and robustness

### `[B-PROV-GEN]` — `cosmo_genetic_optimizers.py`, `cosmo_modular_quantum.py`
116 rows of the genetic sweep were written with `noise='none'` even when the
task was noisy: `_ga_side` never received the seed or the noise level. The
scientific data were always valid; what was wrong was the label, and it is
recovered by reading the folder name. `noise_plots.py` warns when the column
and the folder disagree, and uses the folder.

### `[B-RZZ]` — `cosmo_genetic_optimizers.py`
`Operator(RZZGate(Parameter))` raised `TypeError` because of an unbound
parameter and killed an entire task. The primary path stays byte-for-byte
identical; only if that error appears is it retried by transpiling without
the parametric `rzz`/`rxx`/`ryy` gates.

### `[B-REPLOTGLOB]` — `cosmo_modular_quantum.py`
The `--replot-ladder` help promised shell wildcards, but argparse received
the expansion as several positionals and aborted with *unrecognized
arguments*, rejecting all but the first. It is now `nargs='+'` and each
pattern is expanded with duplicates removed.

### `[B-DIAGONLY]` — `cosmo_modular_quantum.py`
`replot_ladder_from_csv(..., diagnostics_only=True)` redoes only R̂ and KL,
skipping the two expensive figures. ~2 s per task instead of ~7.

---

## New analysis

### `[B-SIGMANQPP]` — `compare_algorithms.py`
Section and figure `sigma_vs_grid`: tracks the ratio
σ(QVMC)/σ(Classical MCMC) along the **whole** `nqpp` sweep.

It answers the obvious objection to the `reported_uncertainty` result —*"your
σ comes out narrow because your grid is coarse"*—. With the sweep up to
`nqpp=9`: the ratio **does not rise toward 1, it flattens at 0.66–0.68** from
`nqpp=5` on. The bias is structural to the variational objective, not a
resolution limitation.

> Careful when reading it: `final_KL` is **not** comparable across `nqpp`,
> because the target distribution is defined on the grid and the grid
> changes. The σ ratio is.

---

## How to apply all this to an already-run campaign

None of the eight plotting/CLI corrections requires recomputing physics:

```bash
python cosmo_hpc_runner.py --plot-only CAMPAIGN         # convergence_*, cost_*
python cosmo_modular_quantum.py --replot-ladder \
       CAMPAIGN/samplers_*/model_*/results_config.csv   # ladder_* (legacy campaigns: resultados_config.csv)
python noise_plots.py CAMPAIGN --out OUTDIR
python compare_algorithms.py CAMPAIGN --out OUTDIR
bash rebuild_all.sh CAMPAIGN                            # all of the above
```

**Two traps that cost time:**

1. `replot_ladder_from_csv` needs **the task's log**, not just the CSV: the
   per-step R̂ and the per-iteration KL are not columns. Without the log,
   those two figures are skipped **silently** and the title of the others
   shows `steps=? | iters=?`. That question mark is the signal.
2. `--plot-only` makes **two passes** (samplers and genetic). If the process
   is cut off halfway, the four genetic figures per campaign are missing
   without any visible error.

---

## New files in this batch

| file | what it is |
|---|---|
| `compare_algorithms.py` | ESS, σ, genetic and model selection |
| `compare_seeds.py` | what survives a change of seed |
| `noise_plots.py` | the noise axis |
| `triage_campaign.py` | is the campaign healthy? |
| `rebuild_all.sh` | redoes every figure without recomputing |
| `tests/test_cost_figure.py` | regression for `[B-NBITS2]` and `[B-PDF]` |
| `tests/test_genetic_noise.py` | regression for `[B-PROV-GEN]` and `[B-RZZ]` |
| `tests/test_seeds.py`, `tests/test_triage.py` | the two new scripts |
| `docs/circuitos/` | the seven circuits, 29 pp. — removed from the repository (Spanish-only material) |
| `docs/guia_figuras/` | what each figure answers, 19 pp. — removed from the repository (Spanish-only material) |
| `docs/guia_codigo/` | code map, 5 pp. — removed from the repository (Spanish-only material) |

**180 tests pass.**
