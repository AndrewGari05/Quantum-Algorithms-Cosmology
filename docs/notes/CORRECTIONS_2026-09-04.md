# Adversarial review and corrections — 2026-09-04

Second pass, this time **systematic** (TASK 2, which had never been done: all
previous bugs had turned up by chance, while doing something else).
`cosmo_core.py` was reviewed in depth by running code, not by reading it.

**No correction here changes a single already-published number.** This was
verified explicitly: for physical points, scalar and batch paths still match
bit for bit (`max|diff| = 0.0`), and the tests pass.

---

## What was fixed

### `[B-BOUNDS]` — HIGH — the "best fit" could lie outside the prior

`fit_statistics` refined with **unconstrained** Nelder-Mead on `chi2`, which
never sees the prior. Since `chi2` was finite outside the box, the simplex
wandered out and a point with **zero posterior probability** was reported as
the best fit.

Measured before the fix, over the 15 model × dataset combinations:

| combination | what it returned |
|---|---|
| `cpl` / CC+BAO+Pantheon | Ωm = 0.178 (the prior starts at 0.18) |
| `cpl` / Pantheon | **H₀ = 6.3e-5 km/s/Mpc** |
| `wcdm` / Pantheon | H₀ = 43.3 (prior: 60–82) |
| `gede` / Pantheon | H₀ = 39.4 |
| `cpl` / CC+BAO | Ωm = 0.159 |

The dangerous part: the χ² penalty was minimal (Δχ² = 0.004 in
`cpl`/CC+BAO+Pantheon), so **AIC and BIC barely changed and the number looked
perfectly healthy**. Exactly the class of silent failure that had already
bitten three times.

It now uses L-BFGS-B with the model's bounds, with bounded Nelder-Mead as a
fallback, and a final check that rejects the candidate if it still fell
outside the support. Verified: **all 15 combinations stay inside.**

It did not affect any published result because ΛCDM never went outside.

### `[B-GRIDSEED]` — HIGH — the grid changed on every call

`estimate_grid_window` took its random numbers from the module's **global**
RNG. Consequences measured on `lcdm`/CC+BAO: the Ωm width varied by **54 %**
between identical calls (0.1121 to 0.1721), and it also depended on how many
numbers any other part of the program had consumed before.

That broke exactly the promise the docstring itself makes: that the simulator
(`cosmo_modular_quantum`) and the QPU (`qpu_cosmo_samplers`) build the
**same** grid. Since each one calls the function on its own, each ended up
with a different discretization — and the KL of both was compared on
different grids. **That comparison is the central result of the project, and
it is exactly the one about to be made against real hardware.**

The window is now a pure function of its arguments plus a seed
(`GRID_WINDOW_SEED`). Verified: five identical calls, and it stays identical
after advancing the global RNG. `seed=None` restores the old behavior if
someone wants to sample the variability on purpose.

### `[B-CLIP]` — HIGH — a non-physical model returned a finite χ²

`CosmoModel.H` did `np.clip(e2, 1e-12, None)`: for E² ≤ 0 — a combination of
parameters **with no physical meaning** — it returned H = 10⁻⁶·H₀ instead of
warning, and from there a finite χ² and log-posterior followed. The class's
documented contract said the opposite.

The inconsistency was observable: for the same θ, the supernova branch **did**
reject and the chronometer branch did not, so the same point had a finite
posterior with CC+BAO and −∞ with CC+BAO+Pantheon.

With the five current models E² > 0 over their whole box, so **it has not
corrupted anything**. But it stops being latent as soon as curvature is added
(Ω_k < 0 makes E² negative at high z) — which is the project's next model.

### `[B-NANMASK]` — LOW — the batch path returned `nan` where the scalar gave `-inf`

`nan <= 0` is `False`, so an E² = nan slipped through the mask of the
vectorized path. Metropolis rejects a nan by comparison, so it never
corrupted a chain, but the two paths must agree. They now agree.

### `[B-PRIORTYPE]` — LOW — a misspelled prior fell back to flat, silently

`'Gaussian'` capitalized, `'planck'`, any typo: all of them gave
`log_prior = 0` silently. A run labeled `gaussian` in the CSV that actually
used a flat prior is unrecoverable afterwards. The CLIs already restricted it
with `choices`; library calls did not. It now raises `ValueError`.

### `[B-SILENT]` — MEDIUM — `load_cc` substituted the file without saying so

A nonexistent `path` or a file with fewer than three columns returned the
embedded table **without any message**. The only signal was the *absence* of
the line "✓ CC+BAO loaded" in a log thousands of lines long. That defeats the
purpose of `data_manifest.py` and `data_checksums.json`, and would silently
replace any alternative compilation a reviewer asked to test. It now warns on
all three paths.

### Comment `[P2]` — the note about radiation said the opposite of the truth

It claimed that `OMEGA_R0 = 9.4e-5` was "photons only" and that at high z it
had to be multiplied by ~1.68 to add the neutrinos. It is the other way
around: the value **already includes them** (photons alone would be 5.40e-5;
photons + 3.046 neutrinos, 9.14e-5). Following the comment's instruction
would have introduced a 73 % error. The number was always right; the comment
was not.

### `csv_path` — an undefined name in dead code

`append_results_csv` had a `return csv_path` after a `return run_csv`.
Unreachable today, a `NameError` as soon as someone reordered the function.
Removed.

---

## What was added

### `[B-NOSTATE]` — the genetic state now survives the process

The genetic figures are drawn from the final population and the
per-generation history, and **none of that was saved to disk**. While fixing
`[B-GAPLOT]` the consequence surfaced: a figure had the wrong error bar, and
redrawing it correctly required **repeating the entire campaign**. In a run
where a noisy cell at 12 qubits takes two days, that turns "change the color
of this curve" into a week of compute.

Every genetic task now writes a `ga_state.npz` (~1–3 MB compressed) and
everything is redrawn with:

```bash
python cosmo_genetic_optimizers.py --replot .../model_lcdm/ga_state.npz
```

Design details:

- **No `pickle`.** An `.npz` with the arrays and a JSON block with the
  scalars. `pickle` would save the object in one line but breaks when the
  numpy version changes or a class is renamed, and these files must still
  open two years from now, when a figure has to be redone for the defense.
- **It is written before plotting**, and even if figures are disabled: if the
  process dies while rendering, the data are already safe.
- The final population, its fitness and its weights are stored as float64
  because the CSV numbers come from them; the per-generation history is
  stored as float32 because it only feeds the animation.
- In `--sweep-all` **one** state with all rungs together is saved, not one
  per rung (they would overwrite each other).
- `--no-state` disables it.

Five new tests, including the one that verifies that the weighted mean and
standard deviation — the ones that go into the CSV — come back **exact**
after the round trip through disk.

### `HOW_TO_RUN_QPU.md`

Full procedure for real hardware, with what is verified offline (the
`ibm_quantum_platform` channel is the current one, `SamplerV2` is the current
primitive, transpilation to ISA exists and produces circuits inside the basis
of a real backend) and, above all, **what is not tested** because the
simulated twin replaces it: the credentials, `least_busy`, the lifecycle of
the `Batch` and the parsing of the real result. The 1-job smoke test exists
precisely for that.

---

## Status

- **107 green tests** (17 new).
- The campaign plan still yields **exactly 100 tasks**, identical to the one
  verified before.
- End-to-end smoke test with the runner: **5/5 tasks OK**, all figures
  generated, including the noise-comparison ones.
- Redrawing from a `ga_state.npz` produced by the runner: **works**.

## What remains pending and was not touched

- **GEDE's `z_t`.** The ΛCDM matter–Λ equality is used
  (`((1-Om)/Om)^(1/3)`), which for GEDE **is not self-consistent**: the proper
  condition would be `Ωm(1+z_t)³ = Ω_DE·f_DE(z_t)`, and `f_DE(z_t) ≠ 1`
  unless Δ=0. The difference reaches **2.6 % in H(z)** at the edge of the Δ
  prior, comparable to the chronometer error bars. **I did not change it on
  purpose**: I cannot check from here whether the closed form is the
  convention of the Li & Shafieloo paper, and changing it would move results
  already obtained. Check it against the paper's equation before the defense.
  Moreover, the test the comment cites as a guarantee uses Δ=0, where
  `f_DE ≡ 1` for *any* `z_t`: it guarantees nothing. A model with the sign
  flipped passes both current tests.
- **The BOSS DR12 covariance.** The three highest-weight points of the CC+BAO
  table (37.6 % of the total weight) are a correlated analysis treated as
  diagonal. That **underestimates the error bars** the thesis quotes. Declare
  it or incorporate the 3×3 block.
- **`k` in AIC/BIC with supernova-only datasets.** χ² is *exactly*
  independent of H₀ (the analytic marginalization of M absorbs it), but H₀ is
  counted as a free parameter. It does not affect the campaigns (they always
  combine datasets); it would affect a SNe-only fit.
- **The other four modules** (`cosmo_modular_quantum`,
  `cosmo_genetic_optimizers`, `cosmo_hpc_runner`, `qpu_cosmo_samplers`) did
  **not** receive the same systematic pass — the review budget ran out.
  `cosmo_core.py`, which is where the physics lives, did.

---

# Second batch: `cosmo_modular_quantum.py`

Adversarial review of the module that contains the central scientific
result. **This is where the most serious finding of the whole project
appeared.**

## `[B-BUDGET]` — CRITICAL — the classical vs quantum comparison did not cost the same

`--qvmc-iter` was passed **unchanged** to both branches of QVMC training. But
an iteration does not cost the same in each:

- the quantum branch spends **1 + 2·n_φ** circuit evaluations per iteration
  (the KL at φ plus the 2·n_φ shifted ones of parameter-shift),
- COBYLA spends exactly **1**.

With the same `--qvmc-iter`, the quantum branch received between **57×**
(ΛCDM, nqpp=2) and **225×** (CPL, nqpp=4) more work. Measured:

| branch | iterations | circuits | KL |
|---|---|---|---|
| parameter-shift | 40 | 2281 | 1.425 |
| COBYLA (nominal budget) | 40 | **41** | 1.783 |
| COBYLA (equalized budget) | 2281 | 2282 | **0.00063** |

At equalized budget, **the classical optimizer wins by three orders of
magnitude**. Also verified at nqpp=3, the campaign resolution: classical
0.204 versus quantum 2.047 with the same ~5100 circuits.

**Why this matters more than any other bug in this project.** The brief
said: *"the central scientific claim is quantum fidelity, NOT quantum
advantage... I do not want any modification to suggest advantage where there
is none."* This bug did exactly that, and not through a badly written comment
but through the arithmetic of the experiment. Any cell where the quantum rung
came out better was an artifact of miscounting the work.

And there is a reading that **strengthens** the conclusion: in the large
campaign (CC+BAO+Pantheon) the 67 % rung came out *worse* in 20 of 24 cells —
i.e. it lost **even while receiving between 57 and 225 times more compute**.
With an equalized budget that conclusion not only holds, it becomes much
stronger.

Fixed with `--budget-mode`:

- `circuits` (**new default**) — both branches receive the same number of
  circuit evaluations. It is the honest comparison.
- `iters` — reproduces the old behavior. **It is the one used by every
  campaign before 2026-09-04**; use it only to reproduce them, never for a new
  comparison.

In addition, every CSV row now carries `budget_mode` and `circuits_train`: the
work actually spent, not the nominal one, so that the comparison can be
audited by a third party.

## `[B-ESSCOMP]` — HIGH — the QVMC ESS was a representation artifact

The quantum branch returned the **compressed** sample (one row per measured
bitstring, with its count as weight) and the classical one **one row per
shot**. They describe the same sample, but Kish's ESS is not invariant under
that compression: on the compressed form it measures how many grid cells were
occupied, not how many samples there are.

Measured with 3 chains × 2000 shots, same `phi_opt` and KL identical to the
sixth digit:

```
Classical VI   rows=6000   reported ESS = 6000.00
QVMC 33%       rows= 185   reported ESS =  100.74
     ...expanding the counts, BOTH give 6000.00
```

This is in the published results: **every** `Classical VI` row reads
`ESS=6000.0` and every `QVMC 33 %` row reads 95–113, with the same KL. A
reader concludes that quantum sampling costs 60× the effective sample size —
**in a cell labeled FAITHFUL, where by construction there must be no
difference**. The means and standard deviations were faithful; only the ESS
column lied. Fixed: both branches now give 6000.0.

## `[B-PROV]` — HIGH — the CSV did not say under which conditions each row was obtained

No schema recorded the noise level, the readout route or the seed. Two rows
with the same key (`Method`, `model`, `dataset`, `prior`, `nqpp`) but
different numbers were indistinguishable: one could come from an ideal run and
the other from a noisy one. **That made the entire noise axis impossible to
reconstruct from the results**, and incidentally broke the reading of the
FAITHFUL cell: `QMCMC 50 % ≡ QMCMC 100 %` holds bit for bit in some groups of
rows and not in others, because the ones that differ are noisy — but there was
no way to know.

`cosmo_noise.NoiseSpec.metadata()` already returned exactly this and was not
called from anywhere. Five columns added: `noise`, `proposal_route`, `seed`,
`budget_mode`, `circuits_train`.

## `[B-EMPTY]` — MEDIUM — a grid with no support returned an excellent KL

If the grid touches no point with prior support, `P` comes out all zeros. The
classical branch gave `nan` with a warning; the quantum one, with its
`+1e-15` guard, returned zeros **without any warning**. Downstream the target
becomes uniform and the KL degenerates into `log n − H(Q)`, which is a
**small and attractive** number: a complete run reported Ωm = −4.45,
H₀ = −95.1 and **KL = 0.022** — better than any legitimate run of the
campaign — without a single error. And it was the 100 % rung that hid the
signal best. It now raises.

## `[B-COBYLA-CLAMP]` — LOW

SciPy raises COBYLA's `maxiter` to `n_vars + 2` and warns; it is now made
explicit so that the declared budget is the real one.

---

## What was checked and is CORRECT

This is also a result, and a good one:

- **The parameter-shift gradient is EXACT.** Verified against central
  differences on the transpiled circuit: the residual scales as O(h²), i.e. it
  is the truncation error of the finite difference, not a gradient error
  (`h=1e-3 → 1.9e-7`, `h=1e-4 → 1.9e-9`, `h=1e-5 → 7.0e-11`). The
  cancellation of the `+1` is real: `max|Σ_i ∂Q_i/∂φ_j| = 2.4e-16`. **The
  `[H5 FIX]` is solid** and supports the main result.
- **The FAITHFUL cells hold for the right reason.** Instrumenting the code:
  the quantum branch of `acceptance` really executes (350 calls at 100 %, 0 at
  50 %) and the chains come out `np.array_equal → True`. It is not switched
  off; it is equivalent. The same with `sampling` (Aer's job count goes from 32
  to 34) and with `normalization` (the QAE circuit executes and its result is
  discarded by design, as declared in the docstring).
- **The `none-counts` control isolates exactly what it should**: `dAcc = 0`,
  `dKL = 0`, and only the proposal changes.
- **The noise really gets there**: `readout` and `full` differ from the ideal
  and go through `density_matrix`; the `[N1]` guard raises; `ΣQ = 1.0000000`
  on every rung.
- **Reproducibility**: two runs with the same seed differ only in wall-clock
  time.

## An observation worth declaring in the thesis

Under readout noise, the acceptance amplitude has a floor at `p`, so an
impossible proposal (Δ = −30) is accepted 3 % of the time and **the acceptance
rate RISES** (0.487 → 0.545). It is a correct consequence of the noise model,
but it means that the noisy QMCMC **no longer samples the posterior**. Say so
explicitly, so that a rising acceptance rate is not read as better mixing — it
is the same trap as the ESS that rises with noise while the KL gets worse.

---

## What remains unreviewed

`cosmo_genetic_optimizers.py`, `cosmo_hpc_runner.py` and
`qpu_cosmo_samplers.py` have **not** received the systematic pass. Given what
turned up in the two modules that did receive it, do not assume they are
clean.


---

# Third batch: genetic, runner and docstrings

## `[B-REFINE]` — HIGH — the genetic χ² measures scipy, not the genetic algorithm

This explains something that had been visible for a long time without being
understood: why in the campaigns the genetic χ² comes out **byte-for-byte
identical** for CGA and for the four QGA rungs (1064.6099 in every row). It
had been read as "it is the χ² of the same best fit". The real reason is a
different one.

The genetic algorithm returns a **grid** point. It is then refined with
`fit_statistics(refine=True)`, which is a **continuous** optimizer. And that
refinement erases the difference between rungs. Measured with lcdm/CC+BAO,
n_bits=4:

| rung | best grid point | grid χ² | χ² after refining |
|---|---|---|---|
| CGA | [0.26496, 70.3148] | 27.6502 | 27.469109 |
| QGA 0 % | [0.26496, 70.3148] | 27.6502 | 27.469109 |
| QGA 33 % | [0.25769, 70.6918] | 27.4717 | 27.469109 |
| QGA 67 % | [0.27000, 70.3125] | 28.3992 | 27.469109 |
| QGA 100 % | [0.27000, 70.3125] | 28.3992 | 27.469109 |

The four land on **different** cells, with a range of **0.93 in χ²** between
the best and the worst. After refining, all four give the same number to the
sixth digit.

Refining is not an error: the refined χ² is the one to compare against the
samplers, and that is why it is kept. **The error would be reading those
columns as if they measured the genetic algorithm.** Do not read them that
way: the genetic χ², χ²_red, AIC and BIC cannot distinguish one rung from
another, by construction.

Fixed by adding **`chi2_grid`** to the CSV and to `ga_state.npz`: the χ² of
the best grid point, unrefined. It is the only genetic number that does
distinguish the rungs. The log now prints both lines, and says how much the
refinement moved.

And with this the FAITHFUL cell is better tested than before: `CGA ≡ QGA(0 %)`
is now verified **on the grid**, before refinement. If they only matched
afterwards, the equality would say nothing about the operators — the refiner
would guarantee it.

## `[B-BUDGETMISMATCH]` — MEDIUM — two flags that silently contradict each other

`--max-task-gb` larger than `--mem-budget-gb` is a contradiction. The qubit
ceiling comes from the former, but the pool **always** admits at least one
task even if it does not fit in the aggregate budget (otherwise a large task
would block the campaign forever). With that combination you can launch a
task that exceeds the entire budget, and the only signal would be an OOMKill
hours later.

Measured: `--max-task-gb 40 --mem-budget-gb 10` with CC+BAO+Pantheon granted
**18 qubits, ~22 GB for ONE task**, against a 10 GB budget. Without a single
warning. It now warns at planning time.

## What was checked in the runner and is CORRECT

- **cgroup v1 and v2 detection**: correct, including the "no limit"
  sentinels (`max` in v2, `9223372036854771712` in v1).
- **The shared-CSV lock withstands real concurrency**: 8 threads × 40 writes
  → 320 rows, **0 duplicated headers, 0 corrupted rows**.
- **Task admission respects the budget** (`admitted_mem() + est_mem
  <= mem_budget`), with the documented exception of the first task.
- **A child that dies does not kill the campaign**: verified in the actual
  campaign runs (`rc=-9` recorded and the rest carried on).

## What was checked in the genetic module and is CORRECT

- **`CGA ≡ QGA(0 %)` bit for bit**, now also on the grid: same `theta_grid`,
  same `chi2_grid`, same final population.
- **`QGA(100 %)` does differ** — otherwise the ladder would measure nothing.
- **Reproducibility**: same seed twice → identical; different seeds →
  different populations.

## Docstrings

The request was to convert the `#` comments to `'''...'''`. This was done
**only where appropriate**, and here is why not everywhere: a `'''...'''`
inside a function body is not a docstring, it is a stray string that Python
evaluates and discards — it does not show up in `help()` or in the editor
tooltip, and it bloats the module. Only the **first** string of a module,
class or function is one.

What was done:

- **The `#` headers of the five large modules** converted to module
  docstrings. They were comments, so `help(cosmo_core)` showed nothing; it now
  shows the 21–68 lines of each header.
- **58 functions, classes and methods that had no docstring** now have one,
  written by reading each of them (with `Args:`, `Returns:` and `Raises:`
  where applicable), not from a template.
- **Zero definitions without a docstring** in the whole repository (360 in
  total).
- Implementation comments stay as `#`, which is where they are useful.

## Status

- **111 green tests**.
- Identical campaign plan: **100 tasks**.
- Full smoke test of the runner: **5/5 tasks OK**, with `chi2_grid` populated
  and distinguishing the rungs (27.7394 / 27.4692 / 28.3992 / 28.5493).

## What remains unreviewed

`qpu_cosmo_samplers.py` was only reviewed for the hardware procedure (API,
transpilation, credentials), not with a full adversarial pass.

---

## Appendix: docstrings and examples (measured state)

What there was when it was asked, and what there is now:

| | before | now |
|---|---|---|
| definitions without any docstring | 58 | **0** |
| modules with the header in `#` (invisible to `help()`) | 5 | **0** |
| docstrings with an `Args:` section | 81 (24 %) | **193 (57 %)** |
| docstrings with a `Returns:` section | 59 (18 %) | **160 (48 %)** |
| **public functions with undocumented arguments** | **112** | **0** |
| docstrings with an executable example | 0 (0 %) | **17 functions** |

The 43 % still without `Args:` are functions without arguments or private
helpers (`_name`), where a one-line summary is the right thing: an `Args:`
block there would be noise.

**The examples are doctests**, not decorative text: they run with the suite
(`test_docstring_examples_run`). An example that does not run is worse than
none because it goes stale silently. The very first one written already
caught a mistake — it claimed `H(0) == 70.0` exactly when it is
`69.99999999999999` because of floating-point rounding.

They are placed where they teach something that was expensive to learn:

- `estimate_qubits_and_mem` — that 20 qubits with CC+BAO+Pantheon are
  **88 GB**, not the 14 the old model estimated (`[B-MEM]`).
- `genetic_noisy_seconds_per_gen` — that 14 qubits are **7.7 h per
  generation** (`[B-TIME]`).
- `noisy_density_bytes` — that the parameter-shift batch at 12 qubits is
  45 GB versus the 256 MB of a single ρ.
- `CosmoModel.H` — that wCDM with w=−1 and CPL with w0=−1, wa=0 reproduce
  ΛCDM **exactly** (`array_equal`, not `allclose`). It is the proof that the
  physics is right, and it now runs on every test run.
- `fit_statistics` — that the best fit always falls inside the prior
  (`[B-BOUNDS]`).
- `ess_weights` — the contrast that `[B-ESSCOMP]` measured wrongly.
