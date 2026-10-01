# Reading of the August–September 2026 campaigns

Analysis of the results downloaded from the HPC. **The dates in the folder
names are wrong** (the node's clock is wrong); the real order is
reconstructed from the content, not from the name.

| folder | dataset | contents | status |
|---|---|---|---|
| `hpc_20260827_175951` | CC+BAO | samplers + genetic, no noise axis | finished |
| `hpc_20260831_173647` | CC+BAO+Pantheon | samplers + genetic, no noise axis | finished, 3 dead tasks |
| `hpc_20260903_161310` | CC+BAO+Pantheon | genetic with noise axis | **running** |

---

## 1. The "empty" folders are not empty: they are OOMKills

No folder is really empty. Three have the complete log but are missing the CSV
and the figures, which is exactly the signature of a `SIGKILL`:

| task | qubits | planner estimate | RSS measured before dying | `returncode` |
|---|---|---|---|---|
| `samplers_cpl_nqpp5` | 20 | 14.2 GB | 58.2 GB | **-9** |
| `samplers_gede_nqpp7` | 21 | 28.1 GB | 55.0 GB | **-9** |
| `samplers_wcdm_nqpp7` | 21 | 28.1 GB | 27.8 GB | **-9** |

`returncode = -9` is `SIGKILL`: the kernel's OOM killer. There is no traceback
and no partial results because the process does not get to execute anything
when it dies.

All three died **at exactly the same point**: the last log line of all three
is `Adaptive QVMC grid window (shared): ...`, i.e. right after finishing MCMC
and QMCMC and right as QVMC started. That is why the MCMC and QMCMC of those
configurations **are** in the log even though they are not in the CSV.

**It was not the noise** (that campaign has no noise axis) nor "too much
nqpp" in the abstract: it was the planner. See §4.

In `hpc_20260903_161310` there are no dead tasks; the folders with 5 files
instead of 7 (`lcdm/pede` × `nb5,nb6,nb7` × `readout`) are the ones that
**were still running** when the results were downloaded. They are missing only
what the child writes at the end. In `genetic_lcdm_nb6_noise-readout` the QGA
at 100 % was at generation 10 of 500 after 4 hours: ~24 min/generation, i.e.
about 8 days for that single cell. That is the cost of the density matrix at
12 qubits, not a hang.

---

## 2. The fit is good

Raw `χ²` against `n_data`, identical across all rungs because it is the `χ²`
of the same best fit:

| model | params | χ² | n_data | χ²_ν | AIC | BIC |
|---|---|---|---|---|---|---|
| CPL | Ωm, H₀, w₀, wₐ | 1058.64 | 1099 | **0.967** | 1066.6 | 1086.6 |
| GEDE | Ωm, H₀, Δ | 1063.87 | 1099 | **0.971** | 1069.9 | 1084.9 |
| wCDM | Ωm, H₀, w | 1063.83 | 1099 | **0.971** | 1069.8 | 1084.8 |
| ΛCDM | Ωm, H₀ | 1064.61 | 1099 | **0.970** | 1068.6 | 1078.6 |
| PEDE | Ωm, H₀ | 1078.91 | 1099 | **0.984** | 1082.9 | 1092.9 |

χ²_ν ≈ 0.97 with 1099 points is a correct fit. (This answers the question
raised by the animation: the ~1064 values are the **raw** χ², not the reduced
one.)

By **BIC**, ΛCDM wins: the extended models lower χ² by between 0.8 and 5.9,
which does not pay for the 2 extra parameters. ΔBIC(ΛCDM→CPL) = +8.0 in favor
of ΛCDM. By AIC the difference is smaller but goes in the same direction,
except for CPL (ΔAIC = −2.0, "inconclusive" on the Jeffreys scale). **Nothing
here requires physics beyond ΛCDM.**

With **CC+BAO alone** (`hpc_20260827`) the χ²_ν come out at 0.52–0.66 with 51
points. That is not a bug: the cosmic-chronometer error bars include
systematics and are inflated, so that dataset constrains conservatively. It is
worth saying so in the thesis rather than being asked about it.

### Where the parameters land

ΛCDM with CC+BAO+Pantheon: **Ωm = 0.2766 ± 0.0105**, **H₀ = 69.57 ± 0.80**.

- H₀ falls between Planck (67.66) and SH0ES (73.04), at ~2.4σ and ~4.3σ
  respectively, with statistical-only σ. This is what is expected from a
  background fit with SNe and no absolute calibrator.
- Ωm is **3.3σ below** Planck (0.3111). This is known for *stat-only*
  Pantheon without CMB, but it should be said explicitly rather than letting
  the figure hint at it on its own.

---

## 3. The central result: fidelity, and its price

### 3.1 The *faithful* cells hold exactly

| comparison | result |
|---|---|
| CGA vs QGA(0 %) | **identical to 6 decimals**, in every configuration, with and without noise |
| QMCMC 50 % vs QMCMC 100 % | **bit-for-bit equal** |
| Classical VI vs QVMC 33 % | identical KL; means equal to the 5th digit |
| QVMC 67 % vs QVMC 100 % | **bit-for-bit equal** |

This is the identity criterion, and it holds. It is the strong claim of the
thesis and it is clean.

### 3.2 QMCMC reproduces classical MCMC with very high fidelity

Deviation of the mean with respect to classical MCMC, in units of MCMC's own
σ (all models, all nqpp):

| model | max\|pull\| | σ_QMCMC / σ_MCMC | ESS |
|---|---|---|---|
| PEDE | 0.003 σ | 0.983 | 5647 |
| wCDM | 0.019 σ | 0.991 | 1917 |
| ΛCDM | 0.025 σ | 0.986 | 5659 |
| GEDE | 0.043 σ | 1.009 | 1884 |
| CPL | 0.100 σ | 0.982 | 381 |

Means within 0.1 σ and widths within 1–2 %. It is the best result of the
campaign and it does not depend on `nqpp` (the QMCMC engine uses `max(2, d)`
qubits).

### 3.3 The quantum gradient **degrades**, and the effect grows with the circuit

Final QVMC KL against the reference posterior, at fixed `nqpp`:

| model | nqpp | classical VI ≡ QVMC 33 % | QVMC 67 % ≡ 100 % | time 33 % | time 67 % |
|---|---|---|---|---|---|
| ΛCDM | 4 | 0.173 | **0.757** | 20 s | 986 s |
| ΛCDM | 6 | 0.490 | 0.620 | 43 s | 3660 s |
| ΛCDM | 9 | 0.519 | **1.127** | 395 s | 45782 s |
| PEDE | 4 | 0.409 | 0.742 | 33 s | 1421 s |
| PEDE | 9 | 0.619 | **1.506** | 650 s | 56603 s |
| GEDE | 6 | 1.378 | 2.080 | 433 s | 36274 s |
| wCDM | 6 | 1.213 | 1.688 | 1235 s | 54580 s |

In **20 of 24** cells the rung trained with parameter-shift comes out
**worse** than the one trained with the classical gradient, and the gap
**grows with the number of qubits** (ΛCDM: 0.757 → 1.127 going from nqpp 4 to
9; PEDE: 0.742 → 1.506).

> **CORRECTION.** An earlier version of this document attributed that gap to
> "the variance of the parameter-shift estimator with a finite number of
> shots". **That is false in the ideal run**, for two independent reasons:
> (1) with `--noise none` QVMC reads amplitudes, not counts, so there are no
> shots and no sampling variance; (2) the parameter-shift rule is implemented
> on the *probabilities* `Q_i = ⟨ψ|Π_i|ψ⟩` with the chain rule into the KL,
> and for RY/RZ gates (eigenvalues ±1/2) it is **exact**, not a finite
> difference — it is verified against central differences in the tests. So
> there is neither shot noise nor gradient bias.
>
> What really separates the rungs is that **they are different optimizers**:
> classical VI uses COBYLA and the 67 % rung uses SGD with learning-rate decay
> on the parameter-shift gradient. That is exactly what the project's taxonomy
> labels as an ALGORITHMIC cell, and it is correct for them to differ.

**And the SGD is NOT CONVERGED at 5000 iterations.** Measured on the KL trace
from the logs themselves, comparing the KL at iteration 4000 against 5000:

| cell | classical VI (COBYLA) | QVMC 67 % (SGD) |
|---|---|---|
| ΛCDM nqpp6 | −0.5 % | −1.3 % |
| ΛCDM nqpp9 | −0.5 % | **−2.9 %** |
| wCDM nqpp6 | −1.3 % | **−3.3 %** |
| GEDE nqpp6 | −0.5 % | **−3.8 %** |

The classical one is already on its plateau; the quantum one keeps going
down, and **goes down more precisely where the gap is largest**. So part of
the gap is not a property of the method but a lack of iterations: the angle
space grows with `nqpp` and SGD needs more steps to traverse it. How large a
part is unknown without running it — closing from 1.127 to 0.519 in ΛCDM
nqpp9 requires a 54 % reduction, and the current rate is ~3 % per 1000
iterations, decreasing.

**Practical consequence:** `--qvmc-iter` is a real lever and lowering it is a
mistake. Any claim about the gap between rungs must come from a run where SGD
has reached its plateau, or must state explicitly that it is not converged.

The cost is **30× to 100×** in time. Concrete example: GEDE at nqpp=6, 433 s
with the classical gradient versus **10 hours** with parameter-shift.

The conclusion that these data do support is that of the *faithful* cells
(§3.1–3.2): the quantum pipeline **reproduces** the classical one where the
criterion is identity, and costs two orders of magnitude more. Fidelity, not
advantage. The statistical comparison of the 67 % rung remains **pending**
until the SGD is converged.

> **Trap to avoid.** There are 4 cells where 67 % comes out better: GEDE
> nqpp 3 and 4, PEDE nqpp 3, ΛCDM nqpp 3. All are at `nqpp = 3`, where
> classical VI also does badly, and none survives increasing the resolution.
> It is noise in the KL estimator, not an advantage. Quoting them without this
> context would be exactly the kind of claim this project does not make.

> **KL is NOT comparable across different `nqpp`.** It is measured against a
> reference discretized on the grid itself, and the grid changes with `nqpp`.
> Only comparisons **between rungs at fixed `nqpp`** are valid — which is
> exactly where the result above lives. That the classical VI KL is not
> monotonic in `nqpp` (ΛCDM: 1.80, 0.17, 0.25, 0.49, 0.49, 0.52, 0.52) is due
> to this, not to an instability of the method.

### 3.4 VI underestimates the width, and more so the finer the grid

σ_VI / σ_MCMC for ΛCDM as `nqpp` rises from 3 to 9: 0.96, 0.91, 0.80, 0.63,
0.64, 0.62, 0.64. This is the variance underestimation inherent to mean-field
VI, which gets worse as the variational family gets finer. **It is not a
quantum effect** — QVMC 33 % follows it point by point. It must be stated
when reporting any σ that comes from VI or QVMC.

### 3.5 Readout noise: the operator holds up

Genetic `lcdm`, comparing `none` against `readout` in the same configuration:

| cell | Ωm (without / with noise) | Ωm spread | H₀ spread | time |
|---|---|---|---|---|
| nb4, q=33 % | 0.276301 / 0.276301 | same | same | same |
| nb4, q=67 % | 0.270002 / 0.270002 | +9 % | +21 % | 34 s → 45 s |
| nb6, q=67 % | 0.277478 / 0.277470 | +20 % | +13 % | 39 s → 91 s |
| nb4, q=100 % | 0.270004 / 0.270155 | ×6 | ×2.4 | 199 s → **2274 s** |

The mean **does not move**; what grows is the convergence spread, between
10 % and 20 % in the 67 % rungs, and considerably more at 100 %. The time cost
of noise at 100 % is 11× at nb4 and grows very fast with `n_bits`.

### 3.6 The apparent QGA bias at 67 %/100 % is **resolution**, not quantum

| `n_bits` | QGA 67 % Ωm | QGA 67 % H₀ | CGA (reference) |
|---|---|---|---|
| 4 | 0.270002 | 70.309 | Ωm = 0.276289 |
| 6 | 0.277478 | 69.450 | H₀ = 69.605 |
| 8 | 0.276849 | 69.574 | |

As `n_bits` increases the estimate **converges to the CGA**: at nb8 the
difference is 0.0006 in Ωm and 0.03 in H₀, i.e. **0.06 σ and 0.04 σ** in MCMC
units. What looked like a bias of the quantum operators at nb4 was the grid of
16 values per axis. This was the missing control and the data already settle
it: the QGA quantum operators are faithful, and the nb4 error is
discretization.

---

## 4. Two bugs found while reviewing these results

### `[B-MEM]` — the planner's memory model (fixed)

This is the one that caused the three OOMKills. The per-state cost of a
samplers task **depends on `N_data`** (arrays of shape `(2^q, N_data)`), and
the model used a constant. Measured:

| `total_q` | CC+BAO (51 pts) | CC+BAO+Pantheon (1099 pts) |
|---|---|---|
| 16 | 1.0 GB | 4.8 GB |
| 18 | 4.0 GB | 19.6 GB |

14 kB/state with 51 points, 74 kB/state with 1099. The old constant
(13.3 kB) captured only the dataset-independent term — the comment said it
*was* the `(n_states, N_data)` arrays, but a constant number cannot be that.
It underestimated by 5.3× at 18 qubits.

Fixed to `bytes/state = (11500 + 57·N_data) × 1.15`, calibrated against the 7
real measurements from the two campaigns and with a safety margin. It
reproduces all of them from above (factor 1.14–1.42) and never from below. 5
new regression tests, including one that verifies that the three dead tasks
are now rejected at planning time.

### `[B-GAPLOT]` — the genetic figure contradicted its own table (fixed)

`genetic_convergence_*.png` drew `theta_map` with the **unweighted** standard
deviation of the final population; the CSV reports the **fitness-weighted**
mean and standard deviation. In `lcdm/nb6` that gave ±0.0165 in the figure
versus ±0.0014 in the table: twelve times wider, with nothing indicating which
was which. Unified on the weighted version (the one that goes into the CSV),
with the MAP as a marker on top and a caption note that the bar is the
**optimizer's convergence spread, not a credible interval**.

### `[B-FIDSCALE]` — the fiducial squashed the figure (fixed)

In the same figure, the Planck fiducial line set the axis scale: the rungs
live between 0.2763 and 0.2775 and the fiducial is at 0.3111, so the axis was
stretched to 0.275–0.311 and the differences between rungs — the content of
the figure — ended up within a single pixel. Now the limits are set by the
trajectories; if the fiducial falls outside, it is annotated at the edge.

---

## 5. What is missing before quoting any number

1. **A converged SGD.** This comes first, because without it §3.3 cannot be
   quoted: raise `--qvmc-iter` until the KL decrease in the last 20 % of
   training is comparable to that of classical VI (~0.5 %). At 5000 it is
   around 3 %.
2. **Multiple seeds.** All of this is `--seed 42`. The parameter-shift
   degradation (§3.3) appears in 20 of 24 cells and grows systematically with
   the number of qubits, so the *trend* is solid; but any concrete KL *number*
   needs at least 3–5 seeds and an error bar — plus the converged SGD of the
   previous item.
3. **The samplers campaign with noise.** The run in progress is genetic only.
   The QVMC rungs under noise are the empty cell of the axis.
4. **The `none-counts` control column.** Without it, comparing the ideal rung
   (which reads amplitudes) against the noisy ones (which read counts)
   measures the noise and the change of operator at the same time.
5. **Relaunch `cpl/nqpp5` and `gede,wcdm/nqpp7`** with enough `--max-task-gb`,
   or accept that the real ceiling of this node is 18–19 qubits with
   CC+BAO+Pantheon and say so in the thesis.
