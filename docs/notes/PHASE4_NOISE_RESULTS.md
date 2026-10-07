# Phase 4 — First run of the noise axis: results

A calibration run of the instrument, not a production run. ΛCDM, CC+BAO,
`nqpp=3` (6 qubits), 600 steps, 4 chains, 25 QVMC iterations, `--seed 42`,
25 generations × 60 individuals for the genetic algorithm, `n_bits=3`. Sizes
are small on purpose: the goal is to see whether the axis **measures
anything** and in which direction, not to produce quotable numbers.

Five columns, not four. The fifth (`none-counts`) is the control, and without
it three of the five conclusions below would have come out backwards.

---

## 1. The control was indispensable

`--noise none` uses the amplitude route by default; any noisy rung uses the
counts route. Comparing the first column against the noisy ones mixes **the
noise with the change of readout operator**. The `none-counts` column runs the
ideal rung through the counts route and separates the two.

QMCMC 50%:

| | none (amplitude) | none-counts | readout | full | FakeBrisbane |
|---|---|---|---|---|---|
| σ(Ωm) | 0.0208 | **0.0176** | **0.0176** | 0.0172 | 0.0175 |
| acceptance | 0.4756 | **0.5092** | **0.5092** | 0.5144 | 0.5078 |
| ESS | 101.3 | **140.8** | **140.8** | 151.7 | 142.7 |

Reading only `none` → `readout` one would conclude that **readout noise
narrows the posterior and improves mixing** — σ drops 15%, ESS rises 39%. That
is false. `none-counts` and `readout` are **identical to the last printed
digit**: the readout channel does absolutely nothing to the proposal. The
whole jump is the change of route.

This is the invariance proven in the module: uniform readout rescales
`⟨Z_q⟩` by `(1−2p)` and calibration to unit std divides it out. Here it is
seen in a complete run, not just in the unit test.

**Operational conclusion:** any noise-axis table that compares against
`none` without the control is measuring two things at once.

---

## 2. The FAITHFUL identity breaks exactly where it should

QMCMC 50% (quantum proposal) vs 100% (quantum proposal + acceptance):

| | none-counts | readout | full | FakeBrisbane |
|---|---|---|---|---|
| Ωm 50% | 0.2629 | 0.2629 | 0.2640 | 0.2637 |
| Ωm 100% | **0.2629** | 0.2648 | 0.2647 | 0.2640 |
| ESS 50% | 140.8 | 140.8 | 151.7 | 142.7 |
| ESS 100% | **140.8** | **117.2** | 118.2 | 115.9 |

In the ideal limit the two rows are **identical**: the quantum acceptance
reproduces Metropolis exactly, which is the project's fidelity claim. With
noise they stop being so.

The cleanest signal is **ESS: 140.8 → 117.2 as soon as readout is switched
on**, a 17% loss of mixing that the 50% row does not suffer. That is: the
FAITHFUL component is the one that pays for the noise, and it pays in sampling
efficiency before the central value.

This turns "identity to 1.1e-16" into a **degradation curve**, which is what
was being sought. The ideal number is still there as the starting point; now
it has a slope.

Mechanism, already measured in the module: readout sets a floor of ~p on the
acceptance, so moves that should almost always be rejected are
over-accepted (A = 0.0025 → 0.0323 with p = 0.03, 13×). The chain accepts
garbage and mixes worse.

---

## 3. The advantage of quantum training erodes monotonically

The 33% → 67% jump of QVMC is the ALGORITHMIC cell: training with exact
parameter-shift instead of COBYLA. Its gain in KL:

| rung | KL 33% | KL 67% | **gain** |
|---|---|---|---|
| none | 12.0885 | 10.5892 | **1.499** |
| readout | 12.5115 | 11.1595 | **1.352** |
| full | 12.7569 | 11.5727 | **1.184** |
| FakeBrisbane | 12.7339 | 11.6417 | **1.092** |

**−27% of the quantum advantage on reaching the noise of a real device**, and
the erosion is monotonic across the four rungs. This is the central result of
the run: it is not that QVMC stops working, it is that *what made it better
than its baseline* shrinks at a measurable rate.

---

## 4. Three invariances that confirm the axis is wired correctly

**Classical MCMC is identical in all five columns** (Ωm 0.2569, σ 0.0160,
acceptance 0.5269, ESS 101.6). It is pure NumPy and never touches a circuit,
so the axis cannot move it. That it does not move even in the last digit is
the best evidence that the noise enters only where it should.

**QVMC 67% and 100% agree in every column.** This was anticipated:
`quantum_amplitude_normalization` executes its circuit but discards the
result — it returns the exact sum. That component is **immune to the axis by
construction**, and any degradation between 67% and 100% would come from
somewhere else.

**QGA 0% is bit-identical to CGA on all four rungs.** With all operators
switched off no circuit runs, so the noise has no way in.

---

## 5. Classical VI is NOT invariant, and that must be declared

| | none | readout | full | FakeBrisbane |
|---|---|---|---|---|
| Classical VI KL | 12.0885 | 12.5115 | 12.7569 | 12.7339 |

Classical MCMC does not move; classical VI does. It is not a bug: in this
framework classical VI is `QVMCModular` with all components switched off, and
**it still represents Q with the ansatz circuit** — it only optimizes and
samples classically. The circuit is the substrate, not a switchable
component, so under noise the substrate is noisy for the entire QVMC ladder.

It is physically defensible (on a noisy device, a classically optimized
variational state is also noisy), but it has a consequence that cannot be
overlooked: **the "0%" of the QVMC ladder is not a fixed reference under
noise**, whereas the "0%" of the QMCMC ladder is. The two ladders are not
symmetric on this axis, and a table that puts them side by side invites
reading them as if they were.

Decision pending: leave it like this and document it, or add an option that
forces the QVMC baseline to always run without noise.

---

## 6. ESS lies under noise

| QVMC | none | readout | full | FakeBrisbane |
|---|---|---|---|---|
| ESS 67% | 91.2 | 106.9 | 116.0 | **125.2** |
| KL 67% | 10.5892 | 11.1595 | 11.5727 | **11.6417** |

ESS **rises** by 37% while the fit **gets worse**. Noise flattens the
distribution, and a flatter distribution yields less correlated samples. Any
reading that uses ESS as a quality metric would conclude that noise helps.

σ(H0) grows monotonically along the same ladder (2.876 → 2.907 → 2.949 →
2.976), so the real broadening is there: it is ESS that is useless as a
quality metric on this axis. **Report ESS together with KL or σ, never
alone.**

---

## 7. QGA comes out almost invariant — with a big caveat

| Ωm | none | readout | full | FakeBrisbane |
|---|---|---|---|---|
| CGA / QGA 0% | 0.2575 | 0.2575 | 0.2575 | 0.2575 |
| QGA 33% | 0.2575 | 0.2558 | 0.2558 | 0.2575 |
| QGA 67% | 0.2800 | 0.2800 | 0.2800 | 0.2800 |
| QGA 100% | 0.2800 | 0.2800 | 0.2800 | **0.2400** |

The prediction holds: QGA is by far the most robust. But attributing it to
the algorithm alone would be premature.

With `n_bits=3` the grid has **8 levels per axis**. The values above are grid
points, not continuous numbers: for noise to move the result it has to flip
enough bits to jump to *another cell* AND survive selection and elitism, which
preserve the best individual intact. The coarse discretization is quantizing
the noise until it disappears.

Put differently: part of the observed robustness belongs to the **operator**
(it measures, and a flipped bit looks like extra mutation) and part belongs to
the **grid**. Separating them requires repeating this with `n_bits` 5–6,
where a cell is much narrower. Until then, "QGA is the most robust" is
supported but not isolated.

χ² discriminates nothing here (27.4691 in all 20 cells) because it is
reported at the MAP refined with Nelder-Mead, which erases the GA
differences. For this axis one must look at raw Ωm/H0, not at the refined χ².

---

## 8. What can NOT be concluded yet

- Small sizes (600 steps, 25 iterations). Third-decimal differences in Ωm are
  within the Monte Carlo noise of a single seed. **Nothing here is quotable
  without repeating with several seeds.**
- A single model (ΛCDM, d=2) and a single resolution (nqpp=3).
- The axis measures noise **without error suppression** — DD and twirling are
  inert in simulation. It is a lower bound on what is achievable on hardware.
- The degradation curve has four points, and three of them are synthetic
  models. A real curve requires sweeping `p` continuously, not four named
  rungs.

## 9. What would be worth doing next

1. Sweep `--noise-readout-p` continuously over QMCMC 100% and fit the slope of
   the ESS loss. It is cheap (2–4 qubits) and gives the real curve instead of
   four points.
2. Repeat the genetic run with `n_bits` 5–6 to separate operator robustness
   from grid robustness.
3. Several seeds before quoting any number.
