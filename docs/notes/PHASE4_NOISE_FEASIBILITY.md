# Phase 4 — NISQ noise axis: feasibility and implementation

**Part A (feasibility)** — preliminary study, no production module touched.
**Part B (implementation)** — the axis is now wired in; see §8 at the end.

Suite: **33 → 62 tests**, all green.

Verification environment: qiskit 2.4.2, qiskit-aer 0.17.2, numpy 1.26.4,
qiskit-ibm-runtime 0.49.0. Baseline before starting: **33/33 tests pass**.

Everything below is measured, not read. Reproduce with:

```
python noise_feasibility_probe.py          # probes A, B, D
python noise_feasibility_probe.py --cost   # adds C (slow)
```

---

## 1. The shortcut hypothesis: **it holds, with one condition**

`SamplerV2` from `qiskit-ibm-runtime` accepts a simulated backend in `mode=`
(*local testing mode*) and runs without credentials or network. The three
wrappers that `QPUConnection` uses work: `mode=backend`,
`mode=Batch(backend=...)` and `mode=Session(backend=...)`. The `run_pub`
decoding survives intact: with `B=2` bindings, `reg.get_counts(k)` returns the
expected values and the bitstrings keep the width of the classical register
(3 bits), **not** the 127 physical qubits that `FakeBrisbane` transpiles to.

That is: the counts-based pipeline can be pointed at simulated noise without
touching the ideal simulator. The hypothesis is correct.

### The condition — and it is a finding, not a detail

In local testing mode, `qiskit-ibm-runtime` **silently ignores error
suppression**:

```
UserWarning: Options {'dynamical_decoupling': {'enable': True,
  'sequence_type': 'XY4'}, 'twirling': {'enable_gates': True,
  'enable_measure': True}} have no effect in local testing mode.
```

`QPUConnection.__init__` sets those four options and never checks that they
took effect. Direct consequence: a noisy run along this route measures
**noise without DD or twirling**, while the run on real hardware measures
**noise with DD and twirling**. They are not the same experiment, and today
nothing in the code or the logs distinguishes them.

This does not invalidate the shortcut — it turns it into a **lower bound** on
hardware quality, which is a perfectly defensible reading. But it has to be
recorded in the CSV as a column, not inferred.

Classification: **HIGH** (it does not affect the correctness of the current
results, which are all ideal; it affects the interpretation of every future
noisy result and of the claim "the QPU pipeline uses XY4 DD + twirling").

---

## 2. Correction to the problem map: there are not three readouts, there are six

The design document lists three places where the ideal simulator reads
amplitudes. `grep` over `cosmo_modular_quantum.py` finds **six** (lines
500/505, 584/590, 627/636, 854, 1017/1025, 1200/1201). Any effort estimate
based on "three places" is an underestimate.

But what matters is not the count, it is that **they are not equivalent to
each other**:

| # | Place | What it reads | Does it survive a mixed state? |
|---|---|---|---|
| 1 | `_raw_block` (proposal) | `Re(ψ)·sign(Im(ψ))` | **No.** It is sensitive to relative phase; it is meaningless for a mixed ρ. |
| 2 | `hadamard_accept_log_batch` | `\|ψ₀\|²` | **Yes.** It is exactly `ρ[0,0]`. |
| 3 | `_kl_batch` | full `\|ψ\|²` | **Yes.** It is exactly `diag(ρ)`. |

Two of the three blockers the document treats as equivalent **are not**.
Readouts 2 and 3 are probabilities, and probabilities do exist in a mixed
state. Only number 1 really breaks — and it already has a counts-based
counterpart in `qpu_cosmo_samplers._counts_to_shift` (⟨Z_q⟩ per qubit).

In addition, `cosmo_genetic_optimizers.py` **already works entirely through
measurement** (`sim.run(..., shots=P, memory=True)` + `get_memory`, lines
878, 951, 997). Zero amplitude readouts. The QGA needs no refactoring: it
accepts a `NoiseModel` as it is.

And `cosmo_core.make_simulator(method, prefer_gpu, n_qubits, **kwargs)`
forwards `**kwargs` to `AerSimulator` and is the **only** place where all of
the project's simulators are created. `noise_model=` enters there without
opening a second path.

---

## 3. Density matrix vs shots: **density matrix**, and the trade-off is not what it seems

Measured with the real ansatz (`build_ansatz(n, n_layers=3)`, 42 parameters),
`B=2` bindings, 4096 shots, `NoiseModel.from_backend(FakeBrisbane)` — the
exact shape of a QVMC-QPU SPSA iteration:

| qubits | density matrix | trajectories | theoretical ρ | measured peak RSS |
|---|---|---|---|---|
| 6  | **1.66 s** | 5.31 s | 0.1 MB | — |
| 8  | **2.33 s** | 6.77 s | 1.0 MB | — |
| 9  | **2.36 s** | 12.19 s | 4.0 MB | — |
| 10 | **3.77 s** | 22.41 s | 16 MB | — |
| 12 | **34.4 s** | 100.9 s | 256 MB | — |
| 13 | **216.5 s** | 333.3 s | 1.0 GB | 1414 MB |

QMCMC proposal engine (block `B=64`, which is how it is really used):

| qubits | density matrix | trajectories |
|---|---|---|
| 2 | **4.15 s** | 51.70 s |
| 3 | **4.27 s** | 55.85 s |
| 4 | **4.30 s** | 66.43 s |

**The density matrix wins on time across the whole range where it fits**,
between 1.5× and 15× faster. The reason is structural: the density matrix
evolves ONCE and then samples the shots from ρ; trajectories re-simulate the
full circuit once per shot, so their cost is `shots × 2^n` while that of ρ is
`4^n` independent of the shots. That is why the advantage grows with the
number of shots and is largest exactly where QMCMC operates (few qubits, many
bindings).

### Where this corrects the design

The design document states: *"if you choose shot-based simulation instead of
a density matrix, the cost moves from memory to time"*. That is true, but the
operational conclusion that follows is not the expected one: **shots do not
buy you qubits.** Turning an OOM into a run that never finishes is not a
usable trade-off. Extrapolating the measured scaling (×2 per qubit for
trajectories), CPL with nqpp=6 (24 qubits) is ~114 hours **per job**, and a
30-iteration QVMC is 31 jobs.

The ~14-qubit ceiling computed in the design is **correct**, but for a
stronger reason than the one given: it is not that the density matrix is
expensive and shots are cheap; it is that **above ~14 qubits neither route is
viable**, one because of memory and the other because of time. The noise axis
is intrinsically a low-qubit experiment.

Translated to the project's models (`n_qubits = n_params × nqpp`):

| model | d | nqpp=3 | nqpp=4 | nqpp=5 | nqpp=6 |
|---|---|---|---|---|---|
| lcdm, pede | 2 | 6 ✅ | 8 ✅ | 10 ✅ | 12 ✅ |
| wcdm, gede | 3 | 9 ✅ | 12 ✅ | 15 ❌ | 18 ❌ |
| cpl | 4 | 12 ✅ | 16 ❌ | 20 ❌ | 24 ❌ |

QMCMC does not appear in this table because its engine uses
`n_qubits = max(2, n_params)` — from 2 to 4 qubits, free under any noise
model, for every model and every nqpp.

**Recommendation: density matrix, with a hard ceiling of 13 qubits** (1 GB of
ρ, 1.4 GB of measured RSS). The runner's third memory model is
`BYTES_PER_STATE_NOISY = 16 · 2^n` on top of the existing samplers model — but
the real ceiling is set by time, not RAM, and it must be an explicit
parameter, not derived from the detected RAM like the other two.

---

## 4. The result that changes the design: ρ is blind to readout error

The FAITHFUL acceptance encodes `A = min(1, e^Δ)` as `cos²(θ/2)` and reads it
as `|ψ₀|²`. Verified:

- `ρ[0,0]` reproduces noiseless `|ψ₀|²` with a difference of **0.000e+00** —
  bit for bit. The change of readout introduces no error of its own.
- `ρ[0,0]` against exact `min(1, e^Δ)`: **1.110e-16**. The identity number is
  independently reconfirmed.

Measured degradation curve:

| channel | mean bias in A | mean relative bias |
|---|---|---|
| ideal | 0.00000 | 0.00000 |
| readout p=0.01 | **0.00000** | **0.00000** |
| readout p=0.03 | **0.00000** | **0.00000** |
| readout p=0.05 | **0.00000** | **0.00000** |
| gate 1e-3 + readout 0.03 | −0.00014 | 0.02598 |
| gate 1e-2 + readout 0.03 | −0.00139 | 0.25977 |

**Readout error does not move `ρ[0,0]` by a single digit, for any p.** This is
physically correct: readout is a *classical* channel *after* the measurement
and does not touch the state. But it means that if the noise axis is
implemented by reading ρ, the `--noise readout` column of the ablation matrix
would come out **identical to the ideal column** — an entire false column, and
false in the worst way, because it does not fail loudly but gives clean,
plausible results.

That counts do see it is verified: with `p=0.03` and 10⁵ shots,
`max|counts − ρ[0,0]| = 0.0305`, against an expected shot noise of 0.0016.
The difference is exactly `p`.

### The way out is analytic, not shot-based

On 1 qubit, readout error is a 2×2 stochastic map on `(ρ₀₀, ρ₁₁)`, so it is
applied in closed form:

```
P_noisy(0) = (1−p)·ρ[0,0] + p·(1−ρ[0,0])
```

Verified against 4×10⁶ shots in five channels: the maximum discrepancy is
0.0005 against an expected shot noise of 0.00025 — i.e. it agrees within ~2σ,
which is what corresponds to the maximum over 8 values.

This matters because the shot-based route **cannot** resolve the bias being
sought: with gates at 1e-3 the bias is 1.4e-4, and distinguishing it from
sampling noise would require ~10⁷ shots **per acceptance evaluation**, at
every step of every chain. Infeasible.

With ρ + the analytic readout map you get **the complete noise axis, exact
and free of shot noise, at the cost of a 1-qubit circuit**.

### And there is publishable physics in the shape of the bias

Depolarization drives ρ toward `I/2`, so the bias is `λ·(1/2 − A)`. The
relative bias scales **linearly** with the gate error rate (2.598% at 1e-3 →
25.977% at 1e-2: a factor of exactly 10, slope ≈ 26). It is not diffuse
degradation: the acceptance is biased **toward 1/2**, over-accepting bad moves
and under-accepting good ones. It is a directional, predictable distortion
with a closed-form law — a considerably stronger result than "noise makes
things worse".

---

## 5. The obstacle the design does not yet cover

The runner dispatches to `cosmo_modular_quantum.py` and
`cosmo_genetic_optimizers.py`. **Never to `qpu_cosmo_samplers.py`.** So the
shortcut, as proposed, delivers noise for QMCMC and QVMC through a route the
runner does not know about, and **delivers nothing for the QGA** — precisely
the method predicted to be most robust.

The clean route is the opposite of the shortcut: `noise_model` through
`cosmo_core.make_simulator`, which reaches the three modules at once, with
conversion to ρ only at places 2 and 3 (where it is exact) and to counts only
at place 1 (which is where it really breaks). The shortcut is still valuable,
but as originally stated: **the first end-to-end test of
`qpu_cosmo_samplers.py`**, which today is validated only in dry-run.

---

## 6. Which previous results would have to be redone

**None.** Nothing verified here contradicts the existing ideal results; on the
contrary, the 1.11e-16 identity is independently reconfirmed. What changes is
the scope of the claims about DD and twirling in any run that is not on real
hardware.

## 7. Decisions pending confirmation

1. `make_simulator` route (reaches the three modules, touches the ideal
   simulator) or shortcut route (touches nothing, leaves the QGA off the
   axis)? I recommend the first, with the second as the end-to-end test of
   the QPU module.
2. Qubit ceiling for noisy tasks: I propose a hard 13.
3. Is the `readout` column computed with the analytic map (exact) or from
   counts (with shot noise)? I recommend analytic.

---

# Part B — Implementation

## 8. What was built

| File | What it is |
|---|---|
| `cosmo_noise.py` | **new.** Single source of truth for the axis: rungs, noise models, analytic readout map, qubit ceiling, shared CLI. |
| `cosmo_core.py` | `[N1]` guard in `make_simulator`. |
| `cosmo_modular_quantum.py` | noisy readout paths + `--noise` + `--proposal-route`. |
| `cosmo_genetic_optimizers.py` | `--noise`; no readout refactor (it already measured). |
| `cosmo_hpc_runner.py` | `--noise-sweep`, third memory model, ceiling of 13, `noise` column in the JSON. |
| `qpu_noisy_simulation.py` | **new.** Noisy twin of the QPU pipeline. |
| `tests/test_noise_axis.py` | **new.** 29 regressions. |

`qpu_cosmo_samplers.py` **was not touched**: it remains intact for real
hardware.

## 9. Three bugs of our own found during implementation

All three were silent — they returned clean, plausible numbers.

**`[N1]`** — `AerSimulator(method='statevector', noise_model=...)` raises
nothing: it runs, reports `COMPLETED` and returns the **ideal** result. An
entire noisy campaign would have come back without noise. `make_simulator`
now rejects that pair.

**`[B-RO]`** — an `add_all_qubit_readout_error` appears in `to_dict()`
**without a `gate_qubits` key**. Interpreting that absence as `[[0]]`
degraded the channel to a single qubit. It matched **exactly at n=1** and
diverged as n and p grew.

**`[B-RECON]`** — rebuilding the model without readout via
`NoiseModel.from_dict()` is **lossy** with the calibrated model: against
FakeBrisbane the rebuilt ρ differs by **6.2e-4** at 3 qubits. On the
synthetic rungs it gave 0.0, so it only showed up with the real backend. The
reconstruction was removed entirely.

## 10. Decision on `_raw_block`: the counts route also exists without noise

`Re(ψ)·sign(Im(ψ))` is a different operator from `⟨Z_q⟩`, not its noisy
version. Comparing the ideal column (amplitudes) against the noisy ones
(counts) would mix the effect of the noise with that of the change of
readout. That is why `--proposal-route counts` is available **also with
`--noise none`**: the axis is measured within a single route, and the change
of route is left as a separate comparison.

## 11. Two results that came out of the implementation

**Calibration absorbs 100% of the uniform readout channel.** With a symmetric
flip p, `⟨Z_q⟩' = (1-2p)·⟨Z_q⟩` — a scalar rescaling — and calibration to
unit std divides it out. Verified to **8e-15 even with p = 0.20**. This
answers the open design question: it does not absorb "part" of the effect, it
absorbs the whole readout component and **none** of the gate component (0.109
in `full`, 0.023 in FakeBrisbane).

Consequence: in the proposal row, the `readout` column comes out
**identical** to ideal-via-counts. It is the same symptom the `[B-RO]` bug
had, so it is documented and pinned by a test so that it is not mistaken for
a regression.

**QVMC normalization is immune to the axis by construction.** The
`quantum_amplitude_normalization` circuit is executed but its result is
discarded: the returned value is the exact sum. QVMC's 100% rung cannot
degrade because of it; any degradation between 67% and 100% comes from
somewhere else.

**Readout error dominates the rejection tail of the acceptance.** An exact
acceptance of A = 0.00248 becomes 0.0323 with p = 0.03: readout sets a floor
of ~p, so moves that should almost always be rejected are accepted **13× too
often**. It is a much larger effect than the gate one and goes against the
intuition that "noise degrades smoothly".

## 12. First end-to-end run of `qpu_cosmo_samplers.py`

The twin really executes it, not in dry-run. QMCMC and QVMC run on all four
rungs. QVMC-LCDM, 8 iterations: KL 11.58 (ideal) → 12.52 (FakeBrisbane).

## 13. Which previous results must be redone

**None.** `--noise none` reproduces the ideal path bit for bit (verified,
round trip from noisy rungs), and an ideal runner run produces exactly the
same output paths as before the axis.

## 14. Pending

- README (noise-axis section, memory table, test count).
- Actually measure the ladders under noise: so far this is the instrument,
  not the results.
