# Running QMCMC, QVMC and QGA on IBM hardware

`python -m thesis hardware` runs the three circuit-based algorithms with **one
recipe at three locations** and reports what each run cost and how much
fidelity it kept:

| location | what executes the circuits |
|---|---|
| `ideal`  | Aer, no noise |
| `noisy`  | Aer with the device's calibrated noise model (the noisy twin) |
| `device` | the IBM device, through `SamplerV2` in a `Batch` |

The runs are only comparable if they are the same process, so the recipe fixes:

* **Circuits.** Each logical circuit (the thesis ansatz, the proposal circuit,
  the genetic operators) is transpiled once for the device, before any location
  runs. That instruction-set (ISA) circuit runs at every location, including its
  SWAPs. `circuits.json` lists each circuit's content fingerprint, depth,
  two-qubit gate count, duration per shot and physical qubits.
* **Schedule.** Circuits are scheduled (ALAP), with explicit delays on their
  qubits, so the noisy twin applies the same idle (T1/T2) noise the device
  sees.
* **Estimator.** Every location samples with the same number of shots. No
  location uses exact probabilities, not even the ideal simulator.
* **Randomness.** Each algorithm starts from the same seed at every
  location, and the backends never draw from that stream: simulator seeds
  have their own, restarted per algorithm, so an algorithm gives the same
  result whether or not the others ran before it. SPSA directions, proposal angles and genetic draws
  therefore coincide at every location. Only the measurement outcomes differ,
  and what the algorithm does with them.
* **Budget.** Steps, iterations, generations, shots per circuit and job sizes
  are the same everywhere. The genetic initialization is one job per
  parameter, sized so it almost never needs a second job, so its job count
  does not depend on the outcomes.
* **Device settings.** Execution is raw: no dynamical decoupling, no
  twirling, no readout mitigation. The simulators do none of these either.

## What runs (defaults: ΛCDM, CC+BAO)

| algorithm | circuit | recipe | jobs per location |
|---|---|---|---|
| `vi` (QVMC) | thesis ansatz, 6 qubits (3 per parameter), 3 layers, CX ring | 4096 samples measured at the common warm start (control); SPSA (a = 0.005, c = 0.05, plug-in shot KL), 60 iterations × 2 circuits × 1024 shots; 4096 measured samples at the end | 63 |
| `mcmc` (QMCMC) | proposal circuit, 2 qubits | 2000 steps × 4 chains (+200 burn-in); increments from ⟨Z⟩ with 64 shots, 1024 increments per job; classical Metropolis acceptance | 10 |
| `genetic` (QGA) | init, mutation and crossover circuits, 6 bits per parameter (crossover: 12 qubits) | population 60, 20 generations | 42 |

Classical references are computed locally and are not part of the
location comparison:

* a long Gaussian-proposal MCMC (20000 × 8), used as the posterior reference;
* the same short MCMC with a Gaussian proposal;
* the grid GA with classical operators.

**Why the VI starts warm.** Training the 42-parameter ansatz from scratch
with SPSA takes about 1000 iterations (2000 circuits, 1000 jobs), which does
not fit in a 10-minute device budget.

* **The warm start.** It is the best of 4 COBYLA runs on the exact KL,
  computed once on a statevector (exact KL ≈ 0.025). Every location starts
  from it.
* **What is measured.** Whether 60 SPSA iterations on each location keep or
  degrade that solution, and what each iteration costs.
* **The gains.** They are small, so the ideal location stays near the warm
  start: exact KL 0.025 → 0.027–0.028 after 60 iterations (3 seeds). The
  remaining drift is SPSA's gradient noise diffusing around a sharp optimum:
  it grows with the gain. The plug-in shot KL is biased toward concentrated
  distributions by about (m − 1)/2N, but the Miller–Madow correction did not
  reduce the drift at these gains (3 seeds: 0.027–0.030 corrected vs
  0.027–0.028 plug-in). The plug-in estimate is therefore the default from a
  warm start, and `--vi-kl-estimator miller-madow` switches. Shot-KL values
  are comparable only between runs with the same estimator (column
  `kl_estimator`).
* **Separating noise from training.** Before training, each location
  measures 4096 samples at the common start (`start_shift_*`,
  `start_width_ratio_*`). That is the effect of noise on sampling alone.
  The difference between those and the final samples is how SPSA reacts to
  the location's noise. The gap between the ideal and noisy locations after
  training is noise plus the optimizer's reaction to it, not noise alone.
* **Other options.** `--vi-init cold` runs the from-scratch experiment instead;
  `--spsa-a` and `--spsa-c` override the gains.

**Why the MCMC acceptance stays classical.** The acceptance of step *t*
depends on the state after step *t − 1*, so it would take one device job per
step. With all chains in one job, that is still about 2,200 jobs for this
recipe, each with its own queue wait. Proposals do not depend on the state,
so they are generated in blocks.

**What the MCMC comparison can and cannot show.** Increments are rescaled to
unit size and given a random sign, so the proposal is symmetric on any
backend. The chain therefore targets the exact posterior at every location,
by construction. Noise can change efficiency (acceptance, ESS per second) but
not what the chain converges to. Shifts are reported with their Monte Carlo
error (`shift_mcse_sigma` ≈ 1/√ESS); a shift smaller than about twice that
is not a noise effect. The informative MCMC numbers are `ess_per_quantum_s`
and `ess_per_backend_wall_s`.

**What the genetic comparison measures.** The best χ² per generation
(`chi2_best_by_generation`) and the first generation that reaches the grid
floor (`generation_reached_floor`). With 6 bits per parameter the grid has
4096 cells and the GA evaluates about 1140 individuals, so reaching the floor
is not guaranteed by exhaustion.

**Seeds.** One device run is one seed. Repeat `ideal` and `noisy` over
several seeds (one `--out` folder per `--seed`) to put an error bar next to
the single device run. This matters most for the GA, whose best χ² varies
by a few tenths between seeds; a single-seed difference between operators or
locations is not evidence of an effect.

## How to run

```bash
pip install -e ".[all]"            # qiskit-ibm-runtime is pinned to the tested range

# once: save the IBM Quantum Platform API key and instance (CRN)
python -c "from qiskit_ibm_runtime import QiskitRuntimeService as S; \
S.save_account(channel='ibm_quantum_platform', token='<API key>', instance='<CRN>', set_as_default=True)"

# 1. what will be submitted (jobs, shots, rough device minutes, compiled circuits); runs nothing
python -m thesis hardware --device least_busy --out results/hw --dry-run

# 2. all three locations in one command, on the device the dry run printed
python -m thesis hardware --device ibm_fez --out results/hw --max-quantum-seconds 480
```

Budget and failures:

* **Budget guard.** `--max-quantum-seconds` stops submitting device jobs once
  the billed time reaches the limit. IBM finalizes the billed time a few
  seconds after each job, so it is polled for up to 30 s per job (at most
  about an hour over ~115 jobs, usually far less). If it is still not final,
  the job is charged the largest of IBM's usage estimate, the execution span
  and a pessimistic model (2 s per job + 250 µs per shot) and flagged
  `quantum_estimated`, so the guard stops early rather than late.
* **Failed algorithms.** An algorithm stopped by the budget, or by a failed
  job, is recorded with an `error` column and the next one runs. Everything
  finished is written after each location.
* **Priority.** The order is vi → genetic → mcmc; `--algorithms mcmc vi`
  changes the priority.

Wall-clock time:

* **Open plan.** It offers job and batch modes, not sessions. SPSA and the
  GA are sequential, so every one of their jobs waits in the queue again.
  With a busy device, the device location can take hours even though it
  bills only minutes.
* **What the queue affects.** `queue_s` records the wait, and it is not part
  of the cost comparison.

Without an account, `--device fake_fez --locations ideal noisy` runs the two
simulated locations against a stored calibration of the device (its date is
in `circuits.json`).

## Outputs (in `--out`)

| file | content |
|---|---|
| `results.csv` | One row per algorithm and location. All: posterior mean and std, shift from the reference (with its Monte Carlo error), width ratio, χ², and the timing columns below. MCMC: R-hat and ESS. VI: exact and shot KL. GA: generation that reached the grid floor. |
| `jobs.csv` | One row per job: circuits, shots, compiled depth and two-qubit gates, wall, queue, execution span, billed quantum seconds (estimated or final), metadata time, job id. |
| `vi_trace.csv` | Per SPSA iteration and location: exact KL of the iterate, the mean shot KL of the two perturbed evaluations SPSA used, and cumulative backend and quantum time. This gives time to reach a KL level. |
| `circuits.json` | Recipe, device and calibration date, transpiler seed, reference posterior, compiled circuits. |
| `summary.md` | The comparison table. |

Timing columns:

* `wall_s`: total for the run.
* `backend_wall_s`: time inside backend calls, including metadata retrieval.
* `classical_s`: `wall_s − backend_wall_s` (likelihoods, the optimizer, bookkeeping).
* `queue_s`: wait in IBM's queue.
* `execution_s`: execution span on the device, or the simulator's own run time.
* `quantum_s`: billed time.

Compilation is done once, before the first location, and listed per circuit
in `circuits.json`.

## Reading the comparison

* **Cost:** `execution_s` and `quantum_s` per circuit, compared across
  locations, and for MCMC the ESS per second. The queue is reported, but it
  is not a property of the algorithm.
* **Fidelity:** `shift_*_sigma` (against `shift_mcse_sigma`) and
  `width_ratio_*` against the reference posterior, the VI exact KL, and the
  genetic χ² against `chi2_grid_floor`.
* **Noise effect:** `noisy` vs `ideal`, same process.
* **Twin accuracy:** `device` vs `noisy`.

None of this shows a quantum speed-up. The classical MCMC reference produces
a better posterior in seconds. The experiment measures what the circuit-based
components cost on current hardware and how much of the ideal result
survives the device's noise.
