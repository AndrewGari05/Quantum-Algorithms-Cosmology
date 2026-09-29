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
  the genetic operators) is transpiled once for the device, and that
  instruction-set (ISA) circuit runs at every location, including its SWAPs.
  `circuits.json` lists the content fingerprint, depth and two-qubit gate
  count of each one.
* **Estimator.** Every location samples with the same number of shots. No
  location uses exact probabilities, not even the ideal simulator.
* **Randomness.** Each algorithm starts from the same seed at every location,
  and the backends consume the random stream identically. SPSA directions,
  proposal angles and genetic draws therefore coincide; only the measurement
  outcomes differ.
* **Budget.** The steps, iterations, generations, shots per circuit and
  circuits per job are the same at every location.
* **Device settings.** Execution is raw: no dynamical decoupling, no
  twirling, no readout mitigation. The simulators do none of these either.

## What runs (defaults: ΛCDM, CC+BAO)

| algorithm | circuit | recipe | jobs per location |
|---|---|---|---|
| `vi` (QVMC) | thesis ansatz, 6 qubits (3 per parameter), 3 layers, CX ring | SPSA, 60 iterations × 2 circuits × 1024 shots, from a common warm start; 4096 measured samples at the end | 62 |
| `mcmc` (QMCMC) | proposal circuit, 2 qubits | 2000 steps × 4 chains (+200 burn-in); increments from ⟨Z⟩ with 64 shots, 1024 increments per job; classical Metropolis acceptance | 10 |
| `genetic` (QGA) | init, mutation and crossover circuits (4 bits per parameter) | population 60, 20 generations | ~44 |

Classical references are computed locally and are not part of the
location comparison:

* a long Gaussian-proposal MCMC (20000 × 8), used as the posterior reference;
* the same short MCMC with a Gaussian proposal;
* the grid GA with classical operators.

**Why the VI starts warm.** Training the 42-parameter ansatz from scratch
with SPSA takes about 1000 iterations (2000 jobs), which does not fit in a
10-minute device budget. The common warm start is the best of 4 COBYLA runs
on the exact KL, computed once on a statevector, with exact KL ≈ 0.03. The
hardware experiment then measures two things: whether 60 SPSA iterations on
each location keep, improve or degrade that solution, and what each
iteration costs. `--vi-init cold` runs the from-scratch experiment instead.

**Why the MCMC acceptance stays classical.** The acceptance of step *t*
depends on the state after step *t − 1*, so it would take one device job per
step. With all chains in one job, that is still about 2,200 jobs for this recipe, each with its own queue wait.
Proposals do not depend on the state, so they are generated in blocks.

## How to run

```bash
pip install -e ".[all]"            # qiskit-ibm-runtime is pinned to the tested range

# once: save the IBM Quantum Platform API key and instance (CRN)
python -c "from qiskit_ibm_runtime import QiskitRuntimeService as S; \
S.save_account(channel='ibm_quantum_platform', token='<API key>', instance='<CRN>', set_as_default=True)"

# 1. what will be submitted (jobs, shots, rough device minutes, compiled depth); runs nothing
python -m thesis hardware --device least_busy --out results/hw --dry-run

# 2. all three locations in one command (the device is used through a Batch)
python -m thesis hardware --device ibm_fez --out results/hw --max-quantum-seconds 480
```

`--max-quantum-seconds` stops submitting device jobs once the billed time
reaches the limit. The algorithm running at that point is recorded with an
`error` column, and everything finished before it is kept. The run order is
vi → genetic → mcmc; use `--algorithms mcmc vi` to change the priority.
Without an account, `--device fake_fez --locations ideal noisy` runs the two
simulated locations against a snapshot of the device.

## Outputs (in `--out`)

| file | content |
|---|---|
| `results.csv` | One row per algorithm and location: posterior mean and std, shift from the reference in reference σ, width ratio, χ², R-hat/ESS (MCMC), exact and shot KL (VI), and the timing columns below. |
| `jobs.csv` | One row per job: circuits, shots, compiled depth and two-qubit gates, wall, queue, execution span, billed quantum seconds, job id. |
| `vi_trace.csv` | Per SPSA iteration and location: shot KL, exact KL of the iterate, cumulative backend and quantum time. This gives time to reach a KL level. |
| `circuits.json` | Recipe, device, transpiler seed, reference posterior, compiled-circuit fingerprints. |
| `summary.md` | The comparison table. |

Timing columns:

* `wall_s` is the total for the run.
* `backend_wall_s` is the time inside backend calls.
* `classical_s` is `wall_s − backend_wall_s`: likelihoods, the optimizer and bookkeeping.
* `queue_s` is the wait in IBM's queue.
* `execution_s` is the execution span on the device, or the simulator's run time.
* `quantum_s` is the time IBM bills.

Compilation happens once and is listed per circuit in `circuits.json`.

## Reading the comparison

* **Cost:** `execution_s` and `quantum_s` per circuit, compared across
  locations. The queue is reported but is not a property of the algorithm.
* **Fidelity:** `shift_*_sigma` and `width_ratio_*` against the reference
  posterior, the VI exact KL, and the genetic χ² against `chi2_grid_floor`.
* **Noise effect:** `noisy` vs `ideal`, same process. **Twin accuracy:**
  `device` vs `noisy`.

None of this shows a quantum speed-up. The classical MCMC reference produces
a better posterior in seconds. The experiment measures what the circuit-based
components cost on current hardware and how much of the ideal result
survives the device's noise.
