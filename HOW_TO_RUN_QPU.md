# Running on real IBM hardware

Full procedure for the **first** run on a QPU. The goal of the first time
**is not to obtain results**: it is to check that the whole path works —
credentials, transpilation, submission, queue, response, parsing. The results
come with the second run.

> `qpu_cosmo_samplers.py` has never been run against a real machine.
> Everything below is verified *offline* (signatures of the installed API,
> transpilation against FakeBrisbane's real coupling map, and an end-to-end
> `--dry-run`). Whatever cannot be verified without an account is marked
> **NOT VERIFIED**.

---

## 0. What is already checked

| Item | Status |
|---|---|
| `qiskit-ibm-runtime` installed | 0.49.0 |
| Channel used by the code | `ibm_quantum_platform` — **the current one** (the old `ibm_quantum` is retired) |
| Primitive | `SamplerV2` — the current one, not the retired V1 |
| Transpilation to ISA | **Yes, mandatory and present** (`generate_preset_pass_manager(optimization_level=3, backend=...)`) |
| Circuit transpiled against a real map | Verified with FakeBrisbane: it comes out in `['ecr','rz','sx','x','measure','barrier']`, all inside the backend's basis |
| Execution context | `Batch` by default, `Session` with `--session` (requires a paid plan) |
| Error suppression | XY4 dynamical decoupling enabled |
| Safety cap | `--max-jobs` (200 by default) aborts before submitting if the plan asks for more |

---

## 1. Before touching anything

### 1.1 Install and check

```bash
pip install -r requirements.txt
python -c "import qiskit, qiskit_ibm_runtime as r; print(qiskit.__version__, r.__version__)"
```

It must print `2.4.2 0.49.0`. If `qiskit-ibm-runtime` is missing, this file
does not run: it is the dependency that was missing from the first version of
`requirements.txt`.

### 1.2 Save the account ONCE

Go to <https://quantum.cloud.ibm.com>, copy your **API key** and your **CRN**
(the instance identifier; on the Open plan it is shown in the dashboard).
Then, once, in Python:

```python
from qiskit_ibm_runtime import QiskitRuntimeService
QiskitRuntimeService.save_account(
    channel="ibm_quantum_platform",
    token="YOUR_API_KEY",
    instance="YOUR_CRN",        # the CRN of your instance
    set_as_default=True,
    overwrite=True,
)
```

Check that it was saved:

```python
from qiskit_ibm_runtime import QiskitRuntimeService
s = QiskitRuntimeService()
print([b.name for b in s.backends(operational=True, simulator=False)])
print(s.least_busy(operational=True, simulator=False).name)
```

If this prints machine names, the hard part is done.

> **NOT VERIFIED.** The code calls `QiskitRuntimeService()` with no arguments,
> or with `token=` if you pass `--token`. With `token=` it **does not pass
> `instance=`**, so if your account needs an explicit CRN, `--token` may fail
> where the saved account works. **Use the saved account, not `--token`.**

### 1.3 Check the plan without connecting

```bash
python qpu_cosmo_samplers.py --dry-run --model lcdm --dataset CC \
    --method qmcmc --steps 64 --block 64 --chains 1 --shots 1024 --samples 200
```

It must print `Estimated QPU jobs for this run: 1`. If it says more, do not
submit it.

---

## 2. The smoke test — this is the next-day step

```bash
python qpu_cosmo_samplers.py \
    --model lcdm --dataset CC --method qmcmc \
    --steps 64 --block 64 --chains 1 \
    --shots 1024 --samples 200 \
    --least-busy \
    --max-jobs 2 \
    --outdir results_qpu_smoke \
    --seed 42
```

**What it submits:** 1 job, 1024 shots. It is the minimum that exercises the
whole path. Seconds of QPU time; the rest is queue.

**What to expect in the log, in order:**

1. `Backend: ibm_XXX (127 qubits)` ← the credentials work
2. the transpilation to ISA (no message, but if it fails, it fails here)
3. a long, silent wait ← the queue, this is normal
4. `QMCMC-QPU — 64 steps, 1 chains:` with Ωm, H₀, χ² ← it arrived and was parsed
5. `MEASURED TIMINGS: 1 jobs, mean wall Xs/job` ← the number that matters

If you reach point 5, **the path works** and you can plan the real run. If it
breaks, it breaks cheaply.

**`--max-jobs 2` is the safety net.** If the plan turned out to be larger
than you think, it aborts *before* submitting instead of burning your quota.

---

## 3. The real run (only after the smoke test passes)

```bash
python qpu_cosmo_samplers.py \
    --model lcdm --dataset CC --method qmcmc \
    --steps 1000 --block 64 --chains 4 \
    --shots 4096 --samples 4000 \
    --least-busy \
    --max-jobs 40 \
    --outdir results_qpu_lcdm \
    --seed 42
```

From `--dry-run`: **16 jobs** for 1000 steps with `block 64`, ~17 min of
projected queue (the projection comes from the smoke-test measurement, so it
will be more reliable after running it).

### Why QMCMC and not QVMC the first time

`--method qvmc` spends **one job per SPSA iteration**: `--iters 30` means 30
queued jobs, each waiting its turn. QMCMC groups 64 proposals per job, so it
yields far more per job. Leave QVMC for when the path has been tested.

---

## 4. The detail you must keep clear when comparing

**The dataset is not the same as in your large campaign.**
`qpu_cosmo_samplers.py` only accepts `CC`, `Pantheon+` and `CC+Pantheon+`.
`--dataset CC` loads the `cosmic_chronometers.txt` table — i.e. the **51
CC+BAO points**, exactly the dataset of the `hpc_20260827` campaign (CC+BAO),
and **not** the `CC+BAO+Pantheon` of the large campaigns.

So the honest comparison is:

> **QPU (`--dataset CC`) against the simulator's CC+BAO campaign**, never
> against the CC+BAO+Pantheon one.

`Pantheon+` requires `Pantheon+SH0ES.dat` and `Pantheon+SH0ES_STAT+SYS.cov`,
which are **not** included in the zip.

---

## 5. What the simulated test does NOT cover

`qpu_noisy_simulation.py` replaces the whole `QPUConnection` class with a
local stand-in. Everything that lives *inside* that class stays untested until
the first hardware run. Specifically:

| Untested | Risk |
|---|---|
| `QiskitRuntimeService()` and the saved account | **High** — it is the most likely failure, and the cheapest to fix |
| `service.least_busy(...)` | Medium — if your plan exposes no operational backends, it raises here |
| `generate_preset_pass_manager` against the **real** backend | Low — verified against FakeBrisbane, which has the same map as ibm_brisbane |
| `Batch(backend=...)` and the context lifecycle | Medium — **NOT VERIFIED**: check that the `Batch` is closed; an open context may keep counting |
| ~~The real format of the `SamplerV2` result~~ | **ALREADY TESTED** — exercised with `SamplerV2` in local mode against FakeBrisbane, which returns a genuine `PubResult`. It surfaced a bug (`[B-CREG]`), now fixed |
| Real dynamical decoupling and twirling | Low — on the simulator they are silently dropped (`[DD-INERT]` in the log) |
| ~~Arrival order of the jobs~~ | **ALREADY TESTED** — verified that batch *k* corresponds to parameter row *k*, with `ry(0)→\|0⟩` and `ry(π)→\|1⟩` as an unmistakable signature |

**The two "High/Medium" rows above are literally what the smoke test is
for.** That is why it is worth submitting it before anything else.

---

## 6. If something fails

| Symptom | Almost certain cause |
|---|---|
| `AccountNotFoundError` | you did not save the account, or `set_as_default=False` |
| `IBMInputValueError` about the `instance` | the CRN is missing in `save_account` |
| `Backend ... not found` | the `--backend` name does not exist on your plan; use `--least-busy` |
| Circuit rejected by the gate basis | transpilation did not run; it should not happen, but if it does, it is a bug — report it |
| Silent for hours | it is the queue, not a hang. Check the job in the web dashboard |
| `Estimated QPU jobs` > `--max-jobs` | it aborts on purpose. Raise `--block` or lower `--steps` |

---

## 7. `[B-CREG]` — a bug that surfaced while testing this

The code read the classical register like this:

```python
reg = getattr(data, 'meas', None) or getattr(data, 'c', None)
```

This project's circuits use `measure_all()`, so their creg is named `meas`
and the happy path works. But with **any other name** that expression returns
`None` and two lines later it blows up with
`AttributeError: 'NoneType' object has no attribute 'get_counts'` — a message
that says nothing, and that happens **after** the job has been executed and
billed on the QPU.

Verified with `SamplerV2` in local mode: with a creg named `readout`, the
`DataBin` exposes `data.readout` and the old expression fails. Fixed: it now
looks up known names, falls back to the only field that can return counts
(with a warning), and if it finds nothing it raises an error that **says which
fields were present** — the only useful thing at that point.

With this, the two "Medium" risk rows in the table above are closed before
spending a single second of real machine time.
