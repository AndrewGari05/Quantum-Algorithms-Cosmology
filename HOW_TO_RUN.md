# How to run this on an HPC

Operational guide. The README has the scientific detail; this is the sequence
of commands.

---

## 0. Install

```bash
pip install -r requirements.txt
```

`requirements.txt` includes `qiskit-ibm-runtime`, which **was missing**:
`qpu_cosmo_samplers.py` has always imported it, so a previous clean install
could never run that module.

## 1. Check the node BEFORE launching anything

```bash
python cosmo_hpc_runner.py --gpu-check
pytest -q
```

The first prints which devices Aer declares, which packages are installed and
whether `nvidia-smi` sees a card. **Run it on every new machine.**

> **About `--gpu`.** The PyPI `qiskit-aer` wheel is compiled **for CPU
> only**. Installing `cuquantum-cu12` / `custatevec-cu12` next to it does
> **not** turn it into a GPU build: Aer has to be *built* with CUDA support.
> If `--gpu-check` says Aer does not expose a GPU, `--gpu` will run on CPU —
> it used to do so silently; now it warns and explains why. Check the
> qiskit-aer documentation to find out which GPU wheel matches version 0.17.x
> before installing anything.

The second must give **116 green tests**. If any fails on a new node, stop
there: something in the environment does not match.

## 2. The full run

No SLURM: the runner launches subprocesses directly, one model per process,
and distributes the cores by itself.

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

nohup python -u cosmo_hpc_runner.py \
    --noise-sweep none,readout,full,FakeBrisbane \
    --nqpp-sweep 3 5 \
    --nbits-sweep 4 8 \
    --dataset CC+BAO+Pantheon \
    --steps 20000 --qvmc-iter 15000 --chains 8 --shots 4096 \
    --generations 120 --population-size 500 \
    --total-cores 14 --mem-budget-gb 50 --max-task-gb 24 \
    --noisy-task-hours 48 \
    --seed 42 \
    > campaign.log 2>&1 &
```

To run on **real IBM hardware**, the full procedure is in
`HOW_TO_RUN_QPU.md` — it includes the 1-job smoke test you should submit
*before* anything else.

100 tasks. To monitor it: `tail -f campaign.log`, and `grep -c OK
results/hpc_*/master_profile.csv` to count the ones done so far.

**Why 120 generations and not 500.** Measured in the logs of the previous
campaign: going from 60 to 500 generations gains **at most 0.73 in χ²** out of
1058, and only in CPL (4 parameters). In ΛCDM, PEDE, wCDM and GEDE the gain is
**exactly 0** — the genetic algorithm converges by generation 10. With a
density matrix each generation costs ~24 min at 12 qubits, so 500 generations
means weeks per cell in exchange for 0.7 in χ², which moves neither the AIC
nor the BIC. 120 leaves plenty of margin for CPL and keeps the noise axis in
hours.

Tuned for a container with **63 Gi and 15 cores**. On a machine with more RAM
raise `--max-task-gb` and `--mem-budget-gb`; the runner detects the container
limits (cgroup) by itself if you do not pass them.

What each part does:

| Flag | Why |
|---|---|
| `--noise-sweep` | the four rungs of the axis. It automatically adds a fifth column, `none-counts`, which is the control. |
| `--nqpp-sweep 3 5` | QVMC resolution. The runner **clamps per model**: ΛCDM reaches 6, CPL drops to 3 under noise, and it tells you why. |
| `--nbits-sweep 4 8` | genetic-algorithm resolution. Without noise it can go much higher (16 B/state); with noise it is capped by `--noisy-task-hours`. The noiseless nb7/nb8 rungs are the ones that prove the apparent QGA bias at 67 % was resolution, not quantum. |
| `--max-task-gb 24` | this sets the samplers' qubit ceiling. At 24 GB with CC+BAO+Pantheon the ceiling is 18 q, which admits `cpl/nqpp4` (16 q, 5.8 GB) and `gede,wcdm/nqpp5` (15 q, 3.0 GB) and correctly rejects `cpl/nqpp5` (20 q, ~88 GB), which is the one that died last time. |
| `--mem-budget-gb 50` | aggregate budget. It leaves headroom below the container's hard limit: exceeding it is an OOMKill (SIGKILL, no traceback and no partial results). |
| `--noisy-task-hours 48` | **[B-TIME]** wall-clock budget per noisy genetic task. This and `--generations` set the genetic algorithm's qubit ceiling on the noise axis. The density-matrix QGA is limited not by RAM but by time: ×4.4 per qubit (73 s/gen at 10 q, 1425 s/gen at 12 q, ~7.7 h/gen at 14 q). Without this ceiling the plan accepts cells that take months and that also block the summary figures of the whole run. |
| (profiling) | It is **on by default** in the runner: peak RAM, VRAM, GPU-hours per task. `--profile` is NOT a valid flag here — the runner forwards it to the children by itself. To turn it off: `--no-profile`. |
| `--seed 42` | reproducibility, including the quantum parts. |

`--dataset`: use `CC+BAO+Pantheon` if you only have
`pantheon_full_parameters.txt`. For `CC+BAO+Pantheon+` (full covariance) you
also need `Pantheon+SH0ES.dat` and `Pantheon+SH0ES_STAT+SYS.cov`, which are
**not** included in the zip.

### Before the real one, a test run

```bash
python cosmo_hpc_runner.py --noise-sweep none,readout --models lcdm \
    --nqpp 3 --steps 500 --qvmc-iter 20 --generations 20 \
    --population-size 40 --n-bits 4 --max-task-gb 12 --outdir /tmp/smoke_test
```

Five minutes. It validates the whole flow, including the new figures.

## 3. Useful variants

**QMCMC is free under noise** — its engine uses 2–4 qubits and `nqpp` does
not touch its circuits. If you want the noisy QMCMC at high resolution:

```bash
python cosmo_hpc_runner.py --models lcdm --noise-sweep none,readout,full \
    --nqpp 9 --only-samplers --qvmc-iter 0 --steps 40000 --max-task-gb 12
```

**Continuous degradation curve** — turns four rungs into a curve. Cheap (2–4
qubits):

```bash
for p in 0.005 0.01 0.02 0.03 0.05 0.08 0.12; do
  python cosmo_hpc_runner.py --models lcdm --noise readout --noise-readout-p $p \
      --only-samplers --nqpp 4 --steps 40000 --max-task-gb 12 \
      --outdir results_ro_$p
done
```

**Genetic algorithm only, for the defense material:**

```bash
python cosmo_hpc_runner.py --only-genetic \
    --noise-sweep none,readout,full,FakeBrisbane \
    --nbits-sweep 4 7 --generations 300 --population-size 400 --max-task-gb 12
```

**The QPU pipeline against simulated noise** (touches neither hardware nor
credentials):

```bash
python qpu_noisy_simulation.py --model lcdm --method both --noise FakeBrisbane \
    --steps 2000 --iters 200 --nqpp 3
```

## 4. Qubit ceilings: what to expect

### Without noise, the ceiling depends on the DATASET

This was not in the first version and cost three dead tasks. The memory cost
of a samplers task is the grid **times the number of data points**: arrays of
shape `(2^q, N_data)`. Measured:

| `total_q` | CC+BAO (51 pts) | CC+BAO+Pantheon (1099 pts) |
|---|---|---|
| 16 | 1.0 GB | 4.8 GB |
| 18 | 4.0 GB | **19.6 GB** |
| 20 | — | ~88 GB |
| 21 | — | ~175 GB |

A state costs 14 kB with CC+BAO and 74 kB with CC+BAO+Pantheon. The same node
grants ~2 fewer qubits when you switch dataset. The runner already accounts
for this (`DATASET_N_DATA`) and prints it at startup; a dataset that is not in
the table uses the largest one, which is the safe direction.

> **This was a bug.** The first version used a constant that did not depend
> on `N_data` and estimated 3.7 GB where 19.6 were measured. In the
> CC+BAO+Pantheon campaign of 2026-08-31 the planner accepted `cpl/nqpp5`
> (20 q) and `gede,wcdm/nqpp7` (21 q); all three died with `rc=-9` (SIGKILL
> from the OOM killer, no traceback and no partial results) after finishing
> MCMC and QMCMC, right as QVMC started. Fixed, with a regression test against
> the 7 real measurements.

### With noise, the parameter-shift batch dominates

With **63 Gi**, the noise axis grants:

| pipeline | ceiling | why |
|---|---|---|
| QMCMC | no `nqpp` limit | its engine uses `max(2, d)` qubits |
| QVMC rungs 0% / 33% | 15 q | a single ρ |
| QVMC rungs 67% / 100% | **11–12 q** | the parameter-shift batch dominates (`2·n_φ` density matrices: 10.4 GiB at 11 q, **44 GiB at 12**) |
| QGA | 15 q | one ρ per operator, no batch |

Per model, with noise and `--max-task-gb 12`: ΛCDM/PEDE reach nqpp=5,
wCDM/GEDE nqpp=3, CPL nqpp=2.

Since `--benchmark` walks the whole ladder, the batch rung sets the ceiling
of a samplers task. More RAM helps little: the batch grows as `4^n`.

### Choose `--max-task-gb` from the table, not from memory

With `--max-task-gb 12` and CC+BAO+Pantheon the ceiling comes out at **17
qubits**, so `gede/wcdm` at nqpp=6 (18 q, 19.6 GB real) does **not** fit: it
used to fit by accident, because the estimate was 5× too low. If you want
those tasks, ask for `--max-task-gb 24` and lower the parallelism
accordingly.

## 5. What to look at when it finishes

| File | What it answers |
|---|---|
| `noise_comparison[_nqppN]_<model>.png` | **start here.** With `--nqpp-sweep` there is **one per resolution**: mixing them would turn a ceiling clamp into what looks like a noise effect. Each rung along the noise axis. Parameters in units of σ (grey band = ±1σ), fit quality on the right. |
| `genetic_evolution_<model>.gif` | the animation of the population converging, all rungs together. For the defense. |
| `genetic_convergence_<model>.png` | trajectory + final estimate of the genetic algorithm. Replaces the corner plot. |
| `corner_ladder_*` | posteriors with shaded 1σ/2σ/3σ bands. |
| `convergence_<model>.png` | effect of `nqpp`. |
| `results_all_models.csv` | everything, with a `noise` column to pivot on (legacy campaigns: `resultados_TODOS_los_modelos.csv`). |

Three warnings when reading:

**Do not compare against `none` without using `none-counts`.** The ideal rung
reads amplitudes and the noisy ones read counts: comparing the two measures
the noise *and* the change of operator at the same time. In the first
campaign this made readout noise look like it **improves** the sampling.

**ESS is not a quality metric on this axis.** It rises with noise while KL
gets worse: noise flattens the distribution and a flat distribution yields
less correlated samples. Report it together with KL or σ, never alone.

**The corner-plot bands are 1σ/2σ/3σ**, and in a two-parameter panel they
contain 39.3 / 86.5 / 98.9 %, not 68 / 95 / 99.7. Those are the values for a
one-dimensional Gaussian.

---

## 6. Redraw a figure without repeating the campaign

Every genetic task now leaves a `ga_state.npz` next to its figures, with the
final population, its weights and the per-generation history. With that,
everything is redrawn in seconds:

```bash
python cosmo_genetic_optimizers.py --replot results/hpc_*/genetic_lcdm_*/model_lcdm/ga_state.npz
```

Before, this was not possible: the population only lived in memory, so
**changing a color or fixing an axis forced repeating the entire run** — days
of compute for a cosmetic change. This was discovered while fixing
`[B-GAPLOT]`. Add `--anim none` to skip the GIF, `--outdir` to write
somewhere else, and `--no-state` on the run if you prefer not to spend the
~1–3 MB per task (at the price of going back to the old problem).
