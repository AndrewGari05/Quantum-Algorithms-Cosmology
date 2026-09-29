# Reproducing the thesis

Two code lines exist:

| | where | produces |
|---|---|---|
| thesis code | tag `v0.8.1-thesis` (frozen campaign) and branch `thesis-errata` (one commit per fix) | every number in the thesis document |
| library | default branch (`qablate` + `thesis/`) | re-analysis with the corrected components |

Thesis numbers are reproduced only from the thesis tag; ERRATA.md lists every
correction and whether it changed results.

## Re-running only what a fix affects (thesis code)

On branch `thesis-errata` the runner can run a subset of rungs; each rung
re-seeds, so its rows equal those of the full ladder:

```bash
python cosmo_hpc_runner.py --only-samplers --models lcdm pede wcdm gede cpl \
    --dataset CC+BAO+Pantheon --nqpp 3 --noise-sweep none,readout,full,FakeBrisbane \
    --rungs QMCMC50 QMCMC100 --steps 20000 --chains 8 --seed 42 \
    --outdir results/rerun_qm1
```

The QMCMC circuits use max(2, d) qubits whatever `nqpp` is, so one run per
model and noise level replaces the QMCMC rows of every `nqpp` cell.

Quantities that need no re-run are recomputed from the stored CSVs:

```bash
python errata_tools.py refit  results/hpc_20260907_130425       # CO-3/4
python errata_tools.py grid-floor --model lcdm --dataset CC+BAO+Pantheon --n-bits 4 5 6   # GA-1
python errata_tools.py mc-error results/hpc_20260907_130425     # QM-3
```

## Running the ladders with the library

```bash
python -m thesis run mcmc --model cpl --dataset CC+BAO+Pantheon --steps 20000 --chains 8 --out out/cpl
python -m thesis campaign --campaign out/c1 --models lcdm cpl --noise none readout FakeBrisbane
python -m thesis analyze out/c1
```

Each task writes `results.csv` with explicit key columns (campaign, model,
dataset, prior, seed, grid, noise and its parameters, code version) and
convergence diagnostics; analysis never parses folder names.
