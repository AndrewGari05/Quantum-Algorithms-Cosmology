# Reproducing the thesis

| | where | role |
|---|---|---|
| legacy baseline | tag `v0.8.1-thesis` | the code of the original campaign |
| corrections | branch `thesis-errata` | one commit per fix, see ERRATA.md |
| **defense numbers** | tag `v0.8.2-thesis-defense` | the only source of the numbers in the thesis (created when the corrected numbers are approved) |
| library | default branch (`qablate` + `thesis/`) | re-analysis with the corrected components |

## Re-running only what a fix affects (thesis code)

The exact commands, with the classical rung included in every re-run so the
corrected comparison is self-contained, are in ERRATA.md section 4.1.
Quantities that need no re-run (CO-3/4, GA-1, QM-3) are recomputed from the
stored CSVs with `python -m thesis legacy {refit,grid-floor,mc-error}`.

## Running the ladders with the library

```bash
python -m thesis run mcmc --model cpl --dataset CC+BAO+Pantheon --steps 20000 --chains 8 --out out/cpl
python -m thesis campaign --campaign out/c1 --models lcdm cpl --noise none readout FakeBrisbane
python -m thesis analyze out/c1
```

Each task writes `results.csv` with explicit key columns (campaign, model,
dataset, prior, seed, grid, noise and its parameters, code version) and
convergence diagnostics; analysis never parses folder names.
