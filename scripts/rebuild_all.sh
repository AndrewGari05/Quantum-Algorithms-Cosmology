#!/usr/bin/env bash
# scripts/rebuild_all.sh — rebuilds ALL the figures and ALL the analyses of a
# campaign that has already run, without repeating a single second of compute.
#
#   bash scripts/rebuild_all.sh PATH/TO/hpc_20260907_130425 [more campaigns...]
#
# What it does, in order:
#   1. triage      — is the campaign healthy?
#   2. --plot-only — rebuilds the campaign's convergence figures
#                    (the ones that came out clipped: [B-REFCLIP], [B-NBITS])
#   3. --replot-ladder — rebuilds the ladder figures of EACH task
#   4. noise_plots      — the noise axis
#   5. compare_algorithms — algorithms, genetic and models
#   6. compare_seeds   — only if there is more than one seed
#
# None of these steps recomputes anything scientific. They all read the CSVs
# (current results_config.csv or legacy resultados_config.csv) and the logs
# the campaign already wrote. Output folder: $OUT (legacy name: $SALIDA).

set -u
PY="${PYTHON:-python3}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ "$#" -lt 1 ]; then
    sed -n '2,20p' "$0"
    exit 2
fi

CAMPAIGNS=("$@")
OUT="${OUT:-${SALIDA:-figures_$(date +%Y%m%d)}}"

echo "================================================================"
echo "  Campaigns: ${CAMPAIGNS[*]}"
echo "  New figures in: $OUT/"
echo "================================================================"

# ── 1. is it healthy? ───────────────────────────────────────────────
echo
echo ">>> 1/6  TRIAGE"
"$PY" "$REPO/triage_campaign.py" "${CAMPAIGNS[@]}"

# ── 2. the campaign's convergence figures ───────────────────────────
#    These are the nqpp / n_bits ones. This fixes them coming out clipped
#    and the genetic family inheriting the samplers' axis.
echo
echo ">>> 2/6  CONVERGENCE FIGURES (--plot-only)"
for C in "${CAMPAIGNS[@]}"; do
    echo "    $C"
    "$PY" "$REPO/cosmo_hpc_runner.py" --plot-only "$C"
done

# ── 3. ladder figures, task by task ─────────────────────────────────
#    The corner and 1-to-1 figures CANNOT be rebuilt: they need the chains,
#    which the CSV does not store. These two can.
#    Both the current and the legacy per-model CSV names are accepted.
echo
echo ">>> 3/6  LADDER FIGURES (--replot-ladder)"
shopt -s nullglob
for C in "${CAMPAIGNS[@]}"; do
    CSVS=("$C"/samplers_*/model_*/results_config.csv
          "$C"/samplers_*/model_*/resultados_config.csv)
    if [ "${#CSVS[@]}" -eq 0 ]; then
        echo "    $C: no samplers_*/model_*/results_config.csv found"
        continue
    fi
    "$PY" "$REPO/cosmo_modular_quantum.py" --replot-ladder "${CSVS[@]}"
done
shopt -u nullglob

# ── 4. the noise axis ───────────────────────────────────────────────
echo
echo ">>> 4/6  NOISE AXIS"
"$PY" "$REPO/noise_plots.py" "${CAMPAIGNS[@]}" --out "$OUT/noise"

# ── 5. comparisons ──────────────────────────────────────────────────
#    Algorithms (ESS, sigma), genetic (chi2_grid) and models (AIC/BIC).
echo
echo ">>> 5/6  COMPARISONS"
"$PY" "$REPO/compare_algorithms.py" "${CAMPAIGNS[@]}" \
      --out "$OUT/comparison" | tee "$OUT-comparison.txt"

# ── 6. seeds ────────────────────────────────────────────────────────
#    Only warns if there is a single seed; it is not an error.
echo
echo ">>> 6/6  SEEDS"
"$PY" "$REPO/compare_seeds.py" "${CAMPAIGNS[@]}"

echo
echo "================================================================"
echo "  Done. Figures in $OUT/ and inside each task folder."
echo "  The text of the comparisons is in $OUT-comparison.txt"
echo "================================================================"
