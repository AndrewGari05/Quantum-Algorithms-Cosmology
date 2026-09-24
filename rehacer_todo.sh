#!/usr/bin/env bash
# rehacer_todo.sh — rehace TODAS las figuras y TODOS los analisis de una
# campana ya corrida, sin repetir un solo segundo de computo.
#
#   bash rehacer_todo.sh RUTA/A/hpc_20260907_130425 [mas campanas...]
#
# Lo que hace, en orden:
#   1. triage      — ¿la campana esta sana?
#   2. --plot-only — rehace las figuras de convergencia de la campana
#                    (las que salian cortadas: [B-REFCLIP], [B-NBITS])
#   3. --replot-ladder — rehace las figuras de escalera de CADA tarea
#   4. graficas_ruido      — el eje de ruido
#   5. comparar_algoritmos — algoritmos, geneticos y modelos
#   6. comparar_semillas   — solo si hay mas de una semilla
#
# Ninguno de estos pasos recalcula nada cientifico. Todos leen los CSV y
# los logs que la campana ya escribio.

set -u
PY="${PYTHON:-python3}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ "$#" -lt 1 ]; then
    sed -n '2,20p' "$0"
    exit 2
fi

CAMPANAS=("$@")
SALIDA="${SALIDA:-figuras_$(date +%Y%m%d)}"

echo "================================================================"
echo "  Campañas: ${CAMPANAS[*]}"
echo "  Figuras nuevas en: $SALIDA/"
echo "================================================================"

# ── 1. ¿está sana? ──────────────────────────────────────────────────
echo
echo ">>> 1/6  TRIAGE"
"$PY" "$REPO/triage_campana.py" "${CAMPANAS[@]}"

# ── 2. figuras de convergencia de la campaña ────────────────────────
#    Son las de nqpp / n_bits. Aquí se arregla que salieran cortadas y
#    que el genético heredara el eje de los samplers.
echo
echo ">>> 2/6  FIGURAS DE CONVERGENCIA (--plot-only)"
for C in "${CAMPANAS[@]}"; do
    echo "    $C"
    "$PY" "$REPO/cosmo_hpc_runner.py" --plot-only "$C"
done

# ── 3. figuras de escalera, tarea por tarea ─────────────────────────
#    Los corner y los 1-a-1 NO se pueden rehacer: necesitan las cadenas,
#    que el CSV no guarda. Estas dos sí.
echo
echo ">>> 3/6  FIGURAS DE ESCALERA (--replot-ladder)"
for C in "${CAMPANAS[@]}"; do
    "$PY" "$REPO/cosmo_modular_quantum.py" --replot-ladder \
        "$C"/samplers_*/model_*/resultados_config.csv
done

# ── 4. el eje de ruido ──────────────────────────────────────────────
echo
echo ">>> 4/6  EJE DE RUIDO"
"$PY" "$REPO/graficas_ruido.py" "${CAMPANAS[@]}" --salida "$SALIDA/ruido"

# ── 5. comparaciones ────────────────────────────────────────────────
#    Algoritmos (ESS, sigma), genéticos (chi2_grid) y modelos (AIC/BIC).
echo
echo ">>> 5/6  COMPARACIONES"
"$PY" "$REPO/comparar_algoritmos.py" "${CAMPANAS[@]}" \
      --salida "$SALIDA/comparacion" | tee "$SALIDA-comparacion.txt"

# ── 6. semillas ─────────────────────────────────────────────────────
#    Avisa solo si hay una sola semilla; no es un error.
echo
echo ">>> 6/6  SEMILLAS"
"$PY" "$REPO/comparar_semillas.py" "${CAMPANAS[@]}"

echo
echo "================================================================"
echo "  Listo. Figuras en $SALIDA/ y dentro de cada carpeta de tarea."
echo "  El texto de las comparaciones quedó en $SALIDA-comparacion.txt"
echo "================================================================"
