# Correcciones del 20–21 de septiembre de 2026

Nueve marcadores. **Sólo uno toca números científicos** (`[B-EPS15]`); los
otros ocho son dibujo, línea de comandos o análisis nuevo, y son seguros para
una campaña en curso.

Cada marcador está en el código con un comentario que explica el porqué, no
sólo el qué. Para encontrarlos: `grep -rn "B-<NOMBRE>" *.py`.

---

## Los que cambian números

### `[B-EPS15]` — `cosmo_modular_quantum.py`

`quantum_amplitude_normalization` devolvía `P_unnorm / (norm + 1e-15)`. Ese
epsilon introducía una diferencia de 0 a 2 ULP que rompía la celda **FIEL**:
el peldaño QVMC 67 % dejaba de coincidir exactamente con el 100 %. La
diferencia se amplifica hasta ~1e-5 en los parámetros.

Ahora es `P_unnorm / norm`. No hace falta el guardia porque `build_target` ya
levanta `[B-EMPTY]` si `total <= 0`.

> **Cambia los números del QVMC 100 % en el quinto decimal.** Una campaña
> corrida antes de esta corrección no es comparable celda a celda con una
> corrida después.

---

## Los que arreglan el dibujo

### `[B-KLLOGX]` — `cosmo_modular_quantum.py`
Eje x logarítmico en `ladder_kl_qvmc_*`.

Los peldaños **no gastan el mismo número de iteraciones**: el presupuesto se
iguala por *circuitos*, no por iteraciones, y el optimizador clásico de scipy
cuenta evaluaciones de función. A `nqpp=3` el peldaño de 33 % registra ~1.2e5
contra 1.5e4 de los cuánticos; a `nqpp=9` la diferencia llega a un factor de
**60**. Con eje lineal el peldaño más largo se comía toda la escala y los
otros tres quedaban aplastados contra el margen izquierdo, que es justo donde
ocurre la bajada que interesa.

No se recorta ni un punto. **Esa diferencia de presupuesto hay que explicarla
en el paper, no esconderla.**

### `[B-NBITS2]` — `cosmo_hpc_runner.py`
La figura `cost_<modelo>` no aplicaba el filtro de familia que `convergence_*`
sí aplicaba desde `[B-NBITS]`. Resultado: las curvas del genético —cuyo eje x
es `n_bits`— se dibujaban sobre un eje etiquetado `nqpp` y con los ticks
puestos en los valores de la otra familia, así que sus puntos caían a la
derecha del último tick, sin etiqueta.

Prueba de regresión en `tests/test_figura_costo.py`; verificada fallando sin
el arreglo.

### `[B-PDF]` — `graficas_ruido.py`, `comparar_algoritmos.py`, `cosmo_hpc_runner.py`
Todas las figuras salen ahora en PNG **y** PDF. El PDF es vectorial y es el
que pide el paper. Los dos se guardan del *mismo* objeto de figura, así que no
pueden discrepar.

---

## Los que arreglan la procedencia y la robustez

### `[B-PROV-GEN]` — `cosmo_genetic_optimizers.py`, `cosmo_modular_quantum.py`
116 filas del barrido genético se escribían con `noise='none'` aunque la tarea
fuera ruidosa: `_ga_side` nunca recibía la semilla ni el nivel de ruido. El
dato científico siempre fue válido; lo que estaba mal era la etiqueta, y se
recupera leyendo el nombre de la carpeta. `graficas_ruido.py` avisa cuando la
columna y la carpeta no coinciden, y usa la carpeta.

### `[B-RZZ]` — `cosmo_genetic_optimizers.py`
`Operator(RZZGate(Parameter))` lanzaba `TypeError` por parámetro sin ligar y
mataba una tarea completa. El camino primario queda byte a byte igual; sólo
si aparece ese error se reintenta transpilando sin las puertas paramétricas
`rzz`/`rxx`/`ryy`.

### `[B-REPLOTGLOB]` — `cosmo_modular_quantum.py`
La ayuda de `--replot-ladder` prometía comodines del shell pero argparse
recibía la expansión como varios posicionales y abortaba con *unrecognized
arguments*, rechazando todos menos el primero. Ahora `nargs='+'` y se expande
cada patrón quitando repetidos.

### `[B-SOLODIAG]` — `cosmo_modular_quantum.py`
`replot_ladder_from_csv(..., solo_diagnosticos=True)` rehace únicamente R̂ y
KL, saltándose las dos figuras caras. ~2 s por tarea en vez de ~7.

---

## Análisis nuevo

### `[B-SIGMANQPP]` — `comparar_algoritmos.py`
Sección y figura `sigma_contra_rejilla`: sigue la razón
σ(QVMC)/σ(Classical MCMC) a lo largo de **todo** el barrido de `nqpp`.

Contesta la objeción obvia al resultado de `incertidumbre_reportada` —*"tu σ
sale angosta porque tu rejilla es gruesa"*—. Con el barrido hasta `nqpp=9`:
la razón **no sube hacia 1, se aplana en 0.66–0.68** desde `nqpp=5`. El sesgo
es estructural del objetivo variacional, no una limitación de resolución.

> Ojo al leerlo: el `final_KL` **no** es comparable a lo largo de `nqpp`,
> porque la distribución objetivo está definida sobre la rejilla y la rejilla
> cambia. La razón de σ sí lo es.

---

## Cómo aplicar todo esto a una campaña ya corrida

Ninguna de las ocho correcciones de dibujo/CLI necesita recalcular física:

```bash
python cosmo_hpc_runner.py --plot-only CAMPANA          # convergence_*, cost_*
python cosmo_modular_quantum.py --replot-ladder \
       CAMPANA/samplers_*/model_*/resultados_config.csv  # ladder_*
python graficas_ruido.py CAMPANA --salida SALIDA
python comparar_algoritmos.py CAMPANA --salida SALIDA
bash rehacer_todo.sh CAMPANA                            # todo lo anterior
```

**Dos trampas que costaron tiempo:**

1. `replot_ladder_from_csv` necesita **el log de la tarea**, no sólo el CSV:
   el R̂ por paso y el KL por iteración no son columnas. Sin log, esas dos
   figuras se omiten **en silencio** y el título de las otras sale con
   `steps=? | iters=?`. Ese signo de interrogación es la señal.
2. `--plot-only` hace **dos pasadas** (samplers y genético). Si el proceso se
   corta a la mitad, faltan las cuatro figuras `_genetico` por campaña sin
   ningún error visible.

---

## Archivos nuevos en esta tanda

| archivo | qué es |
|---|---|
| `comparar_algoritmos.py` | ESS, σ, genético y selección de modelo |
| `comparar_semillas.py` | qué sobrevive al cambio de semilla |
| `graficas_ruido.py` | el eje de ruido |
| `triage_campana.py` | ¿la campaña está sana? |
| `rehacer_todo.sh` | rehace todas las figuras sin recalcular |
| `tests/test_figura_costo.py` | regresión de `[B-NBITS2]` y `[B-PDF]` |
| `tests/test_ruido_genetico.py` | regresión de `[B-PROV-GEN]` y `[B-RZZ]` |
| `tests/test_semillas.py`, `tests/test_triage.py` | los dos scripts nuevos |
| `docs/circuitos/` | los siete circuitos, 29 pp. |
| `docs/guia_figuras/` | qué contesta cada figura, 19 pp. |
| `docs/guia_codigo/` | mapa del código, 5 pp. |

**180 pruebas pasan.**
