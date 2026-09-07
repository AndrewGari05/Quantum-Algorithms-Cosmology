# Revisión adversarial y correcciones — 2026-09-04

Segunda pasada, esta vez **sistemática** (la TAREA 2 que nunca se había hecho:
todos los bugs anteriores habían aparecido de casualidad, mientras se hacía
otra cosa). Se revisó `cosmo_core.py` a fondo ejecutando código, no leyéndolo.

**Ninguna corrección de aquí cambia un solo número ya publicado.** Se
verificó explícitamente: para puntos físicos, escalar y lote siguen
coincidiendo bit a bit (`max|diff| = 0.0`), y los tests pasan.

---

## Lo que se corrigió

### `[B-BOUNDS]` — ALTO — el "mejor ajuste" podía estar fuera del prior

`fit_statistics` refinaba con Nelder-Mead **sin restricciones** sobre `chi2`,
que nunca ve el prior. Como `chi2` era finito fuera de la caja, el simplex se
salía y se reportaba como mejor ajuste un punto con **probabilidad posterior
cero**.

Medido antes de la corrección, sobre las 15 combinaciones modelo × dataset:

| combinación | qué devolvía |
|---|---|
| `cpl` / CC+BAO+Pantheon | Ωm = 0.178 (el prior empieza en 0.18) |
| `cpl` / Pantheon | **H₀ = 6.3e-5 km/s/Mpc** |
| `wcdm` / Pantheon | H₀ = 43.3 (prior: 60–82) |
| `gede` / Pantheon | H₀ = 39.4 |
| `cpl` / CC+BAO | Ωm = 0.159 |

Lo peligroso: la penalización en χ² era mínima (Δχ² = 0.004 en
`cpl`/CC+BAO+Pantheon), así que **AIC y BIC apenas cambiaban y el número se
veía perfectamente sano**. Exactamente la clase de fallo silencioso que ya
había mordido tres veces.

Ahora usa L-BFGS-B con las cotas del modelo, con Nelder-Mead acotado de
respaldo, y una comprobación final que rechaza el candidato si aun así cayera
fuera del soporte. Verificado: **las 15 combinaciones quedan dentro.**

No afectó a ningún resultado publicado porque ΛCDM nunca se salía.

### `[B-GRIDSEED]` — ALTO — la rejilla cambiaba en cada llamada

`estimate_grid_window` tomaba sus números aleatorios del RNG **global** del
módulo. Consecuencias medidas en `lcdm`/CC+BAO: el ancho en Ωm variaba un
**54 %** entre llamadas idénticas (0.1121 a 0.1721), y además dependía de
cuántos números hubiera consumido antes cualquier otra parte del programa.

Eso rompía justo la promesa que el propio docstring hace: que el simulador
(`cosmo_modular_quantum`) y la QPU (`qpu_cosmo_samplers`) construyen la
**misma** rejilla. Como cada uno llama a la función por su cuenta, cada uno se
quedaba con una discretización distinta — y el KL de ambos se comparaba sobre
rejillas diferentes. **Esa comparación es el resultado central del proyecto, y
es justo la que se va a hacer mañana contra hardware real.**

Ahora la ventana es una función pura de sus argumentos más una semilla
(`GRID_WINDOW_SEED`). Verificado: cinco llamadas idénticas, y sigue siendo
idéntica tras mover el RNG global. `seed=None` recupera el comportamiento
viejo si alguien quiere muestrear la variabilidad a propósito.

### `[B-CLIP]` — ALTO — un modelo no físico devolvía un χ² finito

`CosmoModel.H` hacía `np.clip(e2, 1e-12, None)`: para E² ≤ 0 — una combinación
de parámetros **sin sentido físico** — devolvía H = 10⁻⁶·H₀ en vez de avisar, y
de ahí salía un χ² y un log-posterior finitos. El contrato documentado de la
clase decía lo contrario.

La incoherencia era observable: para el mismo θ, la rama de supernovas **sí**
rechazaba y la de cronómetros no, así que el mismo punto tenía posterior
finito con CC+BAO e −∞ con CC+BAO+Pantheon.

Con los cinco modelos actuales E² > 0 en toda su caja, así que **no ha
corrompido nada**. Pero deja de ser latente en cuanto se añada curvatura
(Ω_k < 0 hace E² negativo a alto z) — que es el siguiente modelo del proyecto.

### `[B-NANMASK]` — BAJO — el lote devolvía `nan` donde el escalar daba `-inf`

`nan <= 0` es `False`, así que un E² = nan se colaba por la máscara del camino
vectorizado. Metropolis rechaza un nan por comparación, así que nunca corrompió
una cadena, pero las dos rutas deben coincidir. Ahora coinciden.

### `[B-PRIORTYPE]` — BAJO — un prior mal escrito caía en plano, sin avisar

`'Gaussian'` con mayúscula, `'planck'`, cualquier typo: todos daban
`log_prior = 0` en silencio. Una corrida etiquetada `gaussian` en el CSV que en
realidad usó prior plano es irrecuperable después. Los CLI ya lo acotaban con
`choices`; las llamadas de biblioteca no. Ahora levanta `ValueError`.

### `[B-SILENT]` — MEDIO — `load_cc` sustituía el archivo sin decirlo

Un `path` inexistente o un archivo con menos de tres columnas devolvían la
tabla empotrada **sin ningún mensaje**. La única señal era la *ausencia* de la
línea "✓ CC+BAO loaded" en un log de miles de líneas. Eso anula el propósito de
`data_manifest.py` y de `data_checksums.json`, y sustituiría en silencio
cualquier compilación alternativa que un revisor pidiera probar. Ahora avisa
en las tres rutas.

### Comentario `[P2]` — la nota sobre la radiación decía lo contrario de la verdad

Afirmaba que `OMEGA_R0 = 9.4e-5` era "solo fotones" y que para alto z había que
multiplicarlo por ~1.68 para añadir los neutrinos. Es al revés: el valor **ya
los incluye** (fotones solos serían 5.40e-5; fotones + 3.046 neutrinos,
9.14e-5). Seguir la instrucción del comentario habría metido un error del
73 %. El número siempre estuvo bien; el comentario, no.

### `csv_path` — un nombre indefinido en código muerto

`append_results_csv` tenía un `return csv_path` después de un `return run_csv`.
Inalcanzable hoy, `NameError` en cuanto alguien reordenara la función.
Eliminado.

---

## Lo que se añadió

### `[B-NOSTATE]` — el estado del genético ahora sobrevive al proceso

Las figuras del genético se dibujan desde la población final y la historia por
generación, y **nada de eso se guardaba en disco**. Al corregir `[B-GAPLOT]`
salió la consecuencia: una figura tenía mal la barra de error, y para
redibujarla bien había que **repetir la campaña entera**. En una corrida donde
una celda ruidosa a 12 qubits tarda dos días, eso convierte "cámbiame el color
de esta curva" en una semana de cómputo.

Cada tarea genética escribe ahora un `ga_state.npz` (~1–3 MB comprimido) y se
redibuja todo con:

```bash
python cosmo_genetic_optimizers.py --replot .../model_lcdm/ga_state.npz
```

Detalles de diseño:

- **Sin `pickle`.** Un `.npz` con los arreglos y un bloque JSON con los
  escalares. `pickle` guardaría el objeto en una línea pero se rompe al cambiar
  de versión de numpy o al renombrar una clase, y estos archivos tienen que
  seguir abriéndose dentro de dos años, cuando toque rehacer una figura para la
  defensa.
- **Se escribe antes de dibujar**, y aunque las figuras estén desactivadas: si
  el proceso muere renderizando, los datos ya están a salvo.
- La población final, su fitness y sus pesos van en float64 porque de ellos
  salen los números del CSV; la historia por generación va en float32 porque
  solo alimenta la animación.
- En `--sweep-all` se guarda **un** estado con todos los rungs juntos, no uno
  por rung (que se pisarían).
- `--no-state` lo desactiva.

Cinco tests nuevos, incluido el que verifica que la media y la desviación
ponderadas — las que van al CSV — vuelven **exactas** tras el viaje por disco.

### `COMO_CORRER_QPU.md`

Procedimiento completo para hardware real, con lo que está verificado offline
(el canal `ibm_quantum_platform` es el vigente, `SamplerV2` es la primitiva
actual, la transpilación a ISA existe y produce circuitos dentro de la base de
un backend real) y, sobre todo, **lo que no está probado** porque el gemelo
simulado lo sustituye: las credenciales, `least_busy`, el ciclo de vida del
`Batch` y el parseo del resultado real. La prueba de humo de 1 job existe justo
para eso.

---

## Estado

- **107 tests en verde** (17 nuevos).
- El plan de la campaña sigue dando **exactamente 100 tareas**, idéntico al
  verificado antes.
- Prueba de humo de punta a punta con el runner: **5/5 tareas OK**, todas las
  figuras generadas, incluidas las de comparación de ruido.
- Redibujado desde un `ga_state.npz` producido por el runner: **funciona**.

## Lo que queda pendiente y no se tocó

- **El `z_t` de GEDE.** Se usa la igualdad materia–Λ de ΛCDM
  (`((1-Om)/Om)^(1/3)`), que para GEDE **no es autoconsistente**: la condición
  propia sería `Ωm(1+z_t)³ = Ω_DE·f_DE(z_t)`, y `f_DE(z_t) ≠ 1` salvo si Δ=0.
  La diferencia llega al **2.6 % en H(z)** en el borde del prior de Δ,
  comparable a las barras de los cronómetros. **No lo cambié a propósito**: no
  puedo comprobar desde aquí si la forma cerrada es la convención del paper de
  Li & Shafieloo, y cambiarla movería resultados ya obtenidos. Compruébalo
  contra la ecuación del paper antes de la defensa.
  Además, el test que el comentario cita como garantía usa Δ=0, donde
  `f_DE ≡ 1` para *cualquier* `z_t`: no garantiza nada. Un modelo con el signo
  invertido pasa los dos tests actuales.
- **La covarianza de BOSS DR12.** Los tres puntos con más peso de la tabla
  CC+BAO (37.6 % del peso total) son un análisis correlacionado que se trata
  como diagonal. Eso **subestima las barras de error** que cita la tesis.
  Decláralo o incorpora el bloque 3×3.
- **`k` en AIC/BIC con datasets solo de supernovas.** El χ² es *exactamente*
  independiente de H₀ (la marginalización analítica de M lo absorbe), pero se
  cuenta H₀ como parámetro libre. No afecta a las campañas (siempre combinan
  datasets), sí afectaría a un ajuste solo-SNe.
- **Los otros cuatro módulos** (`cosmo_modular_quantum`, `cosmo_genetic_optimizers`,
  `cosmo_hpc_runner`, `qpu_cosmo_samplers`) **no** recibieron la misma pasada
  sistemática — se agotó el presupuesto de revisión. `cosmo_core.py`, que es
  donde vive la física, sí.

---

# Segunda tanda: `cosmo_modular_quantum.py`

Revisión adversarial del módulo que contiene el resultado científico central.
**Aquí salió lo más serio de todo el proyecto.**

## `[B-BUDGET]` — CRÍTICO — la comparación clásico vs cuántico no costaba lo mismo

`--qvmc-iter` se pasaba **igual** a las dos ramas del entrenamiento del QVMC.
Pero una iteración no cuesta lo mismo en cada una:

- la rama cuántica gasta **1 + 2·n_φ** evaluaciones de circuito por iteración
  (el KL en φ más las 2·n_φ desplazadas del parameter-shift),
- COBYLA gasta exactamente **1**.

Con el mismo `--qvmc-iter`, la rama cuántica recibía entre **57×** (ΛCDM,
nqpp=2) y **225×** (CPL, nqpp=4) más trabajo. Medido:

| rama | iteraciones | circuitos | KL |
|---|---|---|---|
| parameter-shift | 40 | 2281 | 1.425 |
| COBYLA (presupuesto nominal) | 40 | **41** | 1.783 |
| COBYLA (presupuesto igualado) | 2281 | 2282 | **0.00063** |

A presupuesto igualado, **el optimizador clásico gana por tres órdenes de
magnitud**. Verificado también a nqpp=3, la resolución de la campaña: clásico
0.204 contra cuántico 2.047 con los mismos ~5100 circuitos.

**Por qué esto importa más que ningún otro bug de este proyecto.** El brief
decía: *"la afirmación científica central es fidelidad cuántica, NO ventaja
cuántica... no quiero que ninguna modificación insinúe ventaja donde no la
hay."* Este bug hacía exactamente eso, y no por un comentario mal escrito sino
por la aritmética del experimento. Cualquier celda donde el rung cuántico
saliera mejor era un artefacto de contar mal el trabajo.

Y hay una lectura que **fortalece** tu conclusión: en tu campaña grande
(CC+BAO+Pantheon) el rung del 67 % salía *peor* en 20 de 24 celdas — o sea que
perdía **incluso recibiendo entre 57 y 225 veces más cómputo**. Con el
presupuesto igualado esa conclusión no solo se mantiene, se vuelve mucho más
fuerte.

Corregido con `--budget-mode`:

- `circuits` (**nuevo valor por defecto**) — las dos ramas reciben el mismo
  número de evaluaciones de circuito. Es la comparación honesta.
- `iters` — reproduce el comportamiento viejo. **Es el de todas las campañas
  anteriores al 2026-09-04**; úsalo solo para reproducirlas, nunca para una
  comparación nueva.

Además, cada fila del CSV lleva ahora `budget_mode` y `circuits_train`: el
trabajo realmente gastado, no el nominal, para que la comparación sea
auditable por un tercero.

## `[B-ESSCOMP]` — ALTO — el ESS del QVMC era un artefacto de representación

La rama cuántica devolvía la muestra **comprimida** (una fila por cadena de
bits medida, con su conteo como peso) y la clásica **una fila por disparo**.
Describen la misma muestra, pero el ESS de Kish no es invariante bajo esa
compresión: sobre la forma comprimida mide cuántas celdas de la rejilla se
ocuparon, no cuántas muestras hay.

Medido con 3 cadenas × 2000 disparos, misma `phi_opt` y KL idéntico hasta la
sexta cifra:

```
Classical VI   filas=6000   ESS reportado = 6000.00
QVMC 33%       filas= 185   ESS reportado =  100.74
     ...expandiendo los conteos, las DOS dan 6000.00
```

Esto está en tus resultados publicados: **todas** las filas `Classical VI`
leen `ESS=6000.0` y todas las `QVMC 33 %` leen 95–113, con el mismo KL. Un
lector concluye que el muestreo cuántico cuesta 60× el tamaño efectivo de
muestra — **en una celda etiquetada FAITHFUL, donde por construcción no debe
haber diferencia**. Las medias y las desviaciones sí eran fieles; solo la
columna ESS mentía. Corregido: ahora las dos ramas dan 6000.0.

## `[B-PROV]` — ALTO — el CSV no decía en qué condiciones se obtuvo cada fila

Ningún esquema registraba el nivel de ruido, la ruta de lectura ni la semilla.
Dos filas con la misma clave (`Method`, `model`, `dataset`, `prior`, `nqpp`)
pero números distintos eran indistinguibles: una podía venir de una corrida
ideal y otra de una ruidosa. **Eso hacía el eje de ruido entero
irreconstruible desde los resultados**, y de paso rompía la lectura de la
celda FAITHFUL: `QMCMC 50 % ≡ QMCMC 100 %` se cumple bit a bit en unos grupos
de filas y no en otros, porque los que difieren son ruidosos — pero no había
forma de saberlo.

`cosmo_noise.NoiseSpec.metadata()` ya devolvía justo esto y no se llamaba
desde ningún sitio. Añadidas cinco columnas: `noise`, `proposal_route`,
`seed`, `budget_mode`, `circuits_train`.

## `[B-EMPTY]` — MEDIO — una rejilla sin soporte devolvía un KL excelente

Si la rejilla no toca ningún punto con soporte del prior, `P` sale toda a cero.
La rama clásica daba `nan` con un warning; la cuántica, con su guarda `+1e-15`,
devolvía ceros **sin ningún aviso**. Aguas abajo el objetivo se vuelve uniforme
y el KL degenera en `log n − H(Q)`, que es un número **pequeño y atractivo**:
una corrida completa reportaba Ωm = −4.45, H₀ = −95.1 y **KL = 0.022** — mejor
que cualquier corrida legítima de la campaña — sin un solo error. Y era el rung
del 100 % el que mejor escondía la señal. Ahora levanta.

## `[B-COBYLA-CLAMP]` — BAJO

SciPy sube el `maxiter` de COBYLA hasta `n_vars + 2` y avisa; se hace explícito
para que el presupuesto declarado sea el real.

---

## Lo que se comprobó y está BIEN

Esto también es resultado, y del bueno:

- **El gradiente de parameter-shift es EXACTO.** Verificado contra diferencias
  centrales sobre el circuito transpilado: el residuo escala como O(h²), o sea
  que es error de truncamiento de la diferencia finita, no error del gradiente
  (`h=1e-3 → 1.9e-7`, `h=1e-4 → 1.9e-9`, `h=1e-5 → 7.0e-11`). La cancelación
  del `+1` es real: `max|Σ_i ∂Q_i/∂φ_j| = 2.4e-16`. **El `[H5 FIX]` es
  sólido** y sostiene el resultado principal.
- **Las celdas FAITHFUL se cumplen por la razón correcta.** Instrumentando el
  código: la rama cuántica de `acceptance` se ejecuta de verdad (350 llamadas
  al 100 %, 0 al 50 %) y las cadenas salen `np.array_equal → True`. No es que
  esté apagada; es que es equivalente. Lo mismo con `sampling` (el trabajo de
  Aer sube de 32 a 34) y con `normalization` (el circuito QAE se ejecuta y su
  resultado se descarta por diseño, y está declarado en el docstring).
- **El control `none-counts` aísla exactamente lo que debe**: `dAcc = 0`,
  `dKL = 0`, y solo cambia la propuesta.
- **El ruido llega de verdad**: `readout` y `full` difieren del ideal y van por
  `density_matrix`; la guarda `[N1]` levanta; `ΣQ = 1.0000000` en todos los
  rungs.
- **Reproducibilidad**: dos corridas con la misma semilla difieren solo en el
  reloj de pared.

## Una observación que conviene declarar en la tesis

Bajo ruido de lectura, la amplitud de aceptación tiene un suelo en `p`, así que
una propuesta imposible (Δ = −30) se acepta el 3 % de las veces y **la tasa de
aceptación SUBE** (0.487 → 0.545). Es una consecuencia correcta del modelo de
ruido, pero significa que el QMCMC ruidoso **ya no muestrea el posterior**.
Dilo explícitamente, para que una tasa de aceptación que sube no se lea como
mejor mezcla — es la misma trampa que el ESS que sube con el ruido mientras el
KL empeora.

---

## Lo que sigue sin revisar

`cosmo_genetic_optimizers.py`, `cosmo_hpc_runner.py` y `qpu_cosmo_samplers.py`
**no** han recibido la pasada sistemática. Dado lo que salió en los dos
módulos que sí la recibieron, no supongas que están limpios.


---

# Tercera tanda: genético, runner y docstrings

## `[B-REFINE]` — ALTO — el χ² del genético mide a scipy, no al genético

Esto explica algo que llevábamos toda la conversación viendo sin entender: por
qué en tus campañas el χ² del genético sale **byte a byte idéntico** en CGA y
en los cuatro rungs del QGA (1064.6099 en todas las filas). Yo lo había leído
como "es el χ² del mismo mejor ajuste". La razón real es otra.

El genético devuelve un punto de la **rejilla**. Después se refina con
`fit_statistics(refine=True)`, que es un optimizador **continuo**. Y ese
refinamiento borra la diferencia entre peldaños. Medido con lcdm/CC+BAO,
n_bits=4:

| rung | mejor de la rejilla | χ² rejilla | χ² tras refinar |
|---|---|---|---|
| CGA | [0.26496, 70.3148] | 27.6502 | 27.469109 |
| QGA 0 % | [0.26496, 70.3148] | 27.6502 | 27.469109 |
| QGA 33 % | [0.25769, 70.6918] | 27.4717 | 27.469109 |
| QGA 67 % | [0.27000, 70.3125] | 28.3992 | 27.469109 |
| QGA 100 % | [0.27000, 70.3125] | 28.3992 | 27.469109 |

Los cuatro aterrizan en celdas **distintas**, con un rango de **0.93 en χ²**
entre el mejor y el peor. Tras refinar, los cuatro dan el mismo número hasta la
sexta cifra.

Refinar no es un error: el χ² refinado es el que hay que comparar contra los
samplers, y por eso se conserva. **El error sería leer esas columnas como si
midieran al genético.** No las leas así: χ², χ²_red, AIC y BIC del genético no
pueden distinguir un rung de otro, por construcción.

Corregido añadiendo **`chi2_grid`** al CSV y a `ga_state.npz`: el χ² del mejor
punto de la rejilla, sin refinar. Es el único número del genético que sí
distingue los peldaños. El log ahora imprime las dos líneas, y dice cuánto
movió el refinamiento.

Y con esto la celda FAITHFUL queda mejor probada que antes: `CGA ≡ QGA(0 %)`
ahora se verifica **en la rejilla**, antes del refinamiento. Si solo
coincidieran después, la igualdad no diría nada de los operadores — la
garantizaría el refinador.

## `[B-BUDGETMISMATCH]` — MEDIO — dos flags que se contradicen en silencio

`--max-task-gb` mayor que `--mem-budget-gb` es una contradicción. El techo de
qubits sale del primero, pero el pool admite **siempre** al menos una tarea
aunque no quepa en el presupuesto agregado (si no, una tarea grande bloquearía
la campaña para siempre). Con esa combinación puedes lanzar una tarea que
excede el presupuesto entero, y la única señal sería un OOMKill horas después.

Medido: `--max-task-gb 40 --mem-budget-gb 10` con CC+BAO+Pantheon concedía
**18 qubits, ~22 GB para UNA tarea**, contra un presupuesto de 10 GB. Sin un
solo aviso. Ahora avisa al planificar.

## Lo que se comprobó del runner y está BIEN

- **Detección de cgroup v1 y v2**: correcta, incluidos los centinelas de "sin
  límite" (`max` en v2, `9223372036854771712` en v1).
- **El bloqueo del CSV compartido aguanta concurrencia real**: 8 hilos × 40
  escrituras → 320 filas, **0 cabeceras duplicadas, 0 filas corruptas**.
- **La admisión de tareas respeta el presupuesto** (`admitted_mem() + est_mem
  <= mem_budget`), con la excepción documentada de la primera tarea.
- **Un hijo que muere no mata la campaña**: verificado en tus propias corridas
  (`rc=-9` registrado y el resto siguió).

## Lo que se comprobó del genético y está BIEN

- **`CGA ≡ QGA(0 %)` bit a bit**, ahora también en la rejilla: mismo
  `theta_grid`, mismo `chi2_grid`, misma población final.
- **`QGA(100 %)` sí difiere** — si no, la escalera no mediría nada.
- **Reproducibilidad**: misma semilla dos veces → idéntico; semillas distintas
  → poblaciones distintas.

## Docstrings

Tu petición era convertir los `#` a `'''...'''`. Lo hice **solo donde
corresponde**, y te explico por qué no en todos: un `'''...'''` dentro del
cuerpo de una función no es un docstring, es una cadena suelta que Python
evalúa y descarta — no sale en `help()` ni en el tooltip del editor, y engorda
el módulo. Solo la **primera** cadena de un módulo, clase o función lo es.

Lo hecho:

- **Las cabeceras `#` de los cinco módulos grandes** convertidas a docstring de
  módulo. Eran comentarios, así que `help(cosmo_core)` no mostraba nada; ahora
  muestra las 21–68 líneas de cada cabecera.
- **58 funciones, clases y métodos que no tenían docstring** ahora lo tienen,
  escrito leyendo cada una (con `Args:`, `Returns:` y `Raises:` donde aplica),
  no con plantilla.
- **Cero definiciones sin docstring** en todo el repositorio (360 en total).
- Los comentarios de implementación se quedan como `#`, que es donde sirven.

## Estado

- **111 tests en verde**.
- Plan de campaña idéntico: **100 tareas**.
- Prueba de humo completa del runner: **5/5 tareas OK**, con `chi2_grid`
  poblado y distinguiendo los rungs (27.7394 / 27.4692 / 28.3992 / 28.5493).

## Lo que sigue sin revisar

`qpu_cosmo_samplers.py` solo se revisó para el procedimiento de hardware (API,
transpilación, credenciales), no de forma adversarial completa.

---

## Anexo: docstrings y ejemplos (estado medido)

Lo que había cuando se preguntó, y lo que hay ahora:

| | antes | ahora |
|---|---|---|
| definiciones sin ningún docstring | 58 | **0** |
| módulos con la cabecera en `#` (invisible a `help()`) | 5 | **0** |
| docstrings con sección `Args:` | 81 (24 %) | **193 (57 %)** |
| docstrings con sección `Returns:` | 59 (18 %) | **160 (48 %)** |
| **públicas con argumentos sin documentar** | **112** | **0** |
| docstrings con ejemplo ejecutable | 0 (0 %) | **17 funciones** |

El 43 % que sigue sin `Args:` son funciones sin argumentos o helpers privados
(`_nombre`), donde un resumen de una línea es lo correcto: un bloque `Args:` ahí
sería ruido.

**Los ejemplos son doctests**, no texto decorativo: se ejecutan con la suite
(`test_los_ejemplos_de_los_docstrings_corren`). Un ejemplo que no corre es peor
que ninguno porque envejece en silencio. El primero que escribí ya atrapó un
error mío — afirmaba `H(0) == 70.0` exacto cuando son `69.99999999999999` por
redondeo de punto flotante.

Están puestos donde enseñan algo que costó dinero aprender:

- `estimate_qubits_and_mem` — que 20 qubits con CC+BAO+Pantheon son **88 GB**,
  no los 14 que estimaba el modelo viejo (`[B-MEM]`).
- `genetic_noisy_seconds_per_gen` — que 14 qubits son **7.7 h por generación**
  (`[B-TIME]`).
- `noisy_density_bytes` — que el lote de parameter-shift a 12 qubits son 45 GB
  frente a los 256 MB de una ρ suelta.
- `CosmoModel.H` — que wCDM con w=−1 y CPL con w0=−1, wa=0 reproducen ΛCDM
  **exactamente** (`array_equal`, no `allclose`). Es la prueba de que la física
  está bien, y ahora se ejecuta en cada corrida de los tests.
- `fit_statistics` — que el mejor ajuste cae siempre dentro del prior
  (`[B-BOUNDS]`).
- `ess_weights` — el contraste que `[B-ESSCOMP]` medía mal.
