# Lectura de las campañas de agosto–septiembre 2026

Análisis de los resultados descargados de la HPC. **Las fechas de los nombres
de carpeta están mal** (el reloj del nodo lo está); el orden real se
reconstruye del contenido, no del nombre.

| carpeta | dataset | qué contiene | estado |
|---|---|---|---|
| `hpc_20260827_175951` | CC+BAO | samplers + genético, sin eje de ruido | terminada |
| `hpc_20260831_173647` | CC+BAO+Pantheon | samplers + genético, sin eje de ruido | terminada, 3 tareas muertas |
| `hpc_20260903_161310` | CC+BAO+Pantheon | genético con eje de ruido | **corriendo** |

---

## 1. Las carpetas "vacías" no están vacías: son OOMKills

Ninguna carpeta está realmente vacía. Tres tienen el log completo pero les
falta el CSV y las figuras, que es exactamente la firma de un `SIGKILL`:

| tarea | qubits | est. del planificador | RSS medido antes de morir | `returncode` |
|---|---|---|---|---|
| `samplers_cpl_nqpp5` | 20 | 14.2 GB | 58.2 GB | **-9** |
| `samplers_gede_nqpp7` | 21 | 28.1 GB | 55.0 GB | **-9** |
| `samplers_wcdm_nqpp7` | 21 | 28.1 GB | 27.8 GB | **-9** |

`returncode = -9` es `SIGKILL`: el OOM killer del kernel. No hay traceback ni
resultados parciales porque el proceso no llega a ejecutar nada al morir.

Las tres murieron **en el mismo punto exacto**: la última línea del log de las
tres es `Adaptive QVMC grid window (shared): ...`, o sea justo después de
terminar el MCMC y el QMCMC y justo al arrancar el QVMC. Por eso el MCMC y el
QMCMC de esas configuraciones **sí** están en el log aunque no estén en el CSV.

**No fue el ruido** (esa campaña no lleva eje de ruido) ni "demasiado nqpp" en
abstracto: fue el planificador. Ver §4.

En `hpc_20260903_161310` no hay ninguna tarea muerta; las carpetas con 5
archivos en vez de 7 (`lcdm/pede` × `nb5,nb6,nb7` × `readout`) son las que
**seguían corriendo** cuando descargaste. Les falta solo lo que el hijo
escribe al final. En `genetic_lcdm_nb6_noise-readout` el QGA al 100 % iba en
la generación 10 de 500 después de 4 horas: ~24 min/generación, o sea unos
8 días para esa sola celda. Es el coste de la matriz de densidad a 12 qubits,
no un cuelgue.

---

## 2. El ajuste es bueno

`χ²` crudo contra `n_data`, idéntico en todos los rungs porque es el `χ²` del
mismo best-fit:

| modelo | params | χ² | n_data | χ²_ν | AIC | BIC |
|---|---|---|---|---|---|---|
| CPL | Ωm, H₀, w₀, wₐ | 1058.64 | 1099 | **0.967** | 1066.6 | 1086.6 |
| GEDE | Ωm, H₀, Δ | 1063.87 | 1099 | **0.971** | 1069.9 | 1084.9 |
| wCDM | Ωm, H₀, w | 1063.83 | 1099 | **0.971** | 1069.8 | 1084.8 |
| ΛCDM | Ωm, H₀ | 1064.61 | 1099 | **0.970** | 1068.6 | 1078.6 |
| PEDE | Ωm, H₀ | 1078.91 | 1099 | **0.984** | 1082.9 | 1092.9 |

χ²_ν ≈ 0.97 con 1099 puntos es un ajuste correcto. (Esto responde la duda de
la animación: los ~1064 son el χ² **crudo**, no el reducido.)

Por **BIC**, ΛCDM gana: los modelos extendidos bajan el χ² entre 0.8 y 5.9,
que no compra los 2 parámetros extra. ΔBIC(ΛCDM→CPL) = +8.0 a favor de ΛCDM.
Por AIC la diferencia es menor pero va en el mismo sentido salvo CPL
(ΔAIC = −2.0, "no concluyente" en la escala de Jeffreys). **Nada aquí exige
física más allá de ΛCDM.**

Con **CC+BAO solo** (`hpc_20260827`) los χ²_ν salen en 0.52–0.66 con 51
puntos. Eso no es un bug: las barras de los cronómetros cósmicos incluyen
sistemáticos y quedan infladas, así que ese dataset restringe de forma
conservadora. Vale la pena decirlo en la tesis en vez de que te lo pregunten.

### Dónde caen los parámetros

ΛCDM con CC+BAO+Pantheon: **Ωm = 0.2766 ± 0.0105**, **H₀ = 69.57 ± 0.80**.

- H₀ queda entre Planck (67.66) y SH0ES (73.04), a ~2.4σ y ~4.3σ
  respectivamente, con σ solo estadístico. Es lo esperado de un ajuste de
  fondo con SNe sin calibrador absoluto.
- Ωm está **3.3σ por debajo** de Planck (0.3111). Esto es conocido para
  Pantheon *stat-only* sin CMB, pero conviene decirlo explícitamente y no
  dejar que la figura lo insinúe sola.

---

## 3. El resultado central: fidelidad, y su precio

### 3.1 Las celdas *faithful* se cumplen exactamente

| comparación | resultado |
|---|---|
| CGA vs QGA(0 %) | **idénticos a 6 decimales**, en todas las configuraciones, con y sin ruido |
| QMCMC 50 % vs QMCMC 100 % | **bit a bit iguales** |
| VI clásico vs QVMC 33 % | KL idéntico; medias iguales a la 5.ª cifra |
| QVMC 67 % vs QVMC 100 % | **bit a bit iguales** |

Esto es el criterio de identidad, y se cumple. Es la afirmación fuerte de la
tesis y está limpia.

### 3.2 El QMCMC reproduce el MCMC clásico con una fidelidad muy alta

Desviación de la media respecto al MCMC clásico, en unidades de σ del propio
MCMC (todos los modelos, todos los nqpp):

| modelo | max\|pull\| | σ_QMCMC / σ_MCMC | ESS |
|---|---|---|---|
| PEDE | 0.003 σ | 0.983 | 5647 |
| wCDM | 0.019 σ | 0.991 | 1917 |
| ΛCDM | 0.025 σ | 0.986 | 5659 |
| GEDE | 0.043 σ | 1.009 | 1884 |
| CPL | 0.100 σ | 0.982 | 381 |

Medias a menos de 0.1 σ y anchos al 1–2 %. Es el mejor resultado de la
campaña y no depende de `nqpp` (el motor del QMCMC usa `max(2, d)` qubits).

### 3.3 El gradiente cuántico **degrada**, y el efecto crece con el circuito

KL final del QVMC contra el posterior de referencia, a `nqpp` fijo:

| modelo | nqpp | VI clásico ≡ QVMC 33 % | QVMC 67 % ≡ 100 % | tiempo 33 % | tiempo 67 % |
|---|---|---|---|---|---|
| ΛCDM | 4 | 0.173 | **0.757** | 20 s | 986 s |
| ΛCDM | 6 | 0.490 | 0.620 | 43 s | 3660 s |
| ΛCDM | 9 | 0.519 | **1.127** | 395 s | 45782 s |
| PEDE | 4 | 0.409 | 0.742 | 33 s | 1421 s |
| PEDE | 9 | 0.619 | **1.506** | 650 s | 56603 s |
| GEDE | 6 | 1.378 | 2.080 | 433 s | 36274 s |
| wCDM | 6 | 1.213 | 1.688 | 1235 s | 54580 s |

En **20 de 24** celdas el rung entrenado con parameter-shift sale **peor** que
el entrenado con gradiente clásico, y la brecha **crece con el número de
qubits** (ΛCDM: 0.757 → 1.127 al ir de nqpp 4 a 9; PEDE: 0.742 → 1.506).

> **CORRECCIÓN.** Una versión anterior de este documento atribuía esa brecha a
> «la varianza del estimador de parameter-shift con número finito de
> disparos». **Eso es falso en la corrida ideal**, por dos razones
> independientes: (1) con `--noise none` el QVMC lee amplitudes, no conteos,
> así que no hay disparos ni varianza de muestreo; (2) la regla de
> parameter-shift está implementada sobre las *probabilidades*
> `Q_i = ⟨ψ|Π_i|ψ⟩` con regla de la cadena hacia el KL, y para puertas RY/RZ
> (autovalores ±1/2) es **exacta**, no una diferencia finita — está verificada
> contra diferencias centrales en los tests. Así que ni ruido de disparo ni
> sesgo del gradiente.
>
> Lo que de verdad separa los rungs es que **son optimizadores distintos**:
> el VI clásico usa COBYLA y el rung 67 % usa SGD con decaimiento de tasa
> sobre el gradiente de parameter-shift. Eso es exactamente lo que la
> taxonomía del proyecto etiqueta como celda ALGORITHMIC, y es correcto que
> difieran.

**Y el SGD está SIN CONVERGER a 5000 iteraciones.** Medido sobre la traza de
KL de los propios logs, comparando el KL en la iteración 4000 contra la 5000:

| celda | VI clásico (COBYLA) | QVMC 67 % (SGD) |
|---|---|---|
| ΛCDM nqpp6 | −0.5 % | −1.3 % |
| ΛCDM nqpp9 | −0.5 % | **−2.9 %** |
| wCDM nqpp6 | −1.3 % | **−3.3 %** |
| GEDE nqpp6 | −0.5 % | **−3.8 %** |

El clásico ya está en su meseta; el cuántico sigue bajando, y **baja más justo
donde la brecha es mayor**. O sea que una parte de la brecha no es una
propiedad del método sino falta de iteraciones: el espacio de ángulos crece
con `nqpp` y el SGD necesita más pasos para recorrerlo. Cuánta parte, no se
sabe sin correrlo — cerrar de 1.127 a 0.519 en ΛCDM nqpp9 exige un 54 % de
reducción y el ritmo actual es ~3 % por cada 1000 iteraciones, decreciendo.

**Consecuencia práctica:** `--qvmc-iter` es una palanca real y bajarlo es un
error. Cualquier afirmación sobre la brecha entre rungs tiene que venir de una
corrida donde el SGD haya llegado a su meseta, o declarar explícitamente que
está sin converger.

El coste es de **30× a 100×** en tiempo. Ejemplo concreto: GEDE a nqpp=6,
433 s con gradiente clásico contra **10 horas** con parameter-shift.

La conclusión que sí se sostiene con estos datos es la de las celdas
*faithful* (§3.1–3.2): el pipeline cuántico **reproduce** el clásico donde el
criterio es identidad, y cuesta dos órdenes de magnitud más. Fidelidad, no
ventaja. La comparación estadística del rung 67 % queda **pendiente** hasta
tener el SGD convergido.

> **Trampa a evitar.** Hay 4 celdas donde el 67 % sale mejor: GEDE nqpp 3 y 4,
> PEDE nqpp 3, ΛCDM nqpp 3. Todas están en `nqpp = 3`, donde el VI clásico
> también va mal, y ninguna sobrevive al aumentar la resolución. Es ruido del
> estimador de KL, no una ventaja. Citarlas sin este contexto sería exactamente
> el tipo de afirmación que este proyecto no hace.

> **El KL NO es comparable entre distintos `nqpp`.** Se mide contra una
> referencia discretizada en la propia rejilla, y la rejilla cambia con `nqpp`.
> Solo son válidas las comparaciones **entre rungs a `nqpp` fijo** — que es
> justo donde vive el resultado de arriba. Que el KL del VI clásico no sea
> monótono en `nqpp` (ΛCDM: 1.80, 0.17, 0.25, 0.49, 0.49, 0.52, 0.52) es esto,
> no una inestabilidad del método.

### 3.4 VI subestima el ancho, y peor cuanto más fina la rejilla

σ_VI / σ_MCMC para ΛCDM al subir `nqpp` de 3 a 9: 0.96, 0.91, 0.80, 0.63,
0.64, 0.62, 0.64. Es la subestimación de varianza propia de VI de campo medio,
que empeora al afinarse la familia variacional. **No es un efecto cuántico** —
el QVMC 33 % la sigue punto por punto. Hay que decirlo al reportar cualquier σ
que salga de VI o QVMC.

### 3.5 El ruido de lectura: el operador aguanta

Genético `lcdm`, comparando `none` contra `readout` en la misma configuración:

| celda | Ωm (sin / con ruido) | dispersión Ωm | dispersión H₀ | tiempo |
|---|---|---|---|---|
| nb4, q=33 % | 0.276301 / 0.276301 | igual | igual | igual |
| nb4, q=67 % | 0.270002 / 0.270002 | +9 % | +21 % | 34 s → 45 s |
| nb6, q=67 % | 0.277478 / 0.277470 | +20 % | +13 % | 39 s → 91 s |
| nb4, q=100 % | 0.270004 / 0.270155 | ×6 | ×2.4 | 199 s → **2274 s** |

La media **no se mueve**; lo que crece es la dispersión de convergencia, entre
10 % y 20 % en los rungs 67 %, y bastante más al 100 %. El coste en tiempo del
ruido al 100 % es de 11× a nb4 y crece muy rápido con `n_bits`.

### 3.6 El sesgo aparente del QGA al 67 %/100 % es **resolución**, no cuántica

| `n_bits` | QGA 67 % Ωm | QGA 67 % H₀ | CGA (referencia) |
|---|---|---|---|
| 4 | 0.270002 | 70.309 | Ωm = 0.276289 |
| 6 | 0.277478 | 69.450 | H₀ = 69.605 |
| 8 | 0.276849 | 69.574 | |

Al subir `n_bits` la estimación **converge al CGA**: a nb8 la diferencia es
0.0006 en Ωm y 0.03 en H₀, o sea **0.06 σ y 0.04 σ** en unidades del MCMC.
Lo que a nb4 parecía un sesgo de los operadores cuánticos era la rejilla de
16 valores por eje. Este era el control que faltaba y los datos ya lo
resuelven: los operadores cuánticos del QGA son fieles, y el error de nb4 es
de discretización.

---

## 4. Dos bugs encontrados al revisar estos resultados

### `[B-MEM]` — el modelo de memoria del planificador (corregido)

Es el que causó los tres OOMKills. El coste por estado de una tarea de
samplers **depende de `N_data`** (arreglos de forma `(2^q, N_data)`), y el
modelo usaba una constante. Medido:

| `total_q` | CC+BAO (51 pts) | CC+BAO+Pantheon (1099 pts) |
|---|---|---|
| 16 | 1.0 GB | 4.8 GB |
| 18 | 4.0 GB | 19.6 GB |

14 kB/estado con 51 puntos, 74 kB/estado con 1099. La constante vieja
(13.3 kB) capturaba solo el término independiente del dataset — el
comentario decía que *eran* los arreglos `(n_states, N_data)`, pero un número
constante no puede serlo. Subestimaba 5.3× a 18 qubits.

Corregido a `bytes/estado = (11500 + 57·N_data) × 1.15`, calibrado contra las
7 mediciones reales de las dos campañas y con margen de seguridad. Reproduce
todas por arriba (factor 1.14–1.42) y nunca por abajo. 5 tests de regresión
nuevos, incluido uno que verifica que las tres tareas muertas ahora se
rechazan en la planificación.

### `[B-GAPLOT]` — la figura del genético contradecía su propia tabla (corregido)

`genetic_convergence_*.png` dibujaba `theta_map` con la desviación **sin
pesos** de la población final; el CSV reporta la media y la desviación
**ponderadas por fitness**. En `lcdm/nb6` eso daba ±0.0165 en la figura contra
±0.0014 en la tabla: doce veces más ancha, sin nada que indicara cuál era
cuál. Unificado en la versión ponderada (la que va al CSV), con el MAP como
marca encima y una advertencia en el pie de que esa barra es **dispersión de
convergencia del optimizador, no un intervalo de credibilidad**.

### `[B-FIDSCALE]` — el fiducial aplastaba la figura (corregido)

En la misma figura, la línea del fiducial de Planck fijaba la escala del eje:
los rungs viven entre 0.2763 y 0.2775 y el fiducial está en 0.3111, así que el
eje se estiraba a 0.275–0.311 y las diferencias entre rungs — el contenido de
la figura — quedaban en un píxel. Ahora los límites los ponen las
trayectorias; si el fiducial cae fuera, se anota en el borde.

---

## 5. Qué falta antes de citar cualquier número

1. **Un SGD convergido.** Lo primero, porque sin eso §3.3 no se puede citar:
   subir `--qvmc-iter` hasta que la bajada del KL en el último 20 % del
   entrenamiento sea comparable a la del VI clásico (~0.5 %). Con 5000 va por
   el 3 %.
2. **Semillas múltiples.** Todo esto es `--seed 42`. La degradación del
   parameter-shift (§3.3) aparece en 20 de 24 celdas y crece de forma
   sistemática con el número de qubits, así que la *tendencia* es sólida; pero
   cualquier *número* concreto de KL necesita al menos 3–5 semillas y una
   barra — y además el SGD convergido del punto anterior.
3. **La campaña con ruido de samplers.** La corrida en curso es solo genética.
   Los rungs QVMC bajo ruido son la celda vacía del eje.
4. **La columna de control `none-counts`.** Sin ella, comparar el peldaño ideal
   (que lee amplitudes) contra los ruidosos (que leen conteos) mide el ruido y
   el cambio de operador a la vez.
5. **Relanzar `cpl/nqpp5` y `gede,wcdm/nqpp7`** con `--max-task-gb` suficiente,
   o aceptar que el techo real de este nodo es 18–19 qubits con
   CC+BAO+Pantheon y decirlo así en la tesis.
