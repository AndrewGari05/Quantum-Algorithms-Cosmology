# Fase 4 — Eje de ruido NISQ: viabilidad e implementación

**Parte A (viabilidad)** — estudio previo, ningún módulo de producción tocado.
**Parte B (implementación)** — el eje ya está cableado; ver §8 al final.

Suite: **33 → 62 tests**, todos en verde.

Entorno de verificación: qiskit 2.4.2, qiskit-aer 0.17.2, numpy 1.26.4,
qiskit-ibm-runtime 0.49.0. Línea base antes de empezar: **33/33 tests pasan**.

Todo lo de abajo es medido, no leído. Reproducir con:

```
python noise_feasibility_probe.py          # sondas A, B, D
python noise_feasibility_probe.py --cost   # añade C (lenta)
```

---

## 1. La hipótesis del atajo: **procede, con una condición**

`SamplerV2` de `qiskit-ibm-runtime` acepta un backend simulado en `mode=`
(*local testing mode*) y corre sin credenciales ni red. Los tres envoltorios
que usa `QPUConnection` funcionan: `mode=backend`, `mode=Batch(backend=...)`
y `mode=Session(backend=...)`. El decodificado de `run_pub` sobrevive intacto:
con `B=2` bindings, `reg.get_counts(k)` devuelve lo esperado y las cadenas de
bits conservan el ancho del registro clásico (3 bits), **no** los 127 qubits
físicos a los que transpila `FakeBrisbane`.

Es decir: el pipeline por conteos se puede apuntar a ruido simulado sin tocar
el simulador ideal. La hipótesis es correcta.

### La condición — y es un hallazgo, no un detalle

En local testing mode, `qiskit-ibm-runtime` **ignora silenciosamente la
supresión de error**:

```
UserWarning: Options {'dynamical_decoupling': {'enable': True,
  'sequence_type': 'XY4'}, 'twirling': {'enable_gates': True,
  'enable_measure': True}} have no effect in local testing mode.
```

`QPUConnection.__init__` fija esas cuatro opciones y nunca comprueba que
hayan surtido efecto. Consecuencia directa: una corrida ruidosa por esta ruta
mide **ruido sin DD ni twirling**, mientras que la corrida en hardware real
mide **ruido con DD y twirling**. No son el mismo experimento, y hoy nada en
el código ni en los logs lo distingue.

Esto no invalida el atajo — lo convierte en una **cota inferior** de la
calidad del hardware, que es una lectura perfectamente defendible. Pero tiene
que quedar registrado en el CSV como columna, no inferirse.

Clasificación: **ALTO** (no afecta la corrección de los resultados actuales,
que son todos ideales; afecta la interpretación de todo resultado ruidoso
futuro y de la afirmación "el pipeline QPU usa DD XY4 + twirling").

---

## 2. Corrección al mapa del problema: no son tres lecturas, son seis

El documento de diseño lista tres sitios donde el simulador ideal lee
amplitudes. `grep` sobre `cosmo_modular_quantum.py` encuentra **seis**
(líneas 500/505, 584/590, 627/636, 854, 1017/1025, 1200/1201). Cualquier
estimación de esfuerzo basada en "tres sitios" está subestimada.

Pero lo importante no es el conteo, sino que **no son equivalentes entre sí**:

| # | Sitio | Qué lee | ¿Sobrevive a un estado mezclado? |
|---|---|---|---|
| 1 | `_raw_block` (propuesta) | `Re(ψ)·sign(Im(ψ))` | **No.** Es sensible a fase relativa; carece de sentido para ρ mezclado. |
| 2 | `hadamard_accept_log_batch` | `\|ψ₀\|²` | **Sí.** Es exactamente `ρ[0,0]`. |
| 3 | `_kl_batch` | `\|ψ\|²` completo | **Sí.** Es exactamente `diag(ρ)`. |

Dos de los tres bloqueos que el documento da por equivalentes **no lo son**.
Las lecturas 2 y 3 son probabilidades, y las probabilidades sí existen en un
estado mezclado. Solo la 1 se rompe de verdad — y esa ya tiene contraparte por
conteos en `qpu_cosmo_samplers._counts_to_shift` (⟨Z_q⟩ por qubit).

Además, `cosmo_genetic_optimizers.py` **ya trabaja íntegramente por medición**
(`sim.run(..., shots=P, memory=True)` + `get_memory`, líneas 878, 951, 997).
Cero lecturas de amplitud. El QGA no necesita ninguna refactorización: acepta
un `NoiseModel` tal como está.

Y `cosmo_core.make_simulator(method, prefer_gpu, n_qubits, **kwargs)` reenvía
`**kwargs` a `AerSimulator` y es el **único** punto donde nacen todos los
simuladores del proyecto. `noise_model=` entra por ahí sin abrir un segundo
camino.

---

## 3. Matriz de densidad vs disparos: **matriz de densidad**, y el trade-off no es el que parece

Medido con el ansatz real (`build_ansatz(n, n_layers=3)`, 42 parámetros),
`B=2` bindings, 4096 disparos, `NoiseModel.from_backend(FakeBrisbane)` —
la forma exacta de una iteración SPSA del QVMC-QPU:

| qubits | matriz de densidad | trayectorias | ρ teórica | RSS pico medido |
|---|---|---|---|---|
| 6  | **1.66 s** | 5.31 s | 0.1 MB | — |
| 8  | **2.33 s** | 6.77 s | 1.0 MB | — |
| 9  | **2.36 s** | 12.19 s | 4.0 MB | — |
| 10 | **3.77 s** | 22.41 s | 16 MB | — |
| 12 | **34.4 s** | 100.9 s | 256 MB | — |
| 13 | **216.5 s** | 333.3 s | 1.0 GB | 1414 MB |

Engine de propuesta del QMCMC (bloque `B=64`, que es su modo real de uso):

| qubits | matriz de densidad | trayectorias |
|---|---|---|
| 2 | **4.15 s** | 51.70 s |
| 3 | **4.27 s** | 55.85 s |
| 4 | **4.30 s** | 66.43 s |

**La matriz de densidad domina en tiempo en todo el rango donde cabe**, entre
1.5× y 15× más rápida. La razón es estructural: la matriz de densidad evoluciona
UNA vez y luego muestrea los disparos de ρ; las trayectorias re-simulan el
circuito completo una vez por disparo, de modo que su costo es
`shots × 2^n` mientras que el de ρ es `4^n` independiente de los disparos.
Por eso la ventaja crece con el número de disparos y es máxima justo donde el
QMCMC opera (pocos qubits, muchos bindings).

### Dónde esto te corrige

El documento plantea: *"si eliges simulación por disparos en vez de matriz de
densidad, el costo se mueve de memoria a tiempo"*. Es cierto, pero la
conclusión operativa que se sigue no es la esperada: **los disparos no te
compran qubits.** Convertir un OOM en una corrida que no termina no es un
trade-off utilizable. Extrapolando el escalado medido (×2 por qubit para
trayectorias), CPL con nqpp=6 (24 qubits) son ~114 horas **por job**, y un
QVMC de 30 iteraciones son 31 jobs.

El techo de ~14 qubits que calculaste es **correcto**, pero por una razón más
fuerte que la que le atribuyes: no es que la matriz de densidad sea cara y los
disparos baratos; es que **por encima de ~14 qubits ninguna de las dos rutas es
viable**, una por memoria y la otra por tiempo. El eje de ruido es
intrínsecamente un experimento de qubits bajos.

Traducido a tus modelos (`n_qubits = n_params × nqpp`):

| modelo | d | nqpp=3 | nqpp=4 | nqpp=5 | nqpp=6 |
|---|---|---|---|---|---|
| lcdm, pede | 2 | 6 ✅ | 8 ✅ | 10 ✅ | 12 ✅ |
| wcdm, gede | 3 | 9 ✅ | 12 ✅ | 15 ❌ | 18 ❌ |
| cpl | 4 | 12 ✅ | 16 ❌ | 20 ❌ | 24 ❌ |

El QMCMC no aparece en esta tabla porque su engine usa
`n_qubits = max(2, n_params)` — de 2 a 4 qubits, gratis bajo cualquier modelo
de ruido, para todos los modelos y todos los nqpp.

**Recomendación: matriz de densidad, con techo duro de 13 qubits** (1 GB de ρ,
1.4 GB de RSS medido). El tercer modelo de memoria del runner es
`BYTES_PER_STATE_NOISY = 16 · 2^n` extra sobre el modelo de samplers que ya
existe — pero el techo real lo pone el tiempo, no la RAM, y debe ser un
parámetro explícito, no derivado de la RAM detectada como los otros dos.

---

## 4. El resultado que cambia el diseño: ρ es ciego al error de lectura

La aceptación FAITHFUL codifica `A = min(1, e^Δ)` como `cos²(θ/2)` y la lee
como `|ψ₀|²`. Verificado:

- `ρ[0,0]` reproduce `|ψ₀|²` sin ruido con diferencia **0.000e+00** — bit a
  bit. El cambio de lectura no introduce error propio.
- `ρ[0,0]` contra `min(1, e^Δ)` exacto: **1.110e-16**. Tu número de identidad
  se reconfirma de forma independiente.

Curva de degradación medida:

| canal | sesgo medio en A | sesgo relativo medio |
|---|---|---|
| ideal | 0.00000 | 0.00000 |
| lectura p=0.01 | **0.00000** | **0.00000** |
| lectura p=0.03 | **0.00000** | **0.00000** |
| lectura p=0.05 | **0.00000** | **0.00000** |
| compuerta 1e-3 + lectura 0.03 | −0.00014 | 0.02598 |
| compuerta 1e-2 + lectura 0.03 | −0.00139 | 0.25977 |

**El error de lectura no mueve `ρ[0,0]` ni un dígito, para ningún p.** Es
correcto físicamente: la lectura es un canal *clásico posterior* a la medición
y no toca el estado. Pero significa que si el eje de ruido se implementa
leyendo ρ, la columna `--noise readout` de tu matriz de ablación saldría
**idéntica a la columna ideal** — una columna entera falsa, y falsa de la peor
manera, porque no falla ruidosamente sino que da resultados limpios y
plausibles.

Que los conteos sí lo ven está verificado: con `p=0.03` y 10⁵ disparos,
`max|conteos − ρ[0,0]| = 0.0305`, contra un ruido de disparo esperado de
0.0016. La diferencia es exactamente `p`.

### La salida es analítica, no por disparos

Sobre 1 qubit, el error de lectura es un mapa estocástico 2×2 sobre
`(ρ₀₀, ρ₁₁)`, así que se aplica en cerrado:

```
P_ruidosa(0) = (1−p)·ρ[0,0] + p·(1−ρ[0,0])
```

Verificado contra 4×10⁶ disparos en cinco canales: la discrepancia máxima es
0.0005 frente a un ruido de disparo esperado de 0.00025 — es decir, coincide
dentro de ~2σ, que es lo que corresponde al máximo sobre 8 valores.

Esto importa porque la ruta por disparos **no puede** resolver el sesgo que
buscas: con compuertas a 1e-3 el sesgo es 1.4e-4, y distinguirlo del ruido de
muestreo exigiría ~10⁷ disparos **por evaluación de aceptación**, en cada paso
de cada cadena. Inviable.

Con ρ + el mapa analítico de lectura obtienes **el eje de ruido completo, exacto
y sin ruido de disparo, al costo de un circuito de 1 qubit**.

### Y hay física publicable en la forma del sesgo

La despolarización lleva ρ hacia `I/2`, de modo que el sesgo es
`λ·(1/2 − A)`. El sesgo relativo escala **linealmente** con la tasa de error de
compuerta (2.598% a 1e-3 → 25.977% a 1e-2: factor exactamente 10, pendiente
≈ 26). No es degradación difusa: la aceptación se sesga **hacia 1/2**,
aceptando de más los movimientos malos y de menos los buenos. Es una
distorsión direccional, predecible y con ley cerrada — bastante más fuerte
como resultado que "el ruido empeora las cosas".

---

## 5. El obstáculo que el diseño todavía no cubre

El runner despacha a `cosmo_modular_quantum.py` y `cosmo_genetic_optimizers.py`.
**Nunca a `qpu_cosmo_samplers.py`.** Así que el atajo, tal como está planteado,
entrega ruido para QMCMC y QVMC por una ruta que el runner no conoce, y **no
entrega nada para el QGA** — precisamente el método que predices más robusto.

La ruta limpia es la contraria a la del atajo: `noise_model` por
`cosmo_core.make_simulator`, que alcanza los tres módulos a la vez, con
conversión a ρ solo en los sitios 2 y 3 (que es exacta) y a conteos solo en el
sitio 1 (que es donde de verdad se rompe). El atajo sigue siendo valioso, pero
como lo que dijiste al principio: **el primer test extremo-a-extremo de
`qpu_cosmo_samplers.py`**, hoy validado solo en dry-run.

---

## 6. Qué resultados previos habría que rehacer

**Ninguno.** Nada de lo verificado aquí contradice los resultados ideales
existentes; al contrario, la identidad de 1.11e-16 se reconfirma de forma
independiente. Lo que cambia es el alcance de las afirmaciones sobre DD y
twirling en cualquier corrida que no sea de hardware real.

## 7. Decisiones pendientes de tu confirmación

1. ¿Ruta `make_simulator` (alcanza los tres módulos, toca el simulador ideal)
   o ruta atajo (no toca nada, deja al QGA fuera del eje)? Recomiendo la
   primera, con la segunda como test extremo-a-extremo del módulo QPU.
2. Techo de qubits para tareas ruidosas: propongo 13 duro.
3. ¿La columna `readout` se calcula con el mapa analítico (exacta) o por
   conteos (con ruido de disparo)? Recomiendo analítica.

---

# Parte B — Implementación

## 8. Qué se construyó

| Archivo | Qué es |
|---|---|
| `cosmo_noise.py` | **nuevo.** Fuente única del eje: peldaños, modelos de ruido, mapa analítico de lectura, techo de qubits, CLI compartida. |
| `cosmo_core.py` | guarda `[N1]` en `make_simulator`. |
| `cosmo_modular_quantum.py` | rutas de lectura ruidosas + `--noise` + `--proposal-route`. |
| `cosmo_genetic_optimizers.py` | `--noise`; sin refactor de lectura (ya medía). |
| `cosmo_hpc_runner.py` | `--noise-sweep`, tercer modelo de memoria, techo de 13, columna `noise` en el JSON. |
| `qpu_noisy_simulation.py` | **nuevo.** Gemelo ruidoso del pipeline QPU. |
| `tests/test_noise_axis.py` | **nuevo.** 29 regresiones. |

`qpu_cosmo_samplers.py` **no se tocó**: queda intacto para hardware real.

## 9. Tres bugs propios encontrados durante la implementación

Los tres eran silenciosos — devolvían números limpios y plausibles.

**`[N1]`** — `AerSimulator(method='statevector', noise_model=...)` no levanta
nada: corre, reporta `COMPLETED` y devuelve el resultado **ideal**. Una
campaña ruidosa entera habría vuelto sin ruido. `make_simulator` ahora rechaza
esa pareja.

**`[B-RO]`** — un `add_all_qubit_readout_error` aparece en `to_dict()` **sin
clave `gate_qubits`**. Interpretar esa ausencia como `[[0]]` degradaba el canal
a un solo qubit. Coincidía **exacto en n=1** y divergía al crecer n y p.

**`[B-RECON]`** — reconstruir el modelo sin lectura con `NoiseModel.from_dict()`
es **lossy** con el modelo calibrado: contra FakeBrisbane la ρ reconstruida
difiere en **6.2e-4** a 3 qubits. En los peldaños sintéticos daba 0.0, así que
solo aparecía con el backend real. Se eliminó la reconstrucción entera.

## 10. Decisión sobre `_raw_block`: la ruta por conteos existe también sin ruido

`Re(ψ)·sign(Im(ψ))` es un operador distinto de `⟨Z_q⟩`, no su versión ruidosa.
Comparar la columna ideal (amplitudes) contra las ruidosas (conteos) mezclaría
el efecto del ruido con el del cambio de lectura. Por eso `--proposal-route
counts` está disponible **también en `--noise none`**: el eje se mide dentro de
una sola ruta, y el cambio de ruta queda como comparación aparte.

## 11. Dos resultados que salieron de la implementación

**La calibración absorbe el 100% del canal de lectura uniforme.** Con volteo
simétrico p, `⟨Z_q⟩' = (1-2p)·⟨Z_q⟩` — un reescalado escalar — y la calibración
a std unitaria lo divide. Verificado a **8e-15 incluso con p = 0.20**. Responde
la pregunta abierta del diseño: no absorbe "parte" del efecto, absorbe todo el
componente de lectura y **nada** del de compuerta (0.109 en `full`, 0.023 en
FakeBrisbane).

Consecuencia: en la fila de la propuesta, la columna `readout` sale **idéntica**
a la ideal-por-conteos. Es el mismo síntoma que tenía el bug `[B-RO]`, así que
está documentado y fijado con test para que no se confunda con una regresión.

**La normalización del QVMC es inmune al eje por construcción.** El circuito de
`quantum_amplitude_normalization` se ejecuta pero su resultado se descarta: el
valor devuelto es la suma exacta. El peldaño 100% del QVMC no puede degradarse
por su culpa; cualquier degradación entre 67% y 100% viene de otro sitio.

**El error de lectura domina la cola de rechazo de la aceptación.** Una
aceptación exacta de A = 0.00248 pasa a 0.0323 con p = 0.03: la lectura pone un
suelo de ~p, de modo que movimientos que deberían rechazarse casi siempre se
aceptan **13× de más**. Es un efecto mucho mayor que el de compuerta y va en
dirección contraria a la intuición de "el ruido degrada suavemente".

## 12. Primera corrida extremo-a-extremo de `qpu_cosmo_samplers.py`

El gemelo la ejecuta de verdad, no en dry-run. QMCMC y QVMC corren en los
cuatro peldaños. QVMC-LCDM, 8 iteraciones: KL 11.58 (ideal) → 12.52
(FakeBrisbane).

## 13. Qué resultados previos hay que rehacer

**Ninguno.** `--noise none` reproduce la ruta ideal bit a bit (verificado, ida
y vuelta desde peldaños ruidosos), y una corrida ideal del runner produce
exactamente las mismas rutas de salida que antes del eje.

## 14. Pendiente

- README (§ del eje de ruido, tabla de memoria, conteo de tests).
- Medir de verdad las escaleras bajo ruido: hasta aquí está el instrumento, no
  los resultados.
