# Correr en hardware real de IBM

Procedimiento completo para la **primera** corrida en una QPU. El objetivo de
la primera vez **no es obtener resultados**: es comprobar que el camino
completo funciona — credenciales, transpilación, envío, cola, respuesta,
parseo. Los resultados vienen en la segunda.

> `qpu_cosmo_samplers.py` nunca se ha ejecutado contra una máquina real. Todo
> lo que sigue está verificado *offline* (firmas de la API instalada,
> transpilación contra el mapa de acoplamiento real de FakeBrisbane, y
> `--dry-run` de punta a punta). Lo que no se puede verificar sin cuenta va
> marcado como **NO VERIFICADO**.

---

## 0. Lo que ya está comprobado

| Cosa | Estado |
|---|---|
| `qiskit-ibm-runtime` instalado | 0.49.0 |
| Canal usado por el código | `ibm_quantum_platform` — **el vigente** (el viejo `ibm_quantum` está retirado) |
| Primitiva | `SamplerV2` — la actual, no la V1 retirada |
| Transpilación a ISA | **Sí, obligatoria y presente** (`generate_preset_pass_manager(optimization_level=3, backend=...)`) |
| Circuito transpilado contra un mapa real | Verificado con FakeBrisbane: sale en `['ecr','rz','sx','x','measure','barrier']`, todo dentro de la base del backend |
| Contexto de ejecución | `Batch` por defecto, `Session` con `--session` (requiere plan de pago) |
| Supresión de errores | Dynamical decoupling XY4 activado |
| Tope de seguridad | `--max-jobs` (200 por defecto) aborta antes de enviar si el plan pide más |

---

## 1. Antes de tocar nada

### 1.1 Instala y comprueba

```bash
pip install -r requirements.txt
python -c "import qiskit, qiskit_ibm_runtime as r; print(qiskit.__version__, r.__version__)"
```

Debe decir `2.4.2 0.49.0`. Si `qiskit-ibm-runtime` no está, este archivo no
corre: es la dependencia que faltaba en la primera versión del
`requirements.txt`.

### 1.2 Guarda la cuenta UNA vez

Entra a <https://quantum.cloud.ibm.com>, copia tu **API key** y tu **CRN**
(el identificador de la instancia; en el plan Open aparece en el panel).
Después, una sola vez en Python:

```python
from qiskit_ibm_runtime import QiskitRuntimeService
QiskitRuntimeService.save_account(
    channel="ibm_quantum_platform",
    token="TU_API_KEY",
    instance="TU_CRN",          # el CRN de tu instancia
    set_as_default=True,
    overwrite=True,
)
```

Comprueba que quedó:

```python
from qiskit_ibm_runtime import QiskitRuntimeService
s = QiskitRuntimeService()
print([b.name for b in s.backends(operational=True, simulator=False)])
print(s.least_busy(operational=True, simulator=False).name)
```

Si esto imprime nombres de máquinas, ya está lo difícil.

> **NO VERIFICADO.** El código llama a `QiskitRuntimeService()` sin argumentos,
> o con `token=` si pasas `--token`. Con `token=` **no pasa `instance=`**, así
> que si tu cuenta necesita CRN explícito, `--token` puede fallar donde la
> cuenta guardada funciona. **Usa la cuenta guardada, no `--token`.**

### 1.3 Comprueba el plan sin conectarte

```bash
python qpu_cosmo_samplers.py --dry-run --model lcdm --dataset CC \
    --method qmcmc --steps 64 --block 64 --chains 1 --shots 1024 --samples 200
```

Debe decir `Estimated QPU jobs for this run: 1`. Si dice más, no lo mandes.

---

## 2. La prueba de humo — esto es lo de mañana

```bash
python qpu_cosmo_samplers.py \
    --model lcdm --dataset CC --method qmcmc \
    --steps 64 --block 64 --chains 1 \
    --shots 1024 --samples 200 \
    --least-busy \
    --max-jobs 2 \
    --outdir resultados_qpu_humo \
    --seed 42
```

**Qué manda:** 1 job, 1024 disparos. Es lo mínimo que ejercita el camino
entero. Segundos de QPU; el resto es cola.

**Qué esperar en el log, en orden:**

1. `Backend: ibm_XXX (127 qubits)` ← las credenciales funcionan
2. la transpilación a ISA (sin mensaje, pero si falla, falla aquí)
3. una espera larga y silenciosa ← la cola, es normal
4. `QMCMC-QPU — 64 steps, 1 chains:` con Ωm, H₀, χ² ← llegó y se parseó
5. `MEASURED TIMINGS: 1 jobs, mean wall Xs/job` ← el número que importa

Si llegas al punto 5, **el camino funciona** y ya puedes planear la corrida
buena. Si truena, truena barato.

**`--max-jobs 2` es el seguro.** Si el plan resultara ser más grande de lo que
crees, aborta *antes* de enviar en vez de quemarte la cuota.

---

## 3. La corrida de verdad (solo después de que la de humo pase)

```bash
python qpu_cosmo_samplers.py \
    --model lcdm --dataset CC --method qmcmc \
    --steps 1000 --block 64 --chains 4 \
    --shots 4096 --samples 4000 \
    --least-busy \
    --max-jobs 40 \
    --outdir resultados_qpu_lcdm \
    --seed 42
```

Del `--dry-run`: **16 jobs** para 1000 pasos con `block 64`, ~17 min de cola
proyectados (la proyección sale de la medición de la prueba de humo, así que
será más fiable después de correrla).

### Por qué QMCMC y no QVMC para la primera vez

`--method qvmc` gasta **un job por iteración de SPSA**: `--iters 30` son 30
jobs encolados, y cada uno espera su turno. El QMCMC agrupa 64 propuestas por
job, así que rinde muchísimo más por job. Deja el QVMC para cuando el camino
esté probado.

---

## 4. El detalle que hay que tener claro para comparar

**El dataset no es el mismo que el de tu campaña grande.** `qpu_cosmo_samplers.py`
solo acepta `CC`, `Pantheon+` y `CC+Pantheon+`. `--dataset CC` carga la tabla
`cosmic_chronometers.txt` — o sea, los **51 puntos CC+BAO**, exactamente el
dataset de la campaña `hpc_20260827` (CC+BAO), y **no** el
`CC+BAO+Pantheon` de las campañas grandes.

Así que la comparación honesta es:

> **QPU (`--dataset CC`) contra la campaña CC+BAO del simulador**, nunca contra
> la de CC+BAO+Pantheon.

`Pantheon+` requiere `Pantheon+SH0ES.dat` y `Pantheon+SH0ES_STAT+SYS.cov`, que
**no** vienen en el zip.

---

## 5. Lo que la prueba simulada NO cubre

`qpu_noisy_simulation.py` sustituye la clase `QPUConnection` entera por un
sustituto local. Todo lo que vive *dentro* de esa clase queda sin probar hasta
mañana. En concreto:

| Sin probar | Riesgo |
|---|---|
| `QiskitRuntimeService()` y la cuenta guardada | **Alto** — es el fallo más probable, y es el más barato de arreglar |
| `service.least_busy(...)` | Medio — si tu plan no expone backends operativos, levanta aquí |
| `generate_preset_pass_manager` contra el backend **real** | Bajo — verificado contra FakeBrisbane, que tiene el mismo mapa que ibm_brisbane |
| `Batch(backend=...)` y el ciclo de vida del contexto | Medio — **NO VERIFICADO**: revisa que el `Batch` se cierre; un contexto abierto puede seguir contando |
| ~~El formato real del resultado de `SamplerV2`~~ | **YA PROBADO** — se ejerce con `SamplerV2` en modo local contra FakeBrisbane, que devuelve un `PubResult` genuino. Salió un bug (`[B-CREG]`) y está corregido |
| Dynamical decoupling y twirling reales | Bajo — en el simulador se descartan en silencio (`[DD-INERTE]` en el log) |
| ~~Orden de llegada de los jobs~~ | **YA PROBADO** — verificado que el lote *k* corresponde a la fila *k* de parámetros, con `ry(0)→\|0⟩` y `ry(π)→\|1⟩` como firma inconfundible |

**Las dos filas "Alto/Medio" de arriba son literalmente para lo que sirve la
prueba de humo.** Por eso vale la pena mandarla antes que nada.

---

## 6. Si algo falla

| Síntoma | Causa casi segura |
|---|---|
| `AccountNotFoundError` | no guardaste la cuenta, o `set_as_default=False` |
| `IBMInputValueError` sobre el `instance` | falta el CRN en `save_account` |
| `Backend ... not found` | el nombre de `--backend` no existe en tu plan; usa `--least-busy` |
| Circuito rechazado por la base de puertas | la transpilación no corrió; no debería pasar, pero si pasa, es un bug y avísame |
| Se queda callado horas | es la cola, no un cuelgue. Mira el job en el panel web |
| `Estimated QPU jobs` > `--max-jobs` | aborta a propósito. Sube `--block` o baja `--steps` |

---

## 7. `[B-CREG]` — un bug que salió al probar esto

El código leía el registro clásico así:

```python
reg = getattr(data, 'meas', None) or getattr(data, 'c', None)
```

Los circuitos de este proyecto usan `measure_all()`, así que su creg se llama
`meas` y el camino feliz funciona. Pero con **cualquier otro nombre** esa
expresión devuelve `None` y dos líneas después revienta con
`AttributeError: 'NoneType' object has no attribute 'get_counts'` — un mensaje
que no dice nada, y que ocurre **después** de que el trabajo se haya ejecutado
y cobrado en la QPU.

Verificado con `SamplerV2` en modo local: con un creg llamado `lectura`, el
`DataBin` expone `data.lectura` y la expresión vieja falla. Corregido: ahora
busca por nombre conocido, cae al único campo que sepa dar conteos (avisando), y
si no encuentra nada levanta un error que **dice qué campos había** — que es lo
único útil a esas alturas.

Con esto, las dos filas de riesgo "Medio" de la tabla de arriba quedan cerradas
antes de gastar un segundo de máquina real.
