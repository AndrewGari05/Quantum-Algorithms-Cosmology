# Cómo correr esto en una HPC

Guía operativa. El README tiene el detalle científico; esto es la secuencia de
comandos.

---

## 0. Instalar

```bash
pip install -r requirements.txt
```

`requirements.txt` incluye `qiskit-ibm-runtime`, que **faltaba**:
`qpu_cosmo_samplers.py` lo importa desde siempre, así que una instalación
limpia previa nunca pudo ejecutar ese módulo.

## 1. Comprobar el nodo ANTES de lanzar nada

```bash
python cosmo_hpc_runner.py --gpu-check
pytest -q
```

El primero imprime qué dispositivos declara Aer, qué paquetes hay instalados y
si `nvidia-smi` ve tarjeta. **Córrelo en cada máquina nueva.**

> **Sobre `--gpu`.** La rueda `qiskit-aer` de PyPI está compilada **solo para
> CPU**. Instalar `cuquantum-cu12` / `custatevec-cu12` a su lado **no** la
> convierte en una versión con GPU: Aer tiene que estar *construida* con
> soporte CUDA. Si `--gpu-check` dice que Aer no expone GPU, `--gpu` correrá en
> CPU — antes lo hacía en silencio, ahora avisa y explica por qué. Consulta la
> documentación de qiskit-aer para saber qué rueda con GPU corresponde a la
> versión 0.17.x antes de instalar nada.

El segundo debe dar **116 tests en verde**. Si alguno falla en un nodo nuevo,
párate ahí: algo del entorno no coincide.

## 2. La corrida completa

Sin SLURM: el runner lanza subprocesos directos, un modelo por proceso, y
reparte los núcleos solo.

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
    > campana.log 2>&1 &
```

Para correr en **hardware real de IBM**, el procedimiento completo está en
`COMO_CORRER_QPU.md` — incluye la prueba de humo de 1 job que conviene mandar
*antes* que nada.

100 tareas. Para vigilarla: `tail -f campana.log`, y `grep -c OK
results/hpc_*/master_profile.csv` para contar las que van.

**Por que 120 generaciones y no 500.** Medido en los logs de la campana
anterior: ir de 60 a 500 generaciones gana **como maximo 0.73 en χ²** sobre
1058, y solo en CPL (4 parametros). En ΛCDM, PEDE, wCDM y GEDE la ganancia es
**exactamente 0** — el genetico converge en la generacion 10. Con matriz de
densidad cada generacion cuesta ~24 min a 12 qubits, asi que 500 generaciones
son semanas por celda a cambio de 0.7 en χ², que no mueve ni el AIC ni el BIC.
120 deja margen de sobra para CPL y mantiene el eje de ruido en horas.

Ajustado a un contenedor de **63 Gi y 15 nucleos**. En una maquina con mas RAM
sube `--max-task-gb` y `--mem-budget-gb`; el runner detecta solo los limites
del contenedor (cgroup) si no los pasas.

Qué hace cada parte:

| Flag | Por qué |
|---|---|
| `--noise-sweep` | los cuatro peldaños del eje. Añade sola una quinta columna, `none-counts`, que es el control. |
| `--nqpp-sweep 3 5` | resolución del QVMC. El runner **recorta por modelo**: ΛCDM llega a 6, CPL baja a 3 bajo ruido, y te dice por qué. |
| `--nbits-sweep 4 8` | resolución del genético. Sin ruido aguanta mucho más (16 B/estado); con ruido lo corta `--noisy-task-hours`. Los peldaños nb7/nb8 sin ruido son los que prueban que el sesgo aparente del QGA al 67 % era resolución y no cuántica. |
| `--max-task-gb 24` | de aquí sale el techo de qubits de los samplers. A 24 GB con CC+BAO+Pantheon el techo es 18 q, que admite `cpl/nqpp4` (16 q, 5.8 GB) y `gede,wcdm/nqpp5` (15 q, 3.0 GB) y rechaza correctamente `cpl/nqpp5` (20 q, ~88 GB), que es la que murió la vez pasada. |
| `--mem-budget-gb 50` | presupuesto agregado. Deja colchón sobre el límite duro del contenedor: pasarse es un OOMKill (SIGKILL, sin traceback ni resultados parciales). |
| `--noisy-task-hours 48` | **[B-TIME]** presupuesto de reloj por tarea genética con ruido. De aquí y de `--generations` sale el techo de qubits del genético en el eje de ruido. Al QGA con matriz de densidad no lo frena la RAM sino el tiempo: ×4.4 por qubit (73 s/gen a 10 q, 1425 s/gen a 12 q, ~7.7 h/gen a 14 q). Sin este techo el plan acepta celdas de meses que además bloquean las figuras de resumen de toda la corrida. |
| (perfilado) | Está **activado por defecto** en el runner: RAM pico, VRAM, horas-GPU por tarea. `--profile` NO es un flag válido aquí — el runner lo reenvía solo a los hijos. Para apagarlo: `--no-profile`. |
| `--seed 42` | reproducibilidad, también de las partes cuánticas. |

`--dataset`: usa `CC+BAO+Pantheon` si solo tienes
`pantheon_full_parameters.txt`. Para `CC+BAO+Pantheon+` (covarianza completa)
hacen falta además `Pantheon+SH0ES.dat` y `Pantheon+SH0ES_STAT+SYS.cov`, que
**no** vienen en el zip.

### Antes de la de verdad, una de prueba

```bash
python cosmo_hpc_runner.py --noise-sweep none,readout --models lcdm \
    --nqpp 3 --steps 500 --qvmc-iter 20 --generations 20 \
    --population-size 40 --n-bits 4 --max-task-gb 12 --outdir /tmp/prueba
```

Cinco minutos. Valida el flujo entero, incluidas las figuras nuevas.

## 3. Variantes útiles

**El QMCMC es gratis bajo ruido** — su motor usa 2–4 qubits y `nqpp` no toca
sus circuitos. Si quieres el QMCMC ruidoso a resolución alta:

```bash
python cosmo_hpc_runner.py --models lcdm --noise-sweep none,readout,full \
    --nqpp 9 --only-samplers --qvmc-iter 0 --steps 40000 --max-task-gb 12
```

**Curva de degradación continua** — convierte cuatro peldaños en una curva.
Barato (2–4 qubits):

```bash
for p in 0.005 0.01 0.02 0.03 0.05 0.08 0.12; do
  python cosmo_hpc_runner.py --models lcdm --noise readout --noise-readout-p $p \
      --only-samplers --nqpp 4 --steps 40000 --max-task-gb 12 \
      --outdir resultados_ro_$p
done
```

**Solo genético, para el material de la defensa:**

```bash
python cosmo_hpc_runner.py --only-genetic \
    --noise-sweep none,readout,full,FakeBrisbane \
    --nbits-sweep 4 7 --generations 300 --population-size 400 --max-task-gb 12
```

**El pipeline QPU contra ruido simulado** (no toca hardware ni credenciales):

```bash
python qpu_noisy_simulation.py --model lcdm --method both --noise FakeBrisbane \
    --steps 2000 --iters 200 --nqpp 3
```

## 4. Techos de qubits: qué esperar

### Sin ruido, el techo depende del DATASET

Esto no estaba en la primera versión y costó tres tareas muertas. El coste de
memoria de una tarea de samplers es la rejilla **por el número de puntos de
datos**: arreglos de forma `(2^q, N_data)`. Medido:

| `total_q` | CC+BAO (51 pts) | CC+BAO+Pantheon (1099 pts) |
|---|---|---|
| 16 | 1.0 GB | 4.8 GB |
| 18 | 4.0 GB | **19.6 GB** |
| 20 | — | ~88 GB |
| 21 | — | ~175 GB |

Un estado cuesta 14 kB con CC+BAO y 74 kB con CC+BAO+Pantheon. El mismo nodo
concede ~2 qubits menos al cambiar de dataset. El runner ya lo tiene en cuenta
(`DATASET_N_DATA`) y lo imprime al arrancar; un dataset que no esté en la tabla
usa el más grande, que es la dirección segura.

> **Esto era un bug.** La primera versión usaba una constante que no dependía
> de `N_data` y estimaba 3.7 GB donde se midieron 19.6. En la campaña
> CC+BAO+Pantheon del 2026-08-31 el planificador aceptó `cpl/nqpp5` (20 q) y
> `gede,wcdm/nqpp7` (21 q); las tres murieron con `rc=-9` (SIGKILL del OOM
> killer, sin traceback ni resultados parciales) después de terminar el MCMC y
> el QMCMC, justo al empezar el QVMC. Corregido y con test de regresión contra
> las 7 mediciones reales.

### Con ruido, manda el lote de parameter-shift

Con **63 Gi**, el eje de ruido concede:

| pipeline | techo | por qué |
|---|---|---|
| QMCMC | sin límite de `nqpp` | su motor usa `max(2, d)` qubits |
| QVMC rungs 0% / 33% | 15 q | una ρ suelta |
| QVMC rungs 67% / 100% | **11–12 q** | manda el lote de parameter-shift (`2·n_φ` matrices de densidad: 10.4 GiB a 11 q, **44 GiB a 12**) |
| QGA | 15 q | una ρ por operador, sin lote |

Por modelo, con ruido y `--max-task-gb 12`: ΛCDM/PEDE llegan a nqpp=5, 
wCDM/GEDE a nqpp=3, CPL a nqpp=2.

Como `--benchmark` recorre la escalera entera, el peldaño del lote fija el
techo de una tarea de samplers. Más RAM sube poco: el lote crece como `4^n`.

### Elige `--max-task-gb` con la tabla, no de memoria

Con `--max-task-gb 12` y CC+BAO+Pantheon el techo sale en **17 qubits**, así
que `gede/wcdm` a nqpp=6 (18 q, 19.6 GB reales) **no** entra: antes entraba
por accidente, porque la estimación era 5× baja. Si quieres esas tareas, pide
`--max-task-gb 24` y baja el paralelismo en consecuencia.

## 5. Qué mirar al terminar

| Archivo | Qué responde |
|---|---|
| `noise_comparison[_nqppN]_<modelo>.png` | **empieza aquí.** Con `--nqpp-sweep` sale **una por resolución**: mezclarlas convertiría un recorte de techo en lo que parece un efecto del ruido. Cada rung a lo largo del eje de ruido. Parámetros en unidades de σ (banda gris = ±1σ), calidad de ajuste a la derecha. |
| `genetic_evolution_<modelo>.gif` | la animación de la población convergiendo, todos los rungs juntos. Para la defensa. |
| `genetic_convergence_<modelo>.png` | trayectoria + estimación final del genético. Sustituye al corner. |
| `corner_ladder_*` | posteriores con bandas 1σ/2σ/3σ sombreadas. |
| `convergence_<modelo>.png` | efecto de `nqpp`. |
| `resultados_TODOS_los_modelos.csv` | todo, con columna `noise` para pivotar. |

Tres advertencias al leer:

**No compares contra `none` sin usar `none-counts`.** El peldaño ideal lee
amplitudes y los ruidosos leen conteos: comparar los dos mide el ruido *y* el
cambio de operador a la vez. En la primera campaña eso hacía parecer que el
ruido de lectura **mejora** el muestreo.

**El ESS no sirve como métrica de calidad en este eje.** Sube con el ruido
mientras el KL empeora: el ruido aplana la distribución y una distribución
plana da muestras menos correlacionadas. Repórtalo junto a KL o σ, nunca solo.

**Las bandas de los corner son 1σ/2σ/3σ**, y en un panel de dos parámetros
contienen 39.3 / 86.5 / 98.9 %, no 68 / 95 / 99.7. Esos son los de una
gaussiana en una dimensión.

---

## 6. Rehacer una figura sin repetir la campaña

Cada tarea genética deja ahora un `ga_state.npz` junto a sus figuras, con la
población final, sus pesos y la historia por generación. Con eso se redibuja
todo en segundos:

```bash
python cosmo_genetic_optimizers.py --replot results/hpc_*/genetic_lcdm_*/model_lcdm/ga_state.npz
```

Antes esto no se podía: la población solo vivía en memoria, así que **cambiar
un color o corregir un eje obligaba a repetir la corrida entera** — días de
cómputo por un cambio cosmético. Se descubrió al arreglar `[B-GAPLOT]`.
Añade `--anim none` para saltarte el GIF, `--outdir` para escribir en otro
sitio, y `--no-state` en la corrida si prefieres no gastar el ~1–3 MB por
tarea (a cambio de volver al problema de antes).
