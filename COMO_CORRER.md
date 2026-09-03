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

El segundo debe dar **75 tests en verde**. Si alguno falla en un nodo nuevo,
párate ahí: algo del entorno no coincide.

## 2. La corrida completa

Sin SLURM: el runner lanza subprocesos directos, un modelo por proceso, y
reparte los núcleos solo.

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

python cosmo_hpc_runner.py \
    --noise-sweep none,readout,full,FakeBrisbane \
    --nqpp-sweep 3 5 \
    --nbits-sweep 4 7 \
    --dataset CC+BAO+Pantheon \
    --steps 20000 --qvmc-iter 3000 --chains 8 --shots 4096 \
    --generations 200 --population-size 300 \
    --total-cores 14 --mem-budget-gb 50 --max-task-gb 12 \
    --seed 42
```

Ajustado a un contenedor de **63 Gi y 15 nucleos**. En una maquina con mas RAM
sube `--max-task-gb` y `--mem-budget-gb`; el runner detecta solo los limites
del contenedor (cgroup) si no los pasas.

Qué hace cada parte:

| Flag | Por qué |
|---|---|
| `--noise-sweep` | los cuatro peldaños del eje. Añade sola una quinta columna, `none-counts`, que es el control. |
| `--nqpp-sweep 3 5` | resolución del QVMC. El runner **recorta por modelo**: ΛCDM llega a 6, CPL baja a 3 bajo ruido, y te dice por qué. |
| `--nbits-sweep 4 7` | resolución del genético, que aguanta más porque no tiene el lote de parameter-shift. |
| `--max-task-gb 12` | **importa más de lo que parece**: de aquí sale el techo de qubits. A 11 qubits una tarea ruidosa pesa 10.4 GiB y caben 6 en paralelo; a 12 pesa 44 GiB y solo cabe **una**, desperdiciando el resto de núcleos. |
| `--mem-budget-gb 50` | presupuesto agregado. Deja colchón sobre el límite duro del contenedor: pasarse es un OOMKill (SIGKILL, sin traceback ni resultados parciales). |
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
