#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
readme_noise_patch.py — Inserta la documentacion del eje de ruido en README.md.

Por que un parche y no un README reescrito
-------------------------------------------
El README son ~53 kB y esta seccion solo anade contenido en siete puntos
concretos. Reescribirlo entero significaria re-teclearlo desde una copia que
podria estar desfasada respecto al archivo real del repo, y cualquier deriva
entre ambas se perderia en silencio. Este script edita EL archivo real por
anclas de texto y **aborta si un ancla no aparece exactamente una vez**, de
modo que un README que haya cambiado desde que se escribio el parche produce
un error visible en vez de una version corrupta.

Es idempotente: si la marca del eje de ruido ya esta presente, no hace nada.

Uso
---
    python readme_noise_patch.py            # aplica sobre ./README.md
    python readme_noise_patch.py --check    # solo verifica las anclas
    python readme_noise_patch.py --path X   # otro archivo
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Tuple

MARKER = "<!-- eje-de-ruido-nisq -->"


# =============================================================================
# Bloques nuevos
# =============================================================================

PART1_NOISE = """
## The second dial: how much noise

Everything above assumes a **perfect** quantum computer. Real quantum
hardware makes mistakes: gates are slightly off, and when you finally read a
qubit you sometimes read the wrong value. Every result in this project up to
this point was computed on a *noiseless* simulator — the ideal limit.

So there is now a **second dial**, independent of the quantumness one:

```
                        noise  →
                    none   readout   gates+readout   real device
    quantumness 0%    ·        ·            ·             ·
          |    33%    ·        ·            ·             ·
          ↓    67%    ·        ·            ·             ·
              100%    ·        ·            ·             ·
```

Turn it up and the answers get **worse**. That is the point, and it is not a
failure: the interesting question is not *whether* the methods degrade but
**at what rate** — which rung of the quantumness ladder survives noise, and
which one breaks first.

```bash
# One noise level
python cosmo_modular_quantum.py --benchmark --model lcdm --noise full

# The whole 2-D matrix in one launch
python cosmo_hpc_runner.py --models lcdm --noise-sweep none,readout,full
```

`--noise none` is the default and reproduces the previous, noiseless
behaviour **bit for bit**, so nothing published before this change moves.

"""


PART2_NOISE = """
## The noise axis (second ablation dimension)

Until this section the framework had ONE axis, quantumness. It now has two,
and results form a matrix rather than a ladder:

```
                        noise  →
                    none   readout   full   <backend>
    quantumness 0%    ·        ·        ·        ·
          |    33%    ·        ·        ·        ·
          ↓    67%    ·        ·        ·        ·
              100%    ·        ·        ·        ·
```

`cosmo_noise.py` is the single source of truth for this axis, the way
`cosmo_core.py` is for the physics. The four rungs:

| Level | What it models |
|---|---|
| `none` | ideal limit — every result published before this change |
| `readout` | symmetric, uniform measurement-flip error only |
| `full` | depolarizing on 1- and 2-qubit gates **plus** readout |
| `<backend>` | calibrated model of a real IBM device (`FakeBrisbane`, …) |

Available as `--noise` on all three executables and on the HPC runner, which
additionally takes `--noise-sweep none,readout,full` and treats the level as
a task dimension exactly like `--nqpp-sweep`. Every result row carries a
`noise` column, written even for ideal runs so the matrix pivots without
special cases.

### Why readout error travels separately from the NoiseModel

Readout error is a **classical channel applied after measurement**: it does
not touch the state. Aer only applies it when the circuit measures. But two
of the simulator's three amplitude reads (`hadamard_accept_log_batch` and
`_kl_batch`) are probabilities obtained from `rho` **without measuring**, so
they are *blind* to it — the `readout` column of the matrix would come out
identical to the ideal column. Not a loud failure: clean, plausible, wrong
numbers.

Measured: at p = 0.01, 0.03 and 0.05, `rho[0,0]` does not move a single
digit.

The fix is not to measure but to apply the channel in closed form. On n
qubits readout error is a tensor product of 2x2 stochastic matrices, so it
acts exactly on `diag(rho)`:

```
P_noisy = (tensor_q M_q) @ P_ideal
```

That gives the **complete noise axis, exact and free of shot noise**, at the
cost of an O(n·2^n) contraction. It matters because the shot-based route
cannot resolve the bias being measured: at a 1e-3 gate error the acceptance
bias is ~1.4e-4, and separating that from sampling noise would take ~1e7
shots *per acceptance evaluation*.

`NoiseSpec.simulator_kwargs(counts_route=...)` decides which model goes to
Aer, because getting it wrong either double-counts readout error or makes it
vanish silently.

### Density matrix, not shots

Noisy simulation needs either a density matrix (`16·4^n` bytes) or shot
trajectories (`shots·2^n` in time). Measured on the project's real ansatz
(3 layers, B=2 bindings, 4096 shots, FakeBrisbane):

| qubits | density matrix | trajectories | rho size |
|---|---|---|---|
| 6 | **1.66 s** | 5.31 s | 0.1 MB |
| 10 | **3.77 s** | 22.4 s | 16 MB |
| 12 | **34.4 s** | 101 s | 256 MB |
| 13 | **216 s** | 333 s | 1.0 GB |

The density matrix **wins on time everywhere it fits** (1.5x–15x), because it
evolves once and then samples, while trajectories re-simulate the whole
circuit per shot. The advantage is largest exactly where QMCMC operates (few
qubits, many bindings: 4.2 s vs 52 s for a 64-proposal block).

**Shots do not buy qubits.** Above ~14 qubits neither route is usable — one
runs out of memory, the other out of time (CPL at nqpp=6 is ~114 h *per
job*). Hence `MAX_NOISY_QUBITS = 13`, a **time** ceiling as much as a memory
one, which is why it is an explicit constant and not derived from detected
RAM like the other two ceilings. More RAM does not move it.

| Model | d | nqpp=3 | nqpp=4 | nqpp=5 | nqpp=6 |
|---|---|---|---|---|---|
| LCDM, PEDE | 2 | 6 ok | 8 ok | 10 ok | 12 ok |
| wCDM, GEDE | 3 | 9 ok | 12 ok | 15 no | 18 no |
| CPL | 4 | 12 ok | 16 no | 20 no | 24 no |

QMCMC is absent from that table because its proposal engine uses
`n_qubits = max(2, d)` — 2 to 4 qubits, essentially free under any noise
model, for every model and every nqpp.

### The proposal changes operator under noise (and what that costs)

`_raw_block` reads `Re(psi)·sign(Im(psi))`, which is phase-sensitive and
therefore undefined for a mixed state. It is the ONLY one of the simulator's
reads that noise genuinely breaks; the other two are probabilities and
survive exactly as `rho[0,0]` and `diag(rho)`.

Under noise the proposal must be read by measurement instead
(`<Z_q> = 1 - 2·P(q=1)`, the same rule the QPU pipeline runs on hardware).
That is a **different operator**, not a noisy version of the same one, so
comparing the ideal column (amplitudes) against the noisy ones (counts)
would confound two effects. `--proposal-route counts` is therefore available
**in the ideal rung too**: the noise axis is measured inside one route, and
the route change stays a separate, controlled comparison.

### Results that came out of building this

**The unit-std calibration absorbs 100% of uniform readout error.** With a
symmetric flip probability p, `<Z_q>' = (1-2p)·<Z_q>` — a scalar rescaling —
and the calibration divides it straight out. Verified to 8e-15 **even at
p = 0.20**. Gate noise is not absorbed (0.109 for `full`). So in the proposal
row, the `readout` column comes out **identical** to the ideal-counts column.
That is a provable invariance, not a regression, and it is pinned by a
regression test so it is never mistaken for one.

**Readout error dominates the rejection tail of the acceptance.** An exact
acceptance of A = 0.00248 becomes 0.0323 at p = 0.03: readout puts a floor of
~p on the acceptance, so moves that should almost always be rejected are
accepted **13x too often**. This is far larger than the gate-noise effect and
runs against the intuition that noise degrades smoothly.

**The gate-noise bias on the acceptance is linear and directional.**
Depolarizing drives rho toward I/2, so the bias is `lambda·(1/2 - A)`: the
acceptance is pulled **toward 1/2**, over-accepting bad moves and
under-accepting good ones. Relative bias scales linearly with the gate error
rate (2.60% at 1e-3, 25.98% at 1e-2).

**The QVMC normalization component is immune to this axis by construction.**
`quantum_amplitude_normalization` runs its circuit but discards the result —
the returned value is the exact classical sum. The QVMC 100% rung therefore
cannot degrade because of it, and any degradation seen between 67% and 100%
comes from somewhere else.

### `qpu_noisy_simulation.py` — noisy twin of the QPU pipeline

Runs the hardware pipeline against a noisy local simulator instead of IBM.
It does **not** copy `qpu_cosmo_samplers.py` — it imports it and replaces one
piece, the connection. A copied file would diverge from the original at the
first fix applied to one and not the other, silently.

Its value is twofold: it is the **first end-to-end execution** of
`qpu_cosmo_samplers.py` (previously validated only in `--dry-run`, i.e. with
synthetic uniform counts that exercise shapes, not physics), and it gives the
**shot-noise** version of the axis, whereas the `cosmo_modular_quantum` axis
computes exact probabilities to isolate decoherence from sampling.

```bash
python qpu_noisy_simulation.py --model lcdm --method qmcmc --noise full
python qpu_noisy_simulation.py --model wcdm --method qvmc --noise FakeBrisbane
```

> **[DD-INERTE] The error suppression is inert on a simulated backend.**
> `QPUConnection` enables dynamical decoupling XY4 and Pauli twirling. On real
> hardware those act. In qiskit-ibm-runtime's local testing mode they are
> **discarded**, announced only by a `UserWarning` that is lost in a long run's
> logs. So a simulated noisy run measures *noise without error suppression*
> while hardware measures *noise with it* — not the same experiment. It is a
> defensible reading (a **lower bound** on achievable hardware quality) but it
> must be recorded, not inferred. `qpu_noisy_simulation.py` therefore avoids
> `SamplerV2` entirely and states this in the run log.

### Bugs this axis surfaced

Three of them, all silent — they returned clean, plausible numbers.

* **`[N1]`** — `AerSimulator(method='statevector', noise_model=...)` raises
  nothing. It runs, reports `COMPLETED`, and returns the **noiseless**
  result. An entire noisy campaign would have come back ideal with no signal
  at all. `make_simulator` now rejects the pairing.
* **`[B-RO]`** — an `add_all_qubit_readout_error` appears in `to_dict()`
  **without** a `gate_qubits` key. Reading that absence as `[[0]]` degrades
  the channel to a single qubit; the symptom matched exactly at n=1 and
  diverged only as n and p grew.
* **`[B-RECON]`** — rebuilding a readout-free model with
  `NoiseModel.from_dict()` is **lossy** for calibrated backends: 6.2e-4 of
  drift in rho against FakeBrisbane at 3 qubits, while synthetic rungs showed
  0.0. The reconstruction was removed entirely (it was also using an API
  deprecated since qiskit-aer 0.15).

`noise_feasibility_probe.py` reproduces the measurements behind this section.

"""


ARCH_OLD = """            cosmo_core.py   ← PHYSICS + DATA + STATISTICS + Aer device factory
        ┌────────┼────────┐         (shared by everything)
        │        │        │
cosmo_modular_  qpu_cosmo_  cosmo_genetic_      cosmo_profiling.py
quantum.py      samplers.py optimizers.py       (RAM / VRAM / GPU-hours,
(Aer simulator, (real IBM   (CGA/QGA global      used by the two simulators
 quantumness     Quantum HW, optimization for    via --profile)
 benchmark)      SamplerV2)  the MAP + live GUI)
```"""

ARCH_NEW = """   cosmo_core.py  ← PHYSICS + DATA + STATISTICS + Aer device factory
   cosmo_noise.py ← NOISE AXIS (levels, models, analytic readout, ceiling)
        ┌────────┼────────┐         (both shared by everything)
        │        │        │
cosmo_modular_  qpu_cosmo_  cosmo_genetic_      cosmo_profiling.py
quantum.py      samplers.py optimizers.py       (RAM / VRAM / GPU-hours,
(Aer simulator, (real IBM   (CGA/QGA global      used by the two simulators
 quantumness     Quantum HW, optimization for    via --profile)
 benchmark)      SamplerV2)  the MAP + live GUI)
                     ▲
        qpu_noisy_simulation.py — same pipeline, noisy local Aer
        instead of IBM. Imports qpu_cosmo_samplers and swaps ONLY
        the connection, so the hardware module stays untouched.
```

The two axes are orthogonal: `cosmo_core` decides *what* is computed and
`cosmo_noise` decides *under what noise*. Every quantum module — and the HPC
runner — reads its noise level from that one module."""


# =============================================================================
# Anclas
# =============================================================================

def edits() -> List[Tuple[str, str, str]]:
    """Lista de (etiqueta, ancla, texto de reemplazo).

    Cada ancla debe aparecer EXACTAMENTE una vez en el README; si no, el
    parche aborta sin escribir nada.
    """
    return [
        (
            "estado",
            "**Status:** post-Fase-3 hardening (33 tests, all green).",
            "**Status:** post-Fase-4 (62 tests, all green). Fase 4 added the "
            "**noise axis** — a second, orthogonal ablation dimension, so "
            "results are now a 2-D matrix (quantumness x noise) instead of a "
            "ladder; see [The noise axis](#the-noise-axis-second-ablation-"
            "dimension). `--noise none` is the default and reproduces every "
            "previous result bit for bit.\n\n"
            "**Status (Fase 3):** hardening (33 tests, all green).",
        ),
        (
            "parte1",
            "The genetic optimizer (`cosmo_genetic_optimizers.py`) has its own "
            "**third\ndial**, also in thirds — see the technical section for "
            "details.\n",
            "The genetic optimizer (`cosmo_genetic_optimizers.py`) has its own "
            "**third\ndial**, also in thirds — see the technical section for "
            "details.\n" + PART1_NOISE,
        ),
        (
            "arquitectura",
            ARCH_OLD,
            ARCH_NEW,
        ),
        (
            "parte2",
            "## Diagnostics and correctness fixes\n",
            MARKER + "\n" + PART2_NOISE + "\n## Diagnostics and correctness "
            "fixes\n",
        ),
        (
            "tabla-qpu",
            "| Error suppression | — | Dynamical Decoupling XY4 + Pauli "
            "twirling |",
            "| Error suppression | — | Dynamical Decoupling XY4 + Pauli "
            "twirling (**inert on a simulated backend** — see [DD-INERTE]"
            "(#qpu_noisy_simulationpy--noisy-twin-of-the-qpu-pipeline)) |",
        ),
        (
            "memoria",
            "To go above the default cap on a bigger machine, raise it "
            "explicitly:",
            "> **A noisy run has a third, much lower ceiling.** With `--noise` "
            "on, simulation switches to a density matrix: `16·4^n` bytes "
            "*added to* the grid cost above, not replacing it (1.0 GB of rho "
            "at 13 qubits, 4.0 GB at 14). The binding limit is TIME rather "
            "than memory — 216 s per job at 13 qubits, ~14 min at 14 — and "
            "the shot-trajectory alternative, which would fit in RAM, is "
            "slower still. `cosmo_noise.MAX_NOISY_QUBITS = 13` is therefore a "
            "hard constant that more RAM does not raise. The HPC runner "
            "applies it automatically and clamps heavy models per model, the "
            "same way the other two ceilings do.\n\n"
            "To go above the default cap on a bigger machine, raise it "
            "explicitly:",
        ),
        (
            "tests",
            "* **Tests + CI.** `tests/` holds 33 tests:",
            "* **Tests + CI.** `tests/` holds 62 tests (33 through Fase 3, "
            "plus 29 added in Fase 4 for the noise axis — the analytic "
            "readout map against Aer counts, the `[N1]` guard, the ideal "
            "rung's bit-for-bit reproducibility, the proposal's readout "
            "invariance, and the runner's third memory model):",
        ),
    ]


# =============================================================================

def apply(text: str, check_only: bool = False) -> str:
    """Aplica todas las ediciones, o aborta si alguna ancla no encaja.

    Args:
        text: contenido actual del README.
        check_only: si True, solo verifica las anclas.

    Returns:
        El texto resultante (igual al de entrada si `check_only`).

    Raises:
        SystemExit: si un ancla falta o aparece mas de una vez.
    """
    problems: List[str] = []
    for label, anchor, _ in edits():
        n = text.count(anchor)
        if n != 1:
            problems.append(f"  [{label}] el ancla aparece {n} veces, se "
                            f"esperaba 1")
    if problems:
        print("El README no coincide con lo que este parche espera:")
        print("\n".join(problems))
        print("\nNo se escribio nada. Revisa si el README cambio desde que se "
              "escribio el parche.")
        raise SystemExit(2)

    print("Todas las anclas verificadas (%d)." % len(edits()))
    if check_only:
        return text

    out = text
    for label, anchor, replacement in edits():
        out = out.replace(anchor, replacement, 1)
        print(f"  aplicado: {label}")
    return out


def main(argv=None) -> int:
    """Punto de entrada."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--path', default='README.md')
    ap.add_argument('--check', action='store_true',
                    help='solo verificar las anclas, sin escribir')
    args = ap.parse_args(argv)

    with open(args.path, encoding='utf-8') as fh:
        text = fh.read()

    if MARKER in text:
        print("El README ya contiene la seccion del eje de ruido; nada que "
              "hacer.")
        return 0

    out = apply(text, check_only=args.check)
    if args.check:
        return 0

    with open(args.path, 'w', encoding='utf-8') as fh:
        fh.write(out)
    print(f"\n{args.path} actualizado "
          f"({len(text)} -> {len(out)} caracteres).")
    return 0


if __name__ == '__main__':
    sys.exit(main())
