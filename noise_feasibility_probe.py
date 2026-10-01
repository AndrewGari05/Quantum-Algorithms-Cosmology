#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
noise_feasibility_probe.py — Feasibility study of the NISQ noise axis.
================================================================================

VERIFICATION script, not production. It does not modify any module of the
project and is not part of `cosmo_hpc_runner.py`: its only purpose is to back
with measurements the design decisions of Phase 4 (second ablation axis
= noise), so that no claim in the design document rests on code reading or
on intuition.

Context
-------
Today the framework produces its results with `AerSimulator(method='statevector')`
without noise. Adding noise collides with two facts:

  1. Aer does not apply noise with `method='statevector'`.
  2. The ideal simulator reads amplitudes at functionally distinct places
     (`QuantumProposalEngine._raw_block`, `hadamard_accept_log_batch`,
     `QVMCModular._kl_batch`, among others), and those reads are exactly what
     noise makes impossible.

The probes below measure which escape routes exist and how much they cost.

Probes
------
A. `probe_local_testing_mode`
     Checks whether `qiskit_ibm_runtime.SamplerV2` runs locally when given a
     simulated backend (the "shortcut" hypothesis: reuse the counts-based
     pipeline of `qpu_cosmo_samplers.py` pointing it at Aer instead of IBM).
     It ALSO records whether the project's error-suppression options
     (XY4 dynamical decoupling + Pauli twirling) survive that mode.

B. `probe_graded_axis`
     Checks that the four levels of the proposed noise axis
     (none / readout / full / real backend) are buildable and distinguishable.

C. `probe_cost`
     Measures time per job and peak RAM of the two noisy simulation methods
     (density matrix vs statevector trajectories) on the project's real
     ansatz, at the qubit widths the runner actually generates.

D. `probe_faithful_degradation`
     The central result. Verifies that rho[0,0] reproduces |psi_0|^2 exactly
     without noise (the ideal->noisy bridge introduces no error of its own),
     reconfirms the identity with min(1, e^Delta) against the known answer,
     and measures how that identity degrades channel by channel.

     It also detects the trap that would invalidate a whole column of the
     ablation matrix: READOUT error is a classical channel after the
     measurement and does NOT touch rho, so reading rho[0,0] is blind to it.

Usage
-----
    python noise_feasibility_probe.py            # fast probes (A, B, D)
    python noise_feasibility_probe.py --cost     # adds C (slow)

Requirements: the project's verified stack (qiskit 2.4.2, qiskit-aer 0.17.2,
numpy 1.26.4) plus `qiskit-ibm-runtime` for probe A.
"""

from __future__ import annotations

import argparse
import functools
import sys
import time
import warnings
from typing import Dict, Optional

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error

warnings.filterwarnings("ignore")
print = functools.partial(print, flush=True)  # noqa: A001


# =============================================================================
# Noise models: the four rungs of the proposed axis
# =============================================================================

def noise_none() -> Optional[NoiseModel]:
    """Rung 0 of the axis: ideal limit (the one behind all current results)."""
    return None


def noise_readout(p: float = 0.03) -> NoiseModel:
    """Rung 1: ONLY readout error, symmetric, equal on all qubits.

    It is a CLASSICAL channel applied after the measurement: it flips the
    reported bit with probability p. It does not alter the quantum state, and
    therefore it is invisible to any read of rho (see
    `probe_faithful_degradation`, part d).

    Args:
        p: flip probability of the reported bit.
    """
    nm = NoiseModel()
    nm.add_all_qubit_readout_error(ReadoutError([[1 - p, p], [p, 1 - p]]))
    return nm


def noise_full(p1: float = 1e-3, p2: float = 1e-2,
               pr: float = 0.03) -> NoiseModel:
    """Rung 2: depolarization on 1- and 2-qubit gates + readout.

    Args:
        p1: depolarization probability on single-qubit gates.
        p2: same on two-qubit gates (typically ~10x larger).
        pr: readout flip probability.
    """
    nm = noise_readout(pr)
    nm.add_all_qubit_quantum_error(depolarizing_error(p1, 1),
                                   ['ry', 'rz', 'sx', 'x', 'h', 'u'])
    nm.add_all_qubit_quantum_error(depolarizing_error(p2, 2),
                                   ['cx', 'cz', 'ecr'])
    return nm


def noise_backend(name: str = "FakeBrisbane") -> NoiseModel:
    """Rung 3: calibrated model of a real IBM backend.

    Args:
        name: `qiskit_ibm_runtime.fake_provider` class to use.
    """
    from qiskit_ibm_runtime import fake_provider
    return NoiseModel.from_backend(getattr(fake_provider, name)())


# =============================================================================
# Probe A — the shortcut hypothesis
# =============================================================================

def probe_local_testing_mode() -> None:
    """Check whether SamplerV2 runs locally, and whether DD/twirling survive.

    The hypothesis to falsify is: "pointing `qpu_cosmo_samplers.py` at a
    simulated backend instead of IBM yields the noisy run without touching
    the ideal simulator".

    What to look at in the output is not only whether it runs, but the warning
    that qiskit-ibm-runtime emits about the options it ignores.
    """
    from qiskit_ibm_runtime import Batch, Session, SamplerV2
    from qiskit_ibm_runtime.fake_provider import FakeBrisbane
    from qiskit.transpiler.preset_passmanagers import \
        generate_preset_pass_manager

    n = 3
    th = ParameterVector("t", n)
    qc = QuantumCircuit(n)
    for i in range(n):
        qc.ry(th[i], i)
    for i in range(n - 1):
        qc.cx(i, i + 1)
    qc.measure_all()

    pv = np.array([[0.1, 0.2, 0.3], [1.0, 1.1, 1.2]])
    backend = FakeBrisbane()
    pm = generate_preset_pass_manager(optimization_level=3, backend=backend)
    isa = pm.run(qc)

    for label, mode in (("mode=backend", backend),
                        ("mode=Batch", Batch(backend=backend)),
                        ("mode=Session", Session(backend=backend))):
        sampler = SamplerV2(mode=mode)
        # Exactly the options set by QPUConnection.__init__:
        opt = sampler.options
        opt.dynamical_decoupling.enable = True
        opt.dynamical_decoupling.sequence_type = "XY4"
        opt.twirling.enable_gates = True
        opt.twirling.enable_measure = True
        res = sampler.run([(isa, pv)], shots=512).result()
        data = res[0].data
        reg = getattr(data, 'meas', None) or getattr(data, 'c', None)
        c0, c1 = reg.get_counts(0), reg.get_counts(1)
        print(f"  {label:14s} OK | ISA {isa.num_qubits}q | "
              f"B=2 -> {len(c0)}/{len(c1)} outcomes | "
              f"len(bitstring)={len(next(iter(c0)))}")


# =============================================================================
# Probe B — the graded axis is buildable and distinguishable
# =============================================================================

def probe_graded_axis() -> None:
    """Check that the four rungs produce different distributions.

    Reports the total variation distance of each rung against the ideal; a
    rung giving 0.0 would be indistinguishable from the ideal and would not
    contribute a real column to the ablation matrix.
    """
    n = 4
    th = ParameterVector("t", n)
    qc = QuantumCircuit(n)
    for i in range(n):
        qc.ry(th[i], i)
    for i in range(n - 1):
        qc.cx(i, i + 1)
    for i in range(n):
        qc.ry(th[i], i)
    qc.measure_all()

    ref: Optional[Dict[str, int]] = None
    shots = 8192
    for label, nm in (("none", noise_none()), ("readout", noise_readout()),
                      ("full", noise_full()),
                      ("FakeBrisbane", noise_backend())):
        sim = AerSimulator(noise_model=nm, seed_simulator=7)
        isa = transpile(qc, sim, optimization_level=1)
        counts = sim.run(isa,
                         parameter_binds=[{p: [0.3] for p in isa.parameters}],
                         shots=shots).result().get_counts()
        if isinstance(counts, list):
            counts = counts[0]
        if ref is None:
            ref, tv = counts, 0.0
        else:
            keys = set(ref) | set(counts)
            tv = 0.5 * sum(abs(ref.get(k, 0) - counts.get(k, 0))
                           for k in keys) / shots
        print(f"  {label:14s} outcomes={len(counts):3d}  "
              f"total variation distance vs ideal = {tv:.4f}")


# =============================================================================
# Probe C — real cost
# =============================================================================

def probe_cost(widths=(6, 8, 10, 12, 13)) -> None:
    """Time per job and peak RAM: density matrix vs trajectories.

    Uses the real ansatz from `qpu_cosmo_samplers.build_ansatz` (3 layers), a
    job with B=2 bindings and 4096 shots: exactly the shape of one QVMC-QPU
    SPSA iteration.

    Args:
        widths: circuit widths (in qubits) to measure.
    """
    import gc
    import resource
    sys.path.insert(0, ".")
    from qpu_cosmo_samplers import build_ansatz

    nm = noise_backend()
    print(f"  {'n':>3} {'density(s)':>12} {'traject.(s)':>12} "
          f"{'theor. rho':>14} {'peak RSS':>10}")
    for n in widths:
        qc = build_ansatz(n, n_layers=3)
        row = {}
        for method in ("density_matrix", "statevector"):
            sim = AerSimulator(noise_model=nm, method=method,
                               seed_simulator=7)
            isa = transpile(qc, sim, optimization_level=1)
            rng = np.random.default_rng(7)
            binds = [{p: list(rng.random(2)) for p in isa.parameters}]
            t0 = time.time()
            try:
                sim.run(isa, parameter_binds=binds, shots=4096).result()
                row[method] = time.time() - t0
            except Exception as exc:                       # noqa: BLE001
                row[method] = float('nan')
                print(f"      {method} failed at n={n}: {type(exc).__name__}")
            gc.collect()
        rho_gb = (2 ** (2 * n)) * 16 / 2 ** 30
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        print(f"  {n:>3} {row['density_matrix']:>12.2f} "
              f"{row['statevector']:>12.2f} {rho_gb:>11.3f} GB "
              f"{rss:>8.0f}MB")


# =============================================================================
# Probe D — the FAITHFUL acceptance under noise
# =============================================================================

def probe_faithful_degradation() -> None:
    """Degradation curve of the RY-encoded Metropolis acceptance.

    The acceptance is encoded as A = cos^2(theta/2) with
    theta = 2*arccos(sqrt(A)), and today it is read as |psi_0|^2. Under noise
    that amplitude ceases to exist; the question is what we replace it with.

    Parts:
      (a) rho[0,0] == |psi_0|^2 without noise -> the change of readout is exact.
      (b) rho[0,0] == min(1, e^Delta) -> known answer, identity intact.
      (c) bias channel by channel. Depolarization drives rho towards I/2, so
          the bias is lambda*(1/2 - A): the acceptance is biased TOWARDS
          1/2, over-accepting bad moves and under-accepting good ones.
          It is a directional and predictable distortion, not symmetric noise.
      (d) readout error is INVISIBLE to rho but visible in counts.
    """
    deltas = np.array([-6.0, -3.0, -1.0, -0.3, -0.05, 0.0, 0.5, 2.0])
    a_exact = np.minimum(1.0, np.exp(deltas))
    thetas = 2.0 * np.arccos(np.sqrt(np.clip(a_exact, 1e-12, 1.0)))

    par = ParameterVector('t', 1)
    base = QuantumCircuit(1)
    base.ry(par[0], 0)

    def via_statevector() -> np.ndarray:
        """Current project readout: |psi_0|^2 (only exists without noise)."""
        qc = base.copy()
        qc.save_statevector()
        sim = AerSimulator(method='statevector')
        isa = transpile(qc, sim)
        res = sim.run([isa.assign_parameters({par[0]: float(t)})
                       for t in thetas]).result()
        return np.array([abs(np.asarray(res.get_statevector(k))[0]) ** 2
                         for k in range(len(thetas))])

    def via_density(nm: Optional[NoiseModel]) -> np.ndarray:
        """Proposed readout: rho[0,0]. Sees gates, does NOT see readout.

        Args:
            nm: Aer noise model.

        Returns:
            np.ndarray
        """
        qc = base.copy()
        qc.save_density_matrix()
        sim = AerSimulator(method='density_matrix', noise_model=nm)
        isa = transpile(qc, sim)
        res = sim.run([isa.assign_parameters({par[0]: float(t)})
                       for t in thetas]).result()
        return np.array([
            float(np.real(np.asarray(res.data(k)['density_matrix'])[0, 0]))
            for k in range(len(thetas))])

    def via_counts(nm: Optional[NoiseModel],
                   shots: int = 100_000) -> np.ndarray:
        """Counts-based readout: the only one that sees the readout channel.

        Args:
            nm: Aer noise model.
            shots: shots per circuit. Defaults to 100000.

        Returns:
            np.ndarray
        """
        qc = base.copy()
        qc.measure_all()
        sim = AerSimulator(noise_model=nm, seed_simulator=11)
        isa = transpile(qc, sim)
        res = sim.run([isa.assign_parameters({par[0]: float(t)})
                       for t in thetas], shots=shots).result()
        return np.array([res.get_counts(k).get('0', 0) / shots
                         for k in range(len(thetas))])

    sv, dm = via_statevector(), via_density(None)
    print("  (a) |psi_0|^2 vs rho[0,0] no noise  : max|diff| = %.3e"
          % np.max(np.abs(sv - dm)))
    print("  (b) rho[0,0]  vs min(1,e^Delta)     : max|diff| = %.3e"
          % np.max(np.abs(dm - a_exact)))
    print()
    print("  (c) degradation curve:")
    print(f"      {'channel':30s} {'mean bias':>12s} {'mean rel bias':>16s}")
    for label, nm in (("ideal", noise_none()),
                      ("readout p=0.01", noise_readout(0.01)),
                      ("readout p=0.03", noise_readout(0.03)),
                      ("readout p=0.05", noise_readout(0.05)),
                      ("gate 1e-3 + readout 0.03", noise_full(1e-3)),
                      ("gate 1e-2 + readout 0.03", noise_full(1e-2))):
        d = via_density(nm) - a_exact
        print(f"      {label:30s} {d.mean():12.5f} "
              f"{np.mean(d / a_exact):16.5f}")
    print()
    p = 0.03
    cnt, den = via_counts(noise_readout(p)), via_density(noise_readout(p))
    print("  (d) counts vs rho[0,0], same readout channel p=%.2f:" % p)
    print("      max|diff| = %.5f  |  expected shot noise ~ %.5f"
          % (np.max(np.abs(cnt - den)), 0.5 / np.sqrt(100_000)))
    print("      -> rho is BLIND to readout error; counts are not.")


# =============================================================================

def main(argv=None) -> int:
    """Run the selected probes and print the report.

    Args:
        argv: command-line arguments; None uses `sys.argv`. Defaults to
            None.

    Returns:
        int
    """
    ap = argparse.ArgumentParser(
        description="Feasibility probes for the NISQ noise axis.")
    ap.add_argument('--cost', action='store_true',
                    help='include probe C (slow)')
    args = ap.parse_args(argv)

    print("=" * 72)
    print("A. SamplerV2 local testing mode (shortcut hypothesis)")
    print("=" * 72)
    probe_local_testing_mode()

    print()
    print("=" * 72)
    print("B. Graded noise axis")
    print("=" * 72)
    probe_graded_axis()

    print()
    print("=" * 72)
    print("D. Degradation of the FAITHFUL acceptance")
    print("=" * 72)
    probe_faithful_degradation()

    if args.cost:
        print()
        print("=" * 72)
        print("C. Cost: density matrix vs trajectories")
        print("=" * 72)
        probe_cost()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
