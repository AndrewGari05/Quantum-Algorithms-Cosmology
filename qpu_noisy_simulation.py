#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qpu_noisy_simulation.py — Noisy twin of the QPU pipeline.
================================================================================

Runs the `qpu_cosmo_samplers.py` pipeline — the same code, the same
circuits, the same counts-based logic — against a local NOISY simulator
instead of IBM hardware.

`qpu_cosmo_samplers.py` IS NOT MODIFIED
---------------------------------------
That file is kept intact for real QPU runs. This module does not copy it: it
IMPORTS it and replaces a single piece, the connection. A copied file would
start to diverge from the original at the first fix applied to one and not
the other, and the divergence would be silent — exactly the class of bug the
project has been chasing. Here, any fix to the hardware pipeline arrives
automatically.

The only thing replaced is where the counts come from:

    QPUConnection      -> IBM Quantum (SamplerV2 on a physical backend)
    LocalNoisyConnection -> AerSimulator with NoiseModel, same interface

Everything else — `build_proposal_circuit`, `build_ansatz`, `GridEncoding`,
`kl_from_counts`, SPSA, `MCMC_QPU`, `QVMC_QPU`, the figures — is literally
the hardware code.

Why this module exists
----------------------
1. It is the FIRST end-to-end test of `qpu_cosmo_samplers.py`. Until now
   that module had only been validated in `--dry-run`, i.e. with synthetic
   uniform counts that do not exercise the physics: shapes and decoding were
   checked, not results.
2. It provides the version of the noise axis WITH SHOT NOISE. The axis that
   lives in `cosmo_modular_quantum` computes the exact probabilities from
   `diag(rho)` to isolate the degradation due to noise from sampling noise;
   here the combination of both is measured, which is what the hardware
   actually sees.

[DD-INERT] Caveat that this module makes explicit
-------------------------------------------------
`QPUConnection` enables XY4 dynamical decoupling and Pauli twirling of gates
and measurement. On real hardware those options take effect. With a
simulated backend they do NOT: qiskit-ibm-runtime discards them in local
testing mode and says so only with a `UserWarning` that gets lost among the
logs of a long run:

    UserWarning: Options {'dynamical_decoupling': ...,'twirling': ...}
    have no effect in local testing mode.

Consequence: this run measures **noise WITHOUT error suppression**, whereas
the hardware run measures **noise WITH error suppression**. They are not the
same experiment. It is a defensible reading — it gives a LOWER BOUND on the
quality achievable on hardware — but it has to be recorded, not inferred.
That is why `LocalNoisyConnection` does not use `SamplerV2` at all: it talks
to `AerSimulator` directly and REPORTS in the log that error suppression is
inert, instead of setting options it knows will be ignored.

Usage
-----
    # Synthetic rung, QMCMC:
    python qpu_noisy_simulation.py --model lcdm --method qmcmc \\
        --noise full --steps 200 --chains 4

    # Real calibrated backend, QVMC:
    python qpu_noisy_simulation.py --model wcdm --method qvmc \\
        --noise FakeBrisbane --iters 30 --nqpp 3

    # Control: same pipeline WITHOUT noise (isolates the sampling effect)
    python qpu_noisy_simulation.py --model lcdm --method qmcmc --noise none

The hardware flags (`--backend`, `--least-busy`, `--token`, `--session`)
are accepted but ignored: there is no hardware to point at here.
"""

from __future__ import annotations

import argparse
import logging
import time
from typing import Dict, List, Optional

import numpy as np

import cosmo_noise as cnoise
import qpu_cosmo_samplers as qpu

__all__ = ['LocalNoisyConnection', 'build_parser', 'main']


# =============================================================================
# The only replaced piece
# =============================================================================

class LocalNoisyConnection(qpu.QPUConnection):
    """`QPUConnection` that returns counts from a local noisy Aer.

    Implements the SAME interface as the hardware connection —
    `transpile_isa`, `run_pub`, `close`, `timer`, `shots` — so that
    `QPUProposalEngine`, `MCMC_QPU` and `QVMC_QPU` work without any change.

    It does not inherit the parent's `__init__` because that one opens a
    connection to IBM.

    Args:
        noise: rung of the axis (`cosmo_noise.NoiseSpec`).
        shots: shots per binding.
        seed: simulator seed, for reproducibility.
        logger: logger where the effective configuration is reported.
    """

    def __init__(self, noise: "cnoise.NoiseSpec", shots: int = 4096,
                 seed: Optional[int] = None,
                 logger: Optional[logging.Logger] = None):
        """Local stand-in for QPUConnection that simulates noise instead of using the QPU.

        Args:
            noise: noise specification to apply.
            shots: shots per circuit.
            seed: simulator seed.
            logger: destination for log messages.
        """
        from cosmo_core import make_simulator

        self.shots = shots
        self.dry_run = False
        self.use_session = False
        self.log = logger or logging.getLogger("qpu-noisy")
        self.timer = qpu.TimingEstimator()
        self._context = None
        self.noise = noise
        self.seed = seed
        self.pm = None                      # no physical backend: no ISA pass

        # [NOISE] counts_route=True: this pipeline's circuits MEASURE, so Aer
        # applies the readout channel by itself. Also applying it in closed
        # form would be double counting.
        kwargs = noise.simulator_kwargs(counts_route=True)
        # [E-QPU5] The seed is NOT fixed on the simulator: a fixed
        # seed_simulator made every job replay the same random stream, so
        # repeated jobs returned identical counts. Each job gets its own seed
        # derived from `seed` and a job counter (see run_pub).
        self._job = 0
        self.backend = make_simulator(**kwargs)

        self.log.info("Local simulator: method=%s | noise rung='%s' | "
                      "shots=%d", kwargs.get('method'), noise.label, shots)
        # [DD-INERT] Say it ALWAYS and in the run log, not as a lost
        # warning: it is the difference between this experiment and the
        # hardware one, and without it the comparison between them misleads.
        self.log.warning(
            "[DD-INERT] This run has NO dynamical decoupling and NO Pauli "
            "twirling. On real hardware those options do take effect, so the "
            "results here are a LOWER BOUND on the quality achievable on "
            "the QPU, not a prediction of it.")
        if noise.is_ideal:
            self.log.info("Rung 'none': no gate noise and no readout "
                          "noise. What remains is SHOT noise, which is the "
                          "control against which to compare the other "
                          "rungs.")

    # ------------------------------------------------------------------ #
    def transpile_isa(self, qc):
        """Transpile to the simulator's gate set.

        Without a physical backend there is no coupling map to respect, so
        this is just a translation to the supported basis. The original's
        transpile-once pattern is preserved: the caller invokes this once per
        template, not per circuit.

        Args:
            qc: logical circuit.

        Returns:
            Transpiled circuit.
        """
        # [E-HPC3] device-level noise: place and route on the device first
        return self.noise.transpile(qc, self.backend, optimization_level=1)

    # ------------------------------------------------------------------ #
    def run_pub(self, isa_circuit, parameter_values: np.ndarray,
                shots: Optional[int] = None) -> List[Dict[str, int]]:
        """One job with B bindings -> list of B count dictionaries.

        Reproduces the `QPUConnection.run_pub` contract exactly: same
        binding order, same count keys, same record in the
        `TimingEstimator`. That contract is what lets the rest of the
        pipeline not notice the difference.

        Args:
            isa_circuit: already-transpiled circuit.
            parameter_values: (B, n_phi) array of bindings.
            shots: shots per binding.

        Returns:
            List of B dictionaries {bitstring: frequency}.
        """
        shots = shots or self.shots
        pv = np.atleast_2d(np.asarray(parameter_values, dtype=float))
        b = int(pv.shape[0])
        params = list(isa_circuit.parameters)
        t0 = time.time()

        if params:
            if pv.shape[1] != len(params):
                raise ValueError(
                    f"run_pub: the circuit has {len(params)} parameters but "
                    f"bindings of width {pv.shape[1]} arrived.")
            # Aer expects ONE dict per circuit, with the LIST of B values per
            # parameter (not B dicts): passing it the other way fails with AerError.
            binds = [{p: list(pv[:, i]) for i, p in enumerate(params)}]
        else:
            binds = None

        run_kw = {}
        if self.seed is not None:
            run_kw['seed_simulator'] = int(self.seed) * 1_000_003 + self._job
        self._job += 1
        result = self.backend.run(isa_circuit, parameter_binds=binds,
                                  shots=shots, **run_kw).result()
        t_wall = time.time() - t0
        # No queue and no API overhead: the wall time IS the execution
        # time, so it is reported as such instead of letting the estimator
        # derive it from the hardware shot heuristic.
        self.timer.record(b, shots, t_wall, t_exec_reported=t_wall)

        counts = result.get_counts()
        if isinstance(counts, dict):
            counts = [counts]
        if len(counts) != b:
            raise RuntimeError(
                f"run_pub: expected {b} count distributions but "
                f"{len(counts)} arrived.")
        return [dict(c) for c in counts]

    # ------------------------------------------------------------------ #
    def close(self):
        """Nothing to close: there is no remote session or batch."""
        return None


# =============================================================================
# CLI
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    """Parser of the twin: the QPU pipeline's parser plus the noise axis.

    Reusing `qpu.build_parser()` guarantees that any new flag of the
    hardware pipeline shows up here without touching this file.

    Returns:
        Configured `ArgumentParser`.
    """
    p = qpu.build_parser()
    p.description = ("Noisy twin of qpu_cosmo_samplers: same counts-based "
                     "pipeline, against noisy Aer instead of IBM.")
    cnoise.add_noise_cli(p)
    p.add_argument('--noise-seed', type=int, default=None,
                   help='Aer simulator seed (reproducibility of the shot '
                        'noise). Defaults to --seed.')
    return p


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point: runs `qpu.main` with the connection replaced.

    The replacement is done by patching the name `QPUConnection` INSIDE the
    hardware module for the duration of the call, and restoring it
    afterwards. That way the real pipeline's `main` is reused in full — job
    budget, logging, figures, CSV — without duplicating a single line of its
    logic, and without leaving the hardware module altered for the rest of
    the process.

    Args:
        argv: command-line arguments.

    Returns:
        Exit code.
    """
    args = build_parser().parse_args(argv)
    spec = cnoise.spec_from_args(args)

    total_q = args.nqpp * qpu.MODELS[args.model].n_params
    if not spec.is_ideal and total_q > cnoise.MAX_NOISY_QUBITS:
        print(f"[NOISE] {args.model} with nqpp={args.nqpp} is {total_q} "
              f"qubits, above the noise axis ceiling of "
              f"{cnoise.MAX_NOISY_QUBITS}. The density matrix would take "
              f"{cnoise.noisy_density_bytes(total_q) / 2**30:.1f} GB and the "
              f"job would take hours. Lower --nqpp or use --noise none.")
        return 1

    seed = args.noise_seed if args.noise_seed is not None else args.seed

    def _factory(*_a, **kw):
        """Stand-in for QPUConnection: ignores the hardware flags."""
        return LocalNoisyConnection(noise=spec,
                                    shots=kw.get('shots', args.shots),
                                    seed=seed, logger=kw.get('logger'))

    # Forward only the flags the hardware parser knows; the noise-axis ones
    # are already captured in `spec` and `qpu.build_parser()` would reject them.
    passthrough = _strip_noise_flags(argv)

    original = qpu.QPUConnection
    qpu.QPUConnection = _factory
    try:
        return qpu.main(passthrough)
    finally:
        qpu.QPUConnection = original


def _strip_noise_flags(argv: Optional[List[str]]) -> Optional[List[str]]:
    """Remove the noise-axis flags from argv.

    `qpu.main` rebuilds its arguments with `qpu.build_parser()`, which does
    not know `--noise*`; passing them would abort with "unrecognized arguments".

    Args:
        argv: original arguments, or None to use `sys.argv[1:]`.

    Returns:
        Filtered list, or None if `argv` was None and there was nothing to filter.
    """
    import sys

    source = list(sys.argv[1:] if argv is None else argv)
    noise_flags = {'--noise', '--noise-readout-p', '--noise-gate-p1',
                   '--noise-gate-p2', '--noise-seed'}
    out: List[str] = []
    skip = False
    for token in source:
        if skip:
            skip = False
            continue
        if token in noise_flags:
            skip = True
            continue
        if any(token.startswith(f + '=') for f in noise_flags):
            continue
        out.append(token)
    return out


if __name__ == '__main__':
    raise SystemExit(main())
