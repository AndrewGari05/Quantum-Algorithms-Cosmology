#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cosmo_noise.py — Second ablation axis: NISQ noise.
================================================================================

Single source of truth for the noise axis, in the same way that
`cosmo_core.make_simulator` is for simulator creation and
`cosmo_core.log_prob_batch` is for the physics. The three quantum modules
(`cosmo_modular_quantum`, `cosmo_genetic_optimizers` and the noisy twin of the
QPU pipeline) get their noise from here and from nowhere else.

The ablation framework becomes two-dimensional:

```
                        noise  ->
                    none   readout   full   <backend>
    quantumness 0%    .        .        .        .
          |    33%    .        .        .        .
          v    67%    .        .        .        .
              100%    .        .        .        .
```

The four rungs
--------------
* `none`      — ideal limit. It is the one behind ALL results published to
                date. Reproduces bit for bit the route that predates this module.
* `readout`   — readout error only, symmetric, uniform.
* `full`      — depolarization on 1- and 2-qubit gates + readout.
* `<backend>` — calibrated model of a real IBM backend, by name
                (e.g. `FakeBrisbane`, `fake_brisbane`).

Why readout error travels SEPARATELY from the NoiseModel
--------------------------------------------------------
Readout error is a CLASSICAL channel applied after measurement: it does not
touch the state. Aer only applies it when the circuit measures. But two of the
three readings of the ideal simulator (`hadamard_accept_log_batch` and
`_kl_batch`) are probabilities that, under noise, are obtained from `rho`
without measuring, and would therefore be BLIND to the readout channel: the
`readout` column of the ablation matrix would come out identical to the ideal
column — false, and false in the worst way, because it does not fail loudly
but produces clean, plausible numbers.

Verified empirically (`noise_feasibility_probe.py`, probe D): with
p = 0.01, 0.03 and 0.05 the value of `rho[0,0]` does not move a single digit.

The solution is not to measure, but to apply the channel in closed form. On an
n-qubit register the readout error is a tensor product of 2x2 stochastic
matrices, so it acts exactly on the probability vector `diag(rho)`:

    P_noisy = (tensor_q M_q) @ P_ideal

That gives the COMPLETE noise axis, exact and free of shot noise, at the cost
of an O(n * 2^n) contraction. It matters because the shot-based route cannot
resolve the bias we are after: with gates at 1e-3 the bias in the acceptance is
~1.4e-4, and telling it apart from sampling noise would require ~1e7 shots PER
acceptance evaluation.

That is why this module exposes TWO models per rung:

* `gate_model()`  — without readout. For the routes that read `rho` and apply
                    the readout channel with `apply_readout`.
* `full_model()`  — with readout. For the routes that really measure (the QGA,
                    the proposal engine, the QPU twin), where Aer applies it.

Using the wrong model double-counts the readout error or makes it silently
disappear, so the choice is NOT left to the caller: it is requested through
`simulator_kwargs(counts_route=...)`.

Qubit ceiling
-------------
Simulating with noise requires a density matrix (`2^(2n) * 16 B`) or
trajectories (`shots * 2^n` in time). Measured on the project's real ansatz,
the density matrix DOMINATES in time over the whole range where it fits
(1.5x-15x faster), because it evolves once and then samples, whereas
trajectories re-simulate the full circuit once per shot.

Shots do NOT buy qubits: trajectories fit in RAM but their time cost grows as
`shots * 2^n`, so they turn an OOM into a run that never finishes. The density
matrix is the good route.

[REV] How high one can go depends on the node's RAM, and that ceiling DOES
relax with more of it — the previous version of this module hard-fixed it at
13, arguing that time was the limiting factor, and that was a bad
generalization drawn from a 7 GB machine. But the cost is not a single number:
it depends on how many density matrices are alive at the same time.

    width     rho (1x)     parameter-shift batch (2*n_phi * rho)
     12       256 MB                42 GB
     13       1.0 GB               182 GB
     16        64 GB                14 TB

Quantum training in QVMC materializes `2 * n_phi` states in a single job.
Without noise that is cheap (each state is a 16*2^n statevector); with noise
each one is a density matrix and the batch becomes the dominant term. On a
95 GB node:

* QMCMC — its proposal engine uses `max(2, d)` qubits (2 to 4) and the
  acceptance just one. It is FREE under noise at any nqpp; nqpp does not touch
  its circuits.
* QVMC without quantum training (rungs 0% and 33%) — a single rho: 16 qubits.
* QVMC WITH quantum training (rungs 67% and 100%) — the batch rules: 12.
* QGA — a single rho per operator: 16 qubits.

Since a `--benchmark` walks the whole ladder, the batch rung is the one that
sets the ceiling of a samplers task. Chunking that batch (an open item already
identified in the README) is what would unlock larger resolutions.

Measured (3-layer ansatz, B=2 bindings, 4096 shots, FakeBrisbane):

    qubits   density    trajectories   theoretical rho
      6        1.66 s       5.31 s        0.1 MB
      8        2.33 s       6.77 s        1.0 MB
     10        3.77 s      22.41 s       16.0 MB
     12       34.40 s     100.85 s      256.0 MB
     13      216.53 s     333.25 s        1.0 GB
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# =============================================================================
# Axis constants
# =============================================================================

#: Named rungs. Any other value is interpreted as the name of a backend from
#: `qiskit_ibm_runtime.fake_provider`.
NAMED_LEVELS: Tuple[str, ...] = ('none', 'readout', 'full')

#: Default noisy-qubit ceiling when the node's RAM is NOT known.
#:
#: [REV] This value was originally a HARD ceiling, justified as a time limit.
#: That was a bad generalization: it was calibrated on a 7 GB machine, and on a
#: node with real RAM and no hurry the limiting factor is memory again, which
#: DOES relax with more of it. The ceiling is now derived from the available
#: RAM (`noisy_qubit_ceiling`) and this remains only as a conservative fallback
#: for when there is no RAM figure to use.
DEFAULT_NOISY_QUBITS: int = 13

#: Backward-compatible alias of the previous name.
MAX_NOISY_QUBITS: int = DEFAULT_NOISY_QUBITS

#: Layers of the variational ansatz, used to size the parameter-shift batch.
ANSATZ_LAYERS: int = 3

#: Default parameters of the synthetic rungs.
DEFAULT_READOUT_P: float = 0.03
DEFAULT_GATE_P1: float = 1e-3
DEFAULT_GATE_P2: float = 1e-2

#: Gates that get depolarization attached in the `full` rung.
_GATES_1Q = ('id', 'u', 'u1', 'u2', 'u3', 'rx', 'ry', 'rz', 'x', 'y', 'z',
             'h', 's', 'sdg', 't', 'tdg', 'sx', 'sxdg', 'p')
_GATES_2Q = ('cx', 'cz', 'cy', 'ch', 'swap', 'ecr', 'rzz', 'rxx', 'ryy')


# =============================================================================
# Name utilities
# =============================================================================

def canonical_level(level: Optional[str]) -> str:
    """Normalize the name of a noise level for logs and CSV.

    Named rungs are kept lowercase; backend names are normalized to lowercase
    without underscores, so that `FakeBrisbane`, `fake_brisbane` and
    `fakebrisbane` produce the SAME label in the CSV and axis rows are not
    duplicated.

    Args:
        level: level as the user wrote it, or None.

    Returns:
        Canonical label; `'none'` when `level` is None or empty.
    Examples:
        The three named rungs are normalized to lowercase, and anything else
        is interpreted as a backend name:

        >>> canonical_level('NONE'), canonical_level(None)
        ('none', 'none')
        >>> canonical_level('FakeBrisbane')
        'fakebrisbane'
    """
    if not level:
        return 'none'
    s = str(level).strip()
    if s.lower() in NAMED_LEVELS:
        return s.lower()
    return re.sub(r'[_\-\s]', '', s).lower()


def _resolve_fake_backend(level: str):
    """Return the fake backend whose canonical name is `level`.

    Args:
        level: canonical label (lowercase, no underscores).

    Returns:
        Instance of the fake backend.

    Raises:
        ValueError: if no fake_provider backend matches.
    """
    try:
        from qiskit_ibm_runtime import fake_provider
    except ImportError as exc:                              # pragma: no cover
        raise ValueError(
            f"Noise level '{level}' requires qiskit-ibm-runtime, which is not "
            f"installed. Use none/readout/full or install the package."
        ) from exc

    for name in dir(fake_provider):
        if not name.startswith('Fake'):
            continue
        if canonical_level(name) == level:
            return getattr(fake_provider, name)()

    available = sorted(n for n in dir(fake_provider) if n.startswith('Fake'))
    raise ValueError(
        f"Unknown noise level: '{level}'. Expected one of "
        f"{NAMED_LEVELS} or a fake_provider backend. "
        f"Available (first 10): {available[:10]}"
    )


# =============================================================================
# Noise model construction
# =============================================================================

def _readout_error(p: float):
    """Symmetric `ReadoutError` with flip probability `p`.

    Args:
        p: probability that the reported bit is flipped.
    """
    from qiskit_aer.noise import ReadoutError
    return ReadoutError([[1 - p, p], [p, 1 - p]])


def _readout_noise_model(p: float):
    """NoiseModel with ONLY symmetric, uniform readout error.

    Args:
        p: probability that the reported bit is flipped.
    """
    from qiskit_aer.noise import NoiseModel
    nm = NoiseModel()
    nm.add_all_qubit_readout_error(_readout_error(p))
    return nm


def _gate_noise_model(p1: float, p2: float):
    """NoiseModel with ONLY gate depolarization (no readout).

    Args:
        p1: depolarization probability on one-qubit gates.
        p2: same, on two-qubit gates.
    """
    from qiskit_aer.noise import NoiseModel, depolarizing_error
    nm = NoiseModel()
    if p1 > 0:
        nm.add_all_qubit_quantum_error(depolarizing_error(p1, 1),
                                       list(_GATES_1Q))
    if p2 > 0:
        nm.add_all_qubit_quantum_error(depolarizing_error(p2, 2),
                                       list(_GATES_2Q))
    return nm


def _extract_readout(noise_model
                     ) -> Tuple[Dict[int, np.ndarray], Optional[np.ndarray]]:
    """Extract the readout confusion matrices from a NoiseModel.

    It only READS: it neither rebuilds nor modifies the model. The returned
    matrices are the ones that the analytic map `NoiseSpec.apply_readout`
    applies to `diag(rho)` on the routes that do not measure.

    [B-RO] A readout error declared with `add_all_qubit_readout_error` shows up
    in `to_dict()` WITHOUT a `gate_qubits` key — it means "all qubits".
    Treating that absence as `[[0]]` degraded it to a single-qubit error: the
    analytic map noised n qubits while Aer only noised qubit 0. The symptom was
    silent — exact agreement at n=1 and growing divergence with n and with p —
    so the "no gate_qubits" case is returned as the DEFAULT matrix, not as the
    entry for qubit 0.

    [B-RECON] An earlier version also REBUILT the model without readout using
    `NoiseModel.from_dict()` to feed the `rho` route. It was removed for two
    reasons: `from_dict` has been deprecated since qiskit-aer 0.15, and the
    round trip is LOSSY for calibrated models — measured against FakeBrisbane,
    the rebuilt rho differs from the original by 6.2e-4 at 3 qubits, far above
    machine epsilon. It is not needed: on a circuit that does not measure, Aer
    does not apply the readout channel (verified to 2e-16 on the synthetic
    rungs), so the FULL model serves both routes.

    Args:
        noise_model: Aer NoiseModel.

    Returns:
        Tuple `({qubit: M}, M_default_or_None)`, where
        `M[a, b] = P(report a | true b)`.
    """
    per_qubit: Dict[int, np.ndarray] = {}
    default: Optional[np.ndarray] = None
    for entry in noise_model.to_dict().get('errors', []):
        if entry.get('type') != 'roerror':
            continue
        # to_dict gives P(report a | true b) as row b, column a; it is
        # transposed so that `M @ p` is the action on probabilities.
        probs = np.asarray(entry['probabilities'], dtype=float).T
        qubit_groups = entry.get('gate_qubits')
        if not qubit_groups:                      # [B-RO] all-qubit
            default = probs.copy()
        else:
            for qubits in qubit_groups:
                per_qubit[int(qubits[0])] = probs.copy()
    return per_qubit, default


# =============================================================================
# NoiseSpec
# =============================================================================

def _device_path(backend, n: int) -> List[int]:
    """A connected path of `n` physical qubits on `backend`, starting at 0.

    Deterministic depth-first search over the (undirected) coupling graph.
    A path is enough for every circuit in this project (chains and rings);
    the router inserts SWAPs for the ring's closing gate, as on hardware.
    """
    adj: Dict[int, set] = {}
    for a, b in backend.coupling_map.get_edges():
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    best: List[int] = []

    def dfs(path: List[int]) -> bool:
        nonlocal best
        if len(path) > len(best):
            best = list(path)
        if len(path) >= n:
            return True
        for nb in sorted(adj.get(path[-1], ())):
            if nb not in path:
                path.append(nb)
                if dfs(path):
                    return True
                path.pop()
        return False

    dfs([min(adj)])
    if len(best) < n:
        raise ValueError(f"no connected path of {n} qubits on {backend.name}")
    return best[:n]


def physical_qubits_of(result, k: int) -> Optional[List[int]]:
    """Physical qubits recorded by `NoiseSpec.transpile` for circuit `k`.

    None when the circuit was not placed on a device (ideal/synthetic levels).
    Aer returns the circuit header either as an object or as a dict.
    """
    try:
        header = result.results[k].header
        meta = (header.get('metadata') if isinstance(header, dict)
                else getattr(header, 'metadata', None))
        if isinstance(meta, dict):
            return meta.get('physical_qubits')
        return getattr(meta, 'physical_qubits', None)
    except Exception:
        return None


@dataclass
class NoiseSpec:
    """One rung of the noise axis, already resolved and ready to use.

    Build it with `NoiseSpec.from_level(...)`, never by hand: the split between
    gate noise and readout noise is what prevents double counting, and the
    constructor is what guarantees it.

    Attributes:
        label: canonical label for logs and CSV.
        source_model: the FULL NoiseModel (gates + readout), exactly as Aer
            produced it. It is the ONLY model used, on both routes: rebuilding
            a variant without readout turned out to be deprecated and also
            lossy (see [B-RECON] in `_extract_readout`).
        readout: 2x2 confusion matrices per qubit; empty if the readout
            channel is uniform (it then lives in `default_readout`).
        default_readout: confusion matrix applicable to any qubit without its
            own entry, or None. The synthetic rungs are uniform and use this
            one; real backends bring one matrix per qubit.
    """

    label: str
    source_model: Optional[object] = None
    readout: Dict[int, np.ndarray] = field(default_factory=dict)
    default_readout: Optional[np.ndarray] = None
    #: [E-PROV] Channel strengths actually used, e.g. 'readout_p=0.03'.
    #: Recorded in every CSV row so a run is self-describing.
    params: str = ''
    #: [E-HPC3] Fake-backend levels only: the device the noise model was
    #: calibrated on. Circuits are transpiled against it (coupling map, gate
    #: directions, native basis) so that the two-qubit errors apply.
    backend: Optional[object] = None

    # ------------------------------------------------------------------ #
    @classmethod
    def from_level(cls, level: Optional[str],
                   readout_p: float = DEFAULT_READOUT_P,
                   gate_p1: float = DEFAULT_GATE_P1,
                   gate_p2: float = DEFAULT_GATE_P2) -> "NoiseSpec":
        """Resolve an axis level into a usable `NoiseSpec`.

        Args:
            level: `'none'`, `'readout'`, `'full'` or a fake backend name.
            readout_p: readout probability of the synthetic rungs.
            gate_p1: 1-qubit depolarization of the `full` rung.
            gate_p2: 2-qubit depolarization of the `full` rung.

        Returns:
            The corresponding `NoiseSpec`.

        Raises:
            ValueError: if the level is not recognized.
        """
        lab = canonical_level(level)

        if lab == 'none':
            return cls(label='none')

        if lab == 'readout':
            source = _readout_noise_model(readout_p)
        elif lab == 'full':
            source = _gate_noise_model(gate_p1, gate_p2)
            source.add_all_qubit_readout_error(
                _readout_error(readout_p))
        else:
            # Real backend: the calibrated model already carries both parts.
            from qiskit_aer.noise import NoiseModel
            _fake = _resolve_fake_backend(lab)
            source = NoiseModel.from_backend(_fake)

        per_qubit, default = _extract_readout(source)
        if lab == 'readout':
            params = f"readout_p={readout_p:g}"
        elif lab == 'full':
            params = f"readout_p={readout_p:g};gate_p1={gate_p1:g};gate_p2={gate_p2:g}"
        else:
            params = f"backend={lab}"
        return cls(label=lab, source_model=source, readout=per_qubit,
                   default_readout=default, params=params,
                   backend=None if lab in ('readout', 'full') else _fake)

    # ------------------------------------------------------------------ #
    @property
    def is_ideal(self) -> bool:
        """True if this rung is the ideal limit (no noise of any kind).

        When True, EVERY route of this module must reduce exactly to the code
        that predates the noise axis: same simulation method, same readings,
        same numbers bit for bit. This is the condition that makes
        `--noise none` a valid baseline and not an approximation.
        """
        return self.source_model is None

    @property
    def has_readout(self) -> bool:
        """True if the rung includes a readout channel."""
        return bool(self.readout) or self.default_readout is not None

    # ------------------------------------------------------------------ #
    def full_model(self):
        """NoiseModel WITH readout error, for routes that really measure.

        It is the one used by the QGA, the proposal engine and the QPU twin:
        there the circuit measures and Aer applies the readout channel by
        itself, so also applying it with `apply_readout` would double count.

        Returns the ORIGINAL model without rebuilding it: rebuilding is
        exactly where the "all qubits" semantics was lost in [B-RO] and where
        the calibrated model lost precision in [B-RECON].

        Returns:
            NoiseModel, or None if the rung is ideal.
        """
        return self.source_model

    # ------------------------------------------------------------------ #
    def simulator_kwargs(self, counts_route: bool) -> Dict[str, object]:
        """kwargs for `cosmo_core.make_simulator` at this rung.

        The choice of model (with or without readout) is NOT left to the
        caller, because getting it wrong double-counts the readout error or
        makes it silently disappear; it is decided here from `counts_route`.

        The model handed over is the same on both routes; what changes is who
        applies the readout channel. With `counts_route=True` the circuit
        measures and Aer applies it. With `counts_route=False` the circuit does
        not measure, Aer ignores the readout channel by construction (verified
        to 2e-16), and the caller MUST apply it afterwards with
        `apply_readout` — otherwise the `readout` column of the axis comes out
        identical to the ideal one.

        Args:
            counts_route: True if the circuit MEASURES and the result is read
                from counts (QGA, proposal engine, QPU twin). False if the
                result is read from the state (`rho[0,0]`, `diag(rho)`).

        Returns:
            dict with `method` and, when applicable, `noise_model`.
        """
        if self.is_ideal:
            return {'method': 'statevector'}
        return {'method': 'density_matrix', 'noise_model': self.source_model}

    # ------------------------------------------------------------------ #
    def transpile(self, qc, sim, **kwargs):
        """Transpile `qc` for this noise level.

        Ideal and synthetic levels: plain `transpile(qc, sim, **kwargs)`,
        exactly as before (bit-identical results).

        [E-HPC3] Fake-backend levels: the noise model defines two-qubit
        errors only on the device's directed physical pairs. Transpiling
        against a bare simulator produced two-qubit gates on logical pairs
        such as (0, 1) that carry no error, so the "real backend" column had
        no two-qubit noise at all. Here the circuit is placed on a connected
        path of physical qubits and routed against the device target, so
        every two-qubit gate lands on a pair that has its calibrated error.
        Aer `save_*` instructions are not in the device target; they are
        removed before transpiling and re-attached to the final physical
        positions of the logical qubits, so outputs stay in logical order.
        The physical qubits are stored in `metadata['physical_qubits']`.
        """
        from qiskit import transpile as _transpile
        if self.backend is None:
            return _transpile(qc, sim, **kwargs)
        from qiskit_aer import AerSimulator
        if getattr(self, '_target_sim', None) is None:
            object.__setattr__(self, '_target_sim',
                               AerSimulator.from_backend(self.backend))
        saves = [ci for ci in qc.data if ci.operation.name.startswith('save_')]
        core = qc.copy_empty_like()
        for ci in qc.data:
            if not ci.operation.name.startswith('save_'):
                core.append(ci)
        layout = _device_path(self.backend, qc.num_qubits)
        t = _transpile(core, self._target_sim, initial_layout=layout,
                       optimization_level=1, seed_transpiler=0)
        final = t.layout.final_index_layout()
        for ci in saves:
            idx = [final[qc.find_bit(q).index] for q in ci.qubits]
            t.append(ci.operation, [t.qubits[i] for i in idx])
        t.metadata = dict(t.metadata or {}, physical_qubits=list(final))
        return t

    # ------------------------------------------------------------------ #
    def readout_matrix(self, qubit: int) -> Optional[np.ndarray]:
        """2x2 confusion matrix of the given qubit, or None if no readout.

        Args:
            qubit: qubit index.

        Returns:
            `M` with `M[a, b] = P(report a | true b)`, or None.
        """
        if qubit in self.readout:
            return self.readout[qubit]
        return self.default_readout

    # ------------------------------------------------------------------ #
    def apply_readout(self, probs: np.ndarray, n_qubits: int,
                      physical_qubits: Optional[List[int]] = None) -> np.ndarray:
        """Apply the readout channel in closed form to a probability vector.

        On n qubits the readout error is a tensor product of 2x2 stochastic
        maps, so it acts EXACTLY on `diag(rho)` without needing to measure and
        without introducing shot noise. The contraction is done qubit by qubit
        in O(n * 2^n) instead of building the 2^n x 2^n matrix.

        Bit convention: Qiskit is little-endian, the bit of qubit q in index i
        is `(i >> q) & 1`. With `reshape([2]*n)` in C order, axis 0
        corresponds to the MOST significant bit, i.e. qubit n-1; hence the
        mapping `axis = n - 1 - q`.

        Args:
            probs: probability vector of length `2**n_qubits`, or a batch of
                shape `(B, 2**n_qubits)`.
            n_qubits: number of qubits in the register.

        Returns:
            Probability vector (or batch) after the noisy readout. If the rung
            has no readout channel, returns `probs` without copying.
        """
        if not self.has_readout:
            return probs

        p = np.asarray(probs, dtype=float)
        batched = (p.ndim == 2)
        if not batched:
            p = p[None, :]
        b = p.shape[0]
        if p.shape[1] != 2 ** n_qubits:
            raise ValueError(
                f"apply_readout: expected {2 ** n_qubits} probabilities "
                f"for {n_qubits} qubits, got {p.shape[1]}.")

        out = p.reshape((b,) + (2,) * n_qubits)
        for q in range(n_qubits):
            # [E-HPC3] logical qubit q sits on physical_qubits[q] on a device
            m = self.readout_matrix(physical_qubits[q] if physical_qubits
                                    else q)
            if m is None:
                continue
            axis = n_qubits - q                      # +1 for the batch axis
            out = np.moveaxis(out, axis, -1)
            out = out @ m.T
            out = np.moveaxis(out, -1, axis)
        out = out.reshape(b, 2 ** n_qubits)
        return out if batched else out[0]

    # ------------------------------------------------------------------ #
    def metadata(self) -> Dict[str, object]:
        """Rung metadata for the CSV and the figures.

        Returns:
            dict with the label, whether it is ideal, and the effective parameters.
        """
        return {
            'noise': self.label,
            'noise_ideal': self.is_ideal,
            'noise_has_readout': self.has_readout,
            'noise_has_gates': self._has_gate_errors(),
            'noise_params': self.params,
        }

    def _has_gate_errors(self) -> bool:
        """True if the source model has errors on any instruction but measure.

        [E-HPC17] Replaces the `gate_model` attribute removed in [B-RECON];
        `metadata()` and `repr()` raised AttributeError since then.
        """
        if self.source_model is None:
            return False
        return bool(set(self.source_model.noise_instructions) - {'measure'})

    def __repr__(self) -> str:                              # pragma: no cover
        """Short representation for debugging."""
        return (f"NoiseSpec(label={self.label!r}, "
                f"gates={self._has_gate_errors()}, "
                f"readout={self.has_readout})")


# =============================================================================
# Qubit ceiling
# =============================================================================

def ansatz_n_params(n_qubits: int, n_layers: int = ANSATZ_LAYERS) -> int:
    """Number of angles of the `n_qubits` variational ansatz.

    It is measured on the REAL circuit when qiskit is available, so that it
    cannot drift out of sync if the ansatz changes.

    [B-PLAN] With a closed-form fallback when qiskit can NOT be imported.
    Building the circuit drags qiskit into `cosmo_hpc_runner`, which until then
    only planned tasks and launched subprocesses without needing the quantum
    stack. On an HPC that matters: planning and submission happen on a login
    node where the compute environment is not loaded, and the runner died with
    ModuleNotFoundError before writing a single task.

    The formula `n * (2*L + 1)` follows from the structure of the ansatz — L
    layers of (RY, RZ) per qubit plus a final RY layer — and is verified exact
    against the real circuit for L = 1..5 and n = 2..18 by
    `test_noise_axis.py`, so the fallback cannot silently diverge from the
    ansatz that actually runs.

    Args:
        n_qubits: circuit width.
        n_layers: ansatz layers.

    Returns:
        Number of free parameters.
    """
    n, layers = int(n_qubits), int(n_layers)
    try:
        from qpu_cosmo_samplers import build_ansatz
        return int(build_ansatz(n, layers).num_parameters)
    except Exception:                              # [B-PLAN] node without qiskit
        return n * (2 * layers + 1)


def param_shift_batch_factor(n_qubits: int,
                             n_layers: int = ANSATZ_LAYERS) -> int:
    """Memory multiplier of the parameter-shift batch: 2 * n_phi.

    Quantum training in QVMC evaluates the exact gradient by materializing
    `2 * n_phi` states in a single Aer job. Without noise that is cheap because
    each state is a statevector (16 * 2^n); WITH noise each one is a density
    matrix (16 * 4^n) and the multiplier becomes the dominant term:

        n=12 -> rho 256 MB, batch  42 GB
        n=13 -> rho 1.0 GB, batch 182 GB

    That is, on a 95 GB node noisy QVMC with quantum training tops out at 12
    qubits, not at 13 or 16: what rules is the BATCH, not rho. Chunking that
    batch (an open item already identified in the README) is what would unlock
    larger resolutions.

    Args:
        n_qubits: circuit width.
        n_layers: ansatz layers.

    Returns:
        `2 * n_phi`.
    Examples:
        This is the number that sets the qubit ceiling of noisy QVMC: quantum
        training materializes `2 * n_phi` density matrices at once, not one.

        >>> param_shift_batch_factor(4)
        56
        >>> param_shift_batch_factor(12)
        168
    """
    return 2 * ansatz_n_params(n_qubits, n_layers)


def noisy_density_bytes(n_qubits: int, batch_factor: int = 1) -> int:
    """Bytes of an `n_qubits` density matrix (16 B complex).

    Args:
        n_qubits: circuit width.
        batch_factor: how many density matrices are alive at once (1 for a
            single evaluation; `param_shift_batch_factor(n)` for the QVMC
            quantum-training batch).
    Examples:
        A single density matrix at 12 qubits is 256 MB...

        >>> noisy_density_bytes(12) / 1e6
        268.435456

        ...but the parameter-shift batch at that width is 45 GB, which is why
        the ceiling of noisy QVMC is far below the one the RAM allows:

        >>> round(noisy_density_bytes(12, param_shift_batch_factor(12)) / 1e9, 1)
        45.1
    """
    return (2 ** (2 * int(n_qubits))) * 16 * int(batch_factor)


def noisy_qubits_fitting_in(mem_mb: float, quantum_training: bool = True,
                            n_layers: int = ANSATZ_LAYERS) -> int:
    """Largest width whose noisy cost fits in `mem_mb`.

    Args:
        mem_mb: memory available for the task, in MB.
        quantum_training: if True (default) reserves room for the
            parameter-shift batch, which is the most expensive rung of the
            QVMC ladder and therefore the one that rules in a full
            `--benchmark`. Set it to False only if the run does not include
            rungs with quantum training.
        n_layers: ansatz layers.

    Returns:
        Number of qubits, 0 if not even the smallest case fits.
    """
    best = 0
    for n in range(1, 31):
        factor = (param_shift_batch_factor(n, n_layers)
                  if quantum_training else 1)
        if noisy_density_bytes(n, factor) / 1e6 <= mem_mb:
            best = n
        else:
            break
    return best


# ── [B-TIME] TIME ceiling of the noisy genetic optimizer ────────────────────
#
# Memory is NOT the limiting factor of the noisy QGA: a single rho at 14 qubits
# is 4.3 GB, which fits easily on any useful node. The limiting factor is time,
# and it grows as ~4^n because every evaluation propagates a density matrix
# with 4^n elements.
#
# Measured in the 2026-09-03 campaign (pop=500, readout noise, 63 GiB node,
# read from the logs by timestamp between generations):
#
#     10 qubits (n_bits=5, d=2)  ->     73 s/generation
#     12 qubits (n_bits=6, d=2)  ->   1425 s/generation   (23.8 min)
#     14 qubits (n_bits=7, d=2)  ->  had not logged even generation 0
#                                    after 7 h of wall clock
#
# The first two points fix a growth of x4.42 per qubit, which extrapolated to
# 14 gives 27810 s/generation (7.7 h) — consistent with the third observation,
# which thus serves as validation rather than as a fit point.
#
# At that pace, 500 generations at 14 qubits take ~4 MONTHS. That campaign's
# plan accepted them because the ceiling was derived from RAM alone, and the
# task hung, also blocking the summary figures of the whole run (the runner
# does not write them until the entire list finishes).
#
# This does NOT reintroduce the hard constant removed in [REV]. That ceiling
# was bad for two distinct reasons: it was calibrated on a 7 GB machine, and it
# was applied to the SAMPLERS, where what really rules is the parameter-shift
# batch, i.e. memory. Here the limit is time — which does not relax with more
# RAM — it is calibrated with measurements from the real node, and it is not a
# constant: it follows from a per-task wall-clock budget set by the user.
GENETIC_NOISY_REF_QUBITS = 10
GENETIC_NOISY_REF_SEC_PER_GEN = 73.0
GENETIC_NOISY_GROWTH_PER_QUBIT = 4.42

#: Wall-clock budget per noisy genetic task, in hours. It is the default of
#: `--noisy-task-hours`; two days lets 12 qubits through and cuts 13.
DEFAULT_NOISY_TASK_HOURS = 48.0


def genetic_noisy_seconds_per_gen(n_qubits: int) -> float:
    """Seconds per generation of the noisy QGA, at `n_qubits`.

    Extrapolation of the two measurements from the 2026-09-03 campaign (see
    the [B-TIME] block above). It is an order of magnitude, not a promise: it
    serves to decide whether a cell takes hours or months, which is the only
    question that needs answering when planning.

    SCOPE WARNING: the measurements are with `--noise readout`, which is the
    cheapest channel — it is applied analytically on diag(rho). The `full`
    levels and the calibrated backends (FakeBrisbane) add gate errors, i.e.
    more Kraus operators per gate, and run SLOWER than this function
    predicts. With those levels in the sweep, lower `--noisy-task-hours` to
    compensate, or expect the real wall clock to exceed the budget.

    Args:
        n_qubits: circuit width in qubits.

    Returns:
        float
    Examples:
        Reproduces the two measurements from the 2026-09-03 campaign...

        >>> round(genetic_noisy_seconds_per_gen(10))
        73
        >>> round(genetic_noisy_seconds_per_gen(12))
        1426

        ...and extrapolates to 14 qubits the value that motivated [B-TIME]:
        7.7 h PER generation, i.e. ~4 months for the 500 that were requested.

        >>> round(genetic_noisy_seconds_per_gen(14) / 3600, 1)
        7.7
    """
    d = int(n_qubits) - GENETIC_NOISY_REF_QUBITS
    return GENETIC_NOISY_REF_SEC_PER_GEN * (GENETIC_NOISY_GROWTH_PER_QUBIT ** d)


def genetic_noisy_time_ceiling(generations: int,
                               budget_hours: float = DEFAULT_NOISY_TASK_HOURS
                               ) -> int:
    """Largest width whose noisy QGA fits in `budget_hours` of wall clock.

    Args:
        generations: generations to be run.
        budget_hours: wall-clock budget per task, in hours.

    Returns:
        Qubit ceiling; at least 1, so that an absurd budget does not produce
        an empty plan with no explanation.
    Examples:
        With 120 generations and a two-day budget, 12 qubits fit; with the
        500 of the old campaign, only 11:

        >>> genetic_noisy_time_ceiling(120, 48)
        12
        >>> genetic_noisy_time_ceiling(500, 48)
        11

        It is not a constant: a larger budget grants more width.

        >>> genetic_noisy_time_ceiling(120, 480)
        13
    """
    budget_s = max(float(budget_hours), 0.0) * 3600.0
    g = max(int(generations), 1)
    best = 1
    for n in range(1, 31):
        if genetic_noisy_seconds_per_gen(n) * g <= budget_s:
            best = n
        else:
            break
    return best


def noisy_qubit_ceiling(requested: Optional[int] = None,
                        mem_mb: Optional[float] = None,
                        quantum_training: bool = True,
                        generations: Optional[int] = None,
                        budget_hours: float = DEFAULT_NOISY_TASK_HOURS) -> int:
    """Admissible qubit ceiling for a noisy task.

    [REV] It used to return a hard constant, arguing that time was the
    limiting factor. That was a bad generalization, calibrated on a small
    machine: on a node with real RAM and no time pressure the limiting factor
    is memory again, and that DOES relax with more of it. The ceiling is now
    derived from `mem_mb` just like the runner's other two models, and the
    constant remains as a fallback for when there is no RAM figure.

    The noisy cost is not a single number: it depends on how many density
    matrices are alive at once. With quantum training the parameter-shift
    batch multiplies by `2 * n_phi` and is the term that rules.

    Args:
        requested: cap requested by the user, or None. Acts as an upper
            bound; never grants more than what fits in memory.
        mem_mb: memory available per task. None uses `DEFAULT_NOISY_QUBITS`.
        quantum_training: reserve room for the parameter-shift batch.
            False is the genetic route, where the [B-TIME] time ceiling is
            also applied.
        generations: planned generations. Only used with
            `quantum_training=False`; without it the time ceiling is not
            applied.
        budget_hours: wall-clock budget per task for that ceiling.

    Returns:
        Effective qubit ceiling.
    """
    if mem_mb is None:
        ceiling = DEFAULT_NOISY_QUBITS
    else:
        ceiling = noisy_qubits_fitting_in(mem_mb, quantum_training)
    # [B-TIME] The noisy QGA is held back by the clock, not by memory: at 14
    # qubits 4.3 GB of rho fits on any node, but it is ~7.7 h per
    # generation. Without this ceiling the plan accepts months-long cells.
    if not quantum_training and generations is not None:
        ceiling = min(ceiling,
                      genetic_noisy_time_ceiling(generations, budget_hours))
    if requested is not None:
        ceiling = min(int(requested), ceiling)
    return ceiling


def add_noise_cli(parser) -> None:
    """Add the `--noise` flag (and its parameters) to an ArgumentParser.

    It is centralized here so that the three executables and the HPC runner
    offer exactly the same interface and the same defaults.

    Args:
        parser: `argparse.ArgumentParser` to add the group to.
    """
    g = parser.add_argument_group('NISQ noise axis')
    g.add_argument('--noise', type=str, default='none',
                   help="noise level: none | readout | full | "
                        "<backend> (e.g. FakeBrisbane). Default 'none', "
                        "which reproduces the ideal results bit for bit.")
    g.add_argument('--noise-readout-p', type=float, default=DEFAULT_READOUT_P,
                   help=f"readout flip probability of the synthetic rungs "
                        f"(default {DEFAULT_READOUT_P})")
    g.add_argument('--noise-gate-p1', type=float, default=DEFAULT_GATE_P1,
                   help=f"1-qubit depolarization in 'full' "
                        f"(default {DEFAULT_GATE_P1})")
    g.add_argument('--noise-gate-p2', type=float, default=DEFAULT_GATE_P2,
                   help=f"2-qubit depolarization in 'full' "
                        f"(default {DEFAULT_GATE_P2})")


def spec_from_args(args) -> NoiseSpec:
    """Build the `NoiseSpec` from the args of `add_noise_cli`.

    Args:
        args: argparse namespace.

    Returns:
        The resolved `NoiseSpec`.
    """
    return NoiseSpec.from_level(
        getattr(args, 'noise', 'none'),
        readout_p=getattr(args, 'noise_readout_p', DEFAULT_READOUT_P),
        gate_p1=getattr(args, 'noise_gate_p1', DEFAULT_GATE_P1),
        gate_p2=getattr(args, 'noise_gate_p2', DEFAULT_GATE_P2))
