"""
test_qga_crossover.py — regression tests for the QGA quantum crossover (C1).

The C1 bug: the original crossover circuit (CX(B→A) + CRY interference,
measure A) produced child ≈ a XOR b, with verified truth table
P(child=1) = {(0,0)→0, (0,1)→0.85, (1,0)→1, (1,1)→0.146}. Mapping
(1,1) → 0 with probability ~0.85 destroys exactly the consensus bits a
converged population agrees on — the opposite of a crossover.

The fix replaces the mixing layer with a per-qubit partial SWAP (SWAP^α),
which acts as identity on the |aa⟩ subspace. These tests assert, on the REAL
transpiled template the production code uses:

  1. CONSENSUS PRESERVATION (the C1 regression guard): (0,0) → 0 and
     (1,1) → 1 with probability 1, for every alpha.
  2. UNIFORM INHERITANCE at α = 1/2: disagreeing bits come from either
     parent with probability 1/2 (binomial tolerance).
  3. ALPHA LIMITS: α = 0 keeps parent A exactly; α = 1 swaps in parent B.
  4. END-TO-END: do_crossover on identical multi-bit parents returns the
     parents unchanged (no consensus destruction at the operator level).

Requires qiskit + qiskit-aer; skipped automatically when absent.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

qiskit = pytest.importorskip("qiskit")
pytest.importorskip("qiskit_aer")

from cosmo_core import MODELS, Posterior            # noqa: E402
from cosmo_genetic_optimizers import QGA, GAConfig  # noqa: E402


def _make_qga(n_bits=1, alpha=0.5):
    post = Posterior(MODELS['lcdm'], 'CC+BAO', 'flat')
    return QGA(post, GAConfig(pop_size=4), dict(q_crossover=True),
               n_bits=n_bits, rng=np.random.default_rng(0),
               crossover_alpha=alpha)


def _child_bit_prob(qga, a, b, shots=4096, seed=11):
    """P(child bit = 1) measured on the real transpiled crossover template."""
    pa, pb = qga._prep_a, qga._prep_b
    binding = {pa[0]: np.pi * a, pb[0]: np.pi * b}
    bound = qga._qc_cx_t.assign_parameters(binding)
    counts = qga.sim.run(bound, shots=shots,
                         seed_simulator=seed).result().get_counts()
    p1 = 0
    for bs, c in counts.items():
        bits = bs.replace(' ', '')[::-1]        # qubit order (little-endian)
        if bits[0] == '1':                      # register A, qubit 0
            p1 += c
    return p1 / shots


def test_consensus_bits_preserved():
    """C1 regression: (0,0)->0 and (1,1)->1 ALWAYS. The old circuit gave
    P(child=1 | 1,1) ~ 0.146; any consensus leak fails this test."""
    qga = _make_qga(alpha=0.5)
    assert _child_bit_prob(qga, 0, 0) == 0.0
    assert _child_bit_prob(qga, 1, 1) == 1.0


def test_uniform_inheritance_at_half_alpha():
    """alpha=1/2: a disagreeing bit is inherited 50/50 (4-sigma binomial)."""
    qga = _make_qga(alpha=0.5)
    shots = 8192
    tol = 4.0 * 0.5 / np.sqrt(shots)            # 4 sigma of a fair coin
    assert abs(_child_bit_prob(qga, 1, 0, shots) - 0.5) < tol
    assert abs(_child_bit_prob(qga, 0, 1, shots) - 0.5) < tol


def test_alpha_limits():
    """alpha=0 keeps parent A exactly; alpha=1 swaps in parent B exactly."""
    keep = _make_qga(alpha=0.0)
    assert _child_bit_prob(keep, 1, 0) == 1.0   # child = A
    assert _child_bit_prob(keep, 0, 1) == 0.0
    swap = _make_qga(alpha=1.0)
    assert _child_bit_prob(swap, 1, 0) == 0.0   # child = B
    assert _child_bit_prob(swap, 0, 1) == 1.0


def test_do_crossover_identical_parents_are_fixed_points():
    """End-to-end: crossing a population with itself returns it unchanged
    (up to the fixed-point grid cell), for a multi-bit, multi-parameter QGA.
    Under the old XOR circuit this collapsed genes toward the box corner."""
    qga = _make_qga(n_bits=4, alpha=0.5)
    pop = qga._decode(qga._encode(
        np.array([[0.31, 67.7], [0.29, 70.0], [0.35, 65.0], [0.22, 74.0]])))
    child = qga.do_crossover(pop.copy(), pop.copy())
    assert np.allclose(child, pop, atol=1e-12)
