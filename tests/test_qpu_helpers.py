"""
test_qpu_helpers.py — pure-Python tests for qpu_cosmo_samplers helpers.

These cover the parts of the QPU module that do NOT need qiskit-ibm-runtime or
a real backend: the execution-time extraction (B2) and the Metropolis
acceptance rule (B1). We extract the helper sources directly so importing the
full module (which pulls qiskit) is unnecessary.

The hardware-driven paths (QPUConnection.run_pub on a real backend, SPSA
training on hardware) cannot be unit-tested without a QPU and are validated
manually on a real run; see PHASE2_NOTES.
"""
import os
import sys
from datetime import datetime, timedelta
from typing import Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_func(name: str, start_marker: str, end_marker: str):
    import numpy as np
    src = open(os.path.join(ROOT, 'qpu_cosmo_samplers.py')).read()
    block = src[src.index(start_marker):src.index(end_marker)]
    ns = {'Optional': Optional, 'np': np}
    exec(block, ns)
    return ns[name]


# --- B2: execution-time extraction ------------------------------------------

_extract = _load_func('_extract_exec_seconds',
                      'def _extract_exec_seconds', 'class QPUConnection')


class _Span:
    def __init__(self, start, stop):
        self.start, self.stop = start, stop


class _SpanDur:
    def __init__(self, d):
        self.duration = d


class _Result:
    def __init__(self, meta):
        self.metadata = meta


def test_b2_spans_start_stop():
    t0 = datetime(2026, 1, 1)
    spans = [_Span(t0, t0 + timedelta(seconds=0.4)),
             _Span(t0, t0 + timedelta(seconds=0.4))]
    r = _Result({'execution': {'execution_spans': spans}})
    assert abs(_extract(r) - 0.8) < 1e-9


def test_b2_spans_duration_attr():
    r = _Result({'execution': {'execution_spans': [_SpanDur(0.3),
                                                   _SpanDur(0.5)]}})
    assert abs(_extract(r) - 0.8) < 1e-9


def test_b2_scalar_execution_time():
    r = _Result({'execution': {'execution_time': 1.25}})
    assert abs(_extract(r) - 1.25) < 1e-9


def test_b2_unparseable_returns_none():
    # The OLD code crashed on a list here; the new code returns None so the
    # caller falls back to an explicit estimate.
    assert _extract(_Result({'execution': {'execution_spans': [1, 2, 3]}})) is None
    assert _extract(_Result({'execution': {'execution_spans': 0}})) is None
    assert _extract(_Result({})) is None


# --- B1: Metropolis acceptance rule -----------------------------------------

_metro = _load_func('metropolis_log_accept',
                    'def metropolis_log_accept', 'class GridEncoding')


def test_b1_metropolis_rule():
    import math
    # log min(1, e^Δ): 0 for Δ>=0, Δ for Δ<0
    assert _metro(0.0, 1.0) == 0.0            # better proposal -> accept w.p. 1
    assert abs(_metro(0.0, -2.0) - (-2.0)) < 1e-12
    assert _metro(0.0, float('-inf')) == float('-inf')   # out-of-box reject
    # monotonic increasing in Δ (the inverted-acceptance regression guard)
    deltas = [-5, -2, -1, 0, 1, 2]
    vals = [_metro(0.0, d) for d in deltas]
    assert all(b >= a - 1e-12 for a, b in zip(vals, vals[1:]))


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {fn.__name__}: {e}")
        except Exception as e:
            failed += 1
            print(f"ERROR {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)


# ─────────────────────────────────────────────────────────────────────────────
# [B-CREG] Parsing the REAL SamplerV2 result
# ─────────────────────────────────────────────────────────────────────────────
#
# This is the surface that the simulated twin (`qpu_noisy_simulation.py`) does
# NOT cover: it replaces `QPUConnection` entirely, so the code that reads the
# `PubResult` back was never executed. Here it is genuinely exercised, using
# SamplerV2 in local mode against FakeBrisbane — which returns a genuine
# PubResult, with the same shape as the hardware one.

import numpy as np
import pytest


def _local_connection(shots=256):
    """Minimal QPUConnection pointing to SamplerV2 on a fake backend."""
    import logging
    q = pytest.importorskip('qpu_cosmo_samplers')
    fp = pytest.importorskip('qiskit_ibm_runtime.fake_provider')
    rt = pytest.importorskip('qiskit_ibm_runtime')
    q._import_runtime()
    be = fp.FakeBrisbane()
    conn = q.QPUConnection.__new__(q.QPUConnection)
    conn.sampler = rt.SamplerV2(mode=be)
    conn.dry_run = False
    conn.shots = shots
    conn.log = logging.getLogger('test-qpu')
    conn.timer = q.TimingEstimator()
    return conn, be


def _pm(be):
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    return generate_preset_pass_manager(optimization_level=1, backend=be)


@pytest.mark.qiskit
def test_b_creg_counts_are_read_from_a_real_pubresult():
    """[B-CREG] The real SamplerV2 result is parsed without crashing.

    With the creg named 'meas' (the one used by the project's circuits) and
    also with one with a DIFFERENT name, where the previous expression
    `getattr(data,'meas',None) or getattr(data,'c',None)` returned None and
    crashed with an opaque AttributeError — AFTER the job had been executed
    and billed on the QPU.
    """
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector, ClassicalRegister
    conn, be = _local_connection()
    pm = _pm(be)
    for name in ('meas', 'readout'):
        qc = QuantumCircuit(3)
        p = ParameterVector('p', 3)
        for i in range(3):
            qc.ry(p[i], i)
        if name == 'meas':
            qc.measure_all()
        else:
            cr = ClassicalRegister(3, 'readout')
            qc.add_register(cr)
            qc.measure(range(3), cr)
        outs = conn.run_pub(pm.run(qc),
                            np.array([[0.1, 0.2, 0.3], [2.9, 3.0, 3.1]]),
                            shots=256)
        assert len(outs) == 2, f"creg {name!r}: expected 2 batches"
        for o in outs:
            assert sum(o.values()) == 256


@pytest.mark.qiskit
def test_b_creg_batch_k_matches_row_k():
    """[B-CREG] Results arrive IN THE ORDER of the submitted parameters.

    If the order were not preserved, each point of the chain would be
    evaluated with another point's parameters and the whole run would be
    garbage without failing. We use ry(0) -> |0> and ry(pi) -> |1>, which
    leaves an unmistakable signature per row.
    """
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector
    conn, be = _local_connection(512)
    qc = QuantumCircuit(2)
    p = ParameterVector('p', 2)
    qc.ry(p[0], 0)
    qc.ry(p[1], 1)
    qc.measure_all()
    vals = np.array([[0, 0], [np.pi, 0], [0, np.pi], [np.pi, np.pi]], float)
    expected = ['00', '01', '10', '11']
    outs = conn.run_pub(_pm(be).run(qc), vals, shots=512)
    for k, (o, e) in enumerate(zip(outs, expected)):
        dominant = max(o, key=o.get)
        assert dominant == e, (
            f"row {k}: got {dominant!r}, expected {e!r} — the "
            f"results do NOT arrive in the order of the parameters")
