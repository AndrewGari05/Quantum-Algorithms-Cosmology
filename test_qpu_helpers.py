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
