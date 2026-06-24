"""
conftest.py — shared pytest fixtures and the `qiskit` skip marker.

Tests tagged `@pytest.mark.qiskit` are skipped automatically when Qiskit/Aer
are not importable, so the pure-NumPy correctness floor (test_core.py,
test_ablation.py) always runs while the circuit-level tests run only where the
quantum stack is installed.
"""
import importlib

import pytest


def _qiskit_available() -> bool:
    try:
        importlib.import_module("qiskit")
        importlib.import_module("qiskit_aer")
        return True
    except Exception:
        return False


QISKIT_OK = _qiskit_available()


def pytest_collection_modifyitems(config, items):
    skip_q = pytest.mark.skip(reason="Qiskit/Aer not installed")
    for item in items:
        if "qiskit" in item.keywords and not QISKIT_OK:
            item.add_marker(skip_q)
