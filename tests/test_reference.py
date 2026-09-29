"""Bit-identity against the committed reference set (tests/reference/).

Slow (~5 min on 2 cores), so it only runs with ``RUN_SLOW=1``. It must pass
on every commit that does not declare ``Changes-results:`` in its message.
"""
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.mark.skipif(os.environ.get("RUN_SLOW") != "1", reason="set RUN_SLOW=1")
def test_reference_set_bit_identical():
    r = subprocess.run([sys.executable, "tests/reference/make_reference.py", "--check"],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-3000:]
