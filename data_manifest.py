#!/usr/bin/env python3
"""
data_manifest.py — data provenance and integrity for reproducibility.

Why this exists (FASE-1 audit item 4). Reproducible computational physics
requires that every input dataset be identified by SOURCE, VERSION and a
cryptographic CHECKSUM, so a third party can confirm they are fitting the
exact same numbers. This module:

  * documents the provenance of each dataset (DATASETS below);
  * computes/verifies SHA256 checksums of the on-disk files;
  * can (re)generate `data_checksums.json` after the real data files are
    placed in the working directory.

The 51-point CC+BAO table is embedded in cosmo_core._CC_EMBEDDED as a
fallback; its provenance is documented here as well. The Pantheon /
Pantheon+ files are NOT redistributed — download them from the official
releases listed below and verify with `python data_manifest.py --verify`.

Usage:
    python data_manifest.py --generate     # write data_checksums.json
    python data_manifest.py --verify       # check files against the manifest
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

# --- provenance records -----------------------------------------------------
DATASETS = {
    "cosmic_chronometers.txt": {
        "what": "Combined Cosmic Chronometers + BAO H(z) measurements "
                "(z, H, sigma), diagonal errors.",
        "source": "Standard CC compilation (Moresco et al. and BAO H(z) "
                  "points); see thesis bibliography for the per-point "
                  "references.",
        "note": "If absent, cosmo_core._CC_EMBEDDED (51 points) is used as a "
                "documented fallback.",
        "optional": True,
    },
    "pantheon_full_parameters.txt": {
        "what": "Pantheon 2018 SNe Ia (1048), columns name zcmb zhel dz mb "
                "dmb; diagonal errors.",
        "source": "Scolnic et al. (2018), Pantheon public release.",
        "optional": True,
    },
    "Pantheon+SH0ES.dat": {
        "what": "Pantheon+ 2022 SNe Ia data table (redshift + distance "
                "modulus columns).",
        "source": "Brout et al. (2022); https://github.com/PantheonPlusSH0ES/"
                  "DataRelease",
        "optional": True,
    },
    "Pantheon+SH0ES_STAT+SYS.cov": {
        "what": "Pantheon+ 2022 full statistical+systematic covariance "
                "matrix (N then N*N entries).",
        "source": "Brout et al. (2022); same release as the .dat file.",
        "optional": True,
    },
}

MANIFEST = "data_checksums.json"


def sha256(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def generate(root: str = ".") -> dict:
    """Compute checksums for whatever data files are present; write manifest."""
    out = {}
    for name, meta in DATASETS.items():
        p = os.path.join(root, name)
        if os.path.exists(p):
            out[name] = {"sha256": sha256(p), "bytes": os.path.getsize(p),
                         **{k: meta[k] for k in ("what", "source")}}
            print(f"  hashed {name}: {out[name]['sha256'][:16]}…")
        else:
            print(f"  (absent) {name}")
    with open(os.path.join(root, MANIFEST), "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"  wrote {MANIFEST} ({len(out)} file(s))")
    return out


def verify(root: str = ".") -> int:
    """Verify present files against the manifest. Returns process exit code."""
    mpath = os.path.join(root, MANIFEST)
    if not os.path.exists(mpath):
        print(f"No {MANIFEST}; run --generate after placing the data files.")
        return 2
    manifest = json.load(open(mpath))
    rc = 0
    for name, rec in manifest.items():
        p = os.path.join(root, name)
        if not os.path.exists(p):
            print(f"  MISSING  {name}")
            rc = 1
            continue
        got = sha256(p)
        if got == rec["sha256"]:
            print(f"  OK       {name}")
        else:
            print(f"  MISMATCH {name}\n    expected {rec['sha256']}\n"
                  f"    got      {got}")
            rc = 1
    return rc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--generate", action="store_true",
                   help="compute checksums of present data files -> manifest")
    g.add_argument("--verify", action="store_true",
                   help="verify present data files against the manifest")
    ap.add_argument("--root", default=".", help="directory holding the data")
    args = ap.parse_args()
    if args.generate:
        generate(args.root)
        return 0
    return verify(args.root)


if __name__ == "__main__":
    sys.exit(main())
