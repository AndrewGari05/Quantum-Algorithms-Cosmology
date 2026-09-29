"""Observational datasets.

=================  ======  ==========================================================
name               points  description
=================  ======  ==========================================================
``CC+BAO``         51      H(z) compilation of Magaña et al. (2018): 31 cosmic
                           chronometers + 20 BAO H(z) values, diagonal errors.
``CC``             31      The cosmic-chronometer rows of the same table.
``Pantheon``       1048    Pantheon SNe Ia (Scolnic et al. 2018), statistical errors
                           only, absolute magnitude marginalized analytically.
``Pantheon+sys``   1048    Same, with the systematic covariance ``sys_full_long``
                           (downloaded on first use and checked against its SHA-256).
=================  ======  ==========================================================

Combinations are written with ``+`` (``"CC+BAO+Pantheon"``) or passed as a
list. Caveats: the 20 BAO rows of ``CC+BAO`` come from overlapping survey
volumes but are treated as independent, and BAO-derived H(z) values assume a
fiducial sound horizon; ``Pantheon`` omits systematics. Both choices follow
common practice and shrink the error bars; use ``CC`` or ``Pantheon+sys`` to
check their effect.
"""
from __future__ import annotations

import hashlib
import os
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from importlib import resources

import numpy as np

__all__ = ["HzData", "SNData", "load_hz", "load_pantheon", "resolve", "DATASETS"]

#: Redshifts of the BAO-derived rows of the 51-point compilation.
_BAO_Z = (0.24, 0.30, 0.31, 0.35, 0.36, 0.38, 0.43, 0.44, 0.51, 0.52, 0.56, 0.57,
          0.59, 0.60, 0.61, 0.64, 0.73, 2.33, 2.34, 2.36)
_SYS_URL = "https://raw.githubusercontent.com/dscolnic/Pantheon/master/sys_full_long.txt"
_SYS_SHA256 = "0ec3388b984a708f27bcedf7171c8a3e74621aca73dabb41a21246e9ae3fb53d"

DATASETS = ("CC+BAO", "CC", "Pantheon", "Pantheon+sys")
_ALIASES = {"CC+BAO+Pantheon": ["CC+BAO", "Pantheon"],
            "CC+BAO+Pantheon+sys": ["CC+BAO", "Pantheon+sys"],
            "CC+Pantheon": ["CC", "Pantheon"],
            "CC+Pantheon+sys": ["CC", "Pantheon+sys"]}


@dataclass(frozen=True)
class HzData:
    """H(z) measurements with independent Gaussian errors."""

    name: str
    z: np.ndarray
    H: np.ndarray
    sigma: np.ndarray
    reference: str

    def __len__(self) -> int:
        return len(self.z)


@dataclass(frozen=True)
class SNData:
    """Supernova apparent magnitudes; ``cov`` is ``None`` for diagonal errors."""

    name: str
    z: np.ndarray
    mb: np.ndarray
    dmb: np.ndarray
    cov: np.ndarray | None
    reference: str

    def __len__(self) -> int:
        return len(self.z)


def _data_path(name: str) -> str:
    return str(resources.files("qablate.cosmology") / "data" / name)


def load_hz(which: str = "CC+BAO") -> HzData:
    """Load ``'CC+BAO'`` (51 points) or ``'CC'`` (31 points).

    Examples:
        >>> len(load_hz("CC+BAO")), len(load_hz("CC"))
        (51, 31)
    """
    t = np.loadtxt(_data_path("hz_compilation_51.txt"), comments="#")
    ref = "Magaña, Amante, García-Aspeitia & Motta 2018 (compilation)"
    if which == "CC+BAO":
        return HzData("CC+BAO", t[:, 0], t[:, 1], t[:, 2], ref)
    if which == "CC":
        bao = np.array([np.any(np.isclose(z, _BAO_Z, atol=1e-6)) for z in t[:, 0]])
        return HzData("CC", t[~bao, 0], t[~bao, 1], t[~bao, 2], ref + ", cosmic-chronometer rows")
    raise ValueError(f"unknown H(z) dataset {which!r}")


def _cache_dir() -> str:
    d = os.environ.get("QABLATE_CACHE", os.path.join(os.path.expanduser("~"), ".cache", "qablate"))
    os.makedirs(d, exist_ok=True)
    return d


def _fetch_sys(path: str | None = None) -> np.ndarray:
    path = path or os.path.join(_cache_dir(), "pantheon2018_sys_full_long.txt")
    if not os.path.exists(path):
        urllib.request.urlretrieve(_SYS_URL, path)
    with open(path, "rb") as fh:
        digest = hashlib.sha256(fh.read()).hexdigest()
    if digest != _SYS_SHA256:
        raise ValueError(f"checksum mismatch for {path}: {digest}")
    v = np.loadtxt(path)
    n = int(v[0])
    return v[1:].reshape(n, n)


def load_pantheon(systematics: bool = False, sys_path: str | None = None) -> SNData:
    """Pantheon 2018 (1048 SNe Ia), sorted by redshift.

    Args:
        systematics: Add the systematic covariance (downloaded once to
            ``~/.cache/qablate`` or ``$QABLATE_CACHE``, SHA-256 checked).
        sys_path: Local copy of ``sys_full_long.txt`` to use instead.
    """
    raw = np.genfromtxt(_data_path("pantheon2018_lcparam_full_long.txt"),
                        comments="#", dtype=None, encoding="utf-8")
    z = np.array([r[1] for r in raw], dtype=float)
    mb = np.array([r[4] for r in raw], dtype=float)
    dmb = np.array([r[5] for r in raw], dtype=float)
    order = np.argsort(z)
    cov = None
    if systematics:
        c = _fetch_sys(sys_path)
        cov = (c + np.diag(dmb ** 2))[np.ix_(order, order)]
    return SNData("Pantheon+sys" if systematics else "Pantheon", z[order], mb[order],
                  dmb[order], cov, "Scolnic et al. 2018")


def resolve(datasets) -> list[str]:
    """Normalize a dataset spec to a list of component names.

    Examples:
        >>> resolve("CC+BAO+Pantheon")
        ['CC+BAO', 'Pantheon']
        >>> resolve(["CC", "Pantheon+sys"])
        ['CC', 'Pantheon+sys']
    """
    if isinstance(datasets, str):
        if datasets in DATASETS:
            return [datasets]
        if datasets in _ALIASES:
            return list(_ALIASES[datasets])
        raise ValueError(f"unknown dataset {datasets!r}; components {DATASETS}, "
                         f"combinations {sorted(_ALIASES)}")
    names = list(datasets)
    for n in names:
        if n not in DATASETS:
            raise ValueError(f"unknown dataset component {n!r}")
    if sum(n.startswith("CC") for n in names) > 1 or sum(n.startswith("Pantheon") for n in names) > 1:
        raise ValueError("use at most one H(z) and one supernova component")
    return names


def load(names: Sequence[str]):
    """Load the resolved components: ``(HzData or None, SNData or None)``."""
    hz = sn = None
    for n in resolve(list(names) if not isinstance(names, str) else names):
        if n.startswith("CC"):
            hz = load_hz(n)
        else:
            sn = load_pantheon(systematics=(n == "Pantheon+sys"))
    return hz, sn
