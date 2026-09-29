"""qablate: classical-quantum ablation studies for Bayesian inference.

Swap the components of an inference algorithm between classical and
quantum-circuit implementations, one at a time, and measure what changes.
The quantum components are simulable classically at the sizes used here;
the package studies fidelity to classical results, not speed-ups.

Core API (NumPy/SciPy only at import time; Qiskit and ArviZ load lazily)::

    MetropolisHastings, State
    GaussianProposal, RandomCircuitProposal
    MetropolisAcceptance, AmplitudeEncodedMetropolis
    Grid, BornMachineVI
    AerBackend, StatevectorBackend, IBMRuntimeBackend, NoiseSpec
    Study
    diagnostics

Optional subpackages: ``qablate.cosmology`` (background-cosmology models and
likelihoods) and ``qablate.experimental`` (genetic algorithms).
"""
from . import diagnostics
from ._version import __version__
from .ablation import Run, Study
from .acceptance import Acceptance, AmplitudeEncodedMetropolis, MetropolisAcceptance
from .backends import AerBackend, Backend, IBMRuntimeBackend, StatevectorBackend
from .mcmc import MetropolisHastings, State
from .noise import NoiseSpec
from .proposals import GaussianProposal, Proposal, RandomCircuitProposal
from .variational import BornMachineVI, Grid, VIResult

__all__ = [
    "__version__",
    "MetropolisHastings", "State",
    "Proposal", "GaussianProposal", "RandomCircuitProposal",
    "Acceptance", "MetropolisAcceptance", "AmplitudeEncodedMetropolis",
    "Grid", "BornMachineVI", "VIResult",
    "Backend", "AerBackend", "StatevectorBackend", "IBMRuntimeBackend", "NoiseSpec",
    "Study", "Run",
    "diagnostics",
]
