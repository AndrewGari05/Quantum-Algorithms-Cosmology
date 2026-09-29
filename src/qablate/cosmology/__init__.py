"""Flat background cosmology: models, datasets and likelihoods.

Examples:
    >>> from qablate.cosmology import Posterior
    >>> post = Posterior("cpl", "CC+BAO+Pantheon")
    >>> post.model.param_names
    ('Om', 'H0', 'w0', 'wa')
"""
from .data import DATASETS, load_hz, load_pantheon
from .likelihood import Posterior, fit_statistics
from .models import MODELS, PLANCK2018, SH0ES2022, CosmoModel, get_model

__all__ = ["MODELS", "CosmoModel", "get_model", "Posterior", "fit_statistics",
           "DATASETS", "load_hz", "load_pantheon", "PLANCK2018", "SH0ES2022"]
