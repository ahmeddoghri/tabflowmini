"""Tabular synthetic data with simple flow-style transport."""

from .data import Row, make_table
from .synth import evaluate_synth, fit_synth, sample_synth
from .synth_v2 import fit_synth_v2, sample_synth_v2

__all__ = [
    "Row",
    "evaluate_synth",
    "fit_synth",
    "fit_synth_v2",
    "make_table",
    "sample_synth",
    "sample_synth_v2",
]
