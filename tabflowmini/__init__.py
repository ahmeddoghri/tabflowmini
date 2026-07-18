"""Tabular synthetic data with simple flow-style transport."""

from .data import Row, make_table
from .synth import evaluate_synth, fit_synth, sample_synth

__all__ = ["Row", "evaluate_synth", "fit_synth", "make_table", "sample_synth"]
