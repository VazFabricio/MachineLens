"""MachineLens — Deep diagnostics, explainability, and reporting for ML models."""

__version__ = "0.1.1"

from machinelens.core import ModelInterface
from machinelens.core.data_classes import DiagnosticResults
from machinelens.analyzer import ModelAnalyzer
from machinelens.plots import DiagnosticPlotter

__all__ = [
    "ModelInterface",
    "DiagnosticResults",
    "ModelAnalyzer",
    "DiagnosticPlotter",
]

