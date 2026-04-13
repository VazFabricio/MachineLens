"""Diagnostics module for MachineLens."""


from machinelens.diagnostics.classification_diagnostics import ClassificationDiagnostics
from machinelens.diagnostics.regression_diagnostics import RegressionDiagnostics

__all__ = [
    "ClassificationDiagnostics",
    "RegressionDiagnostics",
]
