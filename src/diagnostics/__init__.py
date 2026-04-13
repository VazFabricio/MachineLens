"""Diagnostics module for MachineLens."""


from diagnostics.classification_diagnostics import ClassificationDiagnostics
from diagnostics.regression_diagnostics import RegressionDiagnostics

__all__ = [
    "ClassificationDiagnostics",
    "RegressionDiagnostics",
]
