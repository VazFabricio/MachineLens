"""Processing layer — the calculation engine for MachineLens.

``ModelAnalyzer`` takes a ``ModelInterface``, routes it to the correct specialized
analyzer, and returns a fully populated ``DiagnosticResults`` instance.
"""

from __future__ import annotations

import logging
from typing import List

import pandas as pd

from machinelens.core import ModelInterface
from machinelens.core.data_classes import DiagnosticResults

from .classification_analyzer import ClassificationAnalyzer
from .regression_analyzer import RegressionAnalyzer

logger = logging.getLogger(__name__)


class ModelAnalyzer:
    """Calculation engine that produces ``DiagnosticResults``.

    This analyzer automatically delegates calculations to the specialized
    ``RegressionAnalyzer`` or ``ClassificationAnalyzer`` based on the detected
    problem type, keeping the domain logic perfectly split and decoupled.

    Parameters
    ----------
    interface : ModelInterface
        A validated interface wrapping the fitted model and data splits.

    Examples
    --------
    >>> from machinelens.core import ModelInterface
    >>> from machinelens.analyzer import ModelAnalyzer
    >>> mi = ModelInterface(model, X_train, X_test, y_train, y_test, y_pred)
    >>> analyzer = ModelAnalyzer(mi)
    >>> results = analyzer.analyze()
    """

    def __init__(self, interface: ModelInterface) -> None:
        """Initialize the ModelAnalyzer.

        Parameters
        ----------
        interface : ModelInterface
            A validated interface wrapping the fitted model and data splits.
        """
        self._iface = interface
        self._model = interface.model

    def analyze(self) -> DiagnosticResults:
        """Run all diagnostics and return a populated result object.

        Returns
        -------
        DiagnosticResults
            Fully populated diagnostic state.
        """
        iface = self._iface

        # Seed the result container with automatic feature name discovery
        feature_names: List[str] = []
        if isinstance(iface.X_train, pd.DataFrame):
            feature_names = iface.X_train.columns.tolist()
        elif isinstance(iface.X_test, pd.DataFrame):
            feature_names = iface.X_test.columns.tolist()
        elif hasattr(self._model, "feature_names_in_"):
            feature_names = self._model.feature_names_in_.tolist()

        dr = DiagnosticResults(
            problem_type=iface.problem_type,
            model_name=str(iface.results.get("Model Name", "")),
            algorithm_family=str(iface.results.get("Algorithm Family", "")),
            feature_names=feature_names,
        )

        if iface.problem_type == "regression":
            reg_analyzer = RegressionAnalyzer(iface)
            reg_analyzer.analyze(dr)
        elif iface.problem_type == "classification":
            clf_analyzer = ClassificationAnalyzer(iface)
            clf_analyzer.analyze(dr)

        return dr
