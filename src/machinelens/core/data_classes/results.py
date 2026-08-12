"""Strongly-typed data structures for diagnostic results.

This module defines the **Data Object Layer** — a hierarchy of
``dataclass`` objects that serve as the strict contract and single source
of truth for a model's diagnostic state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

from machinelens.core.data_classes.classification import (
    CalibrationCurveData,
    ClassificationMetrics,
    ClassificationSubsetData,
    MisclassificationResult,
    PrCurveData,
    RocCurveData,
    ThresholdAnalysisData,
)
from machinelens.core.data_classes.regression import (
    OutlierAnalysisResult,
    QQData,
    RegressionMetrics,
    RegressionSubsetData,
)
from machinelens.core.data_classes.shap import ShapData

# Import specialized components to define DiagnosticResults and keep backward compatibility
from machinelens.core.data_classes.shared import LowessData, SubsetData

# Explicitly export everything for backward compatibility
__all__ = [
    "SubsetData",
    "LowessData",
    "RegressionSubsetData",
    "QQData",
    "OutlierAnalysisResult",
    "RegressionMetrics",
    "ClassificationSubsetData",
    "RocCurveData",
    "PrCurveData",
    "MisclassificationResult",
    "ClassificationMetrics",
    "CalibrationCurveData",
    "ThresholdAnalysisData",
    "ShapData",
    "DiagnosticResults",
]


@dataclass
class DiagnosticResults:
    """Complete diagnostic state for a model — single source of truth.

    This object is the **strict contract** between the processing layer
    (``ModelAnalyzer``) and the visualisation layer
    (``DiagnosticPlotter``).  Every calculation result has a dedicated,
    typed field — no opaque dictionaries.

    Attributes
    ----------
    problem_type : str
        ``"classification"`` or ``"regression"``.
    model_name : str
        Human-readable name of the estimator class.
    algorithm_family : str
        Pretty-printed sklearn sub-module family.
    feature_names : list of str
        Feature column names.

    train_data / test_data
        Aligned arrays for each subset (type depends on problem).
    train_metrics / test_metrics
        Scalar evaluation metrics for each subset.

    Regression-specific fields
        ``*_qq``, ``*_linearity_lowess``, ``*_scale_loc_lowess``,
        ``leverage``, ``leverage_lowess``, ``cooks_distance``,
        ``outlier_analysis``.

    Classification-specific fields
        ``*_confusion_matrix``, ``*_roc_curves``, ``*_pr_curves``,
        ``*_misclassification``.
    """

    # ---- identity ----
    problem_type: str
    model_name: str
    algorithm_family: str
    feature_names: List[str] = field(default_factory=list)

    # ---- subset data ----
    train_data: Optional[SubsetData] = None
    test_data: Optional[SubsetData] = None

    # ---- regression metrics ----
    train_metrics: Optional[RegressionMetrics] = None
    test_metrics: Optional[RegressionMetrics] = None

    # ---- classification metrics ----
    train_clf_metrics: Optional[ClassificationMetrics] = None
    test_clf_metrics: Optional[ClassificationMetrics] = None

    # ---- regression-specific ----
    train_qq: Optional[QQData] = None
    test_qq: Optional[QQData] = None
    train_linearity_lowess: Optional[LowessData] = None
    test_linearity_lowess: Optional[LowessData] = None
    train_scale_loc_lowess: Optional[LowessData] = None
    test_scale_loc_lowess: Optional[LowessData] = None
    leverage: Optional[np.ndarray] = None
    leverage_lowess: Optional[LowessData] = None
    cooks_distance: Optional[np.ndarray] = None
    outlier_analysis: Optional[OutlierAnalysisResult] = None

    # ---- classification-specific ----
    train_confusion_matrix: Optional[pd.DataFrame] = None
    test_confusion_matrix: Optional[pd.DataFrame] = None
    train_roc_curves: Optional[List[RocCurveData]] = None
    test_roc_curves: Optional[List[RocCurveData]] = None
    train_pr_curves: Optional[List[PrCurveData]] = None
    test_pr_curves: Optional[List[PrCurveData]] = None
    train_calibration_curves: Optional[List[CalibrationCurveData]] = None
    test_calibration_curves: Optional[List[CalibrationCurveData]] = None
    train_threshold_analysis: Optional[List[ThresholdAnalysisData]] = None
    test_threshold_analysis: Optional[List[ThresholdAnalysisData]] = None
    train_misclassification: Optional[MisclassificationResult] = None
    test_misclassification: Optional[MisclassificationResult] = None

    # ---- SHAP explanations ----
    train_shap: Optional[ShapData] = None
    test_shap: Optional[ShapData] = None
