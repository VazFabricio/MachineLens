"""Classification-specific dataclasses for diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from machinelens.core.data_classes.shared import SubsetData


@dataclass
class ClassificationSubsetData(SubsetData):
    """Extended subset data carrying probability arrays (classification).

    Attributes
    ----------
    y_prob : np.ndarray or None
        Predicted class probabilities (n_samples, n_classes).
        ``None`` when the estimator does not support ``predict_proba``.
    """

    y_prob: Optional[np.ndarray] = None


@dataclass
class RocCurveData:
    """ROC curve data for a single class (or binary problem).

    Attributes
    ----------
    fpr : np.ndarray
        False-positive rates.
    tpr : np.ndarray
        True-positive rates.
    auc_score : float
        Area Under the ROC Curve.
    label : str
        Human-readable label, e.g. ``"binary"`` or ``"class_2"``.
    """

    fpr: np.ndarray
    tpr: np.ndarray
    auc_score: float
    label: str


@dataclass
class PrCurveData:
    """Precision-Recall curve data for a single class (or binary).

    Attributes
    ----------
    precision_arr : np.ndarray
        Precision values at each threshold.
    recall_arr : np.ndarray
        Recall values at each threshold.
    average_precision : float
        Average precision score (area under the PR curve).
    baseline : float
        No-skill baseline (positive class prevalence).
    label : str
        Human-readable label.
    """

    precision_arr: np.ndarray
    recall_arr: np.ndarray
    average_precision: float
    baseline: float
    label: str


@dataclass
class MisclassificationResult:
    """Per-feature statistical analysis of misclassified samples.

    Attributes
    ----------
    results_df : pd.DataFrame
        Feature-level test results with columns like ``feature``,
        ``stat``, ``p_value``, ``adj_p_value``, ``effect_size``,
        ``significant``.
    """

    results_df: pd.DataFrame


@dataclass
class CalibrationCurveData:
    """Calibration curve data for reliability diagrams.

    Attributes
    ----------
    prob_true : np.ndarray
        True probability in each bin.
    prob_pred : np.ndarray
        Mean predicted probability in each bin.
    label : str
        Human-readable label for the class.
    """

    prob_true: np.ndarray
    prob_pred: np.ndarray
    label: str


@dataclass
class ThresholdAnalysisData:
    """Threshold decision analysis data.

    Attributes
    ----------
    thresholds : np.ndarray
        Decision thresholds.
    precision : np.ndarray
        Precision scores for each threshold.
    recall : np.ndarray
        Recall scores for each threshold.
    f1_score : np.ndarray
        F1 scores for each threshold.
    label : str
        Human-readable label for the class.
    """

    thresholds: np.ndarray
    precision: np.ndarray
    recall: np.ndarray
    f1_score: np.ndarray
    label: str


@dataclass
class ClassificationMetrics:
    """Scalar classification evaluation metrics for one subset.

    Attributes
    ----------
    accuracy : float
        Overall accuracy.
    precision : float
        Precision (macro or binary).
    recall : float
        Recall (macro or binary).
    f1_score : float
        F1-score (macro or binary).
    """

    accuracy: float
    precision: float
    recall: float
    f1_score: float
    mcc: float
    roc_auc: Optional[float] = None
    pr_auc: Optional[float] = None
    brier_score: Optional[float] = None
    log_loss: Optional[float] = None
