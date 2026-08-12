"""Regression-specific dataclasses for diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np
import pandas as pd

from machinelens.core.data_classes.shared import LowessData, SubsetData


@dataclass
class RegressionSubsetData(SubsetData):
    """Extended subset data carrying residual arrays (regression only).

    Attributes
    ----------
    residuals : np.ndarray
        Raw residuals  (``y_true - y_pred``).
    std_residuals : np.ndarray
        Standardised residuals  (``residuals / RMSE``).
    abs_residuals : np.ndarray
        Absolute residuals  (``|residuals|``).
    """

    residuals: np.ndarray = field(default_factory=lambda: np.array([]))
    std_residuals: np.ndarray = field(default_factory=lambda: np.array([]))
    abs_residuals: np.ndarray = field(default_factory=lambda: np.array([]))


@dataclass
class QQData:
    """Q-Q (Quantile-Quantile) plot coordinates and reference line.

    Attributes
    ----------
    theoretical : np.ndarray
        Theoretical quantiles from the standard normal.
    sample : np.ndarray
        Ordered sample quantiles.
    slope : float
        Slope of the best-fit reference line.
    intercept : float
        Intercept of the best-fit reference line.
    r_value : float
        Pearson correlation coefficient of the fit.
    ci_lower : np.ndarray
        Lower 95 % CI envelope.
    ci_upper : np.ndarray
        Upper 95 % CI envelope.
    """

    theoretical: np.ndarray
    sample: np.ndarray
    slope: float
    intercept: float
    r_value: float
    ci_lower: np.ndarray
    ci_upper: np.ndarray


@dataclass
class OutlierAnalysisResult:
    """Statistical analysis of residual outliers.

    Attributes
    ----------
    results_df : pd.DataFrame
        Per-feature statistical test results (p-values, effect sizes,
        significance flags), sorted by adjusted p-value.
    threshold : float
        The standardised-residual threshold used to classify outliers.
    lowess_curves : dict of str → LowessData
        LOWESS smoothing curves for features flagged as significant,
        keyed by feature name.
    """

    results_df: pd.DataFrame
    threshold: float
    lowess_curves: Dict[str, LowessData] = field(default_factory=dict)


@dataclass
class RegressionMetrics:
    """Scalar regression evaluation metrics for one subset.

    Attributes
    ----------
    mae : float
        Mean Absolute Error.
    mse : float
        Mean Squared Error.
    rmse : float
        Root Mean Squared Error.
    r2 : float
        Coefficient of determination (R²).
    r2_adjusted : float
        Adjusted R².
    mape : float
        Mean Absolute Percentage Error (%).
    n_samples : int
        Number of observations used.
    """

    mae: float
    mse: float
    rmse: float
    r2: float
    r2_adjusted: float
    mape: float
    n_samples: int
