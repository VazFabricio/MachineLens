"""Shared data structures for diagnostic results."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SubsetData:
    """Aligned arrays for a single data subset (train **or** test).

    Attributes
    ----------
    X_data : pd.DataFrame
        Feature matrix, index-aligned with the target arrays.
    y_true : np.ndarray
        Ground-truth target values (1-D).
    y_pred : np.ndarray
        Model predictions (1-D), same length as ``y_true``.
    """

    X_data: pd.DataFrame
    y_true: np.ndarray
    y_pred: np.ndarray
    residuals: np.ndarray | None = None
    std_residuals: np.ndarray | None = None
    abs_residuals: np.ndarray | None = None


@dataclass
class LowessData:
    """LOWESS smoothing curve with a 95 % bootstrap confidence band.

    Attributes
    ----------
    x_smooth : np.ndarray
        Sorted x-coordinates of the smoothed curve.
    y_smooth : np.ndarray
        Smoothed y-values.
    ci_lower : np.ndarray
        Lower bound of the 95 % confidence interval.
    ci_upper : np.ndarray
        Upper bound of the 95 % confidence interval.
    """

    x_smooth: np.ndarray
    y_smooth: np.ndarray
    ci_lower: np.ndarray
    ci_upper: np.ndarray
