"""SHAP data structures for model explainability."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Union

import numpy as np


@dataclass
class ShapData:
    """Stores SHAP values and metadata for local and global explainability.

    Attributes
    ----------
    feature_names : list of str
        The names of the features used in the model.
    base_value : float or list of float
        The base (expected) value of the model's predictions.
    shap_values : np.ndarray
        The matrix of local SHAP values for each instance and feature.
        Shape: (n_samples, n_features).
    mean_abs_shap : np.ndarray
        Mean absolute SHAP values per feature across the dataset.
    eval_index : list of int
        The original dataset row indices corresponding to each row in
        ``shap_values``. Used by the frontend to map selection events
        (which carry original indices) to the correct SHAP row.
    feature_values : np.ndarray or None
        The raw feature values for the evaluated set, same shape as
        ``shap_values``. Required by the Beeswarm plot for coloring.
    """

    feature_names: List[str]
    base_value: Union[float, List[float]]
    shap_values: np.ndarray
    mean_abs_shap: np.ndarray
    eval_index: List[int] = field(default_factory=list)
    feature_values: Optional[np.ndarray] = None
