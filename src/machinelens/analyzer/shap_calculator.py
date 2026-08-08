"""Helper module to compute SHAP values safely and efficiently."""

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd
import shap

from machinelens.core.data_classes.shap import ShapData

logger = logging.getLogger(__name__)


def compute_shap_values(
    model: Any,
    X_train: pd.DataFrame,
    X_eval: pd.DataFrame,
    is_classification: bool = False,
) -> Optional[ShapData]:
    """Compute SHAP values for a given evaluation set using the training set as background.

    Parameters
    ----------
    model : Any
        The fitted scikit-learn compatible estimator.
    X_train : pd.DataFrame
        The training data to use as background.
    X_eval : pd.DataFrame
        The data to explain.
    is_classification : bool
        Whether the model is a classifier.

    Returns
    -------
    ShapData or None
        The computed SHAP data, or None if computation fails.
    """
    try:
        # Sample background if it's too large to prevent freezing
        background = (
            X_train
            if len(X_train) <= 100
            else shap.sample(X_train, 100, random_state=42)
        )

        # Cap eval set to max 10,000 for reasonable serialization size
        X_eval_capped = X_eval.head(10000)

        # Try TreeExplainer first, fallback to Explainer (which uses Permutation or Exact)
        try:
            explainer = shap.TreeExplainer(model)
            # Just test if it works without error
            _ = explainer.expected_value
        except Exception:
            try:
                explainer = shap.LinearExplainer(model, background)
            except Exception:
                predict_fn = (
                    model.predict_proba
                    if is_classification and hasattr(model, "predict_proba")
                    else model.predict
                )
                explainer = shap.Explainer(predict_fn, background)

        shap_values_obj = explainer(X_eval_capped)

        base_value = shap_values_obj.base_values
        values = shap_values_obj.values

        # For multi-class classification, values shape is (n_samples, n_features, n_classes)
        # For simplicity in local dashboard plotting, if multi-class, we take the explanation
        # for the predicted class or just class 1 for binary.
        if is_classification and values.ndim == 3:
            # We assume binary classification for now, extracting the positive class (index 1)
            # If multi-class, we would need a more complex structure, but let's stick to
            # class 1 for binary
            if values.shape[2] >= 2:
                values = values[:, :, 1]
                if isinstance(base_value, np.ndarray) and base_value.ndim > 0:
                    base_value = (
                        base_value[:, 1] if base_value.ndim == 2 else base_value[1]
                    )
                elif isinstance(base_value, list):
                    base_value = base_value[1]

        # Ensure base_value is a single scalar or list that can be serialized
        if isinstance(base_value, np.ndarray):
            base_value = float(np.mean(base_value))
        elif isinstance(base_value, list):
            base_value = float(base_value[0])  # simplify

        if not isinstance(base_value, float):
            base_value = float(np.mean(base_value))

        mean_abs_shap = np.abs(values).mean(axis=0)

        # Store the original index of the eval rows so frontend can map
        # original dataset indices → SHAP array positions
        eval_index = X_eval_capped.index.tolist()

        # Store raw feature values for beeswarm coloring
        feature_values = X_eval_capped.values

        return ShapData(
            feature_names=X_eval_capped.columns.tolist(),
            base_value=base_value,
            shap_values=values,
            mean_abs_shap=mean_abs_shap,
            eval_index=eval_index,
            feature_values=feature_values,
        )
    except Exception as exc:
        logger.warning("SHAP computation failed: %s", exc)
        return None
