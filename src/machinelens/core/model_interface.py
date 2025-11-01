"""Core model interface module for the MachineLens library."""

from __future__ import annotations

from typing import Any, Dict, Optional, Union, cast

import numpy as np
import pandas as pd
from sklearn.base import is_classifier, is_clusterer, is_regressor
from sklearn.ensemble import BaseEnsemble
from sklearn.linear_model import (
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
)
from sklearn.tree import BaseDecisionTree

PandasLike = Union[pd.DataFrame, pd.Series]
ArrayLike = Union[np.ndarray, pd.Series]

ModelResults = Dict[str, Union[str, Dict[str, Any]]]


class ModelInterface:
    """Encapsulate model data and provide validation utilities."""

    results: ModelResults

    def __init__(
        self,
        model: Any,
        X_train: Optional[PandasLike],
        X_test: Optional[PandasLike],
        y_train: Optional[ArrayLike],
        y_test: Optional[ArrayLike],
        y_pred: Optional[ArrayLike],
        problem_type: str = "regression",
    ) -> None:
        """Initialize the ModelInterface with model, data splits, and predictions.

        Args:
            model (Any): The trained machine learning model object.
            X_train (Optional[PandasLike]): The training features (e.g., pd.DataFrame).
            X_test (Optional[PandasLike]): The test features (e.g., pd.DataFrame).
            y_train (Optional[ArrayLike]): The training target values (e.g., np.ndarray).
            y_test (Optional[ArrayLike]): The ground truth target values for the test set.
            y_pred (Optional[ArrayLike]): The model's predictions on the test set (X_test).
            problem_type (str): The type of machine learning problem.
                Defaults to "regression".
        """
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.y_pred = y_pred
        self.problem_type = problem_type

        self.results = {
            "Model Name": type(model).__name__,
            "Task Type": "Other/Unsupervised",
            "Algorithm Family": "Other/Unknown",
            "Analysis Results": {},
        }

        self._validate_inputs()
        self._analyze_and_classify_model()

    def _validate_inputs(self) -> None:
        """Validate shapes and types of inputs to catch user errors early.

        Ensures that:
            - ``X_train`` and ``X_test`` are pandas DataFrames, Series, or numpy arrays.
            - The lengths of ``X_test`` and ``y_test`` match.
            - The lengths of ``X_test`` and ``y_pred`` match when both are provided.

        Raises:
            TypeError: If ``X_train`` or ``X_test`` is not a pandas DataFrame, Series, or numpy array.
            ValueError: If ``X_test`` and ``y_test`` or ``y_pred`` have inconsistent lengths.
        """
        for name, obj in (
            ("X_train", self.X_train),
            ("X_test", self.X_test),
        ):
            if obj is not None and not isinstance(
                obj, (pd.DataFrame, pd.Series, np.ndarray)
            ):
                raise TypeError(
                    f"{name} should be a pandas DataFrame/Series or numpy.ndarray, got {type(obj)}"
                )

        if self.X_test is not None and self.y_test is not None:
            nX = len(self.X_test)
            ny = len(self.y_test)
            if nX != ny:
                raise ValueError(f"X_test has length {nX} but y_test has length {ny}")

        if self.X_test is not None and self.y_pred is not None:
            nX = len(self.X_test)
            npred = len(self.y_pred)
            if nX != npred:
                raise ValueError(
                    f"X_test has length {nX} but y_pred has length {npred}"
                )

    def _analyze_and_classify_model(self):
        """Analyze the model to determine its task type and algorithm family.

        Inspects the `self.model` object using `sklearn.base` helpers
        (e.g., `is_classifier`, `is_regressor`) and `isinstance` checks
        to categorize the model.

        This method populates the `self.results` dictionary with findings
        like 'Task Type', 'Algorithm Family', and details on available
        attributes (e.g., 'Coefficients Available').

        Returns:
            Dict[str, Any]: The updated `self.results` dictionary.
        """
        model = self.model
        model_name = type(model).__name__
        model_module = type(model).__module__

        self.results["Model Name"] = model_name

        analysis_results: Dict[str, Any] = cast(
            Dict[str, Any], self.results["Analysis Results"]
        )

        # --- 1. Determine the TASK TYPE ---
        if is_classifier(model):
            self.results["Task Type"] = "Supervised Classification"
        elif is_regressor(model):
            self.results["Task Type"] = "Supervised Regression"
        elif is_clusterer(model):
            self.results["Task Type"] = "Unsupervised Clustering"

        # --- 2. Determine the ALGORITHM FAMILY ---

        if isinstance(model, BaseEnsemble):
            self.results["Algorithm Family"] = "Ensemble (Forest/Boosting)"
            if hasattr(model, "feature_importances_"):
                analysis_results["Feature Importance Available"] = True

        elif isinstance(model, (LinearRegression, Ridge, Lasso, LogisticRegression)):
            self.results["Algorithm Family"] = "Linear Model (Regression/Logit)"
            if hasattr(model, "coef_"):
                analysis_results["Coefficients Available"] = True
            if hasattr(model, "intercept_"):
                analysis_results["Coefficients Available"] = True

        elif isinstance(model, BaseDecisionTree):
            self.results["Algorithm Family"] = "Single Decision Tree"
            if hasattr(model, "feature_importances_"):
                analysis_results["Feature Importance Available"] = True

        elif model_module.startswith("sklearn.naive_bayes"):
            self.results["Algorithm Family"] = "Probabilistic (Naive Bayes)"
            if hasattr(model, "class_prior_"):
                analysis_results["Class Priors Available"] = True

        elif model_module.startswith("sklearn.neighbors"):
            self.results["Algorithm Family"] = "Instance-based (K-Nearest Neighbors)"

        elif model_module.startswith("sklearn.svm"):
            self.results["Algorithm Family"] = "Support Vector Machines (Kernel-based)"

        elif model_module.startswith("sklearn.cluster"):
            self.results["Algorithm Family"] = "Clustering"
            if hasattr(model, "n_clusters"):
                analysis_results["Expected Attribute"] = "n_clusters"

        # --- 3. Fallback based on name pattern (for custom or 3rd party models) ---
        elif "forest" in model_name.lower() or "boost" in model_name.lower():
            self.results["Algorithm Family"] = "Ensemble (Forest/Boosting)"
        elif "tree" in model_name.lower():
            self.results["Algorithm Family"] = "Decision Tree"
        elif "linear" in model_name.lower():
            self.results["Algorithm Family"] = "Linear Model"
        elif "svm" in model_name.lower() or "svc" in model_name.lower():
            self.results["Algorithm Family"] = "Support Vector Machine"
        elif "bayes" in model_name.lower():
            self.results["Algorithm Family"] = "Probabilistic (Naive Bayes)"
        elif "knn" in model_name.lower() or "neighbor" in model_name.lower():
            self.results["Algorithm Family"] = "Instance-based (K-Nearest Neighbors)"

        return self.results
