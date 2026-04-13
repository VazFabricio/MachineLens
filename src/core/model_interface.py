"""Core model interface module for the MachineLens library."""

from __future__ import annotations

from typing import Any, Dict, Optional, Union, cast

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, is_classifier, is_clusterer, is_regressor
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_consistent_length, check_is_fitted

PandasLike = Union[pd.DataFrame, pd.Series]
ArrayLike = Union[np.ndarray, pd.Series]

ModelResults = Dict[str, Union[str, Dict[str, Any]]]


class ModelInterface:
    """Encapsulate model data and provide validation utilities."""

    results: ModelResults
    problem_type: str

    def __init__(
        self,
        model: BaseEstimator,
        X_train: Optional[PandasLike],
        X_test: Optional[PandasLike],
        y_train: Optional[ArrayLike],
        y_test: Optional[ArrayLike],
        y_pred: Optional[ArrayLike] = None,
    ) -> None:
        """Initialize the ModelInterface with model, data splits, and predictions.

        Args:
            model: The trained scikit-learn machine learning model object.
            X_train: The training features (e.g., pd.DataFrame).
            X_test: The test features (e.g., pd.DataFrame).
            y_train: The training target values (e.g., np.ndarray).
            y_test: The ground truth target values for the test set.
            y_pred: The model's predictions on the test set (X_test).

        Raises
        ------
            TypeError: If the model is not an instance of BaseEstimator.
        """
        if not isinstance(model, BaseEstimator):
            raise TypeError(
                "O modelo deve ser uma instância de sklearn.base.BaseEstimator."
            )

        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.y_pred = y_pred
        self.problem_type = "unknown"

        self.results = {
            "Model Name": type(model).__name__,
            "Task Type": "Other/Unsupervised",
            "Algorithm Family": "Other/Unknown",
            "Analysis Results": {},
        }

        self._validate_inputs()
        self._analyze_and_classify_model()

    def _validate_inputs(self) -> None:
        """Validate shapes, types of inputs, and model fitting to catch user errors early.

        Ensures that:
            - The model has been fitted.
            - ``X_train`` and ``X_test`` are pandas DataFrames, Series, or numpy arrays.
            - The lengths of ``X_test`` and ``y_test`` match.
            - The lengths of ``X_test`` and ``y_pred`` match when both are provided.

        Raises
        ------
            ValueError: If the model is not fitted or if inputs have inconsistent lengths.
            TypeError: If ``X_train`` or ``X_test`` is not a pandas DataFrame,Series,or numpy array.
        """
        try:
            check_is_fitted(self.model)
        except NotFittedError as e:
            raise ValueError(
                f"O modelo fornecido não está treinado. Execute .fit() antes. Erro: {e}"
            ) from e

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
            check_consistent_length(self.X_test, self.y_test)

        if self.X_test is not None and self.y_pred is not None:
            check_consistent_length(self.X_test, self.y_pred)

    def _analyze_and_classify_model(self) -> Dict[str, Any]:
        """Analyze the model dynamically to determine its task type and algorithm family.

        Extracts the scikit-learn submodule automatically to prevent hardcoding
        every possible module path. Dynamically discovers available attributes.

        Returns
        -------
            Dict[str, Any]: The updated `self.results` dictionary.
        """
        model = self.model
        model_module = type(model).__module__

        self.results["Model Name"] = type(model).__name__

        analysis_results: Dict[str, Any] = cast(
            Dict[str, Any], self.results["Analysis Results"]
        )

        # --- 0. Extract Feature Names ---
        if hasattr(model, "feature_names_in_"):
            analysis_results["Feature Names"] = model.feature_names_in_.tolist()
        elif isinstance(self.X_train, pd.DataFrame):
            analysis_results["Feature Names"] = self.X_train.columns.tolist()

        # --- 1. Determine the TASK TYPE ---
        if is_classifier(model):
            self.problem_type = "classification"
            self.results["Task Type"] = "Supervised Classification"
        elif is_regressor(model):
            self.problem_type = "regression"
            self.results["Task Type"] = "Supervised Regression"
        elif is_clusterer(model):
            self.problem_type = "clustering"
            self.results["Task Type"] = "Unsupervised Clustering"

        # --- 2. Determine the ALGORITHM FAMILY (Dynamic Extraction) ---
        parts = model_module.split(".")
        if len(parts) >= 2 and parts[0] == "sklearn":
            submodule = parts[1]  # Ex: 'ensemble', 'linear_model', 'tree'

            pretty_names = {
                "ensemble": "Ensemble (Forest/Boosting/Bagging)",
                "linear_model": "Linear Model (Regression/Logit/Ridge)",
                "naive_bayes": "Probabilistic (Naive Bayes)",
                "neighbors": "Instance-based (K-Nearest Neighbors)",
                "svm": "Support Vector Machines",
                "dummy": "Baseline (Dummy Estimator)",
            }

            self.results["Algorithm Family"] = pretty_names.get(
                submodule, submodule.replace("_", " ").title()
            )
        else:
            self.results["Algorithm Family"] = "Other/Unknown"

        # --- 3. Dynamic Attribute Discovery ---
        if hasattr(model, "feature_importances_"):
            analysis_results["Feature Importance Available"] = True

        if hasattr(model, "coef_"):
            analysis_results["Coefficients Available"] = True

        if hasattr(model, "class_prior_"):
            analysis_results["Class Priors Available"] = True

        if hasattr(model, "n_clusters"):
            analysis_results["Expected Attribute"] = "n_clusters"

        return self.results
