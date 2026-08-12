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
    """Encapsulate model data and provide validation utilities.

    This class standardizes how the MachineLens library interacts with
    scikit-learn compatible models. It wraps a fitted estimator alongside
    its training/test splits and predictions, automatically validating
    inputs and classifying the model's task type and algorithm family.

    Attributes
    ----------
    model : BaseEstimator
        The wrapped scikit-learn estimator.
    X_train : PandasLike or None
        Training feature matrix.
    X_test : PandasLike or None
        Test feature matrix.
    y_train : ArrayLike or None
        Training target values.
    y_test : ArrayLike or None
        Test target values (ground truth).
    y_pred : ArrayLike or None
        Model predictions on the test set.
    problem_type : str
        Detected problem type: ``"classification"``, ``"regression"``,
        ``"clustering"``, or ``"unknown"``.
    results : ModelResults
        Dictionary with model metadata and analysis results.
    """

    results: ModelResults
    problem_type: str

    def __init__(
        self,
        model: BaseEstimator,
        X_train: Optional[PandasLike] = None,
        X_test: Optional[PandasLike] = None,
        y_train: Optional[ArrayLike] = None,
        y_test: Optional[ArrayLike] = None,
        y_pred: Optional[ArrayLike] = None,
    ) -> None:
        """Initialize the ModelInterface with model, data splits, and predictions.

        Parameters
        ----------
        model : BaseEstimator
            The trained scikit-learn machine learning model object.
        X_train : pandas.DataFrame or pandas.Series or None
            The training features.
        X_test : pandas.DataFrame or pandas.Series or None
            The test features.
        y_train : numpy.ndarray or pandas.Series or None
            The training target values.
        y_test : numpy.ndarray or pandas.Series or None
            The ground truth target values for the test set.
        y_pred : numpy.ndarray or pandas.Series, optional
            The model's predictions on the test set (X_test).

        Raises
        ------
        TypeError
            If the model is not an instance of BaseEstimator.
        """
        if not isinstance(model, BaseEstimator):
            raise TypeError(
                "The model must be an instance of sklearn.base.BaseEstimator."
            )

        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.y_pred = y_pred
        self.problem_type = "unknown"

        self.results: ModelResults = {
            "Model Name": type(model).__name__,
            "Task Type": "Other/Unsupervised",
            "Algorithm Family": "Other/Unknown",
            "Analysis Results": {},
        }

        self._validate_inputs()
        self._analyze_and_classify_model()
        if (
            self.y_pred is None
            and self.X_test is not None
            and hasattr(self.model, "predict")
        ):
            try:
                self.y_pred = self.model.predict(self.X_test)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_inputs(self) -> None:
        """Validate shapes, types of inputs, and model fitting.

        Ensures that:
        - The model has been fitted.
        - ``X_train`` and ``X_test`` are pandas DataFrames, Series, or
          numpy arrays.
        - The lengths of ``X_test`` and ``y_test`` match.
        - The lengths of ``X_test`` and ``y_pred`` match when both are
          provided.

        Raises
        ------
        ValueError
            If the model is not fitted or if inputs have inconsistent
            lengths.
        TypeError
            If ``X_train`` or ``X_test`` is not a supported type.
        """
        try:
            check_is_fitted(self.model)
        except NotFittedError as e:
            raise ValueError(
                f"The provided model is not fitted. Execute .fit() before. Error: {e}"
            ) from e

        for name, obj in (
            ("X_train", self.X_train),
            ("X_test", self.X_test),
        ):
            if obj is not None and not isinstance(
                obj, (pd.DataFrame, pd.Series, np.ndarray)
            ):
                raise TypeError(
                    f"{name} should be a pandas DataFrame/Series or "
                    f"numpy.ndarray, got {type(obj)}"
                )

        if self.X_test is not None and self.y_test is not None:
            check_consistent_length(self.X_test, self.y_test)

        if self.X_test is not None and self.y_pred is not None:
            check_consistent_length(self.X_test, self.y_pred)

    # ------------------------------------------------------------------
    # Model classification
    # ------------------------------------------------------------------

    def _analyze_and_classify_model(self) -> Dict[str, Any]:
        """Analyze the model to determine its task type and algorithm family.

        Extracts the scikit-learn submodule automatically to prevent
        hard-coding every possible module path.  Dynamically discovers
        available fitted attributes.

        Returns
        -------
        dict
            The updated ``self.results`` dictionary.
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

        # --- 2. Determine the ALGORITHM FAMILY ---
        parts = model_module.split(".")
        if len(parts) >= 2 and parts[0] == "sklearn":
            submodule = parts[1]

            pretty_names = {
                "ensemble": "Ensemble (Forest/Boosting/Bagging)",
                "linear_model": "Linear Model (Regression/Logit/Ridge)",
                "naive_bayes": "Probabilistic (Naive Bayes)",
                "neighbors": "Instance-based (K-Nearest Neighbors)",
                "svm": "Support Vector Machines",
                "tree": "Decision Tree",
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
