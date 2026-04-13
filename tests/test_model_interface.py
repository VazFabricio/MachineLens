"""Unit tests for the ModelInterface core class."""

import pytest
import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.base import BaseEstimator

from core.model_interface import ModelInterface


def test_model_interface_classification_happy_path(
    fitted_rf_classifier: BaseEstimator,
    classification_data: Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray],
) -> None:
    """Test ModelInterface initialization with a fitted classification model."""
    X_train, X_test, y_train, y_test = classification_data
    
    interface = ModelInterface(
        model=fitted_rf_classifier,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        y_pred=None,
    )
    
    assert interface.problem_type == "classification"
    assert interface.results["Model Name"] == "RandomForestClassifier"
    assert interface.results["Task Type"] == "Supervised Classification"
    assert interface.results["Algorithm Family"] == "Ensemble (Forest/Boosting/Bagging)"
    assert interface.results["Analysis Results"]["Feature Importance Available"] is True


def test_model_interface_regression_happy_path(
    fitted_linear_regression: BaseEstimator,
    regression_data: Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray],
) -> None:
    """Test ModelInterface initialization with a fitted regression model."""
    X_train, X_test, y_train, y_test = regression_data
    
    interface = ModelInterface(
        model=fitted_linear_regression,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )
    
    assert interface.problem_type == "regression"
    assert interface.results["Model Name"] == "LinearRegression"
    assert interface.results["Task Type"] == "Supervised Regression"
    assert interface.results["Algorithm Family"] == "Linear Model (Regression/Logit/Ridge)"
    assert interface.results["Analysis Results"]["Coefficients Available"] is True


def test_model_interface_unfitted_model(
    unfitted_rf_classifier: BaseEstimator,
    classification_data: Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray],
) -> None:
    """Test exception raising when an unfitted model is provided."""
    X_train, X_test, y_train, y_test = classification_data
    
    with pytest.raises(ValueError, match=r"The provided model is not fitted\. Execute \.fit\(\) before\."):
        ModelInterface(
            model=unfitted_rf_classifier,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
        )


def test_model_interface_invalid_model_type(
    classification_data: Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray],
) -> None:
    """Test exception raising when the model is not a scikit-learn BaseEstimator."""
    X_train, X_test, y_train, y_test = classification_data
    
    with pytest.raises(TypeError, match="The model must be an instance of sklearn.base.BaseEstimator."):
        ModelInterface(
            model="Not a model",  # type: ignore
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
        )


def test_model_interface_inconsistent_lengths(
    fitted_rf_classifier: BaseEstimator,
    classification_data: Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray],
) -> None:
    """Test exception raising when data lengths are inconsistent."""
    X_train, X_test, y_train, y_test = classification_data
    
    malformed_y_test = y_test[:-1]
    
    with pytest.raises(ValueError, match="Found input variables with inconsistent"):
        ModelInterface(
            model=fitted_rf_classifier,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=malformed_y_test,
        )


def test_model_interface_invalid_data_types(
    fitted_rf_classifier: BaseEstimator,
    classification_data: Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray],
) -> None:
    """Test exception raising when X_train or X_test have incorrect types."""
    _, X_test, y_train, y_test = classification_data
    
    with pytest.raises(TypeError, match="should be a pandas DataFrame/Series"):
        ModelInterface(
            model=fitted_rf_classifier,
            X_train=[1, 2, 3],  # type: ignore
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
        )
