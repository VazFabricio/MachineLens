"""Unit tests for the RegressionAnalyzer module."""

import pytest
import numpy as np
import pandas as pd
from typing import Tuple

from sklearn.base import BaseEstimator
from machinelens.core.model_interface import ModelInterface
from machinelens.core.data_classes import DiagnosticResults
from machinelens.analyzer.regression_analyzer import RegressionAnalyzer


@pytest.fixture
def regression_interface(
    fitted_linear_regression: BaseEstimator,
    regression_data: Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray],
) -> ModelInterface:
    """Fixture providing a configured ModelInterface."""
    X_train, X_test, y_train, y_test = regression_data
    y_pred = fitted_linear_regression.predict(X_test)
    
    return ModelInterface(
        model=fitted_linear_regression,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        y_pred=y_pred,
    )


def test_regression_analyzer_happy_path(regression_interface: ModelInterface) -> None:
    """Test full analysis computing functionality."""
    analyzer = RegressionAnalyzer(regression_interface)
    
    dr = DiagnosticResults(
        problem_type="regression",
        model_name="LinearRegression",
        algorithm_family="Linear Model (Regression/Logit/Ridge)",
        feature_names=regression_interface.X_train.columns.tolist()
    )
    
    analyzer.analyze(dr)
    
    # Check metrics
    assert dr.test_metrics is not None
    assert dr.train_metrics is not None
    assert dr.test_metrics.rmse > 0
    assert dr.test_metrics.mae > 0
    
    # Check residuals
    assert dr.test_data is not None
    assert len(dr.test_data.residuals) > 0
    assert len(dr.test_data.std_residuals) > 0
    
    # Check QQ
    assert dr.test_qq is not None
    assert len(dr.test_qq.theoretical) > 0
    
    # Check Leverage / Cooks
    assert dr.leverage is not None
    assert dr.cooks_distance is not None
    
    # Check SHAP
    assert dr.train_shap is not None
    assert dr.test_shap is not None
