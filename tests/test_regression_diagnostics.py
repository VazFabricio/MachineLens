"""Unit tests for the RegressionDiagnostics module."""

import pytest
import numpy as np
import pandas as pd
from typing import Tuple

from sklearn.base import BaseEstimator
from core.model_interface import ModelInterface
from diagnostics.regression_diagnostics import RegressionDiagnostics


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


def test_regression_diagnostics_init(regression_interface: ModelInterface) -> None:
    """Test initialization."""
    diag = RegressionDiagnostics(regression_interface)
    assert diag.model is not None
    assert isinstance(diag.results, dict)


def test_analyze_residual_outliers_happy_path(regression_interface: ModelInterface) -> None:
    """Test outlier analysis functionality."""
    diag = RegressionDiagnostics(regression_interface)
    res = diag.analyze_residual_outliers(min_group_size=1, normality_check=False)
    
    if res is not None:
        assert isinstance(res, pd.DataFrame)


def test_run_all(regression_interface: ModelInterface) -> None:
    """Test the run_all executor triggers all sub-processes."""
    diag = RegressionDiagnostics(regression_interface)
    diag.run_all()
    
    assert "residuals_data" in diag.results
    rd = diag.results["residuals_data"]
    assert "y_test" in rd
    assert "residuals" in rd
    assert "qq_osm" in rd


def test_training_diagnostics_calculations(regression_interface: ModelInterface) -> None:
    """Test computing training diagnostics directly."""
    diag = RegressionDiagnostics(regression_interface)
    diag.compute_training_residuals()
    
    assert "training_diagnostics" in diag.results
    td = diag.results["training_diagnostics"]
    assert "std_residuals" in td
    
    diag.compute_leverage()
    assert "leverage" in td
    
    diag.compute_cooks_distance()
    assert "cooks_distance" in td
    
    diag.compute_vif()
    assert "vif" in td
    
    diag.compute_training_qq()
    assert "qq_osm" in td
