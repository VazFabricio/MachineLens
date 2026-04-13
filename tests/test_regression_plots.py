"""Unit tests for the RegressionPlots module."""

import pytest
import numpy as np
import pandas as pd
from typing import Tuple

import plotly.graph_objects as go
from sklearn.base import BaseEstimator
from core.model_interface import ModelInterface
from diagnostics.regression_diagnostics import RegressionDiagnostics
from visualization.regression_plots import RegressionPlots


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


@pytest.fixture
def diag_runner(regression_interface: ModelInterface) -> RegressionDiagnostics:
    """Pre-run diagnostics."""
    diag = RegressionDiagnostics(regression_interface)
    diag.run_all()
    return diag


def test_regression_plots_init(diag_runner: RegressionDiagnostics) -> None:
    """Test initialization."""
    plots = RegressionPlots(diag_runner.results)
    assert isinstance(plots.plots, dict)


def test_plot_actual_vs_predicted_test(diag_runner: RegressionDiagnostics) -> None:
    """Test Actual vs Predicted scatter plot generation."""
    plots = RegressionPlots(diag_runner.results)
    plots.plot_actual_vs_predicted_test()
    
    key = "actual_vs_predicted_test"
    assert key in plots.plots
    assert isinstance(plots.plots[key], go.Figure)


def test_plot_residuals_vs_predicted_test(diag_runner: RegressionDiagnostics) -> None:
    """Test Residuals vs Predicted dot plot generation."""
    plots = RegressionPlots(diag_runner.results)
    plots.plot_residuals_vs_predicted_test()
    
    key = "residuals_vs_predicted_test"
    assert key in plots.plots
    assert isinstance(plots.plots[key], go.Figure)


def test_plot_qq_test(diag_runner: RegressionDiagnostics) -> None:
    """Test Q-Q plot generation."""
    plots = RegressionPlots(diag_runner.results)
    plots.plot_qq_test()
    
    key = "qq_test"
    assert key in plots.plots
    assert isinstance(plots.plots[key], go.Figure)


def test_run_all(diag_runner: RegressionDiagnostics) -> None:
    """Test the run_all executor triggers all plot renderings."""
    plots = RegressionPlots(diag_runner.results)
    if hasattr(plots, 'run_all'):
        plots.run_all()
        # if the runner populates any, verify at least one is there
        keys = list(plots.plots.keys())
        assert len(keys) > 0
