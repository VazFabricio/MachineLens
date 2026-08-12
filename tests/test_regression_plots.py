"""Unit tests for the RegressionPlots module."""

import pytest
import numpy as np
import pandas as pd
from typing import Tuple

import plotly.graph_objects as go
from sklearn.base import BaseEstimator
from machinelens.core.model_interface import ModelInterface
from machinelens.core.data_classes import DiagnosticResults
from machinelens.analyzer.regression_analyzer import RegressionAnalyzer
from machinelens.plots.plots import DiagnosticPlotter


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
def diag_runner(regression_interface: ModelInterface) -> DiagnosticResults:
    """Pre-run diagnostics."""
    analyzer = RegressionAnalyzer(regression_interface)
    dr = DiagnosticResults(
        problem_type="regression",
        model_name="LinearRegression",
        algorithm_family="Linear Model",
        feature_names=regression_interface.X_train.columns.tolist()
    )
    analyzer.analyze(dr)
    return dr


def test_regression_plots_init(diag_runner: DiagnosticResults) -> None:
    """Test initialization."""
    plots = DiagnosticPlotter(diag_runner)
    assert plots.results == diag_runner


def test_plot_actual_vs_predicted(diag_runner: DiagnosticResults) -> None:
    """Test Actual vs Predicted scatter plot generation."""
    plots = DiagnosticPlotter(diag_runner)
    fig = plots.plot_actual_vs_predicted()
    assert isinstance(fig, go.Figure)


def test_plot_residuals(diag_runner: DiagnosticResults) -> None:
    """Test Residuals dot plot generation."""
    plots = DiagnosticPlotter(diag_runner)
    fig = plots.plot_residuals()
    assert isinstance(fig, go.Figure)


def test_plot_qq(diag_runner: DiagnosticResults) -> None:
    """Test Q-Q plot generation."""
    plots = DiagnosticPlotter(diag_runner)
    fig = plots.plot_qq()
    assert isinstance(fig, go.Figure)
