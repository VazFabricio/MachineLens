"""Unit tests for the ClassificationPlots module."""

import pytest
import numpy as np
import pandas as pd
from typing import Tuple

import plotly.graph_objects as go
from sklearn.base import BaseEstimator
from core.model_interface import ModelInterface
from diagnostics.classification_diagnostics import ClassificationDiagnostics
from visualization.classification_plots import ClassificationPlots


@pytest.fixture
def classification_interface(
    fitted_rf_classifier: BaseEstimator,
    classification_data: Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray],
) -> ModelInterface:
    """Fixture providing a configured ModelInterface."""
    X_train, X_test, y_train, y_test = classification_data
    return ModelInterface(
        model=fitted_rf_classifier,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )


@pytest.fixture
def diag_runner(classification_interface: ModelInterface) -> ClassificationDiagnostics:
    """Pre-run diagnostics."""
    diag = ClassificationDiagnostics(classification_interface)
    diag.run_all()
    return diag


def test_classification_plots_init(diag_runner: ClassificationDiagnostics) -> None:
    """Test initialization."""
    plots = ClassificationPlots(diag_runner.results)
    assert isinstance(plots.plots, dict)


def test_plot_metrics_table(diag_runner: ClassificationDiagnostics) -> None:
    """Test plot metrics table generation."""
    plots = ClassificationPlots(diag_runner.results)
    plots.plot_metrics_table_test()
    
    key = "metrics_table_test"
    assert key in plots.plots
    assert isinstance(plots.plots[key], go.Figure)


def test_plot_confusion_matrix(diag_runner: ClassificationDiagnostics) -> None:
    """Test confusion matrix rendering context."""
    plots = ClassificationPlots(diag_runner.results)
    plots.plot_confusion_matrix_test()
    
    key = "confusion_matrix_test"
    assert key in plots.plots
    assert isinstance(plots.plots[key], go.Figure)


def test_plot_roc_curve(diag_runner: ClassificationDiagnostics) -> None:
    """Test ROC curve plotly figure generation."""
    plots = ClassificationPlots(diag_runner.results)
    plots.plot_roc_curve_test()
    
    key = "roc_curve_test"
    assert key in plots.plots
    assert isinstance(plots.plots[key], go.Figure)


def test_plot_pr_curve(diag_runner: ClassificationDiagnostics) -> None:
    """Test Precision-Recall curve plotly figure generation."""
    plots = ClassificationPlots(diag_runner.results)
    plots.plot_pr_curve_test()
    
    key = "pr_curve_test"
    assert key in plots.plots
    assert isinstance(plots.plots[key], go.Figure)


def test_run_all(diag_runner: ClassificationDiagnostics) -> None:
    """Test the run_all executor triggers all plot renderings."""
    plots = ClassificationPlots(diag_runner.results)
    plots.run_all()
    
    keys = list(plots.plots.keys())
    assert len(keys) > 0
    assert any("confusion_matrix_test" in k for k in keys)
    assert any("roc_curve_test" in k for k in keys)
