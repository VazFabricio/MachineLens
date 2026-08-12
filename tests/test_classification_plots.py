"""Unit tests for the ClassificationPlots module."""

import pytest
import numpy as np
import pandas as pd
from typing import Tuple

import plotly.graph_objects as go
from sklearn.base import BaseEstimator
from machinelens.core.model_interface import ModelInterface
from machinelens.core.data_classes import DiagnosticResults
from machinelens.analyzer.classification_analyzer import ClassificationAnalyzer
from machinelens.plots.plots import DiagnosticPlotter


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
def diag_runner(classification_interface: ModelInterface) -> DiagnosticResults:
    """Pre-run diagnostics."""
    analyzer = ClassificationAnalyzer(classification_interface)
    dr = DiagnosticResults(
        problem_type="classification",
        model_name="RandomForestClassifier",
        algorithm_family="Ensemble (Forest/Boosting/Bagging)",
        feature_names=classification_interface.X_train.columns.tolist()
    )
    analyzer.analyze(dr)
    return dr


def test_classification_plots_init(diag_runner: DiagnosticResults) -> None:
    """Test initialization."""
    plots = DiagnosticPlotter(diag_runner)
    assert plots.results == diag_runner


def test_plot_metrics(diag_runner: DiagnosticResults) -> None:
    """Test plot metrics generation."""
    plots = DiagnosticPlotter(diag_runner)
    fig = plots.plot_metrics()
    assert isinstance(fig, go.Figure)


def test_plot_confusion_matrix(diag_runner: DiagnosticResults) -> None:
    """Test confusion matrix rendering context."""
    plots = DiagnosticPlotter(diag_runner)
    fig = plots.plot_confusion_matrix()
    assert isinstance(fig, go.Figure)


def test_plot_roc_curve(diag_runner: DiagnosticResults) -> None:
    """Test ROC curve plotly figure generation."""
    plots = DiagnosticPlotter(diag_runner)
    fig = plots.plot_roc_curve()
    assert isinstance(fig, go.Figure)


def test_plot_pr_curve(diag_runner: DiagnosticResults) -> None:
    """Test Precision-Recall curve plotly figure generation."""
    plots = DiagnosticPlotter(diag_runner)
    fig = plots.plot_pr_curve()
    assert isinstance(fig, go.Figure)
