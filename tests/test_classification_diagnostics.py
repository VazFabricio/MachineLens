"""Unit tests for the ClassificationDiagnostics module."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch
from typing import Tuple

from sklearn.base import BaseEstimator
from core.model_interface import ModelInterface
from diagnostics.classification_diagnostics import ClassificationDiagnostics


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


def test_classification_diagnostics_init(classification_interface: ModelInterface) -> None:
    """Test initialization."""
    diag = ClassificationDiagnostics(classification_interface)
    assert diag.model is not None
    assert isinstance(diag.results, dict)


def test_classification_metrics_happy_path(classification_interface: ModelInterface) -> None:
    """Test metrics computing functionality."""
    diag = ClassificationDiagnostics(classification_interface)
    
    test_metrics = diag.classification_metrics_test()
    assert "accuracy" in test_metrics
    assert "precision" in test_metrics
    assert "recall" in test_metrics
    assert "f1_score" in test_metrics
    
    train_metrics = diag.classification_metrics_train()
    assert len(train_metrics) > 0


def test_confusion_matrix_happy_path(classification_interface: ModelInterface) -> None:
    """Test confusion matrix parsing."""
    diag = ClassificationDiagnostics(classification_interface)
    cm = diag.confusion_matrix_test()
    assert cm is not None
    assert isinstance(cm, pd.DataFrame)
    
    cm_train = diag.confusion_matrix_train()
    assert cm_train is not None


def test_compute_roc_curve_happy_path(classification_interface: ModelInterface) -> None:
    """Test ROC curve parsing."""
    diag = ClassificationDiagnostics(classification_interface)
    roc = diag.compute_roc_curve_test()
    assert roc is not None
    assert "binary" in roc
    assert "auc" in roc["binary"]


def test_compute_pr_curve_happy_path(classification_interface: ModelInterface) -> None:
    """Test PR curve computing."""
    diag = ClassificationDiagnostics(classification_interface)
    pr = diag.compute_pr_curve_test()
    assert pr is not None
    assert "binary" in pr
    assert "ap" in pr["binary"]


def test_misclassification_analysis_happy_path(classification_interface: ModelInterface) -> None:
    """Test statistical extraction of misclassifications."""
    diag = ClassificationDiagnostics(classification_interface)
    # Using min_group_size=1 to avoid returning empty on small mismatch subsets.
    res = diag.misclassification_analysis_test(min_group_size=1, normality_check=False)
    assert res is not None
    assert isinstance(res, pd.DataFrame)


def test_run_all(classification_interface: ModelInterface) -> None:
    """Test the run_all executor handles execution without exception."""
    diag = ClassificationDiagnostics(classification_interface)
    diag.run_all()
    # verify that metrics are updated
    keys = list(diag.results.keys())
    assert any("classification_metrics_" in k for k in keys)
    assert any("confusion_matrix_" in k for k in keys)
    assert any("roc_curve_" in k for k in keys)
