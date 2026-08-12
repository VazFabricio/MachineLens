"""Unit tests for the ClassificationAnalyzer module."""

import pytest
import numpy as np
import pandas as pd
from typing import Tuple

from sklearn.base import BaseEstimator
from machinelens.core.model_interface import ModelInterface
from machinelens.core.data_classes import DiagnosticResults
from machinelens.analyzer.classification_analyzer import ClassificationAnalyzer


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


def test_classification_analyzer_happy_path(classification_interface: ModelInterface) -> None:
    """Test full analysis computing functionality."""
    analyzer = ClassificationAnalyzer(classification_interface)
    
    dr = DiagnosticResults(
        problem_type="classification",
        model_name="RandomForestClassifier",
        algorithm_family="Ensemble (Forest/Boosting/Bagging)",
        feature_names=classification_interface.X_train.columns.tolist()
    )
    
    analyzer.analyze(dr)
    
    # Check metrics
    assert dr.test_clf_metrics is not None
    assert dr.train_clf_metrics is not None
    assert dr.test_clf_metrics.accuracy > 0
    assert dr.test_clf_metrics.precision > 0
    assert dr.test_clf_metrics.recall > 0
    assert dr.test_clf_metrics.f1_score > 0
    
    # Check confusion matrix
    assert dr.test_confusion_matrix is not None
    assert isinstance(dr.test_confusion_matrix, pd.DataFrame)
    
    # Check ROC curves (since RF provides predict_proba)
    assert dr.test_roc_curves is not None
    assert len(dr.test_roc_curves) > 0
    
    # Check PR curves
    assert dr.test_pr_curves is not None
    assert len(dr.test_pr_curves) > 0
    
    # Check threshold analysis
    assert dr.test_threshold_analysis is not None
    
    # Check SHAP
    assert dr.train_shap is not None
    assert dr.test_shap is not None
