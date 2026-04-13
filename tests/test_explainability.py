"""Unit tests for the explainability modules."""

import pytest
import numpy as np
import pandas as pd
from typing import Tuple

from sklearn.base import BaseEstimator
from core.model_interface import ModelInterface
from explainability.base_explainer import BaseExplainer
from explainability.native_explainer import NativeExplainer


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


def test_base_explainer_init(classification_interface: ModelInterface) -> None:
    """Test BaseExplainer initialization."""
    explainer = BaseExplainer(classification_interface)
    assert explainer.model_interface is not None
    assert explainer.feature_importance() is None


def test_native_explainer_feature_importance_auto(classification_interface: ModelInterface) -> None:
    """Test extracting native feature importance (auto)."""
    explainer = NativeExplainer(classification_interface)
    df = explainer.feature_importance(method="auto")
    
    assert df is not None
    assert isinstance(df, pd.DataFrame)
    assert "feature" in df.columns
    assert "importance" in df.columns


def test_native_explainer_feature_importance_permutation(classification_interface: ModelInterface) -> None:
    """Test permutation importance over native attributes."""
    explainer = NativeExplainer(classification_interface)
    df = explainer.feature_importance(method="permutation", n_repeats=2)
    
    assert df is not None
    assert isinstance(df, pd.DataFrame)
    assert "importance_mean" in df.columns


def test_native_explainer_partial_dependence(classification_interface: ModelInterface) -> None:
    """Test partial dependence arrays generation."""
    explainer = NativeExplainer(classification_interface)
    res = explainer.partial_dependence(features=["feature_0"], grid_resolution=10)
    
    assert res is not None
    assert "feature_0" in res
    assert "grid" in res["feature_0"]


def test_native_explainer_sensitivity_analysis(classification_interface: ModelInterface) -> None:
    """Test model prediction sensitivity metric."""
    explainer = NativeExplainer(classification_interface)
    df = explainer.sensitivity_analysis(noise_level=0.1, n_iterations=2)
    
    assert df is not None
    assert isinstance(df, pd.DataFrame)
    assert "sensitivity" in df.columns


def test_native_explainer_fairness_bias(classification_interface: ModelInterface) -> None:
    """Test fairness calculation missing protected attribute."""
    explainer = NativeExplainer(classification_interface)
    
    df_missing = explainer.fairness_bias(protected_attribute=None)
    assert df_missing is None
    
    df_exists = explainer.fairness_bias(protected_attribute="feature_1")
    assert df_exists is not None
    assert isinstance(df_exists, pd.DataFrame)
    assert "group" in df_exists.columns
