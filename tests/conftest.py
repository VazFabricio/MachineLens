"""Fixtures for the MachineLens test suite."""

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator
from typing import Tuple


@pytest.fixture
def classification_data() -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Provide a generated binary classification dataset."""
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(5)])
    
    # Split into train/test 80/20
    X_train = X_df.iloc[:80].copy()
    X_test = X_df.iloc[80:].copy()
    y_train = y[:80]
    y_test = y[80:]
    return X_train, X_test, y_train, y_test


@pytest.fixture
def regression_data() -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Provide a generated regression dataset."""
    X, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)
    X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(5)])
    
    # Split into train/test 80/20
    X_train = X_df.iloc[:80].copy()
    X_test = X_df.iloc[80:].copy()
    y_train = y[:80]
    y_test = y[80:]
    return X_train, X_test, y_train, y_test


@pytest.fixture
def fitted_rf_classifier(
        classification_data: Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]
) -> BaseEstimator:
    """Provide a fitted RandomForestClassifier."""
    X_train, _, y_train, _ = classification_data
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    return model


@pytest.fixture
def unfitted_rf_classifier() -> BaseEstimator:
    """Provide an unfitted RandomForestClassifier."""
    return RandomForestClassifier(n_estimators=10, random_state=42)


@pytest.fixture
def fitted_linear_regression(
        regression_data: Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]
) -> BaseEstimator:
    """Provide a fitted LinearRegression model."""
    X_train, _, y_train, _ = regression_data
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model
