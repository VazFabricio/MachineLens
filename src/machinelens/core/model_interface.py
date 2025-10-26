"""Core model interface module for the MachineLens library."""

from __future__ import annotations

from typing import Any, Optional, Union

import numpy as np
import pandas as pd

PandasLike = Union[pd.DataFrame, pd.Series]
ArrayLike = Union[np.ndarray, pd.Series]


class ModelInterface:
    """Encapsulate model data and provide validation utilities."""

    def __init__(
        self,
        model: Any,
        X_train: Optional[PandasLike],
        X_test: Optional[PandasLike],
        y_train: Optional[ArrayLike],
        y_test: Optional[ArrayLike],
        y_pred: Optional[ArrayLike],
        problem_type: str = "regression",
    ) -> None:
        """Initialize the ModelInterface with data and model references.

        Args:
            model (Any): The fitted machine learning model object.
            X_train (Optional[PandasLike]): Training features.
            X_test (Optional[PandasLike]): Test features.
            y_train (Optional[ArrayLike]): Training target values.
            y_test (Optional[ArrayLike]): Test target values.
            y_pred (Optional[ArrayLike]): Model's predictions on the test set.
            problem_type (str): The type of problem, e.g., "regression" or
                "classification". Defaults to "regression".
        """
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.y_pred = y_pred
        self.problem_type = problem_type

        self._validate_inputs()

    def _validate_inputs(self) -> None:
        """Validate shapes and types of inputs to catch user errors early.

        Ensures that:
            - ``X_train`` and ``X_test`` are pandas DataFrames, Series, or numpy arrays.
            - The lengths of ``X_test`` and ``y_test`` match.
            - The lengths of ``X_test`` and ``y_pred`` match when both are provided.

        Raises:
            TypeError: If ``X_train`` or ``X_test`` is not a pandas DataFrame, Series, or numpy array.
            ValueError: If ``X_test`` and ``y_test`` or ``y_pred`` have inconsistent lengths.
        """
        for name, obj in (
            ("X_train", self.X_train),
            ("X_test", self.X_test),
        ):
            if obj is not None and not isinstance(
                obj, (pd.DataFrame, pd.Series, np.ndarray)
            ):
                raise TypeError(
                    f"{name} should be a pandas DataFrame/Series or numpy.ndarray, got {type(obj)}"
                )

        if self.X_test is not None and self.y_test is not None:
            nX = len(self.X_test)
            ny = len(self.y_test)
            if nX != ny:
                raise ValueError(f"X_test has length {nX} but y_test has length {ny}")

        if self.X_test is not None and self.y_pred is not None:
            nX = len(self.X_test)
            npred = len(self.y_pred)
            if nX != npred:
                raise ValueError(
                    f"X_test has length {nX} but y_pred has length {npred}"
                )
