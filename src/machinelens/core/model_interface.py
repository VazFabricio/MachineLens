from __future__ import annotations

from typing import Any, Optional, Union

import numpy as np
import pandas as pd

PandasLike = Union[pd.DataFrame, pd.Series]
ArrayLike = Union[np.ndarray, pd.Series]


class ModelInterface:
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
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.y_pred = y_pred
        self.problem_type = problem_type

        self._validate_inputs()

    def _validate_inputs(self) -> None:
        """
        Basic validation of shapes and types to catch user errors early.
        - Ensures X/* are pandas DataFrames (recommended) or numpy arrays
        - Ensures lengths of X_test and y_test and y_pred match when present
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
