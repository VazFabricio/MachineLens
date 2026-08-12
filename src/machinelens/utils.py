"""Utility functions for MachineLens."""

from typing import Any, Optional

import numpy as np
import pandas as pd


def _to_dataframe(X: Any) -> Optional[pd.DataFrame]:
    """Convert array-like *X* to a ``pd.DataFrame``.

    Parameters
    ----------
    X : Any
        The array-like, Series, or DataFrame to convert.

    Returns
    -------
    pd.DataFrame or None
        The converted DataFrame, or None if the input is None or unsupported.
    """
    if X is None:
        return None
    if isinstance(X, pd.DataFrame):
        return X.copy()
    if isinstance(X, pd.Series):
        return X.to_frame()
    if isinstance(X, np.ndarray):
        cols = [f"Feature_{i}" for i in range(X.shape[1] if X.ndim > 1 else 1)]
        return pd.DataFrame(X, columns=cols)
    return None
