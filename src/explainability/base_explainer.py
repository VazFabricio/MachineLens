"""Base class for all explainers in the MachineLens library."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import pandas as pd

if TYPE_CHECKING:
    pass


class BaseExplainer:
    """Abstract base class for explainability modules.

    Provides a unified interface for various explainability techniques.
    All concrete explainers should inherit from this class and implement
    the abstract methods defined here.
    """

    def __init__(self, model_interface: Any) -> None:
        """
        Initialize the BaseExplainer.

        Parameters
        ----------
        model_interface : Any
            An object containing the model and dataset splits.
        """
        self.model_interface = model_interface

    def feature_importance(self, **kwargs: Any) -> Optional[pd.DataFrame]:
        """
        Calculate feature importance.

        Parameters
        ----------
        **kwargs : Any
            Additional arguments for feature importance calculation.

        Returns
        -------
        pandas.DataFrame or None
            A DataFrame containing feature importances.
        """
        pass

    def partial_dependence(self, **kwargs: Any) -> Any:
        """
        Calculate partial dependence.

        Parameters
        ----------
        **kwargs : Any
            Additional arguments.

        Returns
        -------
        Any
            Partial dependence results.
        """
        pass

    def sensitivity_analysis(self, **kwargs: Any) -> Any:
        """
        Perform sensitivity analysis.

        Parameters
        ----------
        **kwargs : Any
            Additional arguments.

        Returns
        -------
        Any
            Sensitivity analysis results.
        """
        pass

    def fairness_bias(self, **kwargs: Any) -> Any:
        """
        Evaluate model fairness and bias.

        Parameters
        ----------
        **kwargs : Any
            Additional arguments.

        Returns
        -------
        Any
            Fairness and bias evaluation results.
        """
        pass
