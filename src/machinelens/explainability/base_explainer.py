"""Base class for all explainers in the MachineLens library."""

from __future__ import annotations

from typing import Any


class BaseExplainer:
    """Abstract base class for explainability modules.

    Provides a unified interface for various explainability techniques.
    All concrete explainers should inherit from this class and implement
    the abstract methods defined here.
    """

    def __init__(self, model_interface: Any) -> None:
        """Initialize the BaseExplainer.

        Args:
            model_interface (Any): An object encapsulating the trained model and related data.
        """
        self.model_interface = model_interface
        self.model = getattr(model_interface, "model", None)

    def feature_importance(self) -> None:
        """Compute and visualize feature importance.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement feature_importance().")

    def partial_dependence(self) -> None:
        """Generate partial dependence plots (PDPs).

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement partial_dependence().")

    def sensitivity_analysis(self) -> None:
        """Perform model sensitivity analysis.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement sensitivity_analysis().")

    def fairness_bias(self) -> None:
        """Evaluate model fairness and potential bias.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement fairness_bias().")
