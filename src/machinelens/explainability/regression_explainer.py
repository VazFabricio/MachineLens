"""Regression explainer model for understanding regression model behavior."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from explainability.base_explainer import BaseExplainer
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class RegressionExplainer(BaseExplainer):
    """Provides explainability analyses for regression models.

    This class includes:
        - Permutation-based feature importance
        - Partial dependence plots (PDP)
        - Sensitivity analysis (feature perturbation)
        - Fairness and bias evaluation by sensitive group

    Attributes:
        results (Dict[str, Any]): Stores numerical results and dataframes.
        plots (Dict[str, plt.Figure]): Stores matplotlib figures.
    """

    def __init__(self, model_interface: Any) -> None:
        """Initialize the RegressionExplainer.

        Args:
            model_interface: Object that provides access to the model and test data.
                Expected attributes: ``X_test``, ``y_test``, and optionally ``y_pred``.
        """
        super().__init__(model_interface)
        self.results: Dict[str, Any] = {}
        self.plots: Dict[str, plt.Figure] = {}

    # ---------------------------
    # Helper utilities
    # ---------------------------
    def _has_predict(self) -> bool:
        """Check whether the wrapped model exposes a predict method.

        Returns:
            bool: True if the model has a callable ``predict`` method, False otherwise.
        """
        return hasattr(self.model, "predict") and callable(self.model.predict)

    def _as_df(self, X: Any) -> Optional[pd.DataFrame]:
        """Convert input X to a pandas DataFrame when possible.

        Args:
            X: Input data (DataFrame, Series, ndarray, or None).

        Returns:
            Optional[pd.DataFrame]: A copy of X as a DataFrame, or None if conversion failed.
        """
        if X is None:
            return None
        if isinstance(X, pd.DataFrame):
            return X.copy()
        if isinstance(X, pd.Series):
            return X.to_frame()
        if isinstance(X, np.ndarray):
            return pd.DataFrame(X)
        return None

    # ---------------------------
    # Main analyses
    # ---------------------------
    def feature_importance(
        self, n_repeats: int = 10, random_state: int = 0
    ) -> Optional[pd.DataFrame]:
        """Compute permutation-based feature importance.

        Args:
            n_repeats: Number of permutation repeats. Defaults to 10.
            random_state: Random seed. Defaults to 0.

        Returns:
            Optional[pd.DataFrame]: DataFrame with columns
            ["feature", "importance_mean", "importance_std"], sorted by mean importance.
        """
        X_test = self._as_df(self.model_interface.X_test)
        y_test = self.model_interface.y_test

        if X_test is None or y_test is None or not self._has_predict():
            self.results["feature_importance"] = None
            return None

        try:
            r = permutation_importance(
                self.model,
                X_test,
                y_test,
                n_repeats=n_repeats,
                random_state=random_state,
            )
            importances = (
                pd.DataFrame(
                    {
                        "feature": list(X_test.columns),
                        "importance_mean": r.importances_mean,
                        "importance_std": r.importances_std,
                    }
                )
                .sort_values("importance_mean", ascending=False)
                .reset_index(drop=True)
            )

            self.results["feature_importance"] = importances

            fig, ax = plt.subplots(figsize=(8, max(3, 0.3 * len(importances))))
            ax.barh(importances["feature"][::-1], importances["importance_mean"][::-1])
            ax.set_xlabel("Permutation importance (mean)")
            ax.set_title("Feature Importance (permutation)")
            plt.tight_layout()
            self.plots["feature_importance"] = fig
            return importances
        except Exception as e:
            self.results["feature_importance_error"] = str(e)
            return None

    def partial_dependence(
        self, feature: Optional[str] = None, grid_points: int = 10
    ) -> Optional[pd.DataFrame]:
        """Approximate partial dependence for a given feature.

        Args:
            feature: Feature name to analyze. If None, uses the top feature from permutation importances.
            grid_points: Number of grid points to evaluate. Defaults to 10.

        Returns:
            Optional[pd.DataFrame]: DataFrame with columns ["value", "avg_pred"], or None if failed.
        """
        X_test = self._as_df(self.model_interface.X_test)
        if X_test is None or not self._has_predict():
            return None

        if feature is None:
            fi = self.results.get("feature_importance")
            feature = (
                X_test.columns[0] if fi is None or fi.empty else fi["feature"].iloc[0]
            )

        if feature not in X_test.columns:
            self.results[f"pdp_{feature}"] = None
            return None

        col = X_test[feature]
        grid = np.linspace(col.quantile(0.05), col.quantile(0.95), grid_points)
        preds = []
        Xbase = X_test.copy()

        try:
            for v in grid:
                Xbase[feature] = v
                preds.append(np.mean(self.model.predict(Xbase)))

            df = pd.DataFrame({"value": grid, "avg_pred": preds})
            self.results[f"pdp_{feature}"] = df

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(df["value"], df["avg_pred"], marker="o")
            ax.set_xlabel(feature)
            ax.set_ylabel("Average prediction")
            ax.set_title(f"Partial Dependence for {feature}")
            plt.tight_layout()
            self.plots[f"pdp_{feature}"] = fig
            return df
        except Exception as e:
            self.results[f"pdp_{feature}_error"] = str(e)
            return None

    def sensitivity_analysis(
        self, features: Optional[List[str]] = None, delta: float = 0.01
    ) -> Optional[pd.DataFrame]:
        """Estimate model sensitivity to feature perturbations.

        Args:
            features: List of features to analyze. If None, analyzes all columns.
            delta: Relative perturbation for numeric features. Defaults to 0.01.

        Returns:
            Optional[pd.DataFrame]: DataFrame with columns ["feature", "mean_abs_change"].
        """
        X_test = self._as_df(self.model_interface.X_test)
        if X_test is None or not self._has_predict():
            return None

        features = features or list(X_test.columns)
        results = []

        try:
            for feat in features:
                col = X_test[feat]
                if np.issubdtype(col.dtype, np.number):
                    up, down = X_test.copy(), X_test.copy()
                    up[feat] = col * (1 + delta)
                    down[feat] = col * (1 - delta)
                else:
                    mode = col.mode().iloc[0] if not col.mode().empty else col.iloc[0]
                    up, down = X_test.copy(), X_test.copy()
                    up[feat] = down[feat] = mode

                pred_up = self.model.predict(up)
                pred_down = self.model.predict(down)
                mean_abs_change = np.mean(np.abs(pred_up - pred_down))
                results.append({"feature": feat, "mean_abs_change": mean_abs_change})

            df = pd.DataFrame(results).sort_values("mean_abs_change", ascending=False)
            self.results["sensitivity"] = df

            fig, ax = plt.subplots(figsize=(8, max(3, 0.3 * len(df))))
            ax.barh(df["feature"][::-1], df["mean_abs_change"][::-1])
            ax.set_xlabel("Mean absolute change in prediction")
            ax.set_title("Sensitivity Analysis")
            plt.tight_layout()
            self.plots["sensitivity"] = fig
            return df
        except Exception as e:
            self.results["sensitivity_error"] = str(e)
            return None

    def fairness_bias(
        self, sensitive_column: str, metric: str = "mae"
    ) -> Optional[pd.DataFrame]:
        """Compute group-wise error metrics for a sensitive feature.

        Args:
            sensitive_column: Column in X_test representing group membership.
            metric: Metric name to report (currently informational). Defaults to "mae".

        Returns:
            Optional[pd.DataFrame]: DataFrame with columns
            [sensitive_column, "count", "mae", "mse", "r2"].
        """
        X_test = self._as_df(self.model_interface.X_test)
        y_test, y_pred = self.model_interface.y_test, self.model_interface.y_pred

        if X_test is None or y_test is None or y_pred is None:
            return None
        if sensitive_column not in X_test.columns:
            self.results[f"fairness_{sensitive_column}"] = None
            return None

        try:
            grp = pd.DataFrame(
                {
                    "group": X_test[sensitive_column],
                    "y_true": np.asarray(y_test).ravel(),
                    "y_pred": np.asarray(y_pred).ravel(),
                }
            )
            agg = (
                grp.groupby("group")
                .apply(
                    lambda d: pd.Series(
                        {
                            "count": len(d),
                            "mae": mean_absolute_error(d["y_true"], d["y_pred"]),
                            "mse": mean_squared_error(d["y_true"], d["y_pred"]),
                            "r2": (
                                r2_score(d["y_true"], d["y_pred"])
                                if len(d) > 1
                                else np.nan
                            ),
                        }
                    )
                )
                .reset_index()
            )

            self.results[f"fairness_{sensitive_column}"] = agg

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(agg["group"].astype(str), agg["mae"])
            ax.set_xlabel(sensitive_column)
            ax.set_ylabel("MAE")
            ax.set_title(f"Group MAE by {sensitive_column}")
            plt.xticks(rotation=45)
            plt.tight_layout()
            self.plots[f"fairness_{sensitive_column}"] = fig
            return agg
        except Exception as e:
            self.results[f"fairness_{sensitive_column}_error"] = str(e)
            return None

    def run_all(self, sensitive_column: Optional[str] = None) -> None:
        """Run a suite of regression explainability analyses.

        Args:
            sensitive_column: Optional name of sensitive column to evaluate fairness.
        """
        X_test = self._as_df(self.model_interface.X_test)
        y_test, y_pred = self.model_interface.y_test, self.model_interface.y_pred

        # Diagnostics
        if X_test is not None and y_test is not None and y_pred is not None:
            try:
                fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                axes[0].scatter(y_test, y_pred, alpha=0.6, s=20)
                axes[0].plot(
                    [min(y_test), max(y_test)],
                    [min(y_test), max(y_test)],
                    color="k",
                    linestyle="--",
                )
                axes[0].set_xlabel("y_true")
                axes[0].set_ylabel("y_pred")
                axes[0].set_title("y_true vs y_pred")

                residuals = np.asarray(y_test).ravel() - np.asarray(y_pred).ravel()
                axes[1].hist(residuals, bins=30, alpha=0.8)
                axes[1].set_title("Residuals distribution")
                axes[1].set_xlabel("y_true - y_pred")

                plt.tight_layout()
                self.plots["diagnostics"] = fig

                self.results["metrics"] = {
                    "mae": float(mean_absolute_error(y_test, y_pred)),
                    "mse": float(mean_squared_error(y_test, y_pred)),
                    "r2": float(r2_score(y_test, y_pred)),
                }
            except Exception as e:
                self.results["diagnostics_error"] = str(e)

        # Run analyses
        self.feature_importance()
        fi = self.results.get("feature_importance")
        top_feat = (
            fi["feature"].iloc[0]
            if isinstance(fi, pd.DataFrame) and not fi.empty
            else None
        )
        self.partial_dependence(feature=top_feat)
        self.sensitivity_analysis()
        if sensitive_column:
            self.fairness_bias(sensitive_column)
