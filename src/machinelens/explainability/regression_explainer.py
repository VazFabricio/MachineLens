from __future__ import annotations

from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from explainability.base_explainer import BaseExplainer
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class RegressionExplainer(BaseExplainer):
    """
    RegressionExplainer provides simple regression explainability analyses.

    This class implements baseline analyses:
    - feature_importance: permutation importance (uses X_test and y_test)
    - partial_dependence: simple PDP by replacing feature values with grid values
    - sensitivity_analysis: perturb features and measure avg change in prediction
    - fairness_bias: group-wise error metrics (MAE) for a sensitive column

    Results (numbers/dfs) are stored in ``self.results`` and matplotlib figures in ``self.plots``.
    """

    def __init__(self, model_interface):
        """
        Initialize the explainer.

        Args:
            model_interface: Object that provides access to model and test data (expects attributes
                ``X_test``, ``y_test``, and optionally ``y_pred``).
        """
        super().__init__(model_interface)
        self.results: Dict[str, Any] = {}
        self.plots: Dict[str, plt.Figure] = {}

    # ---------------------------
    # Helper utilities
    # ---------------------------
    def _has_predict(self) -> bool:
        """
        Check whether the wrapped model exposes a predict method.

        Returns:
            bool: True if the model has a callable ``predict`` attribute, False otherwise.
        """
        return hasattr(self.model, "predict")

    def _as_df(self, X) -> Optional[pd.DataFrame]:
        """
        Convert input X to a pandas DataFrame when possible.

        Args:
            X: Input data (DataFrame, Series, numpy array, or None).

        Returns:
            Optional[pd.DataFrame]: A copy of X as a DataFrame, or None if conversion failed or X is None.
        """
        if X is None:
            return None
        if isinstance(X, pd.DataFrame):
            return X.copy()
        if isinstance(X, pd.Series):
            return X.to_frame()
        # numpy array
        try:
            return pd.DataFrame(X)
        except Exception:
            return None

    # ---------------------------
    # Main analyses
    # ---------------------------
    def feature_importance(
        self, n_repeats: int = 10, random_state: int = 0
    ) -> Optional[pd.DataFrame]:
        """
        Compute permutation importance on ``X_test`` / ``y_test``.

        Args:
            n_repeats (int): Number of permutation repeats (default: 10).
            random_state (int): Random seed for permutation (default: 0).

        Returns:
            Optional[pd.DataFrame]: DataFrame with columns ["feature", "importance_mean", "importance_std"]
                sorted by ``importance_mean`` or ``None`` if prerequisites are missing or an error occurs.

        Side effects:
            Stores the DataFrame in ``self.results['feature_importance']`` and a bar plot in
            ``self.plots['feature_importance']``.
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
            importances = pd.DataFrame(
                {
                    "feature": list(X_test.columns),
                    "importance_mean": r.importances_mean,
                    "importance_std": r.importances_std,
                }
            ).sort_values("importance_mean", ascending=False)

            self.results["feature_importance"] = importances

            # Plot
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
        """
        Approximate partial dependence for a single feature by replacing the column values with grid values
        and computing the mean model prediction.

        Args:
            feature (Optional[str]): Feature name to analyze. If ``None``, uses top feature from permutation
                importances if present.
            grid_points (int): Number of grid points to evaluate (default: 10).

        Returns:
            Optional[pd.DataFrame]: DataFrame with columns ["value", "avg_pred"] or ``None`` if prerequisites
                are missing or an error occurs.

        Side effects:
            Stores the DataFrame in ``self.results[f'pdp_{feature}']`` and a line plot in
            ``self.plots[f'pdp_{feature}']``.
        """
        X_test = self._as_df(self.model_interface.X_test)
        if X_test is None or not self._has_predict():
            return None

        # choose feature
        if feature is None:
            fi = self.results.get("feature_importance")
            if fi is None or fi.empty:
                feature = X_test.columns[0]
            else:
                feature = fi["feature"].iloc[0]

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
                p = self.model.predict(Xbase)
                preds.append(np.mean(p))
            df = pd.DataFrame({"value": grid, "avg_pred": preds})
            self.results[f"pdp_{feature}"] = df

            # Plot
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(df["value"], df["avg_pred"], marker="o")
            ax.set_xlabel(feature)
            ax.set_ylabel("Average prediction")
            ax.set_title(
                f"Partial Dependence (approx) for feature with highest importance — {feature}"
            )
            plt.tight_layout()
            self.plots[f"pdp_{feature}"] = fig
            return df
        except Exception as e:
            self.results[f"pdp_{feature}_error"] = str(e)
            return None

    def sensitivity_analysis(
        self, features: Optional[List[str]] = None, delta: float = 0.01
    ) -> Optional[pd.DataFrame]:
        """
        Estimate sensitivity of model predictions to feature perturbations.

        For numeric features this multiplies values by (1 + delta) and (1 - delta) and measures the
        mean absolute change in prediction. For non-numeric features it uses the mode as a no-op
        perturbation.

        Args:
            features (Optional[List[str]]): List of feature names to analyze. If ``None``, analyzes all
                columns in ``X_test``.
            delta (float): Relative perturbation for numeric features (default: 0.01).

        Returns:
            Optional[pd.DataFrame]: DataFrame with columns ["feature", "mean_abs_change"] sorted descending by
                ``mean_abs_change``, or ``None`` if prerequisites are missing or an error occurs.

        Side effects:
            Stores the DataFrame in ``self.results['sensitivity']`` and a bar plot in
            ``self.plots['sensitivity']``.
        """
        X_test = self._as_df(self.model_interface.X_test)
        if X_test is None or not self._has_predict():
            return None

        if features is None:
            features = list(X_test.columns)

        results = []
        Xbase = X_test.copy()

        try:
            for feat in features:
                col = Xbase[feat]
                if np.issubdtype(col.dtype, np.number):
                    up = Xbase.copy()
                    down = Xbase.copy()
                    up[feat] = col * (1 + delta)
                    down[feat] = col * (1 - delta)
                else:
                    # for categorical/text features: change to mode (no sensible perturbation)
                    mode = col.mode().iloc[0] if not col.mode().empty else col.iloc[0]
                    up = Xbase.copy()
                    down = Xbase.copy()
                    up[feat] = mode
                    down[feat] = mode

                pred_up = self.model.predict(up)
                pred_down = self.model.predict(down)

                mean_abs_change = np.mean(np.abs(pred_up - pred_down))
                results.append({"feature": feat, "mean_abs_change": mean_abs_change})

            df = pd.DataFrame(results).sort_values("mean_abs_change", ascending=False)
            self.results["sensitivity"] = df

            # Plot
            fig, ax = plt.subplots(figsize=(8, max(3, 0.3 * len(df))))
            ax.barh(df["feature"][::-1], df["mean_abs_change"][::-1])
            ax.set_xlabel("Mean absolute change in prediction (perturbation)")
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
        """
        Compute group-wise error metrics for a sensitive column present in ``X_test``.

        Args:
            sensitive_column (str): Column in ``X_test`` indicating group membership.
            metric (str): Metric name to compute; currently ignored except for naming (default: "mae").

        Returns:
            Optional[pd.DataFrame]: Aggregated DataFrame with columns [sensitive_column, "count", "mae", "mse", "r2"]
                or ``None`` if prerequisites are missing or an error occurs.

        Side effects:
            Stores the DataFrame in ``self.results[f'fairness_{sensitive_column}']`` and a bar plot in
            ``self.plots[f'fairness_{sensitive_column}']``.
        """
        X_test = self._as_df(self.model_interface.X_test)
        y_test = self.model_interface.y_test
        y_pred = self.model_interface.y_pred

        if X_test is None or y_test is None or y_pred is None:
            return None

        if sensitive_column not in X_test.columns:
            self.results[f"fairness_{sensitive_column}"] = None
            return None

        try:
            grp = pd.DataFrame(
                {
                    "sensitive": X_test[sensitive_column],
                    "y_true": np.asarray(y_test).reshape(-1),
                    "y_pred": np.asarray(y_pred).reshape(-1),
                }
            )
            agg = (
                grp.groupby("sensitive")
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
                .rename(columns={"index": sensitive_column})
            )

            self.results[f"fairness_{sensitive_column}"] = agg

            # Plot: MAE by group
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(agg[sensitive_column].astype(str), agg["mae"])
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

    # ---------------------------
    # Convenience: run a suite
    # ---------------------------
    def run_all(self, sensitive_column: Optional[str] = None) -> None:
        """
        Run a suite of explainability analyses and store results/plots.

        Args:
            sensitive_column (Optional[str]): Sensitive column name to run fairness analysis on (default: None).

        Returns:
            None

        Side effects:
            Runs diagnostics, feature importance, partial dependence (on top feature if available),
            sensitivity analysis, and fairness bias (if ``sensitive_column`` provided). Results and plots are
            stored in ``self.results`` and ``self.plots``.
        """
        # basic prediction diagnostics
        X_test = self._as_df(self.model_interface.X_test)
        y_test = self.model_interface.y_test
        y_pred = self.model_interface.y_pred

        # diagnostics plot: y_true vs y_pred + residuals
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

                residuals = np.asarray(y_test).reshape(-1) - np.asarray(y_pred).reshape(
                    -1
                )
                axes[1].hist(residuals, bins=30, alpha=0.8)
                axes[1].set_title("Residuals distribution")
                axes[1].set_xlabel("y_true - y_pred")

                plt.tight_layout()
                self.plots["diagnostics"] = fig

                # basic metrics
                self.results["metrics"] = {
                    "mae": float(mean_absolute_error(y_test, y_pred)),
                    "mse": float(mean_squared_error(y_test, y_pred)),
                    "r2": float(r2_score(y_test, y_pred)),
                }
            except Exception as e:
                self.results["diagnostics_error"] = str(e)

        # run feature importance (best-effort)
        self.feature_importance()

        # run pdp on top feature if available
        fi = self.results.get("feature_importance")
        top_feat = None
        if isinstance(fi, pd.DataFrame) and not fi.empty:
            top_feat = fi["feature"].iloc[0]
        self.partial_dependence(feature=top_feat)

        # sensitivity
        self.sensitivity_analysis()

        # fairness if requested
        if sensitive_column is not None:
            self.fairness_bias(sensitive_column)
