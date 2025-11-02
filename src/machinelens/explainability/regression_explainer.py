"""Regression explainer model for understanding regression model behavior (refactored)."""

from __future__ import annotations

from typing import Any, Dict, Optional

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from explainability.base_explainer import BaseExplainer
from scipy import stats
from sklearn.inspection import permutation_importance
from sklearn.model_selection import RepeatedKFold, learning_curve
from statsmodels.stats.multitest import multipletests


class RegressionExplainer(BaseExplainer):
    """Provides explainability analyses for regression models (data-only version).

    Now applies Separation of Concerns:
    - Analysis functions: compute and store results only.
    - run_all: orchestrates execution and visualization order.
    """

    def __init__(self, model_interface: Any) -> None:
        """Initialize the RegressionExplainer.

        Args:
            model_interface (Any): Object providing model, data (train/test), and predictions.
        """
        super().__init__(model_interface)
        self.results: Dict[str, Any] = {}
        self.plots: Dict[str, plt.Figure] = {}

    # ---------------------------
    # Helper utilities
    # ---------------------------
    def _has_predict(self) -> bool:
        """Check if the underlying model has a callable 'predict' method.

        Returns:
            bool: True if the model implements a callable predict() method, False otherwise.
        """
        return hasattr(self.model, "predict") and callable(self.model.predict)

    def _as_df(self, X: Any) -> Optional[pd.DataFrame]:
        """Convert various input formats to a pandas DataFrame.

        Args:
            X (Any): Input data, possibly a DataFrame, Series, or NumPy array.

        Returns:
            Optional[pd.DataFrame]: A copy of X as a DataFrame, or None if conversion fails.
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
    # Analysis-only methods
    # ---------------------------
    def feature_importance(
        self, n_repeats: int = 10, random_state: int = 0
    ) -> Optional[pd.DataFrame]:
        """Compute permutation-based feature importance for regression models.

        Calculates the mean and standard deviation of feature importances using
        sklearn's permutation_importance function.

        Args:
            n_repeats (int): Number of shuffling iterations per feature.
            random_state (int): Random seed for reproducibility.

        Returns:
            Optional[pd.DataFrame]: DataFrame sorted by descending importance with columns:
                - feature: Feature name.
                - importance_mean: Average importance score.
                - importance_std: Standard deviation of importance.
                Returns None if model or data are missing.
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
            return importances
        except Exception as e:
            self.results["feature_importance_error"] = str(e)
            return None

    def analyze_residual_outliers(
        self,
        threshold: Optional[float] = None,
        alpha: float = 0.05,
        min_group_size: int = 5,
        normality_check: bool = True,
        p_adjust_method: str = "fdr_bh",
    ) -> Optional[pd.DataFrame]:
        """Analyze high-residual observations and compare features between groups.

        Identifies observations with large standardized residuals (|studentized_residual| > threshold)
        and tests whether each numeric feature differs significantly between the high- and low-residual
        groups. Uses Welch's t-test or Mann-Whitney U test based on normality, and applies
        multiple-testing correction.

        Args:
            threshold (Optional[float]): Cutoff for standardized residuals. If None, uses the 95th percentile.
            alpha (float): Significance level for hypothesis testing.
            min_group_size (int): Minimum group size required to perform comparisons.
            normality_check (bool): Whether to run Shapiro-Wilk normality tests before choosing test type.
            p_adjust_method (str): Method for p-value adjustment ('bonferroni' or 'fdr_bh').

        Returns:
            Optional[pd.DataFrame]: DataFrame with feature-level statistics, including:
                - feature: Feature name.
                - mean_high / mean_low: Group means.
                - median_high / median_low: Group medians.
                - mean_diff / median_diff: Group differences.
                - stat, p_value, adj_p_value: Test statistics and p-values.
                - effect_size: Cohen's d or rank-biserial effect size.
                - significant: Boolean flag for significance after adjustment.
                Returns None if data are missing or analysis fails.
        """
        X_test = self._as_df(self.model_interface.X_test)
        y_test = getattr(self.model_interface, "y_test", None)
        y_pred = getattr(self.model_interface, "y_pred", None)

        if X_test is None or y_test is None or y_pred is None:
            self.results["residual_outlier_analysis"] = None
            return None

        # align size (fallback)
        y_test_arr = np.asarray(y_test).ravel()
        y_pred_arr = np.asarray(y_pred).ravel()
        if len(y_test_arr) != len(y_pred_arr) or len(y_test_arr) != len(X_test):
            n = min(len(y_test_arr), len(y_pred_arr), len(X_test))
            y_test_arr = y_test_arr[:n]
            y_pred_arr = y_pred_arr[:n]
            X_test = X_test.iloc[:n]

        residuals = y_test_arr - y_pred_arr
        # Standart residuals: divide by RMSE
        rmse = np.sqrt(np.nanmean(residuals**2))
        if rmse == 0:
            std_resid = residuals  # fallback
        else:
            std_resid = residuals / rmse

        abs_std_resid = np.abs(std_resid)

        # default: 95th percentile of *standardized* residuals
        if threshold is None:
            threshold = float(np.percentile(abs_std_resid, 95))

        mask_high = abs_std_resid > threshold
        mask_series = pd.Series(mask_high, index=X_test.index)

        X_high = X_test.loc[mask_series]
        X_low = X_test.loc[~mask_series]

        numeric_cols = X_test.select_dtypes(include=[np.number]).columns.tolist()

        pvals = []
        features_meta = []

        for col in numeric_cols:
            high_vals = X_high[col].dropna().values
            low_vals = X_low[col].dropna().values

            n_high = len(high_vals)
            n_low = len(low_vals)

            if (n_high < min_group_size) or (n_low < min_group_size):
                continue

            mean_high = float(np.nanmean(high_vals)) if n_high > 0 else np.nan
            mean_low = float(np.nanmean(low_vals)) if n_low > 0 else np.nan
            median_high = float(np.nanmedian(high_vals)) if n_high > 0 else np.nan
            median_low = float(np.nanmedian(low_vals)) if n_low > 0 else np.nan
            mean_diff = mean_high - mean_low
            median_diff = median_high - median_low

            # decide test: prefer Welch ttest unless clear non-normal/very small samples
            chosen_test = "welch_ttest"
            stat = np.nan
            pval = 1.0
            effect = np.nan
            try:
                use_mann = False
                if normality_check and 3 <= n_high <= 5000 and 3 <= n_low <= 5000:
                    sh_high = stats.shapiro(high_vals)
                    sh_low = stats.shapiro(low_vals)
                    # if either group deviates from normal at alpha=0.05 -> use non-parametric
                    if (sh_high.pvalue < 0.05) or (sh_low.pvalue < 0.05):
                        use_mann = True
                else:
                    # small samples -> be conservative
                    if n_high < 20 or n_low < 20:
                        use_mann = True
            except Exception:
                use_mann = True

            if use_mann:
                chosen_test = "mannwhitney"
                try:
                    u_stat, pval = stats.mannwhitneyu(
                        high_vals, low_vals, alternative="two-sided"
                    )
                    stat = float(u_stat)
                    # compute rank-biserial effect size: 2U/(n1*n2) - 1
                    try:
                        rb = (2.0 * u_stat) / (n_high * n_low) - 1.0
                    except Exception:
                        rb = np.nan
                    effect = float(rb)
                except Exception:
                    stat, pval, effect = np.nan, 1.0, np.nan
            else:
                chosen_test = "welch_ttest"
                try:
                    t_res = stats.ttest_ind(
                        high_vals, low_vals, equal_var=False, nan_policy="omit"
                    )
                    stat = (
                        float(t_res.statistic)
                        if t_res.statistic is not None
                        else np.nan
                    )
                    pval = float(t_res.pvalue) if t_res.pvalue is not None else 1.0
                    # Cohen's d (pooled sd). For unequal variances you can still report d as general effect size.
                    s1 = np.nanstd(high_vals, ddof=1)
                    s2 = np.nanstd(low_vals, ddof=1)
                    # pooled sd (Cohen's d)
                    try:
                        pooled_sd = np.sqrt(
                            ((n_high - 1) * s1**2 + (n_low - 1) * s2**2)
                            / max(1, (n_high + n_low - 2))
                        )
                        cohen_d = mean_diff / pooled_sd if pooled_sd > 0 else np.nan
                    except Exception:
                        cohen_d = np.nan
                    effect = float(cohen_d)
                except Exception:
                    stat, pval, effect = np.nan, 1.0, np.nan

            pvals.append(pval)
            features_meta.append(
                {
                    "feature": col,
                    "mean_high": mean_high,
                    "mean_low": mean_low,
                    "median_high": median_high,
                    "median_low": median_low,
                    "mean_diff": mean_diff,
                    "median_diff": median_diff,
                    "stat": stat,
                    "p_value": pval,
                    "test": chosen_test,
                    "n_high": int(n_high),
                    "n_low": int(n_low),
                    "effect_size": effect,
                }
            )

        if len(pvals) > 0:
            reject, pvals_adj, _, _ = multipletests(
                pvals, alpha=alpha, method=p_adjust_method
            )
        else:
            pvals_adj = []
            reject = []

        for row, adj_p, rej in zip(features_meta, pvals_adj, reject):
            row["adj_p_value"] = float(adj_p)
            row["significant"] = bool(rej)

        res_df = pd.DataFrame(features_meta)
        if not res_df.empty:
            res_df = res_df.sort_values("adj_p_value").reset_index(drop=True)

        self.results["residual_outlier_analysis"] = {
            "threshold": float(threshold),
            "alpha": float(alpha),
            "min_group_size": int(min_group_size),
            "p_adjust_method": p_adjust_method,
            "results_df": res_df,
        }

        return res_df

    # ---------------------------
    # Orchestration and Visualization
    # ---------------------------
    def run_all(self) -> None:
        """Run full regression explainability pipeline.

        Executes all analyses and generates diagnostic plots, including:
        - Actual vs Predicted
        - Residuals vs Actual and Predicted
        - Residual Distribution
        - Normal Q-Q plot
        - Residual-outlier analysis and violin/box plots
        - Learning curve (if training data available)

        Returns:
            None
        """
        X_test = self._as_df(self.model_interface.X_test)
        y_test = getattr(self.model_interface, "y_test", None)
        y_pred = getattr(self.model_interface, "y_pred", None)

        plt.close("all")
        self.plots = {}

        if X_test is None or y_test is None or y_pred is None:
            return None

        y_test_arr = np.asarray(y_test).ravel()
        y_pred_arr = np.asarray(y_pred).ravel()
        if len(y_test_arr) != len(y_pred_arr) or len(y_test_arr) != len(X_test):
            n = min(len(y_test_arr), len(y_pred_arr), len(X_test))
            y_test_arr = y_test_arr[:n]
            y_pred_arr = y_pred_arr[:n]
            X_test = X_test.iloc[:n]

        residuals = y_test_arr - y_pred_arr
        abs_residuals = np.abs(residuals)

        vmax_abs_residuals = np.percentile(abs_residuals, 97)
        norm_map = mcolors.Normalize(vmin=0, vmax=vmax_abs_residuals)

        # ----------- Actual vs Predicted Values -----------
        fig1, ax1 = plt.subplots(figsize=(6, 5))
        sc1 = ax1.scatter(
            y_test_arr, y_pred_arr, c=abs_residuals, cmap="plasma", s=20, norm=norm_map
        )
        fig1.colorbar(sc1, ax=ax1, label="|Residual|")
        min_val, max_val = float(np.min(y_test_arr)), float(np.max(y_test_arr))
        ax1.plot([min_val, max_val], [min_val, max_val], "k--")
        ax1.set_title("Actual vs Predicted Values")
        ax1.set_xlabel("Actual (y_true)")
        ax1.set_ylabel("Predicted (y_pred)")
        plt.tight_layout()
        self.plots["actual_vs_predicted"] = fig1
        plt.close(fig1)

        # ----------- Residuals vs Actual Value -----------
        fig2, ax2 = plt.subplots(figsize=(6, 5))
        sc2 = ax2.scatter(
            y_test_arr, residuals, c=abs_residuals, cmap="plasma", s=20, norm=norm_map
        )
        fig2.colorbar(sc2, ax=ax2, label="|Residual|")
        ax2.axhline(0, color="k", linestyle="--")
        ax2.set_title("Residuals vs Actual Value")
        ax2.set_xlabel("Actual (y_true)")
        ax2.set_ylabel("Residual (Error)")
        plt.tight_layout()
        self.plots["residuals_vs_actual"] = fig2
        plt.close(fig2)

        # -----------  Residuals vs Predicted -----------
        fig5, ax5 = plt.subplots(figsize=(6, 5))
        sc5 = ax5.scatter(
            y_pred_arr, residuals, c=abs_residuals, cmap="plasma", s=5, norm=norm_map
        )
        fig5.colorbar(sc5, ax=ax5, label="|Residual|")
        ax5.axhline(0, color="k", linestyle="--")
        ax5.set_title("Residuals vs Predicted")
        ax5.set_xlabel("Predicted (y_pred)")
        ax5.set_ylabel("Residual (Error)")
        plt.tight_layout()
        self.plots["residuals_vs_predicted"] = fig5
        plt.close(fig5)

        # ----------- Residual Distribution -----------
        fig3, ax3 = plt.subplots(figsize=(6, 5))
        _, bins, patches = ax3.hist(residuals, bins=30, alpha=0.8, edgecolor="black")
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        colors_hist = cm.plasma(  # type: ignore [attr-defined]
            np.clip(np.abs(bin_centers) / max(1e-9, vmax_abs_residuals), 0, 1)
        )
        for patch, color in zip(patches, colors_hist):  # type: ignore [arg-type]
            patch.set_facecolor(color)
        mean_res = float(np.mean(residuals))
        ax3.axvline(
            mean_res, color="red", linestyle="--", label=f"Mean = {mean_res:.2f}"
        )
        ax3.axvline(0, color="black", linestyle="--", label="Zero")
        ax3.set_title("Residuals Distribution")
        ax3.set_xlabel("Residual")
        ax3.set_ylabel("Frequency")
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        plt.tight_layout()
        self.plots["residual_distribution"] = fig3
        plt.close(fig3)

        # ---------- Normal Q–Q Plot (Enhanced) -----------
        fig4, ax4 = plt.subplots(figsize=(6, 5))
        (osm, osr), (slope, intercept, r) = stats.probplot(residuals, dist="norm")
        dist = np.abs(osr - (intercept + slope * osm))
        colors_qq = cm.plasma(  # type: ignore [attr-defined]
            np.clip(dist / (np.max(dist) if np.max(dist) != 0 else 1.0), 0, 1)
        )
        ax4.scatter(osm, osr, c=colors_qq, s=30)
        ax4.plot(osm, intercept + slope * osm, "k--", label="Fit line")
        ax4.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax4.set_title("Normal Q–Q Plot of Residuals")
        ax4.set_xlabel("Theoretical Quantiles")
        ax4.set_ylabel("Sample Quantiles")
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        plt.tight_layout()
        self.plots["qq_plot"] = fig4
        plt.close(fig4)

        # ------------------ Residual-outlier analysis + plotting ------------------
        try:
            _ = self.analyze_residual_outliers(
                threshold=None, alpha=0.05, min_group_size=5, normality_check=True
            )
        except Exception as e:
            self.results["residual_outlier_analysis_error"] = str(e)

        analysis = None
        if isinstance(self.results.get("residual_outlier_analysis"), dict):
            analysis = self.results.get("residual_outlier_analysis")
        else:
            if hasattr(self, "results_residuals") and isinstance(
                getattr(self, "results_residuals"), dict
            ):
                analysis = getattr(self, "results_residuals").get(
                    "residual_outlier_analysis"
                )

        X_test_raw = getattr(self.model_interface, "X_test", None)
        X_test = self._as_df(X_test_raw)
        if X_test is None or not isinstance(X_test, pd.DataFrame):
            X_test = pd.DataFrame()

        y_test = getattr(self.model_interface, "y_test", None)
        y_pred = getattr(self.model_interface, "y_pred", None)

        abs_std_resid = None
        try:
            if y_test is not None and y_pred is not None:
                y_test_arr = np.asarray(y_test).ravel()
                y_pred_arr = np.asarray(y_pred).ravel()
                n = (
                    min(len(y_test_arr), len(y_pred_arr), len(X_test))
                    if len(X_test) > 0
                    else min(len(y_test_arr), len(y_pred_arr))
                )
                y_test_arr = y_test_arr[:n]
                y_pred_arr = y_pred_arr[:n]
                residuals = y_test_arr - y_pred_arr
                rmse = np.sqrt(np.nanmean(residuals**2))
                std_resid = residuals / (rmse if rmse != 0 else 1.0)
                abs_std_resid = np.abs(std_resid)
        except Exception:
            abs_std_resid = None

        if not analysis:
            self.results["residual_outlier_plots"] = []
        else:
            res_df = analysis.get("results_df")
            if res_df is None or getattr(res_df, "empty", True):
                self.results["residual_outlier_plots"] = []
            else:
                features_to_plot = res_df.loc[res_df["significant"], "feature"].tolist()
                max_plots = 10
                if len(features_to_plot) > max_plots:
                    sort_col = (
                        "adj_p_value" if "adj_p_value" in res_df.columns else "p_value"
                    )
                    features_to_plot = (
                        res_df.sort_values(sort_col)
                        .loc[res_df["significant"], "feature"]
                        .head(max_plots)
                        .tolist()
                    )

                created = []
                threshold = float(analysis.get("threshold", np.nan))

                X_test_local = X_test.copy()
                if abs_std_resid is None or len(abs_std_resid) != len(X_test_local):
                    try:
                        raw_resid = (
                            np.asarray(y_test).ravel() - np.asarray(y_pred).ravel()
                        )
                        abs_std_resid = np.abs(raw_resid)[: len(X_test_local)]
                    except Exception:
                        abs_std_resid = np.zeros(len(X_test_local))
                else:
                    if len(abs_std_resid) > len(X_test_local):
                        abs_std_resid = abs_std_resid[: len(X_test_local)]
                    elif len(abs_std_resid) < len(X_test_local):
                        X_test_local = X_test_local.iloc[: len(abs_std_resid)].copy()

                mask_high = pd.Series(
                    abs_std_resid > threshold, index=X_test_local.index
                )

                # ----- Graphs ------
                errors = {}
                for col in features_to_plot:
                    try:
                        if col not in X_test_local.columns:
                            continue
                        if not np.issubdtype(X_test_local[col].dtype, np.number):
                            continue

                        plot_df = pd.DataFrame(
                            {
                                col: X_test_local[col],
                                "Residual_Group": np.where(
                                    mask_high, "High Residual", "Low Residual"
                                ),
                            }
                        ).dropna()

                        if plot_df.empty:
                            continue

                        row = res_df[res_df["feature"] == col]
                        pval = (
                            float(row["p_value"].values[0])
                            if not row.empty and "p_value" in row
                            else np.nan
                        )
                        adj_p = (
                            float(row["adj_p_value"].values[0])
                            if not row.empty and "adj_p_value" in row
                            else np.nan
                        )
                        eff = (
                            float(row["effect_size"].values[0])
                            if not row.empty and "effect_size" in row
                            else np.nan
                        )
                        n_high = (
                            int(row["n_high"].values[0])
                            if not row.empty and "n_high" in row
                            else (plot_df["Residual_Group"] == "High Residual").sum()
                        )
                        n_low = (
                            int(row["n_low"].values[0])
                            if not row.empty and "n_low" in row
                            else (plot_df["Residual_Group"] == "Low Residual").sum()
                        )
                        test_name = (
                            str(row["test"].values[0])
                            if not row.empty and "test" in row
                            else ""
                        )

                        sns.reset_defaults()
                        sns.set_style("whitegrid")
                        sns.set_context("talk", font_scale=0.9)

                        fig, ax = plt.subplots(figsize=(6.5, 5.5), layout="constrained")
                        order = ["Low Residual", "High Residual"]
                        palette = {
                            "Low Residual": "#006AFFAD",
                            "High Residual": "#FF0000AD",
                        }

                        sns.violinplot(
                            x="Residual_Group",
                            y=col,
                            data=plot_df,
                            order=order,
                            ax=ax,
                            palette=palette,
                            cut=0,
                            linewidth=1,
                            alpha=0.7,
                        )

                        sns.boxplot(
                            x="Residual_Group",
                            y=col,
                            data=plot_df,
                            order=order,
                            ax=ax,
                            width=0.12,
                            showcaps=True,
                            boxprops={
                                "facecolor": "white",
                                "zorder": 3,
                                "linewidth": 1,
                                "edgecolor": "black",
                            },
                            medianprops={"color": "black", "linewidth": 2, "zorder": 4},
                            whiskerprops={"linewidth": 1},
                            showfliers=False,
                        )

                        sns.stripplot(
                            x="Residual_Group",
                            y=col,
                            data=plot_df,
                            order=order,
                            ax=ax,
                            dodge=True,
                            alpha=0.5,
                            zorder=2,
                            size=3,
                            jitter=0.15,
                            color="k",
                        )

                        ax.set_box_aspect(0.8)
                        ax.set_title(
                            f"{col} — High vs Low residuals",
                            fontsize=13,
                            weight="bold",
                            y=1.08,
                        )
                        subtitle = f"{test_name} | p={pval:.2e} | adj p={adj_p:.2e} | effect={eff:.3g} | nH={n_high}, nL={n_low}"
                        ax.text(
                            0.5,
                            1.02,
                            subtitle,
                            ha="center",
                            va="bottom",
                            transform=ax.transAxes,
                            fontsize=9,
                            alpha=0.8,
                        )

                        ax.set_xlabel("")
                        ax.set_ylabel(col, fontsize=11)
                        ax.set_xticklabels(order, fontsize=10)

                        ax.annotate(
                            f"Threshold (std resid) = {threshold:.2f}",
                            xy=(0.99, 0.01),
                            xycoords="axes fraction",
                            ha="right",
                            va="bottom",
                            fontsize=8,
                            alpha=0.6,
                        )

                        ax.yaxis.grid(True, linestyle="--", alpha=0.5)
                        ax.xaxis.grid(False)
                        sns.despine(ax=ax, trim=True)

                        key = f"residual_outlier_{col}"
                        self.plots[key] = fig
                        plt.close(fig)
                        created.append(key)

                    except Exception as e_col:
                        errors[col] = str(e_col)

                self.results["residual_outlier_plots"] = created
                if errors:
                    self.results["residual_outlier_plot_errors"] = errors

        # ----------- Learning Curve (if training data available) -----------
        X_train = self._as_df(getattr(self.model_interface, "X_train", None))
        y_train = getattr(self.model_interface, "y_train", None)

        if (
            X_train is not None
            and y_train is not None
            and hasattr(self.model, "fit")
            and hasattr(self.model, "score")
        ):
            try:
                sns.set_theme(style="darkgrid", palette="deep")

                rkf = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)

                train_sizes, train_scores, test_scores = learning_curve(
                    self.model,
                    X_train,
                    y_train,
                    cv=rkf,
                    n_jobs=-1,
                    scoring="r2",
                    train_sizes=np.linspace(0.1, 1.0, 5),
                )

                train_mean = np.mean(train_scores, axis=1)
                test_mean = np.mean(test_scores, axis=1)
                train_std = np.std(train_scores, axis=1)
                test_std = np.std(test_scores, axis=1)

                fig_lc, ax_lc = plt.subplots(figsize=(8, 6))

                ax_lc.plot(train_sizes, train_mean, "o-", label="Training score (Avg)")
                ax_lc.plot(
                    train_sizes, test_mean, "o-", label="Cross-validation score (Avg)"
                )

                ax_lc.fill_between(
                    train_sizes,
                    train_mean - train_std,
                    train_mean + train_std,
                    alpha=0.15,
                )

                ax_lc.fill_between(
                    train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.15
                )

                ax_lc.set_title(
                    "Learning Curve (R² Score with Standart Deviation)",
                    fontsize=16,
                    fontweight="bold",
                )
                ax_lc.set_xlabel("Training Examples", fontsize=12)
                ax_lc.set_ylabel("R² Score", fontsize=12)

                ax_lc.axhline(
                    y=0,
                    color="black",
                    linestyle="--",
                    linewidth=0.7,
                    label="R² = 0 (Baseline)",
                )

                ax_lc.legend(fontsize=11)
                ax_lc.grid(True)

                plt.tight_layout()
                self.plots["learning_curve"] = fig_lc
                plt.close(fig_lc)

                sns.reset_defaults()

            except Exception as e:
                self.results["learning_curve_error"] = str(e)

        return None
