from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


class RegressionDiagnostics:
    """
    Diagnostics suite for regression models.

    This class provides a comprehensive set of analytical and visual tools
    to evaluate the performance of regression models, including residual
    analysis, error distributions, and learning curves.

    Attributes
    ----------
    model_interface : Any
        The interface containing the model and data splits.
    model : Any
        The predictive model extracted from the interface.
    results : dict
        Dictionary storing numerical and tabular analysis results.
    results : dict
        Dictionary storing numerical and tabular analysis results.
    """

    def __init__(self, model_interface: Any) -> None:
        """
        Initialize the RegressionDiagnostics object.

        Parameters
        ----------
        model_interface : Any
            An object containing the regression model and dataset splits.
        """
        self.model_interface = model_interface
        self.model = getattr(model_interface, "model", None)
        self.results: Dict[str, Any] = {}

    def _has_predict(self) -> bool:
        if self.model is None:
            return False
        return hasattr(self.model, "predict") and callable(self.model.predict)

    def _as_df(self, X: Any) -> Optional[pd.DataFrame]:
        if X is None:
            return None
        if isinstance(X, pd.DataFrame):
            return X.copy()
        if isinstance(X, pd.Series):
            return X.to_frame()
        if isinstance(X, np.ndarray):
            return pd.DataFrame(X)
        return None

    def analyze_residual_outliers(
        self,
        threshold: Optional[float] = None,
        alpha: float = 0.05,
        min_group_size: int = 5,
        normality_check: bool = True,
        p_adjust_method: str = "fdr_bh",
    ) -> Optional[pd.DataFrame]:
        """
        Analyze feature distributions between normal and outlier residuals.

        Performs statistical tests on each numeric feature to determine if its
        distribution significantly differs between samples with high residuals
        (outliers) and low residuals.

        Parameters
        ----------
        threshold : float, optional
            The absolute standardized residual value above which a sample is
            considered an outlier. If None, the 95th percentile is used.
        alpha : float, default=0.05
            Significance level for the statistical tests.
        min_group_size : int, default=5
            Minimum number of samples required in both groups to perform the test.
        normality_check : bool, default=True
            Whether to check for normality to decide between parametric and
            non-parametric tests.
        p_adjust_method : str, default="fdr_bh"
            Method used to adjust p-values for multiple testing.

        Returns
        -------
        pandas.DataFrame or None
            DataFrame containing the statistical analysis results per feature,
            sorted by adjusted p-value. Returns None if data is missing.
        """
        X_test = self._as_df(getattr(self.model_interface, "X_test", None))
        y_test = getattr(self.model_interface, "y_test", None)
        y_pred = getattr(self.model_interface, "y_pred", None)

        if X_test is None or y_test is None or y_pred is None:
            self.results["residual_outlier_analysis"] = None
            return None

        assert X_test is not None

        y_test_arr = np.asarray(y_test).ravel()
        y_pred_arr = np.asarray(y_pred).ravel()
        if len(y_test_arr) != len(y_pred_arr) or len(y_test_arr) != len(X_test):
            n = min(len(y_test_arr), len(y_pred_arr), len(X_test))
            y_test_arr = y_test_arr[:n]
            y_pred_arr = y_pred_arr[:n]
            X_test = X_test.iloc[:n]

        assert X_test is not None

        residuals = y_test_arr - y_pred_arr
        rmse = np.sqrt(np.nanmean(residuals**2))
        if rmse == 0:
            std_resid = residuals
        else:
            std_resid = residuals / rmse

        abs_std_resid = np.abs(std_resid)

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

            try:
                use_mann = False
                if normality_check and 3 <= n_high <= 5000 and 3 <= n_low <= 5000:
                    sh_high = stats.shapiro(high_vals)
                    sh_low = stats.shapiro(low_vals)
                    if (sh_high.pvalue < 0.05) or (sh_low.pvalue < 0.05):
                        use_mann = True
                else:
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
                    s1 = np.nanstd(high_vals, ddof=1)
                    s2 = np.nanstd(low_vals, ddof=1)
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

        for row, adj_p, rej in zip(features_meta, pvals_adj, reject, strict=False):
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

    # ------------------------------------------------------------------
    # Training-data diagnostics
    # ------------------------------------------------------------------

    def _compute_lowess(
        self, x: np.ndarray, y: np.ndarray, frac: float = 0.6667, n_grid: int = 200
    ) -> Dict[str, np.ndarray]:
        """Compute LOWESS smoothed curve with bootstrap 95 % confidence band.

        Parameters
        ----------
        x : np.ndarray
            Predictor values.
        y : np.ndarray
            Response values.
        frac : float
            LOWESS smoothing fraction.
        n_grid : int
            Number of grid points for the output curve.

        Returns
        -------
        dict
            ``x_smooth``, ``y_smooth``, ``ci_lower``, ``ci_upper`` arrays.
        """
        from statsmodels.nonparametric.smoothers_lowess import lowess as sm_lowess

        sort_idx = np.argsort(x)
        xs, ys = x[sort_idx], y[sort_idx]

        smooth = sm_lowess(ys, xs, frac=frac, return_sorted=True)
        x_smooth = smooth[:, 0]
        y_smooth = smooth[:, 1]

        # Bootstrap CI (50 resamples for speed)
        rng = np.random.default_rng(0)
        n = len(xs)
        n_boot = 50
        boot_curves = np.empty((n_boot, len(x_smooth)))
        for b in range(n_boot):
            idx = rng.choice(n, size=n, replace=True)
            bx, by = x[idx], y[idx]
            bsort = np.argsort(bx)
            bsmooth = sm_lowess(by[bsort], bx[bsort], frac=frac, return_sorted=True)
            boot_curves[b] = np.interp(x_smooth, bsmooth[:, 0], bsmooth[:, 1])

        ci_lower = np.percentile(boot_curves, 2.5, axis=0)
        ci_upper = np.percentile(boot_curves, 97.5, axis=0)

        return {
            "x_smooth": x_smooth,
            "y_smooth": y_smooth,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
        }

    def _ensure_training_base(self) -> Optional[Dict[str, Any]]:
        """Return the training_diagnostics dict, creating it with base arrays if needed.

        Populates ``X_train``, ``y_train``, ``y_pred_train``, ``residuals``,
        and ``std_residuals`` on first call.

        Returns
        -------
        dict or None
            The ``training_diagnostics`` sub-dict, or None when data is missing.
        """
        td = self.results.get("training_diagnostics")
        if td is not None:
            return td

        model = self.model
        X_train = self._as_df(getattr(self.model_interface, "X_train", None))
        y_train = getattr(self.model_interface, "y_train", None)

        if model is None or X_train is None or y_train is None:
            return None
        if not (hasattr(model, "predict") and callable(model.predict)):
            return None

        y_train_arr = np.asarray(y_train).ravel()

        if len(y_train_arr) != len(X_train):
            n = min(len(y_train_arr), len(X_train))
            y_train_arr = y_train_arr[:n]
            X_train = X_train.iloc[:n]

        y_pred_train = np.asarray(model.predict(X_train)).ravel()
        residuals = y_train_arr - y_pred_train

        rmse = np.sqrt(np.nanmean(residuals**2))
        std_residuals = residuals / rmse if rmse != 0 else residuals

        td = {
            "X_train": X_train,
            "y_train": y_train_arr,
            "y_pred_train": y_pred_train,
            "residuals": residuals,
            "std_residuals": std_residuals,
        }
        self.results["training_diagnostics"] = td
        return td

    def compute_training_residuals(self) -> None:
        """Compute base training residuals and store them.

        Populates ``training_diagnostics`` with ``X_train``, ``y_train``,
        ``y_pred_train``, ``residuals``, ``std_residuals``, and
        ``sqrt_abs_std_resid``.
        """
        td = self._ensure_training_base()
        if td is None:
            return

        std_residuals = td["std_residuals"]
        td["sqrt_abs_std_resid"] = np.sqrt(np.abs(std_residuals))

    def compute_leverage(self) -> None:
        """Compute leverage (hat values) from ``X_train``.

        Adds ``leverage`` to ``training_diagnostics``.
        """
        td = self._ensure_training_base()
        if td is None:
            return

        X_train = td["X_train"]
        n = len(X_train)

        X_mat = np.asarray(X_train, dtype=float)
        X_aug = np.column_stack([np.ones(n), X_mat])
        try:
            gram_inv = np.linalg.pinv(X_aug.T @ X_aug)
            hat_diag = np.sum((X_aug @ gram_inv) * X_aug, axis=1)
        except Exception:
            hat_diag = np.full(n, np.nan)

        td["leverage"] = hat_diag

    def compute_cooks_distance(self) -> None:
        """Compute Cook's Distance from leverage and standardized residuals.

        Requires ``leverage`` and ``std_residuals`` in ``training_diagnostics``.
        Adds ``cooks_distance``.
        """
        td = self._ensure_training_base()
        if td is None:
            return

        leverage = td.get("leverage")
        std_residuals = td.get("std_residuals")
        X_train = td.get("X_train")

        if leverage is None:
            self.compute_leverage()
            leverage = td.get("leverage")

        if leverage is None or std_residuals is None or X_train is None:
            return

        p_eff = X_train.shape[1] + 1  # including intercept
        denom = p_eff * (1 - leverage) ** 2
        denom = np.where(denom == 0, np.nan, denom)
        td["cooks_distance"] = (std_residuals**2 * leverage) / denom

    def compute_linearity_lowess(self, frac: float = 0.6667) -> None:
        """Compute LOWESS + CI for the linearity plot (residuals vs. fitted).

        Adds ``linearity_lowess`` to ``training_diagnostics``.

        Parameters
        ----------
        frac : float, default=0.6667
            LOWESS smoothing bandwidth fraction.
        """
        td = self._ensure_training_base()
        if td is None:
            return

        y_pred_train = td["y_pred_train"]
        residuals = td["residuals"]

        try:
            td["linearity_lowess"] = self._compute_lowess(
                y_pred_train, residuals, frac=frac
            )
        except Exception:
            td["linearity_lowess"] = None

    def compute_scale_location_lowess(self, frac: float = 0.6667) -> None:
        """Compute LOWESS + CI for the scale-location plot.

        Adds ``scale_loc_lowess`` to ``training_diagnostics``.

        Parameters
        ----------
        frac : float, default=0.6667
            LOWESS smoothing bandwidth fraction.
        """
        td = self._ensure_training_base()
        if td is None:
            return

        y_pred_train = td["y_pred_train"]
        sqrt_abs = td.get("sqrt_abs_std_resid")
        if sqrt_abs is None:
            sqrt_abs = np.sqrt(np.abs(td["std_residuals"]))
            td["sqrt_abs_std_resid"] = sqrt_abs

        try:
            td["scale_loc_lowess"] = self._compute_lowess(
                y_pred_train, sqrt_abs, frac=frac
            )
        except Exception:
            td["scale_loc_lowess"] = None

    def compute_leverage_lowess(self, frac: float = 0.6667) -> None:
        """Compute LOWESS + CI for the leverage plot (std resid vs. hat values).

        Adds ``leverage_lowess`` to ``training_diagnostics``.

        Parameters
        ----------
        frac : float, default=0.6667
            LOWESS smoothing bandwidth fraction.
        """
        td = self._ensure_training_base()
        if td is None:
            return

        leverage = td.get("leverage")
        std_residuals = td.get("std_residuals")

        if leverage is None:
            self.compute_leverage()
            leverage = td.get("leverage")

        if leverage is None or std_residuals is None:
            return

        valid = ~np.isnan(leverage)
        try:
            td["leverage_lowess"] = self._compute_lowess(
                leverage[valid], std_residuals[valid], frac=frac
            )
        except Exception:
            td["leverage_lowess"] = None

    def compute_vif(self, max_features: int = 15) -> None:
        """Compute Variance Inflation Factor for the top features.

        Adds ``vif`` (dict mapping feature name → VIF value) to
        ``training_diagnostics``.

        Parameters
        ----------
        max_features : int, default=15
            Maximum number of features to compute VIF for.
            Features are selected by highest variance.
        """
        from statsmodels.stats.outliers_influence import variance_inflation_factor

        td = self._ensure_training_base()
        if td is None:
            return

        X_train = td["X_train"]
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()

        if len(numeric_cols) > max_features:
            variances = X_train[numeric_cols].var().sort_values(ascending=False)
            numeric_cols = variances.head(max_features).index.tolist()

        vif_data: Dict[str, float] = {}
        if len(numeric_cols) >= 2:
            X_vif = X_train[numeric_cols].dropna()
            X_vif_arr = np.asarray(X_vif, dtype=float)
            X_vif_aug = np.column_stack([np.ones(len(X_vif_arr)), X_vif_arr])
            for i, col in enumerate(numeric_cols):
                try:
                    vif_val = variance_inflation_factor(X_vif_aug, i + 1)
                    vif_data[col] = float(vif_val)
                except Exception:
                    vif_data[col] = float("nan")

        td["vif"] = vif_data

    def compute_training_qq(self) -> None:
        """Compute Q-Q plot coordinates with 95 % CI envelope for training residuals.

        Adds ``qq_osm``, ``qq_osr``, ``qq_slope``, ``qq_intercept``,
        ``qq_r``, ``qq_ci_lower``, ``qq_ci_upper`` to ``training_diagnostics``.
        """
        td = self._ensure_training_base()
        if td is None:
            return

        std_residuals = td["std_residuals"]

        (qq_osm, qq_osr), (qq_slope, qq_intercept, qq_r) = stats.probplot(
            std_residuals, dist="norm"
        )

        n_pts = len(qq_osm)
        order = np.arange(1, n_pts + 1)
        z_vals = qq_osm
        se = (1.0 / stats.norm.pdf(z_vals)) * np.sqrt(
            order * (n_pts - order + 1) / ((n_pts + 1) ** 2 * (n_pts + 2))
        )

        td["qq_osm"] = qq_osm
        td["qq_osr"] = qq_osr
        td["qq_slope"] = qq_slope
        td["qq_intercept"] = qq_intercept
        td["qq_r"] = qq_r
        td["qq_ci_lower"] = z_vals - 1.96 * se
        td["qq_ci_upper"] = z_vals + 1.96 * se

    # ------------------------------------------------------------------
    # Test-data diagnostics (LOWESS enhancements)
    # ------------------------------------------------------------------

    def compute_test_linearity_lowess(self, frac: float = 0.6667) -> None:
        """Compute LOWESS + CI for the test linearity plot.

        Adds ``linearity_lowess`` to ``residuals_data``.

        Parameters
        ----------
        frac : float, default=0.6667
            LOWESS smoothing bandwidth fraction.
        """
        rd = self.results.get("residuals_data")
        if not rd:
            return

        y_pred = rd["y_pred"]
        residuals = rd["residuals"]

        try:
            rd["linearity_lowess"] = self._compute_lowess(y_pred, residuals, frac=frac)
        except Exception:
            rd["linearity_lowess"] = None

    def compute_test_scale_location_lowess(self, frac: float = 0.6667) -> None:
        """Compute LOWESS + CI for the test scale-location plot.

        Adds ``scale_loc_lowess`` to ``residuals_data``.

        Parameters
        ----------
        frac : float, default=0.6667
            LOWESS smoothing bandwidth fraction.
        """
        rd = self.results.get("residuals_data")
        if not rd:
            return

        y_pred = rd["y_pred"]
        residuals = rd["residuals"]
        rmse = np.sqrt(np.nanmean(residuals**2))
        std_resid = residuals / rmse if rmse != 0 else residuals
        sqrt_abs = np.sqrt(np.abs(std_resid))

        try:
            rd["scale_loc_lowess"] = self._compute_lowess(y_pred, sqrt_abs, frac=frac)
        except Exception:
            rd["scale_loc_lowess"] = None

    def run_all(self) -> None:
        """
        Execute the full suite of regression diagnostics calculations.

        Calculates arrays and metrics for predictions versus actuals, residual
        distributions, Q-Q plots, outlier analysis, and training-data diagnostics.
        Results are stored in the `results` attribute as numerical data structures.
        """
        X_test_raw = getattr(self.model_interface, "X_test", None)
        X_test = self._as_df(X_test_raw)
        y_test = getattr(self.model_interface, "y_test", None)
        y_pred = getattr(self.model_interface, "y_pred", None)

        if X_test is None or y_test is None or y_pred is None:
            return None

        assert X_test is not None

        y_test_arr = np.asarray(y_test).ravel()
        y_pred_arr = np.asarray(y_pred).ravel()
        if len(y_test_arr) != len(y_pred_arr) or len(y_test_arr) != len(X_test):
            n = min(len(y_test_arr), len(y_pred_arr), len(X_test))
            y_test_arr = y_test_arr[:n]
            y_pred_arr = y_pred_arr[:n]
            X_test = X_test.iloc[:n]

        assert X_test is not None

        residuals = y_test_arr - y_pred_arr
        abs_residuals = np.abs(residuals)

        # Calculate Q-Q plot coordinates (test data)
        (osm, osr), (slope, intercept, r) = stats.probplot(residuals, dist="norm")

        # Q-Q CI for test data too
        n_pts = len(osm)
        order = np.arange(1, n_pts + 1)
        z_vals = osm
        se = (1.0 / stats.norm.pdf(z_vals)) * np.sqrt(
            order * (n_pts - order + 1) / ((n_pts + 1) ** 2 * (n_pts + 2))
        )

        self.results["residuals_data"] = {
            "y_test": y_test_arr,
            "y_pred": y_pred_arr,
            "residuals": residuals,
            "abs_residuals": abs_residuals,
            "X_test": X_test,
            "qq_osm": osm,
            "qq_osr": osr,
            "qq_slope": slope,
            "qq_intercept": intercept,
            "qq_r": r,
            "qq_ci_lower": z_vals - 1.96 * se,
            "qq_ci_upper": z_vals + 1.96 * se,
        }

        # --- Test-set LOWESS diagnostics ---
        self.compute_test_linearity_lowess()
        self.compute_test_scale_location_lowess()

        try:
            self.analyze_residual_outliers(
                threshold=None, alpha=0.05, min_group_size=5, normality_check=True
            )
        except Exception as e:
            self.results["residual_outlier_analysis_error"] = str(e)

        # Training-data diagnostics (individual methods)
        train_methods: List[Callable[[], Any]] = [
            self.compute_training_residuals,
            self.compute_leverage,
            self.compute_cooks_distance,
            self.compute_linearity_lowess,
            self.compute_scale_location_lowess,
            self.compute_leverage_lowess,
            self.compute_vif,
            self.compute_training_qq,
        ]
        for method in train_methods:
            try:
                method()
            except Exception as e:
                self.results[f"training_{method.__name__}_error"] = str(e)
        return None
