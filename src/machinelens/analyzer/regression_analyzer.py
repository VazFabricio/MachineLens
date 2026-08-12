"""Regression diagnostics calculation engine for MachineLens."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.nonparametric.smoothers_lowess import lowess as sm_lowess
from statsmodels.stats.multitest import multipletests

from machinelens.analyzer.shap_calculator import compute_shap_values
from machinelens.core import ModelInterface
from machinelens.core.data_classes import (
    DiagnosticResults,
    LowessData,
    OutlierAnalysisResult,
    QQData,
    RegressionMetrics,
    RegressionSubsetData,
)
from machinelens.utils import _to_dataframe

logger = logging.getLogger(__name__)


class RegressionAnalyzer:
    """Specialized engine for regression diagnostics.

    Parameters
    ----------
    interface : ModelInterface
        A validated interface wrapping the fitted model and data splits.
    """

    def __init__(self, interface: ModelInterface) -> None:
        """Initialize the RegressionAnalyzer.

        Parameters
        ----------
        interface : ModelInterface
            A validated interface wrapping the fitted model and data splits.
        """
        self._iface = interface
        self._model = interface.model

    def analyze(self, dr: DiagnosticResults) -> None:
        """Populate *dr* with all regression diagnostics.

        Parameters
        ----------
        dr : DiagnosticResults
            The result container to populate.
        """
        # -- Prepare subsets --
        for subset in ("train", "test"):
            try:
                sd = self._prepare_regression_subset(subset)
                if sd is not None:
                    setattr(dr, f"{subset}_data", sd)
            except Exception as exc:
                logger.warning("Regression subset '%s' failed: %s", subset, exc)

        # -- Metrics --
        for subset in ("train", "test"):
            subset_data: Optional[RegressionSubsetData] = getattr(
                dr, f"{subset}_data", None
            )
            if subset_data is not None:
                try:
                    setattr(
                        dr, f"{subset}_metrics", self._regression_metrics(subset_data)
                    )
                except Exception as exc:
                    logger.warning("Metrics (%s) failed: %s", subset, exc)

        # -- Q-Q --
        for subset in ("train", "test"):
            sd = getattr(dr, f"{subset}_data", None)
            if sd is not None and isinstance(sd, RegressionSubsetData):
                try:
                    setattr(dr, f"{subset}_qq", self._compute_qq(sd.std_residuals))
                except Exception as exc:
                    logger.warning("Q-Q (%s) failed: %s", subset, exc)

        # -- LOWESS curves --
        for subset in ("train", "test"):
            sd = getattr(dr, f"{subset}_data", None)
            if sd is not None and isinstance(sd, RegressionSubsetData):
                setattr(
                    dr,
                    f"{subset}_linearity_lowess",
                    self._compute_lowess(sd.y_pred, sd.residuals),
                )
                sqrt_abs = np.sqrt(np.abs(sd.std_residuals))
                setattr(
                    dr,
                    f"{subset}_scale_loc_lowess",
                    self._compute_lowess(sd.y_pred, sqrt_abs),
                )

        # -- Train-only: leverage, Cook's distance, VIF --
        train_sd = dr.train_data
        if train_sd is not None and isinstance(train_sd, RegressionSubsetData):
            try:
                lev = self._compute_leverage(train_sd.X_data)
                if lev is not None:
                    dr.leverage = lev
                    dr.leverage_lowess = self._compute_lowess(
                        lev, train_sd.std_residuals
                    )
                    p = train_sd.X_data.select_dtypes(include=[np.number]).shape[1] + 1
                    dr.cooks_distance = (train_sd.std_residuals**2 / p) * (
                        lev / ((1 - lev) ** 2 + 1e-12)
                    )
            except Exception as exc:
                logger.warning("Leverage/Cook's failed: %s", exc)

        # -- Test-only: outlier analysis --
        test_sd = dr.test_data
        if test_sd is not None and isinstance(test_sd, RegressionSubsetData):
            try:
                dr.outlier_analysis = self._analyze_outliers(test_sd)
            except Exception as exc:
                logger.warning("Outlier analysis failed: %s", exc)

        # -- SHAP --
        if train_sd is not None:
            try:
                # We need the original untransformed X_eval (train and test)
                dr.train_shap = compute_shap_values(
                    self._model,
                    train_sd.X_data,
                    train_sd.X_data,
                    is_classification=False,
                )
                if test_sd is not None:
                    dr.test_shap = compute_shap_values(
                        self._model,
                        train_sd.X_data,
                        test_sd.X_data,
                        is_classification=False,
                    )
            except Exception as exc:
                logger.warning("SHAP calculation failed: %s", exc)

    # -- Regression helpers --


    def _compute_lowess(
        self, x: np.ndarray, y: np.ndarray, frac: float = 0.6667, n_boot: int = 30
    ) -> Optional[LowessData]:
        """Compute LOWESS smoothing with bootstrap 95 % CI band.

        Parameters
        ----------
        x : np.ndarray
            The x-coordinates of the data points.
        y : np.ndarray
            The y-coordinates of the data points.
        frac : float, default=0.6667
            The fraction of data used when estimating each y-value.
        n_boot : int, default=30
            The number of bootstrap iterations to estimate the 95% confidence interval.

        Returns
        -------
        LowessData or None
            The smoothed LOWESS coordinates and confidence bounds, or None if the
            computation fails.
        """
        try:
            idx = np.argsort(x)
            xs, ys = x[idx], y[idx]
            smooth = sm_lowess(ys, xs, frac=frac, return_sorted=True)
            x_sm, y_sm = smooth[:, 0], smooth[:, 1]

            rng = np.random.default_rng(0)
            n = len(xs)
            curves = np.empty((n_boot, len(x_sm)))
            for b in range(n_boot):
                bi = rng.choice(n, size=n, replace=True)
                bx, by = x[bi], y[bi]
                bs = np.argsort(bx)
                bsm = sm_lowess(by[bs], bx[bs], frac=frac, return_sorted=True)
                curves[b] = np.interp(x_sm, bsm[:, 0], bsm[:, 1])

            return LowessData(
                x_smooth=x_sm,
                y_smooth=y_sm,
                ci_lower=np.percentile(curves, 2.5, axis=0),
                ci_upper=np.percentile(curves, 97.5, axis=0),
            )
        except Exception as exc:
            logger.warning("LOWESS smoothing computation failed: %s", exc)
            return None

    def _prepare_regression_subset(self, subset: str) -> Optional[RegressionSubsetData]:
        """Prepare regression features, targets, predictions, and residuals for a subset.

        Parameters
        ----------
        subset : str
            The data subset to prepare (``"train"`` or ``"test"``).

        Returns
        -------
        RegressionSubsetData or None
            The prepared regression subset data container, or None if features
            or targets are unavailable.
        """
        iface = self._iface
        X_raw = getattr(iface, f"X_{subset}", None)
        y_raw = getattr(iface, f"y_{subset}", None)
        X = _to_dataframe(X_raw)
        if X is None or y_raw is None:
            return None

        y_true = np.asarray(y_raw).ravel()

        if subset == "test":
            if iface.y_pred is None:
                return None
            y_pred = np.asarray(iface.y_pred).ravel()
        else:
            if not hasattr(self._model, "predict"):
                return None
            y_pred = np.asarray(self._model.predict(X_raw)).ravel()

        n = min(len(X), len(y_true), len(y_pred))
        y_true, y_pred, X = y_true[:n], y_pred[:n], X.iloc[:n].copy()

        residuals = y_true - y_pred
        rmse = np.sqrt(np.nanmean(residuals**2))
        std_res = residuals / rmse if rmse != 0 else residuals

        return RegressionSubsetData(
            X_data=X,
            y_true=y_true,
            y_pred=y_pred,
            residuals=residuals,
            std_residuals=std_res,
            abs_residuals=np.abs(residuals),
        )

    @staticmethod
    def _regression_metrics(sd: RegressionSubsetData) -> RegressionMetrics:
        """Calculate scalar regression evaluation metrics.

        Parameters
        ----------
        sd : RegressionSubsetData
            The prepared regression subset data.

        Returns
        -------
        RegressionMetrics
            Calculated evaluation metrics containing MAE, MSE, RMSE, R2, Adjusted R2,
            MAPE, and sample count.
        """
        n = len(sd.y_true)
        mae = float(np.mean(sd.abs_residuals))
        mse = float(np.mean(sd.residuals**2))
        rmse = float(np.sqrt(mse))
        ss_res = float(np.sum(sd.residuals**2))
        ss_tot = float(np.sum((sd.y_true - np.mean(sd.y_true)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot != 0 else float("nan")
        p = sd.X_data.shape[1]
        if n > p + 1:
            r2_adjusted = 1.0 - ((1.0 - r2) * (n - 1) / (n - p - 1))
        else:
            r2_adjusted = float("nan")

        nz = sd.y_true != 0
        mape = (
            float(np.mean(np.abs(sd.residuals[nz] / sd.y_true[nz])) * 100)
            if nz.any()
            else float("nan")
        )

        return RegressionMetrics(
            mae=mae,
            mse=mse,
            rmse=rmse,
            r2=r2,
            r2_adjusted=r2_adjusted,
            mape=mape,
            n_samples=n,
        )

    @staticmethod
    def _compute_qq(std_residuals: np.ndarray) -> QQData:
        """Compute Q-Q plot coordinates and reference lines.

        Parameters
        ----------
        std_residuals : np.ndarray
            Standardized residuals.

        Returns
        -------
        QQData
            Theoretical and sample quantiles, slope, intercept, r-value, and
            confidence interval.
        """
        (osm, osr), (slope, intercept, r) = stats.probplot(std_residuals, dist="norm")
        n_pts = len(osm)
        order = np.arange(1, n_pts + 1)
        se = (1.0 / stats.norm.pdf(osm)) * np.sqrt(
            order * (n_pts - order + 1) / ((n_pts + 1) ** 2 * (n_pts + 2))
        )
        return QQData(
            theoretical=osm,
            sample=osr,
            slope=slope,
            intercept=intercept,
            r_value=r,
            ci_lower=osm - 1.96 * se,
            ci_upper=osm + 1.96 * se,
        )

    @staticmethod
    def _compute_leverage(X: pd.DataFrame) -> Optional[np.ndarray]:
        """Compute leverage (hat values) for the features.

        Parameters
        ----------
        X : pd.DataFrame
            The input feature matrix.

        Returns
        -------
        np.ndarray or None
            The leverage value for each sample, or None if numerical features are
            not present or computation fails.
        """
        num = X.select_dtypes(include=[np.number]).columns
        if len(num) == 0:
            return None
        try:
            mat = X[num].dropna().values
            aug = np.column_stack([np.ones(mat.shape[0]), mat])
            Q, _ = np.linalg.qr(aug)
            return np.sum(Q**2, axis=1)
        except Exception as exc:
            logger.warning("Leverage computation failed: %s", exc)
            return None

    def _analyze_outliers(
        self,
        sd: RegressionSubsetData,
        threshold: Optional[float] = None,
        alpha: float = 0.05,
        min_group: int = 5,
    ) -> Optional[OutlierAnalysisResult]:
        """Perform feature-level statistical analysis of standardized residual outliers.

        Parameters
        ----------
        sd : RegressionSubsetData
            The prepared regression subset data.
        threshold : float, optional
            Standardized residual threshold used to classify outliers. If None,
            defaults to the 95th percentile of absolute standardized residuals.
        alpha : float, default=0.05
            Significance level for False Discovery Rate correction.
        min_group : int, default=5
            Minimum number of samples required in both outlier and typical groups
            to perform the statistical test.

        Returns
        -------
        OutlierAnalysisResult or None
            Statistical test results and trend lines for outlying features, or None
            if no valid tests were performed.
        """
        if threshold is None:
            threshold = float(np.percentile(np.abs(sd.std_residuals), 95))

        mask = np.abs(sd.std_residuals) >= threshold
        X_out, X_in = sd.X_data[mask], sd.X_data[~mask]
        num_cols = sd.X_data.select_dtypes(include=[np.number]).columns
        rows: List[Dict[str, Any]] = []

        for col in num_cols:
            ov = X_out[col].dropna().values
            iv = X_in[col].dropna().values
            if len(ov) < min_group or len(iv) < min_group:
                continue
            try:
                stat, pval = stats.mannwhitneyu(ov, iv, alternative="two-sided")
                eff = float((2.0 * stat) / (len(ov) * len(iv)) - 1.0)
                rows.append(
                    {
                        "feature": col,
                        "stat": float(stat),
                        "p_value": float(pval),
                        "effect_size": eff,
                    }
                )
            except Exception as exc:
                logger.warning("Mann-Whitney U failed on feature %s: %s", col, exc)

        if not rows:
            return None

        pvals = [r["p_value"] for r in rows]
        reject, adj, _, _ = multipletests(pvals, alpha=alpha, method="fdr_bh")
        for r, a, rej in zip(rows, adj, reject, strict=False):
            r["adj_p_value"] = float(a)
            r["significant"] = bool(rej)

        df = pd.DataFrame(rows).sort_values("adj_p_value").reset_index(drop=True)

        lowess_curves: Dict[str, LowessData] = {}
        for r in rows:
            if r["significant"]:
                feat = r["feature"]
                ld = self._compute_lowess(sd.X_data[feat].values, sd.std_residuals)
                if ld is not None:
                    lowess_curves[feat] = ld

        return OutlierAnalysisResult(
            results_df=df, threshold=threshold, lowess_curves=lowess_curves
        )
