"""Classification diagnostics calculation engine for MachineLens."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    brier_score_loss,
    log_loss,
    matthews_corrcoef,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from sklearn.preprocessing import label_binarize

from machinelens.analyzer.shap_calculator import compute_shap_values
from machinelens.core import ModelInterface
from machinelens.core.data_classes import (
    CalibrationCurveData,
    ClassificationMetrics,
    ClassificationSubsetData,
    DiagnosticResults,
    MisclassificationResult,
    PrCurveData,
    RocCurveData,
    ThresholdAnalysisData,
)
from machinelens.utils import _to_dataframe

logger = logging.getLogger(__name__)


class ClassificationAnalyzer:
    """Specialized engine for classification diagnostics.

    Parameters
    ----------
    interface : ModelInterface
        A validated interface wrapping the fitted model and data splits.
    """

    def __init__(self, interface: ModelInterface) -> None:
        """Initialize the ClassificationAnalyzer.

        Parameters
        ----------
        interface : ModelInterface
            A validated interface wrapping the fitted model and data splits.
        """
        self._iface = interface
        self._model = interface.model

    def analyze(self, dr: DiagnosticResults) -> None:
        """Populate *dr* with all classification diagnostics.

        Parameters
        ----------
        dr : DiagnosticResults
            The result container to populate.
        """
        for subset in ("train", "test"):
            try:
                sd = self._prepare_classification_subset(subset)
                if sd is not None:
                    setattr(dr, f"{subset}_data", sd)
            except Exception as exc:
                logger.warning("Classification subset '%s' failed: %s", subset, exc)

        for subset in ("train", "test"):
            sd = getattr(dr, f"{subset}_data", None)
            if sd is None or not isinstance(sd, ClassificationSubsetData):
                continue

            # Metrics
            try:
                setattr(
                    dr,
                    f"{subset}_clf_metrics",
                    self._classification_metrics(sd.y_true, sd.y_pred, sd.y_prob),
                )
            except Exception as exc:
                logger.warning("Clf metrics (%s) failed: %s", subset, exc)

            # Confusion matrix
            try:
                cm = sk_confusion_matrix(sd.y_true, sd.y_pred)
                labels = np.unique(np.concatenate((sd.y_true, sd.y_pred)))
                setattr(
                    dr,
                    f"{subset}_confusion_matrix",
                    pd.DataFrame(cm, index=labels, columns=labels),
                )
            except Exception as exc:
                logger.warning("Confusion matrix (%s) failed: %s", subset, exc)

            # ROC / PR (need probabilities)
            if sd.y_prob is not None:
                try:
                    setattr(
                        dr,
                        f"{subset}_roc_curves",
                        self._compute_roc(sd.y_true, sd.y_prob),
                    )
                except Exception as exc:
                    logger.warning("ROC (%s) failed: %s", subset, exc)

                try:
                    setattr(
                        dr,
                        f"{subset}_pr_curves",
                        self._compute_pr(sd.y_true, sd.y_prob),
                    )
                except Exception as exc:
                    logger.warning("PR (%s) failed: %s", subset, exc)

                try:
                    setattr(
                        dr,
                        f"{subset}_calibration_curves",
                        self._compute_calibration(sd.y_true, sd.y_prob),
                    )
                except Exception as exc:
                    logger.warning("Calibration (%s) failed: %s", subset, exc)

                try:
                    setattr(
                        dr,
                        f"{subset}_threshold_analysis",
                        self._compute_threshold_metrics(sd.y_true, sd.y_prob),
                    )
                except Exception as exc:
                    logger.warning("Threshold analysis (%s) failed: %s", subset, exc)

            # Misclassification analysis
            try:
                mc = self._misclassification_analysis(sd.X_data, sd.y_true, sd.y_pred)
                if mc is not None:
                    setattr(dr, f"{subset}_misclassification", mc)
            except Exception as exc:
                logger.warning("Misclassification (%s) failed: %s", subset, exc)

        # -- SHAP --
        train_sd = dr.train_data
        test_sd = dr.test_data
        if train_sd is not None:
            try:
                dr.train_shap = compute_shap_values(
                    self._model,
                    train_sd.X_data,
                    train_sd.X_data,
                    is_classification=True,
                )
                if test_sd is not None:
                    dr.test_shap = compute_shap_values(
                        self._model,
                        train_sd.X_data,
                        test_sd.X_data,
                        is_classification=True,
                    )
            except Exception as exc:
                logger.warning("SHAP calculation failed: %s", exc)

    # -- Classification helpers --

    def _prepare_classification_subset(
        self, subset: str
    ) -> Optional[ClassificationSubsetData]:
        """Prepare classification features, targets, and predictions for a subset.

        Parameters
        ----------
        subset : str
            The data subset to prepare (``"train"`` or ``"test"``).

        Returns
        -------
        ClassificationSubsetData or None
            The prepared classification subset data container, or None if features
            or targets are unavailable.
        """
        iface = self._iface
        X_raw = getattr(iface, f"X_{subset}", None)
        y_raw = getattr(iface, f"y_{subset}", None)
        X = _to_dataframe(X_raw)
        if X is None or y_raw is None:
            return None

        y_true = np.asarray(y_raw).ravel()

        if subset == "test" and iface.y_pred is not None:
            y_pred = np.asarray(iface.y_pred).ravel()
        elif hasattr(self._model, "predict"):
            y_pred = np.asarray(self._model.predict(X_raw)).ravel()
        else:
            return None

        # Probabilities
        y_prob = None
        if hasattr(self._model, "predict_proba") and callable(
            self._model.predict_proba
        ):
            try:
                y_prob = np.asarray(self._model.predict_proba(X_raw))
            except Exception as exc:
                logger.warning("predict_proba failed: %s", exc)

        n = min(len(X), len(y_true), len(y_pred))
        if y_prob is not None:
            n = min(n, len(y_prob))
            y_prob = y_prob[:n]

        return ClassificationSubsetData(
            X_data=X.iloc[:n].copy(),
            y_true=y_true[:n],
            y_pred=y_pred[:n],
            y_prob=y_prob,
        )

    @staticmethod
    def _classification_metrics(
        y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None
    ) -> ClassificationMetrics:
        """Calculate scalar classification evaluation metrics.

        Parameters
        ----------
        y_true : np.ndarray
            Ground truth target values.
        y_pred : np.ndarray
            Predicted target values.
        y_prob : np.ndarray, optional
            Predicted class probabilities.

        Returns
        -------
        ClassificationMetrics
            Calculated evaluation metrics containing accuracy, precision, recall,
            F1, MCC, and probabilistic metrics if y_prob is available.
        """
        unique = np.unique(y_true)
        avg = "binary" if len(unique) <= 2 else "macro"
        p, r, f, _ = precision_recall_fscore_support(
            y_true, y_pred, average=avg, zero_division=0
        )

        mcc = float(matthews_corrcoef(y_true, y_pred))

        roc_auc = None
        pr_auc = None
        brier = None
        lloss = None

        if y_prob is not None:
            try:
                if len(unique) <= 2:
                    y_prob_pos = y_prob[:, 1] if y_prob.ndim > 1 else y_prob
                    roc_auc = float(roc_auc_score(y_true, y_prob_pos))
                    pr_auc = float(average_precision_score(y_true, y_prob_pos))
                    # Brier score is inherently binary or multi-class but scikit-learn's
                    # brier_score_loss is binary only natively unless using BrierScoreLoss
                    # which is different. We'll support binary brier score.
                    brier = float(brier_score_loss(y_true, y_prob_pos))
                else:
                    # Multi-class
                    roc_auc = float(
                        roc_auc_score(
                            y_true, y_prob, multi_class="ovr", average="macro"
                        )
                    )
                    # PR AUC macro for multiclass
                    y_true_bin = label_binarize(y_true, classes=unique)
                    pr_auc = float(
                        average_precision_score(y_true_bin, y_prob, average="macro")
                    )
            except Exception as e:
                logger.warning(f"Probabilistic metric (AUC/PR) failed: {e}")

            try:
                lloss = float(log_loss(y_true, y_prob))
            except Exception as e:
                logger.warning(f"Log loss calculation failed: {e}")

        return ClassificationMetrics(
            accuracy=float(accuracy_score(y_true, y_pred)),
            precision=float(p),
            recall=float(r),
            f1_score=float(f),
            mcc=mcc,
            roc_auc=roc_auc,
            pr_auc=pr_auc,
            brier_score=brier,
            log_loss=lloss,
        )

    @staticmethod
    def _compute_roc(y_true: np.ndarray, y_prob: np.ndarray) -> List[RocCurveData]:
        """Compute ROC curves and AUC scores.

        Parameters
        ----------
        y_true : np.ndarray
            Ground truth target values.
        y_prob : np.ndarray
            Predicted class probabilities.

        Returns
        -------
        list of RocCurveData
            List of calculated ROC curve data objects for each class or the
            binary case.
        """
        classes = np.unique(y_true)
        result: List[RocCurveData] = []
        if len(classes) == 2:
            pp = y_prob[:, 1] if y_prob.shape[1] > 1 else y_prob.ravel()
            fpr, tpr, _ = roc_curve(y_true, pp, pos_label=classes[1])
            result.append(
                RocCurveData(
                    fpr=fpr, tpr=tpr, auc_score=float(auc(fpr, tpr)), label="binary"
                )
            )
        elif len(classes) > 2:
            y_bin = label_binarize(y_true, classes=classes)
            for i, cls in enumerate(classes):
                fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
                result.append(
                    RocCurveData(
                        fpr=fpr,
                        tpr=tpr,
                        auc_score=float(auc(fpr, tpr)),
                        label=f"class_{cls}",
                    )
                )
        return result

    @staticmethod
    def _compute_pr(y_true: np.ndarray, y_prob: np.ndarray) -> List[PrCurveData]:
        """Compute Precision-Recall curves and average precision scores.

        Parameters
        ----------
        y_true : np.ndarray
            Ground truth target values.
        y_prob : np.ndarray
            Predicted class probabilities.

        Returns
        -------
        list of PrCurveData
            List of calculated PR curve data objects for each class or the
            binary case.
        """
        classes = np.unique(y_true)
        result: List[PrCurveData] = []
        if len(classes) == 2:
            pp = y_prob[:, 1] if y_prob.shape[1] > 1 else y_prob.ravel()
            prec, rec, _ = precision_recall_curve(y_true, pp, pos_label=classes[1])
            ap = average_precision_score(y_true, pp, pos_label=classes[1])
            result.append(
                PrCurveData(
                    precision_arr=prec,
                    recall_arr=rec,
                    average_precision=float(ap),
                    baseline=float(np.mean(y_true == classes[1])),
                    label="binary",
                )
            )
        elif len(classes) > 2:
            y_bin = label_binarize(y_true, classes=classes)
            for i, cls in enumerate(classes):
                prec, rec, _ = precision_recall_curve(y_bin[:, i], y_prob[:, i])
                ap = average_precision_score(y_bin[:, i], y_prob[:, i])
                result.append(
                    PrCurveData(
                        precision_arr=prec,
                        recall_arr=rec,
                        average_precision=float(ap),
                        baseline=float(np.mean(y_bin[:, i])),
                        label=f"class_{cls}",
                    )
                )
        return result

    @staticmethod
    def _misclassification_analysis(
        X: pd.DataFrame,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        alpha: float = 0.05,
        min_group: int = 5,
    ) -> Optional[MisclassificationResult]:
        """Perform feature-level statistical analysis on misclassified samples.

        Parameters
        ----------
        X : pd.DataFrame
            The input feature matrix.
        y_true : np.ndarray
            Ground truth target values.
        y_pred : np.ndarray
            Predicted target values.
        alpha : float, default=0.05
            Significance level for False Discovery Rate correction.
        min_group : int, default=5
            Minimum number of samples required in both correct and incorrect
            groups to perform the statistical test.

        Returns
        -------
        MisclassificationResult or None
            Statistical test results for each feature, or None if no valid
            tests were performed.
        """
        mask_wrong = y_true != y_pred
        Xr = X.reset_index(drop=True)
        X_inc, X_cor = Xr.loc[mask_wrong], Xr.loc[~mask_wrong]
        num_cols = X.select_dtypes(include=[np.number]).columns
        rows: List[Dict[str, Any]] = []

        for col in num_cols:
            iv = X_inc[col].dropna().values
            cv = X_cor[col].dropna().values
            if len(iv) < min_group or len(cv) < min_group:
                continue
            try:
                stat, pval = stats.mannwhitneyu(iv, cv, alternative="two-sided")
                eff = float((2.0 * stat) / (len(iv) * len(cv)) - 1.0)
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

        from statsmodels.stats.multitest import multipletests

        pvals = [r["p_value"] for r in rows]
        reject, adj, _, _ = multipletests(pvals, alpha=alpha, method="fdr_bh")
        for r, a, rej in zip(rows, adj, reject, strict=False):
            r["adj_p_value"] = float(a)
            r["significant"] = bool(rej)

        df = pd.DataFrame(rows).sort_values("adj_p_value").reset_index(drop=True)
        return MisclassificationResult(results_df=df)

    @staticmethod
    def _compute_calibration(
        y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
    ) -> List[CalibrationCurveData]:
        """Compute calibration curve data for reliability diagrams.

        Parameters
        ----------
        y_true : np.ndarray
            Ground truth target values.
        y_prob : np.ndarray
            Predicted class probabilities.
        n_bins : int, default=10
            Number of bins to compute the calibration curve.

        Returns
        -------
        list of CalibrationCurveData
            List of calculated calibration curve data objects.
        """
        classes = np.unique(y_true)
        result: List[CalibrationCurveData] = []
        if len(classes) == 2:
            pp = y_prob[:, 1] if y_prob.shape[1] > 1 else y_prob.ravel()
            prob_true, prob_pred = calibration_curve(
                y_true, pp, n_bins=n_bins, pos_label=classes[1]
            )
            result.append(
                CalibrationCurveData(
                    prob_true=prob_true, prob_pred=prob_pred, label="binary"
                )
            )
        elif len(classes) > 2:
            y_bin = label_binarize(y_true, classes=classes)
            for i, cls in enumerate(classes):
                prob_true, prob_pred = calibration_curve(
                    y_bin[:, i], y_prob[:, i], n_bins=n_bins
                )
                result.append(
                    CalibrationCurveData(
                        prob_true=prob_true, prob_pred=prob_pred, label=f"class_{cls}"
                    )
                )
        return result

    @staticmethod
    def _compute_threshold_metrics(
        y_true: np.ndarray, y_prob: np.ndarray
    ) -> List[ThresholdAnalysisData]:
        """Compute threshold decision analysis metrics.

        Parameters
        ----------
        y_true : np.ndarray
            Ground truth target values.
        y_prob : np.ndarray
            Predicted class probabilities.

        Returns
        -------
        list of ThresholdAnalysisData
            List of calculated threshold analysis data objects.
        """
        classes = np.unique(y_true)
        result: List[ThresholdAnalysisData] = []
        if len(classes) == 2:
            pp = y_prob[:, 1] if y_prob.shape[1] > 1 else y_prob.ravel()
            prec, rec, thresh = precision_recall_curve(y_true, pp, pos_label=classes[1])
            f1 = 2 * (prec[:-1] * rec[:-1]) / (prec[:-1] + rec[:-1] + 1e-9)
            result.append(
                ThresholdAnalysisData(
                    thresholds=thresh,
                    precision=prec[:-1],
                    recall=rec[:-1],
                    f1_score=f1,
                    label="binary",
                )
            )
        elif len(classes) > 2:
            y_bin = label_binarize(y_true, classes=classes)
            for i, cls in enumerate(classes):
                prec, rec, thresh = precision_recall_curve(y_bin[:, i], y_prob[:, i])
                f1 = 2 * (prec[:-1] * rec[:-1]) / (prec[:-1] + rec[:-1] + 1e-9)
                result.append(
                    ThresholdAnalysisData(
                        thresholds=thresh,
                        precision=prec[:-1],
                        recall=rec[:-1],
                        f1_score=f1,
                        label=f"class_{cls}",
                    )
                )
        return result
