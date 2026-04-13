from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_curve,
)
from sklearn.metrics import (
    confusion_matrix as sk_confusion_matrix,
)
from sklearn.preprocessing import label_binarize
from statsmodels.stats.multitest import multipletests


class ClassificationDiagnostics:
    """
    Diagnostics suite for classification models.

    This class provides a comprehensive set of analytical and visual tools
    to evaluate the performance of classification models, including metrics,
    confusion matrices, and misclassification statistical analysis.

    Attributes
    ----------
    model_interface : Any
        The interface containing the model and data splits.
    model : Any
        The predictive model extracted from the interface.
    results : dict
        Dictionary storing numerical and tabular analysis results.
    """

    def __init__(self, model_interface: Any) -> None:
        """
        Initialize the ClassificationDiagnostics object.

        Parameters
        ----------
        model_interface : Any
            An object containing the classification model and dataset splits.
        """
        self.model_interface = model_interface
        self.model = getattr(model_interface, "model", None)
        self.results: Dict[str, Any] = {}

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

    def _align_sizes(
        self,
        X: Optional[pd.DataFrame],
        y: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
    ) -> Any:
        if X is None:
            return X, y, y_pred, y_prob

        lens = [len(X), len(y), len(y_pred)]
        if y_prob is not None:
            lens.append(len(y_prob))

        n = min(lens)
        y_prob_out = y_prob[:n] if y_prob is not None else None
        return X.iloc[:n], y[:n], y_pred[:n], y_prob_out

    def _ensure_classification_base(
        self, subset: str = "test"
    ) -> Optional[Dict[str, Any]]:
        """Return the classification data dict for the subset, initializing if needed.

        Populates ``X``, ``y``, and ``y_pred`` for the given subset.

        Parameters
        ----------
        subset : str, default="test"
            Either "test" or "train".

        Returns
        -------
        dict or None
            The data dictionary for the subset, or None if data is missing.
        """
        key = f"classification_data_{subset}"
        if key in self.results:
            return self.results[key]

        X_raw = getattr(self.model_interface, f"X_{subset}", None)
        y_raw = getattr(self.model_interface, f"y_{subset}", None)
        X = self._as_df(X_raw)

        if X is None or y_raw is None or self.model is None:
            return None

        y = np.asarray(y_raw).ravel()
        if not (hasattr(self.model, "predict") and callable(self.model.predict)):
            return None

        # Predict if needed
        if subset == "test":
            y_pred = getattr(self.model_interface, "y_pred", None)
            if y_pred is None:
                y_pred = np.asarray(self.model.predict(X)).ravel()
        else:
            y_pred = np.asarray(self.model.predict(X)).ravel()

        y_pred = np.asarray(y_pred).ravel()

        # Grab probabilities if supported
        y_prob = None
        if hasattr(self.model, "predict_proba") and callable(self.model.predict_proba):
            try:
                y_prob = np.asarray(self.model.predict_proba(X))
            except Exception:
                pass

        X, y, y_pred, y_prob = self._align_sizes(X, y, y_pred, y_prob)

        data = {"X": X, "y": y, "y_pred": y_pred, "y_prob": y_prob}
        self.results[key] = data
        return data

    def classification_metrics_test(self) -> Dict[str, float]:
        """Calculate global metrics for the test set."""
        return self._compute_metrics(subset="test")

    def classification_metrics_train(self) -> Dict[str, float]:
        """Calculate global metrics for the training set."""
        return self._compute_metrics(subset="train")

    def _compute_metrics(self, subset: str) -> Dict[str, float]:
        data = self._ensure_classification_base(subset)
        if not data:
            return {}

        y, y_pred = data["y"], data["y_pred"]
        metrics = {}
        metrics["accuracy"] = float(accuracy_score(y, y_pred))

        unique_y = np.unique(y)
        average_method = "binary" if len(unique_y) <= 2 else "macro"
        p, r, f, _ = precision_recall_fscore_support(
            y, y_pred, average=average_method, zero_division=0
        )
        metrics["precision"] = float(p)
        metrics["recall"] = float(r)
        metrics["f1_score"] = float(f)

        self.results[f"classification_metrics_{subset}"] = metrics
        return metrics

    def confusion_matrix_test(
        self, normalize: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """Compute confusion matrix for the test set."""
        return self._compute_cm(subset="test", normalize=normalize)

    def confusion_matrix_train(
        self, normalize: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """Compute confusion matrix for the training set."""
        return self._compute_cm(subset="train", normalize=normalize)

    def _compute_cm(
        self, subset: str, normalize: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        data = self._ensure_classification_base(subset)
        if not data:
            return None

        y, y_pred = data["y"], data["y_pred"]
        cm = sk_confusion_matrix(y, y_pred, normalize=normalize)
        labels = np.unique(np.concatenate((y, y_pred)))
        df_cm = pd.DataFrame(cm, index=labels, columns=labels)
        self.results[f"confusion_matrix_{subset}"] = df_cm
        return df_cm

    # ------------------------------------------------------------------
    # ROC and PR Curves
    # ------------------------------------------------------------------

    def compute_roc_curve_test(self) -> Optional[Dict[str, Any]]:
        """Compute the ROC curve for the test dataset."""
        return self._compute_roc_curve("test")

    def compute_roc_curve_train(self) -> Optional[Dict[str, Any]]:
        """Compute the ROC curve for the training dataset."""
        return self._compute_roc_curve("train")

    def _compute_roc_curve(self, subset: str) -> Optional[Dict[str, Any]]:
        """Calculate ROC metrics for the specified data subset."""
        data = self._ensure_classification_base(subset)
        if not data or data.get("y_prob") is None:
            return None

        y, y_prob = data["y"], data["y_prob"]
        classes = np.unique(y)

        res = {}
        if len(classes) == 2:
            # Binary classification
            prob_pos = y_prob[:, 1] if y_prob.shape[1] > 1 else y_prob.ravel()
            fpr, tpr, _ = roc_curve(y, prob_pos, pos_label=classes[1])
            res["binary"] = {"fpr": fpr, "tpr": tpr, "auc": float(auc(fpr, tpr))}
        elif len(classes) > 2:
            # Multi-class (One-vs-Rest)
            y_bin = label_binarize(y, classes=classes)
            for i, cls in enumerate(classes):
                fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
                res[f"class_{cls}"] = {
                    "fpr": fpr,
                    "tpr": tpr,
                    "auc": float(auc(fpr, tpr)),
                }

        self.results[f"roc_curve_{subset}"] = res
        return res

    def compute_pr_curve_test(self) -> Optional[Dict[str, Any]]:
        """Compute the Precision-Recall curve for the test dataset."""
        return self._compute_pr_curve("test")

    def compute_pr_curve_train(self) -> Optional[Dict[str, Any]]:
        """Compute the Precision-Recall curve for the training dataset."""
        return self._compute_pr_curve("train")

    def _compute_pr_curve(self, subset: str) -> Optional[Dict[str, Any]]:
        """Calculate PR metrics for the specified data subset."""
        data = self._ensure_classification_base(subset)
        if not data or data.get("y_prob") is None:
            return None

        y, y_prob = data["y"], data["y_prob"]
        classes = np.unique(y)

        res = {}
        if len(classes) == 2:
            # Binary classification
            prob_pos = y_prob[:, 1] if y_prob.shape[1] > 1 else y_prob.ravel()
            prec, rec, _ = precision_recall_curve(y, prob_pos, pos_label=classes[1])
            ap = average_precision_score(y, prob_pos, pos_label=classes[1])
            res["binary"] = {
                "precision": prec,
                "recall": rec,
                "ap": float(ap),
                "baseline": float(np.mean(y == classes[1])),
            }
        elif len(classes) > 2:
            # Multi-class
            y_bin = label_binarize(y, classes=classes)
            for i, cls in enumerate(classes):
                prec, rec, _ = precision_recall_curve(y_bin[:, i], y_prob[:, i])
                ap = average_precision_score(y_bin[:, i], y_prob[:, i])
                res[f"class_{cls}"] = {
                    "precision": prec,
                    "recall": rec,
                    "ap": float(ap),
                    "baseline": float(np.mean(y_bin[:, i])),
                }

        self.results[f"pr_curve_{subset}"] = res
        return res

    def misclassification_analysis_test(
        self,
        alpha: float = 0.05,
        min_group_size: int = 5,
        normality_check: bool = True,
        p_adjust_method: str = "fdr_bh",
    ) -> Optional[pd.DataFrame]:
        """Analyze misclassifications in the test set."""
        return self._compute_misclassification(
            subset="test",
            alpha=alpha,
            min_group_size=min_group_size,
            normality_check=normality_check,
            p_adjust_method=p_adjust_method,
        )

    def misclassification_analysis_train(
        self,
        alpha: float = 0.05,
        min_group_size: int = 5,
        normality_check: bool = True,
        p_adjust_method: str = "fdr_bh",
    ) -> Optional[pd.DataFrame]:
        """Analyze misclassifications in the training set."""
        return self._compute_misclassification(
            subset="train",
            alpha=alpha,
            min_group_size=min_group_size,
            normality_check=normality_check,
            p_adjust_method=p_adjust_method,
        )

    def _compute_misclassification(
        self,
        subset: str,
        alpha: float = 0.05,
        min_group_size: int = 5,
        normality_check: bool = True,
        p_adjust_method: str = "fdr_bh",
    ) -> Optional[pd.DataFrame]:
        data = self._ensure_classification_base(subset)
        if not data:
            return None

        X, y, y_pred = data["X"], data["y"], data["y_pred"]
        mask_incorrect = y != y_pred
        X_incorrect = X.loc[mask_incorrect]
        X_correct = X.loc[~mask_incorrect]

        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        pvals = []
        features_meta = []

        for col in numeric_cols:
            inc_vals = X_incorrect[col].dropna().values
            cor_vals = X_correct[col].dropna().values
            n_inc, n_cor = len(inc_vals), len(cor_vals)

            if n_inc < min_group_size or n_cor < min_group_size:
                continue

            mean_inc, mean_cor = float(np.nanmean(inc_vals)), float(
                np.nanmean(cor_vals)
            )
            use_mann = False
            if normality_check and 3 <= n_inc <= 5000 and 3 <= n_cor <= 5000:
                sh_inc, sh_cor = stats.shapiro(inc_vals), stats.shapiro(cor_vals)
                if sh_inc.pvalue < 0.05 or sh_cor.pvalue < 0.05:
                    use_mann = True
            else:
                use_mann = True

            stat, pval, effect = np.nan, 1.0, np.nan
            if use_mann:
                try:
                    u_stat, pval = stats.mannwhitneyu(
                        inc_vals, cor_vals, alternative="two-sided"
                    )
                    stat, effect = float(u_stat), float(
                        (2.0 * u_stat) / (n_inc * n_cor) - 1.0
                    )
                except Exception:
                    pass
            else:
                try:
                    t_res = stats.ttest_ind(
                        inc_vals, cor_vals, equal_var=False, nan_policy="omit"
                    )
                    stat, pval = float(t_res.statistic), float(t_res.pvalue)
                    s1, s2 = np.nanstd(inc_vals, ddof=1), np.nanstd(cor_vals, ddof=1)
                    pooled_sd = np.sqrt(
                        ((n_inc - 1) * s1**2 + (n_cor - 1) * s2**2)
                        / max(1, (n_inc + n_cor - 2))
                    )
                    effect = (
                        float((mean_inc - mean_cor) / pooled_sd)
                        if pooled_sd > 0
                        else np.nan
                    )
                except Exception:
                    pass

            pvals.append(pval)
            features_meta.append(
                {
                    "feature": col,
                    "stat": stat,
                    "p_value": pval,
                    "n_incorrect": int(n_inc),
                    "n_correct": int(n_cor),
                    "effect_size": effect,
                }
            )

        if pvals:
            reject, pvals_adj, _, _ = multipletests(
                pvals, alpha=alpha, method=p_adjust_method
            )
        else:
            pvals_adj, reject = [], []

        for row, adj_p, rej in zip(features_meta, pvals_adj, reject, strict=False):
            row["adj_p_value"], row["significant"] = float(adj_p), bool(rej)

        res_df = pd.DataFrame(features_meta)
        if not res_df.empty:
            res_df = res_df.sort_values("adj_p_value").reset_index(drop=True)

        self.results[f"misclassification_analysis_{subset}"] = res_df
        return res_df

    def run_all(self) -> None:
        """Execute the full symmetric suite of classification diagnostics."""
        for subset in ["train", "test"]:
            try:
                self._compute_metrics(subset)
            except Exception as e:
                self.results[f"metrics_{subset}_error"] = str(e)
            try:
                self._compute_cm(subset)
            except Exception as e:
                self.results[f"cm_{subset}_error"] = str(e)
            try:
                self._compute_misclassification(subset)
            except Exception as e:
                self.results[f"misclass_{subset}_error"] = str(e)
            try:
                self._compute_roc_curve(subset)
            except Exception as e:
                self.results[f"roc_{subset}_error"] = str(e)
            try:
                self._compute_pr_curve(subset)
            except Exception as e:
                self.results[f"pr_{subset}_error"] = str(e)
