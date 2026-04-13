from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from explainability.base_explainer import BaseExplainer
from sklearn.inspection import partial_dependence, permutation_importance
from sklearn.metrics import accuracy_score


class NativeExplainer(BaseExplainer):
    """
    Native model explainer using built-in scikit-learn capabilities.

    This explainer extracts feature importances and other explanations
    directly from the model's native attributes (e.g., `feature_importances_`
    for trees, `coef_` for linear models) or via standard scikit-learn
    inspection functions like permutation importance.
    """

    def feature_importance(
        self,
        method: str = "auto",
        n_repeats: int = 10,
        random_state: int = 0,
        **kwargs: Any,
    ) -> Optional[pd.DataFrame]:
        """
        Calculate feature importance using native attributes or permutations.

        Parameters
        ----------
        method : str, default="auto"
            The method to use for extracting importance. If "auto", it attempts
            to use `feature_importances_` or `coef_`. If neither is available,
            or if method is "permutation", it falls back to permutation importance.
        n_repeats : int, default=10
            Number of times to permute a feature. Used only for permutation
            importance.
        random_state : int, default=0
            Seed for the random number generator. Used only for permutation
            importance.
        **kwargs : Any
            Additional arguments.

        Returns
        -------
        pandas.DataFrame or None
            A DataFrame containing feature names and their corresponding
            importance scores, sorted in descending order. Returns None if
            essential data or the model is missing.
        """
        X_test = getattr(self.model_interface, "X_test", None)
        model = getattr(self, "model", getattr(self.model_interface, "model", None))

        if X_test is None or model is None:
            return None

        if isinstance(X_test, pd.DataFrame):
            feature_names = X_test.columns.tolist()
        elif isinstance(X_test, pd.Series):
            feature_names = [str(X_test.name)]
        elif hasattr(model, "feature_names_in_"):
            feature_names = model.feature_names_in_.tolist()
        else:
            n_features = np.shape(X_test)[1] if len(np.shape(X_test)) > 1 else 1
            feature_names = [f"Feature_{i}" for i in range(n_features)]

        if method == "auto":
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
                return (
                    pd.DataFrame({"feature": feature_names, "importance": importances})
                    .sort_values("importance", ascending=False)
                    .reset_index(drop=True)
                )

            elif hasattr(model, "coef_"):
                coefs = np.asarray(model.coef_)
                if coefs.ndim > 1:
                    coefs = coefs[0]
                return (
                    pd.DataFrame(
                        {
                            "feature": feature_names,
                            "importance": np.abs(coefs),
                            "coefficient": coefs,
                        }
                    )
                    .sort_values("importance", ascending=False)
                    .reset_index(drop=True)
                )

            else:
                method = "permutation"

        if method == "permutation":
            y_test = getattr(self.model_interface, "y_test", None)
            if y_test is None or not hasattr(model, "predict"):
                return None

            r = permutation_importance(
                model,
                X_test,
                y_test,
                n_repeats=n_repeats,
                random_state=random_state,
            )
            return (
                pd.DataFrame(
                    {
                        "feature": feature_names,
                        "importance_mean": r.importances_mean,
                        "importance_std": r.importances_std,
                    }
                )
                .sort_values("importance_mean", ascending=False)
                .reset_index(drop=True)
            )

        return None

    def partial_dependence(
        self,
        features: Optional[List[Union[int, str]]] = None,
        grid_resolution: int = 100,
        **kwargs: Any,
    ) -> Optional[Dict[str, Any]]:
        """
        Calculate partial dependence for specified features.

        Parameters
        ----------
        features : list of int or str, optional
            The feature indices or names to calculate partial dependence for.
        grid_resolution : int, default=100
            The number of equally spaced points on the grid.
        **kwargs : Any
            Additional arguments passed to sklearn.inspection.partial_dependence.

        Returns
        -------
        dict or None
            A dictionary containing the grid values and average predictions.
        """
        if features is None:
            return None

        X_test = getattr(self.model_interface, "X_test", None)
        model = getattr(self, "model", getattr(self.model_interface, "model", None))

        if X_test is None or model is None:
            return None

        results = {}
        for feature in features:
            try:
                pd_result = partial_dependence(
                    model,
                    X_test,
                    features=[feature],
                    grid_resolution=grid_resolution,
                    **kwargs,
                )
                results[str(feature)] = {
                    "grid": pd_result["values"][0].tolist(),
                    "average": pd_result["average"][0].tolist(),
                }
            except Exception:
                continue

        return results if results else None

    def sensitivity_analysis(
        self, noise_level: float = 0.1, n_iterations: int = 5, **kwargs: Any
    ) -> Optional[pd.DataFrame]:
        """
        Perform sensitivity analysis by adding noise to features.

        Parameters
        ----------
        noise_level : float, default=0.1
            The standard deviation multiplier for the Gaussian noise.
        n_iterations : int, default=5
            Number of times to perturb the data and measure the effect.
        **kwargs : Any
            Additional arguments.

        Returns
        -------
        pandas.DataFrame or None
            A DataFrame with the sensitivity scores for each feature.
        """
        X_test = getattr(self.model_interface, "X_test", None)
        model = getattr(self, "model", getattr(self.model_interface, "model", None))

        if X_test is None or model is None or not hasattr(model, "predict"):
            return None

        if isinstance(X_test, (pd.DataFrame, pd.Series)):
            X_df = pd.DataFrame(X_test).copy()
            feature_names = X_df.columns.tolist()
        else:
            X_df = pd.DataFrame(X_test)
            feature_names = [f"Feature_{i}" for i in range(X_df.shape[1])]

        baseline_preds = np.asarray(
            model.predict(X_df.values if not isinstance(X_test, pd.DataFrame) else X_df)
        )

        sensitivity_scores = []

        for col in X_df.columns:
            std_dev = np.std(X_df[col])
            if std_dev == 0:
                sensitivity_scores.append(0.0)
                continue

            diffs = []
            for _ in range(n_iterations):
                X_perturbed = X_df.copy()
                noise = np.random.normal(0, noise_level * std_dev, size=len(X_df))
                X_perturbed[col] = X_perturbed[col] + noise

                new_preds = np.asarray(
                    model.predict(
                        X_perturbed.values
                        if not isinstance(X_test, pd.DataFrame)
                        else X_perturbed
                    )
                )

                if baseline_preds.dtype.kind in "fc":
                    diffs.append(np.mean(np.abs(baseline_preds - new_preds)))
                else:
                    diffs.append(np.mean(baseline_preds != new_preds))

            sensitivity_scores.append(float(np.mean(diffs)))

        return (
            pd.DataFrame({"feature": feature_names, "sensitivity": sensitivity_scores})
            .sort_values("sensitivity", ascending=False)
            .reset_index(drop=True)
        )

    def fairness_bias(
        self,
        protected_attribute: Optional[str] = None,
        target_metric: str = "accuracy",
        **kwargs: Any,
    ) -> Optional[pd.DataFrame]:
        """
        Evaluate model fairness and bias across groups.

        Parameters
        ----------
        protected_attribute : str, optional
            The name of the feature defining the groups for fairness evaluation.
        target_metric : str, default="accuracy"
            The metric to compute for each group.
        **kwargs : Any
            Additional arguments.

        Returns
        -------
        pandas.DataFrame or None
            A DataFrame containing the metric evaluated for each group.
        """
        if protected_attribute is None:
            return None

        X_test = getattr(self.model_interface, "X_test", None)
        y_test = getattr(self.model_interface, "y_test", None)
        model = getattr(self, "model", getattr(self.model_interface, "model", None))

        if (
            X_test is None
            or y_test is None
            or model is None
            or not hasattr(model, "predict")
        ):
            return None

        if (
            not isinstance(X_test, pd.DataFrame)
            or protected_attribute not in X_test.columns
        ):
            return None

        y_test_arr = np.asarray(y_test).ravel()
        y_pred = np.asarray(model.predict(X_test)).ravel()
        groups = np.asarray(X_test[protected_attribute].values)

        unique_groups = np.unique(groups)
        results_meta = []

        for group in unique_groups:
            mask = groups == group
            if not np.any(mask):
                continue

            group_y_true = y_test_arr[mask]
            group_y_pred = y_pred[mask]

            metric_val = np.nan
            if target_metric == "accuracy" and group_y_true.dtype.kind not in "fc":
                metric_val = accuracy_score(group_y_true, group_y_pred)
            elif target_metric == "mean_prediction":
                metric_val = np.mean(group_y_pred)

            results_meta.append(
                {
                    "group": group,
                    "n_samples": int(np.sum(mask)),
                    target_metric: float(metric_val),
                }
            )

        return pd.DataFrame(results_meta)
