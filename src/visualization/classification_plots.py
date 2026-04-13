from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats as scipy_stats

# ---------------------------------------------------------------------------
# Design tokens – clean white theme
# ---------------------------------------------------------------------------
_BG = "#ffffff"
_SURFACE = "#fafbfc"
_GRID = "rgba(0,0,0,0.06)"
_FONT_COLOR = "#1e293b"
_FONT_MUTED = "#64748b"
_FONT_FAMILY = "Inter, system-ui, sans-serif"
_ACCENT_BLUE = "#3b82f6"
_ACCENT_RED = "#ef4444"
_ACCENT_GREEN = "#10b981"
_COLORSCALE = "Blues"
_BORDER = "rgba(0,0,0,0.08)"

_LAYOUT_BASE = dict(
    paper_bgcolor=_BG,
    plot_bgcolor=_SURFACE,
    font=dict(family=_FONT_FAMILY, color=_FONT_COLOR, size=13),
    margin=dict(l=60, r=40, t=55, b=55),
    hoverlabel=dict(
        bgcolor="white",
        bordercolor=_BORDER,
        font=dict(family=_FONT_FAMILY, color=_FONT_COLOR, size=12),
    ),
)

_AXIS_BASE = dict(
    gridcolor=_GRID,
    gridwidth=1,
    zerolinecolor="rgba(0,0,0,0.12)",
    zerolinewidth=1,
    linecolor="rgba(0,0,0,0.10)",
    tickfont=dict(size=11, color=_FONT_MUTED),
    title_font=dict(size=13, color=_FONT_COLOR),
)

_COLORBAR_STYLE = dict(
    thickness=14,
    tickfont=dict(color=_FONT_MUTED, size=11),
    outlinecolor=_BORDER,
    outlinewidth=1,
    bgcolor="rgba(0,0,0,0)",
)


def _title_dict(text: str) -> dict:
    """Build a standard centred title."""
    return dict(
        text=text,
        font=dict(size=15, color=_FONT_COLOR, family=_FONT_FAMILY),
        x=0.5,
        xanchor="center",
    )


class ClassificationPlots:
    """
    Visualization suite for classification models.

    This class receives raw diagnostic data calculated by ``ClassificationDiagnostics``
    and generates interactive Plotly figures for analysis.

    Attributes
    ----------
    results : dict
        A dictionary containing the raw numerical arrays and DataFrames.
    plots : dict
        A dictionary to store generated :class:`plotly.graph_objects.Figure` objects.
    """

    def __init__(self, results: Dict[str, Any]) -> None:
        """
        Initialize the ClassificationPlots object.

        Parameters
        ----------
        results : dict
            The ``results`` dictionary from ``ClassificationDiagnostics``.
        """
        self.results = results
        self.plots: Dict[str, go.Figure] = {}

    # ------------------------------------------------------------------
    # Individual plots
    # ------------------------------------------------------------------

    def plot_metrics_table_test(self) -> None:
        """Generate a styled Plotly table for the test set metrics."""
        self._plot_metrics_table(subset="test")

    def plot_metrics_table_train(self) -> None:
        """Generate a styled Plotly table for the training set metrics."""
        self._plot_metrics_table(subset="train")

    def _plot_metrics_table(self, subset: str) -> None:
        metrics = self.results.get(f"classification_metrics_{subset}")
        if not metrics:
            return

        display_names = {
            "accuracy": "Accuracy",
            "precision": "Precision",
            "recall": "Recall (TPR)",
            "f1_score": "F1 Score",
        }

        names, values = [], []
        for key, label in display_names.items():
            if key in metrics:
                names.append(label)
                values.append(f"{metrics[key]:.4f}")

        if not names:
            return

        fig = go.Figure(
            go.Table(
                header=dict(
                    values=["<b>Metric</b>", "<b>Value</b>"],
                    fill_color="#f1f5f9",
                    align="center",
                    font=dict(family=_FONT_FAMILY, size=13, color=_FONT_COLOR),
                    line_color=_BORDER,
                    height=36,
                ),
                cells=dict(
                    values=[names, values],
                    fill_color=[["white"] * len(names)],
                    align="center",
                    font=dict(family=_FONT_FAMILY, size=13, color=_FONT_COLOR),
                    line_color=_BORDER,
                    height=32,
                ),
            )
        )
        layout_kwargs = _LAYOUT_BASE | {
            "title": _title_dict(f"Classification Metrics ({subset.capitalize()})"),
            "margin": dict(l=30, r=30, t=55, b=20),
        }
        fig.update_layout(**layout_kwargs)
        self.plots[f"metrics_table_{subset}"] = fig

    def plot_confusion_matrix_test(self, normalize: Optional[str] = None) -> None:
        """Generate confusion matrix heatmap for the test set."""
        self._plot_confusion_matrix(subset="test", normalize=normalize)

    def plot_confusion_matrix_train(self, normalize: Optional[str] = None) -> None:
        """Generate confusion matrix heatmap for the training set."""
        self._plot_confusion_matrix(subset="train", normalize=normalize)

    def _plot_confusion_matrix(
        self, subset: str, normalize: Optional[str] = None
    ) -> None:
        df_cm = self.results.get(f"confusion_matrix_{subset}")
        if df_cm is None:
            return

        z = df_cm.values
        x_labels = [str(c) for c in df_cm.columns]
        y_labels = [str(r) for r in df_cm.index]

        row_sums = z.sum(axis=1, keepdims=True)
        pct = np.where(row_sums > 0, z / row_sums * 100, 0.0)
        text = [
            [f"<b>{z[i, j]}</b><br>{pct[i, j]:.1f}%" for j in range(z.shape[1])]
            for i in range(z.shape[0])
        ]

        fig = go.Figure(
            go.Heatmap(
                z=z,
                x=x_labels,
                y=y_labels,
                text=text,
                texttemplate="%{text}",
                textfont=dict(size=13, color=_FONT_COLOR, family=_FONT_FAMILY),
                colorscale=_COLORSCALE,
                showscale=True,
                hovertemplate=(
                    "Predicted: <b>%{x}</b><br>"
                    "Actual: <b>%{y}</b><br>"
                    "Count: <b>%{z}</b><extra></extra>"
                ),
                colorbar=_COLORBAR_STYLE,
            )
        )
        fig.update_layout(**_LAYOUT_BASE)
        fig.update_layout(
            title=_title_dict(f"Confusion Matrix ({subset.capitalize()})"),
            xaxis=dict(title="Predicted", **_AXIS_BASE),
            yaxis=dict(title="Actual", autorange="reversed", **_AXIS_BASE),
        )
        self.plots[f"confusion_matrix_{subset}"] = fig

    def plot_misclassification_feature_plots_test(self) -> None:
        """Generate box+strip plots for the test set misclassifications."""
        self._plot_misclassification_feature_plots(subset="test")

    def plot_misclassification_feature_plots_train(self) -> None:
        """Generate box+strip plots for the training set misclassifications."""
        self._plot_misclassification_feature_plots(subset="train")

    def _plot_misclassification_feature_plots(self, subset: str) -> None:
        res_df = self.results.get(f"misclassification_analysis_{subset}")
        if res_df is None or getattr(res_df, "empty", True):
            return

        data = self.results.get(f"classification_data_{subset}")
        if not data:
            return
        X, y, y_pred = data["X"], data["y"], data["y_pred"]

        max_plots = 10
        features_to_plot = (
            res_df.loc[res_df["significant"], "feature"].head(max_plots).tolist()
        )
        mask_incorrect = y != y_pred
        created = []

        for col in features_to_plot:
            if col not in X.columns:
                continue
            plot_df = pd.DataFrame(
                {
                    col: X[col],
                    "Status": np.where(mask_incorrect, "Incorrect", "Correct"),
                }
            ).dropna()
            if plot_df.empty:
                continue

            fig = go.Figure()
            for vals, label, box_color, marker_color in [
                (
                    plot_df.loc[plot_df["Status"] == "Correct", col],
                    "Correct",
                    _ACCENT_BLUE,
                    "rgba(59,130,246,0.35)",
                ),
                (
                    plot_df.loc[plot_df["Status"] == "Incorrect", col],
                    "Incorrect",
                    _ACCENT_RED,
                    "rgba(239,68,68,0.35)",
                ),
            ]:
                if vals.empty:
                    continue

                rgb_values = [
                    int(box_color.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4)
                ]
                rgba_color = f"rgba{tuple(rgb_values + [0.12])}"

                fig.add_trace(
                    go.Box(
                        y=vals.tolist(),
                        name=label,
                        boxmean="sd",
                        marker=dict(color=box_color, size=4, opacity=0.6),
                        line=dict(color=box_color, width=1.5),
                        fillcolor=rgba_color,
                        whiskerwidth=0.6,
                        hovertemplate=f"<b>{label}</b><br>{col}: %{{y:.4g}}<extra></extra>",
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=[label] * len(vals),
                        y=vals.tolist(),
                        mode="markers",
                        showlegend=False,
                        marker=dict(color=marker_color, size=3.5),
                        hoverinfo="skip",
                    )
                )

            fig.update_layout(**_LAYOUT_BASE)
            fig.update_layout(
                title=_title_dict(
                    f"<b>{col}</b> — Correct vs Incorrect ({subset.capitalize()})"
                ),
                xaxis=dict(title="Classification Status", **_AXIS_BASE),
                yaxis=dict(title=col, **_AXIS_BASE),
                boxmode="group",
                showlegend=True,
                legend=dict(
                    bgcolor="rgba(255,255,255,0.9)", bordercolor=_BORDER, borderwidth=1
                ),
            )
            key = f"misclass_{col}_{subset}"
            self.plots[key] = fig
            created.append(key)

        self.results[f"misclassification_plots_{subset}"] = created

    def plot_class_distribution_test(self) -> None:
        """Plot the count of actual vs predicted classes for the test set."""
        self._plot_class_distribution("test")

    def plot_class_distribution_train(self) -> None:
        """Plot the count of actual vs predicted classes for the training set."""
        self._plot_class_distribution("train")

    def _plot_class_distribution(self, subset: str) -> None:
        data = self.results.get(f"classification_data_{subset}")
        if not data:
            return
        y, y_pred = data["y"], data["y_pred"]
        classes = np.unique(np.concatenate([y, y_pred]))

        act_counts = [np.sum(y == c) for c in classes]
        pred_counts = [np.sum(y_pred == c) for c in classes]

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=classes.tolist(),
                y=act_counts,
                name="Actual",
                marker_color=_ACCENT_BLUE,
            )
        )
        fig.add_trace(
            go.Bar(
                x=classes.tolist(),
                y=pred_counts,
                name="Predicted",
                marker_color=_ACCENT_RED,
            )
        )

        fig.update_layout(**_LAYOUT_BASE)
        fig.update_layout(
            title=_title_dict(f"Class Distribution ({subset.capitalize()})"),
            xaxis=dict(title="Class", type="category", **_AXIS_BASE),
            yaxis=dict(title="Count", **_AXIS_BASE),
            barmode="group",
            legend=dict(
                bgcolor="rgba(255,255,255,0.9)", bordercolor=_BORDER, borderwidth=1
            ),
        )
        self.plots[f"class_distribution_{subset}"] = fig

    def plot_roc_curve_test(self) -> None:
        """Plot ROC curve and AUC for the test set."""
        self._plot_roc_curve("test")

    def plot_roc_curve_train(self) -> None:
        """Plot ROC curve and AUC for the training set."""
        self._plot_roc_curve("train")

    def _plot_roc_curve(self, subset: str) -> None:
        roc_data = self.results.get(f"roc_curve_{subset}")
        if not roc_data:
            return

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                line=dict(color=_FONT_MUTED, dash="dash"),
                name="Random Guess",
                hoverinfo="skip",
            )
        )

        for key, vals in roc_data.items():
            name = (
                "Binary ROC"
                if key == "binary"
                else f"Class {key.replace('class_', '')} ROC"
            )
            name += f" (AUC = {vals['auc']:.4f})"
            fig.add_trace(
                go.Scatter(
                    x=vals["fpr"].tolist(),
                    y=vals["tpr"].tolist(),
                    mode="lines",
                    name=name,
                    hovertemplate="FPR: %{x:.4f}<br>TPR: %{y:.4f}<extra></extra>",
                )
            )

        fig.update_layout(**_LAYOUT_BASE)
        fig.update_layout(
            title=_title_dict(f"ROC Curve ({subset.capitalize()})"),
            xaxis=dict(title="False Positive Rate", range=[-0.05, 1.05], **_AXIS_BASE),
            yaxis=dict(title="True Positive Rate", range=[-0.05, 1.05], **_AXIS_BASE),
            legend=dict(
                bgcolor="rgba(255,255,255,0.9)", bordercolor=_BORDER, borderwidth=1
            ),
        )
        self.plots[f"roc_curve_{subset}"] = fig

    def plot_pr_curve_test(self) -> None:
        """Plot Precision-Recall curve and Average Precision for the test set."""
        self._plot_pr_curve("test")

    def plot_pr_curve_train(self) -> None:
        """Plot Precision-Recall curve and Average Precision for the training set."""
        self._plot_pr_curve("train")

    def _plot_pr_curve(self, subset: str) -> None:
        pr_data = self.results.get(f"pr_curve_{subset}")
        if not pr_data:
            return

        fig = go.Figure()
        for key, vals in pr_data.items():
            name = (
                "Binary PR"
                if key == "binary"
                else f"Class {key.replace('class_', '')} PR"
            )
            name += f" (AP = {vals['ap']:.4f})"
            fig.add_trace(
                go.Scatter(
                    x=vals["recall"].tolist(),
                    y=vals["precision"].tolist(),
                    mode="lines",
                    name=name,
                    hovertemplate="Recall: %{x:.4f}<br>Precision: %{y:.4f}<extra></extra>",
                )
            )
            fig.add_hline(
                y=vals["baseline"],
                line=dict(dash="dot", width=1.5, color=_FONT_MUTED),
                annotation_text=f"Baseline ({key})",
                annotation_position="bottom right",
                annotation_font=dict(size=10, color=_FONT_MUTED),
            )

        fig.update_layout(**_LAYOUT_BASE)
        fig.update_layout(
            title=_title_dict(f"Precision-Recall Curve ({subset.capitalize()})"),
            xaxis=dict(title="Recall", range=[-0.05, 1.05], **_AXIS_BASE),
            yaxis=dict(title="Precision", range=[-0.05, 1.05], **_AXIS_BASE),
            legend=dict(
                bgcolor="rgba(255,255,255,0.9)", bordercolor=_BORDER, borderwidth=1
            ),
        )
        self.plots[f"pr_curve_{subset}"] = fig

    def plot_probability_distribution_test(self) -> None:
        """Plot KDE of classification confidence (probability) for the test set."""
        self._plot_probability_distribution("test")

    def plot_probability_distribution_train(self) -> None:
        """Plot KDE of classification confidence (probability) for the training set."""
        self._plot_probability_distribution("train")

    def _plot_probability_distribution(self, subset: str) -> None:
        data = self.results.get(f"classification_data_{subset}")
        if not data or data.get("y_prob") is None:
            return

        y, y_pred, y_prob = data["y"], data["y_pred"], data["y_prob"]

        if y_prob.ndim > 1 and y_prob.shape[1] > 1:
            max_probs = np.max(y_prob, axis=1)
        else:
            probs_flat = y_prob.ravel()
            classes = np.unique(np.concatenate([y, y_pred]))
            if len(classes) == 2:
                # Assuming probs_flat is probability of class 1.
                # If pred is class 1, confidence is p. If pred is class 0, confidence is 1-p.
                max_probs = np.where(y_pred == classes[1], probs_flat, 1.0 - probs_flat)
            else:
                max_probs = probs_flat

        correct_probs = max_probs[y == y_pred]
        incorrect_probs = max_probs[y != y_pred]

        x_grid = np.linspace(0, 1, 200)
        fig = go.Figure()

        if len(correct_probs) > 1:
            try:
                kde_c = scipy_stats.gaussian_kde(correct_probs)(x_grid)
                fig.add_trace(
                    go.Scatter(
                        x=x_grid.tolist(),
                        y=kde_c.tolist(),
                        mode="lines",
                        name="Correct Prediction",
                        line=dict(color=_ACCENT_BLUE, width=2.5),
                        fill="tozeroy",
                        fillcolor="rgba(59,130,246,0.25)",
                        hovertemplate="Confidence: %{x:.3f}<br>Density: %{y:.3f}<extra></extra>",
                    )
                )
            except Exception:
                pass

        if len(incorrect_probs) > 1:
            try:
                kde_i = scipy_stats.gaussian_kde(incorrect_probs)(x_grid)
                fig.add_trace(
                    go.Scatter(
                        x=x_grid.tolist(),
                        y=kde_i.tolist(),
                        mode="lines",
                        name="Incorrect Prediction",
                        line=dict(color=_ACCENT_RED, width=2, dash="dash"),
                        fill="tozeroy",
                        fillcolor="rgba(239,68,68,0.25)",
                        hovertemplate="Confidence: %{x:.3f}<br>Density: %{y:.3f}<extra></extra>",
                    )
                )
            except Exception:
                pass

        fig.update_layout(**_LAYOUT_BASE)
        fig.update_layout(
            title=_title_dict(
                f"Prediction Confidence Distribution ({subset.capitalize()})"
            ),
            xaxis=dict(
                title="Prediction Confidence (Max Probability)",
                range=[0, 1],
                **_AXIS_BASE,
            ),
            yaxis=dict(title="Density", **_AXIS_BASE),
            legend=dict(
                bgcolor="rgba(255,255,255,0.9)", bordercolor=_BORDER, borderwidth=1
            ),
        )
        self.plots[f"probability_distribution_{subset}"] = fig

    def run_all(self) -> None:
        """Generate the symmetric suite of classification plots."""
        self.plots = {}
        for subset in ["train", "test"]:
            getattr(self, f"plot_class_distribution_{subset}")()
            getattr(self, f"plot_metrics_table_{subset}")()
            getattr(self, f"plot_confusion_matrix_{subset}")()
            getattr(self, f"plot_roc_curve_{subset}")()
            getattr(self, f"plot_pr_curve_{subset}")()
            getattr(self, f"plot_probability_distribution_{subset}")()
            getattr(self, f"plot_misclassification_feature_plots_{subset}")()
