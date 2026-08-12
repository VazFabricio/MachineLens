"""Visualization layer for the MachineLens library.

This module provides the ``DiagnosticPlotter`` class, which consumes
``DiagnosticResults`` and generates highly polished, interactive Plotly
charts. These charts are styled with a premium dark-slate aesthetic
and designed for zero-friction JSON serialization.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import gaussian_kde, norm

from machinelens.core.data_classes import DiagnosticResults, LowessData, QQData

# ---------------------------------------------------------------------------
# Design Tokens & Styling Constants (Clean Light White Theme)
# ---------------------------------------------------------------------------
_FONT_FAMILY = "Inter, system-ui, -apple-system, sans-serif"

# Clean Light Palette
_COLOR_BG_PAPER = "#ffffff"  # Pure white background
_COLOR_BG_PLOT = "#fafbfc"  # Soft off-white surface
_COLOR_GRID = "rgba(0, 0, 0, 0.06)"  # Very faint dark gridlines
_COLOR_BORDER = "rgba(0, 0, 0, 0.08)"  # Soft dark border

# Typography Colors
_COLOR_TEXT_MAIN = "#1e293b"  # Slate 800 (Deep dark text)
_COLOR_TEXT_MUTED = "#64748b"  # Slate 500 (Muted gray-blue)

# Accent Colors (Vibrant Light Theme Accents)
_COLOR_BLUE = "#3b82f6"  # Bright Blue
_COLOR_GREEN = "#10b981"  # Emerald Green
_COLOR_AMBER = "#f59e0b"  # Amber
_COLOR_RED = "#ef4444"  # Vibrant Red
_COLOR_PURPLE = "#a855f7"  # Indigo-Purple
_COLOR_PINK = "#ec4899"  # Clean Pink

# Modern Premium Color Scale for Continuous Heatmaps/Gradients
_COLORSCALE_RESIDUALS = [
    [0.0, "#3b82f6"],  # Deep Blue (Small residuals)
    [0.4, "#a855f7"],  # Purple
    [0.7, "#f97316"],  # Orange
    [1.0, "#ef4444"],  # Bright Red (High residuals)
]

_COLORSCALE_HEATMAP = [
    [0.0, "#f8fafc"],  # Slate 50
    [0.5, "#93c5fd"],  # Soft Sky Blue
    [1.0, "#3b82f6"],  # Vibrant Deep Blue
]


# ======================================================================
# DiagnosticPlotter
# ======================================================================


class DiagnosticPlotter:
    """Visualization suite for MachineLens diagnostic results.

    This class reads from a strongly-typed ``DiagnosticResults`` object
    and generates interactive, premium Plotly figures. It is completely
    decoupled from any computing tasks, acting purely as a visual
    rendering engine.

    All figure objects returned can be easily serialized to JSON via
    ``fig.to_json()`` for rendering in web frontends.

    Parameters
    ----------
    results : DiagnosticResults
        The complete calculated diagnostic results to visualize.
    """

    def __init__(self, results: DiagnosticResults) -> None:
        """Initialize the DiagnosticPlotter.

        Parameters
        ----------
        results : DiagnosticResults
            The complete calculated diagnostic results to visualize.
        """
        self.results = results

    # ------------------------------------------------------------------
    # Core Theme Applier (Private)
    # ------------------------------------------------------------------

    def _compact_colorbar(self, title_text: str) -> dict:
        """Create a highly compact, thin, and elegantly styled colorbar.

        Parameters
        ----------
        title_text : str
            The label for the colorbar title.

        Returns
        -------
        dict
            Plotly colorbar layout dictionary.
        """
        return dict(
            title=dict(
                text=title_text,
                font=dict(size=8, color=_COLOR_TEXT_MUTED),
            ),
            tickfont=dict(color=_COLOR_TEXT_MUTED, size=8),
            thickness=10,  # Elegant narrow width
            len=0.7,  # Slightly shorter to avoid overlapping
            xpad=5,  # Reduce horizontal padding
            ypad=5,
        )

    def _apply_theme(
        self,
        fig: go.Figure,
        title: str,
        xaxis_title: str,
        yaxis_title: str,
        show_legend: bool = True,
    ) -> None:
        """Apply an ultra-premium clean theme to a Plotly figure.

        Parameters
        ----------
        fig : go.Figure
            The Plotly figure to customize.
        title : str
            The title text of the chart.
        xaxis_title : str
            The label for the x-axis.
        yaxis_title : str
            The label for the y-axis.
        show_legend : bool, default=True
            Whether to show the legend.
        """
        fig.update_layout(
            title=dict(
                text=f"<b>{title}</b>",
                font=dict(family=_FONT_FAMILY, color=_COLOR_TEXT_MAIN, size=15),
                x=0.05,
                xanchor="left",
                y=0.96,
            ),
            paper_bgcolor=_COLOR_BG_PAPER,
            plot_bgcolor=_COLOR_BG_PLOT,
            font=dict(family=_FONT_FAMILY, color=_COLOR_TEXT_MAIN, size=11),
            margin=dict(l=55, r=20, t=65, b=45),
            showlegend=show_legend,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.01,
                xanchor="left",
                x=0.02,
                font=dict(size=9, color=_COLOR_TEXT_MUTED),
                bgcolor="rgba(0,0,0,0)",
                bordercolor="rgba(0,0,0,0)",
                borderwidth=0,
            ),
            hoverlabel=dict(
                bgcolor=_COLOR_BG_PAPER,
                bordercolor=_COLOR_BORDER,
                font=dict(family=_FONT_FAMILY, color=_COLOR_TEXT_MAIN, size=11),
            ),
            hovermode="closest",
        )

        fig.update_xaxes(
            title=dict(
                text=xaxis_title,
                font=dict(size=12, color=_COLOR_TEXT_MUTED, family=_FONT_FAMILY),
            ),
            gridcolor=_COLOR_GRID,
            linecolor=_COLOR_BORDER,
            tickfont=dict(color=_COLOR_TEXT_MUTED, size=10),
            zerolinecolor=_COLOR_GRID,
            zerolinewidth=1,
        )

        fig.update_yaxes(
            title=dict(
                text=yaxis_title,
                font=dict(size=12, color=_COLOR_TEXT_MUTED, family=_FONT_FAMILY),
            ),
            gridcolor=_COLOR_GRID,
            linecolor=_COLOR_BORDER,
            tickfont=dict(color=_COLOR_TEXT_MUTED, size=10),
            zerolinecolor=_COLOR_GRID,
            zerolinewidth=1,
        )

    def _add_lowess_trace(
        self,
        fig: go.Figure,
        lowess: Optional[LowessData],
        name: str = "LOWESS Curve",
        color: str = _COLOR_PURPLE,
    ) -> None:
        """Add a LOWESS curve and confidence band to a scatter plot.

        Parameters
        ----------
        fig : go.Figure
            The Plotly figure to modify.
        lowess : LowessData or None
            The calculated LOWESS trend line and confidence interval.
        name : str, default="LOWESS Curve"
            The trace label for the trend line.
        color : str, default=_COLOR_PURPLE
            The line color string.
        """
        if lowess is None or len(lowess.x_smooth) == 0:
            return

        # Shade confidence interval
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([lowess.x_smooth, lowess.x_smooth[::-1]]).tolist(),
                y=np.concatenate([lowess.ci_upper, lowess.ci_lower[::-1]]).tolist(),
                fill="toself",
                fillcolor="rgba(192, 132, 252, 0.12)",  # Translucent purple
                line=dict(color="rgba(255,255,255,0)"),
                showlegend=False,
                name=f"{name} 95% CI",
                hoverinfo="skip",
            )
        )

        # Draw trend line
        fig.add_trace(
            go.Scatter(
                x=lowess.x_smooth.tolist(),
                y=lowess.y_smooth.tolist(),
                mode="lines",
                line=dict(color=color, width=2.5, shape="spline"),
                name=name,
                hoverinfo="skip",
            )
        )

    # ------------------------------------------------------------------
    # Regression Plotting Methods
    # ------------------------------------------------------------------

    def plot_actual_vs_predicted(self, subset: str = "test") -> go.Figure:
        """Plot Actual vs. Predicted values.

        Includes an identity reference line (y = x) representing ideal fit.
        Points are colored by absolute residual values to highlight mistakes.

        Parameters
        ----------
        subset : str, default="test"
            The data subset to plot (``"train"`` or ``"test"``).

        Returns
        -------
        go.Figure
            The Plotly figure object.
        """
        sd = getattr(self.results, f"{subset}_data", None)
        if sd is None:
            raise ValueError(f"No {subset} subset data available in results.")

        fig = go.Figure()

        # Aligned lists
        y_true = sd.y_true.tolist()
        y_pred = sd.y_pred.tolist()
        abs_res = getattr(sd, "abs_residuals", np.abs(sd.y_true - sd.y_pred)).tolist()

        # Add perfect fit identity line (y = x)
        min_val = float(min(min(y_true), min(y_pred)))
        max_val = float(max(max(y_true), max(y_pred)))
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode="lines",
                line=dict(color=_COLOR_TEXT_MUTED, width=1.5, dash="dash"),
                name="Perfect Prediction (y=x)",
                hoverinfo="skip",
            )
        )

        # Add observations scatter trace
        vmax = float(np.percentile(abs_res, 95)) if len(abs_res) > 0 else 1.0
        fig.add_trace(
            go.Scatter(
                x=y_true,
                y=y_pred,
                mode="markers",
                name="Observations",
                marker=dict(
                    size=7,
                    color=abs_res,
                    colorscale=_COLORSCALE_RESIDUALS,
                    cmin=0,
                    cmax=vmax,
                    colorbar=self._compact_colorbar("|Error|"),
                    line=dict(color=_COLOR_BG_PAPER, width=0.5),
                ),
                hovertemplate=(
                    "<b>Actual:</b> %{x:.4f}<br>"
                    "<b>Predicted:</b> %{y:.4f}<br>"
                    "<b>Absolute Error:</b> %{marker.color:.4f}<extra></extra>"
                ),
                customdata=sd.X_data.index.tolist(),
            )
        )

        self._apply_theme(
            fig,
            title=f"Actual vs. Predicted Values ({subset.capitalize()})",
            xaxis_title="Actual Values",
            yaxis_title="Predicted Values",
            show_legend=True,
        )

        return fig

    def plot_residuals(self, subset: str = "test") -> go.Figure:
        """Plot Residuals vs. Predicted values.

        Overlays a horizontal zero line (y = 0) and the LOWESS trend line
        with bootstrap confidence intervals to identify systematic bias or
        non-linearity.

        Parameters
        ----------
        subset : str, default="test"
            The data subset to plot (``"train"`` or ``"test"``).

        Returns
        -------
        go.Figure
            The Plotly figure object.
        """
        sd = getattr(self.results, f"{subset}_data", None)
        if sd is None:
            raise ValueError(f"No {subset} subset data available in results.")

        fig = go.Figure()

        y_pred = sd.y_pred.tolist()
        residuals = getattr(sd, "residuals", sd.y_true - sd.y_pred).tolist()
        abs_res = getattr(sd, "abs_residuals", np.abs(residuals)).tolist()

        # Horizontal zero-reference line
        fig.add_hline(
            y=0,
            line=dict(color=_COLOR_TEXT_MUTED, width=1.5, dash="dash"),
            annotation_text="Zero Residual",
            annotation_position="bottom right",
            annotation_font=dict(size=9, color=_COLOR_TEXT_MUTED),
        )

        # Plotly scatter points
        vmax = float(np.percentile(abs_res, 95)) if len(abs_res) > 0 else 1.0
        fig.add_trace(
            go.Scatter(
                x=y_pred,
                y=residuals,
                mode="markers",
                name="Residuals",
                marker=dict(
                    size=7,
                    color=abs_res,
                    colorscale=_COLORSCALE_RESIDUALS,
                    cmin=0,
                    cmax=vmax,
                    colorbar=self._compact_colorbar("|Error|"),
                    line=dict(color=_COLOR_BG_PAPER, width=0.5),
                ),
                hovertemplate=(
                    "<b>Predicted:</b> %{x:.4f}<br>"
                    "<b>Residual:</b> %{y:.4f}<br>"
                    "<b>Absolute Error:</b> %{marker.color:.4f}<extra></extra>"
                ),
                customdata=sd.X_data.index.tolist(),
            )
        )

        # Add optional LOWESS overlay
        lowess = getattr(self.results, f"{subset}_linearity_lowess", None)
        self._add_lowess_trace(fig, lowess, name="LOWESS Trend", color=_COLOR_PURPLE)

        self._apply_theme(
            fig,
            title=f"Residuals vs. Predicted Values ({subset.capitalize()})",
            xaxis_title="Predicted Values",
            yaxis_title="Residual (Actual - Predicted)",
            show_legend=True,
        )

        return fig

    def plot_qq(self, subset: str = "test") -> go.Figure:
        """Plot a Normal Q-Q Plot of Standardized Residuals.

        Highlights departures from normality with a 95% confidence
        interval envelope. Points are dynamically colored by their distance
        from the theoretical line.

        Parameters
        ----------
        subset : str, default="test"
            The data subset to plot (``"train"`` or ``"test"``).

        Returns
        -------
        go.Figure
            The Plotly figure object.
        """
        qq: Optional[QQData] = getattr(self.results, f"{subset}_qq", None)
        if qq is None:
            raise ValueError(f"No {subset} Q-Q plot coordinates available in results.")
        sd = getattr(self.results, f"{subset}_data", None)

        fig = go.Figure()

        # Add 95% Confidence Interval Envelope
        ref_line = qq.intercept + qq.slope * qq.theoretical
        ci_lower_y = qq.intercept + qq.slope * qq.ci_lower
        ci_upper_y = qq.intercept + qq.slope * qq.ci_upper

        fig.add_trace(
            go.Scatter(
                x=np.concatenate([qq.theoretical, qq.theoretical[::-1]]).tolist(),
                y=np.concatenate([ci_upper_y, ci_lower_y[::-1]]).tolist(),
                fill="toself",
                fillcolor="rgba(56, 189, 248, 0.08)",  # Sky Blue translucent
                line=dict(color="rgba(255,255,255,0)"),
                showlegend=True,
                name="95% CI Envelope",
                hoverinfo="skip",
            )
        )

        # Add perfect normal theoretical reference line
        fig.add_trace(
            go.Scatter(
                x=qq.theoretical.tolist(),
                y=ref_line.tolist(),
                mode="lines",
                line=dict(color=_COLOR_TEXT_MUTED, width=1.5, dash="dash"),
                name="Normal Distribution",
                hoverinfo="skip",
            )
        )

        # Calculate absolute standardized residuals to map colors consistently with |Error|
        abs_std_res = np.abs(qq.sample)
        vmax = float(np.percentile(abs_std_res, 95)) if len(abs_std_res) > 0 else 1.0

        # Add sample quantiles scatter points
        orig_indices = []
        if (
            sd is not None
            and hasattr(sd, "std_residuals")
            and sd.std_residuals is not None
        ):
            sorted_positions = np.argsort(sd.std_residuals).tolist()
            dataset_idx = sd.X_data.index.tolist()
            orig_indices = [dataset_idx[i] for i in sorted_positions]

        fig.add_trace(
            go.Scatter(
                x=qq.theoretical.tolist(),
                y=qq.sample.tolist(),
                mode="markers",
                name="Sample Quantiles",
                marker=dict(
                    size=6,
                    color=abs_std_res.tolist(),
                    colorscale=_COLORSCALE_RESIDUALS,
                    cmin=0,
                    cmax=vmax,
                    colorbar=self._compact_colorbar("|Error|"),
                    line=dict(color=_COLOR_BG_PAPER, width=0.5),
                ),
                hovertemplate=(
                    "<b>Theoretical Quantile:</b> %{x:.4f}<br>"
                    "<b>Sample Quantile:</b> %{y:.4f}<extra></extra>"
                ),
                customdata=orig_indices,
            )
        )

        self._apply_theme(
            fig,
            title=f"Normal Q-Q Plot of Residuals ({subset.capitalize()})",
            xaxis_title="Theoretical Quantiles (Standard Normal)",
            yaxis_title="Sample Quantiles (Standardized Residuals)",
            show_legend=True,
        )

        return fig

    def plot_scale_location(self, subset: str = "test") -> go.Figure:
        """Generate a Scale-Location Plot.

        Plots Predicted Values vs. sqrt(|Standardized Residuals|).
        Overlays a LOWESS smoothed line to check homoscedasticity. A flat
        trend line indicates constant residual variance.

        Parameters
        ----------
        subset : str, default="test"
            The data subset to plot (``"train"`` or ``"test"``).

        Returns
        -------
        go.Figure
            The Plotly figure object.
        """
        sd = getattr(self.results, f"{subset}_data", None)
        if sd is None:
            raise ValueError(f"No {subset} subset data available in results.")

        fig = go.Figure()

        y_pred = sd.y_pred.tolist()
        std_res = getattr(sd, "std_residuals", np.zeros_like(sd.y_pred))
        sqrt_abs_res = np.sqrt(np.abs(std_res)).tolist()

        abs_std_res = np.abs(std_res)
        vmax = float(np.percentile(abs_std_res, 95)) if len(abs_std_res) > 0 else 1.0

        fig.add_trace(
            go.Scatter(
                x=y_pred,
                y=sqrt_abs_res,
                mode="markers",
                name="Residuals",
                marker=dict(
                    size=7,
                    color=abs_std_res.tolist(),
                    colorscale=_COLORSCALE_RESIDUALS,
                    cmin=0,
                    cmax=vmax,
                    colorbar=self._compact_colorbar("|Error|"),
                    line=dict(color=_COLOR_BG_PAPER, width=0.5),
                ),
                hovertemplate=(
                    "<b>Predicted:</b> %{x:.4f}<br>"
                    "<b>√|Std. Residual|:</b> %{y:.4f}<extra></extra>"
                ),
                customdata=sd.X_data.index.tolist(),
            )
        )

        # Overlay LOWESS smoothed line
        lowess = getattr(self.results, f"{subset}_scale_loc_lowess", None)
        self._add_lowess_trace(
            fig, lowess, name="Homoscedasticity Trend", color=_COLOR_PURPLE
        )

        self._apply_theme(
            fig,
            title=f"Scale-Location Plot ({subset.capitalize()})",
            xaxis_title="Predicted Values",
            yaxis_title="√|Standardized Residuals|",
            show_legend=True,
        )

        return fig

    def plot_leverage(self) -> go.Figure:
        """Plot Standardized Residuals vs Leverage.

        Useful for identifying highly influential data points or outliers in
        predictor space. Marker sizes are proportional to Cook's Distance.

        Returns
        -------
        go.Figure
            The Plotly figure object.
        """
        leverage = self.results.leverage
        if leverage is None:
            raise ValueError("No leverage data available in results.")

        train_sd = self.results.train_data
        if train_sd is None or train_sd.std_residuals is None:
            raise ValueError("No training data available for leverage plotting.")

        fig = go.Figure()

        std_res = train_sd.std_residuals.tolist()
        lev_list = leverage.tolist()
        cooks = self.results.cooks_distance

        marker_sizes = [6] * len(lev_list)
        if cooks is not None:
            # Map cooks distance to marker sizes dynamically (range 6 to 24)
            c_max = float(np.max(cooks)) if len(cooks) > 0 else 1.0
            marker_sizes = (6 + 18 * (cooks / (c_max + 1e-12))).tolist()

        abs_std_res = np.abs(train_sd.std_residuals).tolist()
        vmax = float(np.percentile(abs_std_res, 95)) if len(abs_std_res) > 0 else 1.0

        fig.add_trace(
            go.Scatter(
                x=lev_list,
                y=std_res,
                mode="markers",
                name="Observations",
                marker=dict(
                    size=marker_sizes,
                    color=abs_std_res,
                    colorscale=_COLORSCALE_RESIDUALS,
                    cmin=0,
                    cmax=vmax,
                    colorbar=self._compact_colorbar("|Error|"),
                    line=dict(color=_COLOR_BG_PAPER, width=0.5),
                ),
                text=(
                    [f"Cook's D: {c:.4f}" for c in cooks.tolist()]
                    if cooks is not None
                    else []
                ),
                hovertemplate=(
                    "<b>Leverage:</b> %{x:.4f}<br>"
                    "<b>Std. Residual:</b> %{y:.4f}<br>"
                    "<b>Cook's Distance:</b> %{marker.color:.4f}<extra></extra>"
                ),
                customdata=train_sd.X_data.index.tolist(),
            )
        )

        # Plot reference thresholds at +/- 3 standardized residuals
        fig.add_hline(
            y=3,
            line=dict(color=_COLOR_RED, width=1.2, dash="dash"),
            name="Outlier limit",
        )
        fig.add_hline(y=-3, line=dict(color=_COLOR_RED, width=1.2, dash="dash"))
        fig.add_hline(y=0, line=dict(color=_COLOR_TEXT_MUTED, width=1))

        # LOWESS Trend Line
        self._add_lowess_trace(
            fig, self.results.leverage_lowess, name="Leverage Trend", color=_COLOR_BLUE
        )

        self._apply_theme(
            fig,
            title="Residuals vs. Leverage (Train)",
            xaxis_title="Leverage (Hat Values)",
            yaxis_title="Standardized Residuals",
            show_legend=True,
        )

        return fig

    # ------------------------------------------------------------------
    # Classification Plotting Methods
    # ------------------------------------------------------------------

    def plot_confusion_matrix(self, subset: str = "test") -> go.Figure:
        """Plot a labeled, proportional Confusion Matrix heatmap.

        Shows both raw sample count and row-wise accuracy percentages
        in each cell.

        Parameters
        ----------
        subset : str, default="test"
            The data subset to plot (``"train"`` or ``"test"``).

        Returns
        -------
        go.Figure
            The Plotly figure object.
        """
        cm_df = getattr(self.results, f"{subset}_confusion_matrix", None)
        if cm_df is None:
            raise ValueError(f"No {subset} confusion matrix available in results.")

        z = cm_df.values
        x_labels = [f"Class {c}" for c in cm_df.columns]
        y_labels = [f"Class {r}" for r in cm_df.index]

        # Calculate row-wise percentages (proportions)
        row_sums = z.sum(axis=1, keepdims=True)
        z_pct = np.where(row_sums > 0, z / row_sums * 100, 0.0)

        # Custom cell text formatting
        cell_text = []
        for i in range(len(y_labels)):
            row_text = []
            for j in range(len(x_labels)):
                row_text.append(f"<b>{z[i, j]}</b><br>{z_pct[i, j]:.1f}%")
            cell_text.append(row_text)

        fig = go.Figure(
            go.Heatmap(
                z=z_pct.tolist(),  # Color map based on proportional percentages
                x=x_labels,
                y=y_labels,
                text=cell_text,
                texttemplate="%{text}",
                textfont=dict(size=12, color=_COLOR_TEXT_MAIN, family=_FONT_FAMILY),
                colorscale=_COLORSCALE_HEATMAP,
                showscale=True,
                colorbar=dict(
                    title=dict(
                        text="Recall %", font=dict(size=8, color=_COLOR_TEXT_MUTED)
                    ),
                    ticksuffix="%",
                    tickfont=dict(color=_COLOR_TEXT_MUTED, size=8),
                    thickness=10,
                    len=0.7,
                    xpad=5,
                    ypad=5,
                ),
                hovertemplate=(
                    "<b>Predicted:</b> %{x}<br>"
                    "<b>Actual:</b> %{y}<br>"
                    "<b>Recall Accuracy:</b> %{z:.2f}%<extra></extra>"
                ),
            )
        )

        self._apply_theme(
            fig,
            title=f"Confusion Matrix ({subset.capitalize()})",
            xaxis_title="Predicted Class",
            yaxis_title="Actual Class",
            show_legend=False,
        )

        fig.update_xaxes(type="category")
        fig.update_yaxes(type="category", autorange="reversed")

        return fig

    def plot_roc_curve(self, subset: str = "test") -> go.Figure:
        """Plot Receiver Operating Characteristic (ROC) curves.

        Overlays diagonal no-skill guideline and calculates AUC scores.
        Handles both binary and multi-class classification formats seamlessly.

        Parameters
        ----------
        subset : str, default="test"
            The data subset to plot (``"train"`` or ``"test"``).

        Returns
        -------
        go.Figure
            The Plotly figure object.
        """
        curves = getattr(self.results, f"{subset}_roc_curves", None)
        if not curves:
            raise ValueError(f"No {subset} ROC curve coordinates calculated.")

        fig = go.Figure()

        # Add random guess guideline (y = x)
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                line=dict(color=_COLOR_TEXT_MUTED, width=1.5, dash="dash"),
                name="Random Guess (AUC = 0.50)",
                hoverinfo="skip",
            )
        )

        # Draw ROC curve for each class
        colors = [_COLOR_BLUE, _COLOR_GREEN, _COLOR_PURPLE, _COLOR_PINK, _COLOR_AMBER]
        for idx, curve in enumerate(curves):
            c_color = colors[idx % len(colors)]
            c_name = (
                "Model ROC"
                if curve.label == "binary"
                else f"Class {curve.label.replace('class_', '')}"
            )
            fig.add_trace(
                go.Scatter(
                    x=curve.fpr.tolist(),
                    y=curve.tpr.tolist(),
                    mode="lines",
                    line=dict(color=c_color, width=2.5),
                    name=f"{c_name} (AUC = {curve.auc_score:.4f})",
                    hovertemplate=(
                        "<b>FPR (1-Spec):</b> %{x:.4f}<br>"
                        "<b>TPR (Sens):</b> %{y:.4f}<extra></extra>"
                    ),
                )
            )

        self._apply_theme(
            fig,
            title=f"Receiver Operating Characteristic (ROC) Curve ({subset.capitalize()})",
            xaxis_title="False Positive Rate (1 - Specificity)",
            yaxis_title="True Positive Rate (Sensitivity / Recall)",
            show_legend=True,
        )
        fig.update_xaxes(range=[-0.01, 1.01])
        fig.update_yaxes(range=[-0.01, 1.01])

        return fig

    def plot_pr_curve(self, subset: str = "test") -> go.Figure:
        """Plot Precision-Recall (PR) curves with baseline reference line.

        Highly recommended for class-imbalanced datasets. Includes Average
        Precision (AP) scores.

        Parameters
        ----------
        subset : str, default="test"
            The data subset to plot (``"train"`` or ``"test"``).

        Returns
        -------
        go.Figure
            The Plotly figure object.
        """
        curves = getattr(self.results, f"{subset}_pr_curves", None)
        if not curves:
            raise ValueError(f"No {subset} PR curve coordinates calculated.")

        fig = go.Figure()

        colors = [_COLOR_BLUE, _COLOR_GREEN, _COLOR_PURPLE, _COLOR_PINK, _COLOR_AMBER]
        for idx, curve in enumerate(curves):
            c_color = colors[idx % len(colors)]
            c_name = (
                "Model PR"
                if curve.label == "binary"
                else f"Class {curve.label.replace('class_', '')}"
            )

            # Draw PR curve
            fig.add_trace(
                go.Scatter(
                    x=curve.recall_arr.tolist(),
                    y=curve.precision_arr.tolist(),
                    mode="lines",
                    line=dict(color=c_color, width=2.5),
                    name=f"{c_name} (AP = {curve.average_precision:.4f})",
                    hovertemplate=(
                        "<b>Recall:</b> %{x:.4f}<br>"
                        "<b>Precision:</b> %{y:.4f}<extra></extra>"
                    ),
                )
            )

            # Draw baseline reference line
            fig.add_hline(
                y=curve.baseline,
                line=dict(color=c_color, width=1.2, dash="dot"),
                annotation_text=f"Baseline ({curve.baseline:.2f})",
                annotation_position="bottom left",
                annotation_font=dict(size=8, color=_COLOR_TEXT_MUTED),
            )

        self._apply_theme(
            fig,
            title=f"Precision-Recall Curve ({subset.capitalize()})",
            xaxis_title="Recall",
            yaxis_title="Precision",
            show_legend=True,
        )
        fig.update_xaxes(range=[-0.01, 1.01])
        fig.update_yaxes(range=[-0.01, 1.01])

        return fig

    def plot_calibration_curve(self, subset: str = "test") -> go.Figure:
        """Plot Calibration curves (Reliability diagrams).

        Parameters
        ----------
        subset : str, default="test"
            The data subset to plot (``"train"`` or ``"test"``).

        Returns
        -------
        go.Figure
            The Plotly figure object.
        """
        curves = getattr(self.results, f"{subset}_calibration_curves", None)
        if not curves:
            raise ValueError(f"No {subset} calibration curves calculated.")

        fig = go.Figure()

        # Add perfect calibration line
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                line=dict(color=_COLOR_TEXT_MUTED, width=1.5, dash="dash"),
                name="Perfect Calibration",
                hoverinfo="skip",
            )
        )

        colors = [_COLOR_BLUE, _COLOR_GREEN, _COLOR_PURPLE, _COLOR_PINK, _COLOR_AMBER]
        for idx, curve in enumerate(curves):
            c_color = colors[idx % len(colors)]
            c_name = (
                "Model Calibration"
                if curve.label == "binary"
                else f"Class {curve.label.replace('class_', '')}"
            )

            fig.add_trace(
                go.Scatter(
                    x=curve.prob_pred.tolist(),
                    y=curve.prob_true.tolist(),
                    mode="lines+markers",
                    line=dict(color=c_color, width=2.5),
                    marker=dict(size=6),
                    name=c_name,
                    hovertemplate=(
                        "<b>Pred Prob:</b> %{x:.4f}<br>"
                        "<b>True Prob:</b> %{y:.4f}<extra></extra>"
                    ),
                )
            )

        self._apply_theme(
            fig,
            title=f"Calibration Curve ({subset.capitalize()})",
            xaxis_title="Mean Predicted Probability",
            yaxis_title="Fraction of Positives",
            show_legend=True,
        )
        fig.update_xaxes(range=[-0.01, 1.01])
        fig.update_yaxes(range=[-0.01, 1.01])

        return fig

    def plot_threshold_analysis(self, subset: str = "test") -> go.Figure:
        """Plot Precision, Recall, and F1-Score across thresholds.

        Parameters
        ----------
        subset : str, default="test"
            The data subset to plot (``"train"`` or ``"test"``).

        Returns
        -------
        go.Figure
            The Plotly figure object.
        """
        analyses = getattr(self.results, f"{subset}_threshold_analysis", None)
        if not analyses:
            raise ValueError(f"No {subset} threshold analysis calculated.")

        curve = analyses[0]
        fig = go.Figure()

        c_name = (
            ""
            if curve.label == "binary"
            else f" (Class {curve.label.replace('class_', '')})"
        )

        fig.add_trace(
            go.Scatter(
                x=curve.thresholds.tolist(),
                y=curve.precision.tolist(),
                mode="lines",
                line=dict(color=_COLOR_BLUE, width=2),
                name=f"Precision{c_name}",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=curve.thresholds.tolist(),
                y=curve.recall.tolist(),
                mode="lines",
                line=dict(color=_COLOR_GREEN, width=2),
                name=f"Recall{c_name}",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=curve.thresholds.tolist(),
                y=curve.f1_score.tolist(),
                mode="lines",
                line=dict(color=_COLOR_PURPLE, width=3),
                name=f"F1 Score{c_name}",
            )
        )

        self._apply_theme(
            fig,
            title=f"Decision Threshold Analysis ({subset.capitalize()})",
            xaxis_title="Threshold",
            yaxis_title="Score",
            show_legend=True,
        )
        fig.update_xaxes(range=[0, 1])
        fig.update_yaxes(range=[0, 1.05])

        return fig

    def plot_residual_distribution(self, subset: str = "test") -> go.Figure:
        """Plot a histogram of residuals with an overlaid normal distribution curve.

        Helps verify if the model errors are symmetrically distributed
        and normally concentrated around zero.

        Parameters
        ----------
        subset : str, default="test"
            The data subset to plot (``"train"`` or ``"test"``).

        Returns
        -------
        go.Figure
            The Plotly figure object.
        """
        sd = getattr(self.results, f"{subset}_data", None)
        if sd is None:
            raise ValueError(f"No {subset} subset data available in results.")

        fig = go.Figure()

        residuals = getattr(sd, "residuals", sd.y_true - sd.y_pred)
        residuals_clean = residuals[np.isfinite(residuals)]

        if len(residuals_clean) == 0:
            raise ValueError(f"No finite residuals to plot for subset {subset}.")

        # Compute histogram bins in Python to apply colorscale dynamically
        counts, bin_edges = np.histogram(residuals_clean, bins="auto", density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        bar_colors = np.abs(bin_centers)
        vmax = (
            float(np.percentile(np.abs(residuals_clean), 95))
            if len(residuals_clean) > 0
            else 1.0
        )

        # Add histogram trace using go.Bar with gradient colorscale
        fig.add_trace(
            go.Bar(
                x=bin_centers.tolist(),
                y=counts.tolist(),
                width=[float(bin_edges[1] - bin_edges[0])]
                * len(bin_centers),  # Ensure bars touch like a histogram
                name="Residuals Density",
                marker=dict(
                    color=bar_colors.tolist(),
                    colorscale=_COLORSCALE_RESIDUALS,
                    cmin=0,
                    cmax=vmax,
                    line=dict(color=_COLOR_BG_PAPER, width=0.5),
                ),
                hovertemplate=(
                    "<b>Residual Range:</b> %{x:.4f}<br>"
                    "<b>Density:</b> %{y:.4f}<extra></extra>"
                ),
            )
        )

        # Compute normal curve
        mean_res = float(np.mean(residuals_clean))
        std_res = float(np.std(residuals_clean))

        if std_res > 0:
            x_curve = np.linspace(
                float(np.min(residuals_clean)), float(np.max(residuals_clean)), 200
            )
            y_curve = norm.pdf(x_curve, mean_res, std_res)

            fig.add_trace(
                go.Scatter(
                    x=x_curve.tolist(),
                    y=y_curve.tolist(),
                    mode="lines",
                    line=dict(color=_COLOR_AMBER, width=2.5),
                    name=f"Normal Fit (μ={mean_res:.2f}, σ={std_res:.2f})",
                    hovertemplate=(
                        "<b>Residual:</b> %{x:.4f}<br>"
                        "<b>Normal Density:</b> %{y:.4f}<extra></extra>"
                    ),
                )
            )

        self._apply_theme(
            fig,
            title=f"Residual Distribution ({subset.capitalize()})",
            xaxis_title="Residual (Actual - Predicted)",
            yaxis_title="Density",
            show_legend=True,
        )

        return fig

    def plot_residuals_vs_actual(self, subset: str = "test") -> go.Figure:
        """Plot Residuals vs. Actual target values.

        Helps visualize error behavior across the target's true range.
        Systematic trends suggest missing non-linear relationships.

        Parameters
        ----------
        subset : str, default="test"
            The data subset to plot (``"train"`` or ``"test"``).

        Returns
        -------
        go.Figure
            The Plotly figure object.
        """
        sd = getattr(self.results, f"{subset}_data", None)
        if sd is None:
            raise ValueError(f"No {subset} subset data available in results.")

        fig = go.Figure()

        y_true = sd.y_true.tolist()
        residuals = getattr(sd, "residuals", sd.y_true - sd.y_pred).tolist()
        abs_res = getattr(sd, "abs_residuals", np.abs(residuals)).tolist()

        # Zero baseline reference line
        fig.add_hline(
            y=0,
            line=dict(color=_COLOR_TEXT_MUTED, width=1.5, dash="dash"),
            annotation_text="Zero Residual",
            annotation_position="bottom right",
            annotation_font=dict(size=9, color=_COLOR_TEXT_MUTED),
        )

        # Plot observations
        vmax = float(np.percentile(abs_res, 95)) if len(abs_res) > 0 else 1.0
        fig.add_trace(
            go.Scatter(
                x=y_true,
                y=residuals,
                mode="markers",
                name="Residuals",
                marker=dict(
                    size=7,
                    color=abs_res,
                    colorscale=_COLORSCALE_RESIDUALS,
                    cmin=0,
                    cmax=vmax,
                    colorbar=self._compact_colorbar("|Error|"),
                    line=dict(color=_COLOR_BG_PAPER, width=0.5),
                ),
                hovertemplate=(
                    "<b>Actual Value:</b> %{x:.4f}<br>"
                    "<b>Residual:</b> %{y:.4f}<extra></extra>"
                ),
                customdata=sd.X_data.index.tolist(),
            )
        )

        # Compute on-the-fly LOWESS curve
        try:
            from statsmodels.nonparametric.smoothers_lowess import lowess as sm_lowess

            xs = np.asarray(y_true)
            ys = np.asarray(residuals)
            idx = np.argsort(xs)
            xs_s, ys_s = xs[idx], ys[idx]
            smooth = sm_lowess(ys_s, xs_s, frac=0.6667, return_sorted=True)

            fig.add_trace(
                go.Scatter(
                    x=smooth[:, 0].tolist(),
                    y=smooth[:, 1].tolist(),
                    mode="lines",
                    line=dict(color=_COLOR_PURPLE, width=2.5, shape="spline"),
                    name="LOWESS Trend",
                    hoverinfo="skip",
                )
            )
        except Exception as exc:
            import logging

            logging.getLogger(__name__).warning(
                "On-the-fly LOWESS curve failed: %s", exc
            )

        self._apply_theme(
            fig,
            title=f"Residuals vs. Actual Values ({subset.capitalize()})",
            xaxis_title="Actual Target Values",
            yaxis_title="Residual (Actual - Predicted)",
            show_legend=True,
        )

        return fig

    def plot_posterior_predictive(self, subset: str = "test") -> go.Figure:
        """Plot the Posterior Predictive Density comparison.

        Compares the density/KDE curves of the actual and predicted values
        to verify if the model captures the shape, modality, and spread
        of the true target variable.

        Parameters
        ----------
        subset : str, default="test"
            The data subset to plot (``"train"`` or ``"test"``).

        Returns
        -------
        go.Figure
            The Plotly figure object.
        """
        sd = getattr(self.results, f"{subset}_data", None)
        if sd is None:
            raise ValueError(f"No {subset} subset data available in results.")

        fig = go.Figure()

        y_true = sd.y_true[np.isfinite(sd.y_true)]
        y_pred = sd.y_pred[np.isfinite(sd.y_pred)]

        if len(y_true) < 2 or len(y_pred) < 2:
            raise ValueError("Insufficient finite data points to estimate density.")

        try:
            # Generate common evaluation grid
            min_val = min(float(np.min(y_true)), float(np.min(y_pred)))
            max_val = max(float(np.max(y_true)), float(np.max(y_pred)))
            pad = (max_val - min_val) * 0.15
            x_grid = np.linspace(min_val - pad, max_val + pad, 250)

            # Actual KDE
            kde_true = gaussian_kde(y_true)
            y_true_density = kde_true(x_grid)
            fig.add_trace(
                go.Scatter(
                    x=x_grid.tolist(),
                    y=y_true_density.tolist(),
                    mode="lines",
                    fill="tozeroy",
                    fillcolor="rgba(56, 189, 248, 0.15)",  # Sky Blue translucent
                    line=dict(color=_COLOR_BLUE, width=2.5),
                    name="Actual Values",
                    hovertemplate=(
                        "<b>Value:</b> %{x:.4f}<br>"
                        "<b>Density:</b> %{y:.4f}<extra></extra>"
                    ),
                )
            )

            # Predicted KDE
            kde_pred = gaussian_kde(y_pred)
            y_pred_density = kde_pred(x_grid)
            fig.add_trace(
                go.Scatter(
                    x=x_grid.tolist(),
                    y=y_pred_density.tolist(),
                    mode="lines",
                    fill="tozeroy",
                    fillcolor="rgba(245, 158, 11, 0.15)",  # Amber translucent
                    line=dict(color=_COLOR_AMBER, width=2.5),
                    name="Predicted Values",
                    hovertemplate=(
                        "<b>Value:</b> %{x:.4f}<br>"
                        "<b>Density:</b> %{y:.4f}<extra></extra>"
                    ),
                )
            )

        except Exception:
            # Fallback to overlay histograms
            fig.add_trace(
                go.Histogram(
                    x=y_true.tolist(),
                    histnorm="probability density",
                    name="Actual Values (Hist)",
                    opacity=0.5,
                    marker_color=_COLOR_BLUE,
                )
            )
            fig.add_trace(
                go.Histogram(
                    x=y_pred.tolist(),
                    histnorm="probability density",
                    name="Predicted Values (Hist)",
                    opacity=0.5,
                    marker_color=_COLOR_AMBER,
                )
            )

        self._apply_theme(
            fig,
            title=f"Posterior Predictive Density Comparison ({subset.capitalize()})",
            xaxis_title="Value",
            yaxis_title="Probability Density",
            show_legend=True,
        )

        return fig

    def plot_outliers(self) -> go.Figure:
        """Plot Outliers diagnostic.

        Displays the values of the top significant outlier-predicting feature
        grouped by standardized residual magnitude, highlighting anomalous samples.

        Returns
        -------
        go.Figure
            The Plotly figure object.
        """
        oa = self.results.outlier_analysis
        test_sd = self.results.test_data

        if test_sd is None or test_sd.std_residuals is None:
            raise ValueError("No test data available for outlier plotting.")

        fig = go.Figure()

        std_residuals = test_sd.std_residuals
        abs_std_residuals = np.abs(std_residuals)
        threshold = (
            oa.threshold
            if oa is not None
            else float(np.percentile(abs_std_residuals, 95))
        )

        # Check if outlier analysis found significant features
        top_feature = None
        if oa is not None and oa.results_df is not None and len(oa.results_df) > 0:
            top_feature = oa.results_df.iloc[0]["feature"]

        if top_feature is None:
            num_cols = test_sd.X_data.select_dtypes(include=[np.number]).columns
            if len(num_cols) > 0:
                top_feature = num_cols[0]

        if top_feature is None:
            raise ValueError("No numeric features available to analyze outliers.")

        feature_values = test_sd.X_data[top_feature].values
        outlier_mask = abs_std_residuals >= threshold

        # Plot observations with consistent |Error| colorscale and outlier sizing/borders
        vmax = (
            float(np.percentile(abs_std_residuals, 95))
            if len(abs_std_residuals) > 0
            else 1.0
        )

        # Build dynamic marker settings for outliers vs typical observations
        marker_sizes = np.where(outlier_mask, 9, 6).tolist()
        marker_line_widths = np.where(outlier_mask, 1.2, 0.5).tolist()
        marker_line_colors = np.where(
            outlier_mask, _COLOR_TEXT_MAIN, _COLOR_BG_PAPER
        ).tolist()

        fig.add_trace(
            go.Scatter(
                x=feature_values.tolist(),
                y=std_residuals.tolist(),
                mode="markers",
                name="Observations",
                marker=dict(
                    size=marker_sizes,
                    color=abs_std_residuals.tolist(),
                    colorscale=_COLORSCALE_RESIDUALS,
                    cmin=0,
                    cmax=vmax,
                    colorbar=self._compact_colorbar("|Error|"),
                    line=dict(
                        color=marker_line_colors,
                        width=marker_line_widths,
                    ),
                ),
                hovertemplate=(
                    f"<b>{top_feature}:</b> %{{x:.4f}}<br>"
                    f"<b>Std. Residual:</b> %{{y:.4f}}<br>"
                    f"<b>Absolute Error:</b> %{{marker.color:.4f}}<extra></extra>"
                ),
                customdata=test_sd.X_data.index.tolist(),
            )
        )

        # Overlays thresholds
        fig.add_hline(
            y=threshold,
            line=dict(color=_COLOR_RED, width=1.5, dash="dash"),
            name="Outlier Threshold",
        )
        fig.add_hline(y=-threshold, line=dict(color=_COLOR_RED, width=1.5, dash="dash"))
        fig.add_hline(y=0, line=dict(color=_COLOR_TEXT_MUTED, width=1))

        # LOWESS Trend Line if calculated for this feature
        if oa is not None and top_feature in oa.lowess_curves:
            self._add_lowess_trace(
                fig,
                oa.lowess_curves[top_feature],
                name="Outliers Trend",
                color=_COLOR_PURPLE,
            )

        self._apply_theme(
            fig,
            title=f"Outlier Feature Analysis: Standardized Residuals vs. {top_feature}",
            xaxis_title=f"{top_feature} Values",
            yaxis_title="Standardized Residuals",
            show_legend=True,
        )

        return fig

    def plot_metrics(self, subset: str = "test") -> go.Figure:
        """Plot a highly stylized Summary Metrics Card.

        Renders scalar model metrics as a publication-grade graphical card.

        Parameters
        ----------
        subset : str, default="test"
            The data subset to plot (``"train"`` or ``"test"``).

        Returns
        -------
        go.Figure
            The Plotly figure object.
        """
        fig = go.Figure()

        if self.results.problem_type == "regression":
            m = getattr(self.results, f"{subset}_metrics", None)
            if m is None:
                raise ValueError(
                    f"No regression metrics calculated for {subset} subset."
                )

            headers = ["Evaluation Metric", "Calculated Value"]
            cells = [
                [
                    "R² Coefficient",
                    "MAE (Mean Absolute Error)",
                    "MSE (Mean Squared Error)",
                    "RMSE (Root Mean Squared Error)",
                    "MAPE (Percentage Error)",
                    "Observations Count",
                ],
                [
                    f"{m.r2:.4f}",
                    f"{m.mae:.4f}",
                    f"{m.mse:.4f}",
                    f"{m.rmse:.4f}",
                    f"{m.mape:.2f}%",
                    f"{m.n_samples}",
                ],
            ]
            title = f"Regression Performance Summary ({subset.capitalize()})"
        else:
            m = getattr(self.results, f"{subset}_clf_metrics", None)
            if m is None:
                raise ValueError(
                    f"No classification metrics calculated for {subset} subset."
                )

            headers = ["Evaluation Metric", "Calculated Value"]
            metric_names = [
                "Accuracy Score",
                "Precision (Macro/Binary)",
                "Recall (Macro/Binary)",
                "F1-Score (Macro/Binary)",
                "Matthews Corrcoef (MCC)",
            ]
            metric_vals = [
                f"{m.accuracy * 100:.2f}%",
                f"{m.precision * 100:.2f}%",
                f"{m.recall * 100:.2f}%",
                f"{m.f1_score * 100:.2f}%",
                f"{m.mcc:.4f}",
            ]

            if getattr(m, "roc_auc", None) is not None:
                metric_names.append("ROC-AUC")
                metric_vals.append(f"{m.roc_auc:.4f}")
            if getattr(m, "pr_auc", None) is not None:
                metric_names.append("PR-AUC (Average Precision)")
                metric_vals.append(f"{m.pr_auc:.4f}")
            if getattr(m, "brier_score", None) is not None:
                metric_names.append("Brier Score Loss")
                metric_vals.append(f"{m.brier_score:.4f}")
            if getattr(m, "log_loss", None) is not None:
                metric_names.append("Log-Loss (Cross-Entropy)")
                metric_vals.append(f"{m.log_loss:.4f}")

            cells = [metric_names, metric_vals]
            title = f"Classification Performance Summary ({subset.capitalize()})"

        fig.add_trace(
            go.Table(
                header=dict(
                    values=[f"<b>{h}</b>" for h in headers],
                    fill_color="#f1f5f9",  # Soft slate header background
                    align="left",
                    font=dict(color=_COLOR_TEXT_MAIN, size=13, family=_FONT_FAMILY),
                    line_color=_COLOR_BORDER,
                    height=32,
                ),
                cells=dict(
                    values=cells,
                    fill_color="#ffffff",  # Pure white background
                    align="left",
                    font=dict(color=_COLOR_TEXT_MUTED, size=12, family=_FONT_FAMILY),
                    line_color=_COLOR_BORDER,
                    height=28,
                ),
            )
        )

        self._apply_theme(
            fig,
            title=title,
            xaxis_title="",
            yaxis_title="",
            show_legend=False,
        )

        # Clean layout of margins for tables
        fig.update_layout(margin=dict(l=40, r=40, t=65, b=40))

        return fig

    def plot_class_distribution(self, subset: str = "test") -> go.Figure:
        """Plot the Actual vs. Predicted Class Distributions.

        Displays class proportions as a grouped bar chart to immediately
        highlight prediction biases and class imbalances.

        Parameters
        ----------
        subset : str, default="test"
            The data subset to plot (``"train"`` or ``"test"``).

        Returns
        -------
        go.Figure
            The Plotly figure object.
        """
        sd = getattr(self.results, f"{subset}_data", None)
        if sd is None:
            raise ValueError(f"No {subset} subset data available in results.")

        fig = go.Figure()

        y_true = np.asarray(sd.y_true)
        y_pred = np.asarray(sd.y_pred)

        classes, counts_true = np.unique(y_true, return_counts=True)
        pred_classes, counts_pred = np.unique(y_pred, return_counts=True)

        # Standardize matching class counts
        all_classes = np.unique(np.concatenate((classes, pred_classes)))
        counts_true_map = dict(zip(classes, counts_true, strict=False))
        counts_pred_map = dict(zip(pred_classes, counts_pred, strict=False))

        final_counts_true = [counts_true_map.get(c, 0) for c in all_classes]
        final_counts_pred = [counts_pred_map.get(c, 0) for c in all_classes]
        class_labels = [str(c) for c in all_classes]

        fig.add_trace(
            go.Bar(
                x=class_labels,
                y=final_counts_true,
                name="Actual Class",
                marker_color=_COLOR_BLUE,
                marker_line_color=_COLOR_BG_PAPER,
                marker_line_width=1,
                hovertemplate=(
                    "<b>Class:</b> %{x}<br>" "<b>Actual Count:</b> %{y}<extra></extra>"
                ),
            )
        )

        fig.add_trace(
            go.Bar(
                x=class_labels,
                y=final_counts_pred,
                name="Predicted Class",
                marker_color=_COLOR_AMBER,
                marker_line_color=_COLOR_BG_PAPER,
                marker_line_width=1,
                hovertemplate=(
                    "<b>Class:</b> %{x}<br>"
                    "<b>Predicted Count:</b> %{y}<extra></extra>"
                ),
            )
        )

        self._apply_theme(
            fig,
            title=f"Class Distribution Comparison ({subset.capitalize()})",
            xaxis_title="Target Class Labels",
            yaxis_title="Instance Count",
            show_legend=True,
        )

        fig.update_layout(barmode="group")

        return fig

    def plot_probability_distribution(self, subset: str = "test") -> go.Figure:
        """Plot the distribution of predicted probabilities.

        Categorizes samples into Correctly Predicted vs. Misclassified, showing
        the confidence distribution (winning class probability). Helps visualize
        model calibration and uncertainty.

        Parameters
        ----------
        subset : str, default="test"
            The data subset to plot (``"train"`` or ``"test"``).

        Returns
        -------
        go.Figure
            The Plotly figure object.
        """
        sd = getattr(self.results, f"{subset}_data", None)
        if sd is None:
            raise ValueError(f"No {subset} subset data available in results.")

        if sd.y_prob is None:
            raise ValueError(
                f"No prediction probabilities available for {subset} subset."
            )

        fig = go.Figure()

        y_true = np.asarray(sd.y_true)
        y_pred = np.asarray(sd.y_pred)
        y_prob = np.asarray(sd.y_prob)

        # Get the probability of the predicted winning class (confidence)
        classes = np.unique(y_true)
        class_to_idx = {c: idx for idx, c in enumerate(classes)}

        confidences = []
        for i, pred_class in enumerate(y_pred):
            try:
                idx = class_to_idx[pred_class]
                confidences.append(float(y_prob[i, idx]))
            except Exception:
                confidences.append(float(np.max(y_prob[i])))

        conf_arr = np.asarray(confidences)
        correct_mask = y_true == y_pred

        correct_conf = conf_arr[correct_mask]
        wrong_conf = conf_arr[~correct_mask]

        try:
            # Generate KDE evaluation grid (clamped between 0 and 1)
            x_grid = np.linspace(0.4, 1.0, 150)

            # Correct predictions KDE
            if len(correct_conf) > 1:
                kde_correct = gaussian_kde(correct_conf)
                fig.add_trace(
                    go.Scatter(
                        x=x_grid.tolist(),
                        y=kde_correct(x_grid).tolist(),
                        mode="lines",
                        fill="tozeroy",
                        fillcolor="rgba(16, 185, 129, 0.15)",  # Translucent Emerald Green
                        line=dict(color=_COLOR_GREEN, width=2.5),
                        name="Correct Predictions",
                        hovertemplate=(
                            "<b>Confidence:</b> %{x:.2f}%<br>"
                            "<b>Density:</b> %{y:.4f}<extra></extra>"
                        ),
                    )
                )

            # Incorrect predictions KDE
            if len(wrong_conf) > 1:
                kde_wrong = gaussian_kde(wrong_conf)
                fig.add_trace(
                    go.Scatter(
                        x=x_grid.tolist(),
                        y=kde_wrong(x_grid).tolist(),
                        mode="lines",
                        fill="tozeroy",
                        fillcolor="rgba(248, 113, 113, 0.15)",  # Translucent Red
                        line=dict(color=_COLOR_RED, width=2.5),
                        name="Misclassified Predictions",
                        hovertemplate=(
                            "<b>Confidence:</b> %{x:.2f}%<br>"
                            "<b>Density:</b> %{y:.4f}<extra></extra>"
                        ),
                    )
                )

        except Exception:
            # Fallback to histogram
            fig.add_trace(
                go.Histogram(
                    x=correct_conf.tolist(),
                    name="Correct (Hist)",
                    opacity=0.6,
                    marker_color=_COLOR_GREEN,
                )
            )
            fig.add_trace(
                go.Histogram(
                    x=wrong_conf.tolist(),
                    name="Incorrect (Hist)",
                    opacity=0.6,
                    marker_color=_COLOR_RED,
                )
            )

        self._apply_theme(
            fig,
            title=f"Prediction Confidence Distribution ({subset.capitalize()})",
            xaxis_title="Predicted Class Winning Probability",
            yaxis_title="Probability Density",
            show_legend=True,
        )
        fig.update_xaxes(range=[0.38, 1.02])

        return fig

    def plot_misclassification_features(self, subset: str = "test") -> go.Figure:
        """Plot the Misclassification Feature diagnostic.

        Highlights the feature density boundaries of the top significant feature
        that separates Correctly Predicted vs. Misclassified samples.

        Parameters
        ----------
        subset : str, default="test"
            The data subset to plot (``"train"`` or ``"test"``).

        Returns
        -------
        go.Figure
            The Plotly figure object.
        """
        mc = getattr(self.results, f"{subset}_misclassification", None)
        sd = getattr(self.results, f"{subset}_data", None)

        if sd is None:
            raise ValueError(f"No {subset} subset data available in results.")

        fig = go.Figure()

        y_true = np.asarray(sd.y_true)
        y_pred = np.asarray(sd.y_pred)
        correct_mask = y_true == y_pred

        # Pick top significant feature from misclassification result
        top_feature = None
        if mc is not None and mc.results_df is not None and len(mc.results_df) > 0:
            top_feature = mc.results_df.iloc[0]["feature"]

        if top_feature is None:
            num_cols = sd.X_data.select_dtypes(include=[np.number]).columns
            if len(num_cols) > 0:
                top_feature = num_cols[0]

        if top_feature is None:
            raise ValueError(
                "No numeric features available to evaluate misclassifications."
            )

        feature_values = sd.X_data[top_feature].values
        correct_vals = feature_values[correct_mask]
        wrong_vals = feature_values[~correct_mask]

        correct_vals_clean = correct_vals[np.isfinite(correct_vals)]
        wrong_vals_clean = wrong_vals[np.isfinite(wrong_vals)]

        try:
            # Generate common evaluation grid
            min_val = min(
                float(np.min(correct_vals_clean)), float(np.min(wrong_vals_clean))
            )
            max_val = max(
                float(np.max(correct_vals_clean)), float(np.max(wrong_vals_clean))
            )
            pad = (max_val - min_val) * 0.1
            x_grid = np.linspace(min_val - pad, max_val + pad, 200)

            # Correctly predicted density
            if len(correct_vals_clean) > 1:
                kde_correct = gaussian_kde(correct_vals_clean)
                fig.add_trace(
                    go.Scatter(
                        x=x_grid.tolist(),
                        y=kde_correct(x_grid).tolist(),
                        mode="lines",
                        fill="tozeroy",
                        fillcolor="rgba(16, 185, 129, 0.15)",
                        line=dict(color=_COLOR_GREEN, width=2.5),
                        name="Correct Predictions",
                        hovertemplate=(
                            "<b>Feature Value:</b> %{x:.4f}<br>"
                            "<b>Density:</b> %{y:.4f}<extra></extra>"
                        ),
                    )
                )

            # Misclassified density
            if len(wrong_vals_clean) > 1:
                kde_wrong = gaussian_kde(wrong_vals_clean)
                fig.add_trace(
                    go.Scatter(
                        x=x_grid.tolist(),
                        y=kde_wrong(x_grid).tolist(),
                        mode="lines",
                        fill="tozeroy",
                        fillcolor="rgba(248, 113, 113, 0.15)",
                        line=dict(color=_COLOR_RED, width=2.5),
                        name="Misclassified Predictions",
                        hovertemplate=(
                            "<b>Feature Value:</b> %{x:.4f}<br>"
                            "<b>Density:</b> %{y:.4f}<extra></extra>"
                        ),
                    )
                )

        except Exception:
            # Fallback to histogram
            fig.add_trace(
                go.Histogram(
                    x=correct_vals_clean.tolist(),
                    name="Correct (Hist)",
                    opacity=0.6,
                    marker_color=_COLOR_GREEN,
                )
            )
            fig.add_trace(
                go.Histogram(
                    x=wrong_vals_clean.tolist(),
                    name="Incorrect (Hist)",
                    opacity=0.6,
                    marker_color=_COLOR_RED,
                )
            )

        self._apply_theme(
            fig,
            title=f"Misclassification Boundary: Density of {top_feature} ({subset.capitalize()})",
            xaxis_title=f"{top_feature} Values",
            yaxis_title="Probability Density",
            show_legend=True,
        )

        return fig

    # --- Graph Aliases to exactly match user endpoints ---

    def plot_act_vs_pred(self, subset: str = "test") -> go.Figure:
        """Alias for plot_actual_vs_predicted.

        Parameters
        ----------
        subset : str, default="test"
            The data subset to plot (``"train"`` or ``"test"``).

        Returns
        -------
        go.Figure
            The Plotly figure object.
        """
        return self.plot_actual_vs_predicted(subset)

    def plot_res_dist(self, subset: str = "test") -> go.Figure:
        """Alias for plot_residual_distribution.

        Parameters
        ----------
        subset : str, default="test"
            The data subset to plot (``"train"`` or ``"test"``).

        Returns
        -------
        go.Figure
            The Plotly figure object.
        """
        return self.plot_residual_distribution(subset)

    def plot_res_vs_pred(self, subset: str = "test") -> go.Figure:
        """Alias for plot_residuals.

        Parameters
        ----------
        subset : str, default="test"
            The data subset to plot (``"train"`` or ``"test"``).

        Returns
        -------
        go.Figure
            The Plotly figure object.
        """
        return self.plot_residuals(subset)

    def plot_res_vs_act(self, subset: str = "test") -> go.Figure:
        """Alias for plot_residuals_vs_actual.

        Parameters
        ----------
        subset : str, default="test"
            The data subset to plot (``"train"`` or ``"test"``).

        Returns
        -------
        go.Figure
            The Plotly figure object.
        """
        return self.plot_residuals_vs_actual(subset)

    def plot_scale_loc(self, subset: str = "test") -> go.Figure:
        """Alias for plot_scale_location.

        Parameters
        ----------
        subset : str, default="test"
            The data subset to plot (``"train"`` or ``"test"``).

        Returns
        -------
        go.Figure
            The Plotly figure object.
        """
        return self.plot_scale_location(subset)

    def plot_post_pred(self, subset: str = "test") -> go.Figure:
        """Alias for plot_posterior_predictive.

        Parameters
        ----------
        subset : str, default="test"
            The data subset to plot (``"train"`` or ``"test"``).

        Returns
        -------
        go.Figure
            The Plotly figure object.
        """
        return self.plot_posterior_predictive(subset)

    def plot_prob_dist(self, subset: str = "test") -> go.Figure:
        """Alias for plot_probability_distribution.

        Parameters
        ----------
        subset : str, default="test"
            The data subset to plot (``"train"`` or ``"test"``).

        Returns
        -------
        go.Figure
            The Plotly figure object.
        """
        return self.plot_probability_distribution(subset)

    def plot_misclass(self, subset: str = "test") -> go.Figure:
        """Alias for plot_misclassification_features.

        Parameters
        ----------
        subset : str, default="test"
            The data subset to plot (``"train"`` or ``"test"``).

        Returns
        -------
        go.Figure
            The Plotly figure object.
        """
        return self.plot_misclassification_features(subset)

    # ------------------------------------------------------------------
    # Zero-friction Serialization Helpers
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # SHAP Plotting Methods
    # ------------------------------------------------------------------

    def plot_shap_summary(self, subset: str = "test") -> go.Figure:
        """Plot the SHAP summary (global feature importance) as a bar chart.

        Parameters
        ----------
        subset : str, default="test"
            The subset to evaluate.

        Returns
        -------
        go.Figure
            The Plotly figure.
        """
        shap_data = getattr(self.results, f"{subset}_shap", None)
        if shap_data is None:
            raise ValueError(f"No SHAP data found for subset '{subset}'.")

        mean_abs_shap = shap_data.mean_abs_shap
        feature_names = shap_data.feature_names

        sorted_indices = np.argsort(mean_abs_shap)
        top_indices = sorted_indices[-10:]

        x = mean_abs_shap[top_indices]
        y = [feature_names[i] for i in top_indices]

        fig = go.Figure(
            data=go.Bar(
                x=x,
                y=y,
                orientation='h',
                marker=dict(color=_COLOR_BLUE),
                hovertemplate="<b>%{y}</b><br>Mean |SHAP|: %{x:.4f}<extra></extra>"
            )
        )

        self._apply_theme(
            fig,
            title=f"Global Feature Importance (SHAP) ({subset.title()})",
            xaxis_title="Mean |SHAP value| (average impact on model output)",
            yaxis_title=""
        )
        fig.update_layout(showlegend=False)
        return fig

    def plot_shap_beeswarm(self, subset: str = "test") -> go.Figure:
        """Plot the SHAP beeswarm chart.

        Parameters
        ----------
        subset : str, default="test"
            The subset to evaluate.

        Returns
        -------
        go.Figure
            The Plotly figure.
        """
        shap_data = getattr(self.results, f"{subset}_shap", None)
        if shap_data is None:
            raise ValueError(f"No SHAP data found for subset '{subset}'.")

        mean_abs_shap = shap_data.mean_abs_shap
        feature_names = shap_data.feature_names
        shap_values = shap_data.shap_values
        feature_values = (
            shap_data.feature_values if shap_data.feature_values is not None else shap_values
        )

        sorted_indices = np.argsort(mean_abs_shap)[-10:]

        fig = go.Figure()

        n_samples = shap_values.shape[0]
        np.random.seed(42)
        jitter = (np.random.rand(n_samples) - 0.5) * 0.35

        tickvals = []
        ticktext = []

        for y_pos, f_idx in enumerate(sorted_indices):
            f_name = feature_names[f_idx]
            sv = shap_values[:, f_idx]
            fv = feature_values[:, f_idx]

            f_min = np.min(fv)
            f_max = np.max(fv)
            f_range = f_max - f_min if f_max > f_min else 1.0

            cv = (fv - f_min) / f_range

            y_vals = y_pos + jitter

            marker = dict(
                size=5,
                color=cv,
                colorscale=[[0, '#008bfb'], [1, '#ff0052']],
                cmin=0,
                cmax=1,
                line=dict(width=0),
                showscale=(y_pos == 0),
            )

            if y_pos == 0:
                marker['colorbar'] = dict(
                    title="Feature value",
                    tickvals=[0, 1],
                    ticktext=["Low", "High"],
                    thickness=10,
                    len=0.5,
                    outlinewidth=0,
                )

            hover_text = [
                f"Row {shap_data.eval_index[i] if shap_data.eval_index else i}<br>"
                f"{f_name}: SHAP={sv[i]:.4f}, Val={fv[i]:.4f}"
                for i in range(n_samples)
            ]

            fig.add_trace(go.Scatter(
                x=sv,
                y=y_vals,
                mode='markers',
                marker=marker,
                text=hover_text,
                hovertemplate="%{text}<extra></extra>",
                showlegend=False
            ))

            tickvals.append(y_pos)
            ticktext.append(f_name)

        self._apply_theme(
            fig,
            title=f"SHAP Beeswarm Plot ({subset.title()})",
            xaxis_title="SHAP value (impact on model output)",
            yaxis_title=""
        )

        fig.update_yaxes(
            tickvals=tickvals,
            ticktext=ticktext,
            showgrid=False,
            zeroline=False
        )

        return fig

    def to_json_bundle(self, subset: str = "test") -> str:
        """Serialize a pre-packaged bundle of all available diagnostic charts.

        Perfect for sending over HTTP or rendering directly in drag-and-drop
        dashboards. Uses the exact 18 standard endpoints as dictionary keys.

        Parameters
        ----------
        subset : str, default="test"
            The subset of charts to compile (``"train"`` or ``"test"``).

        Returns
        -------
        str
            A JSON string containing the complete mapped dictionary of figures.
        """
        bundle: Dict[str, Any] = {}

        if self.results.problem_type == "regression":
            # 1. act_vs_pred
            try:
                bundle["act_vs_pred"] = json.loads(
                    self.plot_actual_vs_predicted(subset).to_json()
                )
            except Exception as e:
                bundle["act_vs_pred"] = {"error": str(e)}

            # 2. res_dist
            try:
                bundle["res_dist"] = json.loads(
                    self.plot_residual_distribution(subset).to_json()
                )
            except Exception as e:
                bundle["res_dist"] = {"error": str(e)}

            # 3. res_vs_pred
            try:
                bundle["res_vs_pred"] = json.loads(
                    self.plot_residuals(subset).to_json()
                )
            except Exception as e:
                bundle["res_vs_pred"] = {"error": str(e)}

            # 4. res_vs_act
            try:
                bundle["res_vs_act"] = json.loads(
                    self.plot_residuals_vs_actual(subset).to_json()
                )
            except Exception as e:
                bundle["res_vs_act"] = {"error": str(e)}

            # 5. qq
            try:
                bundle["qq"] = json.loads(self.plot_qq(subset).to_json())
            except Exception as e:
                bundle["qq"] = {"error": str(e)}

            # 6. scale_loc
            try:
                bundle["scale_loc"] = json.loads(
                    self.plot_scale_location(subset).to_json()
                )
            except Exception as e:
                bundle["scale_loc"] = {"error": str(e)}

            # 7. post_pred
            try:
                bundle["post_pred"] = json.loads(
                    self.plot_posterior_predictive(subset).to_json()
                )
            except Exception as e:
                bundle["post_pred"] = {"error": str(e)}

            # 8. leverage
            if subset == "train":
                try:
                    bundle["leverage"] = json.loads(self.plot_leverage().to_json())
                except Exception as e:
                    bundle["leverage"] = {"error": str(e)}

            # 11. outliers
            if subset == "test":
                try:
                    bundle["outliers"] = json.loads(self.plot_outliers().to_json())
                except Exception as e:
                    bundle["outliers"] = {"error": str(e)}

            # 12. metrics (Regression)
            try:
                bundle["metrics"] = json.loads(self.plot_metrics(subset).to_json())
            except Exception as e:
                bundle["metrics"] = {"error": str(e)}

        elif self.results.problem_type == "classification":
            # 1. metrics (Classification)
            try:
                bundle["metrics"] = json.loads(self.plot_metrics(subset).to_json())
            except Exception as e:
                bundle["metrics"] = {"error": str(e)}

            # 2. cm
            try:
                bundle["cm"] = json.loads(self.plot_confusion_matrix(subset).to_json())
            except Exception as e:
                bundle["cm"] = {"error": str(e)}

            # 3. class_dist
            try:
                bundle["class_dist"] = json.loads(
                    self.plot_class_distribution(subset).to_json()
                )
            except Exception as e:
                bundle["class_dist"] = {"error": str(e)}

            # 4. prob_dist
            try:
                bundle["prob_dist"] = json.loads(
                    self.plot_probability_distribution(subset).to_json()
                )
            except Exception as e:
                bundle["prob_dist"] = {"error": str(e)}

            # 5. roc
            try:
                bundle["roc"] = json.loads(self.plot_roc_curve(subset).to_json())
            except Exception as e:
                bundle["roc"] = {"error": str(e)}

            # 6. pr
            try:
                bundle["pr"] = json.loads(self.plot_pr_curve(subset).to_json())
            except Exception as e:
                bundle["pr"] = {"error": str(e)}

            # 7. misclass
            try:
                bundle["misclass"] = json.loads(
                    self.plot_misclassification_features(subset).to_json()
                )
            except Exception as e:
                bundle["misclass"] = {"error": str(e)}

            # 8. calibration
            try:
                bundle["calibration"] = json.loads(
                    self.plot_calibration_curve(subset).to_json()
                )
            except Exception as e:
                bundle["calibration"] = {"error": str(e)}

            # 9. threshold
            try:
                bundle["threshold"] = json.loads(
                    self.plot_threshold_analysis(subset).to_json()
                )
            except Exception as e:
                bundle["threshold"] = {"error": str(e)}

        # Add SHAP raw data to the bundle
        shap_data = getattr(self.results, f"{subset}_shap", None)
        if shap_data is not None:
            try:
                bundle["shap_raw"] = {
                    "feature_names": shap_data.feature_names,
                    "base_value": shap_data.base_value,
                    "shap_values": shap_data.shap_values.tolist(),
                    "mean_abs_shap": shap_data.mean_abs_shap.tolist(),
                    "eval_index": shap_data.eval_index,
                    "feature_values": (
                        shap_data.feature_values.tolist()
                        if shap_data.feature_values is not None
                        else None
                    ),
                }
            except Exception as e:
                bundle["shap_raw"] = {"error": str(e)}

            # Placeholders for frontend JS rendering
            if "error" not in bundle.get("shap_raw", {}):
                bundle["shap_summary"] = {"data": [], "layout": {}}
                bundle["shap_beeswarm"] = {"data": [], "layout": {}}

        return json.dumps(bundle)

    def _get_subset_table(self, subset: str) -> Optional[Dict[str, Any]]:
        """Extract and format subset data to split dictionary format for efficient web transfer.

        Parameters
        ----------
        subset : str
            The subset to extract ("train" or "test").

        Returns
        -------
        dict or None
            A split dictionary containing columns, index, and data, or None.
        """
        sd = getattr(self.results, f"{subset}_data", None)
        if sd is None:
            return None

        # Build consolidated dataframe
        df = sd.X_data.copy()

        # Add actual and predicted columns with special identifiers
        df["__actual__"] = sd.y_true
        df["__predicted__"] = sd.y_pred

        if self.results.problem_type == "regression":
            residuals = sd.y_true - sd.y_pred
            df["__residual__"] = residuals
            df["__abs_error__"] = np.abs(residuals)
        else:
            df["__correct__"] = (sd.y_true == sd.y_pred).astype(bool)

        # Cap the table to 10,000 rows as requested
        cap_df = df.head(10000)

        # Ensure NaNs and infs are converted to None (json null)
        cap_df = cap_df.replace([np.inf, -np.inf], np.nan).where(
            pd.notnull(cap_df), None
        )

        # Convert to split orientation
        table_dict = cap_df.to_dict(orient="split")
        return table_dict

    def to_json_full_bundle(self) -> str:
        """Serialize a full diagnostics bundle containing train, test, and split tables.

        Returns
        -------
        str
            A JSON string containing 'problem_type', 'test', 'train', and 'tables'.
        """
        tables = {}
        for subset in ["test", "train"]:
            tbl = self._get_subset_table(subset)
            if tbl is not None:
                tables[subset] = tbl

        return json.dumps(
            {
                "problem_type": self.results.problem_type,
                "test": json.loads(self.to_json_bundle(subset="test")),
                "train": json.loads(self.to_json_bundle(subset="train")),
                "tables": tables,
            }
        )

    def save_dashboard_bundle(
        self, filepath: str = "dashboard/active_bundle.json"
    ) -> None:
        """Save the full train/test diagnostic bundle directly to a file.

        (defaulting to the dashboard's active bundle path).

        Parameters
        ----------
        filepath : str, default="dashboard/active_bundle.json"
            The filepath to write the bundle JSON to.
        """
        import os

        # Ensure parent directory exists
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(self.to_json_full_bundle())
