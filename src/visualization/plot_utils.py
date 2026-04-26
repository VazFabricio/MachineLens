from typing import Any, Dict, List, Optional

import numpy as np
import plotly.graph_objects as go

from .style import (
    _BORDER,
    _COLORBAR_STYLE,
    _COLORSCALE_REG,
    _FONT_COLOR,
    _FONT_FAMILY,
    _LAYOUT_BASE,
    _add_lowess_with_ci,
    _get_sequential_color,
    _layout,
    _title_dict,
)


def create_metrics_table(names: List[str], values: List[str], title: str) -> go.Figure:
    """
    Create a styled Plotly table for displaying evaluation metrics.

    Parameters
    ----------
    names : List[str]
        A list containing the display names of the metrics.
    values : List[str]
        A list containing the formatted string values corresponding to each metric.
    title : str
        The title text to display above the table.

    Returns
    -------
    plotly.graph_objects.Figure
        The generated Plotly table figure.
    """
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
        "title": _title_dict(title),
        "margin": dict(l=30, r=30, t=55, b=20),
    }
    fig.update_layout(**layout_kwargs)
    return fig


def create_residual_scatter_plot(
    x: np.ndarray,
    y: np.ndarray,
    abs_residuals: np.ndarray,
    title: str,
    xaxis_title: str,
    yaxis_title: str,
    add_identity_line: bool = False,
    add_zero_line: bool = False,
    lowess: Optional[Dict[str, Any]] = None,
    hover_template: Optional[str] = None,
    marker_size: int = 6,
    opacity: float = 0.85,
    line_width: float = 0.3,
) -> go.Figure:
    """
    Create a standardized scatter plot with colors mapped to absolute residuals.

    This function is highly versatile and can be used to generate 'Actual vs. Predicted',
    'Residuals vs. Predicted', or 'Residuals vs. Actual' plots depending on the provided inputs.

    Parameters
    ----------
    x : np.ndarray
        The data array for the x-axis.
    y : np.ndarray
        The data array for the y-axis.
    abs_residuals : np.ndarray
        Array of absolute residuals used to map the color intensity of the scatter points.
    title : str
        The title text for the plot.
    xaxis_title : str
        The label for the x-axis.
    yaxis_title : str
        The label for the y-axis.
    add_identity_line : bool, optional
        If True, adds a diagonal identity reference line. Defaults to False.
    add_zero_line : bool, optional
        If True, adds a horizontal dashed reference line at y=0. Defaults to False.
    lowess : Optional[Dict[str, Any]], optional
        A dictionary with LOWESS smoothing data ('x_smooth', 'y_smooth', 'ci_lower', 'ci_upper')
        to overlay on the scatter plot. Defaults to None.
    hover_template : Optional[str], optional
        A formatted Plotly hover template string. Defaults to None.
    marker_size : int, optional
        The size of the scatter plot markers. Defaults to 6.
    opacity : float, optional
        The opacity of the scatter plot markers. Defaults to 0.85.
    line_width : float, optional
        The outline width for the scatter plot markers. Defaults to 0.3.

    Returns
    -------
    plotly.graph_objects.Figure
        The generated Plotly scatter figure.
    """
    vmax = float(np.percentile(abs_residuals, 97))
    marker_color = np.clip(abs_residuals, 0, vmax)

    fig = go.Figure()

    if add_identity_line:
        min_val = float(np.min(x))
        max_val = float(np.max(x))
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode="lines",
                line=dict(color="rgba(0,0,0,0.25)", dash="dash", width=1.5),
                name="Ideal fit",
                hoverinfo="skip",
            )
        )

    if add_zero_line:
        fig.add_hline(
            y=0,
            line=dict(color="rgba(0,0,0,0.25)", dash="dash", width=1.5),
        )

    scatter_kwargs = dict(
        x=x.tolist(),
        y=y.tolist(),
        mode="markers",
        name="Observations" if add_identity_line else "Residuals",
        marker=dict(
            color=marker_color.tolist(),
            colorscale=_COLORSCALE_REG,
            cmin=0,
            cmax=vmax,
            size=marker_size,
            opacity=opacity,
            colorbar=dict(title="|Residual|", **_COLORBAR_STYLE),
            line=dict(width=line_width, color="rgba(0,0,0,0.1)"),
        ),
    )
    if hover_template:
        scatter_kwargs["hovertemplate"] = hover_template

    fig.add_trace(go.Scatter(**scatter_kwargs))

    if lowess:
        _add_lowess_with_ci(fig, lowess)

    fig.update_layout(
        **_layout(
            title=title,
            xaxis=dict(title=xaxis_title),
            yaxis=dict(title=yaxis_title),
        )
    )
    return fig


def create_qq_plot(
    qq_osm: np.ndarray,
    qq_osr: np.ndarray,
    qq_slope: float,
    qq_intercept: float,
    qq_ci_lower: Optional[np.ndarray],
    qq_ci_upper: Optional[np.ndarray],
    title: str,
) -> go.Figure:
    """
    Create a Quantile-Quantile (Q-Q) plot with an optional 95% confidence interval envelope.

    Parameters
    ----------
    qq_osm : np.ndarray
        Theoretical quantiles from the standard normal distribution.
    qq_osr : np.ndarray
        Sample quantiles derived from the standardized residuals.
    qq_slope : float
        The slope of the theoretical reference line.
    qq_intercept : float
        The intercept of the theoretical reference line.
    qq_ci_lower : Optional[np.ndarray]
        Array of values representing the lower bound of the 95% confidence interval.
    qq_ci_upper : Optional[np.ndarray]
        Array of values representing the upper bound of the 95% confidence interval.
    title : str
        The title text for the Q-Q plot.

    Returns
    -------
    plotly.graph_objects.Figure
        The generated Plotly Q-Q plot figure.
    """
    ref_line = qq_intercept + qq_slope * qq_osm
    dist = np.abs(qq_osr - ref_line)
    max_dist = float(np.max(dist)) if np.max(dist) != 0 else 1.0
    norm_dist = np.clip(dist / max_dist, 0.0, 1.0)

    point_colors = [_get_sequential_color(v) for v in norm_dist.tolist()]

    fig = go.Figure()

    # 95% CI envelope
    if qq_ci_lower is not None and qq_ci_upper is not None:
        ci_lower_line = qq_intercept + qq_slope * qq_ci_lower
        ci_upper_line = qq_intercept + qq_slope * qq_ci_upper
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([qq_osm, qq_osm[::-1]]).tolist(),
                y=np.concatenate([ci_upper_line, ci_lower_line[::-1]]).tolist(),
                fill="toself",
                fillcolor="rgba(59,130,246,0.10)",
                line=dict(width=0),
                showlegend=True,
                name="95% CI",
                hoverinfo="skip",
            )
        )

    # Reference line
    fig.add_trace(
        go.Scatter(
            x=qq_osm.tolist(),
            y=ref_line.tolist(),
            mode="lines",
            name="Theoretical",
            line=dict(color="rgba(0,0,0,0.25)", dash="dash", width=1.5),
            hoverinfo="skip",
        )
    )

    # Q-Q scatter
    fig.add_trace(
        go.Scatter(
            x=qq_osm.tolist(),
            y=qq_osr.tolist(),
            mode="markers",
            name="Sample quantiles",
            marker=dict(
                color=point_colors,
                size=6,
                opacity=0.9,
                line=dict(width=0.3, color="rgba(0,0,0,0.1)"),
            ),
            hovertemplate=(
                "Theoretical: <b>%{x:.4g}</b><br>"
                "Sample: <b>%{y:.4g}</b><extra></extra>"
            ),
        )
    )

    fig.update_layout(
        **_layout(
            title=title,
            xaxis=dict(title="Theoretical Quantiles"),
            yaxis=dict(title="Sample Quantiles"),
        )
    )
    return fig
