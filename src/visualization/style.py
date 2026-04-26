from typing import Any, Dict

import numpy as np
import plotly.graph_objects as go

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
_ACCENT_AMBER = "#f59e0b"

_COLORSCALE_CLASS = "Blues"
_COLORSCALE_REG = [
    [0.0, "#3b82f6"],  # Deep Blue (Small residuals)
    [0.4, "#a855f7"],  # Purple
    [0.7, "#f97316"],  # Orange
    [1.0, "#ef4444"],  # Bright Red (High residuals)
]

_BORDER = "rgba(0,0,0,0.08)"

_LAYOUT_BASE = dict(
    paper_bgcolor=_BG,
    plot_bgcolor=_SURFACE,
    font=dict(family=_FONT_FAMILY, color=_FONT_COLOR, size=12),
    margin=dict(l=55, r=35, t=45, b=45),
    hoverlabel=dict(
        bgcolor="white",
        bordercolor=_BORDER,
        font=dict(family=_FONT_FAMILY, color=_FONT_COLOR, size=11),
    ),
    legend=dict(
        font=dict(size=10),
        bgcolor="rgba(255,255,255,0.85)",
        bordercolor=_BORDER,
        borderwidth=1,
    ),
)

_AXIS_BASE = dict(
    gridcolor=_GRID,
    gridwidth=1,
    zerolinecolor="rgba(0,0,0,0.12)",
    zerolinewidth=1,
    linecolor="rgba(0,0,0,0.10)",
    tickfont=dict(size=10, color=_FONT_MUTED),
    title_font=dict(size=12, color=_FONT_COLOR),
)

_COLORBAR_STYLE = dict(
    thickness=12,
    tickfont=dict(color=_FONT_MUTED, size=10),
    outlinecolor=_BORDER,
    outlinewidth=1,
    bgcolor="rgba(0,0,0,0)",
    len=0.75,
)


def _title_dict(text: str) -> dict:
    """
    Build a standard centred title dictionary for Plotly layouts.

    Parameters
    ----------
    text : str
        The text content to be displayed in the title.

    Returns
    -------
    dict
        A dictionary containing the Plotly title configuration.
    """
    return dict(
        text=text,
        font=dict(size=14, color=_FONT_COLOR, family=_FONT_FAMILY),
        x=0.5,
        xanchor="center",
    )


def _layout(
    title: str = "", xaxis: dict | None = None, yaxis: dict | None = None
) -> dict:
    """
    Build a complete Plotly layout dictionary merging base configurations with per-chart overrides.

    Parameters
    ----------
    title : str, optional
        The plot title text. Defaults to an empty string.
    xaxis : dict | None, optional
        Specific styling overrides for the x-axis configuration. Defaults to None.
    yaxis : dict | None, optional
        Specific styling overrides for the y-axis configuration. Defaults to None.

    Returns
    -------
    dict
        The fully compiled layout configuration dictionary.
    """
    kw: dict = dict(**_LAYOUT_BASE)
    if title:
        kw["title"] = _title_dict(title)
    kw["xaxis"] = {**_AXIS_BASE, **(xaxis or {})}
    kw["yaxis"] = {**_AXIS_BASE, **(yaxis or {})}
    return kw


def _add_lowess_with_ci(
    fig: go.Figure,
    lowess_data: Dict[str, Any],
    line_color: str = _ACCENT_RED,
    fill_color: str = "rgba(239,68,68,0.10)",
    name: str = "LOWESS",
) -> None:
    """
    Add a LOWESS smoothed line trace with a shaded confidence interval band to an existing figure.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        The Plotly figure object to which the traces will be added.
    lowess_data : Dict[str, Any]
            Dict containing smoothing results ('x_smooth', 'y_smooth', 'ci_lower', 'ci_upper').
    line_color : str, optional
        The hex or rgba color string for the central LOWESS line. Defaults to _ACCENT_RED.
    fill_color : str, optional
        The rgba color string for the confidence interval band background. Defaults is red.
    name : str, optional
        The identifier name for the trace, shown in tooltips. Defaults to "LOWESS".

    Returns
    -------
    None
    """
    xs = lowess_data["x_smooth"]
    ys = lowess_data["y_smooth"]
    ci_lo = lowess_data["ci_lower"]
    ci_hi = lowess_data["ci_upper"]

    # CI band (upper boundary → reversed lower boundary to close the fill)
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([xs, xs[::-1]]).tolist(),
            y=np.concatenate([ci_hi, ci_lo[::-1]]).tolist(),
            fill="toself",
            fillcolor=fill_color,
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
            name=f"{name} 95% CI",
        )
    )

    # Smoothed line
    fig.add_trace(
        go.Scatter(
            x=xs.tolist(),
            y=ys.tolist(),
            mode="lines",
            line=dict(color=line_color, width=2),
            name=name,
            hoverinfo="skip",
        )
    )


def _get_sequential_color(t: float) -> str:
    """
    Get an RGB color string interpolated from a custom Blue-to-Red sequential color scale.

    Parameters
    ----------
    t : float
        A normalized value between 0.0 and 1.0 representing the position in the color scale.

    Returns
    -------
    str
        An RGB color string formatted for Plotly (e.g., 'rgb(59,130,246)').
    """
    stops = [
        (0.0, (59, 130, 246)),  # #3b82f6
        (0.4, (168, 85, 247)),  # #a855f7
        (0.7, (249, 115, 22)),  # #f97316
        (1.0, (239, 68, 68)),  # #ef4444
    ]
    t = float(np.clip(t, 0.0, 1.0))
    for i in range(len(stops) - 1):
        t0, c0 = stops[i]
        t1, c1 = stops[i + 1]
        if t0 <= t <= t1:
            frac = (t - t0) / (t1 - t0)
            r = int(c0[0] + frac * (c1[0] - c0[0]))
            g = int(c0[1] + frac * (c1[1] - c0[1]))
            b = int(c0[2] + frac * (c1[2] - c0[2]))
            return f"rgb({r},{g},{b})"
    return "rgb(239,68,68)"
