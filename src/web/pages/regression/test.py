import json
import os
from typing import Any, Dict

import dash_bootstrap_components as dbc
import numpy as np
from dash import dcc, html
from visualization.regression_plots import RegressionPlots

_DESC_FILE = os.path.join(
    os.path.dirname(__file__), "..", "..", "graph_descriptions.json"
)
with open(_DESC_FILE, "r", encoding="utf-8") as f:
    _GRAPH_DESCS = json.load(f)


# ---------------------------------------------------------------------------
# Metrics helpers – compute raw numbers for the badge bar
# ---------------------------------------------------------------------------


def _compute_metrics(results: Dict[str, Any]) -> Dict[str, str]:
    """
    Compute raw performance metrics based on test residuals for the badge bar.

    Parameters
    ----------
    results : Dict[str, Any]
        The results dictionary obtained from the regression diagnostics, containing
        true test values and residuals under the 'residuals_data' key.

    Returns
    -------
    Dict[str, str]
        A dictionary mapping metric names (e.g., 'R²', 'MAE', 'RMSE') to their
        formatted string representations.
    """
    rd = results.get("residuals_data", {})
    y_test = rd.get("y_test")
    residuals = rd.get("residuals")
    if y_test is None or residuals is None:
        return {}

    n = len(y_test)
    mae = float(np.mean(np.abs(residuals)))
    mse = float(np.mean(residuals**2))
    rmse = float(np.sqrt(mse))
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y_test - np.mean(y_test)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot != 0 else float("nan")

    non_zero = y_test != 0
    mape = (
        float(np.mean(np.abs(residuals[non_zero] / y_test[non_zero])) * 100)
        if non_zero.any()
        else float("nan")
    )

    return {
        "R²": f"{r2:.4f}",
        "MAE": f"{mae:.4f}",
        "RMSE": f"{rmse:.4f}",
        "MSE": f"{mse:.4f}",
        "MAPE": f"{mape:.2f}%",
        "N": f"{n}",
    }


def _metric_badge(name: str, value: str) -> dbc.Col:
    """
    Render a single metric visually as a compact Bootstrap card.

    Parameters
    ----------
    name : str
        The title indicating which metric is being displayed (e.g., 'MSE').
    value : str
        The formatted scalar value of the metric.

    Returns
    -------
    dbc.Col
        A dash-bootstrap-components column element housing the styling card.
    """
    return dbc.Col(
        dbc.Card(
            dbc.CardBody(
                [
                    html.P(
                        name,
                        className="mb-0 text-muted",
                        style={
                            "fontSize": "0.65rem",
                            "fontWeight": "600",
                            "textTransform": "uppercase",
                            "letterSpacing": "0.5px",
                        },
                    ),
                    html.H5(
                        value,
                        className="mb-0 fw-bold",
                        style={"fontSize": "1.05rem", "color": "#1e293b"},
                    ),
                ],
                className="text-center py-2 px-1",
            ),
            className="shadow-sm border-0",
            style={"borderRadius": "8px", "backgroundColor": "#ffffff"},
        ),
        className="px-1",
    )


# ---------------------------------------------------------------------------
# Plot generation
# ---------------------------------------------------------------------------


def get_plots(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate all required regression test analytical plots dynamically.

    Parameters
    ----------
    results : Dict[str, Any]
        The complete diagnostic results collection dictionary.

    Returns
    -------
    Dict[str, Any]
        A dictionary mapping the short handle names to generated Plotly Figure components,
        including nested references mapping generated residual outlier plots.
    """
    plotter = RegressionPlots(results)
    plotter.plot_actual_vs_predicted_test()
    plotter.plot_residual_distribution_test()
    plotter.plot_residuals_vs_predicted_test()
    plotter.plot_residuals_vs_actual_test()
    plotter.plot_qq_test()
    plotter.plot_scale_location_test()
    plotter.plot_residual_outliers_test()

    outlier_plots = {
        k: v for k, v in plotter.plots.items() if k.startswith("residual_outlier_")
    }

    return {
        "act_vs_pred": plotter.plots.get("actual_vs_predicted_test"),
        "res_dist": plotter.plots.get("residual_distribution_test"),
        "res_vs_pred": plotter.plots.get("residuals_vs_predicted_test"),
        "res_vs_act": plotter.plots.get("residuals_vs_actual_test"),
        "qq_test": plotter.plots.get("qq_test"),
        "scale_loc": plotter.plots.get("scale_location_test"),
        "outlier_plots": outlier_plots,
    }


# ---------------------------------------------------------------------------
# Graph helper – small wrapper to keep the layout DRY
# ---------------------------------------------------------------------------
_GRAPH_H = "380px"


def _graph(graph_id: str, fig, height: str = _GRAPH_H, desc_key: str | None = None):
    """
    Wrap a Plotly figure in a dcc.Graph object.

    Provides standard dashboard sizing and an optional info tooltip activated
    on hover via a key from `graph_descriptions.json`.

    Parameters
    ----------
    graph_id : str
        The designated HTML ID for the Dash component.
    fig : plotly.graph_objs._figure.Figure
        The pre-computed Plotly figure instance.
    height : str, optional
        The CSS height of the generated graph, by default 380px.
    desc_key : str, optional
        Key referencing an external description stored in `graph_descriptions.json`.
        If given, creates a pop-up description activated on hover.

    Returns
    -------
    html.Div | dbc.Alert
        A styled Graph interface component
        or a visual alert string fallback if model failed to compute visual representation.
    """
    if fig is None:
        return dbc.Alert("Graph unavailable", color="warning", className="h-100")

    children = [
        dcc.Graph(
            id=graph_id,
            figure=fig,
            style={"height": height},
            config={
                "displayModeBar": True,
                "modeBarButtonsToRemove": ["toImage", "sendDataToCloud"],
            },
        )
    ]

    if desc_key and desc_key in _GRAPH_DESCS:
        info_id = f"info-{graph_id}"
        children.extend(
            [
                html.I(
                    className="bi bi-info-circle text-muted hover-icon",
                    id=info_id,
                    style={
                        "position": "absolute",
                        "top": "15px",
                        "left": "15px",
                        "cursor": "help",
                        "zIndex": 1050,
                        "fontSize": "1.2rem",
                    },
                ),
                dbc.Tooltip(
                    _GRAPH_DESCS[desc_key],
                    target=info_id,
                    placement="bottom",
                ),
            ]
        )

    return html.Div(children, style={"position": "relative"})


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


def create_layout(diag: Any) -> dbc.Container:
    """
    Create and compile the overall structure for the Regression Test Diagnostics dashboard view.

    Parameters
    ----------
    diag : Any
        The instantiated diagnostics object (RegressionDiagnostics).

    Returns
    -------
    dbc.Container
        Bootstrap standard container carrying vertically stacked graph rows.
    """
    results = getattr(diag, "results", {})
    try:
        model_name = diag.model.__class__.__name__
    except AttributeError:
        model_name = results.get("metadata", {}).get("model_name", "Unknown Model")
    metrics = _compute_metrics(results)
    plots = get_plots(results)

    # Outlier pair-cards
    outlier_rows = []
    if plots.get("outlier_plots"):
        items = list(plots["outlier_plots"].values())
        for i in range(0, len(items), 2):
            cols = [
                dbc.Col(
                    _graph(f"reg-outlier-{i}", items[i], "320px", desc_key="outliers"),
                    width=12,
                    lg=6,
                    className="mb-3",
                )
            ]
            if i + 1 < len(items):
                cols.append(
                    dbc.Col(
                        _graph(
                            f"reg-outlier-{i+1}",
                            items[i + 1],
                            "320px",
                            desc_key="outliers",
                        ),
                        width=12,
                        lg=6,
                        className="mb-3",
                    )
                )
            outlier_rows.append(dbc.Row(cols))
    else:
        outlier_rows = [
            dbc.Row(
                dbc.Col(
                    html.P(
                        "No significant outlier features detected.",
                        className="text-muted text-center py-4 small",
                    ),
                    width=12,
                )
            )
        ]

    return dbc.Container(
        [
            # ── Title ──────────────────────────────────────────────
            html.Div(
                [
                    html.H5(
                        "Regression — Test Diagnostics",
                        className="mb-0",
                        style={"fontWeight": "700", "color": "#1e293b"},
                    ),
                    html.Span(
                        model_name,
                        className="badge bg-primary bg-opacity-10 text-primary ms-2",
                        style={"fontSize": "0.7rem", "fontWeight": "600"},
                    ),
                ],
                className="d-flex align-items-center mb-3 mt-1",
            ),
            # ── Metric badges ─────────────────────────────────────
            dbc.Row(
                [_metric_badge(k, v) for k, v in metrics.items()],
                className="g-2 mb-4",
            ),
            # ── Row 1: Hero graphs ────────────────────────────────
            html.H6(
                "Parity & Distribution",
                className="text-secondary border-bottom pb-1 mb-2",
                style={
                    "fontSize": "0.75rem",
                    "fontWeight": "700",
                    "textTransform": "uppercase",
                    "letterSpacing": "1px",
                },
            ),
            dbc.Row(
                [
                    dbc.Col(
                        _graph(
                            "reg-act-vs-pred",
                            plots.get("act_vs_pred"),
                            desc_key="act_vs_pred",
                        ),
                        width=12,
                        lg=6,
                        className="mb-3",
                    ),
                    dbc.Col(
                        _graph(
                            "reg-res-dist", plots.get("res_dist"), desc_key="res_dist"
                        ),
                        width=12,
                        lg=6,
                        className="mb-3",
                    ),
                ],
            ),
            # ── Row 2: Residual scatters ──────────────────────────
            html.H6(
                "Residual Analysis",
                className="text-secondary border-bottom pb-1 mb-2 mt-2",
                style={
                    "fontSize": "0.75rem",
                    "fontWeight": "700",
                    "textTransform": "uppercase",
                    "letterSpacing": "1px",
                },
            ),
            dbc.Row(
                [
                    dbc.Col(
                        _graph(
                            "reg-res-vs-pred",
                            plots.get("res_vs_pred"),
                            desc_key="res_vs_pred",
                        ),
                        width=12,
                        lg=6,
                        className="mb-3",
                    ),
                    dbc.Col(
                        _graph(
                            "reg-res-vs-act",
                            plots.get("res_vs_act"),
                            desc_key="res_vs_act",
                        ),
                        width=12,
                        lg=6,
                        className="mb-3",
                    ),
                ],
            ),
            # ── Row 3: Statistical validation ─────────────────────
            html.H6(
                "Statistical Validation",
                className="text-secondary border-bottom pb-1 mb-2 mt-2",
                style={
                    "fontSize": "0.75rem",
                    "fontWeight": "700",
                    "textTransform": "uppercase",
                    "letterSpacing": "1px",
                },
            ),
            dbc.Row(
                [
                    dbc.Col(
                        _graph("reg-qq-test", plots.get("qq_test"), desc_key="qq"),
                        width=12,
                        lg=6,
                        className="mb-3",
                    ),
                    dbc.Col(
                        _graph(
                            "reg-scale-loc",
                            plots.get("scale_loc"),
                            desc_key="scale_loc",
                        ),
                        width=12,
                        lg=6,
                        className="mb-3",
                    ),
                ],
            ),
            # ── Row 4: Outlier deep-dive ──────────────────────────
            html.H6(
                "Outlier Deep-Dive",
                className="text-secondary border-bottom pb-1 mb-2 mt-2",
                style={
                    "fontSize": "0.75rem",
                    "fontWeight": "700",
                    "textTransform": "uppercase",
                    "letterSpacing": "1px",
                },
            ),
            html.Div(outlier_rows),
            # ── Row 5: SHAP placeholder ──────────────────────────
            html.H6(
                "Explainability",
                className="text-secondary border-bottom pb-1 mb-2 mt-2",
                style={
                    "fontSize": "0.75rem",
                    "fontWeight": "700",
                    "textTransform": "uppercase",
                    "letterSpacing": "1px",
                },
            ),
            dbc.Row(
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.P(
                                    "Feature Importance (SHAP)",
                                    className="text-muted text-center mb-0 fw-bold small",
                                ),
                                html.P(
                                    "[Placeholder for future implementation]",
                                    className="text-muted text-center mb-0 small",
                                ),
                            ],
                            className="py-4",
                        ),
                        className="border-0",
                        style={
                            "backgroundColor": "#f8f9fa",
                            "border": "1px dashed #dee2e6 !important",
                            "borderRadius": "8px",
                        },
                    ),
                    width=12,
                    className="mb-4",
                ),
            ),
        ],
        fluid=True,
        style={"maxWidth": "1400px"},
    )
