import json
import os
from typing import Any, Dict

import dash_bootstrap_components as dbc
from dash import dcc, html
from visualization.regression_plots import RegressionPlots

_DESC_FILE = os.path.join(
    os.path.dirname(__file__), "..", "..", "graph_descriptions.json"
)
with open(_DESC_FILE, "r", encoding="utf-8") as f:
    _GRAPH_DESCS = json.load(f)


def get_plots(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate all required regression training diagnostic plots.

    Parameters
    ----------
    results : Dict[str, Any]
        The complete diagnostic results collection dictionary.

    Returns
    -------
    Dict[str, Any]
        A dictionary mapping short handle names to generated Plotly Figure components
        for the training diagnostics.
    """
    plotter = RegressionPlots(results)
    plotter.plot_posterior_predictive_train()
    plotter.plot_linearity_train()
    plotter.plot_scale_location_train()
    plotter.plot_leverage_train()
    plotter.plot_vif_train()
    plotter.plot_qq_train()
    return {
        "post_pred": plotter.plots.get("posterior_predictive_train"),
        "linearity": plotter.plots.get("linearity_train"),
        "scale_loc": plotter.plots.get("scale_location_train"),
        "leverage": plotter.plots.get("leverage_train"),
        "vif": plotter.plots.get("vif_train"),
        "qq_train": plotter.plots.get("qq_train"),
    }


_GRAPH_H = "380px"


def _graph(graph_id: str, fig, height: str = _GRAPH_H, desc_key: str | None = None):
    """
    Wrap a Plotly figure in a dcc.Graph object.

    Provides standard dashboard sizing and an optional info tooltip activated.

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


def create_layout(diag: Any) -> dbc.Container:
    """
    Create and compile the overall structure for the Regression Train Diagnostics dashboard view.

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
    plots = get_plots(results)

    return dbc.Container(
        [
            html.Div(
                [
                    html.H5(
                        "Regression — Train Diagnostics",
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
            # Row 1: Overall Fit
            html.H6(
                "Overall Fit",
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
                            "train-post-pred",
                            plots.get("post_pred"),
                            desc_key="post_pred",
                        ),
                        width=12,
                        lg=6,
                        className="mb-3",
                    ),
                    dbc.Col(
                        _graph("train-qq", plots.get("qq_train"), desc_key="qq"),
                        width=12,
                        lg=6,
                        className="mb-3",
                    ),
                ]
            ),
            # Row 2: Residual Behavior
            html.H6(
                "Residual Behavior",
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
                            "train-linearity",
                            plots.get("linearity"),
                            desc_key="linearity",
                        ),
                        width=12,
                        lg=6,
                        className="mb-3",
                    ),
                    dbc.Col(
                        _graph(
                            "train-scale-loc",
                            plots.get("scale_loc"),
                            desc_key="scale_loc",
                        ),
                        width=12,
                        lg=6,
                        className="mb-3",
                    ),
                ]
            ),
            # Row 3: Model Validity
            html.H6(
                "Model Validity",
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
                            "train-leverage", plots.get("leverage"), desc_key="leverage"
                        ),
                        width=12,
                        lg=6,
                        className="mb-3",
                    ),
                    dbc.Col(
                        _graph("train-vif", plots.get("vif"), desc_key="vif"),
                        width=12,
                        lg=6,
                        className="mb-3",
                    ),
                ]
            ),
            # SHAP placeholder
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
                        style={"backgroundColor": "#f8f9fa", "borderRadius": "8px"},
                    ),
                    width=12,
                    className="mb-4",
                ),
            ),
        ],
        fluid=True,
        style={"maxWidth": "1400px"},
    )
