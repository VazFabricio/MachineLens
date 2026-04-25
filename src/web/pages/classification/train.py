import json
import os
from typing import Any, Dict

import dash_bootstrap_components as dbc
from dash import dcc, html
from visualization.classification_plots import ClassificationPlots

_DESC_FILE = os.path.join(
    os.path.dirname(__file__), "..", "..", "graph_descriptions.json"
)
with open(_DESC_FILE, "r", encoding="utf-8") as f:
    _GRAPH_DESCS = json.load(f)


_GRAPH_H = "380px"


def _graph(graph_id: str, fig, height: str = _GRAPH_H, desc_key: str | None = None):
    """
    Wrap a Plotly in a dcc.Graph object with standard dashboard sizing and an optional info tooltip.

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
        A Graph or a visual alert string fallback if model failed to compute visual representation.
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


def get_plots(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate all required classification training diagnostic plots.

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
    plotter = ClassificationPlots(results)
    plotter.plot_metrics_table_train()
    plotter.plot_confusion_matrix_train()
    plotter.plot_class_distribution_train()
    plotter.plot_roc_curve_train()
    plotter.plot_pr_curve_train()
    plotter.plot_probability_distribution_train()
    plotter.plot_misclassification_feature_plots_train()
    return {
        "metrics": plotter.plots.get("metrics_table_train"),
        "cm": plotter.plots.get("confusion_matrix_train"),
        "class_dist": plotter.plots.get("class_distribution_train"),
        "roc": plotter.plots.get("roc_curve_train"),
        "pr": plotter.plots.get("pr_curve_train"),
        "prob_dist": plotter.plots.get("probability_distribution_train"),
        "misclass": plotter.plots.get("misclassification_feature_plots_train"),
    }


def create_layout(results: Dict[str, Any]) -> dbc.Container:
    """
    Create and compile the structure for the Classification Train Diagnostics dashboard view.

    Parameters
    ----------
    results : Dict[str, Any]
        The extracted analysis output dictionary fetched from ClassificationDiagnostics.

    Returns
    -------
    dbc.Container
        Bootstrap standard container carrying vertically stacked graph rows.
    """
    model_name = results.get("metadata", {}).get("model_name", "Unknown Model")
    plots = get_plots(results)

    return dbc.Container(
        [
            html.Div(
                [
                    html.H5(
                        "Classification — Train Diagnostics",
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
            # Row 1: Performance
            html.H6(
                "Performance",
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
                            "clf-metrics-train",
                            plots.get("metrics"),
                            desc_key="metrics",
                        ),
                        width=12,
                        lg=6,
                        className="mb-3",
                    ),
                    dbc.Col(
                        _graph("clf-cm-train", plots.get("cm"), desc_key="cm"),
                        width=12,
                        lg=6,
                        className="mb-3",
                    ),
                ]
            ),
            # Row 2: Class Insight
            html.H6(
                "Class Insight",
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
                            "clf-class-dist-train",
                            plots.get("class_dist"),
                            desc_key="class_dist",
                        ),
                        width=12,
                        lg=6,
                        className="mb-3",
                    ),
                    dbc.Col(
                        _graph(
                            "clf-prob-dist-train",
                            plots.get("prob_dist"),
                            desc_key="prob_dist",
                        ),
                        width=12,
                        lg=6,
                        className="mb-3",
                    ),
                ]
            ),
            # Row 3: Curves
            html.H6(
                "Decision Curves",
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
                        _graph("clf-roc-train", plots.get("roc"), desc_key="roc"),
                        width=12,
                        lg=6,
                        className="mb-3",
                    ),
                    dbc.Col(
                        _graph("clf-pr-train", plots.get("pr"), desc_key="pr"),
                        width=12,
                        lg=6,
                        className="mb-3",
                    ),
                ]
            ),
            # Row 4: Misclassifications
            html.H6(
                "Misclassification Deep-Dive",
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
                        (
                            _graph(
                                "clf-misclass-train",
                                plots.get("misclass"),
                                "420px",
                                desc_key="misclass",
                            )
                            if plots.get("misclass")
                            else html.P(
                                "No misclassification features available.",
                                className="text-muted text-center py-4 small",
                            )
                        ),
                        width=12,
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
                )
            ),
        ],
        fluid=True,
        style={"maxWidth": "1400px"},
    )
