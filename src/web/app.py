import logging
from typing import Any

import dash
import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, State, dcc, html

from .common import create_navbar

logger = logging.getLogger(__name__)


def create_dashboard(diag: Any) -> Dash:
    """
    Generate a Plotly Dash dashboard dynamically based on the diagnostic object.

    Parameters
    ----------
    diag : Any
        The instantiated diagnostics object (RegressionDiagnostics or ClassificationDiagnostics).

    Returns
    -------
    Dash
        The Plotly Dash application instance.
    """
    app = Dash(
        __name__,
        external_stylesheets=[dbc.themes.FLATLY, dbc.icons.BOOTSTRAP],
        suppress_callback_exceptions=True,
    )

    task_type = getattr(diag.model_interface, "problem_type", "regression")
    results = getattr(diag, "results", {})

    app.title = f"MachineLens - {task_type.capitalize()} Diagnostics"

    app.layout = html.Div(
        [
            dcc.Location(id="url", refresh=False),
            create_navbar(task_type, "/"),
            html.Div(id="page-content", style={"padding": "1.5rem"}),
        ]
    )

    @app.callback(Output("page-content", "children"), [Input("url", "pathname")])
    def render_page_content(pathname: str):
        if not pathname or pathname in ["/", ""]:
            pathname = f"/{task_type}/test"

        if task_type == "regression":
            if pathname == "/regression/test":
                from .pages.regression.test import create_layout

                return create_layout(diag)
            elif pathname == "/regression/train":
                from .pages.regression.train import create_layout

                return create_layout(diag)

        elif task_type == "classification":
            if pathname == "/classification/test":
                from .pages.classification.test import create_layout

                return create_layout(diag)
            elif pathname == "/classification/train":
                from .pages.classification.train import create_layout

                return create_layout(diag)

        return dbc.Container(
            [
                html.H1("404: Not found", className="text-danger"),
                html.Hr(),
                html.P(f"The pathname {pathname} was not recognized..."),
            ],
            className="p-5",
        )

    # -------------------------------------------------------------
    # Cross-Filtering for Regression Test
    # -------------------------------------------------------------
    if task_type == "regression":
        import numpy as np

        app.layout.children.append(dcc.Store(id="xf-indices", data=None))

        _SCATTER_IDS = ["reg-act-vs-pred", "reg-res-vs-pred", "reg-res-vs-act"]

        # ── Selection callback (box / lasso) ────────────────────
        @app.callback(
            Output("xf-indices", "data"),
            [Input(gid, "selectedData") for gid in _SCATTER_IDS],
            prevent_initial_call=True,
        )
        def store_selected_indices(*selections):
            ctx = dash.callback_context
            if not ctx.triggered:
                raise dash.exceptions.PreventUpdate
            trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
            idx = _SCATTER_IDS.index(trigger_id)
            sel = selections[idx]
            if sel and sel.get("points"):
                indices = sorted(
                    {p["pointIndex"] for p in sel["points"] if "pointIndex" in p}
                )
                return {"indices": indices, "source": trigger_id}
            return None

        # ── Zoom callback (relayoutData) ────────────────────────
        @app.callback(
            Output("xf-indices", "data", allow_duplicate=True),
            [Input(gid, "relayoutData") for gid in _SCATTER_IDS],
            prevent_initial_call=True,
        )
        def sync_zoom(*relayout_datas):
            ctx = dash.callback_context
            if not ctx.triggered:
                raise dash.exceptions.PreventUpdate
            trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
            idx = _SCATTER_IDS.index(trigger_id)
            rd_event = relayout_datas[idx]
            if rd_event is None:
                raise dash.exceptions.PreventUpdate
            # Double-click reset
            if "xaxis.autorange" in rd_event or "autosize" in rd_event:
                return None
            x0 = rd_event.get("xaxis.range[0]")
            x1 = rd_event.get("xaxis.range[1]")
            y0 = rd_event.get("yaxis.range[0]")
            y1 = rd_event.get("yaxis.range[1]")
            if x0 is None and y0 is None:
                raise dash.exceptions.PreventUpdate
            # Map graph id → (x_data, y_data)
            res_data = results.get("residuals_data", {})
            y_test = res_data.get("y_test")
            y_pred = res_data.get("y_pred")
            residuals_arr = res_data.get("residuals")
            if y_test is None or y_pred is None or residuals_arr is None:
                raise dash.exceptions.PreventUpdate
            graph_data = {
                "reg-act-vs-pred": (np.asarray(y_test), np.asarray(y_pred)),
                "reg-res-vs-pred": (np.asarray(y_pred), np.asarray(residuals_arr)),
                "reg-res-vs-act": (np.asarray(y_test), np.asarray(residuals_arr)),
            }
            x_arr, y_arr = graph_data[trigger_id]
            mask = np.ones(len(x_arr), dtype=bool)
            if x0 is not None and x1 is not None:
                mask &= (x_arr >= float(x0)) & (x_arr <= float(x1))
            if y0 is not None and y1 is not None:
                mask &= (y_arr >= float(y0)) & (y_arr <= float(y1))
            indices = np.where(mask)[0].tolist()
            if not indices or len(indices) == len(x_arr):
                return None
            return {"indices": indices, "source": trigger_id}

        # ── Apply to each scatter ───────────────────────────────
        for gid in _SCATTER_IDS:

            @app.callback(
                Output(gid, "figure"),
                Input("xf-indices", "data"),
                State(gid, "figure"),
                prevent_initial_call=True,
            )
            def _apply_selection(store_data, current_fig, _gid=gid):
                if current_fig is None:
                    raise dash.exceptions.PreventUpdate
                from dash import Patch

                patched = Patch()
                if store_data is None or store_data.get("source") == _gid:
                    for i, trace in enumerate(current_fig.get("data", [])):
                        if (
                            trace.get("mode") == "markers"
                            or trace.get("type") == "scatter"
                        ):
                            patched["data"][i]["selectedpoints"] = None
                    return patched
                indices = store_data.get("indices", [])
                for i, trace in enumerate(current_fig.get("data", [])):
                    if trace.get("mode") == "markers":
                        patched["data"][i]["selectedpoints"] = indices
                return patched

    # -------------------------------------------------------------
    # Navbar Toggler Callback
    # -------------------------------------------------------------
    @app.callback(
        Output("navbar-collapse", "is_open"),
        [Input("navbar-toggler", "n_clicks")],
        [State("navbar-collapse", "is_open")],
    )
    def toggle_navbar_collapse(n, is_open):
        if n:
            return not is_open
        return is_open

    return app


def launch_dashboard(diag: Any, port: int = 8050) -> None:
    """
    Launch the interactive dashboard for model diagnostics.

    Parameters
    ----------
    diag : Any
        The diagnostic object containing results and model interface.
    port : int, optional
        The port to run the web server on. Default is 8050.

    Returns
    -------
    None
    """
    task_type = getattr(diag.model_interface, "problem_type", "unknown")

    print(f"Launching MachineLens dashboard for {task_type} evaluation...")
    print(f"Available locally at http://127.0.0.1:{port}")
    app = create_dashboard(diag)
    app.run(port=port, debug=False)
