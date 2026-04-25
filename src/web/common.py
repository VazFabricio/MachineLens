import dash_bootstrap_components as dbc
from dash import html


def create_navbar(task_type: str, current_path: str) -> dbc.Navbar:
    """
    Create a persistent top navigation bar indicating available modes and application identity.

    Parameters
    ----------
    task_type : str
        The designated evaluation task type, either 'regression' or 'classification'.
    current_path : str
        The active URL path requested by the browser, used for styling active links.

    Returns
    -------
    dbc.Navbar
        A styled navigation bar component containing conditional navigational buttons.
    """

    def get_link_style(path: str):
        """
        Dynamically construct inline styling dict for navigation links.

        Parameters
        ----------
        path : str
            The URL path mapped to the navigation button.

        Returns
        -------
        dict
            CSS style dictionary enabling distinct highlights for the active path.
        """
        if current_path == path:
            return {
                "backgroundColor": "white",
                "color": "black",
                "border": "1px solid black",
                "transition": "all 0.3s",
            }
        return {
            "color": "#6c757d",
            "border": "1px solid transparent",
            "backgroundColor": "transparent",
        }

    links = []

    if task_type == "regression":
        links = [
            dbc.NavLink(
                [html.I(className="bi bi-graph-up me-2"), "Test Diagnostics"],
                href="/regression/test",
                active="exact",
                className="mx-2 px-3 py-2 rounded-pill fw-bold",
                style=get_link_style("/regression/test"),
            ),
            dbc.NavLink(
                [html.I(className="bi bi-speedometer2 me-2"), "Train Diagnostics"],
                href="/regression/train",
                active="exact",
                className="mx-2 px-3 py-2 rounded-pill fw-bold",
                style=get_link_style("/regression/train"),
            ),
        ]
    elif task_type == "classification":
        links = [
            dbc.NavLink(
                [html.I(className="bi bi-bar-chart-fill me-2"), "Test Diagnostics"],
                href="/classification/test",
                active="exact",
                className="mx-2 px-3 py-2 rounded-pill fw-bold",
                style=get_link_style("/classification/test"),
            ),
            dbc.NavLink(
                [html.I(className="bi bi-pie-chart-fill me-2"), "Train Diagnostics"],
                href="/classification/train",
                active="exact",
                className="mx-2 px-3 py-2 rounded-pill fw-bold",
                style=get_link_style("/classification/train"),
            ),
        ]

    return dbc.Navbar(
        dbc.Container(
            [
                html.A(
                    html.Img(
                        src="/assets/logo.jpg",
                        height="40px",
                        className="d-inline-block align-top",
                    ),
                    href="/",
                    style={"textDecoration": "none"},
                ),
                dbc.NavbarToggler(id="navbar-toggler", n_clicks=0),
                dbc.Collapse(
                    dbc.Nav(
                        links,
                        className="ms-auto",
                        pills=True,
                    ),
                    id="navbar-collapse",
                    is_open=False,
                    navbar=True,
                ),
            ],
            fluid=True,
            className="px-4",
        ),
        color="white",
        dark=False,
        className="border-bottom py-3 shadow-sm",
        style={"zIndex": 1000},
    )
