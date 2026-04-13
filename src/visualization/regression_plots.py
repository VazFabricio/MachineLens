from typing import Any, Dict

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
_ACCENT_AMBER = "#f59e0b"
_COLORSCALE = "Plasma"
_BORDER = "rgba(0,0,0,0.08)"

_LAYOUT_BASE = dict(
    paper_bgcolor=_BG,
    plot_bgcolor=_SURFACE,
    font=dict(family=_FONT_FAMILY, color=_FONT_COLOR, size=13),
    margin=dict(l=65, r=50, t=60, b=60),
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


def _layout(
    title: str = "", xaxis: dict | None = None, yaxis: dict | None = None
) -> dict:
    """Build a complete layout dict merging base + per-chart overrides."""
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
    """Add a LOWESS smoothed line with shaded CI band to a figure."""
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


def _plasma_color(t: float) -> str:
    """Get a Plasma colorscale color from a 0-1 float."""
    plasma_stops = [
        (0.0, (13, 8, 135)),
        (0.25, (126, 3, 168)),
        (0.5, (204, 71, 120)),
        (0.75, (248, 149, 64)),
        (1.0, (240, 249, 33)),
    ]
    t = float(np.clip(t, 0.0, 1.0))
    for i in range(len(plasma_stops) - 1):
        t0, c0 = plasma_stops[i]
        t1, c1 = plasma_stops[i + 1]
        if t0 <= t <= t1:
            frac = (t - t0) / (t1 - t0)
            r = int(c0[0] + frac * (c1[0] - c0[0]))
            g = int(c0[1] + frac * (c1[1] - c0[1]))
            b = int(c0[2] + frac * (c1[2] - c0[2]))
            return f"rgb({r},{g},{b})"
    return "rgb(240,249,33)"


class RegressionPlots:
    """
    Visualization suite for regression models.

    This class receives raw diagnostic data calculated by ``RegressionDiagnostics``
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
        Initialize the RegressionPlots object.

        Parameters
        ----------
        results : dict
            The ``results`` dictionary from ``RegressionDiagnostics`` containing
            all numerical data arrays and dataframes.
        """
        self.results = results
        self.plots: Dict[str, go.Figure] = {}

    def _get_res_data(self) -> Dict[str, Any]:
        """Get residuals_data (test) from results."""
        return self.results.get("residuals_data", {})

    def _get_train_data(self) -> Dict[str, Any]:
        """Get training_diagnostics from results."""
        return self.results.get("training_diagnostics", {})

    # ======================================================================
    #  TEST-DATA PLOTS
    # ======================================================================

    def plot_metrics_table(self) -> None:
        """Generate a styled Plotly table displaying global regression metrics."""
        res_data = self._get_res_data()
        y_test = res_data.get("y_test")
        y_pred = res_data.get("y_pred")
        residuals = res_data.get("residuals")

        if y_test is None or y_pred is None or residuals is None:
            return

        n = len(y_test)
        mae = float(np.mean(np.abs(residuals)))
        mse = float(np.mean(residuals**2))
        rmse = float(np.sqrt(mse))
        ss_res = float(np.sum(residuals**2))
        ss_tot = float(np.sum((y_test - np.mean(y_test)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot != 0 else float("nan")

        non_zero_mask = y_test != 0
        if non_zero_mask.any():
            mape = float(
                np.mean(np.abs(residuals[non_zero_mask] / y_test[non_zero_mask])) * 100
            )
        else:
            mape = float("nan")

        rows = [
            ("MAE", f"{mae:.4f}"),
            ("MSE", f"{mse:.4f}"),
            ("RMSE", f"{rmse:.4f}"),
            ("R²", f"{r2:.4f}"),
            ("MAPE (%)", f"{mape:.2f}"),
            ("n (samples)", f"{n}"),
        ]
        names = [r[0] for r in rows]
        values = [r[1] for r in rows]

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
            "title": _title_dict("Regression Metrics"),
            "margin": dict(l=30, r=30, t=55, b=20),
        }
        fig.update_layout(**layout_kwargs)
        self.plots["metrics_table"] = fig

    def plot_actual_vs_predicted_test(self) -> None:
        """Plot Actual vs. Predicted values for **test** data."""
        res_data = self._get_res_data()
        y_test_arr = res_data.get("y_test")
        y_pred_arr = res_data.get("y_pred")
        abs_residuals = res_data.get("abs_residuals")

        if y_test_arr is None or y_pred_arr is None or abs_residuals is None:
            return

        vmax = float(np.percentile(abs_residuals, 97))
        marker_color = np.clip(abs_residuals, 0, vmax)

        min_val = float(np.min(y_test_arr))
        max_val = float(np.max(y_test_arr))

        fig = go.Figure()

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

        fig.add_trace(
            go.Scatter(
                x=y_test_arr.tolist(),
                y=y_pred_arr.tolist(),
                mode="markers",
                name="Observations",
                marker=dict(
                    color=marker_color.tolist(),
                    colorscale=_COLORSCALE,
                    cmin=0,
                    cmax=vmax,
                    size=6,
                    opacity=0.85,
                    colorbar=dict(title="|Residual|", **_COLORBAR_STYLE),
                    line=dict(width=0.3, color="rgba(0,0,0,0.1)"),
                ),
                hovertemplate=(
                    "Actual: <b>%{x:.4g}</b><br>"
                    "Predicted: <b>%{y:.4g}</b><br>"
                    "|Residual|: <b>%{marker.color:.4g}</b><extra></extra>"
                ),
            )
        )

        fig.update_layout(
            **_layout(
                title="Actual vs. Predicted (Test)",
                xaxis=dict(title="Actual"),
                yaxis=dict(title="Predicted"),
            )
        )
        self.plots["actual_vs_predicted_test"] = fig

    def plot_residuals_vs_actual_test(self) -> None:
        """Plot Residuals vs. Actual values for **test** data."""
        res_data = self._get_res_data()
        y_test_arr = res_data.get("y_test")
        residuals = res_data.get("residuals")
        abs_residuals = res_data.get("abs_residuals")

        if y_test_arr is None or residuals is None or abs_residuals is None:
            return

        vmax = float(np.percentile(abs_residuals, 97))
        marker_color = np.clip(abs_residuals, 0, vmax)

        fig = go.Figure()

        fig.add_hline(
            y=0,
            line=dict(color="rgba(0,0,0,0.25)", dash="dash", width=1.5),
        )

        fig.add_trace(
            go.Scatter(
                x=y_test_arr.tolist(),
                y=residuals.tolist(),
                mode="markers",
                name="Residuals",
                marker=dict(
                    color=marker_color.tolist(),
                    colorscale=_COLORSCALE,
                    cmin=0,
                    cmax=vmax,
                    size=6,
                    opacity=0.85,
                    colorbar=dict(title="|Residual|", **_COLORBAR_STYLE),
                    line=dict(width=0.3, color="rgba(0,0,0,0.1)"),
                ),
                hovertemplate=(
                    "Actual: <b>%{x:.4g}</b><br>"
                    "Residual: <b>%{y:.4g}</b><br>"
                    "|Residual|: <b>%{marker.color:.4g}</b><extra></extra>"
                ),
            )
        )

        fig.update_layout(
            **_layout(
                title="Residuals vs. Actual (Test)",
                xaxis=dict(title="Actual"),
                yaxis=dict(title="Residual"),
            )
        )

        # Add LOWESS + CI
        lowess = res_data.get("linearity_lowess")
        if lowess:
            # Reusing the linearity lowess for vs. actual is standard if distribution allows
            _add_lowess_with_ci(fig, lowess)

        self.plots["residuals_vs_actual_test"] = fig

    def plot_residuals_vs_predicted_test(self) -> None:
        """Plot Residuals vs. Predicted values for **test** data."""
        res_data = self._get_res_data()
        y_pred_arr = res_data.get("y_pred")
        residuals = res_data.get("residuals")
        abs_residuals = res_data.get("abs_residuals")

        if y_pred_arr is None or residuals is None or abs_residuals is None:
            return

        vmax = float(np.percentile(abs_residuals, 97))
        marker_color = np.clip(abs_residuals, 0, vmax)

        fig = go.Figure()

        fig.add_hline(
            y=0,
            line=dict(color="rgba(0,0,0,0.25)", dash="dash", width=1.5),
        )

        fig.add_trace(
            go.Scatter(
                x=y_pred_arr.tolist(),
                y=residuals.tolist(),
                mode="markers",
                name="Residuals",
                marker=dict(
                    color=marker_color.tolist(),
                    colorscale=_COLORSCALE,
                    cmin=0,
                    cmax=vmax,
                    size=5,
                    opacity=0.85,
                    colorbar=dict(title="|Residual|", **_COLORBAR_STYLE),
                    line=dict(width=0.3, color="rgba(0,0,0,0.1)"),
                ),
                hovertemplate=(
                    "Predicted: <b>%{x:.4g}</b><br>"
                    "Residual: <b>%{y:.4g}</b><br>"
                    "|Residual|: <b>%{marker.color:.4g}</b><extra></extra>"
                ),
            )
        )

        fig.update_layout(
            **_layout(
                title="Residuals vs. Predicted (Test)",
                xaxis=dict(title="Predicted"),
                yaxis=dict(title="Residual"),
            )
        )

        # Add LOWESS + CI
        lowess = res_data.get("linearity_lowess")
        if lowess:
            _add_lowess_with_ci(fig, lowess)

        self.plots["residuals_vs_predicted_test"] = fig

    def plot_residual_distribution_test(self) -> None:
        """Plot the distribution histogram of residuals for **test** data."""
        res_data = self._get_res_data()
        residuals = res_data.get("residuals")
        abs_residuals = res_data.get("abs_residuals")

        if residuals is None or abs_residuals is None:
            return

        vmax = float(np.percentile(abs_residuals, 97))
        mean_res = float(np.mean(residuals))

        counts, bin_edges = np.histogram(residuals, bins=30)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        norm_c = np.clip(np.abs(bin_centers) / max(1e-9, vmax), 0, 1)

        bar_colors = [_plasma_color(v) for v in norm_c]
        widths = (bin_edges[1:] - bin_edges[:-1]).tolist()

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=bin_centers.tolist(),
                y=counts.tolist(),
                width=widths,
                marker=dict(
                    color=bar_colors,
                    line=dict(color="rgba(0,0,0,0.15)", width=0.5),
                ),
                name="Residuals",
                hovertemplate="Center: <b>%{x:.4g}</b><br>Count: <b>%{y}</b><extra></extra>",
                opacity=0.9,
            )
        )

        fig.add_vline(
            x=mean_res,
            line=dict(color=_ACCENT_RED, dash="dash", width=1.5),
            annotation_text=f"Mean: {mean_res:.4g}",
            annotation_position="top right",
            annotation_font=dict(color=_ACCENT_RED, size=11),
        )

        fig.add_vline(
            x=0,
            line=dict(color="rgba(0,0,0,0.25)", dash="dash", width=1.2),
            annotation_text="0",
            annotation_position="top left",
            annotation_font=dict(color=_FONT_MUTED, size=11),
        )

        fig.update_layout(
            **_layout(
                title="Residual Distribution (Test)",
                xaxis=dict(title="Residual"),
                yaxis=dict(title="Count"),
            ),
            bargap=0.02,
        )
        self.plots["residual_distribution_test"] = fig

    def plot_qq_test(self) -> None:
        """Plot the Q-Q plot for **test** residuals with 95 % CI envelope."""
        res_data = self._get_res_data()
        qq_osm = res_data.get("qq_osm")
        qq_osr = res_data.get("qq_osr")
        qq_slope = res_data.get("qq_slope")
        qq_intercept = res_data.get("qq_intercept")
        qq_ci_lower = res_data.get("qq_ci_lower")
        qq_ci_upper = res_data.get("qq_ci_upper")

        try:
            if (
                qq_osm is None
                or qq_osr is None
                or qq_slope is None
                or qq_intercept is None
            ):
                return

            ref_line = qq_intercept + qq_slope * qq_osm
            dist = np.abs(qq_osr - ref_line)
            max_dist = float(np.max(dist)) if np.max(dist) != 0 else 1.0
            norm_dist = np.clip(dist / max_dist, 0.0, 1.0)

            point_colors = [_plasma_color(v) for v in norm_dist.tolist()]

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
                    title="Q-Q Plot of Residuals (Test)",
                    xaxis=dict(title="Theoretical Quantiles"),
                    yaxis=dict(title="Sample Quantiles"),
                )
            )
            self.plots["qq_test"] = fig
        except Exception:
            pass

    def plot_residual_outliers_test(self) -> None:
        """Plot significant residual outliers as box+strip charts for **test** data."""
        analysis = self.results.get("residual_outlier_analysis")
        if not analysis or not isinstance(analysis, dict):
            return

        res_df = analysis.get("results_df")
        threshold_val = analysis.get("threshold", np.nan)
        if threshold_val is None or np.isnan(threshold_val):
            return

        threshold = float(threshold_val)
        if res_df is None or getattr(res_df, "empty", True):
            return

        res_data = self._get_res_data()
        X_test = res_data.get("X_test")
        residuals = res_data.get("residuals")

        if X_test is None or residuals is None:
            return

        features_to_plot = res_df.loc[res_df["significant"], "feature"].tolist()
        max_plots = 10
        if len(features_to_plot) > max_plots:
            sort_col = "adj_p_value" if "adj_p_value" in res_df.columns else "p_value"
            features_to_plot = (
                res_df.sort_values(sort_col)
                .loc[res_df["significant"], "feature"]
                .head(max_plots)
                .tolist()
            )

        created = []
        errors: dict = {}
        X_test_local = X_test.copy()

        rmse = np.sqrt(np.nanmean(residuals**2))
        std_resid = residuals / (rmse if rmse != 0 else 1.0)
        abs_std_resid = np.abs(std_resid)

        if len(abs_std_resid) > len(X_test_local):
            abs_std_resid = abs_std_resid[: len(X_test_local)]
        elif len(abs_std_resid) < len(X_test_local):
            X_test_local = X_test_local.iloc[: len(abs_std_resid)].copy()

        mask_high = pd.Series(abs_std_resid > threshold, index=X_test_local.index)

        for col in features_to_plot:
            try:
                if col not in X_test_local.columns or not np.issubdtype(
                    X_test_local[col].dtype, np.number
                ):
                    continue

                plot_df = pd.DataFrame(
                    {
                        col: X_test_local[col],
                        "Residual_Group": np.where(
                            mask_high, "High Residual", "Low Residual"
                        ),
                    }
                ).dropna()

                if plot_df.empty:
                    continue

                low_vals = plot_df.loc[plot_df["Residual_Group"] == "Low Residual", col]
                high_vals = plot_df.loc[
                    plot_df["Residual_Group"] == "High Residual", col
                ]

                fig = go.Figure()

                for vals, label, box_color, marker_color in [
                    (low_vals, "Low Residual", _ACCENT_BLUE, "rgba(59,130,246,0.35)"),
                    (high_vals, "High Residual", _ACCENT_RED, "rgba(239,68,68,0.35)"),
                ]:
                    if vals.empty:
                        continue

                    # Convert hex to a list of RGB integers, then add alpha
                    rgb = [int(box_color.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4)]
                    rgba_str = f"rgba{tuple(rgb + [0.12])}"

                    fig.add_trace(
                        go.Box(
                            y=vals.tolist(),
                            name=label,
                            boxmean="sd",
                            marker=dict(color=box_color, size=4, opacity=0.6),
                            line=dict(color=box_color, width=1.5),
                            fillcolor=rgba_str,
                            whiskerwidth=0.6,
                            hovertemplate=(
                                f"<b>{label}</b><br>{col}: %{{y:.4g}}<extra></extra>"
                            ),
                        )
                    )

                    x_positions = [label] * len(vals)
                    fig.add_trace(
                        go.Scatter(
                            x=x_positions,
                            y=vals.tolist(),
                            mode="markers",
                            name=f"{label} (points)",
                            marker=dict(color=marker_color, size=3.5),
                            showlegend=False,
                            hoverinfo="skip",
                        )
                    )

                fig.update_layout(
                    **_layout(
                        title=f"<b>{col}</b> — Residual Groups (Test)",
                        xaxis=dict(title="Residual Group"),
                        yaxis=dict(title=col),
                    ),
                    boxmode="group",
                    showlegend=True,
                    legend=dict(
                        bgcolor="rgba(255,255,255,0.9)",
                        bordercolor=_BORDER,
                        borderwidth=1,
                        font=dict(color=_FONT_COLOR, size=12),
                    ),
                )

                key = f"residual_outlier_{col}_test"
                self.plots[key] = fig
                created.append(key)

            except Exception as e_col:
                errors[col] = str(e_col)

        self.results["residual_outlier_plots"] = created
        if errors:
            self.results["residual_outlier_plot_errors"] = errors

    # ======================================================================
    #  TRAINING-DATA DIAGNOSTIC PLOTS
    # ======================================================================

    def plot_posterior_predictive_train(self) -> None:
        """Compare KDE of actual y_train vs. predicted y_pred_train."""
        td = self._get_train_data()
        y_train = td.get("y_train")
        y_pred_train = td.get("y_pred_train")

        if y_train is None or y_pred_train is None:
            return

        # Compute KDE
        x_min = float(min(np.min(y_train), np.min(y_pred_train)))
        x_max = float(max(np.max(y_train), np.max(y_pred_train)))
        x_grid = np.linspace(x_min, x_max, 300)

        try:
            kde_actual = scipy_stats.gaussian_kde(y_train)(x_grid)
            kde_pred = scipy_stats.gaussian_kde(y_pred_train)(x_grid)
        except Exception:
            return

        fig = go.Figure()

        # Actual density
        fig.add_trace(
            go.Scatter(
                x=x_grid.tolist(),
                y=kde_actual.tolist(),
                mode="lines",
                name="y_train (actual)",
                line=dict(color=_ACCENT_BLUE, width=2.5),
                hovertemplate="Value: <b>%{x:.4g}</b><br>Density: <b>%{y:.4g}</b><extra></extra>",
            )
        )

        # Predicted density
        fig.add_trace(
            go.Scatter(
                x=x_grid.tolist(),
                y=kde_pred.tolist(),
                mode="lines",
                name="y_pred_train (predicted)",
                line=dict(color=_ACCENT_RED, width=2, dash="dash"),
                hovertemplate="Value: <b>%{x:.4g}</b><br>Density: <b>%{y:.4g}</b><extra></extra>",
            )
        )

        # Shaded area between curves
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([x_grid, x_grid[::-1]]).tolist(),
                y=np.concatenate([kde_actual, kde_pred[::-1]]).tolist(),
                fill="toself",
                fillcolor="rgba(59,130,246,0.08)",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
                name="Discrepancy",
            )
        )

        fig.update_layout(
            **_layout(
                title="Posterior Predictive Check (Train)",
                xaxis=dict(title="Target Value"),
                yaxis=dict(title="Density"),
            ),
            legend=dict(
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor=_BORDER,
                borderwidth=1,
            ),
        )
        self.plots["posterior_predictive_train"] = fig

    def plot_linearity_train(self) -> None:
        """Residuals vs. Fitted values with LOWESS + 95 % CI (training data)."""
        td = self._get_train_data()
        y_pred_train = td.get("y_pred_train")
        residuals = td.get("residuals")
        lowess = td.get("linearity_lowess")

        if y_pred_train is None or residuals is None:
            return

        abs_res = np.abs(residuals)
        vmax = float(np.percentile(abs_res, 97))
        marker_c = np.clip(abs_res, 0, vmax)

        fig = go.Figure()

        # Zero line
        fig.add_hline(y=0, line=dict(color="rgba(0,0,0,0.2)", dash="dash", width=1.5))

        # Scatter
        fig.add_trace(
            go.Scatter(
                x=y_pred_train.tolist(),
                y=residuals.tolist(),
                mode="markers",
                name="Residuals",
                marker=dict(
                    color=marker_c.tolist(),
                    colorscale=_COLORSCALE,
                    cmin=0,
                    cmax=vmax,
                    size=5,
                    opacity=0.7,
                    colorbar=dict(title="|Residual|", **_COLORBAR_STYLE),
                    line=dict(width=0),
                ),
                hovertemplate=(
                    "Fitted: <b>%{x:.4g}</b><br>"
                    "Residual: <b>%{y:.4g}</b><extra></extra>"
                ),
            )
        )

        # LOWESS + CI
        if lowess is not None:
            _add_lowess_with_ci(fig, lowess)

        fig.update_layout(
            **_layout(
                title="Linearity — Residuals vs. Fitted (Train)",
                xaxis=dict(title="Fitted Values"),
                yaxis=dict(title="Residuals"),
            )
        )
        self.plots["linearity_train"] = fig

    def plot_scale_location_train(self) -> None:
        """Scale-Location plot: sqrt|std residuals| vs. fitted (training data)."""
        td = self._get_train_data()
        y_pred_train = td.get("y_pred_train")
        sqrt_abs_std = td.get("sqrt_abs_std_resid")
        lowess = td.get("scale_loc_lowess")

        if y_pred_train is None or sqrt_abs_std is None:
            return

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=y_pred_train.tolist(),
                y=sqrt_abs_std.tolist(),
                mode="markers",
                name="√|Std Residuals|",
                marker=dict(
                    color=_ACCENT_BLUE,
                    size=5,
                    opacity=0.55,
                    line=dict(width=0),
                ),
                hovertemplate=(
                    "Fitted: <b>%{x:.4g}</b><br>"
                    "√|Std Resid|: <b>%{y:.4g}</b><extra></extra>"
                ),
            )
        )

        if lowess is not None:
            _add_lowess_with_ci(fig, lowess)

        fig.update_layout(
            **_layout(
                title="Scale-Location — Homogeneity of Variance (Train)",
                xaxis=dict(title="Fitted Values"),
                yaxis=dict(title="√|Standardized Residuals|"),
            )
        )
        self.plots["scale_location_train"] = fig

    def plot_leverage_train(self) -> None:
        """Residuals vs. Leverage with Cook's Distance contours (training data)."""
        td = self._get_train_data()
        leverage = td.get("leverage")
        std_resid = td.get("std_residuals")
        cooks_d = td.get("cooks_distance")
        lowess = td.get("leverage_lowess")

        if leverage is None or std_resid is None or cooks_d is None:
            return

        X_train = td.get("X_train")
        p = X_train.shape[1] + 1 if X_train is not None else 2  # +1 for intercept

        fig = go.Figure()

        # ---- Cook's Distance contour lines ----
        h_grid = np.linspace(0.001, float(np.nanmax(leverage)) * 1.1, 200)
        for cd_threshold, dash_style, label in [
            (0.5, "dash", "Cook's D = 0.5"),
            (1.0, "dot", "Cook's D = 1.0"),
        ]:
            # From cook's formula: D = (r^2 * h) / (p * (1-h)^2)
            # Solve for r: r = ±sqrt(D * p * (1-h)^2 / h)
            inner = cd_threshold * p * (1 - h_grid) ** 2 / h_grid
            inner = np.where(inner >= 0, inner, np.nan)
            r_pos = np.sqrt(inner)
            r_neg = -r_pos

            fig.add_trace(
                go.Scatter(
                    x=h_grid.tolist(),
                    y=r_pos.tolist(),
                    mode="lines",
                    line=dict(color="rgba(239,68,68,0.4)", dash=dash_style, width=1.2),
                    name=label,
                    hoverinfo="skip",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=h_grid.tolist(),
                    y=r_neg.tolist(),
                    mode="lines",
                    line=dict(color="rgba(239,68,68,0.4)", dash=dash_style, width=1.2),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

        # Zero line
        fig.add_hline(y=0, line=dict(color="rgba(0,0,0,0.15)", dash="dash", width=1))

        # Color by Cook's D: highlight influential points
        is_influential = cooks_d > 1.0
        colors = np.where(is_influential, _ACCENT_RED, _ACCENT_BLUE)

        fig.add_trace(
            go.Scatter(
                x=leverage.tolist(),
                y=std_resid.tolist(),
                mode="markers",
                name="Observations",
                marker=dict(
                    color=colors.tolist(),
                    size=5,
                    opacity=0.65,
                    line=dict(width=0),
                ),
                hovertemplate=(
                    "Leverage: <b>%{x:.4g}</b><br>"
                    "Std Residual: <b>%{y:.4g}</b><extra></extra>"
                ),
            )
        )

        # LOWESS
        if lowess is not None:
            _add_lowess_with_ci(
                fig,
                lowess,
                line_color=_ACCENT_RED,
                fill_color="rgba(239,68,68,0.08)",
            )

        fig.update_layout(
            **_layout(
                title="Influential Observations — Residuals vs. Leverage (Train)",
                xaxis=dict(title="Leverage (Hat Value)"),
                yaxis=dict(title="Standardized Residuals"),
            ),
            legend=dict(
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor=_BORDER,
                borderwidth=1,
            ),
        )
        self.plots["leverage_train"] = fig

    def plot_vif_train(self) -> None:
        """Variance Inflation Factor bar chart with risk-zone shading (training data)."""
        td = self._get_train_data()
        vif_data = td.get("vif")

        if not vif_data:
            return

        # Sort descending
        sorted_items = sorted(vif_data.items(), key=lambda x: x[1], reverse=True)
        features = [item[0] for item in sorted_items]
        vif_values = [item[1] for item in sorted_items]

        # Color by risk zone
        bar_colors = []
        for v in vif_values:
            if np.isnan(v) or np.isinf(v):
                bar_colors.append(_FONT_MUTED)
            elif v > 10:
                bar_colors.append(_ACCENT_RED)
            elif v > 5:
                bar_colors.append(_ACCENT_AMBER)
            else:
                bar_colors.append(_ACCENT_GREEN)

        fig = go.Figure()

        # Risk zone background shading (horizontal bands)
        max_vif = max(v for v in vif_values if np.isfinite(v)) if vif_values else 15
        y_range_max = max(max_vif * 1.15, 12)

        # Green zone: 0–5
        fig.add_hrect(
            y0=0,
            y1=5,
            fillcolor="rgba(16,185,129,0.06)",
            line_width=0,
            annotation_text="Low (< 5)",
            annotation_position="right",
            annotation_font=dict(color=_ACCENT_GREEN, size=10),
        )
        # Amber zone: 5–10
        fig.add_hrect(
            y0=5,
            y1=10,
            fillcolor="rgba(245,158,11,0.06)",
            line_width=0,
            annotation_text="Moderate (5–10)",
            annotation_position="right",
            annotation_font=dict(color=_ACCENT_AMBER, size=10),
        )
        # Red zone: >10
        fig.add_hrect(
            y0=10,
            y1=y_range_max,
            fillcolor="rgba(239,68,68,0.06)",
            line_width=0,
            annotation_text="High (> 10)",
            annotation_position="right",
            annotation_font=dict(color=_ACCENT_RED, size=10),
        )

        # Reference lines
        fig.add_hline(y=5, line=dict(color=_ACCENT_AMBER, dash="dot", width=1))
        fig.add_hline(y=10, line=dict(color=_ACCENT_RED, dash="dot", width=1))

        # Bars
        fig.add_trace(
            go.Bar(
                x=features,
                y=vif_values,
                marker=dict(
                    color=bar_colors,
                    line=dict(color="rgba(0,0,0,0.1)", width=0.5),
                ),
                name="VIF",
                hovertemplate="<b>%{x}</b><br>VIF: <b>%{y:.2f}</b><extra></extra>",
                opacity=0.88,
            )
        )

        fig.update_layout(
            **_layout(
                title="Collinearity — Variance Inflation Factor (Train)",
                xaxis=dict(title="Feature", tickangle=-40),
                yaxis=dict(title="VIF", range=[0, y_range_max]),
            ),
            bargap=0.25,
        )
        self.plots["vif_train"] = fig

    def plot_qq_train(self) -> None:
        """Q-Q plot of standardized training residuals with 95 % CI envelope."""
        td = self._get_train_data()
        qq_osm = td.get("qq_osm")
        qq_osr = td.get("qq_osr")
        qq_slope = td.get("qq_slope")
        qq_intercept = td.get("qq_intercept")
        qq_ci_lower = td.get("qq_ci_lower")
        qq_ci_upper = td.get("qq_ci_upper")

        try:
            if (
                qq_osm is None
                or qq_osr is None
                or qq_slope is None
                or qq_intercept is None
            ):
                return

            ref_line = qq_intercept + qq_slope * qq_osm
            dist = np.abs(qq_osr - ref_line)
            max_dist = float(np.max(dist)) if np.max(dist) != 0 else 1.0
            norm_dist = np.clip(dist / max_dist, 0.0, 1.0)

            point_colors = [_plasma_color(v) for v in norm_dist.tolist()]

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
                    title="Q-Q Plot — Normality of Residuals (Train)",
                    xaxis=dict(title="Theoretical Quantiles"),
                    yaxis=dict(title="Sample Quantiles"),
                )
            )
            self.plots["qq_train"] = fig
        except Exception:
            pass

    def plot_scale_location_test(self) -> None:
        """Scale-Location Plot (Test): Check homoscedasticity.

        Plots sqrt|Standardized Residuals| vs Fitted values with LOWESS + CI.
        """
        res_data = self._get_res_data()
        if not res_data:
            return

        y_pred = res_data.get("y_pred")
        residuals = res_data.get("residuals")
        if y_pred is None or residuals is None:
            return

        rmse = np.sqrt(np.nanmean(residuals**2))
        std_resid = residuals / rmse if rmse != 0 else residuals
        sqrt_abs = np.sqrt(np.abs(std_resid))

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=y_pred.tolist(),
                y=sqrt_abs.tolist(),
                mode="markers",
                name="Residuals",
                marker=dict(color=_ACCENT_BLUE, size=6, opacity=0.4),
                hovertemplate=(
                    "Predicted: <b>%{x:.4g}</b><br>"
                    "√|Std Resid|: <b>%{y:.4g}</b><extra></extra>"
                ),
            )
        )

        fig.update_layout(
            **_layout(
                title="Scale-Location (Test)",
                xaxis=dict(title="Predicted values"),
                yaxis=dict(title="√|Standardized Residuals|"),
            )
        )

        lowess = res_data.get("scale_loc_lowess")
        if lowess:
            _add_lowess_with_ci(fig, lowess)

        self.plots["scale_location_test"] = fig

    # ======================================================================
    #  RUN ALL
    # ======================================================================

    def run_all(self) -> None:
        """Generate all regression plots and store them in the ``plots`` dictionary."""
        self.plots = {}

        # ---- Test-data plots ----
        self.plot_metrics_table()
        self.plot_actual_vs_predicted_test()
        self.plot_residuals_vs_actual_test()
        self.plot_residuals_vs_predicted_test()
        self.plot_scale_location_test()
        self.plot_residual_distribution_test()
        self.plot_qq_test()
        self.plot_residual_outliers_test()

        # ---- Training-data diagnostic plots ----
        self.plot_posterior_predictive_train()
        self.plot_linearity_train()
        self.plot_scale_location_train()
        self.plot_leverage_train()
        self.plot_vif_train()
        self.plot_qq_train()
