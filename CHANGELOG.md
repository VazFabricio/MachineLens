# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.1.0] — 2026-08-12

### Added

#### Core
- `ModelInterface` — validated wrapper for fitted scikit-learn estimators.
  Accepts `X_train`, `X_test`, `y_train`, `y_test` (and optional `y_pred`) as
  `pandas.DataFrame`/`Series` or `numpy.ndarray`, detects the problem type
  automatically (classification vs regression), and exposes rich model metadata
  (algorithm family, feature importance availability, coefficient availability, etc.).

#### Diagnostics
- `ModelAnalyzer` — orchestrates the full diagnostic pipeline and returns a
  `DiagnosticResults` dataclass with all computed metrics.
- `ClassificationAnalyzer` — computes accuracy, precision, recall, F1, ROC-AUC,
  confusion matrix, classification report, calibration data, learning curves,
  cross-validation scores, and SHAP values.
- `RegressionAnalyzer` — computes RMSE, MAE, R², adjusted R², residual analysis,
  Q-Q plot data, partial dependence, learning curves, cross-validation scores,
  and SHAP values.
- `ShapCalculator` — unified SHAP computation supporting Tree, Linear, Kernel,
  and Gradient explainers with automatic fallback.

#### Visualisation
- `DiagnosticPlotter` — Plotly-based plotter with dedicated methods for every
  computed diagnostic:
  - Classification: `plot_metrics`, `plot_confusion_matrix`, `plot_roc_curve`,
    `plot_pr_curve`, `plot_calibration_curve`, `plot_threshold_analysis`,
    `plot_class_distribution`, `plot_probability_distribution`, `plot_misclassification_features`.
  - Regression: `plot_metrics`, `plot_residuals`, `plot_qq`,
    `plot_scale_location`, `plot_leverage`, `plot_residual_distribution`,
    `plot_residuals_vs_actual`, `plot_posterior_predictive`, `plot_outliers`.

#### Dashboard
- Standalone HTML/JS/CSS dashboard (`dashboard/`) for interactively exploring
  `DiagnosticResults` without requiring a Python runtime.

#### Documentation
- Sphinx documentation with `pydata-sphinx-theme`, covering installation,
  quick-start guides (classification & regression), and full API reference.

#### Quality & Tooling
- Full test suite with `pytest` and `pytest-cov` covering `ModelInterface`,
  `ClassificationAnalyzer`, `RegressionAnalyzer`, `ClassificationPlotter`, and
  `RegressionPlotter`.
- `ruff` for linting and formatting (`numpy` docstring convention).
- `mypy` for static type checking.
- `pre-commit` hooks for trailing whitespace, end-of-file fixer, YAML validation,
  large-file guard, ruff, and mypy.
- GitHub Actions CI matrix across Python 3.11, 3.12, and 3.13.

### Dependencies

- Python ≥ 3.11
- numpy ≥ 2.2.6
- pandas ≥ 2.3.3
- plotly ≥ 6.3.1
- scikit-learn ≥ 1.7.2
- scipy ≥ 1.11.0
- shap ≥ 0.49.1
- statsmodels ≥ 0.14.6

[0.1.0]: https://github.com/VazFabricio/MachineLens/releases/tag/v0.1.0
