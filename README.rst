.. -- mode: rst --

|GitHubActions| |Ruff| |License| |Python|

.. |GitHubActions| image:: https://github.com/VazFabricio/MachineLens/actions/workflows/ci.yml/badge.svg
   :target: https://github.com/VazFabricio/MachineLens/actions/workflows/ci.yml
   :alt: CI Status

.. |Ruff| image:: https://img.shields.io/badge/code%20style-ruff-000000.svg
   :target: https://github.com/astral-sh/ruff
   :alt: Code style: Ruff

.. |License| image:: https://img.shields.io/badge/License-BSD%203--Clause-blue.svg
   :target: https://opensource.org/licenses/BSD-3-Clause
   :alt: License: BSD 3-Clause

.. |Python| image:: https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue.svg
   :target: https://www.python.org/
   :alt: Python 3.11 | 3.12 | 3.13

.. image:: https://raw.githubusercontent.com/VazFabricio/MachineLens/main/docs/images/MachineLensLogo.jpg
   :alt: MachineLens Logo
   :align: center
   :width: 200px

|

MachineLens is a XAI Python tool for automated model diagnostics and explainability post training.
It is specifically designed to work seamlessly with the scikit-learn ecosystem, streamlining the
evaluation of machine learning models through comprehensive reports.

It provides a high-level interface to generate diagnostics for classification and regression tasks,
supporting any estimator that follows the scikit-learn API.

- **Documentation**: https://vazfabricio.github.io/MachineLens/
- **Source code**: https://github.com/VazFabricio/MachineLens
- **Issue tracker**: https://github.com/VazFabricio/MachineLens/issues


Installation
------------

.. |PythonMinVersion| replace:: 3.11
.. |NumPyMinVersion| replace:: 2.2.6
.. |PandasMinVersion| replace:: 2.3.3
.. |ScikitLearnMinVersion| replace:: 1.7.2
.. |PlotlyMinVersion| replace:: 6.3.1
.. |StatsmodelsMinVersion| replace:: 0.14.6
.. |ShapMinVersion| replace:: 0.49.1

Dependencies
~~~~~~~~~~~~

MachineLens requires:

- Python (>= |PythonMinVersion|)
- Scikit-learn (>= |ScikitLearnMinVersion|)
- NumPy (>= |NumPyMinVersion|)
- Pandas (>= |PandasMinVersion|)
- Plotly (>= |PlotlyMinVersion|)
- Statsmodels (>= |StatsmodelsMinVersion|)
- SciPy (>= 1.11.0)
- SHAP (>= |ShapMinVersion|)


User installation
~~~~~~~~~~~~~~~~~

You can install MachineLens using ``pip``::

    pip install machinelens

For a faster installation, you can use `uv <https://github.com/astral-sh/uv>`_::

    uv pip install machinelens


Quick Start
-----------

**Classification model**

.. code-block:: python

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    import pandas as pd

    from machinelens import ModelInterface, ModelAnalyzer, DiagnosticPlotter

    # 1. Prepare data and train a model
    X, y = make_classification(n_samples=1000, random_state=42)
    X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=0.2)
    model = RandomForestClassifier(random_state=42).fit(X_train, y_train)

    # 2. Wrap the model
    interface = ModelInterface(model, X_train, X_test, y_train, y_test)

    # 3. Run diagnostics
    results = ModelAnalyzer(interface).analyze()

    # 4. Visualise
    plotter = DiagnosticPlotter(results)
    plotter.plot_metrics().show()
    plotter.plot_roc_curve().show()
    plotter.plot_confusion_matrix().show()

**Regression model**

.. code-block:: python

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split
    import pandas as pd

    from machinelens import ModelInterface, ModelAnalyzer, DiagnosticPlotter

    X, y = make_regression(n_samples=500, n_features=20, noise=15.0, random_state=42)
    X_df = pd.DataFrame(X, columns=[f"Feature_{i+1}" for i in range(X.shape[1])])
    X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=50, random_state=42).fit(X_train, y_train)
    interface = ModelInterface(model, X_train, X_test, y_train, y_test)
    results = ModelAnalyzer(interface).analyze()

    plotter = DiagnosticPlotter(results)
    plotter.plot_metrics().show()
    plotter.plot_residuals().show()
    plotter.plot_qq().show()


Development
-----------

We welcome contributions! MachineLens is built with modern Python tooling to ensure code
quality and ease of development.

Setting up the environment
~~~~~~~~~~~~~~~~~~~~~~~~~~

The project uses `uv <https://github.com/astral-sh/uv>`_ for dependency management:

1. Clone the repository::

       git clone https://github.com/VazFabricio/MachineLens.git
       cd MachineLens

2. Sync the dependencies and create a virtual environment::

       uv sync

This will automatically install all main dependencies and development tools like ``pytest``,
``ruff``, and ``mypy``.

Code Quality
~~~~~~~~~~~~

We use ``ruff`` for linting and formatting. Before submitting code, please ensure it follows
our standards:

- **Linting & Formatting**: Run ``ruff check`` and ``ruff format``.
- **Type Checking**: Run ``mypy src``.
- **Pre-commit**: Install the hooks with::

      pre-commit install

Testing
~~~~~~~

Run the test suite from the root directory::

    pytest

The test suite includes coverage reports by default.


Help and Support
----------------

Communication
~~~~~~~~~~~~~

- **GitHub Discussions**: https://github.com/VazFabricio/MachineLens/discussions
- **Issues**: https://github.com/VazFabricio/MachineLens/issues


License
-------

MachineLens is distributed under the BSD 3-Clause License. See ``LICENSE`` for more information.
