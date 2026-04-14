.. -- mode: rst --

|GitHubActions| |Ruff| |PythonVersion| |PyPI| |License|

.. |GitHubActions| image:: https://github.com/vazfabricio/machinelens/actions/workflows/ci.yml/badge.svg
   :target: https://github.com/vazfabricio/machinelens/actions/workflows/ci.yml

.. |Ruff| image:: https://img.shields.io/badge/code%20style-ruff-000000.svg
   :target: https://github.com/astral-sh/ruff

.. |PythonVersion| image:: https://img.shields.io/pypi/pyversions/machinelens.svg
   :target: https://pypi.org/project/machinelens/

.. |PyPI| image:: https://img.shields.io/pypi/v/machinelens.svg
   :target: https://pypi.org/project/machinelens/

.. |License| image:: https://img.shields.io/badge/License-BSD%203--Clause-blue.svg
   :target: https://opensource.org/licenses/BSD-3-Clause

MachineLens is a XAI Python tool for automated model diagnostics and explainability post training. It is specifically designed to work seamlessly with the scikit-learn ecosystem, streamlining the evaluation of machine learning models through comprehensive reports.

It provides a high-level interface to generate diagnostics for classification and regression tasks, supporting any estimator that follows the scikit-learn API.

Website: https://github.com/vazfabricio/machinelens
Documentation: https://github.com/vazfabricio/machinelens#documentation


Installation
------------

.. |PythonMinVersion| replace:: 3.11
.. |NumPyMinVersion| replace:: 2.2.6
.. |PandasMinVersion| replace:: 2.3.3
.. |MatplotlibMinVersion| replace:: 3.10.7
.. |ScikitLearnMinVersion| replace:: 1.7.2
.. |SeabornMinVersion| replace:: 0.13.2
.. |PlotlyMinVersion| replace:: 6.3.1
.. |StatsmodelsMinVersion| replace:: 0.14.6
.. |ShapMinVersion| replace:: 0.49.1
.. |LimeMinVersion| replace:: 0.2.0.1

Dependencies
~~~~~~~~~~~~

MachineLens requires:

- Python (>= |PythonMinVersion|)
**- Scikit-learn (>= |ScikitLearnMinVersion|)**
- NumPy (>= |NumPyMinVersion|)
- Pandas (>= |PandasMinVersion|)
- Matplotlib (>= |MatplotlibMinVersion|)
- Seaborn (>= |SeabornMinVersion|)
- Plotly (>= |PlotlyMinVersion|)
- Statsmodels (>= |StatsmodelsMinVersion|)
- SHAP (>= |ShapMinVersion|)
- LIME (>= |LimeMinVersion|)


User installation
-----------------

You can install MachineLens using pip::

    pip install machinelens

For a much faster installation, you can use `uv <https://github.com/astral-sh/uv>`_::

    uv pip install machinelens

Development
-----------

We welcome contributors! MachineLens is built with modern Python tooling to ensure code quality and ease of development.
The library implements a robust interface to ensure that any model adhering to the scikit-learn standard can be diagnosed without additional boilerplate.

Setting up the environment
~~~~~~~~~~~~~~~~~~~~~~~~~~

The project uses `uv <https://github.com/astral-sh/uv>`_ for dependency management. To set up your development environment:

1. Clone the repository:
   ::

       git clone https://github.com/vazfabricio/machinelens.git
       cd machinelens

2. Sync the dependencies and create a virtual environment:
   ::

       uv sync

This will automatically install all main dependencies and development tools like ``pytest``, ``ruff``, and ``mypy``.

Code Quality
~~~~~~~~~~~~

We use ``ruff`` for linting and formatting. Before submitting code, please ensure it follows our standards:

- **Linting & Formatting**: Run ``ruff check`` and ``ruff format``.
- **Type Checking**: Run ``mypy src`` to verify type hints.
- **Pre-commit**: We recommend installing the pre-commit hooks:
  ::

      pre-commit install

Testing
~~~~~~~

After installation, launch the test suite from the root directory to ensure everything is working correctly:

::

    pytest

The test suite includes coverage reports by default.


Important links
---------------

- Official source code repo: https://github.com/vazfabricio/machinelens
- Issue tracker: https://github.com/vazfabricio/machinelens/issues


Source code
~~~~~~~~~~~

You can clone the latest sources with::

    git clone https://github.com/vazfabricio/machinelens.git



Contributing
~~~~~~~~~~~~

Please check our development guidelines in the repository. We use ``ruff`` for linting and ``pre-commit`` to ensure code quality.


Help and Support
----------------

Communication
~~~~~~~~~~~~~

- **GitHub Discussions**: https://github.com/vazfabricio/machinelens/discussions
- **Issues**: https://github.com/vazfabricio/machinelens/issues


License
-------

MachineLens is distributed under the BSD 3-Clause License. See ``LICENSE`` for more information.
