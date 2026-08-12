Quick Start
===========

This guide will walk you through the basic usage of MachineLens for both classification and regression models.

1. Installation
---------------

Install MachineLens using pip or uv:

.. code-block:: bash

    uv pip install machinelens

2. Diagnosing a Classification Model
------------------------------------

Here is a simple example of how to use MachineLens to diagnose a classification model:

.. code-block:: python

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    from machinelens.core import ModelInterface
    from machinelens.analyzer import ModelAnalyzer
    from machinelens.plots import DiagnosticPlotter

    # 1. Prepare data and train a model
    X, y = make_classification(n_samples=1000, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = RandomForestClassifier(random_state=42).fit(X_train, y_train)

    # 2. Initialize the Model Interface
    interface = ModelInterface(model, X_train, X_test, y_train, y_test)

    # 3. Analyze the model
    analyzer = ModelAnalyzer(interface)
    results = analyzer.analyze()

    # 4. Plot diagnostics
    plotter = DiagnosticPlotter(results)
    fig = plotter.plot_metrics()
    fig.show()

3. Diagnosing a Regression Model
--------------------------------

MachineLens automatically detects the problem type. Here is an example for a regression task:

.. code-block:: python

    import pandas as pd
    from sklearn.datasets import make_regression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split

    from machinelens.core import ModelInterface
    from machinelens.analyzer import ModelAnalyzer
    from machinelens.plots import DiagnosticPlotter

    # 1. Create a synthetic regression dataset
    X_reg, y_reg = make_regression(
        n_samples=500, n_features=20, n_informative=4, noise=15.0, random_state=42
    )
    feature_names = [f"Feature_{i+1}" for i in range(X_reg.shape[1])]
    X_reg_df = pd.DataFrame(X_reg, columns=feature_names)

    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X_reg_df, y_reg, test_size=0.2, random_state=42
    )

    # 2. Train the model
    reg_model = RandomForestRegressor(n_estimators=50, random_state=42)
    reg_model.fit(X_train_reg, y_train_reg)

    # 3. Wrap with ModelInterface and run ModelAnalyzer
    reg_interface = ModelInterface(
        model=reg_model,
        X_train=X_train_reg,
        X_test=X_test_reg,
        y_train=y_train_reg,
        y_test=y_test_reg
    )
    reg_analyzer = ModelAnalyzer(reg_interface)
    reg_results = reg_analyzer.analyze()

    # 4. Plot diagnostics
    reg_plotter = DiagnosticPlotter(reg_results)
    fig = reg_plotter.plot_metrics()
    fig.show()

4. Exporting to the Web Dashboard
---------------------------------

MachineLens includes a standalone glassmorphic Web Dashboard that allows you to drag, reorder, resize, and inspect pre-computed plot bundles in real-time.

To display your model's diagnostic results on the web dashboard, save the bundle:

.. code-block:: python

    plotter = DiagnosticPlotter(results)
    plotter.save_dashboard_bundle("dashboard/active_bundle.json")

**How to Run the Web Dashboard:**

1. **Start the local HTTP server** (in your terminal from the project root):

.. code-block:: bash

    python -m http.server 8000

2. **Open your browser**:
   Navigate to http://localhost:8000/dashboard/
