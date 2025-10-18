import pandas as pd


class ModelInterface:
    def __init__(
        self,
        model: str,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.DataFrame,
        y_test: pd.DataFrame,
        y_pred: pd.DataFrame,
        problem_type: str = "regression",
    ) -> None:
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.y_pred = y_pred
        self.problem_type = problem_type
