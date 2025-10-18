class ModelInterface:
    def __init__(
        self, model, X_train, X_test, y_train, y_test, y_pred, problem_type="regression"
    ):
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.y_pred = y_pred
        self.problem_type = problem_type
