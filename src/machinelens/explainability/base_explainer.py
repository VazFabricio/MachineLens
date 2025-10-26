class BaseExplainer:
    def __init__(self, model_interface) -> None:
        self.model_interface = model_interface
        self.model = getattr(model_interface, "model", None)

    def feature_importance(self) -> None:
        raise NotImplementedError

    def partial_dependence(self) -> None:
        raise NotImplementedError

    def sensitivity_analysis(self) -> None:
        raise NotImplementedError

    def fairness_bias(self) -> None:
        raise NotImplementedError
