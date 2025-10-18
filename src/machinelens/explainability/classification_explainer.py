from .base_explainer import BaseExplainer


class ClassificationExplainer(BaseExplainer):
    def feature_importance(self) -> None:
        pass

    def partial_dependence(self) -> None:
        pass

    def sensitivity_analysis(self) -> None:
        pass

    def fairness_bias(self) -> None:
        pass
