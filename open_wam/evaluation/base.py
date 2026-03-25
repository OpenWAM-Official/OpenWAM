"""Abstract base class for evaluators."""

from abc import ABC, abstractmethod

from open_wam.inference.base import BaseInferenceEngine


class BaseEvaluator(ABC):
    """Evaluator base class.

    Concrete evaluators implement ``evaluate`` which runs the policy in an
    environment or on an offline dataset and returns metrics.
    """

    def __init__(self, cfg, engine: BaseInferenceEngine):
        self.cfg = cfg
        self.engine = engine

    @abstractmethod
    def evaluate(self, dataset_or_env) -> dict:
        """Run evaluation and return metrics dict."""
        ...
