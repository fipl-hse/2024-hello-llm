"""
Module with description of abstract task evaluator.
"""

# pylint: disable=too-few-public-methods, duplicate-code
from abc import ABC, abstractmethod
from typing import Iterable

from core_utils.llm.metrics import Metrics


class AbstractTaskEvaluator(ABC):
    """
    Abstract Task Evaluator.
    """

    def __init__(self, metrics: Iterable[Metrics]) -> None:
        """
        Initialize an instance of AbstractTaskEvaluator.

        Args:
            metrics (Iterable[Metrics]): List of metrics to check
        """
        self._metrics = metrics

    @abstractmethod
    def run(self) -> dict | None:
        """
        Evaluate the predictions against the references using the specified metric.

        Returns:
            dict | None: A dictionary containing information about the calculated metric
        """
