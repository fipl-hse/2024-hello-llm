"""
Checks that the model is being analyzed correctly
"""
import unittest
from pathlib import Path

import pytest

from admin_utils.reference_scores import ReferenceAnalysisScores, ReferenceAnalysisScoresType
from config.lab_settings import LabSettings
from core_utils.llm.llm_pipeline import AbstractLLMPipeline
from lab_7_llm.main import LLMPipeline


def run_model_analysis_check(lab_path: Path, pipeline_class: type[AbstractLLMPipeline]) -> None:
    """
    Evaluate metrics from a lab.

    Args:
         lab_path (Path): path to lab
         pipeline_class (type[AbstractLLMPipeline]): pipeline class
    """

    settings = LabSettings(lab_path / "settings.json")
    device = "cpu"
    batch_size = 64
    max_length = 120

    pipeline = pipeline_class(settings.parameters.model, None, max_length, batch_size, device)
    model_analysis = pipeline.analyze_model()
    references = ReferenceAnalysisScores(scores_type=ReferenceAnalysisScoresType.MODEL)

    model_name = settings.parameters.model.replace("test_", "")

    if references.get(model_name) != model_analysis:
        assert False, f"Model {settings.parameters.model} analysis is incorrect"


class ModelWorkingTest(unittest.TestCase):
    """
    Tests analyse function
    """

    @pytest.mark.lab_7_llm
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_analyze_ideal(self) -> None:
        """
        Ideal analyze scenario
        """
        self.assertIsNone(run_model_analysis_check(Path(__file__).parent.parent, LLMPipeline))
