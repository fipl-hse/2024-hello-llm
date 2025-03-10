"""
Checks the inference of the model
"""
# pylint: disable=R0801
import unittest
from pathlib import Path

import pytest

from admin_utils.reference_scores import ReferenceAnalysisScores, ReferenceAnalysisScoresType
from config.lab_settings import LabSettings
from core_utils.llm.llm_pipeline import AbstractLLMPipeline
from lab_7_llm.main import LLMPipeline


def run_model_inference_check(lab_path: Path, pipeline_class: type[AbstractLLMPipeline]) -> None:
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
    references = ReferenceAnalysisScores(scores_type=ReferenceAnalysisScoresType.INFERENCE)
    model_name = settings.parameters.model.replace("test_", "")
    samples = references.get(model_name)
    for query, prediction in samples.items():
        if "[TEST SEP]" in query:
            first_value, second_value = query.split("[TEST SEP]")
            res = pipeline.infer_sample((first_value, second_value))
        else:
            res = pipeline.infer_sample((query,))
        if res != prediction:
            assert False, (
                f"Inference of {settings.parameters.model} model is incorrect.\n"
                f"Expected: {prediction}\n"
                f"Actual: {res}"
            )


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
        Ideal inference scenario
        """
        self.assertIsNone(run_model_inference_check(Path(__file__).parent.parent, LLMPipeline))
