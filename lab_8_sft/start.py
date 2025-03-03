"""
Fine-tuning starter.
"""
# pylint: disable=too-many-locals, undefined-variable, unused-import, too-many-branches, too-many-statements
from pathlib import Path

from transformers import AutoTokenizer

from config.constants import PROJECT_ROOT
from config.lab_settings import LabSettings, SFTParams
from core_utils.llm.time_decorator import report_time
from lab_8_sft.main import (
    LLMPipeline,
    RawDataImporter,
    RawDataPreprocessor,
    SFTPipeline,
    TaskDataset,
    TaskEvaluator,
    TokenizedTaskDataset,
)

@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    settings = LabSettings(PROJECT_ROOT / "lab_8_sft" / "settings.json")

    importer = RawDataImporter(settings.parameters.dataset)
    importer.obtain()
    if importer.raw_data is None:
        return

    preprocessor = RawDataPreprocessor(importer.raw_data)
    analysis = preprocessor.analyze()

    result = analysis
    assert result is not None, "Finetuning does not work correctly"


if __name__ == "__main__":
    main()
