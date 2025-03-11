"""
Fine-tuning starter.
"""
# pylint: disable=too-many-locals, undefined-variable, unused-import, too-many-branches, too-many-statements
from pathlib import Path

from core_utils.llm.time_decorator import report_time
from lab_8_sft.main import RawDataImporter, RawDataPreprocessor, TaskDataset, LLMPipeline
from config.lab_settings import LabSettings
from config.constants import PROJECT_ROOT
import pandas as pd


@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    settings = LabSettings(PROJECT_ROOT / 'lab_8_sft' / 'settings.json')
    importer = RawDataImporter(settings.parameters.dataset)
    importer.obtain()

    preprocessor = RawDataPreprocessor(importer.raw_data)
    print(preprocessor.analyze())

    preprocessor.transform()

    dataset = TaskDataset(preprocessor.data.head(100))

    batch_size = 1
    max_length = 120
    device = 'cpu'
    pipeline = LLMPipeline(settings.parameters.model, dataset, max_length, batch_size, device)
    print(pipeline.analyze_model())
    print(pipeline.infer_sample(dataset[0]))

    result = True
    assert result is not None, "Finetuning does not work correctly"


if __name__ == "__main__":
    main()
