"""
Fine-tuning starter.
"""
# pylint: disable=too-many-locals, undefined-variable, unused-import, too-many-branches, too-many-statements
from pathlib import Path

from config.lab_settings import LabSettings
from core_utils.llm.time_decorator import report_time
from lab_8_sft.main import (
    RawDataImporter,
    RawDataPreprocessor,
    TaskDataset,
    LLMPipeline,
)


@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    settings_path = Path(__file__).parent / 'settings.json'
    settings = LabSettings(settings_path)

    importer = RawDataImporter(settings.parameters.dataset)
    importer.obtain()

    if importer.raw_data is None:
        return None

    preprocessor = RawDataPreprocessor(importer.raw_data)
    preprocessor.transform()

    batch_size = 1
    max_length = 120
    device = 'cpu'

    dataset = TaskDataset(preprocessor.data.head(100))
    pipeline = LLMPipeline(settings.parameters.model,
                           dataset,
                           max_length,
                           batch_size,
                           device)

    _sample_prediction = pipeline.infer_sample(dataset[0])
    predictions = pipeline.infer_dataset()

    print(_sample_prediction)
    result = predictions
    assert result is not None, "Finetuning does not work correctly"


if __name__ == "__main__":
    main()
