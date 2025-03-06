"""
Fine-tuning starter.
"""
# pylint: disable=too-many-locals, undefined-variable, unused-import, too-many-branches, too-many-statements
from pathlib import Path

from config.constants import PROJECT_ROOT
from config.lab_settings import LabSettings
from core_utils.llm.time_decorator import report_time
from lab_8_sft.main import (
    LLMPipeline,
    RawDataImporter,
    RawDataPreprocessor,
    TaskDataset
)


@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    lab_settings = LabSettings(PROJECT_ROOT / 'lab_8_sft' / 'settings.json')
    importer = RawDataImporter(lab_settings.parameters.dataset)
    importer.obtain()

    if importer.raw_data is None:
        return

    preprocessor = RawDataPreprocessor(importer.raw_data)
    properties = preprocessor.analyze()
    print(properties)

    preprocessor.transform()
    dataset = TaskDataset(preprocessor.data.head(100))

    pipeline = LLMPipeline(lab_settings.parameters.model,
                           dataset,
                           max_length=120,
                           batch_size=64,
                           device="cpu")

    pipeline.analyze_model()

    dataframe = pipeline.infer_dataset()

    predictions_path = PROJECT_ROOT / "lab_8_sft" / "dist" / "predictions.csv"
    predictions_path.parent.mkdir(exist_ok=True)
    dataframe.to_csv(predictions_path)

    result = dataframe
    assert result is not None, "Finetuning does not work correctly"


if __name__ == "__main__":
    main()
