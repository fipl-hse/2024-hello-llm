"""
Starter for demonstration of laboratory work.
"""
# pylint: disable= too-many-locals, undefined-variable, unused-import
import json
from pathlib import Path

from config.lab_settings import LabSettings
from lab_7_llm.main import (
    LLMPipeline,
    RawDataImporter,
    RawDataPreprocessor,
    report_time,
    TaskDataset,
)


@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    result = None
    settings = LabSettings(Path(__file__).parent / 'settings.json')
    importer = RawDataImporter(settings.parameters.dataset)
    importer.obtain()

    preprocessor = RawDataPreprocessor(importer.raw_data)
    analysis = preprocessor.analyze()
    preprocessor.transform()

    dataset = TaskDataset(preprocessor.data.head(100))
    pipeline = LLMPipeline(settings.parameters.model, dataset, max_length=120, batch_size=1, device="cpu")

    model_properties = pipeline.analyze_model()
    prediction = pipeline.infer_sample(dataset[2])
    print(prediction)
    result = prediction


    assert result is not None, "Demo does not work correctly"


if __name__ == "__main__":
    main()