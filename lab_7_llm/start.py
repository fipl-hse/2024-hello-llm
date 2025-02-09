"""
Starter for demonstration of laboratory work.
"""
# pylint: disable= too-many-locals, undefined-variable, unused-import
from pathlib import Path

from config.constants import PROJECT_ROOT
from config.lab_settings import LabSettings
from core_utils.llm.time_decorator import report_time
from lab_7_llm.main import (
    LLMPipeline,
    RawDataImporter,
    RawDataPreprocessor,
    TaskDataset,
    TaskEvaluator,
)


@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    settings = LabSettings(PROJECT_ROOT / 'lab_7_llm' / 'settings.json')
    predictions_path = PROJECT_ROOT / 'dist' / 'predictions.csv'

    importer = RawDataImporter(settings.parameters.dataset)
    importer.obtain()

    preprocessor = RawDataPreprocessor(importer.raw_data)
    key_properties = preprocessor.analyze()
    preprocessor.transform()

    dataset = TaskDataset(preprocessor.data.head(100))

    pipeline = LLMPipeline(settings.parameters.model, dataset, max_length=120, batch_size=1, device='cpu')
    model_properties = pipeline.analyze_model()
    sample_inference = pipeline.infer_sample(dataset[0])

    result = model_properties, sample_inference
    assert result is not None, "Demo does not work correctly"


if __name__ == "__main__":
    main()
