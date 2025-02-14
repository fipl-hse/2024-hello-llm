"""
Starter for demonstration of laboratory work.
"""
# pylint: disable= too-many-locals, undefined-variable, unused-import
from pathlib import Path

from config.constants import PROJECT_ROOT
from config.lab_settings import LabSettings
from lab_7_llm.main import LLMPipeline, RawDataImporter, RawDataPreprocessor, report_time, TaskDataset


@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """

    settings = LabSettings(PROJECT_ROOT/'lab_7_llm'/'settings.json')

    importer = RawDataImporter(settings.parameters.dataset)
    importer.obtain()

    preprocessor = RawDataPreprocessor(importer.raw_data)
    properties = preprocessor.analyze()
    preprocessor.transform()

    dataset = TaskDataset(preprocessor.data.head(100))

    device = "cpu"
    batch_size = 1
    max_length = 120

    pipeline = LLMPipeline(settings.parameters.model, dataset, max_length, batch_size, device)
    model_analysis = pipeline.analyze_model()
    sample_inference = pipeline.infer_sample(dataset[0])

    result = sample_inference
    assert result is not None, "Demo does not work correctly"

if __name__ == "__main__":
    main()

