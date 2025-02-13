"""
Starter for demonstration of laboratory work.
"""
# pylint: disable= too-many-locals, undefined-variable, unused-import
from pathlib import Path

from config.constants import PROJECT_ROOT
from config.lab_settings import LabSettings
from core_utils.llm.raw_data_preprocessor import ColumnNames
from lab_7_llm.main import RawDataImporter, report_time, RawDataPreprocessor, TaskDataset, LLMPipeline


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
    sample_inference = pipeline.infer_dataset(dataset[0])

    # dataset_inference = pipeline.infer_dataset()
    # dataset_inference.to_csv('predictions.csv', columns=ColumnNames.PREDICTION.value)

    result = sample_inference
    assert result is not None, "Demo does not work correctly"

if __name__ == "__main__":
    main()

