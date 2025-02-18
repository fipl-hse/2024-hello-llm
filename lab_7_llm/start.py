"""
Starter for demonstration of laboratory work.
"""
# pylint: disable= too-many-locals, undefined-variable, unused-import
from pathlib import Path

from sphinx.addnodes import index

from config.constants import PROJECT_ROOT
from config.lab_settings import LabSettings
from core_utils.llm.time_decorator import report_time
from lab_7_llm.main import LLMPipeline, RawDataImporter, RawDataPreprocessor, TaskDataset, TaskEvaluator


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
    batch_size = 64
    max_length = 120

    pipeline = LLMPipeline(settings.parameters.model, dataset, max_length, batch_size, device)
    model_analysis = pipeline.analyze_model()
    sample_inference = pipeline.infer_sample(dataset[0])

    predictions_path = PROJECT_ROOT/'lab_7_llm'/'dist'/'predictions.csv'
    predictions_path.parent.mkdir(exist_ok=True)
    dataset_inference = pipeline.infer_dataset()
    dataset_inference.to_csv(predictions_path, index=False)

    evaluator = TaskEvaluator(predictions_path, settings.parameters.metrics)
    metric = evaluator.run()

    result = metric
    assert result is not None, "Demo does not work correctly"

if __name__ == "__main__":
    main()

