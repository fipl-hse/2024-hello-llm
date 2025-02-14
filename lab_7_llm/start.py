"""
Starter for demonstration of laboratory work.
"""

# pylint: disable= too-many-locals, undefined-variable, unused-import
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
    settings_path = PROJECT_ROOT / 'lab_7_llm' / 'settings.json'
    parameters = LabSettings(settings_path).parameters
    if not parameters.dataset:
        return

    importer = RawDataImporter(parameters.dataset)
    importer.obtain()

    preprocessor = RawDataPreprocessor(importer.raw_data)
    print(preprocessor.analyze())
    preprocessor.transform()

    dataset = TaskDataset(preprocessor.data.head(100))

    batch_size = 64
    max_length = 120
    device = 'cpu'

    pipeline = LLMPipeline(parameters.model, dataset, max_length, batch_size, device)
    print(pipeline.analyze_model())
    print(pipeline.infer_sample(dataset[22]))

    predictions_path = PROJECT_ROOT / 'lab_7_llm' / 'dist' / 'predictions.csv'
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    predictions = pipeline.infer_dataset()
    predictions.to_csv(predictions_path)

    evaluator = TaskEvaluator(predictions_path, parameters.metrics)
    comparison = evaluator.run()
    print(comparison)

    result = comparison
    assert result is not None, "Demo does not work correctly"


if __name__ == "__main__":
    main()
