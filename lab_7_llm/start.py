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

    importer = RawDataImporter(settings.parameters.dataset)
    importer.obtain()

    preprocessor = RawDataPreprocessor(importer.raw_data)
    analysis = preprocessor.analyze()
    print(analysis)

    preprocessor.transform()
    dataset = TaskDataset(preprocessor.data.head(100))

    pipeline = LLMPipeline(
        settings.parameters.model,
        dataset,
        max_length=120,
        batch_size=64,
        device="cpu"
    )
    model_summary = pipeline.analyze_model()
    print(model_summary)

    infer_data = pipeline.infer_dataset()

    predictions = PROJECT_ROOT / "lab_7_llm" / "dist" / "predictions.csv"
    predictions.parent.mkdir(exist_ok=True)
    infer_data.to_csv(predictions)

    evaluator = TaskEvaluator(predictions, settings.parameters.metrics)

    result = evaluator.run()
    print(result)

    assert result is not None, "Demo does not work correctly"


if __name__ == "__main__":
    main()
