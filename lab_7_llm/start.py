"""
Starter for demonstration of laboratory work.
"""
# pylint: disable= too-many-locals, undefined-variable, unused-import
from pathlib import Path

from config.constants import PROJECT_ROOT
from config.lab_settings import LabSettings
from lab_7_llm.main import (
    LLMPipeline,
    RawDataImporter,
    RawDataPreprocessor,
    report_time,
    TaskDataset,
    TaskEvaluator,
)


@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    settings = LabSettings(PROJECT_ROOT / "lab_7_llm" / "settings.json")

    importer = RawDataImporter(settings.parameters.dataset)
    importer.obtain()
    if importer.raw_data is None:
        return

    preprocessor = RawDataPreprocessor(importer.raw_data)
    analysis = preprocessor.analyze()
    print(analysis)      # for mark 4

    preprocessor.transform()
    dataset = TaskDataset(preprocessor.data.head(100))
    pipeline = LLMPipeline(settings.parameters.model,
                           dataset,
                           max_length=120,
                           batch_size=1,
                           device='cpu')
    print(pipeline.analyze_model())
    sample1 = pipeline.infer_sample(dataset[0])
    print(sample1)          # for mark 6

    # for mark 8
    pipeline = LLMPipeline(settings.parameters.model,
                           dataset,
                           max_length=120,
                           batch_size=64,
                           device="cpu")

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
