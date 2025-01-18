"""
Starter for demonstration of laboratory work.
"""
# pylint: disable= too-many-locals, undefined-variable, unused-import
from pathlib import Path

from config.lab_settings import LabSettings

from lab_7_llm.main import (
    RawDataPreprocessor,
    RawDataImporter,
    report_time,
    TaskDataset,
    LLMPipeline,
    TaskEvaluator
)

@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    settings = LabSettings(Path(__file__).parent / "settings.json")
    importer = RawDataImporter(settings.parameters.dataset)
    importer.obtain()
    preprocessor = RawDataPreprocessor(importer.raw_data)

    analysis = preprocessor.analyze()
    preprocessor.transform()
    dataset = TaskDataset(preprocessor.data.head(100))

    pipeline = LLMPipeline(settings.parameters.model,
                           dataset,
                           max_length=120,
                           batch_size=1,
                           device="cpu")

    pipeline.analyze_model()
    df = pipeline.infer_dataset()
    df.to_csv(Path(__file__).parent / "predictions.csv")

    evaluator = TaskEvaluator(Path(__file__).parent / "predictions.csv",
                              settings.parameters.metrics)

    result = evaluator.run()
    assert result is not None, "Demo does not work correctly"


if __name__ == "__main__":
    main()
