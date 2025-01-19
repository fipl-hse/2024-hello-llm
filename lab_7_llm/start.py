"""
Starter for demonstration of laboratory work.
"""
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

# pylint: disable= too-many-locals, undefined-variable, unused-import

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
    print(analysis)

    preprocessor.transform()
    dataset = TaskDataset(preprocessor.data.head(100))

    pipeline = LLMPipeline(settings.parameters.model,
                           dataset,
                           max_length=120,
                           batch_size=64,
                           device="cpu")

    pipeline.analyze_model()
    data_frame = pipeline.infer_dataset()

    predictions_path = PROJECT_ROOT / "lab_7_llm" / "dist" / "predictions.csv"
    predictions_path.parent.mkdir(exist_ok=True)
    data_frame.to_csv(predictions_path)

    evaluator = TaskEvaluator(predictions_path, settings.parameters.metrics)
    result = evaluator.run()
    print(result)

    assert result is not None, "Demo does not work correctly"


if __name__ == "__main__":
    main()
