"""
Starter for demonstration of laboratory work.
"""
# pylint: disable= too-many-locals, undefined-variable, unused-import
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
    analysis_before = preprocessor.analyze()
    print("dataset properties before preprocessing:", analysis_before)

    preprocessor.transform()
    task_dataset = TaskDataset(preprocessor.data.head(100))

    pipeline = LLMPipeline(settings.parameters.model,
                           task_dataset,
                           max_length=120,
                           batch_size=64,
                           device="cpu")
    pipeline.analyze_model()

    sample = pipeline.infer_sample(task_dataset[0])
    print("sample inference result:", sample)

    predictions_df = pipeline.infer_dataset()

    predictions_file = PROJECT_ROOT / "lab_7_llm" / "dist" / "predictions.csv"
    predictions_file.parent.mkdir(exist_ok=True)
    predictions_df.to_csv(predictions_file)

    evaluator = TaskEvaluator(predictions_file, settings.parameters.metrics)
    result = evaluator.run()
    print("evaluation results:", result)

    assert result is not None, "Demo does not work correctly"


if __name__ == "__main__":
    main()
