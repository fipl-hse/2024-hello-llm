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
    settings = LabSettings(PROJECT_ROOT / 'lab_7_llm' / 'settings.json')
    dataset_name = settings.parameters.dataset
    model_name = settings.parameters.model
    metric_list = settings.parameters.metrics
    predictions_file = PROJECT_ROOT / "dist" / "predictions.csv"

    importer = RawDataImporter(dataset_name)
    importer.obtain()
    if importer.raw_data is None:
        return


    preprocessor = RawDataPreprocessor(importer.raw_data)
    preprocessor.transform()

    analysis_after = preprocessor.analyze()
    print("Dataset properties after preprocessing:", analysis_after)

    task_dataset = TaskDataset(preprocessor.data.head(100))
    print("TaskDataset length:", len(task_dataset))

    pipeline = LLMPipeline(
        model_name=model_name,
        dataset=task_dataset,
        max_length=120,
        batch_size=1,
        device="cpu"
    )

    model_metadata = pipeline.analyze_model()
    print("Model properties:", model_metadata)

    sample = task_dataset[0]
    prediction = pipeline.infer_sample(sample)
    print("Single-sample inference:", prediction)

    df_predictions = pipeline.infer_dataset()
    print("Batch inference result (first rows):")
    print(df_predictions.head())

    predictions_file.parent.mkdir(exist_ok=True)
    df_predictions.to_csv(predictions_file)

    evaluator = TaskEvaluator(predictions_file, metric_list)
    result = evaluator.run()
    print("Evaluation scores:", result)

    assert result is not None, "Demo does not work correctly"


if __name__ == "__main__":
    main()
