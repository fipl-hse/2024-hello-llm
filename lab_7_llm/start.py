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

    # mark4
    importer = RawDataImporter(settings.parameters.dataset)
    importer.obtain()
    if importer.raw_data is None:
        return

    preprocessor = RawDataPreprocessor(importer.raw_data)
    data_analysis = preprocessor.analyze()
    print("Dataset analysis:")
    for field, value in data_analysis.items():
        print(field, value, sep=': ')

    # mark6
    preprocessor.transform()
    dataset = TaskDataset(preprocessor.data)
    pipeline = LLMPipeline(settings.parameters.model,
                           dataset,
                           max_length=120,
                           batch_size=1,
                           device='cpu',
                           )
    summary = pipeline.infer_sample(dataset[0])
    model_analysis = pipeline.analyze_model()
    print("Model analysis:")
    for field, value in model_analysis.items():
        print(field, value, sep=': ')

    # mark8
    pipeline = LLMPipeline(settings.parameters.model,
                           dataset,
                           max_length=120,
                           batch_size=64,
                           device="cpu")

    predictions_dataframe = pipeline.infer_dataset()
    predictions_path = PROJECT_ROOT / "lab_7_llm" / "dist" / "predictions.csv"
    predictions_path.parent.mkdir(exist_ok=True)
    predictions_dataframe.to_csv(predictions_path)

    evaluator = TaskEvaluator(predictions_path, settings.parameters.metrics)
    result = evaluator.run()
    print("Evaluation results:")
    for metric, value in result.items():
        print(metric, value, sep=': ')

    assert result is not None, "Demo does not work correctly"


if __name__ == "__main__":
    main()
