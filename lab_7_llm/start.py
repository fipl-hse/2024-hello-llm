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
    lab_settings = LabSettings(PROJECT_ROOT / 'lab_7_llm' / 'settings.json')
    importer = RawDataImporter(lab_settings.parameters.dataset)
    importer.obtain()

    if importer.raw_data is None:
        return

    preprocessor = RawDataPreprocessor(importer.raw_data)
    properties = preprocessor.analyze()
    print(properties)

    preprocessor.transform()
    dataset = TaskDataset(preprocessor.data.head(100))

    pipeline = LLMPipeline(lab_settings.parameters.model,
                           dataset,
                           max_length=120,
                           batch_size=64,
                           device="cpu")

    pipeline.analyze_model()

    dataframe = pipeline.infer_dataset()

    predictions_path = PROJECT_ROOT / "lab_7_llm" / "dist" / "predictions.csv"
    predictions_path.parent.mkdir(exist_ok=True)
    dataframe.to_csv(predictions_path)

    evaluator = TaskEvaluator(predictions_path, lab_settings.parameters.metrics)
    result = evaluator.run()
    print(result)

    assert result is not None, "Demo does not work correctly"


if __name__ == "__main__":
    main()
