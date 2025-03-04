"""
Starter for demonstration of laboratory work.
"""
# pylint: disable= too-many-locals, undefined-variable, unused-import
from config.constants import PROJECT_ROOT
from config.lab_settings import LabSettings
from lab_7_llm.main import (
    LLMPipeline,
    Path,
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
    # analysis = preprocessor.analyze()
    preprocessor.transform()

    dataset = TaskDataset(preprocessor.data.head(100))
    pipeline = LLMPipeline(settings.parameters.model,
                           dataset,
                           max_length=120,
                           batch_size=64,
                           device='cpu')

    # result = pipeline.infer_sample(dataset[1])

    predictions_path = Path(PROJECT_ROOT / 'lab_7_llm' / 'dist' / 'predictions.csv')
    if not predictions_path.parent.exists():
        predictions_path.parent.mkdir()

    pipeline.infer_dataset().to_csv(predictions_path, index=False)

    evaluator = TaskEvaluator(predictions_path, settings.parameters.metrics)
    result = evaluator.run()
    print(result)

    assert result is not None, "Demo does not work correctly"


if __name__ == "__main__":
    main()
