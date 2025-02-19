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
    TaskEvaluator
)
from pathlib import Path


@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    settings = LabSettings(PROJECT_ROOT / "lab_7_llm" / "settings.json")

    importer = RawDataImporter(settings.parameters.dataset)
    importer.obtain()

    preprocessor = RawDataPreprocessor(importer.raw_data)
    preprocessor.transform()

    dataset = TaskDataset(preprocessor.data)

    device = "cpu"
    batch_size = 64
    max_length = 120

    pipeline = LLMPipeline(settings.parameters.model, dataset, max_length, batch_size, device)
    pipeline.analyze_model()

    inference = pipeline.infer_dataset()
    output_folder = "dist"
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    path = Path('dist/predictions.csv')
    inference.to_csv(path)

    evaluator = TaskEvaluator(path, settings.parameters.metrics)
    result = evaluator.run()
    assert result is not None, "Demo does not work correctly"
    print(result)


if __name__ == "__main__":
    main()
