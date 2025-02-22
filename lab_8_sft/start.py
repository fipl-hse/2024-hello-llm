"""
Fine-tuning starter.
"""
# pylint: disable=too-many-locals, undefined-variable, unused-import, too-many-branches, too-many-statements
from pathlib import Path

from config.constants import PROJECT_ROOT
from config.lab_settings import LabSettings
from core_utils.llm.time_decorator import report_time
from lab_8_sft.main import (
    RawDataImporter,
    RawDataPreprocessor,
    TaskDataset,
    LLMPipeline,
    TaskEvaluator,
)


@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    settings = LabSettings(PROJECT_ROOT / "lab_8_sft" / "settings.json")
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
                           batch_size=1,
                           device="cpu")
    pipeline.analyze_model()

    sample = pipeline.infer_sample(task_dataset[0])
    print("sample inference result:", sample)

    predictions_df = pipeline.infer_dataset()

    predictions_file = PROJECT_ROOT / "lab_8_sft" / "dist" / "predictions.csv"
    predictions_file.parent.mkdir(exist_ok=True)
    predictions_df.to_csv(predictions_file)

    evaluator = TaskEvaluator(predictions_file, settings.parameters.metrics)
    result = evaluator.run()
    print("evaluation results:", result)

    assert result is not None, "Finetuning does not work correctly"


if __name__ == "__main__":
    main()
