"""
Fine-tuning starter.
"""
# pylint: disable=too-many-locals, undefined-variable, unused-import, too-many-branches, too-many-statements
from pathlib import Path

from config.constants import PROJECT_ROOT
from config.lab_settings import LabSettings
from core_utils.llm.time_decorator import report_time
from lab_8_sft.main import (
    LLMPipeline,
    RawDataImporter,
    RawDataPreprocessor,
    TaskDataset,
    TaskEvaluator,
)


@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    settings = LabSettings(PROJECT_ROOT / 'lab_8_sft' / 'settings.json')

    importer = RawDataImporter(settings.parameters.dataset)
    importer.obtain()

    preprocessor = RawDataPreprocessor(importer.raw_data)
    analysis = preprocessor.analyze()
    print(f'Dataset analysis: {analysis}')

    preprocessor.transform()
    dataset = TaskDataset(preprocessor.data.head(100))

    pipeline = LLMPipeline(
        settings.parameters.model,
        dataset,
        max_length=120,
        batch_size=64,
        device="cpu"
    )
    model_summary = pipeline.analyze_model()
    print(f'Model analysis: {model_summary}')

    sample_text = dataset[0]
    print(f'Single sample input: {sample_text}')

    single_prediction = pipeline.infer_sample(sample_text)
    print(f'Single sample prediction: {single_prediction}')

    inference_data = pipeline.infer_dataset()

    predictions = PROJECT_ROOT / 'lab_8_sft' / 'dist' / 'predictions.csv'
    predictions.parent.mkdir(exist_ok=True)
    inference_data.to_csv(predictions)

    evaluator = TaskEvaluator(predictions, settings.parameters.metrics)
    print(evaluator.run())

    result = evaluator.run()

    assert result is not None, "Finetuning does not work correctly"


if __name__ == "__main__":
    main()
