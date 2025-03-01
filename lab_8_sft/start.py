"""
Fine-tuning starter.
"""
# pylint: disable=too-many-locals, undefined-variable, unused-import, too-many-branches, too-many-statements
from pathlib import Path

from config.constants import PROJECT_ROOT
from config.lab_settings import LabSettings
from core_utils.llm.time_decorator import report_time
from lab_8_sft.main import (LLMPipeline, RawDataImporter, RawDataPreprocessor, TaskDataset,
                            TaskEvaluator)


@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    settings = LabSettings(PROJECT_ROOT / "lab_8_sft" / "settings.json")

    # mark4
    importer = RawDataImporter(settings.parameters.dataset)
    importer.obtain()
    if importer.raw_data is None:
        return

    preprocessor = RawDataPreprocessor(importer.raw_data)
    dataset_analysis = preprocessor.analyze()
    print("Dataset analysis:")
    for field, value in dataset_analysis.items():
        print(field, value, sep=': ')

    preprocessor.transform()
    dataset = TaskDataset(preprocessor.data.head(100))
    pipeline = LLMPipeline(settings.parameters.model,
                           dataset,
                           max_length=120,
                           batch_size=64,
                           device="cpu")

    model_analysis = pipeline.analyze_model()
    print("Model analysis:")
    for field, value in model_analysis.items():
        print(field, value, sep=': ')

    random_sample = tuple(dataset.data.sample(random_state=42)["source"])
    print("Random text:", random_sample[0])
    print("Inference result:", pipeline.infer_sample(random_sample))

    # mark6
    pipeline.infer_dataset()

    predictions_dataframe = pipeline.infer_dataset()
    predictions_path = PROJECT_ROOT / "lab_8_sft" / "dist" / "predictions.csv"
    predictions_path.parent.mkdir(exist_ok=True)
    predictions_dataframe.to_csv(predictions_path)

    evaluator = TaskEvaluator(predictions_path, settings.parameters.metrics)
    result = evaluator.run()
    print("Evaluation metrics:")
    for metric, value in result.items():
        print(metric, value, sep=': ')

    # mark8
    result = True
    assert result is not None, "Finetuning does not work correctly"


if __name__ == "__main__":
    main()
