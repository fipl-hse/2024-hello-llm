"""
Fine-tuning starter.
"""

# pylint: disable=too-many-locals, undefined-variable, unused-import, too-many-branches, too-many-statements
from pathlib import Path

from transformers import AutoTokenizer

from config.constants import PROJECT_ROOT
from config.lab_settings import LabSettings, SFTParams
from core_utils.llm.time_decorator import report_time
from lab_8_sft.main import (
    LLMPipeline,
    RawDataImporter,
    RawDataPreprocessor,
    SFTPipeline,
    TaskDataset,
    TaskEvaluator,
    TokenizedTaskDataset,
)


@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    current_folder = PROJECT_ROOT / "lab_8_sft"

    settings = LabSettings(current_folder / "settings.json")

    importer = RawDataImporter(settings.parameters.dataset)
    importer.obtain()

    if importer.raw_data is None:
        raise ValueError("Raw data is None")

    preprocessor = RawDataPreprocessor(importer.raw_data)
    analysis = preprocessor.analyze()
    print(f"Dataset analysis: {analysis}")

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
    print(f"Model analysis: {model_summary}")

    sample_text = dataset[0]
    print(f"Single sample input: {sample_text}")

    single_prediction = pipeline.infer_sample(sample_text)
    print(f"Single sample prediction: {single_prediction}")

    inference_data = pipeline.infer_dataset()

    predictions_path = current_folder / "dist" / "predictions.csv"
    predictions_path.parent.mkdir(exist_ok=True)
    inference_data.to_csv(predictions_path)

    evaluator = TaskEvaluator(predictions_path, settings.parameters.metrics)
    old_metrics = evaluator.run()
    print("Old evaluation metrics before fine-tuning:")
    for metric, value in old_metrics.items():
        print(f"{metric}: {value:.3f}")

    sft_params = SFTParams(
        batch_size=3,
        max_length=120,
        max_fine_tuning_steps=50,
        learning_rate=1e-3,
        finetuned_model_path=current_folder / "dist" / settings.parameters.model,
        device="cpu"
    )

    num_samples = 1000
    fine_tune_samples = sft_params.batch_size * sft_params.max_fine_tuning_steps

    tokenizer = AutoTokenizer.from_pretrained(settings.parameters.model)
    tokenizer.save_pretrained(sft_params.finetuned_model_path)

    tokenized_dataset = TokenizedTaskDataset(
        preprocessor.data.loc[num_samples : num_samples + fine_tune_samples],
        tokenizer=tokenizer,
        max_length=sft_params.max_length,
    )

    sft_pipeline = SFTPipeline(
        model_name=settings.parameters.model,
        dataset=tokenized_dataset,
        sft_params=sft_params
    )

    sft_pipeline.run()

    fine_tuned_pipeline = LLMPipeline(
        model_name=str(current_folder / "dist" / settings.parameters.model),
        dataset=TaskDataset(preprocessor.data.head(num_samples)),
        max_length=120,
        batch_size=64,
        device="cpu",
    )

    model_analysis = fine_tuned_pipeline.analyze_model()
    print("Model analysis after fine-tuning:")
    for field, value in model_analysis.items():
        print(f"{field}: {value}")

    first_sample = dataset.data.iloc[0]["source"]
    print("\nFirst text:", first_sample)
    print("Inference result after fine-tuning:", fine_tuned_pipeline.infer_sample(first_sample))

    predictions_dataframe = fine_tuned_pipeline.infer_dataset()

    predictions_path = current_folder / "dist" / "predictions.csv"
    predictions_path.parent.mkdir(exist_ok=True)
    predictions_dataframe.to_csv(predictions_path)

    evaluator = TaskEvaluator(predictions_path, settings.parameters.metrics)
    new_metrics = evaluator.run()
    print("New evaluation metrics after fine-tuning:")

    for metric, value in new_metrics.items():
        print(f"{metric}: {value:.3f}")

    print("\nDifference in metrics (new - old):")
    if old_metrics and isinstance(old_metrics, dict):
        for metric, new_value in new_metrics.items():
            old_value = old_metrics.get(metric)
            if old_value is not None:
                print(f"{metric}: {new_value:.3f} - {old_value:.3f} = {new_value - old_value:.3f}")

    result = new_metrics

    assert result is not None, "Finetuning does not work correctly"


if __name__ == "__main__":
    main()