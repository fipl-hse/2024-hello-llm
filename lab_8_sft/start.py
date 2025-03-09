"""
Fine-tuning starter.
"""
import json

# pylint: disable=too-many-locals, undefined-variable, unused-import, too-many-branches, too-many-statements
from pathlib import Path

from transformers import AutoTokenizer

from config.constants import PROJECT_ROOT
from config.lab_settings import LabSettings, SFTParams
from lab_8_sft.main import (
    LLMPipeline,
    RawDataImporter,
    RawDataPreprocessor,
    report_time,
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
    config = LabSettings(PROJECT_ROOT / "lab_8_sft" / "settings.json")

    data_loader = RawDataImporter(config.parameters.dataset)
    data_loader.obtain()

    processor = RawDataPreprocessor(data_loader.raw_data)
    dataset_stats = processor.analyze()
    print(f"Dataset overview: {dataset_stats}")

    processor.transform()
    sample_dataset = TaskDataset(processor.data.head(100))

    model_pipeline = LLMPipeline(
        config.parameters.model,
        sample_dataset,
        max_length=120,
        batch_size=64,
        device="cpu"
    )
    print(f"Initial model analysis: {model_pipeline.analyze_model()}")

    test_sample = sample_dataset[0]
    print(f"Test sample input: {test_sample}")
    print(f"Predicted output: {model_pipeline.infer_sample(test_sample)}")

    prediction_results = model_pipeline.infer_dataset()
    output_path = PROJECT_ROOT / "lab_8_sft" / "dist" / "predictions.csv"
    output_path.parent.mkdir(exist_ok=True)
    prediction_results.to_csv(output_path)

    evaluator = TaskEvaluator(output_path, config.parameters.metrics)
    initial_metrics = evaluator.run()
    print("Performance before fine-tuning:")
    for metric, score in initial_metrics.items():
        print(f"{metric}: {score:.3f}")

    tuning_params = SFTParams(
        batch_size=3,
        max_length=120,
        max_fine_tuning_steps=50,
        learning_rate=1e-3,
        finetuned_model_path=PROJECT_ROOT / "dist" / config.parameters.model,
        device="cpu",
        target_modules=["query", "key", "value"]
    )

    sample_count = 1000
    tuning_sample_count = tuning_params.batch_size * tuning_params.max_fine_tuning_steps

    tokenizer = AutoTokenizer.from_pretrained(config.parameters.model)
    prepared_dataset = TokenizedTaskDataset(
        processor.data.loc[sample_count: sample_count + tuning_sample_count],
        tokenizer=tokenizer,
        max_length=tuning_params.max_length,
    )

    fine_tuning_pipeline = SFTPipeline(
        model_name=config.parameters.model,
        dataset=prepared_dataset,
        sft_params=tuning_params
    )

    print('Starting fine-tuning...')
    fine_tuning_pipeline.run()

    refined_pipeline = LLMPipeline(
        model_name=PROJECT_ROOT / "dist" / config.parameters.model,
        dataset=TaskDataset(processor.data.head(sample_count)),
        max_length=120,
        batch_size=64,
        device="cpu",
    )

    refined_model_analysis = refined_pipeline.analyze_model()
    print("Model evaluation after fine-tuning:")
    for attribute, result in refined_model_analysis.items():
        print(f"{attribute}: {result}")

    first_text_sample = sample_dataset.data.iloc[0]["source"]
    print("\nOriginal Text:", first_text_sample)
    print("Prediction after fine-tuning:", refined_pipeline.infer_sample(first_text_sample))

    final_predictions = refined_pipeline.infer_dataset()
    final_output_path = PROJECT_ROOT / "dist" / "predictions.csv"
    final_output_path.parent.mkdir(exist_ok=True)
    final_predictions.to_csv(final_output_path)

    final_evaluator = TaskEvaluator(final_output_path, config.parameters.metrics)
    final_metrics = final_evaluator.run()
    print("Performance after fine-tuning:")

    for metric, score in final_metrics.items():
        print(f"{metric}: {score:.3f}")

    print("\nMetric differences (post-fine-tuning - pre-fine-tuning):")
    for metric in final_metrics:
        if metric in initial_metrics:
            print(f"{metric}: {final_metrics[metric]:.3f} - {initial_metrics[metric]:.3f} = "
                  f"{final_metrics[metric] - initial_metrics[metric]:.3f}")

    result = final_metrics

    assert result is not None, "Finetuning does not work correctly"


if __name__ == "__main__":
    main()

