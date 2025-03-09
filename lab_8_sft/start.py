"""
Fine-tuning starter.
"""
# pylint: disable=too-many-locals, undefined-variable, unused-import, too-many-branches, too-many-statements
from transformers import AutoTokenizer

from config.constants import PROJECT_ROOT
from config.lab_settings import LabSettings, SFTParams
from core_utils.llm.time_decorator import report_time
from lab_8_sft.main import (
    LLMPipeline,
    RawDataImporter,
    RawDataPreprocessor,
    TaskDataset,
    TaskEvaluator,
    TokenizedTaskDataset,
    SFTPipeline,
)


@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    settings = LabSettings(PROJECT_ROOT / "lab_8_sft" / "settings.json")
    dataset_name = settings.parameters.dataset
    model_name = settings.parameters.model
    metric_list = settings.parameters.metrics
    predictions_file = PROJECT_ROOT / "lab_8_sft" / "dist" / "predictions.csv"

    importer = RawDataImporter(dataset_name)
    importer.obtain()
    if importer.raw_data is None:
        return

    preprocessor = RawDataPreprocessor(importer.raw_data)
    preprocessor.transform()

    dataset_analysis = preprocessor.analyze()
    print("Dataset properties after preprocessing:", dataset_analysis)

    task_dataset = TaskDataset(preprocessor.data.head(100))
    print("TaskDataset length:", len(task_dataset))

    pipeline_before_sft = LLMPipeline(
        model_name=model_name,
        dataset=task_dataset,
        max_length=120,
        batch_size=1,
        device="cpu"
    )

    model_analysis_before_sft = pipeline_before_sft.analyze_model()
    print("Model properties (before SFT):", model_analysis_before_sft)

    sample = task_dataset[0]
    single_prediction_before_sft = pipeline_before_sft.infer_sample(sample)
    print("Single-sample inference (before SFT):", single_prediction_before_sft)

    df_predictions_before_sft = pipeline_before_sft.infer_dataset()
    print("Batch inference result (first rows) (before SFT):")
    print(df_predictions_before_sft.head())

    predictions_file.parent.mkdir(exist_ok=True)
    df_predictions_before_sft.to_csv(predictions_file, index=False)

    evaluator_before_sft = TaskEvaluator(predictions_file, metric_list)
    result_before_sft = evaluator_before_sft.run()
    print("Evaluation scores (before SFT):")
    for metric, value in result_before_sft.items():
        print(f"{metric}: {value}")

    sft_params = SFTParams(
        batch_size=3,
        max_length=120,
        max_fine_tuning_steps=5,
        learning_rate=1e-3,
        finetuned_model_path=PROJECT_ROOT / "lab_8_sft" / "dist" / f"{model_name}_finetuned",
        device="cpu",
        target_modules=["query", "value"]
    )

    num_samples_for_inference = 10
    fine_tune_samples = sft_params.batch_size * sft_params.max_fine_tuning_steps

    train_df = preprocessor.data.loc[
               num_samples_for_inference: num_samples_for_inference + fine_tune_samples
               ]

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    tokenized_dataset = TokenizedTaskDataset(
        train_df, tokenizer, max_length=sft_params.max_length
    )

    sft_pipeline = SFTPipeline(
        model_name=model_name,
        dataset=tokenized_dataset,
        sft_params=sft_params
    )
    sft_pipeline.run()

    tokenizer.save_pretrained(sft_params.finetuned_model_path)

    pipeline_after_sft = LLMPipeline(
        model_name=str(sft_params.finetuned_model_path),
        dataset=TaskDataset(preprocessor.data.head(num_samples_for_inference)),
        max_length=120,
        batch_size=64,
        device="cpu"
    )

    model_analysis_after_sft = pipeline_after_sft.analyze_model()
    print("Model analysis after SFT:")
    for field, value in model_analysis_after_sft.items():
        print(f"{field}: {value}")

        random_sample = tuple(task_dataset.data.sample(random_state=42)["source"])
        print("Inference result on random sample (after SFT):",
              pipeline_after_sft.infer_sample(random_sample))

        predictions_dataframe_after_sft = pipeline_after_sft.infer_dataset()
        predictions_dataframe_after_sft.to_csv(predictions_file, index=False)

        evaluator_after_sft = TaskEvaluator(predictions_file, settings.parameters.metrics)
        result_after_sft = evaluator_after_sft.run()
        print("Evaluation scores after SFT:")
        for metric, value in result_after_sft.items():
            print(f"{metric}: {value}")

    assert result_after_sft is not None, "Finetuning does not work correctly"


if __name__ == "__main__":
    main()
