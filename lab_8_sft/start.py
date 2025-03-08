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
                           batch_size=64,
                           device="cpu")
    pipeline.analyze_model()

    sample = pipeline.infer_sample(task_dataset[0])
    print("sample inference result (base model):", sample)

    predictions_df = pipeline.infer_dataset()
    predictions_file = PROJECT_ROOT / "lab_8_sft" / "dist" / "predictions.csv"
    predictions_file.parent.mkdir(exist_ok=True)
    predictions_df.to_csv(predictions_file)

    evaluator = TaskEvaluator(predictions_file, settings.parameters.metrics)
    base_metrics = evaluator.run()
    print("evaluation results:", base_metrics)

    sft_params = SFTParams(
        max_length=120,
        batch_size=3,
        max_fine_tuning_steps=50,
        device="cpu",
        finetuned_model_path=PROJECT_ROOT / "lab_8_sft" /
                             "dist" / f"{settings.parameters.model}_finetuned",
        learning_rate=1e-3,
        target_modules=["q", "k"]
    )

    num_samples = 10
    fine_tune_samples = sft_params.batch_size * sft_params.max_fine_tuning_steps
    base_tokenizer = AutoTokenizer.from_pretrained(settings.parameters.model)
    tokenized_dataset = TokenizedTaskDataset(
        preprocessor.data.loc[num_samples: num_samples + fine_tune_samples],
        tokenizer=base_tokenizer,
        max_length=sft_params.max_length
    )

    sft_pipeline = SFTPipeline(model_name=settings.parameters.model,
                               dataset=tokenized_dataset,
                               sft_params=sft_params)
    sft_pipeline.run()

    base_tokenizer.save_pretrained(sft_params.finetuned_model_path)

    pipeline_ft = LLMPipeline(
        str(sft_params.finetuned_model_path),
        TaskDataset(preprocessor.data.head(num_samples)),
        max_length=120,
        batch_size=64,
        device="cpu"
    )
    pipeline_ft.analyze_model()

    predictions_dataframe = pipeline_ft.infer_dataset()
    predictions_dataframe.to_csv(predictions_file)

    evaluator = TaskEvaluator(predictions_file, settings.parameters.metrics)
    result = evaluator.run()

    for metric in settings.parameters.metrics:
        key = metric.value
        base_val = base_metrics.get(key)
        ft_val = result.get(key)
        if base_val is not None and ft_val is not None:
            diff = ft_val - base_val
            print(f"metric {key}: base = {base_val:.3f}, "
                  f"fine-tuned = {ft_val:.3f}, diff = {diff:.3f}")
        else:
            print(f"metric {key} not computed properly: "
                  f"base = {base_val}, fine-tuned = {ft_val}")

    assert result is not None, "Finetuning does not work correctly"


if __name__ == "__main__":
    main()
