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
    # for mark 6
    lab_settings = LabSettings(PROJECT_ROOT / 'lab_8_sft' / 'settings.json')
    importer = RawDataImporter(lab_settings.parameters.dataset)
    importer.obtain()

    if importer.raw_data is None:
        return

    preprocessor = RawDataPreprocessor(importer.raw_data)
    preprocessor.analyze()

    preprocessor.transform()
    dataset = TaskDataset(preprocessor.data.head(100))

    pipeline = LLMPipeline(lab_settings.parameters.model,
                           dataset,
                           max_length=120,
                           batch_size=64,
                           device="cpu")

    pipeline.analyze_model()

    dataframe = pipeline.infer_dataset()

    predictions_path = PROJECT_ROOT / "lab_8_sft" / "dist" / "predictions.csv"
    predictions_path.parent.mkdir(exist_ok=True)
    dataframe.to_csv(predictions_path)

    evaluator = TaskEvaluator(predictions_path, lab_settings.parameters.metrics)
    result = evaluator.run()
    print("Evaluation results (mark 6):")
    for metric, value in result.items():
        print(metric, value, sep=': ')

    # for mark 8
    # Inference
    preprocessor.transform()
    num_samples = 10
    dataset = TaskDataset(preprocessor.data.head(100))
    pipeline = LLMPipeline(model_name=lab_settings.parameters.model,
                           dataset=dataset,
                           max_length=120,
                           batch_size=64,
                           device="cpu")

    model_analysis = pipeline.analyze_model()
    print("Model analysis (inference):")
    for field, value in model_analysis.items():
        print(field, value, sep=': ')

    pipeline.infer_dataset()

    predictions_dataframe = pipeline.infer_dataset()
    predictions_path = PROJECT_ROOT / "lab_8_sft" / "dist" / "predictions.csv"
    predictions_path.parent.mkdir(exist_ok=True)
    predictions_dataframe.to_csv(predictions_path)

    evaluator = TaskEvaluator(predictions_path, lab_settings.parameters.metrics)
    result = evaluator.run()
    print("Evaluation metrics (inference):")
    for metric, value in result.items():
        print(metric, value, sep=': ')

    # Fine-tuning
    sft_params = SFTParams(
        max_length=120,
        batch_size=3,
        max_fine_tuning_steps=50,
        device="cpu",
        finetuned_model_path=PROJECT_ROOT / "lab_8_sft" / "dist" / lab_settings.parameters.model,
        learning_rate=1e-3
    )

    fine_tune_samples = sft_params.batch_size * sft_params.max_fine_tuning_steps
    tokenized_dataset = TokenizedTaskDataset(
        preprocessor.data.loc[num_samples: num_samples + fine_tune_samples],
        tokenizer=AutoTokenizer.from_pretrained(lab_settings.parameters.model),
        max_length=sft_params.max_length
    )

    sft_pipeline = SFTPipeline(model_name=lab_settings.parameters.model,
                               dataset=tokenized_dataset,
                               sft_params=sft_params)
    sft_pipeline.run()

    num_samples = 10
    pipeline = LLMPipeline(
        model_name=str(PROJECT_ROOT / "lab_8_sft" / "dist" / lab_settings.parameters.model),
        dataset=TaskDataset(preprocessor.data.head(num_samples)),
        max_length=120,
        batch_size=64,
        device="cpu"
    )

    model_analysis = pipeline.analyze_model()
    print("Model analysis (fine-tuning):")
    for field, value in model_analysis.items():
        print(field, value, sep=': ')

    pipeline.infer_dataset()

    predictions_dataframe = pipeline.infer_dataset()
    predictions_path = PROJECT_ROOT / "lab_8_sft" / "dist" / "predictions.csv"
    predictions_path.parent.mkdir(exist_ok=True)
    predictions_dataframe.to_csv(predictions_path)

    evaluator = TaskEvaluator(predictions_path, lab_settings.parameters.metrics)
    result = evaluator.run()
    print("Evaluation metrics (fine-tuning):")
    for metric, value in result.items():
        print(metric, value, sep=': ')

    assert result is not None, "Finetuning does not work correctly"


if __name__ == "__main__":
    main()
