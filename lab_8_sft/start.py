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

LAB_FOLDER = PROJECT_ROOT / "lab_8_sft"

@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    settings = LabSettings(LAB_FOLDER / "settings.json")

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
    num_samples = 10
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


    pipeline.infer_dataset()

    predictions_dataframe = pipeline.infer_dataset()
    predictions_path = LAB_FOLDER / "dist" / "predictions.csv"
    predictions_path.parent.mkdir(exist_ok=True)
    predictions_dataframe.to_csv(predictions_path)

    evaluator = TaskEvaluator(predictions_path, settings.parameters.metrics)
    result = evaluator.run()
    print("Evaluation metrics:")
    for metric, value in result.items():
        print(metric, value, sep=': ')

    sft_params = SFTParams(
        batch_size=3,
        max_length=120,
        max_fine_tuning_steps=5,
        learning_rate=1e-3,
        finetuned_model_path=LAB_FOLDER / "dist" / settings.parameters.model,
        device="cpu"
    )

    fine_tune_samples = sft_params.batch_size * sft_params.max_fine_tuning_steps
    tokenized_dataset = TokenizedTaskDataset(
        preprocessor.data.loc[num_samples: num_samples + fine_tune_samples],
        tokenizer=AutoTokenizer.from_pretrained(settings.parameters.model),
        max_length=sft_params.max_length
    )

    sft_pipeline = SFTPipeline(model_name=settings.parameters.model,
                               dataset=tokenized_dataset,
                               sft_params=sft_params)
    sft_pipeline.run()


    num_samples = 10
    pipeline = LLMPipeline(
        settings.parameters.model,
        TaskDataset(preprocessor.data.head(num_samples)),
        max_length=120,
        batch_size=64,
        device="cpu"
    )

    model_analysis = pipeline.analyze_model()
    print("Model analysis:")
    for field, value in model_analysis.items():
        print(field, value, sep=': ')

    random_sample = tuple(dataset.data.sample(random_state=42)["source"])
    print("Random text:", random_sample[0])
    print("Inference result:", pipeline.infer_sample(random_sample))

    pipeline.infer_dataset()

    predictions_dataframe = pipeline.infer_dataset()
    predictions_path = LAB_FOLDER / "dist" / "predictions.csv"
    predictions_path.parent.mkdir(exist_ok=True)
    predictions_dataframe.to_csv(predictions_path)

    evaluator = TaskEvaluator(predictions_path, settings.parameters.metrics)
    result = evaluator.run()
    print("Evaluation metrics:")
    for metric, value in result.items():
        print(metric, value, sep=': ')

    assert result is not None, "Finetuning does not work correctly"


if __name__ == "__main__":
    main()
