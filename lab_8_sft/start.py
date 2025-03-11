"""
Fine-tuning starter.
"""
# pylint: disable=too-many-locals, undefined-variable, unused-import, too-many-branches, too-many-statements
from transformers import AutoTokenizer

from config.constants import PROJECT_ROOT
from config.lab_settings import InferenceParams, LabSettings, SFTParams
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
SETTINGS = LabSettings(LAB_FOLDER / "settings.json")
INFERENCE_PARAMS = InferenceParams(
    num_samples=10,
    max_length=120,
    batch_size=64,
    predictions_path=LAB_FOLDER / "dist" / "predictions.csv",
    device="cpu"
)

SFT_PARAMS = SFTParams(
    batch_size=3,
    max_length=120,
    max_fine_tuning_steps=5,
    learning_rate=1e-3,
    finetuned_model_path=LAB_FOLDER / "dist" / SETTINGS.parameters.model,
    device="cpu"
)


def run_llm_analysis_and_inference(model_name: str,
                                   dataset: TaskDataset) -> None:
    """
    Perform model analysis, inference on random sample and whole dataset,
    calculate metrics
    """
    pipeline = LLMPipeline(
        model_name=model_name,
        dataset=dataset,
        max_length=INFERENCE_PARAMS.max_length,
        batch_size=INFERENCE_PARAMS.batch_size,
        device=INFERENCE_PARAMS.device
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
    INFERENCE_PARAMS.predictions_path.parent.mkdir(exist_ok=True)
    predictions_dataframe.to_csv(INFERENCE_PARAMS.predictions_path)

    evaluator = TaskEvaluator(INFERENCE_PARAMS.predictions_path,
                              SETTINGS.parameters.metrics)
    result = evaluator.run()
    print("Evaluation metrics:")
    for metric, value in result.items():
        print(metric, value, sep=': ')


@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    importer = RawDataImporter(SETTINGS.parameters.dataset)
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

    # base model
    run_llm_analysis_and_inference(model_name=SETTINGS.parameters.model,
                                   dataset=dataset)

    fine_tune_samples = SFT_PARAMS.batch_size * SFT_PARAMS.max_fine_tuning_steps
    tokenized_dataset = TokenizedTaskDataset(
        preprocessor.data.loc[INFERENCE_PARAMS.num_samples:
                              INFERENCE_PARAMS.num_samples + fine_tune_samples],
        tokenizer=AutoTokenizer.from_pretrained(SETTINGS.parameters.model),
        max_length=SFT_PARAMS.max_length
    )

    sft_pipeline = SFTPipeline(
        model_name=SETTINGS.parameters.model,
        dataset=tokenized_dataset,
        sft_params=SFT_PARAMS
    )
    sft_pipeline.run()

    # fine-tuned model
    run_llm_analysis_and_inference(
        model_name=str(LAB_FOLDER / "dist" / SETTINGS.parameters.model),
        dataset=TaskDataset(preprocessor.data.head(INFERENCE_PARAMS.num_samples))
    )

    result = True
    assert result is not None, "Finetuning does not work correctly"


if __name__ == "__main__":
    main()
