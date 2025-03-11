"""
Fine-tuning starter.
"""
# pylint: disable=too-many-locals, undefined-variable, unused-import, too-many-branches, too-many-statements
from pathlib import Path

from transformers import AutoTokenizer

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
    result = None

    settings = LabSettings(Path(__file__).parent / 'settings.json')
    importer = RawDataImporter(settings.parameters.dataset)
    importer.obtain()
    if importer.raw_data is None:
        return

    preprocessor = RawDataPreprocessor(importer.raw_data)
    print(preprocessor.analyze())
    preprocessor.transform()

    num_samples = 10

    dataset = TaskDataset(preprocessor.data.head(100))
    pipeline = LLMPipeline(settings.parameters.model, dataset,
                           max_length=120, batch_size=64, device="cpu")
    print("Analysis before finetuning: ", pipeline.analyze_model())


    dataset_inference = pipeline.infer_dataset()
    print(dataset_inference)
    single_prediction = pipeline.infer_sample(dataset[0])
    print("Prediction before finetuning: ", single_prediction)

    predictions_path = Path(__file__).parent / 'dist' / 'predictions.csv'
    predictions_path.parent.mkdir(exist_ok=True)

    dataset_inference.to_csv(predictions_path)

    evaluator = TaskEvaluator(predictions_path, settings.parameters.metrics)
    result = evaluator.run()
    print("Metrics before finetuning: ", result)

    finetuned_model_path = Path(__file__).parent / 'dist' / settings.parameters.model
    sft_params = SFTParams(
        batch_size=3,
        max_length=120,
        max_fine_tuning_steps=200,
        learning_rate=1e-2,
        device="cpu",
        finetuned_model_path=finetuned_model_path
    )

    fine_tune_samples = sft_params.batch_size * sft_params.max_fine_tuning_steps

    tokenized_dataset = TokenizedTaskDataset(preprocessor.data.loc[
                        num_samples:num_samples + fine_tune_samples],
                        tokenizer=AutoTokenizer.from_pretrained(settings.parameters.model),
                        max_length=sft_params.max_length
    )
    sft_pipeline = SFTPipeline(settings.parameters.model, tokenized_dataset, sft_params)
    sft_pipeline.run()

    dataset = TaskDataset(preprocessor.data.head(num_samples))
    finetuned_pipeline = LLMPipeline(
                        str(Path(__file__).parent / 'dist' / settings.parameters.model),
                        dataset, max_length=120, batch_size=64, device='cpu'
    )
    print("Analysis after finetuning: ", finetuned_pipeline.analyze_model())

    dataset_inference = finetuned_pipeline.infer_dataset()
    print(dataset_inference)
    print("Prediction after finetuning: ", finetuned_pipeline.infer_sample(dataset[0]))
    predictions_path = Path(__file__).parent / 'dist' / 'predictions.csv'
    predictions_path.parent.mkdir(exist_ok=True)

    dataset_inference.to_csv(predictions_path)

    evaluator = TaskEvaluator(predictions_path, settings.parameters.metrics)
    result = evaluator.run()
    print("Metrics after finetuning: ", result)

    assert result is not None, "Finetuning does not work correctly"

if __name__ == "__main__":
    main()
