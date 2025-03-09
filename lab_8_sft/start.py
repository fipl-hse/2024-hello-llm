"""
Fine-tuning starter.
"""
# pylint: disable=too-many-locals, undefined-variable, unused-import, too-many-branches, too-many-statements
from pathlib import Path

from config.lab_settings import LabSettings, SFTParams
from core_utils.llm.time_decorator import report_time
from lab_8_sft.main import (
    LLMPipeline,
    RawDataImporter,
    RawDataPreprocessor,
    TaskDataset,
    TaskEvaluator,
    TokenizedTaskDataset,
    SFTPipeline
)
from transformers import AutoTokenizer


@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    settings_path = Path(__file__).parent / 'settings.json'
    settings = LabSettings(settings_path)

    importer = RawDataImporter(settings.parameters.dataset)
    importer.obtain()

    if importer.raw_data is None:
        return None

    preprocessor = RawDataPreprocessor(importer.raw_data)
    preprocessor.transform()

    batch_size = 1
    max_length = 120
    device = 'cpu'

    dataset = TaskDataset(preprocessor.data.head(10))
    pipeline = LLMPipeline(settings.parameters.model,
                           dataset,
                           max_length,
                           batch_size,
                           device)

    model_analysis = pipeline.analyze_model()
    print("Model analysis: ")
    print(model_analysis)

    sample_prediction = pipeline.infer_sample(dataset[0])
    print("Inference of one sample: ")
    print(sample_prediction)

    predictions = pipeline.infer_dataset()
    predictions_path = Path(__file__).parent / 'dist' / 'predictions.csv'
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(predictions_path, index=False)

    evaluator = TaskEvaluator(predictions_path, settings.parameters.metrics)
    inference_result = evaluator.run()
    print("Dataset inference result: ")
    print(inference_result)

    num_samples = 10
    fine_tuning_steps = 10
    fine_tune_samples = batch_size * fine_tuning_steps
    dataset = TokenizedTaskDataset(
        preprocessor.data.loc[
            num_samples: num_samples + fine_tune_samples
        ],
        tokenizer=AutoTokenizer.from_pretrained(
            settings.parameters.model
        ),
        max_length=max_length
    )
    model_path = Path(__file__).parent / 'dist' / settings.parameters.model
    lr = 1e-3
    sft_params = SFTParams(
        max_length=max_length,
        batch_size=batch_size,
        max_fine_tuning_steps=fine_tuning_steps,
        device=device,
        finetuned_model_path=model_path,
        learning_rate=lr,
        target_modules=["query", "key", "value", "dense"]

    )
    pipeline = SFTPipeline(settings.parameters.model, dataset, sft_params)
    result = pipeline
    assert result is not None, "Finetuning does not work correctly"


if __name__ == "__main__":
    main()
