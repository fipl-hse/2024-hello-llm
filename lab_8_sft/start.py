"""
Fine-tuning starter.
"""
# pylint: disable=too-many-locals, undefined-variable, unused-import, too-many-branches, too-many-statements
from pathlib import Path

from transformers import AutoTokenizer

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
    settings_path = Path(__file__).parent / 'settings.json'
    settings = LabSettings(settings_path)

    importer = RawDataImporter(settings.parameters.dataset)
    importer.obtain()

    if importer.raw_data is None:
        return None

    preprocessor = RawDataPreprocessor(importer.raw_data)
    preprocessor.transform()

    # inference params
    batch_size = 64
    max_length = 120
    device = 'cpu'
    num_samples = 10

    dataset = TaskDataset(preprocessor.data.loc[:num_samples])
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

    # fine-tuning params
    batch_size = 3
    fine_tuning_steps = 50
    learning_rate = 1e-3

    fine_tune_samples = batch_size * fine_tuning_steps
    sft_dataset = TokenizedTaskDataset(
        preprocessor.data.loc[
            num_samples: num_samples + fine_tune_samples
        ],
        tokenizer=AutoTokenizer.from_pretrained(
            settings.parameters.model
        ),
        max_length=max_length
    )
    model_path = Path(__file__).parent / 'dist' / settings.parameters.model
    sft_params = SFTParams(
        max_length=max_length,
        batch_size=batch_size,
        max_fine_tuning_steps=fine_tuning_steps,
        device=device,
        finetuned_model_path=model_path,
        learning_rate=learning_rate,
        target_modules=["query", "key", "value", "dense"]

    )
    sft_pipeline = SFTPipeline(settings.parameters.model, sft_dataset, sft_params)
    sft_pipeline.run()

    pipeline_sft_llm = LLMPipeline(model_path,
                                   dataset,
                                   max_length,
                                   batch_size=64,
                                   device=device)

    predictions_sft = pipeline_sft_llm.infer_dataset()
    predictions_path = Path(__file__).parent / 'dist' / 'predictions.csv'
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_sft.to_csv(predictions_path, index=False)

    evaluator = TaskEvaluator(predictions_path, settings.parameters.metrics)
    result = evaluator.run()
    print("Dataset fine-tuning result: ")
    print(result)
    assert result is not None, "Finetuning does not work correctly"
    return None


if __name__ == "__main__":
    main()
