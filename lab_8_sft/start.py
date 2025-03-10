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
    settings = LabSettings(PROJECT_ROOT / 'lab_8_sft' / 'settings.json')

    importer = RawDataImporter(settings.parameters.dataset)
    importer.obtain()

    if importer.raw_data is None:
        raise ValueError("Raw data is None")

    preprocessor = RawDataPreprocessor(importer.raw_data)
    analysis = preprocessor.analyze()
    print(f'Dataset analysis: {analysis}')

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
    print(f'Model analysis: {model_summary}')

    sample_text = dataset[0]
    print(f'Single sample input: {sample_text}')

    single_prediction = pipeline.infer_sample(sample_text)
    print(f'Single sample prediction: {single_prediction}')

    infer_data = pipeline.infer_dataset()

    predictions = PROJECT_ROOT / "lab_8_sft" / "dist" / "predictions.csv"
    predictions.parent.mkdir(exist_ok=True)
    infer_data.to_csv(predictions)

    evaluator = TaskEvaluator(predictions, settings.parameters.metrics)

    result = evaluator.run()
    print(result)

    #8

    sft_params = SFTParams(
        max_length=120,
        batch_size=3,
        max_fine_tuning_steps=50,
        device="cpu",
        finetuned_model_path=PROJECT_ROOT / 'lab_8_sft' / 'dist' / settings.parameters.model,
        learning_rate=1e-3,
        target_modules=["q", "v"],
    )

    num_samples = 10

    fine_tune_samples = sft_params.batch_size * sft_params.max_fine_tuning_steps
    base_tokenizer = AutoTokenizer.from_pretrained(settings.parameters.model)
    dataset = TokenizedTaskDataset(
        preprocessor.data.loc[num_samples: num_samples + fine_tune_samples],
        tokenizer=base_tokenizer.from_pretrained(settings.parameters.model),
        max_length=sft_params.max_length
    )

    sft_pipeline = SFTPipeline(
        model_name=settings.parameters.model,
        dataset=dataset,
        sft_params=sft_params
    )

    sft_pipeline.run()

    assert result is not None, "Demo does not work correctly"

if __name__ == "__main__":
    main()