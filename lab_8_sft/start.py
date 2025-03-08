"""
Fine-tuning starter.
"""
# pylint: disable=too-many-locals, undefined-variable, unused-import, too-many-branches, too-many-statements
from pathlib import Path

from transformers import AutoTokenizer

from config.constants import PROJECT_ROOT
from config.lab_settings import LabSettings
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
from lab_8_sft.service import fine_tuned_pipeline


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

    tokenizer = AutoTokenizer.from_pretrained(settings.parameters.model)

    batch = settings.parameters.batch_size
    fine_tuning_steps = settings.parameters.fine_tuning_steps
    num_samples = 10
    fine_tune_samples = batch * fine_tuning_steps

    tokenized_dataset = TokenizedTaskDataset(
        preprocessor.data.loc[num_samples:num_samples + fine_tune_samples],
        tokenizer,
        max_length=120
    )

    sft_params = settings.parameters.sft_params
    sft_pipeline = SFTPipeline(settings.parameters.model, tokenized_dataset, sft_params)
    sft_pipeline.run()

    dataset = TaskDataset(preprocessor.data.head(100))

    finetuned_model_path = PROJECT_ROOT / 'lab_8_sft' / 'dist' / f'{settings.parameters.model}_finetuned'

    fine_tuned_pipeline = LLMPipeline(
        model_name=finetuned_model_path,
        dataset=dataset,
        max_length=120,
        batch_size=64,
        device='cpu'
    )

    model_summary = fine_tuned_pipeline.analyze_model()
    print(f'Model analysis: {model_summary}')

    sample_text = dataset[0]
    print(f'Single sample input: {sample_text}')

    single_prediction = fine_tuned_pipeline.infer_sample(sample_text)
    print(f'Single sample prediction: {single_prediction}')

    infer_data = fine_tuned_pipeline.infer_dataset()

    predictions = PROJECT_ROOT / "lab_8_sft" / "dist" / "predictions.csv"
    predictions.parent.mkdir(exist_ok=True)
    infer_data.to_csv(predictions)

    evaluator = TaskEvaluator(predictions, settings.parameters.metrics)

    result = evaluator.run()
    print(result)

    assert result is not None, "Demo does not work correctly"

if __name__ == "__main__":
    main()
