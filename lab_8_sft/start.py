"""
Fine-tuning starter.
"""
# pylint: disable=too-many-locals, undefined-variable, unused-import, too-many-branches, too-many-statements
from pathlib import Path

from transformers import AutoTokenizer, set_seed

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
    set_seed(42)
    settings_path = PROJECT_ROOT / 'lab_8_sft' / 'settings.json'
    parameters = LabSettings(settings_path).parameters

    importer = RawDataImporter(parameters.dataset)
    importer.obtain()
    if importer.raw_data is None:
        return

    preprocessor = RawDataPreprocessor(importer.raw_data)
    # print(preprocessor.analyze())
    preprocessor.transform()

    dataset = TaskDataset(preprocessor.data.head(100))

    max_length = 120
    batch_size = 64
    device = 'cpu'

    pipeline = LLMPipeline(parameters.model, dataset,
                           max_length=max_length, batch_size=batch_size, device=device)

    print('base model analysis:', pipeline.analyze_model())
    sample = dataset[22]
    print(f'example of sample inference:\n'
          f'text: {sample[0]}\n'
          f'label: {pipeline.infer_sample(sample)}')


    predictions_path = PROJECT_ROOT / 'lab_8_sft' / 'dist' / 'predictions.csv'
    predictions_path.parent.mkdir(parents=True, exist_ok=True)

    predictions = pipeline.infer_dataset()
    predictions.to_csv(predictions_path)

    evaluator = TaskEvaluator(predictions_path, parameters.metrics)
    metrics_result = evaluator.run()
    print('results of base model:', metrics_result)

    finetuned_model_path = PROJECT_ROOT / 'lab_8_sft' / 'dist' / parameters.model
    finetuned_model_path.mkdir(parents=True, exist_ok=True)

    sft_parameters = SFTParams(
        batch_size=3,
        max_length=120,
        max_fine_tuning_steps=50,
        learning_rate=1e-3,
        device='cpu',
        finetuned_model_path=finetuned_model_path
    )

    fine_tune_samples = sft_parameters.batch_size * sft_parameters.max_fine_tuning_steps

    num_samples = 10
    tokenized_dataset = TokenizedTaskDataset(
        data=preprocessor.data.loc[num_samples: num_samples + fine_tune_samples],
        tokenizer=AutoTokenizer.from_pretrained(parameters.model),
        max_length=sft_parameters.max_length
    )

    sft_pipeline = SFTPipeline(model_name=parameters.model, dataset=tokenized_dataset,
                               sft_params=sft_parameters)
    sft_pipeline.run()

    finetuned_pipeline = LLMPipeline(str(finetuned_model_path),
                                     TaskDataset(preprocessor.data.sample(num_samples)),
                                     max_length=max_length, batch_size=batch_size, device=device)

    print('finetuned model analysis:', finetuned_pipeline.analyze_model())

    sample = dataset[12]
    print(f'example of sample inference:\n'
          f'text: {sample[0]}\n'
          f'label: {finetuned_pipeline.infer_sample(sample)}')

    finetuned_predictions = finetuned_pipeline.infer_dataset()
    finetuned_predictions.to_csv(predictions_path)

    evaluator = TaskEvaluator(predictions_path, parameters.metrics)
    finetuned_metrics_result = evaluator.run()
    print('results of finetuned model:', finetuned_metrics_result)

    result = finetuned_metrics_result
    assert result is not None, "Finetuning does not work correctly"


if __name__ == "__main__":
    main()
