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

    pipeline = LLMPipeline(parameters.model, dataset,
                           max_length=120, batch_size=64, device='cpu')

    # print(pipeline.analyze_model())
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
    print('results:', metrics_result)

    finetuned_model_dir = f'finetuned_{parameters.model.split("/")[-1]}'
    finetuned_model_path = PROJECT_ROOT / 'lab_8_sft' / 'dist' / finetuned_model_dir

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
        preprocessor.data.loc[num_samples: num_samples + fine_tune_samples],
        tokenizer=AutoTokenizer.from_pretrained(parameters.model),
        max_length=sft_parameters.max_length
    )

    sft_pipeline = SFTPipeline(model_name=parameters.model, dataset=tokenized_dataset,
                               sft_params=sft_parameters)
    sft_pipeline.run()

    # result = None
    # assert result is not None, "Finetuning does not work correctly"


if __name__ == "__main__":
    main()
