"""
Fine-tuning starter.
"""
# pylint: disable=too-many-locals, undefined-variable, unused-import, too-many-branches, too-many-statements
from pathlib import Path

import pandas as pd
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

    settings = LabSettings(PROJECT_ROOT/'lab_8_sft'/'settings.json')

    importer = RawDataImporter(settings.parameters.dataset)
    importer.obtain()

    if not isinstance(importer.raw_data, pd.DataFrame):
        raise TypeError('The downloaded dataset is not pd.DataFrame')

    preprocessor = RawDataPreprocessor(importer.raw_data)
    preprocessor.analyze()
    preprocessor.transform()

    dataset = TaskDataset(preprocessor.data.head(100))

    pipeline = LLMPipeline(settings.parameters.model,
                           dataset,
                           max_length=120,
                           batch_size=64,
                           device="cpu")
    pipeline.analyze_model()
    pipeline.infer_sample(dataset[0])

    predictions_path = PROJECT_ROOT/'lab_8_sft'/'dist'/'predictions.csv'
    predictions_path.parent.mkdir(exist_ok=True)
    dataset_inference = pipeline.infer_dataset()
    dataset_inference.to_csv(predictions_path, index=False)

    evaluator = TaskEvaluator(predictions_path, settings.parameters.metrics)
    evaluator.run()

    sft_params = SFTParams(
        batch_size=3,
        max_length=120,
        max_fine_tuning_steps=200,
        learning_rate=1e-3,
        finetuned_model_path=PROJECT_ROOT/'lab_8_sft'/'dist'/ settings.parameters.model,
        device='cpu',
    )

    num_samples = 11
    fine_tune_samples = sft_params.batch_size * sft_params.max_fine_tuning_steps
    tokenizer = AutoTokenizer.from_pretrained(settings.parameters.model)
    tokenised_dataset = TokenizedTaskDataset(preprocessor.data.loc[
                                   num_samples: num_samples + fine_tune_samples
                                   ],
                                             tokenizer=tokenizer,
                                             max_length=sft_params.max_length)

    sft_pipeline = SFTPipeline(settings.parameters.model,
                               dataset=tokenised_dataset,
                               sft_params=sft_params)
    sft_pipeline.run()

    finetuned_pipline = LLMPipeline(
        str(sft_params.finetuned_model_path),
        TaskDataset(preprocessor.data.head(11)),
        max_length=120,
        batch_size=64,
        device='cpu'
    )
    finetuned_pipline.analyze_model()
    finetuned_pipline.infer_dataset()

    predictions_ft_path = PROJECT_ROOT/'lab_8_sft'/'dist'/'predictions.csv'
    predictions_path.parent.mkdir(exist_ok=True)
    predictions_df = finetuned_pipline.infer_dataset()
    # print(predictions_df.predictions)
    predictions_df.to_csv(predictions_path, index=False)

    evaluator_ft = TaskEvaluator(predictions_ft_path, settings.parameters.metrics)
    metric_ft = evaluator_ft.run()

    result = metric_ft
    assert result is not None, "Finetuning does not work correctly"

if __name__ == "__main__":
    main()
