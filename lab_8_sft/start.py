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
    settings = LabSettings(PROJECT_ROOT / "lab_8_sft" / "settings.json")

    importer = RawDataImporter(settings.parameters.dataset)
    importer.obtain()

    if not isinstance(importer.raw_data, pd.DataFrame):
        raise TypeError('Obtained dataset is not pd.DataFrame')

    preprocessor = RawDataPreprocessor(importer.raw_data)
    ds_analysis = preprocessor.analyze()
    print(ds_analysis)
    preprocessor.transform()

    dataset = TaskDataset(preprocessor.data.head(100))

    pipeline = LLMPipeline(settings.parameters.model,
                           dataset,
                           max_length=120,
                           batch_size=64,
                           device='cpu')
    model_analysis = pipeline.analyze_model()
    print(model_analysis)

    sample = pipeline.infer_sample(dataset[1])
    print('Sample res:', sample)

    infered_df = pipeline.infer_dataset()

    path_to_outputs = PROJECT_ROOT / 'lab_8_sft' / 'dist' / 'predictions.csv'
    path_to_outputs.parent.mkdir(exist_ok=True)
    infered_df.to_csv(path_to_outputs, index=False)

    evaluation = TaskEvaluator(path_to_outputs, settings.parameters.metrics)
    results = evaluation.run()
    print('Metrics before tuning', results)

    sft_params = SFTParams(
                batch_size=3,
                max_length=120,
                max_fine_tuning_steps=50,
                learning_rate=1e-3,
                finetuned_model_path=PROJECT_ROOT / "lab_8_sft" / "dist" / settings.parameters.model,
                device="cpu"
            )
    num_samples = 10
    fine_tune_samples = sft_params.batch_size * sft_params.max_fine_tuning_steps

    tok_dataset = TokenizedTaskDataset(preprocessor.data.loc[
                                             num_samples: num_samples + fine_tune_samples],
                                       max_length=sft_params.max_length,
                                       tokenizer=AutoTokenizer
                                       .from_pretrained(settings.parameters.model))

    sft_pipeline = SFTPipeline(settings.parameters.model,
                               tok_dataset,
                               sft_params)
    sft_pipeline.run()

    pipeline = LLMPipeline(PROJECT_ROOT / "lab_8_sft" / "dist" / settings.parameters.model,
                           TaskDataset(preprocessor.data.head(num_samples)),
                           max_length=120,
                           batch_size=64,
                           device='cpu')

    model_analysis = pipeline.analyze_model()
    print(model_analysis)

    sample = pipeline.infer_sample(dataset[1])
    print('Sample res:', sample)

    infered_df = pipeline.infer_dataset()
    path_to_outputs = PROJECT_ROOT / 'lab_8_sft' / 'dist' / 'predictions.csv'
    path_to_outputs.parent.mkdir(exist_ok=True)
    infered_df.to_csv(path_to_outputs, index=False)

    evaluation = TaskEvaluator(path_to_outputs, settings.parameters.metrics)
    result = evaluation.run()
    print('Metrics after tuning', result)

    assert result is not None, "Finetuning does not work correctly"


if __name__ == "__main__":
    main()
