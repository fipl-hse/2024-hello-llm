"""
Fine-tuning starter.
"""
# pylint: disable=too-many-locals, undefined-variable, unused-import, too-many-branches, too-many-statements
from pathlib import Path

from transformers import AutoTokenizer

from config.constants import PROJECT_ROOT
from config.lab_settings import LabSettings, SFTParams
from lab_8_sft.main import (
    LLMPipeline,
    RawDataImporter,
    RawDataPreprocessor,
    report_time,
    TaskDataset,
    TaskEvaluator,
    TokenizedTaskDataset,
    SFTPipeline,
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
        raise ValueError('Received None instead of dataframe')

    preprocessor = RawDataPreprocessor(importer.raw_data)
    print(preprocessor.analyze())
    preprocessor.transform()
    dataset = TaskDataset(preprocessor.data.head(100))

    pipeline = LLMPipeline(settings.parameters.model, dataset, 120, 64, 'cpu')
    infer_dataframe = pipeline.infer_dataset()
    print(pipeline.infer_sample(dataset[0]))
    path_to_outputs = PROJECT_ROOT / 'lab_8_sft' / 'dist' / 'predictions.csv'
    path_to_outputs.parent.mkdir(exist_ok=True)
    infer_dataframe.to_csv(path_to_outputs, index=False)
    evaluation = TaskEvaluator(path_to_outputs, settings.parameters.metrics)
    res = evaluation.run()
    print(f'Ванильные метрики: {res}')

    ft_model_path = PROJECT_ROOT / "lab_8_sft" / "dist" / settings.parameters.model

    sft_params = SFTParams(
        batch_size=3,
        max_length=120,
        max_fine_tuning_steps=50,
        learning_rate=1e-3,
        finetuned_model_path=ft_model_path,
        device="cpu",
        target_modules=''
    )
    num_samples = 15
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

    pipeline = LLMPipeline(str(ft_model_path),
                           TaskDataset(preprocessor.data.head(num_samples)),
                           max_length=120,
                           batch_size=64,
                           device='cpu')

    infer_dataframe = pipeline.infer_dataset()
    path_to_outputs = PROJECT_ROOT / 'lab_8_sft' / 'dist' / 'predictions.csv'
    path_to_outputs.parent.mkdir(exist_ok=True)
    infer_dataframe.to_csv(path_to_outputs, index=False)

    evaluation = TaskEvaluator(path_to_outputs, settings.parameters.metrics)
    res = evaluation.run()
    print(f'Тюнингованные метрики, {res}')

    result = res
    assert result is not None, "Finetuning does not work correctly"


if __name__ == "__main__":
    main()
