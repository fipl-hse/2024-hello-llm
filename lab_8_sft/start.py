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
    settings = LabSettings(PROJECT_ROOT / "lab_8_sft" / "settings.json")
    predictions_path = Path(PROJECT_ROOT / 'lab_8_sft' / 'dist')
    if not predictions_path.exists():
        predictions_path.mkdir()

    importer = RawDataImporter(settings.parameters.dataset)
    importer.obtain()
    if importer.raw_data is None:
        return

    preprocessor = RawDataPreprocessor(importer.raw_data)
    print(preprocessor.analyze())
    preprocessor.transform()
    dataset = TaskDataset(preprocessor.data.head(100))

    pipeline = LLMPipeline(settings.parameters.model, dataset,  max_length=120, batch_size=64,  device='cpu')
    print(pipeline.analyze_model())
    pipeline.infer_dataset().to_csv(predictions_path / 'predictions.csv', index=False)
    evaluator = TaskEvaluator(predictions_path / 'predictions.csv', settings.parameters.metrics)
    print(evaluator.run())

    num_samples = 10
    sft_params = SFTParams(batch_size=3, max_length=120, max_fine_tuning_steps=50, device="cpu",
                           learning_rate=1e-3,
                           finetuned_model_path=predictions_path / settings.parameters.model)
    fine_tune_samples = sft_params.batch_size * sft_params.max_fine_tuning_steps
    tokenized_dataset = TokenizedTaskDataset(
        preprocessor.data.loc[num_samples:num_samples + fine_tune_samples],
        AutoTokenizer.from_pretrained(settings.parameters.model),
        sft_params.max_length)

    sft_pipeline = SFTPipeline(settings.parameters.model, tokenized_dataset, sft_params)
    sft_pipeline.run()

    pipeline = LLMPipeline(str(predictions_path / settings.parameters.model),
                           TaskDataset(preprocessor.data.head(10)), max_length=120, batch_size=64, device='cpu')
    print(pipeline.analyze_model())
    pipeline.infer_dataset().to_csv(predictions_path / 'SFT-predictions.csv', index=False)
    evaluator = TaskEvaluator(predictions_path / 'SFT-predictions.csv', settings.parameters.metrics)
    result = evaluator.run()
    print(result)

    assert result is not None, "Finetuning does not work correctly"


if __name__ == "__main__":
    main()
