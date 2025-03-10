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
    settings = LabSettings(PROJECT_ROOT/"lab_8_sft"/"settings.json")

    importer = RawDataImporter(settings.parameters.dataset)
    importer.obtain()

    if importer.raw_data is None:
        return

    preprocessor = RawDataPreprocessor(importer.raw_data)
    preprocessor.transform()

    predictions_path = PROJECT_ROOT / "lab_8_sft" / "dist" / "predictions.csv"
    predictions_path.parent.mkdir(exist_ok=True)

    num_samples = 10
    sft_params = SFTParams(
        batch_size=3,
        max_length=120,
        max_fine_tuning_steps=200,
        learning_rate=1e-3,
        finetuned_model_path=PROJECT_ROOT / "lab_8_sft" / "dist" / settings.parameters.model,
        device="cpu",
        target_modules=["query", "key", "value"]
    )
    fine_tune_samples = sft_params.batch_size * sft_params.max_fine_tuning_steps
    tokenized_dataset = TokenizedTaskDataset(
        preprocessor.data.loc[num_samples: num_samples + fine_tune_samples],
        tokenizer=AutoTokenizer.from_pretrained(settings.parameters.model),
        max_length=sft_params.max_length
    )
    sft_pipeline = SFTPipeline(model_name=settings.parameters.model,
                               dataset=tokenized_dataset,
                               sft_params=sft_params)
    sft_pipeline.run()

    tokenizer = AutoTokenizer.from_pretrained(settings.parameters.model)
    tokenizer.save_pretrained(settings.parameters.model)

    pipeline = LLMPipeline(
        model_name=str(PROJECT_ROOT / "lab_8_sft" / "dist" / settings.parameters.model),
        dataset=TaskDataset(preprocessor.data.head(10)),
        max_length=120,
        batch_size=64,
        device="cpu"
    )

    pipeline.infer_dataset()

    predictions_dataframe = pipeline.infer_dataset()
    predictions_dataframe.to_csv(predictions_path)
    evaluator = TaskEvaluator(predictions_path, settings.parameters.metrics)

    result = evaluator.run()

    assert result is not None, "Finetuning does not work correctly"


if __name__ == "__main__":
    main()
