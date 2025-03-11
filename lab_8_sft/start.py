"""
Fine-tuning starter.
"""
# pylint: disable=too-many-locals, undefined-variable, unused-import, too-many-branches, too-many-statements
from pathlib import Path

from transformers import BertTokenizerFast, set_seed

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
    set_seed(42)

    # Inference of a base model
    settings = LabSettings(PROJECT_ROOT / "lab_8_sft" / "settings.json")

    importer = RawDataImporter(settings.parameters.dataset)
    importer.obtain()

    if importer.raw_data is None:
        return None

    preprocessor = RawDataPreprocessor(importer.raw_data)
    analysis = preprocessor.analyze()
    print("Dataset information:")
    for analyze in analysis:
        print(rf"{analyze}: {analysis[analyze]}")
    preprocessor.transform()

    dataset = TaskDataset(preprocessor.data.head(100))

    pipeline = LLMPipeline(settings.parameters.model, dataset, 120, 64, "cpu")
    analysis = pipeline.analyze_model()
    print("Base model information:")
    for analyze in analysis:
        print(rf"{analyze}: {analysis[analyze]}")

    inference = pipeline.infer_dataset()
    output_path = Path(__file__).parent / "dist" / "predictions.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    inference.to_csv(output_path, index=False)

    evaluator = TaskEvaluator(output_path, settings.parameters.metrics)
    evaluations = evaluator.run()
    print("Base model evaluation:")
    for evaluation in evaluations:
        print(rf"{evaluation}: {evaluations[evaluation]}")
    print()

    # Inference of sft model
    # sft parameters
    batch = 3
    num_samples = 10
    max_length = 120
    fine_tuning_steps = 100
    learning_rate = 1e-3
    fine_tune_samples = batch * fine_tuning_steps

    tokenizer = BertTokenizerFast.from_pretrained(settings.parameters.model)
    sft_dataset = TokenizedTaskDataset(
        preprocessor.data.loc[num_samples : num_samples + fine_tune_samples], tokenizer, max_length
    )

    output_path = Path(__file__).parent / "dist"
    sft_params = SFTParams(
        max_length=max_length,
        batch_size=batch,
        max_fine_tuning_steps=fine_tuning_steps,
        finetuned_model_path=output_path,
        device="cpu",
        learning_rate=learning_rate,
        target_modules=None,
    )
    sft_pipeline = SFTPipeline(settings.parameters.model, sft_dataset, sft_params)
    sft_pipeline.run()
    model_path = Path(__file__).parent / "dist"

    # parameters for sft model inference
    batch_size = 64

    dataset = TaskDataset(preprocessor.data.head(11))
    sft_inference = LLMPipeline(str(model_path), dataset, max_length, batch_size, "cpu")
    sft_analyze = sft_inference.analyze_model()
    print("SFT model information:")
    for analyze in sft_analyze:
        print(rf"{analyze}: {analysis[analyze]}")

    inference = sft_inference.infer_dataset()
    output_path = Path(__file__).parent / "dist" / "predictions.csv"
    inference.to_csv(output_path, index=False)

    sft_evaluator = TaskEvaluator(output_path, settings.parameters.metrics)
    sft_evaluations = sft_evaluator.run()
    print("SFT model evaluation:")
    for evaluation in sft_evaluations:
        print(rf"{evaluation}: {sft_evaluations[evaluation]}")

    result = sft_evaluations
    assert result is not None, "Finetuning does not work correctly"
    return None


if __name__ == "__main__":
    main()
