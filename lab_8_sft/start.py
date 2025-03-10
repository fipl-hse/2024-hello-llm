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
    TaskDataset,
    TaskEvaluator
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

    pipeline = LLMPipeline(settings.parameters.model, dataset, max_length=120,
                           batch_size=64, device='cpu')
    print(pipeline.analyze_model())
    pipeline.infer_dataset().to_csv(predictions_path / 'predictions.csv', index=False)
    evaluator = TaskEvaluator(predictions_path / 'predictions.csv', settings.parameters.metrics)
    # print(evaluator.run())
    result = evaluator.run()
    print(result)

    assert result is not None, "Finetuning does not work correctly"


if __name__ == "__main__":
    main()
