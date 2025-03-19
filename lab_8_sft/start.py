"""
Fine-tuning starter.
"""
# pylint: disable=too-many-locals, undefined-variable, unused-import, too-many-branches, too-many-statements
import os
from pathlib import Path

from config.constants import PROJECT_ROOT
from config.lab_settings import LabSettings
from core_utils.llm.time_decorator import report_time
from lab_8_sft.main import LLMPipeline, RawDataImporter, RawDataPreprocessor, SFTPipeline, TaskDataset, TaskEvaluator

@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    settings = LabSettings(PROJECT_ROOT / "lab_8_sft" / "settings.json")

    importer = RawDataImporter(settings.parameters.dataset)
    importer.obtain()
    if importer.raw_data is None:
        return

    preprocessor = RawDataPreprocessor(importer.raw_data)
    analysis = preprocessor.analyze()
    print(analysis)

    preprocessor.transform()
    dataset = TaskDataset(preprocessor.data.head(100))
    pipeline = LLMPipeline(settings.parameters.model,
                           dataset,
                           max_length=120,
                           batch_size=1,
                           device='cpu')
    model_analysis = pipeline.analyze_model()
    print(model_analysis)

    sample = dataset[0]
    infer_sample_result  = pipeline.infer_sample(sample)
    print(infer_sample_result)

    if not os.path.exists(PROJECT_ROOT / 'lab_8_sft' / 'dist'):
        os.mkdir(PROJECT_ROOT / 'lab_8_sft' / 'dist')

    path_to_predictions = PROJECT_ROOT / 'lab_8_sft' / 'dist' / 'predictions.csv'
    pipeline.infer_dataset().to_csv(path_to_predictions, index=False)

    evaluator = TaskEvaluator(data_path=Path(path_to_predictions),
                              metrics=settings.parameters.metrics)
    result = evaluator.run()
    print(result)

    assert result is not None, "Finetuning does not work correctly"


if __name__ == "__main__":
    main()
