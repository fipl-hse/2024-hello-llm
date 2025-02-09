"""
Starter for demonstration of laboratory work.
"""
# pylint: disable= too-many-locals, undefined-variable, unused-import
from pathlib import Path
import json

from config.constants import PROJECT_ROOT
from config.lab_settings import LabSettings
from core_utils.llm.time_decorator import report_time
from lab_7_llm.main import RawDataImporter, RawDataPreprocessor, TaskDataset, LLMPipeline, TaskEvaluator


@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    settings_path = PROJECT_ROOT / 'lab_7_llm' / 'settings.json'
    parameters = LabSettings(settings_path).parameters

    importer = RawDataImporter(parameters.dataset)
    importer.obtain()

    preprocessor = RawDataPreprocessor(importer.raw_data)
    preprocessor.transform()

    dataset = TaskDataset(preprocessor.data.head(10))

    batch_size = 2
    max_length = 120
    device = 'cpu'

    pipeline = LLMPipeline(parameters.model, dataset, max_length, batch_size, device)

    predictions = pipeline.infer_dataset()

    predictions_path = PROJECT_ROOT / 'lab_7_llm' / 'dist' / 'predictions.csv'
    evaluator = TaskEvaluator(predictions_path, parameters.metrics)
    evaluator.run()


    # result = None
    # assert result is not None, "Demo does not work correctly"


if __name__ == "__main__":
    main()
