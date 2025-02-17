"""
Starter for demonstration of laboratory work.
"""
import json

# pylint: disable= too-many-locals, undefined-variable, unused-import
from pathlib import Path

from config.constants import PROJECT_ROOT
from config.lab_settings import LabSettings
from core_utils.llm.time_decorator import report_time
from lab_7_llm.main import LLMPipeline, RawDataImporter, RawDataPreprocessor, TaskDataset


@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    settings = LabSettings(PROJECT_ROOT / 'lab_7_llm' / 'settings.json')
    importer = RawDataImporter(settings.parameters.dataset)
    importer.obtain()
    preprocessor = RawDataPreprocessor(importer.raw_data)
    preprocessor.transform()
    dataset = TaskDataset(preprocessor.data.head(100))
    pipeline = LLMPipeline(settings.parameters.model, dataset, 120, 64, 'cpu')
    sample = pipeline.infer_sample(dataset[1])
    infer_dataframe = pipeline.infer_dataset()

    path_to_outputs = PROJECT_ROOT / 'lab_7_llm' / 'dist' / 'predictions.csv'
    path_to_outputs.parent.mkdir(exist_ok=True)
    infer_dataframe.to_csv(path_to_outputs, index=False)

    result = pipeline
    assert result is not None, "Demo does not work correctly"


if __name__ == "__main__":
    main()
