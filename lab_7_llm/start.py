"""
Starter for demonstration of laboratory work.
"""
# pylint: disable= too-many-locals, undefined-variable, unused-import
from pathlib import Path

from core_utils.llm.time_decorator import report_time
from lab_7_llm.main import RawDataImporter, RawDataPreprocessor, TaskDataset, LLMPipeline
from config.lab_settings import LabSettings
from config.constants import PROJECT_ROOT
import pandas as pd


@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    settings = LabSettings(PROJECT_ROOT/'lab_7_llm'/'settings.json')
    importer = RawDataImporter(settings.parameters.dataset)
    importer.obtain()

    preprocessor = RawDataPreprocessor(importer.raw_data)
    print(preprocessor.analyze())

    preprocessor.transform()

    dataset = TaskDataset(preprocessor.data.head(100))

    new_dataset = pd.concat([importer.raw_data.head(100), importer.raw_data.tail(100)], ignore_index=True)
    preprocessor2 = RawDataPreprocessor(new_dataset)
    print(preprocessor2.analyze())

    # batch_size = 1
    # max_length = 120
    # device = 'cpu'
    # pipeline = LLMPipeline(settings.parameters.model, dataset, max_length, batch_size, device)
    # print(pipeline.analyze_model())
    # print(pipeline.infer_sample(())

    result = True
    assert result is not None, "Demo does not work correctly"


if __name__ == "__main__":
    main()
