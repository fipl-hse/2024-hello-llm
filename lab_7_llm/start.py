"""
Starter for demonstration of laboratory work.
"""
# pylint: disable= too-many-locals, undefined-variable, unused-import
from pathlib import Path

import json
from core_utils.llm.time_decorator import report_time
from config.constants import PROJECT_ROOT
from lab_7_llm.main import RawDataImporter, RawDataPreprocessor, TaskDataset


@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    with open(PROJECT_ROOT/'lab_7_llm'/'settings.json', 'r', encoding='utf-8') as file:
        settings = json.load(file)

    importer = RawDataImporter(settings['parameters']['dataset'])
    importer.obtain()

    preprocessor = RawDataPreprocessor(importer.raw_data)
    analysis = preprocessor.analyze()
    print(analysis)

    preprocessor.transform()
    dataset = TaskDataset(preprocessor.data.head(10))

    # idx = 1
    # item = dataset[idx]
    #
    # print(f"Item at index {idx}:")
    # print(item)

    result = dataset
    assert result is not None, "Demo does not work correctly"


if __name__ == "__main__":
    main()
