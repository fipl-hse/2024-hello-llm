"""
Starter for demonstration of laboratory work.
"""
# pylint: disable= too-many-locals, undefined-variable, unused-import
from pathlib import Path
from lab_7_llm.main import RawDataImporter, RawDataPreprocessor
from core_utils.llm.time_decorator import report_time
import json


@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    with open('settings.json', 'r', encoding='utf-8') as config_file:
        config_dict = json.load(config_file)

    data_importer = RawDataImporter(config_dict['parameters']['dataset'])
    data_importer.obtain()

    data_preprocessor = RawDataPreprocessor(data_importer.raw_data)
    dataset_properties = data_preprocessor.analyze()

    result = dataset_properties
    assert result is not None, "Demo does not work correctly"

if __name__ == "__main__":
    main()
