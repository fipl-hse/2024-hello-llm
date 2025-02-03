"""
Starter for demonstration of laboratory work.
"""
# pylint: disable= too-many-locals, undefined-variable, unused-import
from pathlib import Path
from core_utils.llm.time_decorator import report_time
from lab_7_llm.main import RawDataImporter, RawDataPreprocessor
import json


@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    with open(Path(__file__).parent / 'settings.json', 'r', encoding='utf-8') as file:
        config_file = json.load(file)

    importer = RawDataImporter(config_file['parameters']['dataset'])
    importer.obtain()
    preprocessor = RawDataPreprocessor(importer.raw_data)
    print(preprocessor.analyze())


if __name__ == "__main__":
    main()
