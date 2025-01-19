"""
Starter for demonstration of laboratory work.
"""
# pylint: disable= too-many-locals, undefined-variable, unused-import
from pathlib import Path

from lab_7_llm.main import RawDataImporter, RawDataPreprocessor, report_time
from config.constants import PROJECT_ROOT
from config.lab_settings import LabSettings


@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    lab_settings = LabSettings(PROJECT_ROOT / 'lab_7_llm' / 'settings.json')
    importer = RawDataImporter(lab_settings.parameters.dataset)
    importer.obtain()

    preprocessor = RawDataPreprocessor(importer.raw_data)
    properties = preprocessor.analyze()
    # print(properties)
    result = properties
    assert result is not None, "Demo does not work correctly"


if __name__ == "__main__":
    main()
