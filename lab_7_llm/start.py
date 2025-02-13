"""
Starter for demonstration of laboratory work.
"""
# pylint: disable= too-many-locals, undefined-variable, unused-import
from pathlib import Path
from config.constants import PROJECT_ROOT
from config.lab_settings import LabSettings
from core_utils.llm.time_decorator import report_time
from lab_7_llm.main import (
    RawDataImporter,
    RawDataPreprocessor
)


@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    settings = LabSettings(PROJECT_ROOT / 'lab_7_llm' / 'settings.json')
    importer = RawDataImporter(settings.parameters.dataset)
    importer.obtain()
    if importer.raw_data is None:
        return
    preprocessor = RawDataPreprocessor(importer.raw_data)
    print(preprocessor.analyze())

    result = preprocessor.analyze()
    assert result is not None, "Demo does not work correctly"


if __name__ == "__main__":
    main()
