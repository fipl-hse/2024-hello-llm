"""
Starter for demonstration of laboratory work.
"""
# pylint: disable= too-many-locals, undefined-variable, unused-import
from pathlib import Path
from config.constants import PROJECT_ROOT
from config.lab_settings import LabSettings
from lab_7_llm.main import RawDataImporter, report_time, RawDataPreprocessor


@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """

    settings = LabSettings(PROJECT_ROOT/'lab_7_llm'/'settings.json')
    importer = RawDataImporter(settings.parameters.dataset)
    importer.obtain()
    preprocesser = RawDataPreprocessor(importer.raw_data)
    preprocesser.analyze()
    result = None
    assert result is not None, "Demo does not work correctly"

if __name__ == "__main__":
    main()

