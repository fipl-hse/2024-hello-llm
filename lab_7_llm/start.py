"""
Starter for demonstration of laboratory work.
"""
# pylint: disable= too-many-locals, undefined-variable, unused-import
from pathlib import Path

from config.constants import PROJECT_ROOT
from main import RawDataImporter, RawDataPreprocessor
from core_utils.llm.time_decorator import report_time
from config.lab_settings import LabSettings





@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    settings = LabSettings(PROJECT_ROOT / 'lab_7_llm' / 'settings.json')
    dataset_name = settings.parameters.dataset

    importer = RawDataImporter(dataset_name)

    importer.obtain()

    preprocessor = RawDataPreprocessor(importer.raw_data)
    analysis_results = preprocessor.analyze()

    result = analysis_results

    assert result is not None, "Demo does not work correctly"


if __name__ == "__main__":
    main()
