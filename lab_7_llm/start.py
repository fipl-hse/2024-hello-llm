"""
Starter for demonstration of laboratory work.
"""
# pylint: disable= too-many-locals, undefined-variable, unused-import
from pathlib import Path
from config.constants import PROJECT_ROOT
from config.lab_settings import LabSettings
import pandas as pd
from core_utils.llm.time_decorator import report_time
from lab_7_llm.main import (RawDataImporter, RawDataPreprocessor)

@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    lab_path = PROJECT_ROOT / 'lab_7_llm'
    settings = LabSettings(lab_path / 'settings.json')
    importer = RawDataImporter(settings.parameters.dataset)
    importer.obtain()

    if not isinstance(importer.raw_data, pd.DataFrame):
        raise TypeError('The downloaded dataset is not pd.DataFrame.')

    preprocessor = RawDataPreprocessor(importer.raw_data)
    key_properties = preprocessor.analyze()
    result = key_properties
    assert result is not None, "Demo does not work correctly"


if __name__ == "__main__":
    main()
