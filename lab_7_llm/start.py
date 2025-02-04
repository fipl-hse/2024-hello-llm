"""
Starter for demonstration of laboratory work.
"""
# pylint: disable= too-many-locals, undefined-variable, unused-import
from pathlib import Path
from main import RawDataImporter, RawDataPreprocessor
import json
from core_utils.llm.time_decorator import report_time
import numpy as np
import pandas as pd


@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    # path = Path(r'lab_7_llm\\settings.json')

    with open('settings.json', 'r') as f:
        settings = json.load(f)

    importer = RawDataImporter(settings['parameters']['dataset'])
    importer.obtain()

    preprocessor = RawDataPreprocessor(importer.raw_data)

    result = preprocessor.analyze()
    assert result is not None, "Demo does not work correctly"

if __name__ == "__main__":
    main()
