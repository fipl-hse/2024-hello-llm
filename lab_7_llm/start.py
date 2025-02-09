"""
Starter for demonstration of laboratory work.
"""
# pylint: disable= too-many-locals, undefined-variable, unused-import
from pathlib import Path
from lab_7_llm.main import RawDataImporter, RawDataPreprocessor
from core_utils.llm.time_decorator import report_time


@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    importer = RawDataImporter("lionelchg/dolly_open_qa")
    importer.obtain()
    preprocessor = RawDataPreprocessor(importer.raw_data)
    df_analysis = preprocessor.analyze()
    result = None
    assert result is not None, "Demo does not work correctly"


if __name__ == "__main__":
    main()
