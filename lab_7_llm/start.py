"""
Starter for demonstration of laboratory work.
"""
# pylint: disable= too-many-locals, undefined-variable, unused-import
from pathlib import Path

from core_utils.llm.time_decorator import report_time
from lab_7_llm.main import RawDataImporter, RawDataPreprocessor


@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    importer = RawDataImporter('jtatman/databricks-dolly-8k-qa-open-close')
    importer.obtain()
    #print(importer._raw_data.head())
    preprocessor = RawDataPreprocessor(importer.raw_data)
    result = preprocessor.analyze()
    print(result)
    #result = None
    assert result is not None, "Demo does not work correctly"


if __name__ == "__main__":
    main()
