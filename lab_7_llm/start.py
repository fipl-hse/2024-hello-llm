"""
Starter for demonstration of laboratory work.
"""
# pylint: disable= too-many-locals, undefined-variable, unused-import
from config.constants import PROJECT_ROOT
from config.lab_settings import LabSettings
from lab_7_llm.main import (
    RawDataImporter,
    RawDataPreprocessor,
    report_time
)


@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    importer = RawDataImporter("trixdade/reviews_russian")
    raw_data = importer.obtain()

    if raw_data is not None:
        preprocessor = RawDataPreprocessor(raw_data)
        dataset_analysis = preprocessor.analyze()
        print("Dataset Analysis:", dataset_analysis)

        preprocessor.transform()
        print("Processed data preview:", preprocessor.data.head())
    else:
        print("Failed to obtain data.")


if __name__ == "__main__":
    main()
