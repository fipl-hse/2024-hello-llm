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
    settings = LabSettings(PROJECT_ROOT / "lab_7_llm" / "settings.json")
    importer = RawDataImporter(settings.parameters.dataset)
    importer.obtain()

    preprocessor = RawDataPreprocessor(importer.raw_data)
    dataset_properties = preprocessor.analyze()
    print("dataset properties:", dataset_properties)

    assert dataset_properties is not None, "Demo does not work correctly"


if __name__ == "__main__":
    main()
