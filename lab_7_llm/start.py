"""
Starter for demonstration of laboratory work.
"""
# pylint: disable= too-many-locals, undefined-variable, unused-import
from pathlib import Path
from lab_7_llm.main import RawDataImporter, RawDataPreprocessor, report_time
import json

@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    result = None

    with open(Path(__file__).parent / "settings.json", encoding="utf-8") as f:
        settings_dict = json.load(f)


    importer = RawDataImporter(settings_dict["parameters"]["dataset"])
    importer.obtain()

    preprocessor = RawDataPreprocessor(importer.raw_data)
    analysis = preprocessor.analyze()

    result = analysis

    assert result is not None, "Demo does not work correctly"



if __name__ == "__main__":
    main()
