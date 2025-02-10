"""
Starter for demonstration of laboratory work.
"""
# pylint: disable= too-many-locals, undefined-variable, unused-import
import json
from pathlib import Path
from lab_7_llm.main import RawDataImporter, RawDataPreprocessor, TaskDataset, LLMPipeline
from core_utils.llm.time_decorator import report_time


@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    with open(Path(__file__).parent / "settings.json", encoding="utf-8") as f:
        settings = json.load(f)

    importer = RawDataImporter(settings["parameters"]["dataset"])
    importer.obtain()

    preprocessor = RawDataPreprocessor(importer.raw_data)
    result = preprocessor.analyze()

    dataset = TaskDataset(preprocessor.data.head(100))
    batch_size = 1
    max_length = 194
    device = 'cpu'
    pipeline = LLMPipeline(settings.parameters.model, dataset, max_length, batch_size, device)

    assert result is not None, "Demo does not work correctly"


if __name__ == "__main__":
    main()
