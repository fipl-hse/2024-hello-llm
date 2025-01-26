"""
Starter for demonstration of laboratory work.
"""
# pylint: disable= too-many-locals, undefined-variable, unused-import
from pathlib import Path

from config.lab_settings import LabSettings
from config.constants import PROJECT_ROOT

from core_utils.llm.time_decorator import report_time
from lab_7_llm.main import RawDataImporter, RawDataPreprocessor, TaskDataset

@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    settings = LabSettings(PROJECT_ROOT / "lab_7_llm" / "settings.json")

    importer = RawDataImporter(settings.parameters.dataset)
    importer.obtain()

    preprocessor = RawDataPreprocessor(importer.raw_data)
    analysis = preprocessor.analyze()

    #dataset = TaskDataset(preprocessor.data.head(100))
    #pipeline = LLMPipeline(settings.parameters.model, dataset, max_length, batch_size, device)

    result = analysis
    assert result is not None, "Demo does not work correctly"


if __name__ == "__main__":
    main()
