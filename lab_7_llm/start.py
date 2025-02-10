"""
Starter for demonstration of laboratory work.
"""
# pylint: disable= too-many-locals, undefined-variable, unused-import
from config.constants import PROJECT_ROOT
from config.lab_settings import LabSettings
from lab_7_llm.main import (
    RawDataImporter,
    RawDataPreprocessor,
    report_time,
    LLMPipeline,
    TaskDataset,
)


@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    settings = LabSettings(PROJECT_ROOT / "lab_7_llm" / "settings.json")

    importer = RawDataImporter(settings.parameters.dataset)
    importer.obtain()

    # data = AbstractRawDataPreprocessor(importer.raw_data)

    preprocessor = RawDataPreprocessor(importer.raw_data)
    preprocessor.transform()

    dataset = TaskDataset(preprocessor.data)

    device = "cpu"
    batch_size = 1
    max_length = 120

    pipeline = LLMPipeline(settings.parameters.model, dataset, max_length, batch_size, device)
    pipeline.analyze_model()

    print(pipeline.infer_sample(dataset[0][1]))
    # assert result is not None, "Demo does not work correctly"


if __name__ == "__main__":
    main()
