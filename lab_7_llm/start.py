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

    importer = RawDataImporter(settings.parameters.dataset)
    importer.obtain()

    preprocessor = RawDataPreprocessor(importer.raw_data)
    dataset_analysis = preprocessor.analyze()
    preprocessor.transform()

    dataset = TaskDataset(preprocessor.data.head(100))
    BATCH_SIZE = 1
    MAX_LENGTH = 120
    DEVICE = 'cpu'
    pipeline = LLMPipeline(settings.parameters.model, dataset, MAX_LENGTH, BATCH_SIZE, DEVICE)
    model_analysis = pipeline.analyze_model()
    SAMPLE = dataset[0]
    generated_sample = pipeline.infer_sample(SAMPLE)

    result = generated_sample

    assert result is not None, "Demo does not work correctly"


if __name__ == "__main__":
    main()
