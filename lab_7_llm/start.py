"""
Starter for demonstration of laboratory work.
"""
# pylint: disable= too-many-locals, undefined-variable, unused-import
import json
from pathlib import Path

from core_utils.llm.time_decorator import report_time
from lab_7_llm.main import RawDataImporter, RawDataPreprocessor, TaskDataset, LLMPipeline


@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    with open(Path(__file__).parent / 'settings.json', 'r', encoding='utf-8') as file:
        config_file = json.load(file)

    importer = RawDataImporter(config_file['parameters']['dataset'])
    importer.obtain()
    preprocessor = RawDataPreprocessor(importer.raw_data)
    #preprocessor.analyze()
    preprocessor.transform()
    dataset = TaskDataset(preprocessor.data.head(100))
    pipeline = LLMPipeline(config_file['parameters']['model'], dataset, max_length=120, batch_size=64, device='cpu')
    #print(pipeline.analyze_model())
    infer_sample_result = pipeline.infer_sample(dataset[1])
    assert infer_sample_result is not None, "Demo does not work correctly"


if __name__ == "__main__":
    main()
