"""
Starter for demonstration of laboratory work.
"""
import json
# pylint: disable= too-many-locals, undefined-variable, unused-import
from pathlib import Path

from core_utils.llm.time_decorator import report_time
from config.constants import PROJECT_ROOT
from lab_7_llm.main import RawDataImporter, RawDataPreprocessor, TaskDataset, LLMPipeline


@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    with open(PROJECT_ROOT/'lab_7_llm'/'settings.json', 'r', encoding='utf-8') as file:
        settings = json.load(file)

    importer = RawDataImporter(settings['parameters']['dataset'])
    importer.obtain()
    preprocessor = RawDataPreprocessor(importer.raw_data)

    # print(preprocessor.analyze())
    preprocessor.transform()
    dataset = TaskDataset(preprocessor.data.head(100))

    pipeline = LLMPipeline(settings['parameters']['model'], dataset, max_length=120, batch_size=64, device='cpu')
    print(pipeline.analyze_model())

    sample = pipeline.infer_sample(dataset[1])
    print(sample)

    result = sample
    assert result is not None, "Demo does not work correctly"


if __name__ == "__main__":
    main()
