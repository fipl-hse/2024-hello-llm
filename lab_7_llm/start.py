"""
Starter for demonstration of laboratory work.
"""
# pylint: disable= too-many-locals, undefined-variable, unused-import
from pathlib import Path
import json

from config.constants import PROJECT_ROOT

from core_utils.llm.time_decorator import report_time

from lab_7_llm.main import LLMPipeline, RawDataImporter, RawDataPreprocessor, TaskDataset


@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    with open(PROJECT_ROOT / 'lab_7_llm' / 'settings.json', 'r', encoding='utf-8') as file:
        settings = json.load(file)

    importer = RawDataImporter(settings['parameters']['dataset'])
    importer.obtain()
    preprocessor = RawDataPreprocessor(importer.raw_data)
    analysis = preprocessor.analyze()
    print(analysis)
    preprocessor.transform()
    dataset = TaskDataset(preprocessor.data.head(100))
    print(len(dataset))
    print(dataset[1])
    pipeline = LLMPipeline(settings['parameters']['model'], dataset, max_length=120, batch_size=1, device='cpu')
    print(pipeline.analyze_model())
    result = pipeline
    assert result is not None, "Demo does not work correctly"


if __name__ == "__main__":
    main()
