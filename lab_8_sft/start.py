"""
Fine-tuning starter.
"""
# pylint: disable=too-many-locals, undefined-variable, unused-import, too-many-branches, too-many-statements
from pathlib import Path

import json
from config.constants import PROJECT_ROOT
from core_utils.llm.time_decorator import report_time
from lab_8_sft.main import (RawDataImporter, RawDataPreprocessor, TaskDataset, LLMPipeline)


@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    with open(PROJECT_ROOT / 'lab_8_sft' / 'settings.json', 'r', encoding='utf-8') as file:
        config_file = json.load(file)

    importer = RawDataImporter(config_file['parameters']['dataset'])
    importer.obtain()
    if importer.raw_data is None:
        return

    preprocessor = RawDataPreprocessor(importer.raw_data)
    print(preprocessor.analyze())
    preprocessor.transform()

    dataset = TaskDataset(preprocessor.data.head(100))

    pipeline = LLMPipeline(config_file['parameters']['model'], dataset, max_length=120,
                           batch_size=1, device='cpu')

    result = pipeline.analyze_model()
    print(result)
    assert result is not None, "Finetuning does not work correctly"


if __name__ == "__main__":
    main()
