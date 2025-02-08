"""
Starter for demonstration of laboratory work.
"""
import json
import random

# pylint: disable= too-many-locals, undefined-variable, unused-import
from pathlib import Path

from core_utils.llm.time_decorator import report_time
from lab_7_llm.main import LLMPipeline, RawDataImporter, RawDataPreprocessor, TaskDataset


@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    with open(Path(__file__).parent / 'settings.json', 'r', encoding='utf-8') as config_file:
        config_dict = json.load(config_file)

    data_importer = RawDataImporter(config_dict['parameters']['dataset'])
    data_importer.obtain()

    data_preprocessor = RawDataPreprocessor(data_importer.raw_data)
    dataset_properties = data_preprocessor.analyze()
    data_preprocessor.transform()

    preprocessed_dataset = TaskDataset(data_preprocessor.data.head(100))
    MAX_LENGTH = 120
    BATCH_SIZE = 1
    DEVICE = 'cpu'
    pipeline = LLMPipeline(config_dict['parameters']['model'], preprocessed_dataset, MAX_LENGTH, BATCH_SIZE, DEVICE)
    model_params = pipeline.analyze_model()

    sample = preprocessed_dataset[random.randint(0, len(preprocessed_dataset))]
    print(*sample)
    prediction = pipeline.infer_sample(sample)
    print(prediction)

    result = prediction
    assert result is not None, "Demo does not work correctly"

if __name__ == "__main__":
    main()
