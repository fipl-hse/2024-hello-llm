"""
Starter for demonstration of laboratory work.
"""
import random
from pathlib import Path

from config.lab_settings import LabSettings
from core_utils.llm.time_decorator import report_time
from lab_7_llm.main import (
    LLMPipeline,
    RawDataImporter,
    RawDataPreprocessor,
    TaskDataset,
    TaskEvaluator,
)

# pylint: disable= too-many-locals, undefined-variable, unused-import


MAX_LENGTH = 120
BATCH_SIZE = 64
DEVICE = 'cpu'
SETTINGS_PATH = Path(__file__).parent / 'settings.json'
PREDICTIONS_PATH = Path(__file__).parent / 'dist' / 'predictions.csv'

@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    settings = LabSettings(SETTINGS_PATH)

    data_importer = RawDataImporter(settings.parameters.dataset)
    data_importer.obtain()

    if data_importer.raw_data is None:
        raise ValueError('No dataset created by obtain() method')

    data_preprocessor = RawDataPreprocessor(data_importer.raw_data)
    _dataset_properties = data_preprocessor.analyze()
    data_preprocessor.transform()

    preprocessed_dataset = TaskDataset(data_preprocessor.data.head(100))
    pipeline = LLMPipeline(settings.parameters.model,
                           preprocessed_dataset,
                           MAX_LENGTH, BATCH_SIZE, DEVICE)
    _model_params = pipeline.analyze_model()

    sample = preprocessed_dataset[random.randint(0, len(preprocessed_dataset))]
    _sample_prediction = pipeline.infer_sample(sample)

    dataset_predictions = pipeline.infer_dataset()

    PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    dataset_predictions.to_csv(PREDICTIONS_PATH)

    evaluator = TaskEvaluator(PREDICTIONS_PATH,
                              settings.parameters.metrics)
    evaluation_result = evaluator.run()
    print(evaluation_result)

    result = evaluation_result
    assert result is not None, "Demo does not work correctly"

if __name__ == "__main__":
    main()
