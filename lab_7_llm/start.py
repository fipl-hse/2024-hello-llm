"""
Starter for demonstration of laboratory work.
"""
# pylint: disable= too-many-locals, undefined-variable, unused-import
from pathlib import Path
import pandas as pd
from config.constants import PROJECT_ROOT
from config.lab_settings import LabSettings
from core_utils.llm.time_decorator import report_time
from lab_7_llm.main import (RawDataImporter, RawDataPreprocessor, TaskDataset,
                            LLMPipeline, TaskEvaluator)

@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    lab_path = PROJECT_ROOT / 'lab_7_llm'
    settings = LabSettings(lab_path / 'settings.json')
    dist_path = lab_path / 'dist'
    dist_path.mkdir(exist_ok=True)
    predictions_path = dist_path / 'predictions.csv'

    importer = RawDataImporter(settings.parameters.dataset)
    importer.obtain()

    if not isinstance(importer.raw_data, pd.DataFrame):
        raise TypeError('The downloaded dataset is not pd.DataFrame.')

    preprocessor = RawDataPreprocessor(importer.raw_data)
    key_properties = preprocessor.analyze()
    preprocessor.transform()

    dataset = TaskDataset(preprocessor.data.head(100))

    pipeline = LLMPipeline(settings.parameters.model,
                           dataset,
                           max_length = 120,
                           batch_size = 1,
                           device = 'cpu')
    pipeline.analyze_model()
    pipeline.infer_sample(dataset[0])

    dataset_inference = pipeline.infer_dataset()
    dataset_inference.to_csv(predictions_path)

    evaluator = TaskEvaluator(predictions_path, settings.parameters.metrics)
    metrics = evaluator.run()

    result = metrics
    assert result is not None, "Demo does not work correctly"


if __name__ == "__main__":
    main()
