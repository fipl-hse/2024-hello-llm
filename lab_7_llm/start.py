"""
Starter for demonstration of laboratory work.
"""
# pylint: disable= too-many-locals, undefined-variable, unused-import
import json
from pathlib import Path

from core_utils.llm.metrics import Metrics
from core_utils.llm.time_decorator import report_time
from lab_7_llm.main import (
    LLMPipeline,
    RawDataImporter,
    RawDataPreprocessor,
    TaskDataset,
    TaskEvaluator,
)


@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    with open(Path(__file__).parent / "settings.json", encoding="utf-8") as f:
        settings = json.load(f)

    importer = RawDataImporter(settings['parameters']['dataset'])
    importer.obtain()

    if importer.raw_data:
        preprocessor = RawDataPreprocessor(importer.raw_data)
        preprocessor.transform()

        batch_size = 1
        max_length = 120
        device = 'cpu'

        dataset = TaskDataset(preprocessor.data.head(100))
        pipeline = LLMPipeline(settings['parameters']['model'],
                               dataset,
                               max_length,
                               batch_size,
                               device)
        predictions = pipeline.infer_dataset()

        predictions_path = Path(__file__).parent / 'dist' / 'predictions.csv'
        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(predictions_path, index=False)

        metrics = [Metrics(metric) for metric in settings['parameters']['metrics']]
        evaluator = TaskEvaluator(predictions_path, metrics)
        result = evaluator.run()

    assert result is not None, "Demo does not work correctly"


if __name__ == "__main__":
    main()
