"""
Starter for demonstration of laboratory work.
"""
# pylint: disable= too-many-locals, undefined-variable, unused-import
import json
from pathlib import Path
from random import randint

from config.constants import PROJECT_ROOT
from core_utils.llm.metrics import Metrics
from core_utils.llm.time_decorator import report_time
from lab_7_llm.main import (LLMPipeline, RawDataImporter, RawDataPreprocessor, TaskDataset,
                            TaskEvaluator)


@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    with open(PROJECT_ROOT / 'lab_7_llm' / 'settings.json', 'r', encoding='utf-8') as file:
        settings = json.load(file)

    importer = RawDataImporter(settings['parameters']['dataset'])
    importer.obtain()

    if importer.raw_data is not None:
        preprocessor = RawDataPreprocessor(importer.raw_data)
    else:
        preprocessor = None
    print(preprocessor.analyze())
    preprocessor.transform()

    dataset = TaskDataset(preprocessor.data.head(100))
    print(dataset[0])

    pipeline = LLMPipeline(settings['parameters']['model'], dataset, 120, 64, 'cpu')
    print(pipeline.analyze_model())

    print(pipeline.infer_sample(dataset[randint(0, len(dataset)-1)]))

    predictions = pipeline.infer_dataset()

    Path(PROJECT_ROOT / 'lab_7_llm'/ 'dist').mkdir(exist_ok=True)
    predictions_path = PROJECT_ROOT / 'lab_7_llm'/ 'dist' / 'predictions.csv'
    predictions.to_csv(predictions_path)

    metrics_list = []
    for metric in settings['parameters']['metrics']:
        metrics_list.append(Metrics(metric))
    evaluator = TaskEvaluator(predictions_path, metrics_list)
    result = evaluator.run()
    print(result)

    assert result is not None, "Demo does not work correctly"


if __name__ == "__main__":
    main()
