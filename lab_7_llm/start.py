"""
Starter for demonstration of laboratory work.
"""
# pylint: disable= too-many-locals, undefined-variable, unused-import
from pathlib import Path
import json


from core_utils.llm.time_decorator import report_time
from lab_7_llm.main import RawDataImporter, RawDataPreprocessor, TaskDataset, LLMPipeline, TaskEvaluator


@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    settings_path = Path(__file__).parent / 'settings.json'

    with open(settings_path, 'r', encoding='utf-8') as f:
        parameters = json.load(f)['parameters']

    importer = RawDataImporter(parameters['dataset'])
    importer.obtain()

    preprocessor = RawDataPreprocessor(importer.raw_data)
    # properties = preprocessor.analyze()
    # print(properties)
    preprocessor.transform()
    # print(preprocessor.data.head())

    dataset = TaskDataset(preprocessor.data.head(100))

    batch_size = 2
    max_length = 120
    device = 'cpu'

    pipeline = LLMPipeline(parameters['model'], dataset, max_length, batch_size, device)
    # model_summary = pipeline.analyze_model()
    # print(model_summary)

    prediction = pipeline.infer_sample(dataset[0])
    print(prediction)
    # predictions = pipeline.infer_dataset()
    # print(predictions.head(10))

    predictions_path = Path(__file__).parent / 'dist' / 'predictions.csv'
    # evaluator = TaskEvaluator(predictions_path, parameters['metrics'])

    # result = None
    # assert result is not None, "Demo does not work correctly"


if __name__ == "__main__":
    main()
