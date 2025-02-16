"""
Starter for demonstration of laboratory work.
"""
# pylint: disable= too-many-locals, undefined-variable, unused-import
from pathlib import Path

from config.constants import PROJECT_ROOT
from config.lab_settings import LabSettings
from lab_7_llm.main import (
    LLMPipeline,
    RawDataImporter,
    RawDataPreprocessor,
    report_time,
    TaskDataset,
    TaskEvaluator,
)

SETTINGS_PATH = PROJECT_ROOT / 'lab_7_llm/settings.json'

@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    settings = LabSettings(SETTINGS_PATH)

    importer = RawDataImporter(settings.parameters.dataset)
    importer.obtain()

    preprocessor = RawDataPreprocessor(importer.raw_data)
    preprocessor.transform()

    dataset = TaskDataset(preprocessor.data.head(100))
    pipeline = LLMPipeline(settings.parameters.model,
                           dataset,
                           max_length=120,
                           batch_size=1,
                           device="cpu")

    print(pipeline.analyze_model())
    result = preprocessor.analyze()

    predictions_path = PROJECT_ROOT / 'lab_7_llm/dist/predictions.csv'
    predictions_path.parent.mkdir(exist_ok=True)
    sample_inference_result = pipeline.infer_sample(dataset[0])
    print('Sample inference result:', sample_inference_result, sep='\n')

    pipeline.infer_dataset().to_csv(predictions_path, index=False)

    evaluator = TaskEvaluator(predictions_path, settings.parameters.metrics)
    result = evaluator.run()
    print('Resulting quality:', result, sep='\n')

    assert result is not None, "Demo does not work correctly"


if __name__ == "__main__":
    main()
