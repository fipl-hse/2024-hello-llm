"""
Fine-tuning starter.
"""
# pylint: disable=too-many-locals, undefined-variable, unused-import, too-many-branches, too-many-statements
from pathlib import Path
from config.constants import PROJECT_ROOT
from config.lab_settings import LabSettings
from lab_8_sft.main import (
    LLMPipeline,
    RawDataImporter,
    RawDataPreprocessor,
    report_time,
    TaskDataset,
    TaskEvaluator,
)

@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    settings = LabSettings(PROJECT_ROOT / 'lab_8_sft/settings.json')

    importer = RawDataImporter(settings.parameters.dataset)
    importer.obtain()
    if importer.raw_data is None:
        return

    preprocessor = RawDataPreprocessor(importer.raw_data)
    dataset_analysis = preprocessor.analyze()
    print('Dataset analysis result:', dataset_analysis, sep='\n')

    preprocessor.transform()

    dataset = TaskDataset(preprocessor.data.head(100))

    batch_size = 64
    max_length = 120
    device = 'cpu'
    pipeline = LLMPipeline(model_name=settings.parameters.model,
                           dataset=dataset,
                           max_length=max_length,
                           batch_size=batch_size,
                           device=device)
    model_analysis = pipeline.analyze_model()
    print('Model analysis result:', model_analysis, sep='\n')

    predictions_path = PROJECT_ROOT / 'lab_8_sft/dist/predictions.csv'
    predictions_path.parent.mkdir(exist_ok=True)
    sample_inference_result = pipeline.infer_sample(dataset[0])
    print('Sample inference result:', sample_inference_result, sep='\n')

    pipeline.infer_dataset().to_csv(predictions_path, index=False)

    evaluator = TaskEvaluator(predictions_path, settings.parameters.metrics)
    result = evaluator.run()
    print('Resulting quality:', result, sep='\n')

    assert result is not None, "Finetuning does not work correctly"


if __name__ == "__main__":
    main()
