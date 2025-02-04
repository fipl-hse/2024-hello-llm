"""
Starter for demonstration of laboratory work.
"""
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

# pylint: disable= too-many-locals, undefined-variable, unused-import

@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    settings = LabSettings(PROJECT_ROOT / 'lab_7_llm' / 'settings.json')
    importer = RawDataImporter(settings.parameters.dataset)
    importer.obtain()

    preprocessor = RawDataPreprocessor(importer.raw_data)
    key_properties = preprocessor.analyze()
    print(key_properties)

    preprocessor.transform()
    dataset = TaskDataset(preprocessor.data.head(100))

    pipeline = LLMPipeline(settings.parameters.model,
                           dataset,
                           max_length=120,
                           batch_size=1,
                           device="cpu")

    pipeline.analyze_model()
    sample_inference = pipeline.infer_sample(dataset[0])

    result = sample_inference

    assert result is not None, "Demo does not work correctly"


if __name__ == "__main__":
    main()
