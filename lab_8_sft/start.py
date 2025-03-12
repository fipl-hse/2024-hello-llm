"""
Fine-tuning starter.
"""
# pylint: disable=too-many-locals, undefined-variable, unused-import, too-many-branches, too-many-statements
from pathlib import Path
from config.constants import PROJECT_ROOT
from config.lab_settings import LabSettings
from core_utils.llm.time_decorator import report_time
from lab_8_sft.main import (
    LLMPipeline,
    RawDataImporter,
    RawDataPreprocessor,
    TaskDataset
)


@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    settings = LabSettings(PROJECT_ROOT / 'lab_8_sft' / 'settings.json')

    importer = RawDataImporter(settings.parameters.dataset)
    importer.obtain()
    preprocessor = RawDataPreprocessor(importer.raw_data)
    preprocessor.analyze()
    preprocessor.transform()
    dataset = TaskDataset(preprocessor.data.head(100))
    pipeline = LLMPipeline(settings.parameters.model, dataset, max_length=120, batch_size=1, device='cpu')
    model_properties = pipeline.analyze_model()
    sample_inference = pipeline.infer_sample(dataset[0])
    result = model_properties, sample_inference
    print(result)
    assert result is not None, "Finetuning does not work correctly"


if __name__ == "__main__":
    main()
