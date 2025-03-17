"""
Fine-tuning starter.
"""
# pylint: disable=too-many-locals, undefined-variable, unused-import, too-many-branches, too-many-statements
from pathlib import Path

from config.constants import PROJECT_ROOT
from config.lab_settings import LabSettings
from core_utils.llm.time_decorator import report_time
from lab_8_sft.main import LLMPipeline, RawDataImporter, RawDataPreprocessor, SFTPipeline, TaskDataset, TaskEvaluator

@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    settings = LabSettings(PROJECT_ROOT / "lab_8_sft" / "settings.json")

    importer = RawDataImporter(settings.parameters.dataset)
    importer.obtain()
    if importer.raw_data is None:
        return

    preprocessor = RawDataPreprocessor(importer.raw_data)
    analysis = preprocessor.analyze()
    print(analysis)

    preprocessor.transform()
    dataset = TaskDataset(preprocessor.data.head(100))
    pipeline = LLMPipeline(settings.parameters.model,
                           dataset,
                           max_length=120,
                           batch_size=1,
                           device='cpu')
    print(pipeline.analyze_model())
    result = pipeline.infer_sample(dataset[0])
    print(result)

    assert result is not None, "Finetuning does not work correctly"


if __name__ == "__main__":
    main()
