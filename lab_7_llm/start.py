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
)


@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    settings = LabSettings(PROJECT_ROOT / 'lab_7_llm/settings.json')

    importer = RawDataImporter(settings.parameters.dataset)
    importer.obtain()

    preprocessor = RawDataPreprocessor(importer.raw_data)
    preprocessor.transform()

    dataset = TaskDataset(preprocessor.data.head(100))

    batch_size = 1
    max_length = 120
    device = 'cpu'

    pipeline = LLMPipeline(model_name=settings.parameters.model,
                           dataset=dataset,
                           max_length=max_length,
                           batch_size=batch_size,
                           device=device)
    print(pipeline.analyze_model())
    result = pipeline.infer_sample('Я люблю есть кирпичи')
    assert result is not None, "Demo does not work correctly"


if __name__ == "__main__":
    main()
