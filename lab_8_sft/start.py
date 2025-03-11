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
    settings = LabSettings(PROJECT_ROOT / 'lab_8_sft' / 'settings.json')

    importer = RawDataImporter(settings.parameters.dataset)
    importer.obtain()

    if importer.raw_data is None:
        raise ValueError('Received None instead of dataframe')

    preprocessor = RawDataPreprocessor(importer.raw_data)
    preprocessor.transform()
    dataset = TaskDataset(preprocessor.data.head(10))
    pipeline = LLMPipeline(settings.parameters.model, dataset, 120, 64, 'cpu')
    infer_dataframe = pipeline.infer_dataset()
    print(pipeline.infer_sample(dataset[0]))
    path_to_outputs = PROJECT_ROOT / 'lab_8_sft' / 'dist' / 'predictions.csv'
    path_to_outputs.parent.mkdir(exist_ok=True)
    infer_dataframe.to_csv(path_to_outputs, index=False)
    evaluation = TaskEvaluator(path_to_outputs, settings.parameters.metrics)
    res = evaluation.run()
    print(res)

    result = res
    assert result is not None, "Finetuning does not work correctly"


if __name__ == "__main__":
    main()
