"""
Fine-tuning starter.
"""
# pylint: disable=too-many-locals, undefined-variable, unused-import, too-many-branches, too-many-statements
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

from config.constants import PROJECT_ROOT
from config.lab_settings import LabSettings
from core_utils.llm.time_decorator import report_time
from lab_8_sft.main import (
    RawDataImporter,
    RawDataPreprocessor,
    TaskDataset,
    LLMPipeline,
    TaskEvaluator
)


@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    settings = LabSettings(Path(__file__).parent / "settings.json")
    print(str(LabSettings(Path(__file__).parent / "settings.json")))

    importer = RawDataImporter(settings.parameters.dataset)
    importer.obtain()

    if not isinstance(importer.raw_data, pd.DataFrame):
        raise TypeError('Obtained dataset is not pd.DataFrame')

    preprocessor = RawDataPreprocessor(importer.raw_data)
    ds_analysis = preprocessor.analyze()
    print(ds_analysis)
    preprocessor.transform()

    dataset = TaskDataset(preprocessor.data.head(100))

    pipeline = LLMPipeline(settings.parameters.model,
                           dataset,
                           max_length=120,
                           batch_size=64,
                           device='cpu')
    model_analysis = pipeline.analyze_model()
    print(model_analysis)

    sample = pipeline.infer_sample(dataset[1])
    print('Sample res:', sample)

    infered_df = pipeline.infer_dataset()

    path_to_outputs = PROJECT_ROOT / 'lab_8_sft' / 'dist' / 'predictions.csv'
    path_to_outputs.parent.mkdir(exist_ok=True)
    infered_df.to_csv(path_to_outputs, index=False)

    evaluation = TaskEvaluator(path_to_outputs, settings.parameters.metrics)
    results = evaluation.run()
    print(results)

    # tokenizer = AutoTokenizer.from_pretrained(settings.parameters.model)
    # print(tokenize_sample(dataset[0], tokenizer, 120))

    result = results
    assert result is not None, "Finetuning does not work correctly"


if __name__ == "__main__":
    main()
