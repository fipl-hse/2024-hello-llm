"""
Starter for demonstration of laboratory work.
"""
# pylint: disable= too-many-locals, undefined-variable, unused-import
from pathlib import Path

from core_utils.llm.time_decorator import report_time
from lab_7_llm.main import RawDataImporter, RawDataPreprocessor, TaskDataset, LLMPipeline


@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    importer = RawDataImporter('jtatman/databricks-dolly-8k-qa-open-close')
    importer.obtain()

    preprocessor = RawDataPreprocessor(importer.raw_data)
    analysis = preprocessor.analyze()
    print(analysis)
    preprocessor.transform()

    dataset = TaskDataset(preprocessor.data.head(100))
    print(dataset[0])

    pipeline = LLMPipeline("JackFram/llama-68m", dataset, 120, 1, 'cpu')
    model_analysis = pipeline.analyze_model()
    print(model_analysis)

    sample = pipeline.infer_sample(dataset[0])
    print(sample)
    result = sample

    assert result is not None, "Demo does not work correctly"


if __name__ == "__main__":
    main()
