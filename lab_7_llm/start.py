"""
Starter for demonstration of laboratory work.
"""
# pylint: disable= too-many-locals, undefined-variable, unused-import
import json

from config.constants import PROJECT_ROOT
from lab_7_llm.main import (
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
    with open(PROJECT_ROOT / "lab_7_llm" / "settings.json", "r", encoding="utf-8") as file:
        settings = json.load(file)

    importer = RawDataImporter(settings["parameters"]["dataset"])
    importer.obtain()

    if importer.raw_data is None:
        raise ValueError("Raw data is None")

    preprocessor = RawDataPreprocessor(importer.raw_data)

    analysis = preprocessor.analyze()
    print(analysis)

    preprocessor.transform()
    dataset = TaskDataset(preprocessor.data.head(100))
    print(len(dataset))
    print(dataset[1])

    pipeline = LLMPipeline(
        settings["parameters"]["model"], dataset, max_length=120, batch_size=64, device="cpu"
    )
    print(pipeline.analyze_model())

    infer_sample = pipeline.infer_sample(dataset[1])
    print(infer_sample)

    infer_df = pipeline.infer_dataset()

    path_to_outputs = PROJECT_ROOT / "lab_7_llm" / "dist" / "predictions.csv"
    path_to_outputs.parent.mkdir(exist_ok=True)
    infer_df.to_csv(path_to_outputs, index=False)

    evaluation = TaskEvaluator(path_to_outputs, settings["parameters"]["metrics"])
    eval_res = evaluation.run()
    print(eval_res)
    result = eval_res
    assert result is not None, "Demo does not work correctly"


if __name__ == "__main__":
    main()
