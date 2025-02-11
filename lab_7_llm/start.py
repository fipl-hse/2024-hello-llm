"""
Starter for demonstration of laboratory work.
"""
# pylint: disable= too-many-locals, undefined-variable, unused-import
import json
from pathlib import Path

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
    result = None

    with open(Path(__file__).parent / "settings.json", encoding="utf-8") as f:
        settings_dict = json.load(f)


    importer = RawDataImporter(settings_dict["parameters"]["dataset"])
    importer.obtain()

    preprocessor = RawDataPreprocessor(importer.raw_data)
    analysis = preprocessor.analyze()
    preprocessor.transform()

    dataset = TaskDataset(preprocessor.data.head(100))
    pipeline = LLMPipeline(settings_dict['parameters']['model'], dataset, max_length=120, batch_size=1, device="cpu")

    model_properties = pipeline.analyze_model()
    prediction = pipeline.infer_sample(dataset.__getitem__(0))
    result = prediction


    assert result is not None, "Demo does not work correctly"



if __name__ == "__main__":
    main()
