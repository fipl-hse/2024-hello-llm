"""
Starter for demonstration of laboratory work.
"""
# pylint: disable= too-many-locals, undefined-variable, unused-import
import sys
from pathlib import Path

import pandas as pd

from lab_7_llm.main import (
    RawDataImporter,
    RawDataPreprocessor,
    TaskDataset,
    LLMPipeline,
    TaskEvaluator,
    report_time
)


@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    try:
        importer = RawDataImporter("trixdade/reviews_russian")
        raw_data = importer.obtain()

        if raw_data is None or raw_data.empty:
            print("Warning: No data obtained. Exiting early.")
            return

        preprocessor = RawDataPreprocessor(raw_data)
        preprocessor.transform()

        dataset = TaskDataset(preprocessor.data)

        model = LLMPipeline(
            model_name="stevhliu/my_awesome_billsum_model",
            dataset=dataset,
            max_length=512,
            batch_size=8,
            device="cpu"
        )

        model_properties = model.analyze_model()
        print("Model Properties:", model_properties)

        predictions = model.infer_dataset()

        predictions_path = Path("predictions.csv")
        predictions.to_csv(predictions_path, index=False)

        evaluator = TaskEvaluator(predictions_path, ["rouge"])
        results = evaluator.run()
        print("Evaluation Results:", results)

    except (ValueError, FileNotFoundError, KeyError) as e:
        print(f"An error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
