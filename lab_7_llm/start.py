"""
Starter for demonstration of laboratory work.
"""
# pylint: disable= too-many-locals, undefined-variable, unused-import
import sys
from pathlib import Path

import pandas as pd

from core_utils.llm.metrics import Metrics
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
    try:
        importer = RawDataImporter()
        importer.obtain()

        raw_data = importer.data

        if raw_data is None or raw_data.empty:
            print("Warning: No data obtained. Exiting early.")
            return

        preprocessor = RawDataPreprocessor(raw_data)

        preprocessor.transform()

        dataset_analysis = preprocessor.analyze()
        print("Dataset analysis:")
        for field, value in dataset_analysis.items():
            print(field, value, sep=': ')

        dataset = TaskDataset(preprocessor.data.head(100))

        model = LLMPipeline(
            model_name="stevhliu/my_awesome_billsum_model",
            dataset=dataset,
            max_length=512,
            batch_size=8,
            device="cpu"
        )

        model_analysis = model.analyze_model()
        print("Model analysis:")
        for field, value in model_analysis.items():
            print(field, value, sep=': ')

        predictions = model.infer_dataset()

        predictions_path = Path("predictions.csv")
        predictions.to_csv(predictions_path, index=False)

        metrics = [Metrics("rouge")]
        evaluator = TaskEvaluator(predictions_path, metrics)
        results = evaluator.run()
        print("Evaluation Results:", results)

        assert results is not None, "Demo does not work correctly"

    except (ValueError, FileNotFoundError, KeyError) as e:
        print(f"An error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
