"""
Collect and store model analytics.
"""
# pylint: disable=import-error, assignment-from-no-return, duplicate-code
from pathlib import Path
from typing import Any

import simplejson as json

try:
    from pandas import DataFrame
except ImportError:
    print('Library "pandas" not installed. Failed to import.')
    DataFrame = dict  # type: ignore

from lab_7_llm.main import LLMPipeline, TaskDataset  # type: ignore


def get_references(path: Path) -> Any:
    """
    Load reference_scores.json file.

    Args:
        path (Path): Path to file

    Returns:
        Any: File content
    """
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def save_reference(path: Path, refs: dict) -> None:
    """
    Save analytics.

    Args:
        path (Path): Path to file with analytics
        refs (dict): Model analysis for models
    """
    with open(path, mode="w", encoding="utf-8") as file:
        json.dump(refs, file, indent=4, ensure_ascii=False, sort_keys=True, use_decimal=True)
    with open(path, mode="a", encoding="utf-8") as file:
        file.write("\n")


def main() -> None:
    """
    Run collected models analytics.
    """
    batch_size = 64
    max_length = 120
    device = "cpu"

    references_path = Path(__file__).parent / "reference_scores.json"
    dest = Path(__file__).parent / "reference_model_analytics.json"

    references = get_references(path=references_path)
    result = {}

    for model, _ in references.items():
        print(model)
        pipeline = LLMPipeline(model, TaskDataset(DataFrame([])), max_length, batch_size, device)
        model_analysis = pipeline.analyze_model()
        result[model] = model_analysis

    save_reference(dest, result)


if __name__ == "__main__":
    main()
