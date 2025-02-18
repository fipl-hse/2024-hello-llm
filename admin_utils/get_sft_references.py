"""
Collect and store model analytics.
"""
# pylint: disable=duplicate-code
from decimal import Decimal, ROUND_FLOOR
from pathlib import Path
from typing import Any

from pydantic.dataclasses import dataclass
from tqdm import tqdm

from admin_utils.get_model_analytics import get_references, save_reference
from admin_utils.get_references import (
    collect_combinations,
    get_classification_models,
    get_nli_models,
    get_nmt_models,
    get_summurization_models,
    prepare_result_section,
)
from config.lab_settings import InferenceParams, SFTParams
from core_utils.llm.metrics import Metrics
from reference_lab_classification_sft.start import get_result_for_classification
from reference_lab_nli_sft.start import get_result_for_nli
from reference_lab_nmt_sft.start import get_result_for_nmt
from reference_lab_summarization_sft.start import get_result_for_summarization


@dataclass
class MainParams:
    """
    Main parameters.
    """

    model: str
    dataset: str
    metrics: list[Metrics]


def get_target_modules(model_name: str) -> list[str] | None:
    """
    Gets modules to fine-tune with LoRA.

    Args:
        model_name (str): Model name

    Returns:
        list[str] | None: modules to fine-tune with LoRA.
    """
    if model_name in (
        "dmitry-vorobiev/rubert_ria_headlines",
        "XSY/albert-base-v2-imdb-calssification",
    ):
        return ["query", "key", "value", "dense"]
    if model_name in (
        "Helsinki-NLP/opus-mt-en-fr",
        "Helsinki-NLP/opus-mt-ru-en",
        "Helsinki-NLP/opus-mt-ru-es",
    ):
        return ["k_proj", "v_proj", "q_proj", "out_proj"]
    # Peft will find default values for other
    return None


def get_task(
    model: str,
    main_params: MainParams,
    inference_params: InferenceParams,
    sft_params: SFTParams,
) -> Any:
    """
    Gets task.

    Args:
        model (str): name of model
        main_params (MainParams): Parameters from main
        inference_params (InferenceParams): Parameters for inference
        inference_params (SFTParams): Parameters for fine-tuning

    Returns:
        Any: Metric for a specific task
    """
    if "test_" in model:
        model = model.replace("test_", "")

    classification_models = get_classification_models()
    summarization_models = get_summurization_models()
    nli_models = get_nli_models()
    nmt_models = get_nmt_models()

    if model in classification_models:
        return get_result_for_classification(inference_params, sft_params, main_params)
    if model in summarization_models:
        sft_params.max_fine_tuning_steps = 30
        sft_params.batch_size = 1
        return get_result_for_summarization(inference_params, sft_params, main_params)
    if model in nli_models:
        return get_result_for_nli(inference_params, sft_params, main_params)
    if model in nmt_models:
        return get_result_for_nmt(inference_params, sft_params, main_params)

    raise ValueError(f"Unknown model {model} ...")


def main() -> None:
    """
    Run collected reference scores.
    """
    project_root = Path(__file__).parent.parent
    references_path = project_root / "admin_utils" / "reference_sft_scores.json"

    dist_dir = project_root / "dist"
    dist_dir.mkdir(exist_ok=True)

    dest = dist_dir / "predictions.json"

    references = get_references(path=references_path)

    combinations = collect_combinations(references)

    inference_params = InferenceParams(
        num_samples=10,
        max_length=120,
        batch_size=3,
        predictions_path=dist_dir / "predictions.csv",
        device="cpu",
    )
    sft_params = SFTParams(
        max_fine_tuning_steps=50,
        batch_size=3,
        max_length=120,
        finetuned_model_path=dist_dir,
        device="cpu",
    )
    result = {}
    for model_name, dataset_name, metric in tqdm(sorted(combinations)):
        print(model_name, dataset_name, metric)

        prepare_result_section(result, model_name, dataset_name, metric)
        sft_params.finetuned_model_path = dist_dir / model_name
        sft_params.target_modules = get_target_modules(model_name)
        main_params = MainParams(model_name, dataset_name, [Metrics(metric)])
        inference_func = get_task(model_name, main_params, inference_params, sft_params)
        score = Decimal(inference_func[metric])
        result[model_name][dataset_name][metric] = score.quantize(Decimal("1.00000"), ROUND_FLOOR)

    save_reference(dest, result)


if __name__ == "__main__":
    main()
