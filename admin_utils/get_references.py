"""
Collect and store model analytics.
"""
# pylint: disable=import-error, too-many-branches, no-else-return, inconsistent-return-statements, too-many-locals, too-many-statements, wrong-import-order, too-many-return-statements
from decimal import Decimal, ROUND_FLOOR
from pathlib import Path
from typing import Any

from pydantic.dataclasses import dataclass
from tqdm import tqdm

from admin_utils.get_model_analytics import get_references, save_reference
from config.lab_settings import InferenceParams
from core_utils.llm.metrics import Metrics

from reference_lab_classification.start import get_result_for_classification  # isort:skip
from reference_lab_generation.start import get_result_for_generation  # isort:skip
from reference_lab_ner.start import get_result_for_ner  # isort:skip
from reference_lab_nli.start import get_result_for_nli  # isort:skip
from reference_lab_nmt.start import get_result_for_nmt  # isort:skip
from reference_lab_open_qa.start import get_result_for_open_qa  # isort:skip
from reference_lab_summarization.start import get_result_for_summarization  # isort:skip


@dataclass
class MainParams:
    """
    Main parameters.
    """

    model: str
    dataset: str
    metrics: list[Metrics]


def get_task(model: str, main_params: MainParams, inference_params: InferenceParams) -> Any:
    """
    Gets task.

    Args:
        model (str): name of model
        main_params (MainParams): Parameters from main
        inference_params (InferenceParams): Parameters from inference

    Returns:
        Any: Metric for a specific task
    """
    if "test_" in model:
        model = model.replace("test_", "")

    nmt_model = [
        "Helsinki-NLP/opus-mt-en-fr",
        "Helsinki-NLP/opus-mt-ru-en",
        "Helsinki-NLP/opus-mt-ru-es",
        "t5-small",
    ]

    generation_model = ["VMware/electra-small-mrqa", "timpal0l/mdeberta-v3-base-squad2"]

    classification_model = [
        "cointegrated/rubert-tiny-toxicity",
        "cointegrated/rubert-tiny2-cedr-emotion-detection",
        "papluca/xlm-roberta-base-language-detection",
        "fabriceyhc/bert-base-uncased-ag_news",
        "XSY/albert-base-v2-imdb-calssification",
        "aiknowyou/it-emotion-analyzer",
        "blanchefort/rubert-base-cased-sentiment-rusentiment",
        "tatiana-merz/turkic-cyrillic-classifier",
        "s-nlp/russian_toxicity_classifier",
        "IlyaGusev/rubertconv_toxic_clf",
    ]

    nli_model = [
        "cointegrated/rubert-base-cased-nli-threeway",
        "cointegrated/rubert-tiny-bilingual-nli",
        "cross-encoder/qnli-distilroberta-base",
        "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
    ]

    summarization_model = [
        "mrm8488/bert-mini2bert-mini-finetuned-cnn_daily_mail-summarization",
        "nandakishormpai/t5-small-machine-articles-tag-generation",
        "mrm8488/bert-small2bert-small-finetuned-cnn_daily_mail-summarization",
        "stevhliu/my_awesome_billsum_model",
        "UrukHan/t5-russian-summarization",
        "dmitry-vorobiev/rubert_ria_headlines",
    ]

    open_generative_qa_model = [
        "EleutherAI/pythia-160m-deduped",
        "JackFram/llama-68m",
        "EleutherAI/gpt-neo-125m",
    ]

    ner_model = ["dslim/distilbert-NER", "Babelscape/wikineural-multilingual-ner"]

    if model in nmt_model:
        return get_result_for_nmt(inference_params, main_params)
    elif model in generation_model:
        return get_result_for_generation(inference_params, main_params)
    elif model in classification_model:
        return get_result_for_classification(inference_params, main_params)
    elif model in nli_model:
        return get_result_for_nli(inference_params, main_params)
    elif model in summarization_model:
        return get_result_for_summarization(inference_params, main_params)
    elif model in open_generative_qa_model:
        return get_result_for_open_qa(inference_params, main_params)
    elif model in ner_model:
        return get_result_for_ner(inference_params, main_params)
    else:
        raise ValueError(f"Unknown model {model} ...")


def main() -> None:
    """
    Run collected reference scores.
    """
    project_root = Path(__file__).parent.parent
    references_path = project_root / "admin_utils" / "reference_scores.json"

    dist_dir = project_root / "dist"
    dist_dir.mkdir(exist_ok=True)

    dest = dist_dir / "reference_scores.json"

    max_length = 120
    batch_size = 3
    num_samples = 100
    device = "cuda"

    inference_params = InferenceParams(
        num_samples, max_length, batch_size, dist_dir / "result.csv", device
    )

    references = get_references(path=references_path)

    combos = []
    for model_name, datasets in sorted(references.items()):
        for dataset_name, metrics in sorted(datasets.items()):
            for metric in sorted(metrics):
                combos.append((model_name, dataset_name, metric))

    result = {}
    for model_name, dataset_name, metric in tqdm(sorted(combos)):
        print(model_name, dataset_name, metric)

        if model_name not in result:
            result[model_name] = {}
        if dataset_name not in result[model_name]:
            result[model_name][dataset_name] = {}
        if metric not in result[model_name][dataset_name]:
            result[model_name][dataset_name][metric] = {}
        if "test_" in model_name:
            inference_params.num_samples = 10

        main_params = MainParams(model_name, dataset_name, [Metrics(metric)])
        inference_func = get_task(model_name, main_params, inference_params)
        score = Decimal(inference_func[metric])
        truncated_metric = score.quantize(Decimal("1.00000"), ROUND_FLOOR)
        result[model_name][dataset_name][metric] = truncated_metric

    save_reference(dest, result)


if __name__ == "__main__":
    main()
