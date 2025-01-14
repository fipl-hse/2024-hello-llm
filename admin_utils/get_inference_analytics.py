"""
Collect and store inference analytics.
"""
# pylint: disable=import-error, duplicate-code, too-many-branches, no-else-return, inconsistent-return-statements, too-many-locals, too-many-statements, wrong-import-order
from pathlib import Path

from pandas import DataFrame
from pydantic.dataclasses import dataclass

from admin_utils.get_model_analytics import get_references, save_reference

from lab_7_llm.main import LLMPipeline, TaskDataset  # isort:skip
from reference_lab_classification.main import ClassificationLLMPipeline  # isort:skip
from reference_lab_generation.main import GenerationLLMPipeline  # isort:skip
from reference_lab_nli.main import NLILLMPipeline  # isort:skip
from reference_lab_open_qa.main import OpenQALLMPipeline  # isort:skip


@dataclass
class InferenceParams:
    """
    Inference parameters.
    """

    num_samples: int
    max_length: int
    batch_size: int
    predictions_path: Path
    device: str


def get_inference_from_task(
    model_name: str, inference_params: InferenceParams, samples: list, task: str
) -> dict:
    """
    Gets inferences.

    Args:
        model_name (str): Model name
        inference_params (InferenceParams): Parameters from inference
        samples (list): Samples for inference
        task (str): Task for inference

    Returns:
        dict: Processed predictions with queries
    """
    dataset = TaskDataset(DataFrame([]))

    pipeline: LLMPipeline
    if task == "nmt":
        pipeline = LLMPipeline(
            model_name,
            dataset,
            inference_params.max_length,
            inference_params.batch_size,
            inference_params.device,
        )
    elif task == "generation":
        pipeline = GenerationLLMPipeline(
            model_name,
            dataset,
            inference_params.max_length,
            inference_params.batch_size,
            inference_params.device,
        )
    elif task == "classification":
        pipeline = ClassificationLLMPipeline(
            model_name,
            dataset,
            inference_params.max_length,
            inference_params.batch_size,
            inference_params.device,
        )
    elif task == "nli":
        pipeline = NLILLMPipeline(
            model_name,
            dataset,
            inference_params.max_length,
            inference_params.batch_size,
            inference_params.device,
        )
    elif task == "summarization":
        pipeline = LLMPipeline(
            model_name,
            dataset,
            inference_params.max_length,
            inference_params.batch_size,
            inference_params.device,
        )
    else:
        pipeline = OpenQALLMPipeline(
            model_name,
            dataset,
            inference_params.max_length,
            inference_params.batch_size,
            inference_params.device,
        )

    result = {}
    for sample in samples:
        if "[TEST SEP]" in sample:
            first_value, second_value = sample.split("[TEST SEP]")
            prediction = pipeline.infer_sample((first_value, second_value))
        else:
            prediction = pipeline.infer_sample((sample,))

        result[sample] = prediction

    return result


def get_task(model: str, inference_params: InferenceParams, samples: list) -> dict:
    """
    Gets task.

    Args:
        model (str): name of model
        inference_params (InferenceParams): Samples for inference
        samples (list): Parameters from inference

    Returns:
        dict: Results with model predictions
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

    _ = [
        "EleutherAI/pythia-160m-deduped",
        "JackFram/llama-68m",
        "EleutherAI/gpt-neo-125m",
    ]

    if model in nmt_model:
        return get_inference_from_task(model, inference_params, samples, "nmt")
    elif model in generation_model:
        return get_inference_from_task(model, inference_params, samples, "generation")
    elif model in classification_model:
        return get_inference_from_task(model, inference_params, samples, "classification")
    elif model in nli_model:
        return get_inference_from_task(model, inference_params, samples, "nli")
    elif model in summarization_model:
        return get_inference_from_task(model, inference_params, samples, "summarization")
    else:
        return get_inference_from_task(model, inference_params, samples, "open_qa")


def main() -> None:
    """
    Run collected reference scores.
    """
    references_path = Path(__file__).parent / "reference_inference_analytics.json"
    dest = Path(__file__).parent / "reference_inference_analytics_.json"

    max_length = 120
    batch_size = 1
    num_samples = 100
    device = "cpu"

    inference_params = InferenceParams(
        num_samples=num_samples,
        max_length=max_length,
        batch_size=batch_size,
        device=device,
        predictions_path=Path(),
    )

    references = get_references(path=references_path)
    result = {}

    for model, pairs in references.items():
        predictions = get_task(model, inference_params, pairs)
        print(model)
        result[model] = predictions

    save_reference(dest, result)


if __name__ == "__main__":
    main()
