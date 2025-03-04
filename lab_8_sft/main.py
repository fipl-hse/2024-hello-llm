"""
Laboratory work.

Fine-tuning Large Language Models for a downstream task.
"""
# pylint: disable=too-few-public-methods, undefined-variable, duplicate-code, unused-argument, too-many-arguments
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
import torch
from datasets import load_dataset
from evaluate import load
from pandas import DataFrame
from torch.utils.data import DataLoader, Dataset
from torchinfo import summary
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from config.lab_settings import SFTParams
from core_utils.llm.llm_pipeline import AbstractLLMPipeline
from core_utils.llm.metrics import Metrics
from core_utils.llm.raw_data_importer import AbstractRawDataImporter
from core_utils.llm.raw_data_preprocessor import AbstractRawDataPreprocessor, ColumnNames
from core_utils.llm.sft_pipeline import AbstractSFTPipeline
from core_utils.llm.task_evaluator import AbstractTaskEvaluator
from core_utils.llm.time_decorator import report_time


class RawDataImporter(AbstractRawDataImporter):
    """
    Custom implementation of data importer.
    """

    @report_time
    def obtain(self) -> None:
        """
        Import dataset.
        """
        dataset = load_dataset(self._hf_name, split="test").to_pandas()

        if not isinstance(dataset, pd.DataFrame):
            raise TypeError("Downloaded dataset is not a pandas DataFrame.")

        self._raw_data = dataset


class RawDataPreprocessor(AbstractRawDataPreprocessor):
    """
    Custom implementation of data preprocessor.
    """

    def analyze(self) -> dict:
        """
        Analyze preprocessed dataset.

        Returns:
            dict: dataset key properties.
        """
        num_samples = self._raw_data.shape[0]
        num_columns = self._raw_data.shape[1]

        num_duplicates = self._raw_data.duplicated().sum()
        num_empty_rows = self._raw_data.isnull().all(axis=1).sum()

        cleaned_data = self._raw_data.dropna()

        min_sample_length, max_sample_length = cleaned_data["article"].str.len().agg(["min", "max"])

        return {
            "dataset_number_of_samples": num_samples,
            "dataset_columns": num_columns,
            "dataset_duplicates": num_duplicates,
            "dataset_empty_rows": num_empty_rows,
            "dataset_sample_min_len": min_sample_length,
            "dataset_sample_max_len": max_sample_length,
        }

    @report_time
    def transform(self) -> None:
        """
        Apply preprocessing transformations to the raw dataset.
        """
        self._data = self._raw_data.copy()
        self._data.rename(columns={'article': ColumnNames.SOURCE.value,
                                   'abstract': ColumnNames.TARGET.value},
                                    inplace=True)
        self._data.reset_index(drop=True, inplace=True)


class TaskDataset(Dataset):
    """
    A class that converts pd.DataFrame to Dataset and works with it.
    """

    def __init__(self, data: pd.DataFrame) -> None:
        """
        Initialize an instance of TaskDataset.

        Args:
            data (pandas.DataFrame): Original data
        """
        self._data = data

    def __len__(self) -> int:
        """
        Return the number of items in the dataset.

        Returns:
            int: The number of items in the dataset
        """
        return len(self._data)

    def __getitem__(self, index: int) -> tuple[str, ...]:
        """
        Retrieve an item from the dataset by index.

        Args:
            index (int): Index of sample in dataset

        Returns:
            tuple[str, ...]: The item to be received
        """
        item = str(self._data.loc[index, ColumnNames.SOURCE.value])

        return tuple([item])

    @property
    def data(self) -> DataFrame:
        """
        Property with access to preprocessed DataFrame.

        Returns:
            pandas.DataFrame: Preprocessed DataFrame
        """
        return self._data


def tokenize_sample(
    sample: pd.Series, tokenizer: AutoTokenizer, max_length: int
) -> dict[str, torch.Tensor]:
    """
    Tokenize sample.

    Args:
        sample (pandas.Series): sample from a dataset
        tokenizer (transformers.models.auto.tokenization_auto.AutoTokenizer): Tokenizer to tokenize
            original data
        max_length (int): max length of sequence

    Returns:
        dict[str, torch.Tensor]: Tokenized sample
    """


class TokenizedTaskDataset(Dataset):
    """
    A class that converts pd.DataFrame to Dataset and works with it.
    """

    def __init__(self, data: pd.DataFrame, tokenizer: AutoTokenizer, max_length: int) -> None:
        """
        Initialize an instance of TaskDataset.

        Args:
            data (pandas.DataFrame): Original data
            tokenizer (transformers.models.auto.tokenization_auto.AutoTokenizer): Tokenizer to
                tokenize the dataset
            max_length (int): max length of a sequence
        """

    def __len__(self) -> int:
        """
        Return the number of items in the dataset.

        Returns:
            int: The number of items in the dataset
        """

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """
        Retrieve an item from the dataset by index.

        Args:
            index (int): Index of sample in dataset

        Returns:
            dict[str, torch.Tensor]: An element from the dataset
        """


class LLMPipeline(AbstractLLMPipeline):
    """
    A class that initializes a model, analyzes its properties and infers it.
    """

    def __init__(
        self, model_name: str, dataset: TaskDataset, max_length: int, batch_size: int, device: str
    ) -> None:
        """
        Initialize an instance of LLMPipeline.

        Args:
            model_name (str): The name of the pre-trained model.
            dataset (TaskDataset): The dataset to be used for translation.
            max_length (int): The maximum length of generated sequence.
            batch_size (int): The size of the batch inside DataLoader.
            device (str): The device for inference.
        """
        super().__init__(model_name, dataset, max_length, batch_size, device)

        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        self._model = self._model = AutoModelForSeq2SeqLM.from_pretrained(self._model_name)
        self._model.to(self._device)

    def analyze_model(self) -> dict:
        """
        Analyze model computing properties.

        Returns:
            dict: Properties of a model
        """
        sample_input = torch.ones((1, self._model.config.encoder.max_position_embeddings),
                                  dtype=torch.long,
                                  device=self._device)

        input_data = {"input_ids": sample_input,
                      "decoder_input_ids": sample_input}

        if not isinstance(self._model, torch.nn.Module):
            raise ValueError("The model has not been correctly initialized")

        model_summary = summary(self._model, input_data=input_data, verbose=0)

        try:
            embedding_size = self._model.config.max_position_embeddings
        except AttributeError:
            if hasattr(self._model.config, "encoder"):
                embedding_size = getattr(self._model.config.encoder, "max_position_embeddings", None)
                if embedding_size is None:
                    embedding_size = getattr(self._model.config.encoder, "hidden_size", None)
            else:
                embedding_size = None

        return {
            'input_shape': {key: list(value.shape) for key, value in input_data.items()},
            'embedding_size': embedding_size,
            'output_shape': model_summary.summary_list[-1].output_size,
            'num_trainable_params': model_summary.trainable_params,
            'vocab_size': self._model.config.vocab_size,
            'size': model_summary.total_param_bytes,
            'max_context_length': self._model.config.max_length
        }

    @report_time
    def infer_sample(self, sample: tuple[str, ...]) -> str | None:
        """
        Infer model on a single sample.

        Args:
            sample (tuple[str, ...]): The given sample for inference with model

        Returns:
            str | None: A prediction
        """
        return self._infer_batch([sample])[0]

    @report_time
    def infer_dataset(self) -> pd.DataFrame:
        """
        Infer model on a whole dataset.

        Returns:
            pd.DataFrame: Data with predictions
        """
        data_loader = DataLoader(dataset=self._dataset, batch_size=self._batch_size)
        predictions = []

        with torch.no_grad():
            for batch in data_loader:
                batch_predictions = self._infer_batch(batch)
                predictions.extend(batch_predictions)

        result_df = pd.DataFrame(self._dataset.data)
        result_df[ColumnNames.PREDICTION.value] = predictions

        return result_df

    @torch.no_grad()
    def _infer_batch(self, sample_batch: Sequence[tuple[str, ...]]) -> list[str]:
        """
        Infer single batch.

        Args:
            sample_batch (Sequence[tuple[str, ...]]): batch to infer the model

        Returns:
            list[str]: model predictions as strings
        """
        inputs = self._tokenizer(
            [sample[0] for sample in sample_batch],
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self._max_length
        ).to(self._device)

        outputs = self._model.generate(
            **inputs,
            max_length=self._max_length
        )

        return self._tokenizer.batch_decode(outputs, skip_special_tokens=True)


class TaskEvaluator(AbstractTaskEvaluator):
    """
    A class that compares prediction quality using the specified metric.
    """

    def __init__(self, data_path: Path, metrics: Iterable[Metrics]) -> None:
        """
        Initialize an instance of Evaluator.

        Args:
            data_path (pathlib.Path): Path to predictions
            metrics (Iterable[Metrics]): List of metrics to check
        """
        super().__init__(metrics)
        self._data_path = data_path


    def run(self) -> dict | None:
        """
        Evaluate the predictions against the references using the specified metric.

        Returns:
            dict | None: A dictionary containing information about the calculated metric
        """
        data_to_evaluate = pd.read_csv(self._data_path)

        predictions = data_to_evaluate[ColumnNames.PREDICTION.value]
        targets = data_to_evaluate[ColumnNames.TARGET.value]

        evaluation = {}
        for metric in self._metrics:
            scores = load(metric.value, seed=77).compute(predictions=predictions,
                                                         references=targets)
            if metric.value == "rouge":
                evaluation[metric.value] = scores["rougeL"]
            else:
                evaluation[metric.value] = scores[metric.value]

        return evaluation


class SFTPipeline(AbstractSFTPipeline):
    """
    A class that initializes a model, fine-tuning.
    """

    def __init__(self, model_name: str, dataset: Dataset, sft_params: SFTParams) -> None:
        """
        Initialize an instance of ClassificationSFTPipeline.

        Args:
            model_name (str): The name of the pre-trained model.
            dataset (torch.utils.data.dataset.Dataset): The dataset used.
            sft_params (SFTParams): Fine-Tuning parameters.
        """

    def run(self) -> None:
        """
        Fine-tune model.
        """
