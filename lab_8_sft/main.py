"""
Laboratory work.

Fine-tuning Large Language Models for a downstream task.
"""
# pylint: disable=too-few-public-methods, undefined-variable, duplicate-code, unused-argument, too-many-arguments
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
import torch as torch

from datasets import load_dataset
from pandas import DataFrame
from torch.utils.data import Dataset
from torchinfo import summary
from transformers import BertForSequenceClassification, BertTokenizerFast, AutoTokenizer

from config.lab_settings import SFTParams
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
        self._raw_data = load_dataset(self._hf_name, split='train').to_pandas()


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
        properties = dict()
        properties['dataset_number_of_samples'] = len(self._raw_data)
        properties['dataset_columns'] = len(self._raw_data.columns)
        properties['dataset_duplicates'] = int(self._raw_data.duplicated().sum())
        properties['dataset_empty_rows'] = len(self._raw_data) - len(self._raw_data.dropna())
        properties['dataset_sample_min_len'] = int(self._raw_data['content'].str.len().min())
        properties['dataset_sample_max_len'] = int(self._raw_data['content'].str.len().max())
        return properties

    @report_time
    def transform(self) -> None:
        """
        Apply preprocessing transformations to the raw dataset.
        """
        self._data = self._raw_data.drop\
            (columns=['part', 'movie_name', 'review_id', 'author', 'date', 'title', 'grade10'])
        self._data = self._data.rename(columns={'content': ColumnNames.SOURCE, 'grade3': ColumnNames.TARGET})
        self._data = self._data.dropna()
        self._data[ColumnNames.TARGET] = self._data[ColumnNames.TARGET].map({'Bad': 2, 'Neutral': 1, 'Good': 0})
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
        return tuple(self._data[ColumnNames.SOURCE][index])

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


class LLMPipeline(AbstractSFTPipeline):
    """
    A class that initializes a model, analyzes its properties and infers it.
    """

    def run(self) -> None:
        pass

    _model: torch.nn.Module

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
        super().__init__(model_name, dataset)
        self._model = BertForSequenceClassification.from_pretrained(self._model_name)
        self._tokenizer = BertTokenizerFast.from_pretrained(self._model_name)
        self._max_length = max_length
        self._batch_size = batch_size
        self._device = device

    def analyze_model(self) -> dict:
        """
        Analyze model computing properties.

        Returns:
            dict: Properties of a model
        """
        if not isinstance(self._model, torch.nn.Module):
            raise TypeError

        properties = dict()
        input_data = torch.ones((1, self._model.config.max_position_embeddings), dtype=torch.long)
        model_statistics = summary(self._model, input_data={'input_ids': input_data, 'attention_mask': input_data})
        properties['input_shape'] = {'input_ids': model_statistics.input_size['input_ids'],
                                     'attention_mask': model_statistics.input_size['input_ids']}
        properties['embedding_size'] = self._model.config.max_position_embeddings
        properties['output_shape'] = model_statistics.summary_list[-1].output_size
        properties['num_trainable_params'] = model_statistics.trainable_params
        properties['vocab_size'] = self._model.config.vocab_size
        properties['size'] = model_statistics.total_param_bytes
        properties['max_context_length'] = self._model.config.max_length
        return properties

    @report_time
    def infer_sample(self, sample: tuple[str, ...]) -> str | None:
        """
        Infer model on a single sample.

        Args:
            sample (tuple[str, ...]): The given sample for inference with model

        Returns:
            str | None: A prediction
        """
        if not self._model:
            raise ValueError

        return self._infer_batch([sample])[0]

    @report_time
    def infer_dataset(self) -> pd.DataFrame:
        """
        Infer model on a whole dataset.

        Returns:
            pd.DataFrame: Data with predictions
        """
        pass

    @torch.no_grad()
    def _infer_batch(self, sample_batch: Sequence[tuple[str, ...]]) -> list[str]:
        """
        Infer single batch.

        Args:
            sample_batch (Sequence[tuple[str, ...]]): batch to infer the model

        Returns:
            list[str]: model predictions as strings
        """
        tokenized = self._tokenizer(list(sample_batch[0]), max_length=self._max_length,
                                    padding=True, truncation=True, return_tensors='pt')
        predict = torch.argmax(self._model(**tokenized).logits, dim=1).tolist()
        return [str(i) if i != 0 else '2' for i in predict]


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

    def run(self) -> dict | None:
        """
        Evaluate the predictions against the references using the specified metric.

        Returns:
            dict | None: A dictionary containing information about the calculated metric
        """


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
