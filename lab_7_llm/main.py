"""
Laboratory work.

Working with Large Language Models.
"""
# pylint: disable=too-few-public-methods, undefined-variable, too-many-arguments, super-init-not-called
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
import torch
from torch.utils.data import Dataset
from torchinfo import summary
from transformers import BertForSequenceClassification, BertTokenizer
from datasets import load_dataset

from core_utils.llm.raw_data_importer import AbstractRawDataImporter
from core_utils.llm.raw_data_preprocessor import AbstractRawDataPreprocessor, ColumnNames
from core_utils.llm.llm_pipeline import AbstractLLMPipeline
from core_utils.llm.task_evaluator import AbstractTaskEvaluator
from core_utils.llm.metrics import Metrics
from core_utils.llm.time_decorator import report_time
from pandas.core.frame import DataFrame


class RawDataImporter(AbstractRawDataImporter):
    """
    A class that imports the HuggingFace dataset.
    """

    @report_time
    def obtain(self) -> None:
        """
        Download a dataset.

        Raises:
            TypeError: In case of downloaded dataset is not pd.DataFrame
        """
        dataset = load_dataset(self._hf_name, split="train").to_pandas()

        if not isinstance(dataset, pd.DataFrame):
            raise TypeError("Downloaded dataset is not a pandas DataFrame.")

        self._raw_data = dataset


class RawDataPreprocessor(AbstractRawDataPreprocessor):
    """
    A class that analyzes and preprocesses a dataset.
    """

    def analyze(self) -> dict:
        """
        Analyze a dataset.

        Returns:
            dict: Dataset key properties
        """
        num_samples = self._raw_data.shape[0]
        num_columns = self._raw_data.shape[1]

        num_duplicates = self._raw_data.duplicated().sum()
        num_empty_rows = self._raw_data.isnull().all(axis=1).sum()

        cleaned_data = self._raw_data.dropna()

        min_sample_length, max_sample_length = cleaned_data["text"].str.len().agg(["min", "max"])

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
        self._data.rename(columns={'text': ColumnNames.SOURCE.value,
                                   'toxic': ColumnNames.TARGET.value},
                                    inplace=True)


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
        row = self._data.iloc[index]
        return row[ColumnNames.SOURCE.value], str(row[ColumnNames.TARGET.value])

    @property
    def data(self) -> DataFrame:
        """
        Property with access to preprocessed DataFrame.

        Returns:
            pandas.DataFrame: Preprocessed DataFrame
        """
        return self._data


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
            model_name (str): The name of the pre-trained model
            dataset (TaskDataset): The dataset used
            max_length (int): The maximum length of generated sequence
            batch_size (int): The size of the batch inside DataLoader
            device (str): The device for inference
        """
        self._model_name = model_name
        self._dataset = dataset
        self._device = device
        self._max_length = max_length
        self._batch_size = batch_size

        self._model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
        self._model.to(self._device)

        self._tokenizer = BertTokenizer.from_pretrained(model_name)

    def analyze_model(self) -> dict:
        """
        Analyze model computing properties.

        Returns:
            dict: Properties of a model
        """
        sample_input = torch.ones((1, self._model.config.max_position_embeddings),
                                  dtype=torch.long,
                                  device=self._device)
        input_tensors = {'input_ids': sample_input,
                         'attention_mask': sample_input}

        model_summary = summary(self._model, input_data=input_tensors, verbose=0)
        model_metadata = {
            'input_dimensions': {key: list(value.shape) for key, value in input_tensors.items()},
            'embedding_size': self._model.config.max_position_embeddings,
            'output_shape': model_summary.summary_list[-1].output_size,
            'num_trainable_params': model_summary.trainable_params,
            'vocab_size': self._model.config.vocab_size,
            'size': model_summary.total_param_bytes,
            'max_context_length': self._model.config.max_length
        }
        return model_metadata


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
            return None

        inputs = self._tokenizer(
            sample[0],
            truncation=True,
            padding="max_length",
            max_length=self._max_length,
            return_tensors="pt"
        )

        inputs = {key: value.to(self._device) for key, value in inputs.items()}

        self._model.eval()
        with torch.no_grad():
            outputs = self._model(**inputs)

        predicted_class = torch.argmax(outputs.logits, dim=1).item()
        return str(predicted_class)

    @report_time
    def infer_dataset(self) -> pd.DataFrame:
        """
        Infer model on a whole dataset.

        Returns:
            pd.DataFrame: Data with predictions
        """

    @torch.no_grad()
    def _infer_batch(self, sample_batch: Sequence[tuple[str, ...]]) -> list[str]:
        """
        Infer model on a single batch.

        Args:
            sample_batch (Sequence[tuple[str, ...]]): Batch to infer the model

        Returns:
            list[str]: Model predictions as strings
        """


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

    @report_time
    def run(self) -> dict | None:
        """
        Evaluate the predictions against the references using the specified metric.

        Returns:
            dict | None: A dictionary containing information about the calculated metric
        """
