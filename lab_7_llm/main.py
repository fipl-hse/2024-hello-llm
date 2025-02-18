"""
Laboratory work.

Working with Large Language Models.
"""
# pylint: disable=too-few-public-methods, undefined-variable, too-many-arguments, super-init-not-called
from pathlib import Path
from typing import Iterable, Sequence

import datasets
import pandas as pd
import torch
from pandas import DataFrame
from torch.utils.data import Dataset
from torchinfo import summary
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from core_utils.llm.llm_pipeline import AbstractLLMPipeline
from core_utils.llm.metrics import Metrics
from core_utils.llm.raw_data_importer import AbstractRawDataImporter
from core_utils.llm.raw_data_preprocessor import AbstractRawDataPreprocessor, ColumnNames
from core_utils.llm.task_evaluator import AbstractTaskEvaluator
from core_utils.llm.time_decorator import report_time


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
        dataset = datasets.load_dataset(self._hf_name, split="test")
        self._raw_data = pd.DataFrame(dataset)

        if not isinstance(self._raw_data, pd.DataFrame):
            raise TypeError("downloaded dataset is not pd.DataFrame.")


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
        dataset_info = {
            'dataset_number_of_samples': self._raw_data.shape[0],
            'dataset_columns': self._raw_data.shape[1],
            'dataset_duplicates': self._raw_data.duplicated().sum(),
            'dataset_empty_rows': self._raw_data.isnull().all(axis=1).sum(),
            'dataset_sample_min_len': self._raw_data['article'].dropna(how='all').map(len).min(),
            'dataset_sample_max_len': self._raw_data['article'].dropna(how='all').map(len).max()

        }

        return dataset_info

    @report_time
    def transform(self) -> None:
        """
        Apply preprocessing transformations to the raw dataset.
        """
        self._data = self._raw_data.rename(columns={"article": ColumnNames.SOURCE.value,
                                                    "abstract": ColumnNames.TARGET.value})
        self._data = self._data.dropna(subset=[ColumnNames.SOURCE.value, ColumnNames.TARGET.value])
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
        super().__init__(model_name, dataset, max_length, batch_size, device)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(model_name,
                                                        model_max_length=max_length)

    def analyze_model(self) -> dict:
        """
        Analyze model computing properties.

        Returns:
            dict: Properties of a model
        """
        embeddings_length = self._model.config.decoder.max_position_embeddings
        tensor = torch.ones(1, embeddings_length, dtype=torch.long)

        ids = {"input_ids": tensor,
               "token_type_ids": tensor,
               "attention_mask": tensor}

        statistics = summary(self._model,
                             input_data=ids,
                             decoder_input_ids=tensor,
                             verbose=False)

        model_info = {
            "input_shape": list(statistics.input_size['input_ids']),
            "embedding_size": embeddings_length,
            "output_shape": statistics.summary_list[-1].output_size,
            "num_trainable_params": statistics.trainable_params,
            "vocab_size": self._model.config.decoder.vocab_size,
            "size": statistics.total_param_bytes,
            "max_context_length": self._model.config.max_length
        }

        return model_info

    @report_time
    def infer_sample(self, sample: tuple[str, ...]) -> str | None:
        """
        Infer model on a single sample.

        Args:
            sample (tuple[str, ...]): The given sample for inference with model

        Returns:
            str | None: A prediction
        """
        tokenizer = AutoTokenizer.from_pretrained(self._model_name)

        tokens = tokenizer(sample[0], max_length=120, padding=True,
                           return_tensors='pt', truncation=True)

        output = self._model.generate(**tokens)
        result = tokenizer.batch_decode(output, skip_special_tokens=True)

        return result[0]

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
