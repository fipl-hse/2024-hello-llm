"""
Laboratory work.

Working with Large Language Models.
"""
# pylint: disable=too-few-public-methods, undefined-variable, too-many-arguments, super-init-not-called
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
import torch as torch
from pandas import DataFrame

from core_utils.llm.llm_pipeline import AbstractLLMPipeline
from core_utils.llm.metrics import Metrics
from core_utils.llm.raw_data_importer import AbstractRawDataImporter
from core_utils.llm.raw_data_preprocessor import AbstractRawDataPreprocessor, ColumnNames
from datasets import load_dataset
from torch.utils.data import Dataset

from core_utils.llm.task_evaluator import AbstractTaskEvaluator
from core_utils.llm.time_decorator import report_time

from transformers import AutoModel, AutoTokenizer
from torchinfo import summary


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
        self._raw_data = load_dataset(self._hf_name, split='train').to_pandas()


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
        properties = dict()
        properties['dataset_number_of_samples'] = len(self._raw_data)
        properties['dataset_columns'] = len(self._raw_data.columns)
        properties['dataset_duplicates'] = int(self._raw_data.duplicated().sum())
        properties['dataset_empty_rows'] = len(self._raw_data) - len(self._raw_data.dropna())
        properties['dataset_sample_min_len'] = int(self._raw_data['article_content'].str.len().min())
        properties['dataset_sample_max_len'] = int(self._raw_data['article_content'].str.len().max())

        return properties

    @report_time
    def transform(self) -> None:
        """
        Apply preprocessing transformations to the raw dataset.
        """
        self._data = self._raw_data.drop(columns=['title', 'date', 'url'])
        self._data = self._data.rename(columns={'article_content': ColumnNames.SOURCE,
                                                'summary': ColumnNames.TARGET})
        self._data = self._data.dropna()
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
        return tuple(self._data.loc[index, ColumnNames.SOURCE], self._data.loc[index, ColumnNames.TARGET])

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

    _model: torch.nn.Module

    def __init__(self, model_name: str, dataset: TaskDataset, max_length: int, batch_size: int, device: str) -> None:
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
        self._model = AutoModel.from_pretrained(self._model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)

    def analyze_model(self) -> dict:
        """
        Analyze model computing properties.

        Returns:
            dict: Properties of a model
        """
        properties = dict()
        input_data = torch.ones((1, self._model.config.d_model), dtype=torch.long)
        model_statistics = summary(self._model, input_data=input_data, decoder_input_ids=input_data)
        properties['input_shape'] = list(model_statistics.input_size)
        properties['embedding_size'] = self._model.config.d_model
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
        return self._infer_batch([sample])[0]

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
        # model_input = self._tokenizer(list(sample_batch[0]),
        #                               return_tensors='pt', padding=True, truncation=True, max_length=64)
        # print(model_input)
        # model_output = self._model.forward(input_ids=model_input['input_ids'],
        #                                    attention_mask=model_input['attention_mask'],
        #                                    decoder_input_ids=model_input['input_ids'])
        #
        # return self._tokenizer.batch_decode(model_output, skip_special_tokens=True)


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
