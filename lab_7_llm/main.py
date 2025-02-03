"""
Laboratory work.

Working with Large Language Models.
"""

from core_utils.llm.raw_data_importer import AbstractRawDataImporter
from core_utils.llm.raw_data_preprocessor import AbstractRawDataPreprocessor
from core_utils.llm.time_decorator import report_time
# pylint: disable=too-few-public-methods, undefined-variable, too-many-arguments, super-init-not-called
from datasets import load_dataset
from pathlib import Path
from typing import Iterable, Sequence
import pandas as pd


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
        self._raw_data = load_dataset(path=self._hf_name, split='if_test').to_pandas()

        if isinstance(self._raw_data, pd.DataFrame):
            raise TypeError


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
        properties['dataset_number_of_samples'] = self._raw_data.shape[0]
        properties['dataset_columns'] = self._raw_data.shape[1]
        properties['dataset_duplicates'] = self._raw_data.duplicated().sum()
        empty_str = self._raw_data[['ru', 'en', 'ru_annotated']].apply(lambda row: sum(len(x) for x in row), axis=1)
        self._raw_data = self._raw_data.drop(empty_str[empty_str == 0], axis=0)
        properties['dataset_empty_rows'] = self._raw_data.isna().sum().sum() + len(empty_str[empty_str == 0])
        if properties['dataset_empty_rows'] > 0:
            self._raw_data = self._raw_data.dropna()
        properties['dataset_sample_max_len'] = self._raw_data['ru'].apply(lambda x: len(x)).max()
        properties['dataset_sample_min_len'] = self._raw_data['ru'].apply(lambda x: len(x)).min()
        return properties

    @report_time
    def transform(self) -> None:
        """
        Apply preprocessing transformations to the raw dataset.
        """


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

    def __len__(self) -> int:
        """
        Return the number of items in the dataset.

        Returns:
            int: The number of items in the dataset
        """

    def __getitem__(self, index: int) -> tuple[str, ...]:
        """
        Retrieve an item from the dataset by index.

        Args:
            index (int): Index of sample in dataset

        Returns:
            tuple[str, ...]: The item to be received
        """

    @property
    def data(self) -> DataFrame:
        """
        Property with access to preprocessed DataFrame.

        Returns:
            pandas.DataFrame: Preprocessed DataFrame
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
            model_name (str): The name of the pre-trained model
            dataset (TaskDataset): The dataset used
            max_length (int): The maximum length of generated sequence
            batch_size (int): The size of the batch inside DataLoader
            device (str): The device for inference
        """

    def analyze_model(self) -> dict:
        """
        Analyze model computing properties.

        Returns:
            dict: Properties of a model
        """

    @report_time
    def infer_sample(self, sample: tuple[str, ...]) -> str | None:
        """
        Infer model on a single sample.

        Args:
            sample (tuple[str, ...]): The given sample for inference with model

        Returns:
            str | None: A prediction
        """

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
