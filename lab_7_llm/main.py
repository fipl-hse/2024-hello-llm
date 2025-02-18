"""
Laboratory work.

Working with Large Language Models.
"""
# pylint: disable=too-few-public-methods, undefined-variable, too-many-arguments, super-init-not-called
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
import torch
from datasets import load_dataset
from evaluate import load
from pandas import DataFrame
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset
from torchinfo import summary
from transformers import AutoModelForSequenceClassification, AutoTokenizer

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

        self._raw_data = load_dataset(
            path=self._hf_name,
            split='validation'
        ).to_pandas()

        if not isinstance(self._raw_data, pd.DataFrame):
            raise TypeError('The downloaded dataset is not pd.DataFrame')


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

        dataset_number_of_samples = len(self._raw_data)
        dataset_columns = len(self._raw_data.columns)

        dataset_duplicates = self._raw_data.duplicated().sum()

        num_nans = self._raw_data.isnull().any().sum()
        num_empty_lines = len(
            self._raw_data[
                (self._raw_data['text'] == '') | (self._raw_data['labels'] == '')
            ]
        )
        dataset_empty_rows = num_nans + num_empty_lines

        raw_data_no_nans = self._raw_data.dropna()

        len_counts = raw_data_no_nans['text'].apply(len)
        dataset_sample_min_len = len_counts.min()
        dataset_sample_max_len = len_counts.max()

        return {
            'dataset_number_of_samples': dataset_number_of_samples,
            'dataset_columns': dataset_columns,
            'dataset_duplicates': dataset_duplicates.item(),
            'dataset_empty_rows': dataset_empty_rows.item(),
            'dataset_sample_min_len': dataset_sample_min_len.item(),
            'dataset_sample_max_len': dataset_sample_max_len.item()
        }

    @report_time
    def transform(self) -> None:
        """
        Apply preprocessing transformations to the raw dataset.
        """
        label2id = {
            "ar": 2,
            "bg": 12,
            "de": 4,
            "el": 10,
            "en": 13,
            "es": 8,
            "fr": 14,
            "hi": 9,
            "it": 5,
            "ja": 0,
            "nl": 1,
            "pl": 3,
            "pt": 6,
            "ru": 16,
            "sw": 18,
            "th": 17,
            "tr": 7,
            "ur": 11,
            "vi": 19,
            "zh": 15
        }

        self._data = self._raw_data.rename(columns={
            'labels': str(ColumnNames.TARGET),
            'text': str(ColumnNames.SOURCE)
        })

        self._data[str(ColumnNames.TARGET)] = self._data[str(ColumnNames.TARGET)].apply(
            lambda lang: label2id[lang]
        )

        self._data.drop_duplicates(inplace=True)
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
        return (self._data[str(ColumnNames.SOURCE)][index],)

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

        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(self._model_name)

        self._model: Module
        self._model.eval()
        self._model.to(self._device)

    def analyze_model(self) -> dict:
        """
        Analyze model computing properties.

        Returns:
            dict: Properties of a model
        """

        model_config = self._model.config

        ids = torch.ones(1, model_config.max_position_embeddings, dtype=torch.long)
        model_summary = summary(
            self._model,
            input_data={'input_ids': ids, 'attention_mask': ids}
        )

        return {
            'input_shape': {
                input_type: list(shape) for input_type, shape
                in model_summary.input_size.items()
            },
            'embedding_size': model_config.max_position_embeddings,
            'output_shape': model_summary.summary_list[-1].output_size,
            'num_trainable_params': model_summary.trainable_params,
            'vocab_size': model_config.vocab_size,
            'size': model_summary.total_param_bytes,
            'max_context_length': model_config.max_length
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

        if not self._model:
            return None

        return self._infer_batch([sample])[0]

    @report_time
    def infer_dataset(self) -> pd.DataFrame:
        """
        Infer model on a whole dataset.

        Returns:
            pd.DataFrame: Data with predictions
        """

        data_loader = DataLoader(self._dataset,
                                 batch_size=self._batch_size)

        dataset_predictions = []
        for batch in data_loader:
            dataset_predictions.extend(self._infer_batch(batch))

        return pd.DataFrame(
            {
                str(ColumnNames.TARGET): self._dataset.data[str(ColumnNames.TARGET)],
                str(ColumnNames.PREDICTION): dataset_predictions
            }
        )

    @torch.no_grad()
    def _infer_batch(self, sample_batch: Sequence[tuple[str, ...]]) -> list[str]:
        """
        Infer model on a single batch.

        Args:
            sample_batch (Sequence[tuple[str, ...]]): Batch to infer the model

        Returns:
            list[str]: Model predictions as strings
        """
        if not self._model:
            raise ValueError('Model is not available')

        batch_list = list(sample_batch[0])
        tokens = self._tokenizer(batch_list,
                                 max_length=self._max_length,
                                 padding=True,
                                 truncation=True,
                                 return_tensors='pt').to(self._device)

        output = self._model(**tokens)

        batch_predictions = torch.argmax(output.logits, dim=1).tolist()
        return [str(pred) for pred in batch_predictions]


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

    @report_time
    def run(self) -> dict | None:
        """
        Evaluate the predictions against the references using the specified metric.

        Returns:
            dict | None: A dictionary containing information about the calculated metric
        """
        predictions_df = pd.read_csv(self._data_path)

        metrics_dict = {}
        for metric in self._metrics:
            metric_computer = load(str(metric)) # надо посмотреть, как load обрабатывает список

            score = metric_computer.compute(
                references=predictions_df[str(ColumnNames.TARGET)],
                predictions=predictions_df[str(ColumnNames.PREDICTION)],
                average='micro'
            )
            metrics_dict[list(score.keys())[0]] = list(score.values())[0]

        return metrics_dict if metrics_dict else None
