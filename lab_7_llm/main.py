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
from pandas import DataFrame
from torch.utils.data import Dataset
from torchinfo import summary
from transformers import AutoTokenizer, AutoModelForSequenceClassification

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
        dataset = load_dataset(self._hf_name,  split='validation')
        self._raw_data = dataset.to_pandas()


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
        dataset_no_empty_rows = self._raw_data["text"].dropna()

        properties_dict = {
            "dataset_number_of_samples": self._raw_data.shape[0],
            "dataset_columns": self._raw_data.shape[1],
            "dataset_duplicates": self._raw_data.duplicated().sum(),
            "dataset_empty_rows": self._raw_data.isnull().any(axis=1).sum(),
            "dataset_sample_min_len": dataset_no_empty_rows.map(len).min(),
            "dataset_sample_max_len": dataset_no_empty_rows.map(len).max()
        }

        return properties_dict

    @report_time
    def transform(self) -> None:
        """
        Apply preprocessing transformations to the raw dataset.
        """
        self._data = self._raw_data.rename(columns={"text": ColumnNames.SOURCE.value,
                                                    "label": ColumnNames.TARGET.value})
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
        data = self._data.iloc[index]
        return tuple(data)

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
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        #self._model = AutoModelForSequenceClassification.from_config(AutoConfig.from_pretrained(model_name)).to(device)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)

    def analyze_model(self) -> dict:
        """
        Analyze model computing properties.

        Returns:
            dict: Properties of a model
        """
        input_data = torch.ones((1, 512), dtype=torch.long)
        model_summary = summary(self._model, input_data=input_data, verbose=0)

        return {
            "embedding_size": list(self._model.named_parameters())[1][1].shape[0],
            "input_shape": {'attention_mask': list(input_data.size()), 'input_ids': list(input_data.size())},
            "max_context_length": self._model.config.max_length,
            "num_trainable_params": model_summary.trainable_params,
            "output_shape": model_summary.summary_list[-1].output_size,
            "size": model_summary.total_param_bytes,
            "vocab_size": self._model.config.vocab_size,
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
        input_data = self._tokenizer(" ".join(sample), return_tensors="pt", truncation=True)
        with torch.no_grad():
            logits = self._model(**input_data).logits
        predicted_label = torch.argmax(logits, dim=-1).item()
        return str(predicted_label)

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
