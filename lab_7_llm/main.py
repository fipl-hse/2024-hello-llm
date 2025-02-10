"""
Laboratory work.

Working with Large Language Models.
"""
# pylint: disable=too-few-public-methods, undefined-variable, too-many-arguments, super-init-not-called
from pathlib import Path
from typing import Iterable, Sequence
from datasets import load_dataset

import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from core_utils.llm.llm_pipeline import AbstractLLMPipeline
from core_utils.llm.metrics import Metrics
from core_utils.llm.raw_data_importer import AbstractRawDataImporter
from core_utils.llm.raw_data_preprocessor import AbstractRawDataPreprocessor
from core_utils.llm.task_evaluator import AbstractTaskEvaluator
from core_utils.llm.time_decorator import report_time
from torchinfo import summary
from transformers import AutoModelForTokenClassification, AutoTokenizer
import torch
import fastapi


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
        self._raw_data = pd.DataFrame(
            load_dataset(self._hf_name, split="validation", trust_remote_code=True)
        )


class RawDataPreprocessor(AbstractRawDataPreprocessor):
    """
    A class that analyzes and preprocesses a dataset.
    """

    def __init__(self, raw_data: pd.DataFrame):
        super().__init__(raw_data)

    def analyze(self) -> dict:
        """
        Analyze a dataset.

        Returns:
            dict: Dataset key properties
        """
        temp_df = self._raw_data.copy()

        for column in temp_df.columns:
            temp_df[column] = temp_df[column].str.join(" ")

        duplicates = temp_df[temp_df.duplicated()].shape[0]

        empty_rows = self._raw_data[self._raw_data.isnull().all(axis=1)].shape[0]

        no_na_df = self._raw_data.dropna()

        return {
            "dataset_columns": self._raw_data.shape[1],
            "dataset_duplicates": duplicates,
            "dataset_empty_rows": empty_rows,
            "dataset_number_of_samples": self._raw_data.shape[0],
            "dataset_sample_max_len": int(no_na_df["tokens"].str.len().max()),
            "dataset_sample_min_len": int(no_na_df["tokens"].str.len().min()),
        }

    @report_time
    def transform(self) -> None:
        """
        Apply preprocessing transformations to the raw dataset.
        """
        self._data = self._raw_data.rename(
            columns={"ner_tags": "target", "tokens": "source"}
        ).reset_index(drop=True)


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
        return self._data.shape[0]

    def __getitem__(self, index: int) -> tuple[str, ...]:
        """
        Retrieve an item from the dataset by index.

        Args:
            index (int): Index of sample in dataset

        Returns:
            tuple[str, ...]: The item to be received
        """
        return tuple(self._data.iloc[index].array)

    @property
    def data(self) -> pd.DataFrame:
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
        self._model = AutoModelForTokenClassification.from_pretrained(self._model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self._model_name)

    def analyze_model(self) -> dict:
        """
        Analyze model computing properties.

        Returns:
            dict: Properties of a model
        """
        model_config = self._model.config

        input_size = [self._batch_size, self._max_length]
        output_size = [self._batch_size, self._max_length, model_config.dim]
        info = summary(self._model, input_size=input_size, verbose=0)

        return {
            "input_shape": info.input_size,
            "num_trainable_params": info.trainable_params,
            "size": info.total_param_bytes,
            "vocab_size": model_config.vocab_size,
            "max_context_length": self._max_length,
            "output_shape": output_size,
            "embedding_size": model_config.dim,
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
        if self._model:
            input_data = self.tokenizer(
                sample,
                truncation=True,
                add_special_tokens=True,
                is_split_into_words=True,
                return_tensors="pt",
            )

            with torch.no_grad():
                logits = self._model(**input_data).logits

            pred = torch.argmax(logits, dim=2)
            # predicted_ token_class = [self._model.config.id2label[t.item()] for t in pred[0]]

            # pred_labels = pred[0][1:-1]

            return [int(t) for t in pred[0]]

        return

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
