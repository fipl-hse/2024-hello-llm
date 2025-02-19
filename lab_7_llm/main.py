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
from torch.utils.data import DataLoader, Dataset
from torchinfo import summary
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from core_utils.llm.llm_pipeline import AbstractLLMPipeline
from core_utils.llm.metrics import Metrics
from core_utils.llm.raw_data_importer import AbstractRawDataImporter
from core_utils.llm.raw_data_preprocessor import AbstractRawDataPreprocessor
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
        raw_dataset = load_dataset(path=self._hf_name, split='test')
        self._raw_data = raw_dataset.to_pandas()

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
        return {
            'dataset_number_of_samples': self._raw_data.shape[0],
            'dataset_columns': self._raw_data.shape[1],
            'dataset_duplicates': int(self._raw_data.duplicated().sum()),
            'dataset_empty_rows': int(self._raw_data.isnull().any(axis=1).sum()),
            "dataset_sample_min_len": int(self._raw_data['text'].dropna().map(len).min()),
            "dataset_sample_max_len": int(self._raw_data['text'].map(len).max())
        }

    @report_time
    def transform(self) -> None:
        """
        Apply preprocessing transformations to the raw dataset.
        """
        self._data = self._raw_data.copy()
        self._data.rename(columns={'label': 'target',
                                   'text': 'source'},
                          inplace=True)
        self._data.reset_index(drop=True).drop_duplicates()


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
        return tuple(self._data.iloc[index])

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
        self._model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)

    def analyze_model(self) -> dict:
        """
        Analyze model computing properties.

        Returns:
            dict: Properties of a model
        """
        max_position_embeddings = self._model.config.max_position_embeddings
        tensor = torch.ones((1, max_position_embeddings), dtype=torch.long)
        input_data = {"input_ids": tensor, "attention_mask": tensor}
        model_summary = summary(self._model, input_data=input_data, verbose=False)
        return {
            "input_shape": {"input_ids": list(tensor.size()),
                            "attention_mask": list(tensor.size())},
            "embedding_size": max_position_embeddings,
            "output_shape": model_summary.summary_list[-1].output_size,
            "num_trainable_params": model_summary.trainable_params,
            "vocab_size": self._model.config.vocab_size,
            "size": model_summary.total_param_bytes,
            "max_context_length": self._model.config.max_length
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
        data_loader = DataLoader(batch_size=self._batch_size,
                                 dataset=self._dataset)
        predictions = []
        for batch in data_loader:
            sample = self._infer_batch(batch)
            predictions.extend(sample)
        res = pd.DataFrame(self._dataset.data)
        res['prediction'] = predictions
        return res

    @torch.no_grad()
    def _infer_batch(self, sample_batch: Sequence[tuple[str, ...]]) -> list[str]:
        """
        Infer model on a single batch.

        Args:
            sample_batch (Sequence[tuple[str, ...]]): Batch to infer the model

        Returns:
            list[str]: Model predictions as strings
        """
        inputs = self._tokenizer(list(list(zip(*sample_batch))[0]),
                                 return_tensors="pt", padding=True,
                                 truncation=True,
                                 collate_fn=lambda x: x)
        outputs = self._model(**inputs)
        return [str(pred.item()) for pred in outputs.logits.argmax(dim=1)]


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
        self.data_path = data_path
        self._metrics = metrics

    @report_time
    def run(self) -> dict | None:
        """
        Evaluate the predictions against the references using the specified metric.

        Returns:
            dict | None: A dictionary containing information about the calculated metric
        """
        data_frame = pd.read_csv(self.data_path)

        predictions = data_frame['prediction']
        targets = data_frame['target']
        res = {}
        for metric in self._metrics:
            scores = load(str(metric.value)).compute(predictions=predictions,
                                                     references=targets, average='micro')
            if metric.value == "f1":
                res[metric.value] = scores["f1"]
            else:
                res[metric.value] = scores[metric.value]
        return res
