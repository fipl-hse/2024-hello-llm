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
        dataset = load_dataset(self._hf_name, split='validation')
        self._raw_data = pd.DataFrame(dataset)


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
        dataset_properties = {
            "dataset_number_of_samples": self._raw_data.shape[0],
            "dataset_columns": self._raw_data.shape[1],
            "dataset_duplicates": self._raw_data.duplicated().sum(),
            "dataset_empty_rows": self._raw_data.isnull().sum(axis=1).sum(),
            "dataset_sample_min_len": self._raw_data['comment_text'].dropna().str.len().min(),
            "dataset_sample_max_len": self._raw_data['comment_text'].dropna().str.len().max(),
        }

        return dataset_properties

    @report_time
    def transform(self) -> None:
        """
        Apply preprocessing transformations to the raw dataset.
        """
        drop_cols = ['id', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        self._raw_data.drop(columns=drop_cols, errors='ignore', inplace=True)

        self._raw_data.rename(columns={
            'toxic': ColumnNames.TARGET.value,
            'comment_text': ColumnNames.SOURCE.value
        }, inplace=True)

        self._raw_data.reset_index(drop=True, inplace=True)
        self._data = self._raw_data


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
        super().__init__(model_name, dataset, max_length, batch_size, device)

        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)

    def analyze_model(self) -> dict:
        """
        Analyze model computing properties.

        Returns:
            dict: Properties of a model
        """
        dummy_inputs = torch.ones((1,
                                  self._model.config.max_position_embeddings),
                                  dtype=torch.long)

        input_data = {'input_ids': dummy_inputs,
                      'attention_mask': dummy_inputs}

        if isinstance(self._model, torch.nn.Module):
            model_summary = summary(self._model, input_data=input_data, verbose=0)
            model_properties = {
                'input_shape': {k: list(v.shape) for k, v in input_data.items()},
                'embedding_size': self._model.config.max_position_embeddings,
                'output_shape': model_summary.summary_list[-1].output_size,
                'num_trainable_params': model_summary.trainable_params,
                'vocab_size': self._model.config.vocab_size,
                'size': model_summary.total_param_bytes,
                'max_context_length': self._model.config.max_length
            }
            return model_properties

        raise TypeError("model is not a valid torch.nn.Module")

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
        dataloader = DataLoader(dataset=self._dataset, batch_size=self._batch_size)

        all_targets = []
        all_predictions = []
        for batch in dataloader:
            sources, targets = batch
            sample_batch = list(zip(sources, targets))
            preds = self._infer_batch(sample_batch)
            all_predictions.extend(preds)
            all_targets.extend(targets)

        return pd.DataFrame({ColumnNames.TARGET: all_targets,
                             ColumnNames.PREDICTION: all_predictions})

    @torch.no_grad()
    def _infer_batch(self, sample_batch: Sequence[tuple[str, ...]]) -> list[str]:
        """
        Infer model on a single batch.

        Args:
            sample_batch (Sequence[tuple[str, ...]]): Batch to infer the model

        Returns:
            list[str]: Model predictions as strings
        """
        if self._model is None:
            raise ValueError("model has not been initialized")

        texts = [sample[0] for sample in sample_batch]
        inputs = self._tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self._max_length
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)
            preds = torch.argmax(outputs.logits, dim=1)

        return [str(pred.item()) for pred in preds]


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
        self._data_path = data_path
        self._metrics = metrics

    @report_time
    def run(self) -> dict | None:
        """
        Evaluate the predictions against the references using the specified metric.

        Returns:
            dict | None: A dictionary containing information about the calculated metric
        """
        data_frame = pd.read_csv(self._data_path)

        targets = data_frame[ColumnNames.TARGET.value]
        predictions = data_frame[ColumnNames.PREDICTION.value]

        results = {}
        for metric_item in self._metrics:
            metric = load(metric_item.value)
            scores = metric.compute(predictions=predictions,
                                    references=targets,
                                    average="micro")

            results[metric_item.value] = scores.get(metric_item.value)

        return results
