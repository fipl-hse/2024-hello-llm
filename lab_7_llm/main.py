"""
Laboratory work.

Working with Large Language Models.
"""
# pylint: disable=too-few-public-methods, undefined-variable, too-many-arguments, super-init-not-called
from pathlib import Path
from typing import Iterable, Sequence

import evaluate
import pandas as pd
import torch
from datasets import load_dataset
from pandas import DataFrame
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset
from torchinfo import summary
from transformers import BertForSequenceClassification, BertTokenizer

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

        if not isinstance(self._raw_data, pd.DataFrame):
            raise TypeError("The downloaded dataset is not a pandas DataFrame.")


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
            'dataset_duplicates': self._raw_data.duplicated().sum(),
            'dataset_empty_rows': self._raw_data.isnull().all(axis=1).sum(),
            'dataset_sample_min_len': self._raw_data['text'].dropna(how='all').map(len).min(),
            'dataset_sample_max_len': self._raw_data['text'].dropna(how='all').map(len).max()

        }

    @report_time
    def transform(self) -> None:
        """
        Apply preprocessing transformations to the raw dataset.
        """
        renamed_dataset = self._raw_data.rename(columns={
            'label': ColumnNames.TARGET.value,
            'text': ColumnNames.SOURCE.value
        }, inplace=False)

        label_map = {'tat': '0',
                     'rus': '1',
                     'kir': '2',
                     'krc': '3',
                     'bak': '4',
                     'sah': '5',
                     'kaz': '6',
                     'tyv': '7',
                     'chv': '8'
                     }

        renamed_dataset[ColumnNames.TARGET.value] = (renamed_dataset[ColumnNames.TARGET.value]
                                                     .map(label_map))

        self._data = renamed_dataset


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
        return (self._data.iloc[index][ColumnNames.SOURCE.value],
                self._data.iloc[index][ColumnNames.TARGET.value])

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

        self._model: Module = (BertForSequenceClassification.from_pretrained(model_name)
                               .to(self._device))
        self._model.eval()
        self._tokenizer = BertTokenizer.from_pretrained(model_name)

    def analyze_model(self) -> dict:
        """
        Analyze model computing properties.

        Returns:
            dict: Properties of a model
        """
        model_config = self._model.config

        embeddings_length = model_config.max_position_embeddings
        ids = torch.ones(1, embeddings_length, dtype=torch.long)
        tokens = {"input_ids": ids, "attention_mask": ids}
        model_summary = summary(self._model, input_data=tokens, device=self._device, verbose=0)

        return {
            "input_shape": {
                "attention_mask": list(model_summary.input_size['attention_mask']),
                "input_ids": list(model_summary.input_size['input_ids'])},
            "embedding_size": model_config.max_position_embeddings,
            "output_shape": model_summary.summary_list[-1].output_size,
            "num_trainable_params": model_summary.trainable_params,
            "vocab_size": model_config.vocab_size,
            "size": model_summary.total_param_bytes,
            "max_context_length": model_config.max_length
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

        return self._infer_batch([sample])[0]

    @report_time
    def infer_dataset(self) -> pd.DataFrame:
        """
        Infer model on a whole dataset.

        Returns:
            pd.DataFrame: Data with predictions
        """
        data_loader = DataLoader(batch_size=self._batch_size, dataset=self._dataset)

        predictions = []
        targets = []

        for batch in data_loader:
            batch_texts, batch_labels = batch
            batch_tuples = [(text,) for text in batch_texts]
            preds = self._infer_batch(batch_tuples)
            predictions.extend(preds)
            targets.extend(batch_labels)

        return DataFrame({ColumnNames.TARGET.value: targets,
                          ColumnNames.PREDICTION.value: predictions})

    @torch.no_grad()
    def _infer_batch(self, sample_batch: Sequence[tuple[str, ...]]) -> list[str]:
        """
        Infer model on a single batch.

        Args:
            sample_batch (Sequence[tuple[str, ...]]): Batch to infer the model

        Returns:
            list[str]: Model predictions as strings
        """
        texts = [sample[0] for sample in sample_batch]
        inputs = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        outputs = self._model(**inputs)
        preds = torch.argmax(outputs.logits, dim=-1)

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
        super().__init__(metrics)
        self._data_path = data_path

    @report_time
    def run(self) -> dict | None:
        """
        Evaluate the predictions against the references using the specified metric.

        Returns:
            dict | None: A dictionary containing information about the calculated metric
        """
        data = pd.read_csv(self._data_path)

        predictions = data[ColumnNames.PREDICTION.value].tolist()
        references = data[ColumnNames.TARGET.value].tolist()
        metric_name = str(list(self._metrics)[0])
        metric_evaluator = evaluate.load(metric_name)
        score = metric_evaluator.compute(predictions=predictions,
                                         references=references,
                                         average='micro')
        if not isinstance(score, dict):
            raise TypeError(f"Expected dict, but got {type(score)}: {score}")

        return score
