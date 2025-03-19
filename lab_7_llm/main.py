"""
Laboratory work. no meow I hate pycharm so much and venv is the worst ting to exist also

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
        self._raw_data = load_dataset(self._hf_name, split="validation").to_pandas()
        if not isinstance(self._raw_data, pd.DataFrame):
            raise TypeError('downloaded dataset is not pd.DataFrame')


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
        key_properties = {'dataset_number_of_samples': self._raw_data.shape[0],
                        'dataset_columns': self._raw_data.shape[1],
                        'dataset_duplicates': self._raw_data.duplicated().sum(),
                        'dataset_empty_rows': self._raw_data.isnull().any(axis=1).sum(),
                        'dataset_sample_min_len': self._raw_data['comment_text'].dropna().str.len().min(),
                        'dataset_sample_max_len': self._raw_data['comment_text'].dropna().str.len().max()}
        return key_properties

    @report_time
    def transform(self) -> None:
        """
        Apply preprocessing transformations to the raw dataset.
        """
        # self._data = self._raw_data.copy()
        self._data = self._raw_data.drop(['id'], axis=1).rename(columns={
            'label': ColumnNames.TARGET.value,
            'comment_text': ColumnNames.SOURCE.value}).reset_index(drop=True)


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
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_name)

    def analyze_model(self) -> dict:
        """
        Analyze model computing properties.

        Returns:
            dict: Properties of a model
        """
        tensor = torch.zeros(1, self._model.config.max_position_embeddings, dtype=torch.long)
        model_summary = summary(
            self._model,
            input_data={'attention_mask': tensor, 'input_ids': tensor},
            device='cpu'
        )
        model_props = {'input_shape':
                           {'attention_mask':list(tensor.size()),
                            'input_ids':list(tensor.size())},
                       'embedding_size': self._model.config.max_position_embeddings,
                       'output_shape': model_summary.summary_list[-1].output_size,
                       'num_trainable_params':model_summary.trainable_params,
                       'vocab_size': self._model.config.vocab_size,
                       'size':model_summary.total_param_bytes,
                       'max_context_length': self._model.config.max_length}
        return model_props


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
        dataframe = pd.DataFrame(self._dataset.data)
        predicted = []
        for batch in dataloader:
            predicted.extend(self._infer_batch(batch))
        dataframe[ColumnNames.PREDICTION.value] = predicted
        return dataframe

    @torch.no_grad()
    def _infer_batch(self, sample_batch: Sequence[tuple[str, ...]]) -> list[str]:
        """
        Infer model on a single batch.

        Args:
            sample_batch (Sequence[tuple[str, ...]]): Batch to infer the model

        Returns:
            list[str]: Model predictions as strings
        """
        if not isinstance(self._model, torch.nn.Module):
            raise TypeError('The model is not a torch.nn.Module instance.')

        inputs = self._tokenizer(
            *sample_batch,
            max_length=self._max_length,
            padding=True,
            truncation=True,
            return_tensors='pt'
        ).to(self._device)
        outputs = self._model(**inputs)
        predicted = torch.argmax(outputs.logits, dim=1).tolist()
        return [str(pred) for pred in predicted]


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
        dataframe = pd.read_csv(self._data_path)
        metrics_scores = {}
        for metric in self._metrics:
            evaluator = evaluate.load(metric.value)
            score = evaluator.compute(
                references=dataframe[ColumnNames.TARGET.value],
                predictions=dataframe[ColumnNames.PREDICTION.value],
                average=None
            )
            metrics_scores[metric.value] = score[metric.value]
        return metrics_scores
