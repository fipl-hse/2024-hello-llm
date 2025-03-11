"""
Laboratory work.

Fine-tuning Large Language Models for a downstream task.
"""
# pylint: disable=too-few-public-methods, undefined-variable, duplicate-code, unused-argument, too-many-arguments
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

from config.lab_settings import SFTParams
from core_utils.llm.llm_pipeline import AbstractLLMPipeline
from core_utils.llm.metrics import Metrics
from core_utils.llm.raw_data_importer import AbstractRawDataImporter
from core_utils.llm.raw_data_preprocessor import AbstractRawDataPreprocessor, ColumnNames
from core_utils.llm.sft_pipeline import AbstractSFTPipeline
from core_utils.llm.task_evaluator import AbstractTaskEvaluator
from core_utils.llm.time_decorator import report_time


class RawDataImporter(AbstractRawDataImporter):
    """
    Custom implementation of data importer.
    """

    @report_time
    def obtain(self) -> None:
        """
        Import dataset.
        """
        self._raw_data = load_dataset(path=self._hf_name, split='train').to_pandas()
        if not isinstance(self._raw_data, pd.DataFrame):
            raise TypeError('Dataset is not Dataframe')


class RawDataPreprocessor(AbstractRawDataPreprocessor):
    """
    Custom implementation of data preprocessor.
    """

    def analyze(self) -> dict:
        """
        Analyze preprocessed dataset.

        Returns:
            dict: dataset key properties.
        """
        return {
            'dataset_number_of_samples': len(self._raw_data),
            'dataset_columns': len(self._raw_data.columns),
            'dataset_duplicates': int(self._raw_data.duplicated().sum()),
            'dataset_empty_rows': int(self._raw_data.isnull().all(axis=1).sum()),
            'dataset_sample_min_len': int(self._raw_data['comment'].str.len().min()),
            'dataset_sample_max_len': int(self._raw_data['comment'].str.len().max()),
        }

    @report_time
    def transform(self) -> None:
        """
        Apply preprocessing transformations to the raw dataset.
        """
        self._data = (self._raw_data.rename(columns={
            'toxic': ColumnNames.TARGET.value,
            'comment': ColumnNames.SOURCE.value}).
                      drop_duplicates().
                      reset_index(drop=True))
        self._data[ColumnNames.TARGET.value] = self._data[ColumnNames.TARGET.value].apply(int)


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
        return (str(self._data.loc[index, ColumnNames.SOURCE.value]),
                str(self._data.loc[index, ColumnNames.TARGET.value]))

    @property
    def data(self) -> DataFrame:
        """
        Property with access to preprocessed DataFrame.

        Returns:
            pandas.DataFrame: Preprocessed DataFrame
        """
        return self._data


def tokenize_sample(
    sample: pd.Series, tokenizer: AutoTokenizer, max_length: int
) -> dict[str, torch.Tensor]:
    """
    Tokenize sample.

    Args:
        sample (pandas.Series): sample from a dataset
        tokenizer (transformers.models.auto.tokenization_auto.AutoTokenizer): Tokenizer to tokenize
            original data
        max_length (int): max length of sequence

    Returns:
        dict[str, torch.Tensor]: Tokenized sample
    """


class TokenizedTaskDataset(Dataset):
    """
    A class that converts pd.DataFrame to Dataset and works with it.
    """

    def __init__(self, data: pd.DataFrame, tokenizer: AutoTokenizer, max_length: int) -> None:
        """
        Initialize an instance of TaskDataset.

        Args:
            data (pandas.DataFrame): Original data
            tokenizer (transformers.models.auto.tokenization_auto.AutoTokenizer): Tokenizer to
                tokenize the dataset
            max_length (int): max length of a sequence
        """

    def __len__(self) -> int:
        """
        Return the number of items in the dataset.

        Returns:
            int: The number of items in the dataset
        """

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """
        Retrieve an item from the dataset by index.

        Args:
            index (int): Index of sample in dataset

        Returns:
            dict[str, torch.Tensor]: An element from the dataset
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
            model_name (str): The name of the pre-trained model.
            dataset (TaskDataset): The dataset to be used for translation.
            max_length (int): The maximum length of generated sequence.
            batch_size (int): The size of the batch inside DataLoader.
            device (str): The device for inference.
        """
        super().__init__(model_name, dataset, max_length, batch_size, device)

        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
        self._model.eval()

    def analyze_model(self) -> dict:
        """
        Analyze model computing properties.

        Returns:
            dict: Properties of a model
        """
        if not isinstance(self._model, torch.nn.Module):
            raise ValueError("No model here")
        dummy_inputs = torch.ones((1,
                                   self._model.config.max_position_embeddings),
                                  dtype=torch.long)

        input_data = {'input_ids': dummy_inputs,
                      'attention_mask': dummy_inputs}

        model_summary = summary(self._model, input_data=input_data)

        return {
            'input_shape': {k: list(v.shape) for k, v in input_data.items()},
            'embedding_size': self._model.config.max_position_embeddings,
            'output_shape': model_summary.summary_list[-1].output_size,
            'num_trainable_params': model_summary.trainable_params,
            'vocab_size': self._model.config.vocab_size,
            'size': model_summary.total_param_bytes,
            'max_context_length': self._model.config.max_length
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
        loader_data = DataLoader(dataset=self._dataset, batch_size=self._batch_size)
        predictions = []
        with torch.no_grad():
            for batch in loader_data:
                batch_predictions = self._infer_batch(batch)
                predictions.extend(batch_predictions)

        infer_dataframe = pd.DataFrame(self._dataset.data)
        infer_dataframe[ColumnNames.PREDICTION.value] = predictions

        return infer_dataframe

    @torch.no_grad()
    def _infer_batch(self, sample_batch: Sequence[tuple[str, ...]]) -> list[str]:
        """
        Infer single batch.

        Args:
            sample_batch (Sequence[tuple[str, ...]]): batch to infer the model

        Returns:
            list[str]: model predictions as strings
        """
        encoded_batch = self._tokenizer(
            sample_batch[0],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self._max_length)

        encoded_batch.to(self._device)
        outputs = self._model(**encoded_batch).logits
        return [str(pred.argmax().item()) for pred in outputs]

        # preds = torch.argmax(self._model(**encoded_batch).logits, dim=1)
        # return [str(pred.item()) for pred in preds]


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
        self._metrics = [load(metric.value) for metric in metrics]

    def run(self) -> dict | None:
        """
        Evaluate the predictions against the references using the specified metric.

        Returns:
            dict | None: A dictionary containing information about the calculated metric
        """
        eval_data = pd.read_csv(self._data_path)

        predictions = eval_data[ColumnNames.PREDICTION.value]
        targets = eval_data[ColumnNames.TARGET.value]

        eval_results = {}
        for metric in self._metrics:
            scores = metric.compute(predictions=predictions,
                                    references=targets,
                                    average="micro")
            #key = metric.__class__.__name__.lower()
            eval_results[metric.name] = scores[metric.name]

        return eval_results


class SFTPipeline(AbstractSFTPipeline):
    """
    A class that initializes a model, fine-tuning.
    """

    def __init__(self, model_name: str, dataset: Dataset, sft_params: SFTParams) -> None:
        """
        Initialize an instance of ClassificationSFTPipeline.

        Args:
            model_name (str): The name of the pre-trained model.
            dataset (torch.utils.data.dataset.Dataset): The dataset used.
            sft_params (SFTParams): Fine-Tuning parameters.
        """

    def run(self) -> None:
        """
        Fine-tune model.
        """
