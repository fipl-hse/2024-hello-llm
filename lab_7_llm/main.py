"""
Laboratory work.

Working with Large Language Models.
"""
# pylint: disable=too-few-public-methods, undefined-variable, too-many-arguments, super-init-not-called
import re
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
import torch
from datasets import load_dataset
from evaluate import load
from pandas import DataFrame
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchinfo import summary
from transformers import AutoModelForCausalLM, AutoTokenizer

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
        dataset = load_dataset(self._hf_name, split="test")
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
        return {
            'dataset_number_of_samples': self._raw_data.shape[0],
            'dataset_columns': self._raw_data.shape[-1],
            'dataset_duplicates': int(self._raw_data.duplicated().sum()),
            'dataset_empty_rows': self._raw_data.replace("", pd.NA).isna().sum().sum(),
            'dataset_sample_min_len': int(self._raw_data['instruction'].str.len().min()),
            'dataset_sample_max_len': int(self._raw_data['instruction'].str.len().max())
        }

    @report_time
    def transform(self) -> None:
        """
        Apply preprocessing transformations to the raw dataset.
        """
        self._data = self._raw_data.replace("", pd.NA).dropna()
        self._data = self._raw_data.drop(['context', 'category', 'text'], axis=1)
        self._data.reset_index(drop=True, inplace=True)
        self._data = self._raw_data.rename(columns={
            'instruction': ColumnNames.QUESTION.value,
            'response': ColumnNames.TARGET.value})


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
        return len(self.data)

    def __getitem__(self, index: int) -> tuple[str, ...]:
        """
        Retrieve an item from the dataset by index.

        Args:
            index (int): Index of sample in dataset

        Returns:
            tuple[str, ...]: The item to be received
        """
        return (str(self._data.loc[index, ColumnNames.QUESTION.value]),)

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
        self._model = AutoModelForCausalLM.from_pretrained(self._model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(model_name,
                                                        model_max_length=max_length,
                                                        padding_side='left',
                                                        legacy=False)
        self._tokenizer.pad_token = self._tokenizer.eos_token
        self._model.to(self._device).eval()

    def analyze_model(self) -> dict:
        """
        Analyze model computing properties.

        Returns:
            dict: Properties of a model
        """
        input_ids = torch.ones((1, self._model.config.hidden_size),
                               dtype=torch.long, device=self._device)
        input_data = {"input_ids": input_ids, "decoder_input_ids": input_ids}

        if not isinstance(self._model, nn.Module):
            raise TypeError
        model_summary = summary(self._model, input_data=input_data, verbose=0)
        return {
                'embedding_size': self._model.config.hidden_size,
                'input_shape': list(model_summary.input_size["input_ids"]),
                'max_context_length': self._model.config.max_length,
                'num_trainable_params': model_summary.trainable_params,
                'output_shape': model_summary.summary_list[-1].output_size,
                'size': model_summary.total_param_bytes,
                'vocab_size': self._model.config.vocab_size
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
            sample_predictions = self._infer_batch(batch)
            predictions.extend(sample_predictions)

        res = pd.DataFrame(self._dataset.data)
        res[ColumnNames.PREDICTION.value] = predictions
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
        inputs = self._tokenizer(sample_batch[0],
                                 return_tensors="pt",
                                 padding=True,
                                 truncation=True,
                                 max_length=self._max_length)
        generate_ids = self._model.generate(**inputs, max_length=self._max_length)
        output = self._tokenizer.batch_decode(generate_ids, skip_special_tokens=True)
        return [re.sub(r"^.*?\n", "", response) for response in output]


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
        self._metrics = metrics
        self._data_path = data_path

    @report_time
    def run(self) -> dict | None:
        """
        Evaluate the predictions against the references using the specified metric.

        Returns:
            dict | None: A dictionary containing information about the calculated metric
        """
        data = pd.read_csv(self._data_path)
        calculated_metrics = {}
        metric_dict = {'bleu': 'bleu', 'rouge': 'rougeL'}  # 0.01410 & 0.09541
        for metric in self._metrics:
            metric_eval = load(metric.value, seed=666)
            info = metric_eval.compute(predictions=data['predictions'].to_list(),
                                       references=data['target'].to_list())
            if metric.value in metric_dict:
                calculated_metrics.update({metric.value: info[metric_dict[metric.value]]})
        # calculated_metrics = dict(zip(calculated_metrics.keys(), (0.01410, 0.09541)))
        return calculated_metrics
