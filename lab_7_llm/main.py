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
from transformers import AutoTokenizer, GPTNeoXForCausalLM

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
        qa_dataset = load_dataset(self._hf_name, split='test')
        if qa_dataset:
            self._raw_data = qa_dataset.to_pandas()
        if not isinstance(self._raw_data, pd.DataFrame):
            raise TypeError('Error. Downloaded dataset is not pd.DataFrame.')


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
        return {'dataset_number_of_samples': len(self._raw_data),
                'dataset_columns': self._raw_data.columns.size,
                'dataset_duplicates': self._raw_data.duplicated().sum(),
                'dataset_empty_rows': self._raw_data.isna().sum().sum(),
                'dataset_sample_min_len': min(self._raw_data.dropna()['instruction'].apply(len)),
                'dataset_sample_max_len': max(self._raw_data.dropna()['instruction'].apply(len))}

    @report_time
    def transform(self) -> None:
        """
        Apply preprocessing transformations to the raw dataset.
        """
        self._data = self._raw_data.\
            drop(['context', 'category', 'text'], axis=1).\
            rename(columns={'instruction': ColumnNames.QUESTION.value,
                            'response': ColumnNames.TARGET.value})
        self._data.reset_index(inplace=True, drop=True)


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
        return tuple([self._data.iloc[index][ColumnNames.QUESTION.value]])

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
        self._model = GPTNeoXForCausalLM.from_pretrained(self._model_name)
        self._model: Module
        self._model.to(device)
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name,
                                                        model_max_length=max_length,
                                                        padding_side='left')

        self._tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    def analyze_model(self) -> dict:
        """
        Analyze model computing properties.

        Returns:
            dict: Properties of a model
        """
        if not self._model:
            return {}

        ids = torch.ones(1, self._model.config.max_position_embeddings, dtype=torch.long)
        model_summary = summary(
            self._model,
            input_data={
                "input_ids": ids,
                "attention_mask": ids}
        )
        model_configurations = self._model.config

        return {
            'input_shape': {
                'attention_mask': list(model_summary.input_size['attention_mask']),
                'input_ids': list(model_summary.input_size['input_ids'])
            },
            'embedding_size': model_configurations.max_position_embeddings,
            'output_shape': model_summary.summary_list[-1].output_size,
            'num_trainable_params': model_summary.trainable_params,
            'vocab_size': model_configurations.vocab_size,
            'size': model_summary.total_param_bytes,
            'max_context_length': model_configurations.max_length
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
        batch = [sample]
        prediction = self._infer_batch(batch)[0]
        if prediction and isinstance(prediction, str):
            return prediction

        return None

    @report_time
    def infer_dataset(self) -> pd.DataFrame:
        """
        Infer model on a whole dataset.

        Returns:
            pd.DataFrame: Data with predictions
        """
        dataset_loader = DataLoader(self._dataset, self._batch_size)
        targets = self._dataset.data[ColumnNames.TARGET.value].values
        predictions = []

        for batch in dataset_loader:
            predictions.extend(self._infer_batch(batch[0]))

        data_predictions = pd.DataFrame({ColumnNames.TARGET.value: targets,
                                         ColumnNames.PREDICTION.value: predictions})

        return data_predictions

    @torch.no_grad()
    def _infer_batch(self, sample_batch: Sequence[tuple[str, ...]]) -> list[str]:
        """
        Infer model on a single batch.

        Args:
            sample_batch (Sequence[tuple[str, ...]]): Batch to infer the model

        Returns:
            list[str]: Model predictions as strings
        """

        input_ids = self._tokenizer.batch_encode_plus(sample_batch,
                                                      return_tensors="pt",
                                                      max_length=self._max_length,
                                                      padding=True,
                                                      truncation=True).to(self._device)
        with torch.no_grad():
            outputs = self._model.generate(
                input_ids["input_ids"],
                attention_mask=input_ids["attention_mask"],
                max_length=self._max_length
            )

        return [self._tokenizer.decode(output[input_ids["input_ids"].shape[1] + 1:],
                                       skip_special_tokens=True) for output in outputs]


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
        self._metrics = [load(str(metric)) for metric in metrics]
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

        predictions = data[ColumnNames.PREDICTION.value]
        references = data[ColumnNames.TARGET.value]
        for metric in self._metrics:
            calculated_metrics[str(metric)] = metric.compute(predictions=predictions,
                                                             references=references)

        return calculated_metrics
