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
from torch.utils.data import DataLoader, Dataset
from torchinfo import summary
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

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
        self._raw_data = load_dataset(path=self._hf_name, split='test').to_pandas()
        if not isinstance(self._raw_data, pd.DataFrame):
            raise TypeError('Dataset is not pd.Dataframe')


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
            'dataset_number_of_samples': len(self._raw_data),
            'dataset_columns': self._raw_data.shape[1],
            'dataset_duplicates': self._raw_data.duplicated().sum(),
            'dataset_empty_rows': self._raw_data.isna().sum().sum(),
            'dataset_sample_min_len': self._raw_data['info'].str.len().min(),
            'dataset_sample_max_len': self._raw_data['info'].str.len().max(),
        }

    @report_time
    def transform(self) -> None:
        """
        Apply preprocessing transformations to the raw dataset.
        """
        self._data = (self._raw_data.rename(columns={
            'info': ColumnNames.SOURCE.value,
            'summary': ColumnNames.TARGET.value}).reset_index(drop=True))


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
        return (self._data.source.iloc[index],)

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
            self, model_name: str, dataset: TaskDataset,
            max_length: int, batch_size: int, device: str
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
        self._model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self._device)
        self._tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=max_length)

    def analyze_model(self) -> dict:
        """
        Analyze model computing properties.

        Returns:
            dict: Properties of a model
        """
        if self._model is None:
            raise ValueError("No model here")
        tensor = torch.ones(
            (1, self._model.config.encoder.max_position_embeddings),
            dtype=torch.long)
        inputs = {"input_ids": tensor, "attention_mask": tensor}
        summary_m = summary(
            self._model, input_data=inputs, decoder_input_ids=tensor, verbose=False
        )

        return {'input_shape': list(tensor.shape), 'embedding_size': list(tensor.shape)[1],
                'output_shape': summary_m.summary_list[-1].output_size,
                'num_trainable_params': summary_m.trainable_params,
                'vocab_size': self._model.config.encoder.vocab_size,
                'size': summary_m.total_param_bytes,
                'max_context_length': self._model.config.max_length}

    @report_time
    def infer_sample(self, sample: tuple[str, ...]) -> list[str] | None:
        """
        Infer model on a single sample.

        Args:
            sample (tuple[str, ...]): The given sample for inference with model

        Returns:
            str | None: A prediction
        """
        if not self._model:
            return None

        return self._infer_batch([sample][0])


    @report_time
    def infer_dataset(self) -> pd.DataFrame:
        """
        Infer model on a whole dataset.

        Returns:
            pd.DataFrame: Data with predictions
        """
        loader_data = DataLoader(dataset=self._dataset, batch_size=self._batch_size)
        predictions = []

        for batch in loader_data:
            with torch.no_grad():
                batch_predictions = self._infer_batch(batch)
                predictions.extend(batch_predictions)

        infer_dataframe = pd.DataFrame(self._dataset.data)
        infer_dataframe[ColumnNames.PREDICTION.value] = predictions

        return infer_dataframe


    @torch.no_grad()
    def _infer_batch(self, sample_batch: Sequence[tuple[str, ...]]) -> list[str]:
        """
        Infer model on a single batch.

        Args:
            sample_batch (Sequence[tuple[str, ...]]): Batch to infer the model

        Returns:
            list[str]: Model predictions as strings
        """
        encoded_batch = self._tokenizer(
            sample_batch[0],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self._max_length)

        output_ids = self._model.generate(
            input_ids=encoded_batch["input_ids"],
            max_length=self._max_length
        )

        not_strings_predictions = self._tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        return [str(string_prediction) for string_prediction in not_strings_predictions]


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
        eval_data = pd.read_csv(self.data_path)

        predictions = eval_data[ColumnNames.PREDICTION.value]
        targets = eval_data[ColumnNames.TARGET.value]

        eval_results = {}
        for metric in self._metrics:
            scores = load(metric.value, seed=77).compute(predictions=predictions,
                                                         references=targets)
            if metric.value == "rouge":
                eval_results[metric.value] = scores["rougeL"]
            else:
                eval_results[metric.value] = scores[metric.value]

        return eval_results
