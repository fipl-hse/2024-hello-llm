"""
Laboratory work.

Working with Large Language Models.
"""
# pylint: disable=too-few-public-methods, undefined-variable, too-many-arguments, super-init-not-called
from pathlib import Path
from typing import Iterable, Sequence

import datasets
import pandas as pd
import torch
from datasets import load_dataset
from evaluate import load
from pandas import DataFrame
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
        dataset = load_dataset(
            path=self._hf_name, revision="v2.0", split="test", trust_remote_code=True
        )

        self._raw_data = dataset.to_pandas() if isinstance(dataset, datasets.Dataset) else None

        if self._raw_data is None:
            raise TypeError(
                f"Failed to convert dataset to DataFrame. Expected 'Dataset', "
                f"got {type(dataset)} instead."
            )


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
        if self._raw_data is None:
            raise ValueError("Raw data is not set. Cannot analyze an empty dataset.")

        non_empty_data = self._raw_data.dropna(subset=["text"])

        dataset_properties = {
            "dataset_number_of_samples": len(non_empty_data),
            "dataset_columns": len(non_empty_data.columns),
            "dataset_duplicates": non_empty_data.duplicated().sum(),
            "dataset_empty_rows": len(self._raw_data) - len(non_empty_data),
            "dataset_sample_min_len": min(len(sample) for sample in non_empty_data["text"]),
            "dataset_sample_max_len": max(len(sample) for sample in non_empty_data["text"]),
        }

        return dataset_properties

    @report_time
    def transform(self) -> None:
        """
        Apply preprocessing transformations to the raw dataset.
        """
        if self._raw_data is None:
            raise ValueError("Raw data is not set. Cannot transform an empty dataset.")

        self._data = (
            self._raw_data.drop(columns=["title", "date", "url"])
            .rename(columns={"text": ColumnNames.SOURCE.value, "summary": ColumnNames.TARGET.value})
            .reset_index(drop=True)
        )


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
        self._model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self._device)
        self._model.eval()
        self._tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=max_length)

    def analyze_model(self) -> dict:
        """
        Analyze model computing properties.

        Returns:
            dict: Properties of a model
        """
        if not isinstance(self._model, torch.nn.Module):
            raise TypeError("Expected self._model to be an instance of torch.nn.Module.")
        input_tensor = torch.ones(
            (1, self._model.config.encoder.max_position_embeddings), dtype=torch.long
        )
        inputs = {"input_ids": input_tensor, "attention_mask": input_tensor}
        summary_model = summary(
            self._model, input_data=inputs, decoder_input_ids=input_tensor, verbose=0
        )
        return {
            "input_shape": list(input_tensor.size()),
            "embedding_size": list(input_tensor.shape)[1],
            "output_shape": summary_model.summary_list[-1].output_size,
            "num_trainable_params": summary_model.trainable_params,
            "vocab_size": self._model.config.encoder.vocab_size,
            "size": summary_model.total_param_bytes,
            "max_context_length": self._model.config.max_length,
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
        loader = DataLoader(batch_size=self._batch_size, dataset=self._dataset)

        all_predictions = [
            prediction for batch in loader for prediction in self._infer_batch(batch)
        ]

        results_df = pd.DataFrame(self._dataset.data)
        results_df[ColumnNames.PREDICTION.value] = all_predictions
        return results_df

    @torch.no_grad()
    def _infer_batch(self, sample_batch: Sequence[tuple[str, ...]]) -> list[str]:
        """
        Infer model on a single batch.

        Args:
            sample_batch (Sequence[tuple[str, ...]]): Batch to infer the model

        Returns:
            list[str]: Model predictions as strings
        """
        inputs = self._tokenizer.prepare_seq2seq_batch(
            list(sample_batch[0]),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self._max_length,
        ).to(self._device)

        if not isinstance(self._model, torch.nn.Module):
            raise TypeError("Expected self._model to be an instance of torch.nn.Module.")

        output_ids = self._model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=self._max_length,
        )

        return [
            prediction.strip()
            for prediction in self._tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        ]


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
        results_df = pd.read_csv(self.data_path)
        texts = results_df[ColumnNames.PREDICTION.value]
        targets = results_df[ColumnNames.TARGET.value]
        evaluation = {}

        string_metrics = [format(metric) for metric in self._metrics]

        for metr in string_metrics:
            metric = load(metr, seed=77).compute(predictions=texts, references=targets)
            if metr == Metrics.ROUGE.value:
                evaluation[metr] = metric[Metrics.ROUGE.value + "L"]
            else:
                evaluation[metr] = metric[metr]

        return evaluation
