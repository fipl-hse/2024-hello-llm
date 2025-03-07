"""
Laboratory work.

Fine-tuning Large Language Models for a downstream task.
"""
# pylint: disable=too-few-public-methods, undefined-variable, duplicate-code, unused-argument, too-many-arguments
from itertools import chain
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
import torch
from datasets import load_dataset
from evaluate import load
from pandas import DataFrame
from torch.utils.data import DataLoader, Dataset
from torch.nn.functional import softmax
from torchinfo import summary
from transformers import AutoTokenizer, BertForSequenceClassification, BertTokenizerFast

from config.lab_settings import SFTParams
from core_utils.llm.llm_pipeline import AbstractLLMPipeline
from core_utils.llm.metrics import Metrics
from core_utils.llm.raw_data_importer import AbstractRawDataImporter
from core_utils.llm.raw_data_preprocessor import AbstractRawDataPreprocessor
from core_utils.llm.task_evaluator import AbstractTaskEvaluator
from core_utils.llm.time_decorator import report_time
from core_utils.llm.sft_pipeline import AbstractSFTPipeline


class RawDataImporter(AbstractRawDataImporter):
    """
    Custom implementation of data importer.
    """

    @report_time
    def obtain(self) -> None:
        """
        Import dataset.
        """
        self._raw_data = load_dataset(
            self._hf_name, split="train", trust_remote_code=True
        ).to_pandas()


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
        temp_df = self._raw_data.copy()

        duplicates = temp_df[temp_df.duplicated()].shape[0]

        empty_rows = self._raw_data[self._raw_data.isnull().any(axis=1)].shape[0]

        no_na_df = self._raw_data.dropna()

        return {
            "dataset_columns": self._raw_data.shape[1],
            "dataset_duplicates": duplicates,
            "dataset_empty_rows": empty_rows,
            "dataset_number_of_samples": self._raw_data.shape[0],
            "dataset_sample_max_len": int(no_na_df["content"].str.len().max()),
            "dataset_sample_min_len": int(no_na_df["content"].str.len().min()),
        }

    @report_time
    def transform(self) -> None:
        """
        Apply preprocessing transformations to the raw dataset.
        """
        removed_columns = self._raw_data.drop(
            columns=["part", "movie_name", "review_id", "author", "date", "title", "grade10"]
        )
        renamed_columns = removed_columns.rename(columns={"grade3": "target", "content": "source"})

        filtered_df = renamed_columns[~renamed_columns.isnull().any(axis=1)]

        filtered_df.loc[filtered_df["target"] == "Good", "target"] = 1
        filtered_df.loc[filtered_df["target"] == "Bad", "target"] = 2
        filtered_df.loc[filtered_df["target"] == "Neutral", "target"] = 0

        self._data = filtered_df.reset_index(drop=True)


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
        return tuple([self._data.iloc[index]["source"]])

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
        self._model = BertForSequenceClassification.from_pretrained(self._model_name)
        self._model.to(self._device)
        self._tokenizer = BertTokenizerFast.from_pretrained(self._model_name)

    def analyze_model(self) -> dict:
        """
        Analyze model computing properties.

        Returns:
            dict: Properties of a model
        """
        if not isinstance(self._model, torch.nn.Module):
            raise TypeError("Expected self._model to be an instance of torch.nn.Module.")

        model_config = self._model.config

        if not isinstance(model_config.max_position_embeddings, int):
            raise TypeError("Expected model_config.max_position_embeddings to be int type")

        input_ids = torch.ones(
            [1, model_config.max_position_embeddings], dtype=torch.long, device=self._device
        )

        input_data = {"input_ids": input_ids, "attention_mask": input_ids}

        info = summary(model=self._model, input_data=input_data, verbose=0, device=self._device)
        input_size = list(info.input_size["input_ids"])

        return {
            "embedding_size": model_config.max_position_embeddings,
            "input_shape": {"attention_mask": input_size, "input_ids": input_size},
            "max_context_length": model_config.max_length,
            "num_trainable_params": info.trainable_params,
            "output_shape": info.summary_list[-1].output_size,
            "size": info.total_param_bytes,
            "vocab_size": model_config.vocab_size,
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
        dataloader = DataLoader(
            dataset=self._dataset, batch_size=self._batch_size, collate_fn=lambda x: x
        )
        data = {"target": self._dataset.data["target"].tolist(), "predictions": []}
        for batch in dataloader:
            pred = self._infer_batch(batch)
            data["predictions"].extend(pred)

        return pd.DataFrame(data=data)

    @torch.no_grad()
    def _infer_batch(self, sample_batch: Sequence[tuple[str, ...]]) -> list[str]:
        """
        Infer single batch.

        Args:
            sample_batch (Sequence[tuple[str, ...]]): batch to infer the model

        Returns:
            list[str]: model predictions as strings
        """
        inputs = self._tokenizer(
            [sample[0] for sample in sample_batch],
            max_length=self._max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self._device)

        with torch.no_grad():
            outputs = softmax(self._model(**inputs).logits, dim=1)

        predicted = torch.argmax(outputs, dim=1).tolist()
        return [str(class_pred) if class_pred != 0 else 2 for class_pred in predicted]


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

    def run(self) -> dict | None:
        """
        Evaluate the predictions against the references using the specified metric.

        Returns:
            dict | None: A dictionary containing information about the calculated metric
        """
        dataframe = pd.read_csv(self._data_path)
        res = {}
        for metric in self._metrics:
            evaluation = load(str(metric))

            preds = dataframe["predictions"].tolist()
            refs = dataframe["target"].tolist()
            results = evaluation.compute(references=refs, predictions=preds, average="micro")

            res.update(results)

        return res


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
