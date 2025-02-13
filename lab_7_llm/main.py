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
from pandas import DataFrame
from torch.utils.data import DataLoader, Dataset
from torchinfo import summary
from transformers import BertForSequenceClassification, BertTokenizerFast

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
        dataset = load_dataset(self._hf_name, split="train")
        self._raw_data = dataset.to_pandas()

        if not isinstance(self._raw_data, pd.DataFrame):
            raise TypeError("The downloaded dataset is not pd.DataFrame")



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
            "dataset_number_of_samples": len(self._raw_data),
            "dataset_columns": len(self._raw_data.columns),
            "dataset_duplicates": self._raw_data.duplicated().sum(),
            "dataset_empty_rows": self._raw_data.isnull().any(axis=1).sum(),
            "dataset_sample_min_len": self._raw_data["content"].dropna().map(len).min(),
            "dataset_sample_max_len": self._raw_data["content"].dropna().map(len).max()
        }

    @report_time
    def transform(self) -> None:
        """
        Apply preprocessing transformations to the raw dataset.
        """
        self._data = self._raw_data.copy()
        self._data.drop(["part", "movie_name", "review_id", "author", "date", "title", "grade10"], axis=1, inplace=True)
        self._data.rename(columns={"grade3": ColumnNames.TARGET.value, "content": ColumnNames.SOURCE.value}, inplace=True)
        self._data.dropna(inplace=True)
        #self._data.dropna(how="all", inplace=True)
        #labels for the model: 0: neutral, 1: positive, 2: negative
        self._data[ColumnNames.TARGET.value] = self._data[ColumnNames.TARGET.value].map({"Good": 1, "Bad": 2, "Neutral": 0})
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
        return (str(self._data.loc[index, ColumnNames.SOURCE.value]),)



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
        self._model = BertForSequenceClassification.from_pretrained(model_name)
        self._model.to(self._device)
        self._tokenizer = BertTokenizerFast.from_pretrained(model_name)

    def analyze_model(self) -> dict:
        """
        Analyze model computing properties.

        Returns:
            dict: Properties of a model
        """
        ids = torch.ones((1, self._model.config.max_position_embeddings), dtype=torch.long, device=self._device)
        result = summary(self._model, input_data={"input_ids": ids, "attention_mask": ids}, device="cpu",  verbose=0)
        return {
            "input_shape": {"input_ids": list(result.input_size["input_ids"]), "attention_mask": list(result.input_size["input_ids"])},
            "embedding_size": self._model.config.max_position_embeddings,
            "output_shape": result.summary_list[-1].output_size,
            "num_trainable_params": result.trainable_params,
            "vocab_size": self._model.config.vocab_size,
            "size": result.total_param_bytes,
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
            return
        """
        text = [i for i in sample]

        tokens = self._tokenizer(text, max_length=self._max_length, padding=True,
                                 truncation=True, return_tensors='pt').to(self._device)
        self._model.eval()

        with torch.no_grad():
            output = self._model(**tokens)

        predictions = torch.argmax(output.logits).item()
        return str(predictions)
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
        predictions = []

        for batch in dataloader:
            prediction = self._infer_batch(batch)
            predictions.extend(prediction)

        return pd.DataFrame({ColumnNames.TARGET.value: self._dataset[ColumnNames.TARGET],
                             ColumnNames.PREDICTION.value:predictions})


    @torch.no_grad()
    def _infer_batch(self, sample_batch: Sequence[tuple[str, ...]]) -> list[str]:
        """
        Infer model on a single batch.

        Args:
            sample_batch (Sequence[tuple[str, ...]]): Batch to infer the model

        Returns:
            list[str]: Model predictions as strings
        """
        """
        print("Batch:", sample_batch)
        inputs = self._tokenizer(
            list(sample_batch[0]),
            max_length=self._max_length,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )
        outputs = self._model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=1).tolist()
        print([str(i) for i in predictions])
        """
        inputs = self._tokenizer(
            list(sample_batch[0]),
            max_length=self._max_length,
            padding=True,
            truncation=True,
            return_tensors='pt'
        ).to(self._device)

        outputs = self._model(**inputs)
        predicted = torch.nn.functional.softmax(outputs.logits, dim=1)
        predicted = torch.argmax(predicted, dim=1).numpy()
        print(predicted)
        return [str(i) for i in predicted]


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

    @report_time
    def run(self) -> dict | None:
        """
        Evaluate the predictions against the references using the specified metric.

        Returns:
            dict | None: A dictionary containing information about the calculated metric
        """
