"""
Laboratory work.

Working with Large Language Models.
"""
# pylint: disable=too-few-public-methods, undefined-variable, too-many-arguments, super-init-not-called
import torch
from datasets import load_dataset
from pandas import DataFrame
from typing import Sequence, Iterable
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from pathlib import Path

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from evaluate import load as load_metric

from core_utils.llm.llm_pipeline import AbstractLLMPipeline
from core_utils.llm.metrics import Metrics
from core_utils.llm.raw_data_importer import AbstractRawDataImporter
from core_utils.llm.raw_data_preprocessor import AbstractRawDataPreprocessor
from core_utils.llm.task_evaluator import AbstractTaskEvaluator
from core_utils.llm.time_decorator import report_time
import pandas as pd


class RawDataImporter(AbstractRawDataImporter):
    """
    A class that imports the HuggingFace dataset.
    """

    def __init__(self, hf_name: str) -> None:
        """
        Initialize the data importer.

        Args:
            hf_name (str): The name of the dataset on Hugging Face.
        """
        super().__init__(hf_name)
        self.hf_name = hf_name

    @report_time
    def obtain(self) -> None:
        """
        Download a dataset.

        Raises:
            TypeError: In case of downloaded dataset is not pd.DataFrame
        """
        dataset = load_dataset(self.hf_name, split="train")
        df = dataset.to_pandas()

        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected a pandas DataFrame, but got {type(df)}")

        return df



class RawDataPreprocessor(AbstractRawDataPreprocessor):
    """
    A class that analyzes and preprocesses a dataset.
    """
    def __init__(self, raw_data: pd.DataFrame) -> None:
        if raw_data is None:
            raise ValueError("Raw data cannot be None")
        self._raw_data = raw_data
        self._data = raw_data.copy()

    @property
    def data(self) -> pd.DataFrame:
        """
        Property to access the preprocessed dataset.
        """
        return self._data

    def analyze(self) -> dict:
        """
        Analyze a dataset.

        Returns:
            dict: Dataset key properties
        """
        cleaned_data = self._data.dropna()

        return {
            "dataset_number_of_samples": len(self._data),
            "dataset_columns": len(self._data.columns),
            "dataset_duplicates": self._data.duplicated().sum(),
            "dataset_empty_rows": self._data.isnull().sum().sum(),
            "dataset_sample_min_len": cleaned_data["source"].str.len().min() if "source" in cleaned_data else None,
            "dataset_sample_max_len": cleaned_data["source"].str.len().max() if "source" in cleaned_data else None,
        }

    @report_time
    def transform(self) -> None:
        """
        Apply preprocessing transformations to the raw dataset.

        Returns:
            pd.DataFrame: Preprocessed dataset
        """
        if "Reviews" not in self._data.columns or "Summary" not in self._data.columns:
            raise KeyError("Dataset does not contain 'Reviews' and 'Summary' columns. Check dataset structure.")

        self._data.dropna(inplace=True)
        self._data.rename(columns={"Reviews": "source", "Summary": "target"}, inplace=True)
        self._data.reset_index(drop=True, inplace=True)


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
        return row["source"], row["target"]

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
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
        self.dataset = dataset
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = device

    def analyze_model(self) -> dict:
        """
        Analyze model computing properties.

        Returns:
            dict: Properties of a model
        """
        return {
            "model_parameters": sum(p.numel() for p in self.model.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            "device": self.device
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
        input_text = sample[0]
        inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True, max_length=self.max_length).to(
            self.device)
        with torch.no_grad():
            output = self.model.generate(**inputs, max_length=self.max_length)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

    @report_time
    def infer_dataset(self) -> pd.DataFrame:
        """
        Infer model on a whole dataset.

        Returns:
            pd.DataFrame: Data with predictions
        """
        dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)
        predictions = []

        total_batches = 0

        for batch in dataloader:
            total_batches += 1
            batch_predictions = self._infer_batch(batch)
            predictions.extend(batch_predictions)

        print(f"Total batches processed: {total_batches}")
        print(f"Total predictions: {len(predictions)}")
        print(f"Expected dataset size: {len(self.dataset.data)}")

        if len(predictions) != len(self.dataset.data):
            raise ValueError(
                f"Number of predictions ({len(predictions)}) does not match number of samples in dataset ({len(self.dataset.data)}).")

        self.dataset.data['predicted_summary'] = pd.Series(predictions)

        return self.dataset.data


    @torch.no_grad()
    def _infer_batch(self, sample_batch: Sequence[tuple[str, ...]]) -> list[str]:
        """
        Infer model on a single batch.

        Args:
            sample_batch (Sequence[tuple[str, ...]]): Batch to infer the model

        Returns:
            list[str]: Model predictions as strings
        """
        input_texts, _ = sample_batch

        inputs = self.tokenizer(
            list(input_texts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        ).to(self.device)

        outputs = self.model.generate(**inputs, max_length=self.max_length)
        predictions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return predictions


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
        self.metrics = metrics

    @report_time
    def run(self) -> dict | None:
        """
        Evaluate the predictions against the references using the specified metric.

        Returns:
            dict | None: A dictionary containing information about the calculated metric
        """
        data = pd.read_csv(self.data_path)

        if 'predicted_summary' not in data.columns or 'target' not in data.columns:
            raise ValueError("CSV file must contain 'predicted_summary' and 'target' columns.")

        results = {}

        for metric_name in self.metrics:
            if metric_name == "rougeL":
                metric = load_metric("rouge")
                metric_result = metric.compute(
                    predictions=data['predicted_summary'].tolist(),
                    references=data['target'].tolist(),
                    rouge_types=["rougeL"],
                    use_aggregator=True
                )
                results["rougeL"] = metric_result["rougeL"]

            elif metric_name == "bleu":
                metric = load_metric("bleu")
                predictions = [pred.split() for pred in data["predicted_summary"].tolist()]
                references = [[ref.split()] for ref in data["target"].tolist()]
                metric_result = metric.compute(predictions=predictions, references=references)
                results["bleu"] = metric_result["bleu"]

        return results
