"""
Laboratory work.

Working with Large Language Models.
"""
# pylint: disable=too-few-public-methods, undefined-variable, too-many-arguments, super-init-not-called,
# pylint: disable=too-many-arguments
from pathlib import Path
from typing import Iterable, Sequence, Union

import pandas as pd
import torch
from datasets import load_dataset
from evaluate import load
from pandas import DataFrame
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from core_utils.llm.llm_pipeline import AbstractLLMPipeline
from core_utils.llm.metrics import Metrics
from core_utils.llm.raw_data_importer import AbstractRawDataImporter
from core_utils.llm.raw_data_preprocessor import AbstractRawDataPreprocessor
from core_utils.llm.task_evaluator import AbstractTaskEvaluator
from core_utils.llm.time_decorator import report_time


class RawDataImporter(AbstractRawDataImporter):
    """
    A class that imports the HuggingFace dataset.
    """

    def __init__(self) -> None:
        """Initialize an instance of RawDataImporter."""
        super().__init__("trixdade/reviews_russian")
        self._data: pd.DataFrame | None = None

    @report_time
    def obtain(self) -> None:
        """
        Download a dataset.

        Raises:
            TypeError: In case of downloaded dataset is not pd.DataFrame
        """
        dataset = load_dataset("trixdade/reviews_russian", split="train")
        self._data = dataset.to_pandas()
        if not isinstance(self._data, pd.DataFrame) or self._data.empty:
            raise ValueError("Dataset could not be loaded or is empty.")

    @property
    def data(self):
        """
        Returns the dataset stored in `_data`.

        Returns:
            pd.DataFrame: The dataset as a pandas DataFrame.
        """
        return self._data


class RawDataPreprocessor(AbstractRawDataPreprocessor):
    """
    A class that analyzes and preprocesses a dataset.
    """

    def __init__(self, raw_data: DataFrame) -> None:
        if raw_data is None:
            raise ValueError("Raw data cannot be None")
        self._raw_data = raw_data
        self._data = None

    @property
    def data(self) -> pd.DataFrame:
        """
        Property to access the preprocessed dataset.
        """
        if self._data is None:
            raise ValueError("Data has not been processed yet. Call transform() first.")
        return self._data

    def analyze(self) -> dict:
        """
        Analyze a dataset.

        Returns:
            dict: Dataset key properties
        """
        cleaned_dataset = self._data.dropna()
        return {
            "dataset_number_of_samples": len(self._data) if self._data is not None else 0,
            "dataset_columns": len(self._data.columns) if self._data is not None else 0,
            "dataset_duplicates": self._data.duplicated().sum(),
            "dataset_empty_rows": self._data.isnull().sum().sum(),
            "dataset_sample_min_len": cleaned_dataset["source"].str.len().min()
            if "source" in cleaned_dataset else None,
            "dataset_sample_max_len": cleaned_dataset["source"].str.len().max()
            if "source" in cleaned_dataset else None,
        }

    @report_time
    def transform(self) -> None:
        """
        Apply preprocessing transformations to the raw dataset.

        Returns:
            pd.DataFrame: Preprocessed dataset
        """
        self._raw_data.rename(columns={"Reviews": "source", "Summary": "target"}, inplace=True)

        if "source" not in self._raw_data.columns or "target" not in self._raw_data.columns:
            raise KeyError("Dataset does not contain 'source' and 'target' columns.")

        self._data = self._raw_data.dropna().reset_index(drop=True)

        print("Transformed Data Sample:")
        print(self._data.head())


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
            self,
            model_name: str,
            dataset: TaskDataset,
            max_length: int,
            batch_size: int,
            device: str
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
        self._tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
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
        config = self._model.config
        return {
            "model_parameters": sum(p.numel() for p in self._model.parameters()),
            "trainable_parameters": sum(p.numel() for p in self._model.parameters()
                                        if p.requires_grad),
            "device": self.device,
            "num_layers": getattr(config, "num_hidden_layers", "Unknown"),
            "vocab_size": getattr(config, "vocab_size", "Unknown"),
            "hidden_size": getattr(config, "hidden_size", "Unknown"),
            "attention_heads": getattr(config, "num_attention_heads", "Unknown"),
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
        inputs = self._tokenizer(
            input_text, return_tensors="pt",
            truncation=True,
            max_length=self.max_length
        ).to(self.device)
        tokens = self._tokenizer.convert_ids_to_tokens(inputs['input_ids'][0].tolist())
        print(f"Tokenized input: {tokens}")

        with torch.no_grad():
            output = self._model.generate(**inputs, max_length=self.max_length)

        prediction = str(self._tokenizer.decode(output[0], skip_special_tokens=True))
        print(f"Input: {input_text}, Prediction: {prediction}")
        return prediction if prediction else None

    @report_time
    def infer_dataset(self) -> pd.DataFrame:
        """
        Infer model on a whole dataset.

        Returns:
            pd.DataFrame: Data with predictions
        """
        dataloader = DataLoader(self.dataset,
                                batch_size=self.batch_size,
                                shuffle=False
                                )
        predictions = []

        for batch in dataloader:
            input_texts = [sample[0] for sample in batch]
            inputs = self._tokenizer(
                input_texts, return_tensors="pt",
                padding=True, truncation=True,
                max_length=self.max_length
            ).to(self.device)
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_length=self.max_length,
                    num_beams=5,
                    repetition_penalty=1.2,
                    early_stopping=True,
                    forced_bos_token_id=self._tokenizer.bos_token_id
                )
            batch_predictions = self._tokenizer.batch_decode(outputs, skip_special_tokens=True)
            predictions.extend(batch_predictions)

            print(f"Tokens: {outputs[0]}")
            print(f"Decoded: {batch_predictions[0]}")

        self.dataset.data.loc[:, "predicted_summary"] = pd.Series(predictions)
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
        input_texts = [sample[0] for sample in sample_batch]
        inputs = self._tokenizer(
            input_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        ).to(self.device)
        outputs = self._model.generate(**inputs, max_length=self.max_length)
        return list(self._tokenizer.batch_decode(outputs, skip_special_tokens=True))


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
    def run(self) -> Union[dict, None]:
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
            if metric_name == "rouge":
                metric = load("rouge")
                metric_result = metric.compute(
                    predictions=data['predicted_summary'].tolist(),
                    references=data['target'].tolist(),
                    rouge_types=["rougeL"],
                    use_aggregator=True
                )
                results["rougeL"] = metric_result["rougeL"]

            elif metric_name == "bleu":
                metric = load("bleu")
                predictions = [pred.split() for pred in data["predicted_summary"].tolist()]
                references = [[ref.split()] for ref in data["target"].tolist()]
                metric_result = metric.compute(predictions=predictions, references=references)
                results["bleu"] = metric_result["bleu"]

        print("Метрики:", results)
        return results
