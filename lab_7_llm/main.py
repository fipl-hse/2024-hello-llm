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
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

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

    @report_time
    def obtain(self) -> None:
        """
        Download a dataset.

        Raises:
            TypeError: In case of downloaded dataset is not pd.DataFrame
        """
        subset = None
        split = "test"
        if subset:
            dataset = load_dataset(self._hf_name, subset=subset, split=split)
        else:
            dataset = load_dataset(self._hf_name, split=split)

        self._raw_data = pd.DataFrame(dataset)

        if not isinstance(self._raw_data, pd.DataFrame):
            raise TypeError("Downloaded dataset is not a pd.DataFrame.")


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
        dataset_info = {
            "dataset_number_of_samples": self._raw_data.shape[0],
            "dataset_columns": self._raw_data.shape[1],
            "dataset_duplicates": self._raw_data.duplicated().sum(),
            "dataset_empty_rows": self._raw_data.isnull().all(axis=1).sum(),
            "dataset_sample_min_len": self._raw_data["info"].map(len).min(),
            "dataset_sample_max_len": self._raw_data["info"].map(len).max()
        }
        return dataset_info


    @report_time
    def transform(self) -> None:
        """
        Apply preprocessing transformations to the raw dataset.
        """
        self._data = self._raw_data.rename(columns={'info': 'source', 'summary': 'target'})
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
        item = self._data.iloc[index]
        return tuple(item)

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
        self.model_name = model_name
        self.dataset = dataset
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = device
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name,
                                                       model_max_length=max_length)


    def analyze_model(self) -> dict:
        """
        Analyze model computing properties.

        Returns:
            dict: Properties of a model
        """
        input_ids = torch.ones((1, 768), dtype=torch.long)
        input_data = {"input_ids": input_ids, "decoder_input_ids": input_ids}
        model_summary = summary(self.model, input_data=input_data, verbose=0)

        return {
            "input_shape": list(input_ids.size()),
            "embedding_size": list(self.model.named_parameters())[1][1].shape[0],
            "output_shape": model_summary.summary_list[-1].output_size,
            "num_trainable_params": model_summary.trainable_params,
            "vocab_size": self.model.config.vocab_size,
            "size": model_summary.total_param_bytes,
            "max_context_length": self.model.config.max_length
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
        return self._infer_batch(sample)[0]


    @report_time
    def infer_dataset(self) -> pd.DataFrame:
        """
        Infer model on a whole dataset.

        Returns:
            pd.DataFrame: Data with predictions
        """
        def collate_fn(batch_sample):
            items, targets = [], []
            for sample in batch_sample:
                items.append(sample[0])
                targets.append(sample[1])

            return {"items": items, "targets": targets}

        dl = DataLoader(batch_size=self.batch_size,
                        dataset=self.dataset,
                        collate_fn=collate_fn)

        targets, predictions = [], []
        for sample_batch in dl:
            targets.extend(sample_batch["targets"])
            sample_predictions = self._infer_batch(sample_batch["items"])
            predictions.append(sample_predictions)

        return pd.DataFrame({"target": targets, "predictions": predictions})


    @torch.no_grad()
    def _infer_batch(self, sample_batch: Sequence[tuple[str, ...]]) -> list[str]:
        """
        Infer model on a single batch.

        Args:
            sample_batch (Sequence[tuple[str, ...]]): Batch to infer the model

        Returns:
            list[str]: Model predictions as strings
        """
        pipe = pipeline("text2text-generation",
                        model=self.model,
                        tokenizer=self.tokenizer,
                        truncation=True)

        res = pipe(list(sample_batch))
        return [r["generated_text"] for r in res]


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
        df = pd.read_csv(self.data_path)
        predictions, references = df.predictions, df.target
        evaluation_res = {}
        for metric_name in self._metrics:
            metric = load(metric_name.value, seed=77)
            scores = metric.compute(predictions=predictions, references=references)
            if metric_name.value == "rouge":
                evaluation_res[metric_name.value] = scores["rougeL"]
            else:
                evaluation_res[metric_name.value] = scores[metric_name.value]
        return evaluation_res
