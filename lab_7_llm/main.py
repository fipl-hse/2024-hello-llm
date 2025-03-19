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
from evaluate import load
from pandas import DataFrame
from torch.utils.data import DataLoader, Dataset
from torchinfo import summary
from transformers import AutoModelForSequenceClassification, AutoTokenizer

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
        data = datasets.load_dataset(self._hf_name, split='train')
        self._raw_data = pd.DataFrame(data)

        if not isinstance(self._raw_data, pd.DataFrame):
            raise TypeError('Downloaded dataset is not a pandas DataFrame.')


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

        properties = {
            'dataset_number_of_samples': len(self._raw_data),
            'dataset_columns': len(self._raw_data.columns),
            'dataset_duplicates': int(self._raw_data.duplicated().sum()),
            'dataset_empty_rows': int(self._raw_data.isnull().all(axis=1).sum()),
            'dataset_sample_min_len': int(self._raw_data['comment'].str.len().min()),
            'dataset_sample_max_len': int(self._raw_data['comment'].str.len().max()),
        }
        return properties

    @report_time
    def transform(self) -> None:
        """
        Apply preprocessing transformations to the raw dataset.
        """
        self._data = self._raw_data.drop_duplicates().rename(
            columns={'comment': ColumnNames.SOURCE.value,
                     'toxic': ColumnNames.TARGET.value}).reset_index(drop=True)

        self._data[ColumnNames.TARGET.value] = self._data[ColumnNames.TARGET.value].replace(
            {'true': 1, 'false': 0}).astype(int)


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
        item = str(self._data.loc[index, ColumnNames.SOURCE.value])
        return tuple([item])

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
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_name)

    def analyze_model(self) -> dict:
        """
        Analyze model computing properties.

        Returns:
            dict: Properties of a model
        """
        if not isinstance(self._model, torch.nn.Module):
            raise TypeError('The model is not a Module model')

        embeddings_length = self._model.config.max_position_embeddings
        dummy_input = torch.ones(1, embeddings_length, dtype=torch.long)
        input_data = {"input_ids": dummy_input, "attention_mask": dummy_input}
        model_summary = summary(model=self._model,
                                input_data=input_data,
                                device=self._device,
                                verbose=False)

        model_properties = {
            "input_shape": {'attention_mask': list(input_data['attention_mask'].shape),
                            'input_ids': list(input_data['input_ids'].shape)},
            "embedding_size": self._model.config.max_position_embeddings,
            "output_shape": model_summary.summary_list[-1].output_size,
            "num_trainable_params": model_summary.trainable_params,
            "vocab_size": self._model.config.vocab_size,
            "size": model_summary.total_param_bytes,
            "max_context_length": self._model.config.max_length
        }

        return model_properties

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
        data_load = DataLoader(self._dataset, batch_size=self._batch_size)
        predictions = []
        for batch in data_load:
            predictions.extend(self._infer_batch(batch))

        data_with_predictions = pd.DataFrame(
            {'target': self._dataset.data[ColumnNames.TARGET.value],
             'prediction': pd.Series(predictions)})
        return data_with_predictions

    @torch.no_grad()
    def _infer_batch(self, sample_batch: Sequence[tuple[str, ...]]) -> list[str]:
        """
        Infer model on a single batch.

        Args:
            sample_batch (Sequence[tuple[str, ...]]): Batch to infer the model

        Returns:
            list[str]: Model predictions as strings
        """
        if self._model is None:
            raise ValueError("Model is not initialized.")
        
        inputs = self._tokenizer(sample_batch[0],
                                 padding=True,
                                 truncation=True,
                                 max_length=self._max_length,
                                 return_tensors="pt").to(self._device)

        outputs = self._model(**inputs).logits
        predictions = [str(prediction.argmax().item()) for prediction
                       in outputs]
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
        super().__init__(metrics)
        self._data_path = data_path

    @report_time
    def run(self) -> dict | None:
        """
        Evaluate the predictions against the references using the specified metric.

        Returns:
            dict | None: A dictionary containing information about the calculated metric
        """
        predictions = pd.read_csv(self._data_path)
        evaluations = {}
        for metric in self._metrics:
            metric = load(metric.value)
            evaluation = metric.compute(references=predictions['target'].tolist(),
                                        predictions=predictions['prediction'].tolist(),
                                        average='micro')
            evaluations.update(dict(evaluation))

        return evaluations
