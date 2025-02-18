"""
Laboratory work.

Working with Large Language Models.
"""
# pylint: disable=too-few-public-methods, undefined-variable, too-many-arguments, super-init-not-called
from pathlib import Path
from typing import Iterable, Sequence

import evaluate
import pandas as pd
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from torchinfo import summary
from transformers import AutoTokenizer, BertForSequenceClassification

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
        dataset = load_dataset(self._hf_name, name='simplified', split='validation')
        self._raw_data = pd.DataFrame.from_dict(dataset)

        if not isinstance(self._raw_data, pd.DataFrame):
            raise TypeError('The downloaded dataset is not pd.DataFrame')


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
        raw_data = self._raw_data.copy()
        raw_data['labels'] = self._raw_data['labels'].apply(tuple)

        data_properties = {
            'dataset_number_of_samples': self._raw_data.shape[0],
            'dataset_columns': self._raw_data.shape[1],
            'dataset_duplicates': len(raw_data[raw_data.duplicated()]),
            'dataset_empty_rows': int(self._raw_data.isna().all(axis=1).sum()),
            'dataset_sample_min_len': int(
                self._raw_data
                .dropna()
                .drop_duplicates(subset='ru_text')['ru_text']
                .str.len().min()
            ),
            'dataset_sample_max_len': int(
                self._raw_data
                .dropna()
                .drop_duplicates(subset='ru_text')['ru_text']
                .str.len().max()
            )
        }

        return data_properties


    @report_time
    def transform(self) -> None:
        """
        Apply preprocessing transformations to the raw dataset.
        """
        unwanted_labels = [0, 4, 5, 6, 7, 8, 10, 12, 15, 18, 21, 22, 23]
        class_rules = {
            1: [1, 13, 17, 20],
            2: [9, 16, 24, 25],
            3: [14, 19],
            4: [2, 3],
            6: [26],
            7: [27]
        }

        self._data = self._raw_data.drop(['id', 'text'], axis=1)
        self._data['labels'] = self._data['labels'].apply(tuple)
        self._data = self._data.rename(
            columns={"labels": ColumnNames.TARGET.value, "ru_text": ColumnNames.SOURCE.value})
        self._data = (
            self._data[
                self._data['target'].apply(
                    lambda x: not any(label in unwanted_labels for label in x)
                )
            ]
        )
        self._data['target'] = self._data['target'].apply(
            lambda x: next(
                (class_num for class_num, emotions in class_rules.items() if x[0] in emotions),8)
        )
        self._data = self._data.dropna().drop_duplicates(subset='source')
        self._data = self._data.reset_index(drop=True)


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
        # return tuple(self._data.loc[index, ColumnNames.SOURCE.value])
        return (str(self._data[ColumnNames.SOURCE.value].iloc[index]),)

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

        super().__init__(model_name, dataset, max_length, batch_size, device)
        self._model = BertForSequenceClassification.from_pretrained(model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)

    def analyze_model(self) -> dict:
        """
        Analyze model computing properties.

        Returns:
            dict: Properties of a model
        """
        ids = torch.ones(1, self._model.config.max_position_embeddings, dtype=torch.long)
        tokens = {"input_ids": ids, "attention_mask": ids}
        result = summary(self._model, input_data=tokens, device="cpu", verbose=0)

        model_properties = {
            'input_shape': {
                'input_ids': list(result.input_size['input_ids']),
                'attention_mask': list(result.input_size['attention_mask'])
            },
            'embedding_size': self._model.config.max_position_embeddings,
            'output_shape': result.summary_list[-1].output_size,
            'vocab_size': self._model.config.vocab_size,
            'num_trainable_params': result.trainable_params,
            'size': result.total_param_bytes,
            'max_context_length': self._model.config.max_length
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
        dataset_loader = DataLoader(self._dataset, batch_size=self._batch_size)

        predictions = []
        for batch in dataset_loader:
            predictions.extend(self._infer_batch(batch))

        new_df = pd.DataFrame()
        new_df[ColumnNames.TARGET.value] = self._dataset.data[ColumnNames.TARGET.value]
        new_df[ColumnNames.PREDICTION.value] = predictions
        return new_df


    @torch.no_grad()
    def _infer_batch(self, sample_batch: Sequence[tuple[str, ...]]) -> list[str]:
        """
        Infer model on a single batch.

        Args:
            sample_batch (Sequence[tuple[str, ...]]): Batch to infer the model

        Returns:
            list[str]: Model predictions as strings
        """

        input_tokens = self._tokenizer(
            text=sample_batch[0],
            max_length= self._max_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        output = self._model(**input_tokens)
        predicted_class = torch.argmax(torch.softmax(output.logits, -1), dim=1).numpy()
        return [str(cl) for cl in predicted_class]


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
        # super().__init__(metrics)
        self._metrics = metrics
        self._data_path = data_path

    @report_time
    def run(self) -> dict | None:
        """
        Evaluate the predictions against the references using the specified metric.

        Returns:
            dict | None: A dictionary containing information about the calculated metric
        """
        predictions_df = pd.read_csv(self._data_path)
        for metric in self._metrics:
            f1_metric = evaluate.load(metric.value)
            predictions = predictions_df[ColumnNames.PREDICTION.value]
            references = predictions_df[ColumnNames.TARGET.value]
            value = f1_metric.compute(
                references=references,
                predictions=predictions,
                average='micro')
            return value
