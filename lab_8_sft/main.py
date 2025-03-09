"""
Laboratory work.

Fine-tuning Large Language Models for a downstream task.
"""
# pylint: disable=too-few-public-methods, undefined-variable, duplicate-code, unused-argument, too-many-arguments
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
import torch
from datasets import load_dataset
from evaluate import load
from pandas import DataFrame
from torch.utils.data import DataLoader, Dataset
from torchinfo import summary
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config.lab_settings import SFTParams
from core_utils.llm.llm_pipeline import AbstractLLMPipeline
from core_utils.llm.metrics import Metrics
from core_utils.llm.raw_data_importer import AbstractRawDataImporter
from core_utils.llm.raw_data_preprocessor import AbstractRawDataPreprocessor, ColumnNames
from core_utils.llm.sft_pipeline import AbstractSFTPipeline
from core_utils.llm.task_evaluator import AbstractTaskEvaluator
from core_utils.llm.time_decorator import report_time
from peft import get_peft_model, LoraConfig


class RawDataImporter(AbstractRawDataImporter):
    """
    Custom implementation of data importer.
    """

    @report_time
    def obtain(self) -> None:
        """
        Import dataset.
        """
        sentiment_dataset = load_dataset(self._hf_name, split='test')
        if sentiment_dataset:
            self._raw_data = sentiment_dataset.to_pandas()
        if not isinstance(self._raw_data, pd.DataFrame):
            raise TypeError('Error. Downloaded dataset is not pd.DataFrame.')


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
        return {'dataset_number_of_samples': len(self._raw_data),
                'dataset_columns': self._raw_data.columns.size,
                'dataset_duplicates': self._raw_data.duplicated().sum(),
                'dataset_empty_rows': self._raw_data.isna().sum().sum(),
                'dataset_sample_min_len': min(self._raw_data.dropna()['text'].apply(len)),
                'dataset_sample_max_len': max(self._raw_data.dropna()['text'].apply(len))}

    @report_time
    def transform(self) -> None:
        """
        Apply preprocessing transformations to the raw dataset.
        """
        self._data = self._raw_data.rename(
            columns={
                'text': str(ColumnNames.SOURCE),
                'label': str(ColumnNames.TARGET)
            }
        ).drop_duplicates().reset_index(drop=True)


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
        return tuple([self._data.iloc[index][ColumnNames.SOURCE.value]])

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
    tokenized_sample = tokenizer(sample[ColumnNames.SOURCE.value],
                                 padding="max_length",
                                 truncation=True,
                                 max_length=max_length)
    tokenized_sample["labels"] = sample[ColumnNames.TARGET.value]
    return tokenized_sample.data


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
        self._data = data.apply(
            lambda x: tokenize_sample(sample=x,
                                      tokenizer=tokenizer,
                                      max_length=max_length)
        )

    def __len__(self) -> int:
        """
        Return the number of items in the dataset.

        Returns:
            int: The number of items in the dataset
        """
        return len(self._data)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """
        Retrieve an item from the dataset by index.

        Args:
            index (int): Index of sample in dataset

        Returns:
            dict[str, torch.Tensor]: An element from the dataset
        """
        return self._data[index]


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
        self._model = AutoModelForSequenceClassification.from_pretrained(self._model_name)
        self._model.eval()
        self._model.to(device)
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name,
                                                        model_max_length=max_length,
                                                        padding_side='left')
        self._tokenizer.pad_token = self._tokenizer.eos_token

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
        if prediction:
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
        targets = self._dataset.data[str(ColumnNames.TARGET)].values
        predictions = []

        for batch in dataset_loader:
            predictions.extend(self._infer_batch(batch))

        data_predictions = pd.DataFrame({ColumnNames.TARGET.value: targets,
                                         ColumnNames.PREDICTION.value: predictions})
        return data_predictions

    @torch.no_grad()
    def _infer_batch(self, sample_batch: Sequence[tuple[str, ...]]) -> list[str]:
        """
        Infer single batch.

        Args:
            sample_batch (Sequence[tuple[str, ...]]): batch to infer the model

        Returns:
            list[str]: model predictions as strings
        """
        input_ids = self._tokenizer(*sample_batch,
                                    return_tensors="pt",
                                    max_length=self._max_length,
                                    padding=True,
                                    truncation=True).to(self._device)

        outputs = self._model(**input_ids)
        decoded_batch = torch.argmax(outputs.logits, dim=1).tolist()
        return [str(prediction) for prediction in decoded_batch]


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
        self._metrics = [load(str(metric)) for metric in self._metrics]
        self._data_path = data_path

    def run(self) -> dict | None:
        """
        Evaluate the predictions against the references using the specified metric.

        Returns:
            dict | None: A dictionary containing information about the calculated metric
        """
        data = pd.read_csv(self._data_path)
        calculated_metrics = {}

        predictions = data[ColumnNames.PREDICTION.value].to_list()
        references = data[ColumnNames.TARGET.value].to_list()

        for metric in self._metrics:
            calculated_metrics.update(metric.compute(predictions=predictions,
                                                     references=references,
                                                     average='micro'))

        return calculated_metrics


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
        super.__init__(model_name, dataset)
        self._model = AutoModelForSequenceClassification.from_pretrained(self._model_name)
        self._lora_config = LoraConfig(target_modules=sft_params,
                                       r=4,
                                       lora_alpha=8,
                                       lora_dropout=0.)
    def run(self) -> None:
        """
        Fine-tune model.
        """
