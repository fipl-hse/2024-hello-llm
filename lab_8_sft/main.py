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
from peft import get_peft_model, LoraConfig
from torch.utils.data import DataLoader, Dataset
from torchinfo import summary
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from config.lab_settings import SFTParams
from core_utils.llm.llm_pipeline import AbstractLLMPipeline
from core_utils.llm.metrics import Metrics
from core_utils.llm.raw_data_importer import AbstractRawDataImporter
from core_utils.llm.raw_data_preprocessor import AbstractRawDataPreprocessor, ColumnNames
from core_utils.llm.sft_pipeline import AbstractSFTPipeline
from core_utils.llm.task_evaluator import AbstractTaskEvaluator
from core_utils.llm.time_decorator import report_time


class RawDataImporter(AbstractRawDataImporter):
    """
    Custom implementation of data importer.
    """

    @report_time
    def obtain(self) -> None:
        """
        Import dataset.
        """
        self._raw_data = load_dataset(path=self._hf_name, split='train').to_pandas()
        if not isinstance(self._raw_data, pd.DataFrame):
            raise TypeError(f'Expected pd.DataFrame, got type {self._raw_data} instead.')


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
        return {
            'dataset_number_of_samples': len(self._raw_data),
            'dataset_columns': len(self._raw_data.columns),
            'dataset_duplicates': self._raw_data.duplicated().sum(),
            "dataset_empty_rows": self._raw_data.isnull().any(axis=1).sum(),
            'dataset_sample_min_len': min(
                len(sample) for sample in self._raw_data["toxic_comment"]
            ),
            'dataset_sample_max_len': max(
                len(sample) for sample in self._raw_data["toxic_comment"]
            )
        }

    @report_time
    def transform(self) -> None:
        """
        Apply preprocessing transformations to the raw dataset.
        """
        self._data = self._raw_data.copy().rename(
            columns={
                "reasons": ColumnNames.TARGET.value,
                "toxic_comment": ColumnNames.SOURCE.value,
            }
        )

        self._data = self._data.loc[
            self._data[ColumnNames.TARGET.value].isin(
                ['{"toxic_content":true}', '{"not_toxic":true}']
            )
        ].replace(
            {
                ColumnNames.TARGET.value: {
                    '{"toxic_content":true}': 1,
                    '{"not_toxic":true}': 0,
                }
            }
        ).reset_index(drop=True)

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
    tokenized = tokenizer(
        sample[ColumnNames.SOURCE.value],
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

    return {
        "input_ids": tokenized["input_ids"].squeeze(0),
        "attention_mask": tokenized["attention_mask"].squeeze(0),
        "labels": sample[ColumnNames.TARGET.value]
    }

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
        self._tokenized_data = [
            tokenize_sample(row, tokenizer, max_length) for _, row in data.iterrows()
        ]

    def __len__(self) -> int:
        """
        Return the number of items in the dataset.

        Returns:
            int: The number of items in the dataset
        """
        return len(self._tokenized_data)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """
        Retrieve an item from the dataset by index.

        Args:
            index (int): Index of sample in dataset

        Returns:
            dict[str, torch.Tensor]: An element from the dataset
        """
        return self._tokenized_data[index]


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
        self._model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model.eval()

    def analyze_model(self) -> dict:
        """
        Analyze model computing properties.

        Returns:
            dict: Properties of a model
        """
        if not isinstance(self._model, torch.nn.Module):
            raise TypeError("Model is of incorrect type")

        tensor = torch.ones((1,
                             self._model.config.max_position_embeddings),
                             dtype=torch.long
        )

        input_data = {'input_ids': tensor, 'attention_mask': tensor}

        model_summary = summary(self._model, input_data=input_data)

        return {
            'input_shape': {
                'input_ids': list(tensor.size()),
                'attention_mask': list(tensor.size())
            },
            'embedding_size': self._model.config.max_position_embeddings,
            'output_shape': model_summary.summary_list[-1].output_size,
            'num_trainable_params': model_summary.trainable_params,
            'vocab_size': self._model.config.vocab_size,
            'size': model_summary.total_param_bytes,
            'max_context_length': self._model.config.max_length
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
        data_loader = DataLoader(dataset=self._dataset, batch_size=self._batch_size)
        predictions = []

        for batch in data_loader:
            batch_predictions = self._infer_batch(batch)
            predictions.extend(batch_predictions)

        result_df = pd.DataFrame(self._dataset.data)
        result_df[ColumnNames.PREDICTION.value] = predictions

        return result_df

    @torch.no_grad()
    def _infer_batch(self, sample_batch: Sequence[tuple[str, ...]]) -> list[str]:
        """
        Infer single batch.

        Args:
            sample_batch (Sequence[tuple[str, ...]]): batch to infer the model

        Returns:
            list[str]: model predictions as strings
        """
        if not isinstance(self._model, torch.nn.Module):
            raise TypeError('Model must be a torch.nn.Module')

        inputs = self._tokenizer(
            list(sample_batch[0]),
            return_tensors="pt",
            truncation=True,
            padding=True,
        ).to(self._device)

        decoded_predictions = self._model(**inputs).logits
        return [str(prediction.argmax().item()) for prediction in decoded_predictions]


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
        self._metrics = [load(metric.value) for metric in metrics]
        self._data_path = data_path

    def run(self) -> dict | None:
        """
        Evaluate the predictions against the references using the specified metric.

        Returns:
            dict | None: A dictionary containing information about the calculated metric
        """
        data = pd.read_csv(self._data_path)
        targets = data[ColumnNames.TARGET.value]
        predictions = data[ColumnNames.PREDICTION.value]

        results = {}
        for metric in self._metrics:
            score = metric.compute(predictions=predictions, references=targets)
            metric_key = next(iter(score))
            results[metric_key] = score[metric_key]

        return results

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
        super().__init__(model_name, dataset)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self._model_name
        )

        self._lora_config = LoraConfig(
            r=4,
            lora_alpha=8,
            lora_dropout=0.1,
            target_modules = sft_params.target_modules
        )

        self._batch_size = sft_params.batch_size
        self._device = sft_params.device
        self._model = get_peft_model(self._model, self._lora_config).to(self._device)
        self._max_sft_steps = sft_params.max_fine_tuning_steps
        self._max_length = sft_params.max_length
        self._learning_rate = sft_params.learning_rate
        self._finetuned_model_path = sft_params.finetuned_model_path

    def run(self) -> None:
        """
        Fine-tune model.
        """
        if (self._max_sft_steps is None
                or self._batch_size is None
                or self._learning_rate is None
                or self._finetuned_model_path is None):
            return

        if not isinstance(self._model, torch.nn.Module):
            raise TypeError('Model must be a torch.nn.Module')

        training_args = TrainingArguments(
            max_steps=self._max_sft_steps,
            per_device_train_batch_size=self._batch_size,
            learning_rate=self._learning_rate,
            save_strategy='no',
            use_cpu = True,
            load_best_model_at_end=False,
            output_dir=self._finetuned_model_path,
        )

        trainer = Trainer(
            model=self._model,
            args=training_args,
            train_dataset=self._dataset
        )

        trainer.train()

        final_model = self._model.merge_and_unload()
        final_model.save_pretrained(self._finetuned_model_path)
