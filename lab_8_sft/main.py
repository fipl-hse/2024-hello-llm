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
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset
from torchinfo import summary
from transformers import AutoTokenizer, MarianMTModel, Trainer, TrainingArguments

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

        self._raw_data = load_dataset(
            path=self._hf_name,
            split='test'
        ).to_pandas()

        if not isinstance(self._raw_data, pd.DataFrame):
            raise TypeError('The downloaded dataset is not pd.DataFrame')


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

        raw_data_no_nans = (
            self._raw_data
            .replace('', pd.NA)
            .dropna()
        )
        len_counts = raw_data_no_nans['en'].apply(len)

        return {
            'dataset_number_of_samples': len(self._raw_data),
            'dataset_columns': len(self._raw_data.columns),
            'dataset_duplicates': self._raw_data.duplicated().sum().item(),
            'dataset_empty_rows': len(self._raw_data) - len(raw_data_no_nans),
            'dataset_sample_min_len': len_counts.min().item(),
            'dataset_sample_max_len': len_counts.max().item()
        }

    @report_time
    def transform(self) -> None:
        """
        Apply preprocessing transformations to the raw dataset.
        """

        self._data = (
            self._raw_data
            .rename(columns={
                'en': str(ColumnNames.SOURCE),
                'fr': str(ColumnNames.TARGET)
            })
            .drop_duplicates()
            .replace('', pd.NA)
            .dropna()
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
        return (self._data[str(ColumnNames.SOURCE)][index],)

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
    tokens = tokenizer(text=sample[str(ColumnNames.SOURCE)],
                     text_target=sample[str(ColumnNames.TARGET)],
                     max_length=max_length,
                     padding='max_length',
                     truncation=True,
                     return_tensors='pt').data
    return {
                "input_ids": tokens["input_ids"][0],
                "attention_mask": tokens["attention_mask"][0],
                "labels": tokens["labels"][0],
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
        self._data = data.apply(
            lambda sample: tokenize_sample(sample, tokenizer, max_length),
            axis=1
        )
        self._data.reset_index(drop=True, inplace=True)

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
        item: dict = self._data.loc[index]
        return item


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

        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        self._model = MarianMTModel.from_pretrained(self._model_name)

        self._model: Module
        self._model.eval()
        self._model.to(self._device)

    def analyze_model(self) -> dict:
        """
        Analyze model computing properties.

        Returns:
            dict: Properties of a model
        """

        model_config = self._model.config

        ids = torch.ones(1, model_config.max_position_embeddings, dtype=torch.long)
        model_summary = summary(
            self._model,
            input_data={'input_ids': ids, 'decoder_input_ids': ids}
        )

        return {
            'input_shape': list(model_summary.input_size['input_ids']),
            'embedding_size': model_config.max_position_embeddings,
            'output_shape': model_summary.summary_list[-1].output_size,
            'num_trainable_params': model_summary.trainable_params,
            'vocab_size': model_config.vocab_size,
            'size': model_summary.total_param_bytes,
            'max_context_length': model_config.max_length
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

        data_loader = DataLoader(self._dataset,
                                 batch_size=self._batch_size)

        dataset_predictions = []
        for batch in data_loader:
            dataset_predictions.extend(self._infer_batch(batch))

        return pd.DataFrame(
            {
                str(ColumnNames.TARGET): self._dataset.data[str(ColumnNames.TARGET)],
                str(ColumnNames.PREDICTION): dataset_predictions
            }
        )

    @torch.no_grad()
    def _infer_batch(self, sample_batch: Sequence[tuple[str, ...]]) -> list[str]:
        """
        Infer single batch.

        Args:
            sample_batch (Sequence[tuple[str, ...]]): batch to infer the model

        Returns:
            list[str]: model predictions as strings
        """

        if not self._model:
            raise ValueError('Model is not available')

        tokens = self._tokenizer(*sample_batch,
                                 max_length=self._max_length,
                                 padding=True,
                                 truncation=True,
                                 return_tensors='pt').to(self._device)

        outputs = self._model.generate(**tokens,  max_length=self._max_length)

        decoded_seq: list = self._tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return decoded_seq


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
        self._metrics_computers = [load(str(metric)) for metric in self._metrics]

    def run(self) -> dict | None:
        """
        Evaluate the predictions against the references using the specified metric.

        Returns:
            dict | None: A dictionary containing information about the calculated metric
        """
        predictions_df = pd.read_csv(self._data_path)

        metrics_dict = {}
        for computer in self._metrics_computers:

            metrics = computer.compute(
                references=predictions_df[str(ColumnNames.TARGET)],
                predictions=predictions_df[str(ColumnNames.PREDICTION)],
            )

            metrics_dict[str(Metrics.BLEU)] = metrics[str(Metrics.BLEU)]

        return metrics_dict if metrics_dict else None


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

        pretrained_model = MarianMTModel.from_pretrained(self._model_name).to(sft_params.device)
        self._lora_config = LoraConfig(
            r=4,
            lora_alpha=8,
            lora_dropout=0.1,
            target_modules=sft_params.target_modules
        )
        self._model: Module = get_peft_model(
            pretrained_model,
            self._lora_config
        ).to(sft_params.device)

        self._batch_size: int = sft_params.batch_size
        self._max_length: int = sft_params.max_length
        self._max_sft_steps: int = sft_params.max_fine_tuning_steps
        self._device: str = sft_params.device
        self._finetuned_model_path: Path = sft_params.finetuned_model_path
        self._learning_rate: float = sft_params.learning_rate

    def run(self) -> None:
        """
        Fine-tune model.
        """

        training_args = TrainingArguments(
            output_dir=str(self._finetuned_model_path),
            max_steps=self._max_sft_steps,
            per_device_train_batch_size=self._batch_size,
            learning_rate=self._learning_rate,
            save_strategy="no",
            use_cpu=True,
            load_best_model_at_end=True,
            remove_unused_columns=False,
        )

        trainer = Trainer(
            model=self._model,
            args=training_args,
            train_dataset=self._dataset
        )
        trainer.train()

        trainer.model.merge_and_unload()
        trainer.model.base_model.save_pretrained(self._finetuned_model_path)

        AutoTokenizer.from_pretrained(self._model_name).save_pretrained(self._finetuned_model_path)
