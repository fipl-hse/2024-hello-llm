"""
Laboratory work.

Fine-tuning Large Language Models for a downstream task.
"""
# pylint: disable=too-few-public-methods, undefined-variable, duplicate-code, unused-argument, too-many-arguments
from pathlib import Path
from typing import Iterable, Sequence

import evaluate
import pandas as pd
import torch
from datasets import load_dataset
from peft import get_peft_model, LoraConfig
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset
from torchinfo import summary
from transformers import AutoTokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments

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

        dataset = load_dataset(self._hf_name, split='test')
        self._raw_data = dataset.to_pandas()

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

        data_properties = {
            'dataset_number_of_samples': self._raw_data.shape[0],
            'dataset_columns': self._raw_data.shape[1],
            'dataset_duplicates': len(self._raw_data[self._raw_data.duplicated()]),
            'dataset_empty_rows': int(self._raw_data.isna().all(axis=1).sum()),
            'dataset_sample_min_len': int(
                self._raw_data
                .dropna()
                .drop_duplicates(subset='EN')['EN']
                .str.len().min()
            ),
            'dataset_sample_max_len': int(
                self._raw_data
                .dropna()
                .drop_duplicates(subset='EN')['EN']
                .str.len().max()
            )
        }

        return data_properties

    @report_time
    def transform(self) -> None:
        """
        Apply preprocessing transformations to the raw dataset.
        """
        self._data = self._raw_data.rename(
            columns={"EN": ColumnNames.SOURCE.value, "DE": ColumnNames.TARGET.value})
        self._data = self._data.drop_duplicates()
        self._data[ColumnNames.SOURCE.value] = ('Translate from English to German: '
                                                + self._data[ColumnNames.SOURCE.value])
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
        return (str(self._data[ColumnNames.SOURCE.value].iloc[index]),)

    @property
    def data(self) -> pd.DataFrame:
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
    source_tokens = tokenizer(
        text=sample[ColumnNames.SOURCE.value],
        max_length= max_length,
        padding= 'max_length',
        truncation=True,
        return_tensors="pt"
    )

    target_tokens = tokenizer(
        text=sample[ColumnNames.TARGET.value],
        max_length= max_length,
        padding= 'max_length',
        truncation=True,
        return_tensors="pt"
    )

    return {
        "input_ids": source_tokens['input_ids'].squeeze(0),
        "decoder_input_ids": source_tokens['attention_mask'].squeeze(0),
        "labels": target_tokens['input_ids'].squeeze(0)
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
        self._data = list(data.apply(lambda x: tokenize_sample(x, tokenizer, max_length), axis=1))

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
        return dict(self._data[index])


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
        self._model = T5ForConditionalGeneration.from_pretrained(model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)

    def analyze_model(self) -> dict:
        """
        Analyze model computing properties.

        Returns:
            dict: Properties of a model
        """
        if isinstance(self._model, Module):
            result = summary(
                self._model,
                input_data={
                    "input_ids": torch.ones(1, self._model.config.d_model, dtype=torch.long),
                    "decoder_input_ids": torch.ones(1, self._model.config.d_model, dtype=torch.long)
                },
            )
        else:
            print("Model is not of type 'Module'")

        model_properties = {
            'input_shape': list(result.input_size['input_ids']),
            'embedding_size': self._model.config.d_model,
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
        Infer single batch.

        Args:
            sample_batch (Sequence[tuple[str, ...]]): batch to infer the model

        Returns:
            list[str]: model predictions as strings
        """
        input_tokens = self._tokenizer(
            text=sample_batch[0],
            max_length= self._max_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        if self._model is None:
            raise ValueError("Model is not initialized properly.")
        output = self._model.generate(**input_tokens)
        results = self._tokenizer.batch_decode(output, skip_special_tokens=True)
        return list(res for res in results)

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
        predictions_df = pd.read_csv(self._data_path)
        metrics_dict = {}
        predictions = predictions_df[ColumnNames.PREDICTION.value]
        references = predictions_df[ColumnNames.TARGET.value]
        for metric in self._metrics:
            bleu_metric = evaluate.load(metric.value)
            value = bleu_metric.compute(
                references=references,
                predictions=predictions,
            )
            metrics_dict[metric.value] = value[metric.value]
        return metrics_dict


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
        self._lora_config = LoraConfig(
            r=4,
            lora_alpha=8,
            lora_dropout=0.1,
            target_modules=sft_params.target_modules
        )

        self._max_steps = sft_params.max_fine_tuning_steps
        self._per_device_train_batch_size = sft_params.batch_size
        self._learning_rate = sft_params.learning_rate
        self._finetuned_model_path = sft_params.finetuned_model_path
        self._model = T5ForConditionalGeneration.from_pretrained(self._model_name)

        if self._lora_config is None:
            raise ValueError("self._lora_config should not be None")
        self._model = get_peft_model(self._model, self._lora_config)

    def run(self) -> None:
        """
        Fine-tune model.
        """
        if (self._finetuned_model_path is None
                or self._max_steps is None
                or self._per_device_train_batch_size is None
                or self._learning_rate is None):
            return

        training_args = TrainingArguments(
            output_dir=str(self._finetuned_model_path),
            learning_rate=self._learning_rate,
            per_device_train_batch_size=self._per_device_train_batch_size,
            max_steps=self._max_steps,
            save_strategy='no',
            use_cpu=True,
            load_best_model_at_end=False,
        )

        if not isinstance(self._model, Module):
            raise TypeError(f"Model is not of type torch.nn.Module")

        trainer = Trainer(
            model=self._model,
            args=training_args,
            train_dataset=self._dataset
        )
        trainer.train()

        trainer.model.merge_and_unload()
        trainer.model.base_model.save_pretrained(self._finetuned_model_path)

        tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        tokenizer.save_pretrained(self._finetuned_model_path)
