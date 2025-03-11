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
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    T5TokenizerFast,
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

        dataset = load_dataset(self._hf_name, split='test', revision='v2.0', trust_remote_code=True)
        self._raw_data = pd.DataFrame(dataset)

        if not isinstance(self._raw_data, pd.DataFrame):
            raise TypeError("The downloaded dataset is not a pandas DataFrame.")


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
        ds_properties = {
            'dataset_number_of_samples': self._raw_data.shape[0],
            'dataset_columns': self._raw_data.shape[1],
            'dataset_duplicates': self._raw_data.duplicated().sum(),
            'dataset_empty_rows': self._raw_data.isnull().all(axis=1).sum(),
            'dataset_sample_min_len': self._raw_data['text'].dropna(how='all').map(len).min(),
            'dataset_sample_max_len': self._raw_data['text'].dropna(how='all').map(len).max()
        }

        return ds_properties

    @report_time
    def transform(self) -> None:
        """
        Apply preprocessing transformations to the raw dataset.
        """

        renamed_dataset = self._raw_data.rename(columns={
            'summary': ColumnNames.TARGET.value,
            'text': ColumnNames.SOURCE.value
        }, inplace=False)

        # renamed_dataset.drop(columns=['title', 'date', 'url'])
        # renamed_dataset.reset_index(drop=True)
        renamed_dataset = renamed_dataset.drop(columns=['title', 'date', 'url']).reset_index(drop=True)

        self._data = renamed_dataset


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
        return self._data.shape[0]

    def __getitem__(self, index: int) -> tuple[str, ...]:
        """
        Retrieve an item from the dataset by index.

        Args:
            index (int): Index of sample in dataset

        Returns:
            tuple[str, ...]: The item to be received
        """
        return (self._data.iloc[index][ColumnNames.SOURCE.value],
                self._data.iloc[index][ColumnNames.TARGET.value])

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
    # tokenized_source = tokenizer(sample[ColumnNames.SOURCE.value],
    #                              padding="max_length",
    #                              truncation=True,
    #                              return_tensors="pt",
    #                              max_length=max_length
    #                              )
    # tokenized_target = tokenizer(sample[ColumnNames.TARGET.value],
    #                              padding="max_length",
    #                              truncation=True,
    #                              return_tensors="pt",
    #                              max_length=max_length
    #                              )
    # return {
    #     "input_ids": tokenized_source["input_ids"].squeeze(),
    #     "attention_mask": tokenized_source["attention_mask"].squeeze(),
    #     "labels": tokenized_target["input_ids"].squeeze()
    # }

    tokenized = tokenizer(
        [sample[ColumnNames.SOURCE.value], sample[ColumnNames.TARGET.value]],
        padding="max_length",
        truncation=True,
        return_tensors="pt",
        max_length=max_length
    )

    return {
        "input_ids": tokenized["input_ids"][0].squeeze(),
        "attention_mask": tokenized["attention_mask"][0].squeeze(),
        "labels": tokenized["input_ids"][1].squeeze()
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
        self._tokenizer = tokenizer
        self._max_length = max_length
        self._data = list(
            data.apply(lambda sample: tokenize_sample(sample, tokenizer, max_length), axis=1)
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
        item = self._data[index]
        if not isinstance(item, dict):
            raise TypeError(f"Expected dict, but got {type(item)} at index {index}")

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

        self.tokenizer = T5TokenizerFast.from_pretrained(model_name)
        self._model: Module = (AutoModelForSeq2SeqLM.from_pretrained(model_name)
                               .to(self._device))
        self._model.eval()

    def analyze_model(self) -> dict:
        """
        Analyze model computing properties.

        Returns:
            dict: Properties of a model
        """
        model_config = self._model.config

        embeddings_length = model_config.hidden_size
        input_data = torch.ones((1, embeddings_length), dtype=torch.long)
        model_summary = summary(self._model,
                                input_data=input_data, decoder_input_ids=input_data)

        return {
            'input_shape': model_summary.summary_list[0].input_size,
            'embedding_size': embeddings_length,
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

        return self._infer_batch([sample])[0]

    @report_time
    def infer_dataset(self) -> pd.DataFrame:
        """
        Infer model on a whole dataset.

        Returns:
            pd.DataFrame: Data with predictions
        """
        dataloader = DataLoader(batch_size=self._batch_size, dataset=self._dataset)

        predictions = []
        targets = []

        for batch in dataloader:
            batch_texts, batch_labels = batch
            batch_tuples = [(text,) for text in batch_texts]
            preds = self._infer_batch(batch_tuples)
            predictions.extend(preds)
            targets.extend(batch_labels)

        return pd.DataFrame({
            ColumnNames.TARGET.value: targets,
            ColumnNames.PREDICTION.value: predictions
        })

    @torch.no_grad()
    def _infer_batch(self, sample_batch: Sequence[tuple[str, ...]]) -> list[str]:
        """
        Infer single batch.

        Args:
            sample_batch (Sequence[tuple[str, ...]]): batch to infer the model

        Returns:
            list[str]: model predictions as strings
        """

        texts = [sample[0] for sample in sample_batch]
        inputs = self.tokenizer(texts,
                                      return_tensors='pt',
                                      max_length=self._max_length,
                                      padding=True,
                                      truncation=True).to(self._device)

        outputs = self._model.generate(**inputs)

        outputs_decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return [str(pred) for pred in outputs_decoded]


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
        data = pd.read_csv(self._data_path)
        predictions = data[ColumnNames.PREDICTION.value].tolist()
        references = data[ColumnNames.TARGET.value].tolist()

        scores = {}
        for metric in self._metrics:
            if metric == Metrics.ROUGE.value:
                metric_evaluator = load(metric.value, seed=77)
            else:
                metric_evaluator = load(metric.value)
            calculated = metric_evaluator.compute(predictions=predictions, references=references)

            if metric.value == Metrics.ROUGE.value:
                scores[metric.value] = float(calculated['rougeL'])
            else:
                scores[metric.value] = float(calculated[metric.value])

        return scores


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
        self._model = AutoModelForSeq2SeqLM.from_pretrained(self._model_name)
        self._batch_size = sft_params.batch_size
        self._lora_config = LoraConfig(r=4,
                                       lora_alpha=8,
                                       lora_dropout=0.1,
                                       target_modules=sft_params.target_modules
                                       )
        self._device = sft_params.device
        self._model = get_peft_model(self._model, self._lora_config).to(self._device)
        self._max_length = sft_params.max_length
        self._max_sft_steps = sft_params.max_fine_tuning_steps
        self._finetuned_model_path = sft_params.finetuned_model_path
        self._learning_rate = sft_params.learning_rate

    def run(self) -> None:
        """
        Fine-tune model.
        """

        if (self._finetuned_model_path is None
                or self._batch_size is None
                or self._max_sft_steps is None
                or self._learning_rate is None):
            return

        if not isinstance(self._model, torch.nn.Module):
            raise TypeError("Wrong class of model")

        training_args = TrainingArguments(
            output_dir=str(self._finetuned_model_path),
            per_device_train_batch_size=self._batch_size,
            max_steps=self._max_sft_steps,
            learning_rate=self._learning_rate,
            save_strategy="no",
            use_cpu=bool(self._device == "cpu"),
            load_best_model_at_end=False
        )

        trainer = Trainer(
            model=self._model,
            args=training_args,
            train_dataset=self._dataset,
        )

        trainer.train()

        trainer.model.base_model.merge_and_unload()
        trainer.model.base_model.save_pretrained(self._finetuned_model_path)
        AutoTokenizer.from_pretrained(self._model_name).save_pretrained(self._finetuned_model_path)
