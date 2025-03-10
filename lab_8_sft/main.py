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
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Trainer, TrainingArguments

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
        dataset = load_dataset(self._hf_name, split="train")
        self._raw_data = pd.DataFrame(dataset)


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
        data = self._raw_data.dropna(subset=['article_content'])
        analyze = {
            'dataset_number_of_samples': self._raw_data.shape[0],
            'dataset_columns': self._raw_data.shape[1],
            'dataset_duplicates': self._raw_data.duplicated().sum(),
            "dataset_empty_rows": self._raw_data.isnull().any(axis=1).sum(),
            'dataset_sample_min_len': data["article_content"].str.len().min(),
            'dataset_sample_max_len': data["article_content"].str.len().max(),
        }
        return analyze

    @report_time
    def transform(self) -> None:
        """
        Apply preprocessing transformations to the raw dataset.
        """
        self._data = (
            self._raw_data.drop(columns=['title', 'date', 'url'])
            .rename(columns={'article_content': ColumnNames.SOURCE.value,
                             'summary': ColumnNames.TARGET.value})
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
        return tuple(self._data.iloc[index])

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
    source = sample[ColumnNames.SOURCE.value]
    target = sample[ColumnNames.TARGET.value]

    tokenized = tokenizer(
        source,
        text_target=target,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

    return {
        "input_ids": tokenized["input_ids"].squeeze(),
        "attention_mask": tokenized["attention_mask"].squeeze(),
        "labels": tokenized["labels"].squeeze()
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
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be pandas DataFrame")

        self._data = data
        self._tokenizer = tokenizer
        self._max_length = max_length


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
        sample = self._data.iloc[index]
        return tokenize_sample(sample, self._tokenizer, self._max_length)


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
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name, model_max_length=self._max_length
        )
        self._model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def analyze_model(self) -> dict:
        """
        Analyze model computing properties.

        Returns:
            dict: Properties of a model
        """
        if not isinstance(self._model, torch.nn.Module):
            raise ValueError("Model is of incorrect type")
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
        data_loader = DataLoader(dataset=self._dataset, batch_size=self._batch_size)
        predictions = []

        with torch.no_grad():
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
        inputs = self._tokenizer(
            list(sample_batch[0]),
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self._device)


        generated_ids = self._model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=self._max_length
        )

        decoded_predictions = self._tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        return [prediction.strip() for prediction in decoded_predictions]


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


    def run(self) -> dict | None:
        """
        Evaluate the predictions against the references using the specified metric.

        Returns:
            dict | None: A dictionary containing information about the calculated metric
        """
        eval_data = pd.read_csv(self.data_path)

        predictions = eval_data[ColumnNames.PREDICTION.value]
        targets = eval_data[ColumnNames.TARGET.value]

        eval_results = {}
        for metric in self._metrics:
            scores = load(metric.value, seed=77).compute(predictions=predictions,
                                                         references=targets)
            if metric.value == "rouge":
                eval_results[metric.value] = scores["rougeL"]
            else:
                eval_results[metric.value] = scores[metric.value]

        return eval_results


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

        self._model = AutoModelForSeq2SeqLM.from_pretrained(self._model_name)
        self._model = get_peft_model(self._model, self._lora_config)

        self._max_length = sft_params.max_length
        self._batch_size = sft_params.batch_size
        self._max_fine_tuning_steps = sft_params.max_fine_tuning_steps
        self._device = sft_params.device
        self._finetuned_model_path = sft_params.finetuned_model_path
        self._learning_rate = sft_params.learning_rate

    def run(self) -> None:
        """
        Fine-tune model.
        """
        if (self._finetuned_model_path is None
                or self._max_fine_tuning_steps is None
                or self._batch_size is None
                or self._learning_rate is None):
            return

        training_args = TrainingArguments(
            output_dir=str(self._finetuned_model_path),
            max_steps=self._max_fine_tuning_steps,
            per_device_train_batch_size=self._batch_size,
            learning_rate=self._learning_rate,
            save_strategy="no",
            use_cpu=True,
            load_best_model_at_end=False
        )

        trainer = Trainer(
            model=self._model,
            args=training_args,
            train_dataset=self._dataset
        )

        trainer.train()

        merged_model = self._model.merge_and_unload()
        merged_model.save_pretrained(self._finetuned_model_path)

        tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        tokenizer.save_pretrained(self._finetuned_model_path)