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
        dataset = load_dataset(self._hf_name, split="test").to_pandas()

        if not isinstance(dataset, pd.DataFrame):
            raise TypeError("Downloaded dataset is not a pandas DataFrame.")

        self._raw_data = dataset


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
        num_samples = self._raw_data.shape[0]
        num_columns = self._raw_data.shape[1]

        num_duplicates = self._raw_data.duplicated().sum()
        num_empty_rows = self._raw_data.isnull().all(axis=1).sum()

        cleaned_data = self._raw_data.dropna()

        min_sample_length, max_sample_length = cleaned_data["article"].str.len().agg(["min", "max"])

        return {
            "dataset_number_of_samples": num_samples,
            "dataset_columns": num_columns,
            "dataset_duplicates": num_duplicates,
            "dataset_empty_rows": num_empty_rows,
            "dataset_sample_min_len": min_sample_length,
            "dataset_sample_max_len": max_sample_length,
        }

    @report_time
    def transform(self) -> None:
        """
        Apply preprocessing transformations to the raw dataset.
        """
        self._data = self._raw_data.copy()
        self._data.rename(columns={'article': ColumnNames.SOURCE.value,
                                   'abstract': ColumnNames.TARGET.value},
                                    inplace=True)
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
    source_encodings = tokenizer(
        sample['source'],
        padding="max_length",
        truncation=True,
        max_length=max_length
    )
    target_encodings = tokenizer(
        sample['target'],
        padding="max_length",
        truncation=True,
        max_length=max_length
    )

    return {
        "input_ids": torch.tensor(source_encodings["input_ids"], dtype=torch.long),
        "attention_mask": torch.tensor(source_encodings["attention_mask"], dtype=torch.long),
        "labels": torch.tensor(target_encodings["input_ids"], dtype=torch.long)
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
        self._data = [tokenize_sample(sample, tokenizer, max_length)
                      for _, sample in data.iterrows()]

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

        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        self._model = self._model = AutoModelForSeq2SeqLM.from_pretrained(self._model_name)
        self._model.to(self._device)

    def analyze_model(self) -> dict:
        """
        Analyze model computing properties.

        Returns:
            dict: Properties of a model
        """
        sample_input = torch.ones((1, self._model.config.encoder.max_position_embeddings),
                                  dtype=torch.long,
                                  device=self._device)

        input_data = {"input_ids": sample_input,
                      "decoder_input_ids": sample_input}

        if not isinstance(self._model, torch.nn.Module):
            raise ValueError("The model has not been correctly initialized")

        model_summary = summary(self._model, input_data=input_data, verbose=0)

        try:
            embedding_size = self._model.config.max_position_embeddings
        except AttributeError:
            if hasattr(self._model.config, "encoder"):
                embedding_size = getattr(self._model.config.encoder,
                                         "max_position_embeddings", None)
                if embedding_size is None:
                    embedding_size = getattr(self._model.config.encoder,
                                             "hidden_size", None)
            else:
                embedding_size = None

        return {
            'input_shape': list(input_data['input_ids'].shape),
            'embedding_size': embedding_size,
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
        loader = DataLoader(dataset=self._dataset, batch_size=self._batch_size)
        collected_preds = []

        with torch.no_grad():
            for batch in loader:
                batch_preds = self._infer_batch(batch)
                collected_preds.extend(batch_preds)

        df_results = pd.DataFrame(self._dataset.data.iloc[:len(collected_preds)])
        df_results[ColumnNames.PREDICTION.value] = collected_preds

        return df_results

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
            [sample[0] for sample in sample_batch],
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self._max_length
        ).to(self._device)

        outputs = self._model.generate(
            **inputs,
            max_length=self._max_length
        )

        decoded: list[str] = self._tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return decoded

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

        preds = data[ColumnNames.PREDICTION.value]
        refs = data[ColumnNames.TARGET.value]

        results = {}
        for metric in self._metrics:
            metric_result = load(metric.value, seed=77).compute(predictions=preds, references=refs)
            if metric.value == "rouge":
                results[metric.value] = metric_result["rougeL"]
            else:
                results[metric.value] = metric_result[metric.value]

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

        self._lora_config = LoraConfig(
            r=4,
            lora_alpha=8,
            lora_dropout=0.1,
            target_modules=sft_params.target_modules
        )

        self._model = AutoModelForSeq2SeqLM.from_pretrained(self._model_name)

        self._batch_size = sft_params.batch_size
        self._max_length = sft_params.max_length
        self._max_sft_steps = sft_params.max_fine_tuning_steps
        self._finetuned_model_path = sft_params.finetuned_model_path
        self._learning_rate = sft_params.learning_rate

        self._device = sft_params.device

    def run(self) -> None:
        """
        Fine-tune model.
        """
        device = self._device if self._device is not None else "cpu"

        assert self._model is not None, "Model must be initialized before running fine-tuning."

        if self._lora_config is not None:
            self._model = get_peft_model(self._model, self._lora_config)
        self._model.to(device)

        batch_size = self._batch_size or 1
        max_steps = self._max_sft_steps or 100
        learning_rate = self._learning_rate or 1e-3

        training_args = TrainingArguments(
            output_dir=str(self._finetuned_model_path),
            per_device_train_batch_size=batch_size,
            max_steps=max_steps,
            learning_rate=learning_rate,
            use_cpu=(device == "cpu"),
            save_strategy="no",
            load_best_model_at_end=False
        )

        trainer = Trainer(
            model=self._model,
            args=training_args,
            train_dataset=self._dataset
        )

        trainer.train()

        if self._lora_config is not None:
            merged_model = self._model.merge_and_unload()
            merged_model.save_pretrained(self._finetuned_model_path)
        else:
            self._model.save_pretrained(self._finetuned_model_path)
