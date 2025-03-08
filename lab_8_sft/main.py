"""
Laboratory work.

Fine-tuning Large Language Models for a downstream task.
"""
# pylint: disable=too-few-public-methods, undefined-variable, duplicate-code, unused-argument, too-many-arguments
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
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
        self._raw_data = load_dataset(path=self._hf_name, name='ru',
                                      split='validation').to_pandas()


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
        return {'dataset_number_of_samples': self._raw_data.shape[0],
                'dataset_columns': self._raw_data.shape[1],
                'dataset_duplicates': self._raw_data.duplicated().sum().tolist(),
                'dataset_empty_rows':
                    sum(self._raw_data.replace('', np.nan).isna().sum().to_list()),
                'dataset_sample_max_len':
                    pd.concat([self._raw_data['premise'].apply(len),
                               self._raw_data['hypothesis'].apply(len)]).max(),
                'dataset_sample_min_len':
                    pd.concat([self._raw_data['premise'].apply(len),
                               self._raw_data['hypothesis'].apply(len)]).min()}

    @report_time
    def transform(self) -> None:
        """
        Apply preprocessing transformations to the raw dataset.
        """
        self._data = self._raw_data.drop_duplicates().replace('', np.nan).dropna()
        self._data = self._data.rename(columns={'label': ColumnNames.TARGET.value}).reset_index()


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
        return (self._data.iloc[index][ColumnNames.PREMISE.value],
                self._data.iloc[index][ColumnNames.HYPOTHESIS.value])

    @property
    def data(self) -> DataFrame:
        """
        Property with access to preprocessed DataFrame.

        Returns:
            pandas.DataFrame: Preprocessed DataFrame
        """
        return self._data


def tokenize_sample(sample: pd.Series, tokenizer: AutoTokenizer,
                    max_length: int) -> dict[str, torch.Tensor]:
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
    tokenized_input = tokenizer(sample[ColumnNames.PREMISE.value],
                                sample[ColumnNames.HYPOTHESIS.value],
                                padding="max_length",
                                truncation=True,
                                max_length=max_length,
                                return_tensors="pt")

    return {"input_ids": tokenized_input["input_ids"].squeeze(0),
            "attention_mask": tokenized_input["attention_mask"].squeeze(0),
            "labels":  sample[ColumnNames.TARGET.value]}


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
        self._data = list(data.apply(lambda sample:
                                     tokenize_sample(sample, tokenizer, max_length),
                                     axis=1))

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
        self._model = AutoModelForSequenceClassification.from_pretrained(model_name)

    def analyze_model(self) -> dict:
        """
        Analyze model computing properties.

        Returns:
            dict: Properties of a model
        """
        if not isinstance(self._model, torch.nn.Module):
            raise TypeError("type of the model should be Module")

        model_config = self._model.config
        model_properties = {}

        model_properties['max_context_length'] = model_config.max_length
        model_properties['vocab_size'] = model_config.vocab_size
        model_properties['embedding_size'] = model_config.max_position_embeddings

        ids = torch.ones((1, model_properties['embedding_size']), dtype=torch.long)
        input_data = {'input_ids': ids, 'attention_mask': ids}

        model_stats = summary(self._model, input_data=input_data, verbose=0)

        model_properties['input_shape'] = {'input_ids': [ids.shape[0], ids.shape[1]],
                                           'attention_mask': [ids.shape[0], ids.shape[1]]}
        model_properties['num_trainable_params'] = model_stats.trainable_params
        model_properties['output_shape'] = model_stats.summary_list[-1].output_size
        model_properties['size'] = model_stats.total_param_bytes

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
        loader = DataLoader(self._dataset, self._batch_size)
        predictions = []
        for batch in loader:
            predictions.extend(self._infer_batch(batch))
        res = pd.DataFrame(self._dataset.data)
        res[ColumnNames.PREDICTION.value] = predictions
        return res[[ColumnNames.TARGET.value, ColumnNames.PREDICTION.value]]

    @torch.no_grad()
    def _infer_batch(self, sample_batch: Sequence[tuple[str, ...]]) -> list[str]:
        """
        Infer single batch.

        Args:
            sample_batch (Sequence[tuple[str, ...]]): batch to infer the model

        Returns:
            list[str]: model predictions as strings
        """
        if len(sample_batch) == 1:
            sample_batch = sample_batch[0]
        tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        tokens = tokenizer(sample_batch[0], sample_batch[1], padding=True, truncation=True,
                           return_tensors='pt')
        output = self._model(**tokens).logits

        return [str(prediction.item()) for prediction in list(torch.argmax(output, dim=1))]


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
        predictions = data[ColumnNames.PREDICTION.value]
        references = data[ColumnNames.TARGET.value]
        scores = {}
        for metric in self._metrics:
            scores[str(metric)] = load(str(metric)).compute(predictions=predictions,
                                                            references=references)[str(metric)]
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

        self._model = AutoModelForSequenceClassification.from_pretrained(self._model_name)
        self._batch_size = sft_params.batch_size
        self._max_length = sft_params.max_length
        self._max_sft_steps = sft_params.max_fine_tuning_steps
        self._device = sft_params.device
        self._finetuned_model_path = sft_params.finetuned_model_path
        self._learning_rate = sft_params.learning_rate

        self._lora_config = LoraConfig(r=4, lora_alpha=8, lora_dropout=0.1,
                                       target_modules=sft_params.target_modules)

    def run(self) -> None:
        """
        Fine-tune model.
        """
        model = get_peft_model(self._model, self._lora_config)
        training_args = TrainingArguments(output_dir=self._finetuned_model_path,
                                          max_steps=self._max_sft_steps,
                                          per_device_train_batch_size=self._batch_size,
                                          learning_rate=self._learning_rate,
                                          save_strategy="no",
                                          use_cpu=bool(self._device == "cpu"),
                                          load_best_model_at_end=False)

        trainer = Trainer(model=model,
                          args=training_args,
                          train_dataset=self._dataset)

        trainer.train()

        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(self._finetuned_model_path)
