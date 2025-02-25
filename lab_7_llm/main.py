"""
Laboratory work.

Working with Large Language Models.
"""
# pylint: disable=too-few-public-methods, undefined-variable, too-many-arguments, super-init-not-called
import re
from itertools import chain
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
import torch
from datasets import load_dataset
from evaluate import load
from torch.utils.data import DataLoader, Dataset
from torchinfo import summary
from transformers import AutoModelForTokenClassification, AutoTokenizer

from core_utils.llm.llm_pipeline import AbstractLLMPipeline
from core_utils.llm.metrics import Metrics
from core_utils.llm.raw_data_importer import AbstractRawDataImporter
from core_utils.llm.raw_data_preprocessor import AbstractRawDataPreprocessor
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
        self._raw_data = load_dataset(
            self._hf_name, split="validation", trust_remote_code=True
        ).to_pandas()


class RawDataPreprocessor(AbstractRawDataPreprocessor):
    """
    A class that analyzes and preprocesses a dataset.
    """

    def __init__(self, raw_data: pd.DataFrame):
        super().__init__(raw_data)

    def analyze(self) -> dict:
        """
        Analyze a dataset.

        Returns:
            dict: Dataset key properties
        """
        temp_df = self._raw_data.copy()

        for column in temp_df.columns:
            temp_df[column] = temp_df[column].str.join(" ")

        duplicates = temp_df[temp_df.duplicated()].shape[0]

        empty_rows = self._raw_data[self._raw_data.isnull().all(axis=1)].shape[0]

        no_na_df = self._raw_data.dropna()

        return {
            "dataset_columns": self._raw_data.shape[1],
            "dataset_duplicates": duplicates,
            "dataset_empty_rows": empty_rows,
            "dataset_number_of_samples": self._raw_data.shape[0],
            "dataset_sample_max_len": int(no_na_df["tokens"].str.len().max()),
            "dataset_sample_min_len": int(no_na_df["tokens"].str.len().min()),
        }

    @report_time
    def transform(self) -> None:
        """
        Apply preprocessing transformations to the raw dataset.
        """
        self._data = self._raw_data.rename(
            columns={"ner_tags": "target", "tokens": "source"}
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
        return self._data.shape[0]

    def __getitem__(self, index: int) -> tuple[str, ...]:
        """
        Retrieve an item from the dataset by index.

        Args:
            index (int): Index of sample in dataset

        Returns:
            tuple[str, ...]: The item to be received
        """
        return tuple(self._data.iloc[index]["source"])

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
        self._model = AutoModelForTokenClassification.from_pretrained(self._model_name)
        self._model.to(self._device).eval()
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_name,
        )

    def analyze_model(self) -> dict:
        """
        Analyze model computing properties.

        Returns:
            dict: Properties of a model
        """
        if not isinstance(self._model, torch.nn.Module):
            raise TypeError("Expected self._model to be an instance of torch.nn.Module.")

        model_config = self._model.config

        if not isinstance(model_config.max_position_embeddings, int):
            raise TypeError("Expected model_config.max_position_embeddings to be int type")

        input_ids = torch.ones(
            [1, model_config.max_position_embeddings], dtype=torch.long, device=self._device
        )

        input_data = {"input_ids": input_ids, "attention_mask": input_ids}

        info = summary(model=self._model, input_data=input_data, verbose=0)
        input_size = list(info.input_size["input_ids"])

        return {
            "embedding_size": model_config.max_position_embeddings,
            "input_shape": {"attention_mask": input_size, "input_ids": input_size},
            "max_context_length": model_config.max_length,
            "num_trainable_params": info.trainable_params,
            "output_shape": info.summary_list[-1].output_size,
            "size": info.total_param_bytes,
            "vocab_size": model_config.vocab_size,
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
        if self._model:
            return self._infer_batch(sample_batch=[sample])[0]

        return None

    @report_time
    def infer_dataset(self) -> pd.DataFrame:
        """
        Infer model on a whole dataset.

        Returns:
            pd.DataFrame: Data with predictions
        """
        dataloader = DataLoader(
            dataset=self._dataset, batch_size=self._batch_size, collate_fn=lambda x: x
        )

        data = {"target": self._dataset.data["target"].tolist(), "predictions": []}
        for batch in dataloader:
            pred = self._infer_batch(batch)
            data["predictions"].extend(pred)

        return pd.DataFrame(data=data)

    @torch.no_grad()
    def _infer_batch(self, sample_batch: Sequence[tuple[str, ...]]) -> list[str]:
        """
        Infer model on a single batch.

        Args:
            sample_batch (Sequence[tuple[str, ...]]): Batch to infer the model

        Returns:
            list[str]: Model predictions as strings
        """

        new_sample_batch = []
        for sample in sample_batch:
            if len(sample) > 1:
                new_sample_batch.append(sample)
            else:
                new_sample_batch.append(tuple(re.findall(r"[\w-]+|[-.,!?:;]", sample[0])))

        input_data = self._tokenizer(
            new_sample_batch,
            return_tensors="pt",
            is_split_into_words=True,
            padding=True,
            truncation=True,
        )

        all_words_ids = []
        for sent in range(len(new_sample_batch)):
            tokens_words_mapping = input_data.word_ids(sent)
            label_ids = [(None, None)]

            for word_id in tokens_words_mapping:
                if word_id is not None and word_id != label_ids[-1][0]:
                    label_ids.append((word_id, tokens_words_mapping.index(word_id)))

            label_ids.remove((None, None))
            all_words_ids.append(label_ids)

        if self._model is not None:
            with torch.no_grad():
                logits = self._model(**input_data).logits

        all_labels = [list(map(int, sample)) for sample in torch.argmax(logits, dim=2)]

        res = []
        for index, word_ids in enumerate(all_words_ids):
            res.append(
                str(
                    [
                        all_labels[index][word_id[1]] for word_id in word_ids
                        if word_id[1] is not None and index is not None
                    ]
                )
            )

        return res


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
        dataframe = pd.read_csv(self._data_path)
        res = {}
        for metric in self._metrics:
            evaluation = load(str(metric))

            list_pred = dataframe["predictions"].str.findall(r"\d").tolist()
            list_ref = dataframe["target"].str.findall(r"\d").tolist()

            preds = list(chain.from_iterable([list(map(int, i)) for i in list_pred]))
            ref = list(chain.from_iterable([list(map(int, i)) for i in list_ref]))

            results = evaluation.compute(references=ref, predictions=preds)

            res.update(results)

        return res
