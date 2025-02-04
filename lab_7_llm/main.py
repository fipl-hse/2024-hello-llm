"""
Laboratory work.

Working with Large Language Models.
"""
import torch
import pandas as pd

from datasets import load_dataset

from core_utils.llm.llm_pipeline import AbstractLLMPipeline
from core_utils.llm.metrics import Metrics
from core_utils.llm.raw_data_importer import AbstractRawDataImporter
from core_utils.llm.raw_data_preprocessor import AbstractRawDataPreprocessor, ColumnNames
from core_utils.llm.task_evaluator import AbstractTaskEvaluator
from core_utils.llm.time_decorator import report_time

from pandas import DataFrame
from pathlib import Path
from typing import Iterable, Sequence
from torch.utils.data import Dataset
from torchinfo import summary
from transformers import MarianMTModel, MarianTokenizer
# pylint: disable=too-few-public-methods, undefined-variable, too-many-arguments, super-init-not-called


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
        self._raw_data = load_dataset(path=self._hf_name, split='if_test').to_pandas()

        if not isinstance(self._raw_data, pd.DataFrame):
            raise TypeError


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
        properties = dict()
        properties['dataset_number_of_samples'] = self._raw_data.shape[0]
        properties['dataset_columns'] = self._raw_data.shape[1]
        properties['dataset_duplicates'] = self._raw_data.duplicated().sum().tolist()
        empty = self._raw_data[['ru', 'en', 'ru_annotated']].apply(lambda row: sum(len(x) for x in row), axis=1)
        properties['dataset_empty_rows'] = (self._raw_data.drop(empty[empty == 0], axis=0).isna().sum().sum().tolist()
                                            + len(empty[empty == 0]))
        if properties['dataset_empty_rows'] > 0:
            self._raw_data = self._raw_data.dropna()
        properties['dataset_sample_max_len'] = self._raw_data['ru'].apply(lambda x: len(x)).max().tolist()
        properties['dataset_sample_min_len'] = self._raw_data['ru'].apply(lambda x: len(x)).min().tolist()
        return properties

    @report_time
    def transform(self) -> None:
        """
        Apply preprocessing transformations to the raw dataset.
        """
        self._data = self._raw_data.drop(columns=['ru_annotated', 'styles'])
        self._data = self._data.rename(columns={'ru': 'source', 'en': 'target'}).reset_index()


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
        return (self._data.iloc[index][ColumnNames.SOURCE.value], )

    @property
    def data(self) -> DataFrame:
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
    _model: torch.nn.Module

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
        self._tokenizer = MarianTokenizer.from_pretrained(model_name)
        self._model = MarianMTModel.from_pretrained(model_name)

    def analyze_model(self) -> dict:
        """
        Analyze model computing properties.

        Returns:
            dict: Properties of a model
        """
        model_config = self._model.config
        model_properties = dict()

        model_properties['max_context_length'] = model_config.max_length
        model_properties['vocab_size'] = model_config.vocab_size
        model_properties['embedding_size'] = model_config.max_position_embeddings

        ids = torch.ones((self._batch_size, model_properties['embedding_size']), dtype=torch.long, device=self._device)
        input_data = {"input_ids": ids, "decoder_input_ids": ids}

        model_stats = summary(self._model, input_data=input_data, verbose=0)

        model_properties['input_shape'] = [model_stats.input_size['input_ids'][0],
                                           model_stats.input_size['input_ids'][1]]
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
        pass

    @torch.no_grad()
    def _infer_batch(self, sample_batch: Sequence[tuple[str, ...]]) -> list[str]:
        """
        Infer model on a single batch.

        Args:
            sample_batch (Sequence[tuple[str, ...]]): Batch to infer the model

        Returns:
            list[str]: Model predictions as strings
        """
        sample_batch_squeezed = [sample for el in sample_batch for sample in el]
        inputs = self._tokenizer.tok.prepare_seq2seq_batch(src_texts=sample_batch_squeezed,
                                                           padding=True, truncation=True,
                                                           return_tensors='pt').input_ids
        return list(self._tokenizer.batch_decode(self._model.generate(inputs), skip_special_tokens=True))


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
        pass

    @report_time
    def run(self) -> dict | None:
        """
        Evaluate the predictions against the references using the specified metric.

        Returns:
            dict | None: A dictionary containing information about the calculated metric
        """
        pass
