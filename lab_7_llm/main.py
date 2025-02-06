"""
Laboratory work.

Working with Large Language Models.
"""
# pylint: disable=too-few-public-methods, undefined-variable, too-many-arguments, super-init-not-called
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
import torch
from pandas import DataFrame
from datasets import load_dataset, Dataset
from torchinfo import summary
from transformers import LlamaConfig, LlamaTokenizer, AutoModelForCausalLM, AutoTokenizer, AutoConfig


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
        self._raw_data = load_dataset(self._hf_name, split='train').to_pandas()
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
        dataset_number_of_samples = self._raw_data.shape[0]
        dataset_columns = self._raw_data.shape[1]
        dataset_duplicates = len(self._raw_data[self._raw_data.duplicated()])
        df = self._raw_data.replace('', pd.NA)
        dataset_empty_rows = len(df[df.isna().any(axis=1)])
        cleaned = self._raw_data.replace('', pd.NA).dropna().drop_duplicates()
        dataset_sample_min_len = int(cleaned['instruction'].str.len().min())
        dataset_sample_max_len = int(cleaned['instruction'].str.len().max())
        return {'dataset_number_of_samples': dataset_number_of_samples,
                'dataset_columns': dataset_columns,
                'dataset_duplicates':dataset_duplicates,
                'dataset_empty_rows': dataset_empty_rows,
                'dataset_sample_min_len': dataset_sample_min_len,
                'dataset_sample_max_len': dataset_sample_max_len}

    @report_time
    def transform(self) -> None:
        """
        Apply preprocessing transformations to the raw dataset.
        """
        df = self._raw_data.copy()
        df = df[df['category'] == 'open_qa']
        df = df.drop(columns=['context', 'category', '__index_level_0__'])
        df = df.rename(columns={"instruction": "question", "response": "target"})
        self._data = df


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
        self._data = Dataset.from_pandas(data)

    def __len__(self) -> int:
        """
        Return the number of items in the dataset.

        Returns:
            int: The number of items in the dataset
        """
        return self._data.num_rows

    def __getitem__(self, index: int) -> tuple[str, ...]:
        """
        Retrieve an item from the dataset by index.

        Args:
            index (int): Index of sample in dataset

        Returns:
            tuple[str, ...]: The item to be received
        """
        return tuple(self._data.select([index]))

    @property
    def data(self) -> DataFrame:
        """
        Property with access to preprocessed DataFrame.

        Returns:
            pandas.DataFrame: Preprocessed DataFrame
        """
        return self._data.to_pandas()


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
        self._model_name = model_name
        self._model = AutoModelForCausalLM.from_pretrained(model_name)
        self._dataset = dataset
        self._device = device
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._batch_size = batch_size
        self._max_length = max_length
        self._config = AutoConfig.from_pretrained(model_name)
        #print(config)

    def analyze_model(self) -> dict:
        """
        Analyze model computing properties.

        Returns:
            dict: Properties of a model
        """
        #config = self._model.config
        vocab_size = self._config.vocab_size
        embeddings_length = self._config.max_position_embeddings
        print(self._config)
        ids = torch.ones((1, embeddings_length), dtype=torch.long)
        input_data = {"input_ids": ids, "attention_mask": ids}

        statistics = summary(self._model, input_data=input_data, verbose=0)
        print(statistics)
        print(statistics.summary_list)
        input_shape = {'attention_mask': list(statistics.input_size['attention_mask']),
                       'input_ids': list(statistics.input_size['input_ids'])}
        output_shape = statistics.summary_list[-1].output_size

        #max context length should be 20???????
        max_context_length = self._config.max_length
        trainable_params = statistics.trainable_params
        total_param_bytes = statistics.total_param_bytes

        return {'embedding_size': embeddings_length,
                'input_shape': input_shape,
                'max_context_length': max_context_length,
                'num_trainable_params': trainable_params,
                'output_shape': output_shape,
                'size': total_param_bytes,
                'vocab_size': vocab_size}

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

        inputs = self._tokenizer(sample, return_tensors="pt")
        generate_ids = self._model.generate(inputs.input_ids, max_length=self._max_length)
        return self._tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    @report_time
    def infer_dataset(self) -> pd.DataFrame:
        """
        Infer model on a whole dataset.

        Returns:
            pd.DataFrame: Data with predictions
        """

    @torch.no_grad()
    def _infer_batch(self, sample_batch: Sequence[tuple[str, ...]]) -> list[str]:
        """
        Infer model on a single batch.

        Args:
            sample_batch (Sequence[tuple[str, ...]]): Batch to infer the model

        Returns:
            list[str]: Model predictions as strings
        """


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

    @report_time
    def run(self) -> dict | None:
        """
        Evaluate the predictions against the references using the specified metric.

        Returns:
            dict | None: A dictionary containing information about the calculated metric
        """
