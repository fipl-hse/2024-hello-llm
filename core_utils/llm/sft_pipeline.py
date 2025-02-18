"""
Module with description of abstract LLM pipeline.
"""

# pylint: disable=too-few-public-methods, too-many-arguments, duplicate-code, invalid-name
from abc import ABC, abstractmethod

try:
    from peft import get_peft_model, LoraConfig
except ImportError:
    print('Library "peft" not installed. Failed to import.')

try:
    from torch.utils.data.dataset import Dataset
except ImportError:
    print('Library "torch" not installed. Failed to import.')
    Dataset = None  # type: ignore

try:
    from transformers import Trainer, TrainingArguments
except ImportError:
    print('Library "transformers" not installed. Failed to import.')


from config.lab_settings import SFTParams
from core_utils.llm.llm_pipeline import HFModelLike


class AbstractSFTPipeline(ABC):
    """
    Abstract Fine-Tuning LLM Pipeline.
    """

    #: Model
    _model: HFModelLike | None
    _dataset: Dataset | None

    def __init__(self, model_name: str, dataset: Dataset, sft_params: SFTParams) -> None:
        """
        Initialize an instance of AbstractLLMPipeline.

        Args:
            model_name (str): The name of the pre-trained model.
            dataset (torch.utils.data.dataset.Dataset): The dataset used.
            sft_params (SFTParams) Fine-Tuning parameters.
        """
        self._model_name = model_name
        self._model = None
        self._dataset = dataset
        self._batch_size = sft_params.batch_size
        self._max_length = sft_params.max_length
        self._max_sft_steps = sft_params.max_fine_tuning_steps
        self._device = sft_params.device
        self._finetuned_model_path = sft_params.finetuned_model_path
        self._finetuned_model_path.mkdir(exist_ok=True, parents=True)
        self._lora_config = LoraConfig(
            r=4, lora_alpha=8, lora_dropout=0.1, target_modules=sft_params.target_modules
        )
        self._learning_rate = sft_params.learning_rate

    @abstractmethod
    def run(self) -> None:
        """
        Fine-tune model.
        """


class SFTPipeline(AbstractSFTPipeline):
    """
    Abstract Fine-Tuning LLM Pipeline.
    """

    def run(self) -> None:
        """
        Finetune model.
        """
        training_args = TrainingArguments(
            output_dir=self._finetuned_model_path,
            max_steps=self._max_sft_steps,
            per_device_train_batch_size=self._batch_size,
            learning_rate=self._learning_rate,
            save_strategy="no",
            use_cpu=self._device == "cpu",
            load_best_model_at_end=True,
        )

        if self._dataset is None:
            raise AssertionError("Dataset is not initialized")
        trainer = Trainer(
            model=get_peft_model(self._model, self._lora_config),
            args=training_args,
            train_dataset=self._dataset,
        )

        trainer.train()

        trainer.model.merge_and_unload()
        trainer.model.base_model.save_pretrained(self._finetuned_model_path)
