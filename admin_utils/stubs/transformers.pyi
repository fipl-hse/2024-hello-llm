from pathlib import Path
from typing import Any, Callable, Protocol

import torch
from datasets import Dataset

class EvalPrediction:
    logits: torch.Tensor
    labels: list[str]

    def __getitem__(self, index: int) -> torch.Tensor | list[str]:
        return (self.logits, self.labels)[index]

class TrainingArguments:
    def __init__(
        self,
        output_dir: str,
        learning_rate: float,
        per_device_train_batch_size: int,
        per_device_eval_batch_size: int,
        num_train_epochs: float,
        evaluation_strategy: str,
        eval_steps: int,
        label_names: list[str],
    ): ...

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        args: TrainingArguments,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        compute_metrics: Callable[[EvalPrediction], dict[str, float]],
    ): ...
    def save_model(self, path: Path) -> None: ...
    def train(self) -> None: ...

class BatchEncoding:
    input_features: torch.Tensor
    def to(self, device: str) -> BatchEncoding: ...
    def __getitem__(self, el: str) -> Any: ...
    def __setitem__(self, el: str, val: Any) -> None: ...
    def keys(self) -> list: ...

class LayerLike(Protocol):
    out_features: int

class ContainerLike(Protocol):
    dense: LayerLike
    out_proj: torch.nn.Linear

    def parameters(self) -> list: ...

class AutoModel:
    classifier: ContainerLike
    pooler: ContainerLike
    longformer: ContainerLike
    num_labels: int

    @staticmethod
    def from_pretrained(model_name: str) -> AutoModel: ...
    def load_state_dict(self, state_dict: dict, strict: bool, assign: bool) -> None: ...
    def state_dict(self) -> dict: ...
    def __call__(self, *args: Any, **kwds: Any) -> torch.Tensor: ...
    def to(self, device: str) -> AutoModel: ...
    def eval(self) -> AutoModel: ...
    def generate(self, values: torch.Tensor) -> list: ...

class PreTrainedTokenizerBase:
    cls_token_id: int
    mask_token_id: int

class AutoTokenizer(PreTrainedTokenizerBase):
    def __call__(self, *args: Any, **kwds: Any) -> BatchEncoding: ...
