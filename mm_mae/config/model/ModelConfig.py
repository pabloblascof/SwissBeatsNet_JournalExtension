from dataclasses import dataclass
from omegaconf import MISSING
from omegaconf import II
from typing import List, Dict


@dataclass
class ModelConfig:
    name: str = MISSING
    offline_eval: bool = False

    # if possible: weight sharing between encoder and decoder of the different modalities (vision only)
    share_weights: bool = False
    share_weights_text: bool = False


@dataclass
class ModelModuleConfig:
    _target_: str = MISSING
    encoders = MISSING
    decoders = MISSING
    evaluator = MISSING
    modality_names: List[str] = II("dataset.modality_names")
    num_samples_lr_train: int = II("eval.num_samples_train")
    offline_eval: bool = II("model.offline_eval")
    learning_rate: float = II("training.base_lr")
