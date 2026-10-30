from dataclasses import dataclass
from typing import List
from omegaconf import MISSING
from omegaconf import II


@dataclass
class EvalConfig:
    # latent representation
    num_samples_train: int = 10000
    max_iteration: int = 10000
    evaluate_lr: bool = True
    logging_frequency_lr: int = 1
    logging_frequency_plots: int = 50
    validation_every_n_epochs: int = 1


@dataclass
class EvalModuleConfig:
    _target_: str = MISSING
    modality_names: List[str] = II("dataset.modality_names")
    max_iter: int = II("eval.max_iteration")
    evaluate_lr: bool = II("eval.evaluate_lr")
    logging_frequency_lr: int = II("eval.logging_frequency_lr")
    logging_frequency_plots: int = II("eval.logging_frequency_plots")


@dataclass
class OfflineEvalConfig:
    ckpt_path: str = MISSING
    seed: int = 0

@dataclass
class OfflineEvalModuleConfig:
    _target_: str = MISSING