from dataclasses import dataclass
from omegaconf import MISSING

from config.eval.EvalConfig import EvalConfig
from config.eval.EvalConfig import EvalModuleConfig


@dataclass
class PolyMNISTEvalConfig(EvalConfig):
    # latent representation
    num_samples_train: int = 10000
    max_iteration: int = 10000
    evaluate_lr: bool = True
    logging_frequency_lr: int = 50
    logging_frequency_plots: int = 50


@dataclass
class PMEvalModuleConfig(EvalModuleConfig):
    _target_: str = MISSING


@dataclass
class PMEvalMAEModuleConfig(PMEvalModuleConfig):
    _target_: str = MISSING


@dataclass
class PMsharedEvalMAEModuleConfig(PMEvalMAEModuleConfig):
    _target_: str = "eval.PolyMNISTEval.PolyMNISTEvalMAEshared"


@dataclass
class PMmsEvalMAEModuleConfig(PMEvalMAEModuleConfig):
    _target_: str = "eval.PolyMNISTEval.PolyMNISTEvalMAEms"
