from dataclasses import dataclass
from omegaconf import MISSING
from omegaconf import II
from omegaconf import OmegaConf

OmegaConf.register_new_resolver("eval", eval)
from typing import List, Dict

from config.model.ModelConfig import ModelConfig
from config.model.ModelConfig import ModelModuleConfig


@dataclass
class MAEModelConfig(ModelConfig):
    name: str = MISSING
    patch_size: int = 4
    mask_ratio: float = 0.75
    path: str = "mae.pt"


@dataclass
class MMMAEModuleConfig(ModelModuleConfig):
    _target_: str = MISSING
    encoders = MISSING
    decoders = MISSING
    evaluator = MISSING
    annealing = MISSING
    mask_ratio: float = II("model.mask_ratio")
    learning_rate_weight_decay: float = II("training.weight_decay")
    batch_size: int = II("training.batch_size")
    num_nodes: int = II("training.num_nodes")
    num_gpus: int = II("training.num_gpus")
    n_epochs: int = II("training.n_epochs")
    n_epochs_warmup: int = II("training.n_epochs_warmup")
    sync_logs_on_epoch_end: bool = II("log.sync_logs_on_epoch_end")
    log_debug: bool = II("log.debug")


@dataclass
class MAEIndependentModelConfig(MAEModelConfig):
    name: str = "independent"
    compute_rec: bool = True


@dataclass
class MAEIndependentModelModuleConfig(MMMAEModuleConfig):
    _target_: str = "mm_maes.mm_mae_independent.MMindependentMAE"
    compute_rec: bool = II("model.compute_rec")


@dataclass
class MAEMMVMModelConfig(MAEModelConfig):
    name: str = "mmvm"
    use_latent_heads: bool = True
    regularization_tokens: str = "all"  # all, all_but_cls, cls
    regularization_metric: str = "cosine_sim"
    compute_rec: bool = True
    beta: float = 0.1
    temperature: float = 0.5
    # EDIT: these parameters are needed for challege submission to work
    output_dir: str = ""
    encoder_path: str = ""
    train_data_path: str = ""


@dataclass
class MAEMMVMModelModuleConfig(MMMAEModuleConfig):
    _target_: str = "mm_maes.mm_mae_mmvm.MMVMMAE"
    use_latent_heads: bool = II("model.use_latent_heads")
    regularization_tokens: str = II("model.regularization_tokens")
    use_bce_loss: bool = False
    patch_size: int = II("model.patch_size")
    regularization_metric: str = II("model.regularization_metric")
    compute_rec: bool = II("model.compute_rec")
    temperature: float = II("model.temperature")


@dataclass
class MAEAggregationModelConfig(MAEModelConfig):
    name: str = "joint"
    aggregation: str = "avg"
    use_latent_heads: bool = True
    regularization_tokens: str = "all"  # all, all_but_cls, cls


@dataclass
class MAEAggregationModelModuleConfig(MMMAEModuleConfig):
    _target_: str = "mm_maes.mm_mae_joint.MMaggMAE"
    use_latent_heads: bool = II("model.use_latent_heads")
    regularization_tokens: str = II("model.regularization_tokens")
    patch_size: int = II("model.patch_size")
