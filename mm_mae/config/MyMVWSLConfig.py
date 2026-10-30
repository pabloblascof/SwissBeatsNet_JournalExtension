from dataclasses import dataclass, field
from typing import List, Optional
from omegaconf import MISSING

from config.data.TransformConfig import TransformModuleConfig
from config.data.DatasetConfig import DataConfig
from config.data.DatasetConfig import TrainDataModuleConfig
from config.data.DatasetConfig import EvalDataModuleConfig
from config.data.DatasetConfig import TestDataModuleConfig
from config.model.ModelConfig import ModelConfig
from config.model.ModelConfig import ModelModuleConfig
from config.model.UtilsConfig import AnnealingModuleConfig
from config.model.UtilsConfig import AnnealingConfig
from config.networks.NetworkConfig import EncoderModuleConfig
from config.networks.NetworkConfig import DecoderModuleConfig
from config.networks.NetworkConfig import MMNetworksModuleConfig
from config.networks.NetworkConfig import TokenizerModuleConfig
from config.eval.EvalConfig import EvalConfig, OfflineEvalConfig, OfflineEvalModuleConfig
from config.eval.EvalConfig import EvalModuleConfig


@dataclass
class LogConfig:
    # wandb
    wandb_entity: str = "agostini335"
    # mv_mimic_mps, mv_mimic_scicore
    wandb_group: str = "mm_mae_polymnist"
    wandb_run_name: str = ""
    wandb_project_name: str = "ms"
    wandb_log_freq: int = 10
    wandb_offline: bool = True
    wandb_local_instance: bool = True
    wandb_checkpoint_metric: str = "val/loss/rec_loss"

    # logs
    dir_logs: str = "/logs"

    # Speedup training by syncing at the end of epochs only
    sync_logs_on_epoch_end: bool = True

    # debug level wandb
    debug: bool = False


@dataclass
class TrainingConfig:
    # mps, cuda, cpu
    device: str = "cuda"
    batch_size: int = 32
    base_lr: float = 1e-4
    n_epochs: int = 500
    num_gpus: int = 1
    num_nodes: int = 1
    gradient_clipping: float = 0.0  # 0 = no clipping
    strategy: str = "auto"  # can be set to "ddp_find_unused_parameters_true" 
    do_pretraining: bool = True # set to False if you want to do offline eval only

@dataclass
class MAETrainingConfig(TrainingConfig):
    n_epochs_warmup: int = 50
    weight_decay: float = 0.05


@dataclass
class MyMVWSLConfig:
    seed: int = 0
    checkpoint_metric: str = "val/loss/loss"
    # logger
    log: LogConfig = MISSING
    # data
    dataset: DataConfig = MISSING
    datamodule_train: TrainDataModuleConfig = MISSING
    datamodule_eval: EvalDataModuleConfig = MISSING
    datamodule_test: TestDataModuleConfig = MISSING
    transformmodule: TransformModuleConfig = MISSING
    # model
    model: ModelConfig = MISSING
    modelmodule: ModelModuleConfig = MISSING
    encodermodule: EncoderModuleConfig = MISSING
    decodermodule: DecoderModuleConfig = MISSING
    # text encoder,decoder and tokenizer are optional so that we can run vision-only experiments
    textencodermodule: Optional[EncoderModuleConfig] = None
    textdecodermodule: Optional[DecoderModuleConfig] = None
    tokenizermodule: Optional[TokenizerModuleConfig] = None
    annealing: AnnealingConfig = MISSING
    annealingmodule: Optional[AnnealingModuleConfig] = None
    # networks: MMNetworksModuleConfig = MISSING
    # mm_networks: MMNetworksConfig = MISSING
    # training
    training: TrainingConfig = MISSING
    # eval
    eval: EvalConfig = MISSING
    evalmodule: EvalModuleConfig = MISSING
    # offline eval
    offline_eval: Optional[OfflineEvalConfig] = None
    offline_evalmodule: Optional[OfflineEvalModuleConfig] = None
    # tasks to be performed
    do_pretraining: bool = True
    do_offline_eval: bool = True
    # EDIT: 
    downstream_base_lr: float = 1
    checkpoint_init_path: str = ""
