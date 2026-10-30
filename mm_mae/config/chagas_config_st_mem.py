from dataclasses import dataclass, field
from typing import List, Optional, Any
from config.model.ModelConfig import ModelConfig
from dataclasses import field

@dataclass
class DDPConfig:
    world_size: int = 1
    rank: int = -1
    gpu: int = 0
    dist_url: str = "env://"
    dist_backend: str = "nccl"
    distributed: bool = False

@dataclass
class ModelConfigBlock:
    num_leads: int = 12
    num_classes: int = 2  # st_mem code won't allow us to set this to 1

@dataclass
class DatasetConfig:
    pass  # we are using our own dataset 

@dataclass
class RandAugmentConfig:
    use: bool = True
    op_names: List[str] = field(default_factory=lambda: [
        "shift", "cutout", "drop", "flip", "erase", "sine", "partial_sine", "partial_white_noise"
    ])
    level: int = 10
    num_layers: int = 2
    prob: float = 0.5

@dataclass
class TransformsConfig:
    train_transforms: List[Any] = field(default_factory=lambda: [
        {"random_crop": {"crop_length": 2250}},
        {"highpass_filter": {"fs": 250, "cutoff": 0.67}},
        {"lowpass_filter": {"fs": 250, "cutoff": 40}},
        {"standardize": {"axis": [-1, -2]}},
    ])
    eval_transforms: List[Any] = field(default_factory=lambda: [
        {"n_crop": {"crop_length": 2250, "num_segments": 3}},
        {"highpass_filter": {"fs": 250, "cutoff": 0.67}},
        {"lowpass_filter": {"fs": 250, "cutoff": 40}},
        {"standardize": {"axis": [-1, -2]}},
    ])

@dataclass
class DataloaderConfig:
    # this is a duplicate parameter (main config also has it) but it is useful since then we don't have to always reconfigure it for st_mem and mm_mae
    batch_size: int = 32  
    pin_memory: bool = True

@dataclass
class TrainingConfig:
    freeze_enc: bool = True
    accum_iter: int = 1
    lr: Optional[float] = None
    weight_decay: float = 0.05
    dist_eval: bool = False
    max_norm: Optional[float] = None
    optimizer: str = "adamw"
    min_lr: float = 0.0

@dataclass
class LossConfig:
    name: str = "cross_entropy"

@dataclass
class MetricConfig:
    task: str = "multiclass"
    compute_on_cpu: bool = True
    sync_on_compute: bool = False
    num_classes: int = 2
    target_metrics: Any = field(default_factory=lambda: [])  # we don't use st_mem evaluation code for evaluation

@dataclass
class TestConfig:
    target_metric: str = "loss"

@dataclass
class ChagasConfigStMem(ModelConfig):
    # set in main_downstream.py
    encoder_path: str = ""
    train_data_path: str = ""

    # top-level
    seed: int = 0
    output_dir: str = "./st_mem"
    exp_name: str = "finetune"
    resume: Optional[str] = None
    start_epoch: int = 0
    model_name: str = "st_mem_vit_base"
    mode: str = "finetune"
    device: str = "cuda"

    # nested configs
    ddp: DDPConfig = field(default_factory=DDPConfig)
    model: ModelConfigBlock = field(default_factory=ModelConfigBlock)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    rand_augment: RandAugmentConfig = field(default_factory=RandAugmentConfig)
    transforms: TransformsConfig = field(default_factory=TransformsConfig)
    dataloader: DataloaderConfig = field(default_factory=DataloaderConfig)
    train: TrainingConfig = field(default_factory=TrainingConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    metric: MetricConfig = field(default_factory=MetricConfig)
    test: TestConfig = field(default_factory=TestConfig)


    # parameters that are forced to be present but are either not used at all (because they are mm_mae params 
    # that are not relevant to st_mem) or overwritten in main config (hence it is okay to give them "empty" values)
    patch_size: int = 0  # overwritten in main_config
    mask_ratio: float = 0.0  # overwritten in main_config
    regularization_metric: str = ""
    use_latent_heads: bool = False
    regularization_tokens: str = ""
    compute_rec: bool = False
    temperature: float = 0.0