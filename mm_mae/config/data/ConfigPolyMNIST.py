import os
from dataclasses import dataclass, field
from typing import List, Dict

from omegaconf import MISSING
from omegaconf import II

from config.data.DatasetConfig import DataConfig
from config.data.DatasetConfig import TrainDataModuleConfig
from config.data.DatasetConfig import EvalDataModuleConfig
from config.data.DatasetConfig import TestDataModuleConfig


@dataclass
class PolyMNISTDataConfig(DataConfig):
    num_views: int = 3
    modality_names: List[str] = field(default_factory=lambda: ["m0", "m1", "m2"])
    ref_mod_d_size: int = 2352  # 3*28*28
    modalities_size: Dict[str, int] = field(
        default_factory=lambda: {"m0": 2352, "m1": 2352, "m2": 2352}
    )
    image_channels: int = 3
    num_labels: int = 1
    img_size: int = 28
    dir_data_base: str = "/usr/scratch/projects/multimodality/data"
    dir_clfs_base: str = (
        "/usr/scratch/projects/multimodality/mvvae/experiments/trained_clfs/PolyMNIST"
    )
    missing_ratio: float = 0.0

    # initialized by PolyMNIST variations
    suffix_data: str = MISSING
    suffix_clfs: str = MISSING


@dataclass
class PMvanillaDataConfig(PolyMNISTDataConfig):
    name: str = "PM_vanilla"
    suffix_data: str = "PolyMNIST_vanilla"
    suffix_clfs: str = "vanilla_resnet"
    modality_specific_labels: bool = False


@dataclass
class PMtranslatedData50Config(PolyMNISTDataConfig):
    name: str = "PM_translated_50"
    suffix_data: str = "PolyMNIST_translated"
    suffix_clfs: str = "translated50_resnet"
    modality_specific_labels: bool = False


@dataclass
class PMtranslatedData50MSConfig(PolyMNISTDataConfig):
    name: str = "PM_translated_50_ms"
    suffix_data: str = "PolyMNIST_translated_scale050_ms"
    suffix_clfs: str = "translated50_resnet"
    modality_specific_labels: bool = True


@dataclass
class PMtranslatedData55Config(PolyMNISTDataConfig):
    name: str = "PM_translated_55"
    suffix_data: str = "PolyMNIST_translated_55"
    suffix_clfs: str = "translated55_resnet"
    modality_specific_labels: bool = False


@dataclass
class PMtranslatedData60Config(PolyMNISTDataConfig):
    name: str = "PM_translated_60"
    suffix_data: str = "PolyMNIST_translated_60"
    suffix_clfs: str = "translated60_resnet"
    modality_specific_labels: bool = False


@dataclass
class PMtranslatedData65Config(PolyMNISTDataConfig):
    name: str = "PM_translated_65"
    suffix_data: str = "PolyMNIST_translated_65"
    suffix_clfs: str = "translated65_resnet"
    modality_specific_labels: bool = False


@dataclass
class PMtranslatedData70Config(PolyMNISTDataConfig):
    name: str = "PM_translated_70"
    suffix_data: str = "translated_70"
    suffix_clfs: str = "translated70_resnet"
    modality_specific_labels: bool = False


@dataclass
class PMtranslatedData75Config(PolyMNISTDataConfig):
    name: str = "PM_translated75"
    suffix_data: str = "PolyMNIST_translated_scale075"
    suffix_clfs: str = "translated75_resnet"
    modality_specific_labels: bool = False


@dataclass
class PMtranslatedData50FixedConfig(PolyMNISTDataConfig):
    name: str = "PM_translated_50_fixed"
    suffix_data: str = "PolyMNIST_translated_50_fixed"
    suffix_clfs: str = "translated_50_fixed_resnet"
    modality_specific_labels: bool = False


@dataclass
class PMrotatedDataConfig(PolyMNISTDataConfig):
    name: str = "PM_rotated"
    suffix_data: str = "PolyMNIST_rotated"
    suffix_clfs: str = "rotated_resnet"
    modality_specific_labels: bool = False


@dataclass
class PMTrainDataModuleConfig(TrainDataModuleConfig):
    _target_: str = "datasets.PolyMNISTDataset.PolyMNIST"
    transform = MISSING
    dir_data: str = os.path.join(II("dataset.dir_data_base"), II("dataset.suffix_data"))
    num_views: int = II("dataset.num_views")
    modality_specific_labels: bool = II("dataset.modality_specific_labels")
    split: str = "train"
    missing_ratio: float = II("dataset.missing_ratio")
    tokenizer = MISSING


@dataclass
class PMEvalDataModuleConfig(EvalDataModuleConfig):
    _target_: str = "datasets.PolyMNISTDataset.PolyMNIST"
    transform = MISSING
    dir_data: str = os.path.join(II("dataset.dir_data_base"), II("dataset.suffix_data"))
    num_views: int = II("dataset.num_views")
    modality_specific_labels: bool = II("dataset.modality_specific_labels")
    split: str = "test"
    missing_ratio: float = II("dataset.missing_ratio")
    tokenizer = MISSING


@dataclass
class PMTestDataModuleConfig(TestDataModuleConfig):
    _target_: str = "datasets.PolyMNISTDataset.PolyMNIST"
    transform = MISSING
    dir_data: str = os.path.join(II("dataset.dir_data_base"), II("dataset.suffix_data"))
    num_views: int = II("dataset.num_views")
    modality_specific_labels: bool = II("dataset.modality_specific_labels")
    split: str = "test"
    missing_ratio: float = II("dataset.missing_ratio")
    tokenizer = MISSING
