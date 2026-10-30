import os
from dataclasses import dataclass, field
from typing import List, Dict

from omegaconf import MISSING
from omegaconf import II

from config.data.DatasetConfig import DataConfig
from config.data.DatasetConfig import DataModuleConfig
from config.data.DatasetConfig import TrainDataModuleConfig
from config.data.DatasetConfig import EvalDataModuleConfig
from config.data.DatasetConfig import TestDataModuleConfig
from dataclasses import dataclass
from omegaconf import MISSING
from omegaconf import II
from omegaconf import OmegaConf

from config.networks.NetworkConfig import TokenizerModuleConfig


@dataclass
class MimicCXRDataConfig(DataConfig):
    name: str = "mimic_cxr"
    # dataset type: dask or folder
    type: str = "folder"

    # shuffle train dataloader
    shuffle_train_dl: bool = True

    metadata_text: str = "XX"
    # num views = 2 : lateral (LATERAL + LL) and frontal (AP + PA)
    num_views: int = 2  # todo check
    dir_data: str = "XX"
    dir_cache: str = "XX"
    use_cache: bool = True
    # path pointing to the root dir of the data files
    dir_images: str = ""

    # split settings
    splitting_method: str = "random"
    train_val_split: float = 0.8
    test_val_split: float = 0.5
    split_seed: int = 0

    # one_frontal_one_lateral or all_combi_no_missing
    studies_policy: str = "all_combi_no_missing"
    reduced_dataset: bool = False
    num_train_sample: int = 100

    # labels
    target_list: List[str] = field(
        default_factory=lambda: [
            "Atelectasis",
            "Cardiomegaly",
            "Consolidation",
            "Edema",
            "Enlarged Cardiomediastinum",
            "Fracture",
            "Lung Lesion",
            "Lung Opacity",
            "No Finding",
            "Pleural Effusion",
            "Pleural Other",
            "Pneumonia",
            "Pneumothorax",
            "Support Devices",
        ]
    )
    n_clfs_outputs: int = 14
    num_labels: int = 14

    img_size: int = 224  # use 224
    image_channels: int = 1

    ref_mod_d_size: int = 50176  # img-size * img-size = 224*224
    modality_names: List[str] = field(default_factory=lambda: ["frontal", "lateral"])
    modalities_size: Dict[str, int] = field(
        default_factory=lambda: {"frontal": 50176, "lateral": 50176}
    )
    
    # img settings
    img_RGB: bool = False  # set to False if you want to use grayscale images
    pre_load_images: bool = False


# @dataclass
# class MimicCXRDataModuleConfig(DataModuleConfig):
#     _target_: str = MISSING
#     dir_data: str = II("dataset.dir_data")
#     fn_metadata: str = os.path.join(
#         II("dataset.dir_data"), II("dataset.studies_policy") + "_metadata_splits.csv"
#     )
#     img_size: int = II("dataset.img_size")
#     target_list: List[str] = II("dataset.target_list")
#     split: str = MISSING
#     debug: bool = II("dataset.reduced_dataset")
#     # TODO: fix to include transform
#     # transform:


@dataclass
class MimicCXRFolderTrainModuleConfig(TrainDataModuleConfig):
    _target_: str = "datasets.MimicCXRDataset.MimicCXRDatasetFolder"
    dir_data: str = II("dataset.dir_data")
    fn_metadata: str = os.path.join(
        II("dataset.dir_cache"),
        "seed-"
        + II("seed")
        + "_trainval-"
        + II("dataset.train_val_split")
        + "_testval-"
        + II("dataset.test_val_split")
        + "_sp-"
        + II("dataset.studies_policy"),
        II("dataset.studies_policy") + "_metadata_splits.csv",
    )
    fn_metadata_text: str = II("dataset.metadata_text")
    img_size: int = II("dataset.img_size")
    target_list: List[str] = II("dataset.target_list")
    split: str = "train"
    debug: bool = II("dataset.reduced_dataset")
    num_train_sample: int = II("dataset.num_train_sample")
    modality_names: List[str] = II("dataset.modality_names")
    transform = MISSING
    tokenizer = MISSING


@dataclass
class MimicCXRFolderEvalModuleConfig(EvalDataModuleConfig):
    _target_: str = "datasets.MimicCXRDataset.MimicCXRDatasetFolder"
    dir_data: str = II("dataset.dir_data")
    fn_metadata: str = os.path.join(
        II("dataset.dir_cache"),
        "seed-"
        + II("seed")
        + "_trainval-"
        + II("dataset.train_val_split")
        + "_testval-"
        + II("dataset.test_val_split")
        + "_sp-"
        + II("dataset.studies_policy"),
        II("dataset.studies_policy") + "_metadata_splits.csv",
    )
    fn_metadata_text: str = II("dataset.metadata_text")
    img_size: int = II("dataset.img_size")
    target_list: List[str] = II("dataset.target_list")
    split: str = "val"
    debug: bool = False
    modality_names: List[str] = II("dataset.modality_names")
    transform = MISSING
    tokenizer = MISSING
    num_train_sample: int = 0


@dataclass
class MimicCXRFolderTestModuleConfig(TestDataModuleConfig):
    _target_: str = "datasets.MimicCXRDataset.MimicCXRDatasetFolder"
    dir_data: str = II("dataset.dir_data")
    fn_metadata: str = os.path.join(
        II("dataset.dir_cache"),
        "seed-"
        + II("seed")
        + "_trainval-"
        + II("dataset.train_val_split")
        + "_testval-"
        + II("dataset.test_val_split")
        + "_sp-"
        + II("dataset.studies_policy"),
        II("dataset.studies_policy") + "_metadata_splits.csv",
    )
    fn_metadata_text: str = II("dataset.metadata_text")
    img_size: int = II("dataset.img_size")
    target_list: List[str] = II("dataset.target_list")
    split: str = "test"
    debug: bool = False
    modality_names: List[str] = II("dataset.modality_names")
    transform = MISSING
    tokenizer = MISSING
    num_train_sample: int = 0
