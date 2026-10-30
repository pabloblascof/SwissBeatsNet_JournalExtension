from dataclasses import dataclass, field
from typing import Optional, Any, Dict
from torchvision import transforms

from omegaconf import MISSING
from omegaconf import II


@dataclass
class TransformModuleConfig:
    _target_: str = MISSING


@dataclass
class TransformMimicFolderBasicModule(TransformModuleConfig):
    _target_: str = "datasets.MimicCXRDataset.get_transform_folder_basic"
    img_size: int = II("dataset.img_size")


@dataclass
class TransformMimicDaskBasicModule(TransformModuleConfig):
    _target_: str = "datasets.MimicCXRDataset.get_transform_dask_basic"
    img_size: int = II("dataset.img_size")


@dataclass
class TransformMimicFolderMAEModule(TransformModuleConfig):
    _target_: str = "datasets.MimicCXRDataset.get_transform_folder_mae"
    img_size: int = II("dataset.img_size")
    resize_crop: bool = True
    resize_crop_kwargs: Dict[str, Any] = field(
        default_factory=lambda: dict(
            size=II("dataset.img_size"),
            scale=(0.8, 1.0),
            ratio=(1.0, 1.0),
            interpolation=transforms.InterpolationMode.BILINEAR,
            antialias=True,
        )
    )
    rotation: bool = False
    rotation_kwargs: Dict[str, Any] = field(
        default_factory=lambda: dict(
            degrees=(0, 45),
            interpolation=transforms.InterpolationMode.NEAREST,
            expand=False,
            center=None,
            fill=0,
        )
    )
    horizontal_flip: bool = True
    normalize: bool = True
    normalize_kwargs: Optional[Dict[str, Any]] = field(
        default_factory=lambda: dict(
            mean=[0.5056], std=[0.252]
        )
    )
    jitter: bool = False
    jitter_kwargs: Optional[Dict[str, Any]] = field(
        default_factory=lambda: dict(
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
            hue=0.1
        )
    )


@dataclass
class TransformPolyMNISTModule(TransformModuleConfig):
    _target_: str = "datasets.PolyMNISTDataset.get_transform"
