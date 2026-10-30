import os
from dataclasses import dataclass, field
from typing import List, Dict

from omegaconf import MISSING
from omegaconf import II


@dataclass
class DataConfig:
    name: str = MISSING
    num_workers: int = 8
    # num views
    num_views: int = MISSING
    modality_names: List[str] = MISSING
    ref_mod_d_size: int = MISSING
    modalities_size: Dict[str, int] = MISSING


@dataclass
class DataModuleConfig:
    _target_: str = MISSING


@dataclass
class TrainDataModuleConfig(DataModuleConfig):
    _target_: str = MISSING
    transform = MISSING


@dataclass
class EvalDataModuleConfig(DataModuleConfig):
    _target_: str = MISSING
    transform = MISSING


@dataclass
class TestDataModuleConfig(DataModuleConfig):
    _target_: str = MISSING
    transform = MISSING
