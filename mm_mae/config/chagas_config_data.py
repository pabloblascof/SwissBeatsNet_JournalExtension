from dataclasses import dataclass, field
from config.data.DatasetConfig import DataConfig
from typing import List, Any

@dataclass
class ChagasConfigData(DataConfig):
    # general data parameters
    name: str = "chagas"
    debug_limit: int = -1
    lazy: bool = True  # switches between lazy loading (True) and data pre-loading (False)
    target_sr: int = 250
    preprocessing: Any = "team"  # ["none", "team", "st_mem"]

    # parameters for "team_processing"
    intensity_normalization: Any = "standardize_regular_leadwise"  # [stretch_peak, standardize_regular_leadwise, standardize_regular_patientwise, no_normalization]
    standard_len: int = 2250
    pad_mode: Any = "symmetric"
    heartbeat_mode: Any = "multiple"

    # parameters for "st_mem_processing" mode
    st_mem_processing: Any = field(default_factory=lambda: {
        "rand_augment": {
            "use": False,
            "kwargs": {
                "op_names": [
                    "shift", "cutout", "drop", "flip", "erase",
                    "sine", "partial_sine", "partial_white_noise"
                ],
                "level": 10,
                "num_layers": 2,
                "prob": 0.5
            }
        },
        "train_transforms": [
            {"random_crop": {"crop_length": 2250}},
            {"highpass_filter": {"fs": 250, "cutoff": 0.67}},
            {"lowpass_filter": {"fs": 250, "cutoff": 40}},
            {"standardize": {"axis": [-1, -2]}}
        ],
        "eval_transforms": [
            {"highpass_filter": {"fs": 250, "cutoff": 0.67}},
            {"lowpass_filter": {"fs": 250, "cutoff": 40}},
            {"standardize": {"axis": [-1, -2]}}
        ]
    })

    # mm_mae needs this to be present in config
    modality_names: List[str] = field(default_factory=lambda: [f"m{i}" for i in range(12)])
    num_views: int = 12

    # the following parameters are expected to be present by mm_mae code
    # but since we don't use them for anything we can set "empty" values
    dir_data: str = ""
    dir_cache: str = ""
    metadata_text: str = ""
    type: str = ""
    reduced_dataset: bool = False
    studies_policy: str = ""
    train_val_split: float = 0.0
    test_val_split: float = 0.0
    img_size: int = 0
    target_list: List[str] = field(default_factory=list)
    num_train_sample: int = 0
    image_channels: int = 0
    num_labels: int = 0