from dataclasses import dataclass, field
from typing import List, Any
from config.MyMVWSLConfig import TrainingConfig

@dataclass
class ChagasConfigTraining(TrainingConfig):
    # pretraining
    pretraining_data_mode: List[Any] = field(default_factory=lambda: [
        "code_15_non_chagas", "ptbxl_padded", "samitrop_padded", "fallback_padded"
    ])
    # calling this pretraining_epochs would make more sense but entire mm_mae repo expects a parameter called n_epochs (i.e. use that to avoid creating subtle bugs)
    n_epochs: int = 500 # mm_mae: if steepness=0.02 need at least 500 to obtain "full" sigmoid curve

    # fine-tuning parameters
    fine_tuning_data_mode: List[Any] = field(default_factory=lambda: [
        "code_15_chagas", "ptbxl_chagas", "samitrop_chagas", "fallback_chagas"
    ])
    fine_tuning_epochs: int = 50  # mm_mae: must match MimicBaseClf/oe_n_epochs for correct LR scheduling

    # this parameter is also used for offline evaluation BUT also for online evaluation (hence it is okay to have it in this config sub_file)
    binarization_threshold: float = 0.5

    # parameters used by mm_mae code
    n_epochs_warmup: int = 50
    weight_decay: float = 0.05