from dataclasses import dataclass, field
from typing import Any, List, Optional
from omegaconf import MISSING
from omegaconf import II
from omegaconf import OmegaConf
from config.eval.EvalConfig import EvalConfig, OfflineEvalConfig
from config.eval.EvalConfig import EvalModuleConfig, OfflineEvalModuleConfig


@dataclass
class MimicEvalBaseConfig(EvalConfig):
    evaluate_coherence: bool = False

    # latent representation
    save_encodings: bool = True

    # classifiers trained in the downstream task - only for MIMIC
    classifier_list: List[str] = field(
        default_factory=lambda: [
            "LR",
        ]
    )

    # metrics used for evaluation - only for MIMIC
    metric_list: List[str] = field(
        default_factory=lambda: [
            "AUROC",
            "AP",
        ]
    )
    f_n_jobs: int = 8


@dataclass
class MimicEvalBaseModuleConfig(EvalModuleConfig):
    _target_: str = MISSING
    clfs: List[str] = II("eval.classifier_list")
    metrics: List[str] = II("eval.metric_list")
    f_n_jobs: int = II("eval.f_n_jobs")
    seed: int = II("seed")


@dataclass
class MimicEvalConfig(MimicEvalBaseConfig):
    # RF PARAMETERS
    f_n_estimators: int = 1000
    f_min_samples_split: int = 5
    f_min_samples_leaf: int = 1
    f_max_features: str = "sqrt"
    f_max_depth: int = 5
    f_criterion: str = "entropy"
    f_bootstrap: bool = True


@dataclass
class MimicVAEEvalModuleConfig(MimicEvalBaseModuleConfig):
    _target_: str = "eval.MimicEval.MimicVAEEval"
    f_n_estimators: int = II("eval.f_n_estimators")
    f_min_samples_split: int = II("eval.f_min_samples_split")
    f_min_samples_leaf: int = II("eval.f_min_samples_leaf")
    f_max_features: str = II("eval.f_max_features")
    f_max_depth: int = II("eval.f_max_depth")
    f_criterion: str = II("eval.f_criterion")
    f_bootstrap: bool = II("eval.f_bootstrap")


@dataclass
class MimicMAEEvalModuleConfig(MimicEvalBaseModuleConfig):
    _target_: str = "eval.MimicEval.MimicMAEEval"
    f_n_estimators: int = II("eval.f_n_estimators")
    f_min_samples_split: int = II("eval.f_min_samples_split")
    f_min_samples_leaf: int = II("eval.f_min_samples_leaf")
    f_max_features: str = II("eval.f_max_features")
    f_max_depth: int = II("eval.f_max_depth")
    f_criterion: str = II("eval.f_criterion")
    f_bootstrap: bool = II("eval.f_bootstrap")


@dataclass
class MimicEvalOfflineConfig(OfflineEvalConfig):
    # model checkpoint path to evaluate
    ckpt_path: str = ""

    # Fine Tuning Pipeline Params
    fused: bool = False
    freeze_encoder: bool = False #if 'True', encoders is freezed (probing). If 'False' full fine tuning is performed.
    clf_head_type: str = "non_linear" #'linear' or 'non_linear' classifier head
    feature_policy: str = "cls" #cls or all (avg of tokens)
    checkpoint_init: bool = True # False for supervised clf baseline - the checkpoint is not loaded

    #training setup
    oe_auroc_frequency: int = 10 #auroc is logged every oe_auroc_frequency epochs

@dataclass
class MimicBaseClf(OfflineEvalModuleConfig):
    #TODO discuss how to set default values
    _target_: str = MISSING
    oe_encoders: Optional[Any] = None
    oe_batch_size: int = II("training.batch_size")
    oe_num_nodes: int = II("training.num_nodes")
    oe_num_gpus: int = II("training.num_gpus")
    oe_num_labels: int = II("dataset.num_labels")
    oe_n_epochs: int = 50
    oe_learning_rate: float = 1e-4
    oe_learning_rate_weight_decay: float = II("training.weight_decay")
    oe_checkpoint_init: bool = II("offline_eval.checkpoint_init")
    oe_freeze_encoder: bool = II("offline_eval.freeze_encoder")
    oe_feature_policy: str = II("offline_eval.feature_policy")
    oe_ckpt_path: str = II("offline_eval.ckpt_path")
    oe_clf_head_type: str = II("offline_eval.clf_head_type")
    oe_logging_frequency: int = II("log.wandb_log_freq")
    oe_modality_names: list = field(default_factory=lambda: II("dataset.modality_names"))
    oe_auroc_frequency: int = II("offline_eval.oe_auroc_frequency")
    oe_n_epochs_warmup: int = II("training.n_epochs_warmup")
    


@dataclass
class MimicFrontalClf(MimicBaseClf):
    """Classifier config for unimodal frontal input."""
    _target_: str = "classifiers.Classifiers.UnimodalClassifier"
    oe_target_modality: str = "frontal"


@dataclass
class MimicLateralClf(MimicBaseClf):
    """Classifier config for unimodal lateral input."""
    _target_: str = "classifiers.Classifiers.UnimodalClassifier"
    oe_target_modality: str = "lateral"

@dataclass
class MimicEnsemblelClf(MimicBaseClf):
    """Classifier config for unimodal lateral input."""
    _target_: str = "classifiers.Classifiers.EnsembleClassifier"