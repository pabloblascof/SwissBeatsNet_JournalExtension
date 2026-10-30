import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.model_summary import ModelSummary
import wandb

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from utils.MimicCXRSplitter import MimicCXRSplitter

from config.MyMVWSLConfig import MyMVWSLConfig
from config.MyMVWSLConfig import LogConfig
from config.ModelConfig import DRPMModelConfig
from config.ModelConfig import JointModelConfig
from config.ModelConfig import MixedPriorModelConfig
from config.ModelConfig import UnimodalModelConfig
from config.ModelConfig import MixedPriorStdNormModelConfig
from config.DatasetConfig import PMvanillaDataConfig, MimicCXRDataConfig
from config.DatasetConfig import PMtranslatedData50Config
from config.DatasetConfig import PMtranslatedData55Config
from config.DatasetConfig import PMtranslatedData60Config
from config.DatasetConfig import PMtranslatedData65Config
from config.DatasetConfig import PMtranslatedData70Config
from config.DatasetConfig import PMtranslatedData75Config
from config.DatasetConfig import PMtranslatedData50FixedConfig
from config.DatasetConfig import PMrotatedDataConfig
from config.DatasetConfig import CelebADataConfig
from config.MyMVWSLConfig import EvalConfig
from mv_vaes.mv_joint_vae import MVJointVAE as MVJointVAE
from mv_vaes.mv_unimodal_vae import MVunimodalVAE as MVunimodalVAE
from mv_vaes.mv_mixedprior_vae import MVMixedPriorVAE as MVMixedPriorVAE
from mv_vaes.mv_mixedpriorstdnorm_vae import (
    MVMixedPriorStdNormVAE as MVMixedPriorStdNormVAE,
)


cs = ConfigStore.instance()
# Registering the Config class with the name 'config'.
# TODO add mimic-related configs
cs.store(group="log", name="log", node=LogConfig)
cs.store(group="model", name="drpm", node=DRPMModelConfig)
cs.store(group="model", name="joint", node=JointModelConfig)
cs.store(group="model", name="mixedprior", node=MixedPriorModelConfig)
cs.store(group="model", name="unimodal", node=UnimodalModelConfig)
cs.store(group="model", name="mixedpriorstdnorm", node=MixedPriorStdNormModelConfig)
cs.store(group="eval", name="eval", node=EvalConfig)
cs.store(group="dataset", name="PMvanilla", node=PMvanillaDataConfig)

cs.store(group="dataset", name="PMtranslated50", node=PMtranslatedData50Config)
cs.store(group="dataset", name="PMtranslated55", node=PMtranslatedData55Config)
cs.store(group="dataset", name="PMtranslated60", node=PMtranslatedData60Config)
cs.store(group="dataset", name="PMtranslated65", node=PMtranslatedData65Config)
cs.store(group="dataset", name="PMtranslated70", node=PMtranslatedData70Config)
cs.store(group="dataset", name="PMtranslated75", node=PMtranslatedData75Config)
cs.store(
    group="dataset", name="PMtranslated50fixed", node=PMtranslatedData50FixedConfig
)
cs.store(group="dataset", name="Mimic_cxr", node=MimicCXRDataConfig)
cs.store(group="dataset", name="PMrotated", node=PMrotatedDataConfig)
cs.store(group="dataset", name="CelebA", node=CelebADataConfig)
# cs.store(group="dataset", name="dataset", node=DataConfig)
cs.store(name="base_config", node=MyMVWSLConfig)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def run_experiment(cfg: MyMVWSLConfig):
    print(OmegaConf.to_yaml(cfg))
    mimic_cxr_splitter = MimicCXRSplitter(cfg)


if __name__ == "__main__":
    run_experiment()
