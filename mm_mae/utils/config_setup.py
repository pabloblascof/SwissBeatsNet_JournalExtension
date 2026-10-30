import torch.nn as nn
import warnings
from hydra.utils import instantiate
from typing import List

from hydra.core.config_store import ConfigStore
from config.data.TransformConfig import (
    TransformMimicDaskBasicModule,
    TransformMimicFolderBasicModule,
    TransformMimicFolderMAEModule,
    TransformPolyMNISTModule,
)

from config.MyMVWSLConfig import MyMVWSLConfig
from config.MyMVWSLConfig import LogConfig
from config.model.MMMAEConfig import MAEIndependentModelConfig
from config.model.MMMAEConfig import MAEMMVMModelConfig
from config.model.MMMAEConfig import MAEAggregationModelConfig
from config.model.MMMAEConfig import MAEIndependentModelModuleConfig
from config.model.MMMAEConfig import MAEMMVMModelModuleConfig
from config.model.MMMAEConfig import MAEAggregationModelModuleConfig
from config.model.UtilsConfig import SigmoidAnnealingModuleConfig
from config.model.UtilsConfig import LinearAnnealingModuleConfig
from config.model.UtilsConfig import ExpAnnealingModuleConfig
from config.model.UtilsConfig import CosAnnealingModuleConfig
from config.model.UtilsConfig import NoAnnealingModuleConfig
from config.model.UtilsConfig import SigmoidAnnealingConfig
from config.model.UtilsConfig import LinearAnnealingConfig
from config.model.UtilsConfig import ExpAnnealingConfig
from config.model.UtilsConfig import CosAnnealingConfig
from config.model.UtilsConfig import NoAnnealingConfig
from config.networks.ConfigMAENetworks import ViTtinyEncoderModuleConfig
from config.networks.ConfigMAENetworks import ViTtinyDecoderModuleConfig
from config.networks.ConfigMAENetworks import ViTbaseEncoderModuleConfig
from config.networks.ConfigMAENetworks import ViTbaseDecoderModuleConfig
from config.networks.ConfigMAENetworks import ViTlargeEncoderModuleConfig, ViTlargeDecoderModuleConfig
from config.data.ConfigMimic import MimicCXRDataConfig
from config.data.ConfigMimic import MimicCXRFolderTestModuleConfig
from config.data.ConfigMimic import MimicCXRFolderEvalModuleConfig
from config.data.ConfigMimic import MimicCXRFolderTrainModuleConfig
from config.data.ConfigPolyMNIST import PMvanillaDataConfig
from config.data.ConfigPolyMNIST import PMtranslatedData50Config
from config.data.ConfigPolyMNIST import PMtranslatedData50MSConfig
from config.data.ConfigPolyMNIST import PMtranslatedData55Config
from config.data.ConfigPolyMNIST import PMtranslatedData60Config
from config.data.ConfigPolyMNIST import PMtranslatedData65Config
from config.data.ConfigPolyMNIST import PMtranslatedData70Config
from config.data.ConfigPolyMNIST import PMtranslatedData75Config
from config.data.ConfigPolyMNIST import PMtranslatedData50FixedConfig
from config.data.ConfigPolyMNIST import PMrotatedDataConfig
from config.data.ConfigPolyMNIST import PMTrainDataModuleConfig
from config.data.ConfigPolyMNIST import PMEvalDataModuleConfig
from config.data.ConfigPolyMNIST import PMTestDataModuleConfig
from config.eval.PolyMNISTEvalConfig import PMsharedEvalMAEModuleConfig
from config.eval.PolyMNISTEvalConfig import PMmsEvalMAEModuleConfig
from config.eval.PolyMNISTEvalConfig import PolyMNISTEvalConfig
from config.eval.MimicEvalConfig import MimicEnsemblelClf, MimicEvalConfig, MimicEvalOfflineConfig, MimicFrontalClf, MimicLateralClf
from config.eval.MimicEvalConfig import MimicVAEEvalModuleConfig
from config.eval.MimicEvalConfig import MimicMAEEvalModuleConfig#,MimicOfflineEvalModuleConfig
from config.networks.NetworkConfig import TokenizerModuleConfig
from config.networks.NetworkConfig import EncoderModuleConfig, TokenizerModuleConfig
from config.networks.NetworkConfig import DecoderModuleConfig
from config.MyMVWSLConfig import TrainingConfig
from config.MyMVWSLConfig import MAETrainingConfig
from config.networks.ConfigTextNetworks import (
    BiomedBertEncoderModuleConfig,
    BiomedBertDecoderModuleConfig,
    BiomedBertTokenizerModuleConfig,
    BertEncoderModuleConfig,
    BertDecoderModuleConfig,
    BertTokenizerModuleConfig,
    BertLargeEncoderModuleConfig,
    BertLargeDecoderModuleConfig,
    BertLargeTokenizerModuleConfig,
)

# EDIT
from config.chagas_config_data import ChagasConfigData
from config.chagas_config_training import ChagasConfigTraining
from config.chagas_config_st_mem import ChagasConfigStMem

from omegaconf import MISSING
from utils.generic import (
    get_text_modalities,
    has_text_modality,
    is_text_modality,
    remove_text_modalities_from_list,
)

def resolve_clf(ft_strategy: str) -> str:
    mapping = {
        "frontal": "classifiers.Classifiers.CLFTeset",
        "lateral": "classifiers.Classifiers.CLFTeset",
        "ensemble": "classifiers.Classifiers.CLFTeset"
    }
    if ft_strategy not in mapping:
        raise ValueError(f"Unknown strategy: {ft_strategy}")
    return mapping[ft_strategy]

def get_encoders(
    encoders: EncoderModuleConfig,
    text_encoder: EncoderModuleConfig,
    modality_names: List[str],
    share_weights: bool,
    share_weights_text: bool,
    compute_rec: bool = True,
) -> nn.ModuleDict:

    enc_modules = nn.ModuleDict()
    m_names = modality_names.copy()

    if has_text_modality(m_names) and text_encoder is not None:
        if not share_weights_text:
            for mod in get_text_modalities(m_names):
                enc_modules.add_module(
                    mod, instantiate(text_encoder, compute_rec=compute_rec)
                )
            remove_text_modalities_from_list(m_names)
        else:
            text_enc = instantiate(text_encoder, compute_rec=compute_rec)
            for mod in get_text_modalities(m_names):
                enc_modules.add_module(mod, text_enc)
            remove_text_modalities_from_list(m_names)
    else:
        warnings.warn(
            "Instantiating encoder modules without text modality", UserWarning
        )

    # share weights is done only across Vision encoders.
    if share_weights:
        enc_mod = instantiate(encoders)
        for mod in m_names:
            enc_modules.add_module(mod, enc_mod)
    else:
        for mod in m_names:
            enc_modules.add_module(mod, instantiate(encoders))

    return enc_modules


def get_decoders(
    decoders: DecoderModuleConfig,
    text_decoder: DecoderModuleConfig,
    modality_names: List[str],
    share_weights: bool,
    share_weights_text: bool,
) -> nn.ModuleDict:
    dec_modules = nn.ModuleDict()
    m_names = modality_names.copy()

    if has_text_modality(m_names) and text_decoder is not None:
        if not share_weights_text:
            for mod in get_text_modalities(m_names):
                dec_modules.add_module(mod, instantiate(text_decoder))
            remove_text_modalities_from_list(m_names)
        else:
            text_enc = instantiate(text_decoder)
            for mod in get_text_modalities(m_names):
                dec_modules.add_module(mod, text_enc)
            remove_text_modalities_from_list(m_names)
    else:
        warnings.warn(
            "Instantiating encoder modules without text modality", UserWarning
        )

    if share_weights:
        dec_mod = instantiate(decoders)
        for mod in m_names:
            dec_modules.add_module(mod, dec_mod)
    else:

        for mod in m_names:
            dec_modules.add_module(mod, instantiate(decoders))
    return dec_modules


def load_config():
    cs = ConfigStore.instance()
    # Registering the Config class with the name 'config'.

    cs.store(group="log", name="log", node=LogConfig)
    cs.store(group="model", name="mae_independent", node=MAEIndependentModelConfig)
    cs.store(group="model", name="mae_mmvm", node=MAEMMVMModelConfig)
    cs.store(group="model", name="mae_joint", node=MAEAggregationModelConfig)
    cs.store(
        group="modelmodule", name="mae_agg_module", node=MAEAggregationModelModuleConfig
    )
    cs.store(
        group="modelmodule",
        name="mae_independent_module",
        node=MAEIndependentModelModuleConfig,
    )
    cs.store(group="modelmodule", name="mae_mmvm_module", node=MAEMMVMModelModuleConfig)
    cs.store(
        group="encodermodule",
        name="vit_tiny_encoder_module",
        node=ViTtinyEncoderModuleConfig,
    )
    cs.store(
        group="decodermodule",
        name="vit_tiny_decoder_module",
        node=ViTtinyDecoderModuleConfig,
    )
    cs.store(
        group="encodermodule",
        name="vit_base_encoder_module",
        node=ViTbaseEncoderModuleConfig,
    )
    cs.store(
        group="decodermodule",
        name="vit_base_decoder_module",
        node=ViTbaseDecoderModuleConfig,
    )
    cs.store(
        group="encodermodule",
        name="vit_large_encoder_module",
        node=ViTlargeEncoderModuleConfig,
    )
    cs.store(
        group="decodermodule",
        name="vit_large_decoder_module",
        node=ViTlargeDecoderModuleConfig,
    )
    cs.store(
        group="textencodermodule",
        name="biomed_bert_encoder_module",
        node=BiomedBertEncoderModuleConfig,
    )
    cs.store(
        group="textdecodermodule",
        name="biomed_bert_decoder_module",
        node=BiomedBertDecoderModuleConfig,
    )
    cs.store(
        group="tokenizermodule",
        name="biomed_bert_tokenizer_module",
        node=BiomedBertTokenizerModuleConfig,
    )
    cs.store(
        group="textencodermodule",
        name="bert_encoder_module",
        node=BertEncoderModuleConfig,
    )
    cs.store(
        group="textdecodermodule",
        name="bert_decoder_module",
        node=BertDecoderModuleConfig,
    )
    cs.store(
        group="tokenizermodule",
        name="bert_tokenizer_module",
        node=BertTokenizerModuleConfig,
    )
    cs.store(
        group="textencodermodule",
        name="bert_large_encoder_module",
        node=BertLargeEncoderModuleConfig,
    )
    cs.store(
        group="textdecodermodule",
        name="bert_large_decoder_module",
        node=BertLargeDecoderModuleConfig,
    )
    cs.store(
        group="tokenizermodule",
        name="bert_large_tokenizer_module",
        node=BertLargeTokenizerModuleConfig,
    )
    cs.store(group="annealing", name="sigmoid", node=SigmoidAnnealingConfig)
    cs.store(group="annealing", name="cosine", node=CosAnnealingConfig)
    cs.store(group="annealing", name="exp", node=ExpAnnealingConfig)
    cs.store(group="annealing", name="linear", node=LinearAnnealingConfig)
    cs.store(group="annealing", name="noannealing", node=NoAnnealingConfig)
    cs.store(
        group="annealingmodule", name="sigmoidmodule", node=SigmoidAnnealingModuleConfig
    )
    cs.store(group="annealingmodule", name="cosmodule", node=CosAnnealingModuleConfig)
    cs.store(group="annealingmodule", name="expmodule", node=ExpAnnealingModuleConfig)
    cs.store(
        group="annealingmodule", name="linearmodule", node=LinearAnnealingModuleConfig
    )
    cs.store(
        group="annealingmodule", name="noannealingmodule", node=NoAnnealingModuleConfig
    )
    # cs.store(group="training", name="training", node=TrainingConfig)
    # cs.store(group="training", name="training_mae", node=MAETrainingConfig)

    cs.store(group="eval", name="PMEvalMAE", node=PolyMNISTEvalConfig)
    cs.store(group="eval", name="MimicEval", node=MimicEvalConfig)
    cs.store(group="offline_eval", name="MimicOfflineEval", node=MimicEvalOfflineConfig)
    cs.store(group="offline_evalmodule", name="mimic_frontal_clf", node=MimicFrontalClf)
    cs.store(group="offline_evalmodule", name="mimic_lateral_clf", node=MimicLateralClf)
    cs.store(group="offline_evalmodule", name="mimic_ensemble_clf", node=MimicEnsemblelClf)
    cs.store(
        group="evalmodule",
        name="PMsharedEvalMAEModule",
        node=PMsharedEvalMAEModuleConfig,
    )
    cs.store(group="evalmodule", name="PMMSEvalMAEModule", node=PMmsEvalMAEModuleConfig)
    cs.store(group="evalmodule", name="MimicEvalModule", node=MimicVAEEvalModuleConfig)
    cs.store(
        group="evalmodule", name="MimicMAEEvalModule", node=MimicMAEEvalModuleConfig
    )
    # cs.store(group="evalmodule", name="CelebAEvalModule", node=CelebAEvalModuleConfig)

    cs.store(group="dataset", name="PMvanilla", node=PMvanillaDataConfig)
    cs.store(group="dataset", name="PMtranslated50", node=PMtranslatedData50Config)
    cs.store(group="dataset", name="PMtranslated50ms", node=PMtranslatedData50MSConfig)
    cs.store(group="dataset", name="PMtranslated55", node=PMtranslatedData55Config)
    cs.store(group="dataset", name="PMtranslated60", node=PMtranslatedData60Config)
    cs.store(group="dataset", name="PMtranslated65", node=PMtranslatedData65Config)
    cs.store(group="dataset", name="PMtranslated70", node=PMtranslatedData70Config)
    cs.store(group="dataset", name="PMtranslated75", node=PMtranslatedData75Config)
    cs.store(
        group="dataset", name="PMtranslated50fixed", node=PMtranslatedData50FixedConfig
    )
    cs.store(group="dataset", name="PMrotated", node=PMrotatedDataConfig)
    cs.store(group="dataset", name="Mimic_cxr", node=MimicCXRDataConfig)
    cs.store(
        group="datamodule_train", name="PMTrainDataModule", node=PMTrainDataModuleConfig
    )
    cs.store(
        group="datamodule_eval", name="PMEvalDataModule", node=PMEvalDataModuleConfig
    )
    cs.store(
        group="datamodule_test", name="PMTestDataModule", node=PMTestDataModuleConfig
    )
    cs.store(
        group="datamodule_train",
        name="MimicTrainDataModule",
        node=MimicCXRFolderTrainModuleConfig,
    )
    cs.store(
        group="datamodule_eval",
        name="MimicEvalDataModule",
        node=MimicCXRFolderEvalModuleConfig,
    )
    cs.store(
        group="datamodule_test",
        name="MimicTestDataModule",
        node=MimicCXRFolderTestModuleConfig,
    )
    cs.store(
        group="transformmodule",
        name="transformmodule_pm",
        node=TransformPolyMNISTModule,
    )
    cs.store(
        group="transformmodule",
        name="transformmodule_mimic_dask_basic",
        node=TransformMimicDaskBasicModule,
    )
    cs.store(
        group="transformmodule",
        name="transformmodule_mimic_folder_basic",
        node=TransformMimicFolderBasicModule,
    )
    cs.store(
        group="transformmodule",
        name="transformmodule_mimic_folder_mae",
        node=TransformMimicFolderMAEModule,
    )
    cs.store(name="base_config", node=MyMVWSLConfig)

    # EDIT: this is how we tell hydra which classes we mean by the strings we have configured in ecg_chagas.yaml
    cs.store(group="dataset", name="chagas_config_data", node=ChagasConfigData)
    cs.store(group="training", name="chagas_config_training", node=ChagasConfigTraining)
    cs.store(group="model", name="st_mem", node=ChagasConfigStMem)