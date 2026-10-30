import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf, DictConfig
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
import torch
from utils import dataset
from config.MyMVWSLConfig import MyMVWSLConfig
from config.MyMVWSLConfig import LogConfig
from config.ModelConfig import JointModelConfig
from config.ModelConfig import MixedPriorModelConfig
from config.ModelConfig import UnimodalModelConfig
from config.ModelConfig import MixedPriorStdNormModelConfig
from config.DatasetConfig import MimicCXRDataConfig
from config.MyMVWSLConfig import EvalConfig
from mv_vaes.mv_mixedprior_vae import MVMixedPriorVAE as MVMixedPriorVAE
from mv_vaes.mv_unimodal_vae import MVunimodalVAE, MVVAE
import statistics

import gc
import numpy as np

cs = ConfigStore.instance()
# Registering the Config class with the name 'config'.

cs.store(group="log", name="log", node=LogConfig)
cs.store(group="model", name="joint", node=JointModelConfig)
cs.store(group="model", name="mixedprior", node=MixedPriorModelConfig)
cs.store(group="model", name="unimodal", node=UnimodalModelConfig)
cs.store(group="model", name="mixedpriorstdnorm", node=MixedPriorStdNormModelConfig)
cs.store(group="eval", name="eval", node=EvalConfig)
cs.store(group="dataset", name="Mimic_cxr", node=MimicCXRDataConfig)
cs.store(name="base_config", node=MyMVWSLConfig)
import numpy as np
import wandb
from utils.eval import (
    train_clfs_mimic,
    eval_clfs_mimic,
)



def prepare_embeddings(model_val, before_aggregation=True):
    # offline encodings calculation VAL SET
    enc_mu_out_val = {key: [] for key in model_val.modality_names}
    enc_mu_enc_val = {key: [] for key in model_val.modality_names}
    enc_lv_enc_val = {key: [] for key in model_val.modality_names}
    labels_val = []

    for idx, val_out in enumerate(model_val.offline_validation_step_ouputs):
        out, batch = val_out
        labels = batch
        dists_out = out[1]
        dists_enc = out[2]
        for m, key in enumerate(model_val.modality_names):
            mu_out_m, lv_out_m = dists_out[key]
            mu_enc_m, lv_enc_m = dists_enc[key]

            enc_mus_out_m = enc_mu_out_val[key]
            enc_mus_enc_m = enc_mu_enc_val[key]
            enc_lvs_enc_m = enc_lv_enc_val[key]

            enc_mus_out_m.append(mu_out_m)
            enc_mus_enc_m.append(mu_enc_m)
            enc_lvs_enc_m.append(lv_enc_m)

            enc_mu_out_val[key] = enc_mus_out_m
            enc_mu_enc_val[key] = enc_mus_enc_m
            enc_lv_enc_val[key] = enc_lvs_enc_m

        labels_val.append(labels)
    for m, key in enumerate(model_val.modality_names):
        enc_mu_out_m_val = enc_mu_out_val[key]
        enc_mu_out_m_val = torch.cat(enc_mu_out_m_val, dim=0)
        enc_mu_out_val[key] = enc_mu_out_m_val

        enc_mu_enc_m_val = enc_mu_enc_val[key]
        enc_mu_enc_m_val = torch.cat(enc_mu_enc_m_val, dim=0)
        enc_mu_enc_val[key] = enc_mu_enc_m_val

        enc_lv_enc_m_val = enc_lv_enc_val[key]
        enc_lv_enc_m_val = torch.cat(enc_lv_enc_m_val, dim=0)
        enc_lv_enc_val[key] = enc_lv_enc_m_val

    labels_val = torch.cat(labels_val, dim=0)
    if before_aggregation:
        return enc_mu_enc_val, labels_val, enc_lv_enc_val


@hydra.main(version_base=None, config_path="conf", config_name="config")
def run_experiment(cfg: MyMVWSLConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    # sample gradients at random from normal distribution
    # log example

    pl.seed_everything(cfg.seed, workers=True)

    # get data loaders
    train_loader, train_dst, val_loader, val_dst = dataset.get_dataset(cfg)

    wandb_logger = WandbLogger(
        name=cfg.log.wandb_run_name,
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
        project=cfg.log.wandb_project_name,
        group=cfg.log.wandb_group,
        offline=cfg.log.wandb_offline,
        entity=cfg.log.wandb_entity,
        save_dir=cfg.log.dir_logs,
    )
    trainer = pl.Trainer(
        max_epochs=cfg.model.epochs,
        devices=1,
        accelerator="gpu",
        logger=wandb_logger,
        check_val_every_n_epoch=1,
        deterministic=True,
    )

    # preload the model and generate the embeddings
    if cfg.eval.kl_model_type == 'unimodal':
        model_val = MVunimodalVAE.load_from_checkpoint(cfg.eval.kl_independent_trained_model_path, strict=False)
    elif cfg.eval.kl_model_type == 'mix':
        model_val = MVMixedPriorVAE.load_from_checkpoint(cfg.eval.kl_independent_trained_model_path, strict=False)
    else:
        raise NotImplementedError

    model_val.eval()
    trainer.validate(model_val, train_loader, verbose=True)

    # prepare embeddings
    enc_val, labels_val, enc_lv_val = prepare_embeddings(model_val)

    # CLEAN RAM
    del model_val
    gc.collect()

    # Frontal MU and Frontal LV

    frontal_mu = enc_val['frontal'][0:cfg.eval.num_enc_kl]
    frontal_lv = enc_lv_val['frontal'][0:cfg.eval.num_enc_kl]

    lateral_mu = enc_val['lateral'][0:cfg.eval.num_enc_kl]
    lateral_lv = enc_lv_val['lateral'][0:cfg.eval.num_enc_kl]

    # Compute KL
    # instantiate an empty model

    model = MVVAE(cfg)

    kls = []
    for i in range(0, len(frontal_mu)):
        print(i)
        print(frontal_mu[i])
        print(lateral_mu[i])
        print(frontal_lv[i])
        print(lateral_lv[i])
        if cfg.eval.kl_P_frontal == True:
            res = model.calc_kl_divergence(mu0=frontal_mu[i], logvar0=frontal_lv[i], mu1=lateral_mu[i], logvar1=lateral_lv[i])
        else:
            res = model.calc_kl_divergence(mu1=frontal_mu[i], logvar1=frontal_lv[i], mu0=lateral_mu[i],
                                           logvar0=lateral_lv[i])

        trainer.logger.experiment.log({"kl_value": res})
        kls.append(res.item())
        print(res.item())

    print("mean val")
    print(statistics.mean(kls))
    print("std val")
    print(statistics.stdev(kls))

    #log
    trainer.logger.experiment.log({"kl_mean":statistics.mean(kls)})

    trainer.logger.experiment.log({"kl_std":statistics.stdev(kls)})


if __name__ == "__main__":
    run_experiment()
