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


def custom_log(cfg, scores_m, wandb_run, modality_name='frontal', str_ds='before_aggregation'):
    key = modality_name
    label_names = cfg.dataset.target_list
    # scores_m : RF -> AUROC -> Label

    for clf_type in cfg.eval.classifier_list:
        for metric in cfg.eval.metric_list:
            wandb_run.log(
                {f"val/downstream/{str_ds}/{key}/{clf_type}/{metric}": scores_m[clf_type][metric].mean()}
            )

            for k, l_name in enumerate(label_names):
                wandb_run.log(
                    {f"val/downstream/{str_ds}/{key}/{clf_type}/{metric}/{l_name}": scores_m[clf_type][metric][k]}
                )


def prepare_embeddings(model_train, model_val, before_aggregation=True):
    # offline encodings calculation TRAIN SET
    enc_mu_out_train = {key: [] for key in model_train.modality_names}
    enc_mu_enc_train = {key: [] for key in model_train.modality_names}
    labels_train = []

    for idx, val_out in enumerate(model_train.offline_validation_step_ouputs):
        out, batch = val_out
        labels = batch
        dists_out = out[1]
        dists_enc = out[2]
        for m, key in enumerate(model_train.modality_names):
            mu_out_m, lv_out_m = dists_out[key]
            mu_enc_m, lv_enc_m = dists_enc[key]
            enc_mus_out_m = enc_mu_out_train[key]
            enc_mus_enc_m = enc_mu_enc_train[key]
            enc_mus_out_m.append(mu_out_m)
            enc_mus_enc_m.append(mu_enc_m)
            enc_mu_out_train[key] = enc_mus_out_m
            enc_mu_enc_train[key] = enc_mus_enc_m
        labels_train.append(labels)
    for m, key in enumerate(model_train.modality_names):
        enc_mu_out_m_train = enc_mu_out_train[key]
        enc_mu_out_m_train = torch.cat(enc_mu_out_m_train, dim=0)
        enc_mu_out_train[key] = enc_mu_out_m_train
        enc_mu_enc_m_train = enc_mu_enc_train[key]
        enc_mu_enc_m_train = torch.cat(enc_mu_enc_m_train, dim=0)
        enc_mu_enc_train[key] = enc_mu_enc_m_train
    labels_train = torch.cat(labels_train, dim=0)

    # offline encodings calculation VAL SET
    enc_mu_out_val = {key: [] for key in model_val.modality_names}
    enc_mu_enc_val = {key: [] for key in model_val.modality_names}
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
            enc_mus_out_m.append(mu_out_m)
            enc_mus_enc_m.append(mu_enc_m)
            enc_mu_out_val[key] = enc_mus_out_m
            enc_mu_enc_val[key] = enc_mus_enc_m
        labels_val.append(labels)
    for m, key in enumerate(model_val.modality_names):
        enc_mu_out_m_val = enc_mu_out_val[key]
        enc_mu_out_m_val = torch.cat(enc_mu_out_m_val, dim=0)
        enc_mu_out_val[key] = enc_mu_out_m_val
        enc_mu_enc_m_val = enc_mu_enc_val[key]
        enc_mu_enc_m_val = torch.cat(enc_mu_enc_m_val, dim=0)
        enc_mu_enc_val[key] = enc_mu_enc_m_val
    labels_val = torch.cat(labels_val, dim=0)
    if before_aggregation:
        return enc_mu_enc_train, labels_train, enc_mu_enc_val, labels_val


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
        accelerator="gpu",  # todo generalize for sciCORE
        logger=wandb_logger,
        check_val_every_n_epoch=1,
        deterministic=True,
    )

    # preload the model and generate the embeddings
    model_train = MVMixedPriorVAE.load_from_checkpoint(cfg.eval.trained_model_path, strict=False)
    model_val = MVMixedPriorVAE.load_from_checkpoint(cfg.eval.trained_model_path, strict=False)

    #force offline eval in model_train and model_val
    model_train.offline_eval = True
    model_val.offline_eval = True

    model_train.eval()
    trainer.validate(model_train, train_loader, verbose=True)
    model_val.eval()
    trainer.validate(model_val, val_loader, verbose=True)


    # prepare embeddings
    enc_train, labels_train, enc_val, labels_val = prepare_embeddings(model_train, model_val)

    # CLEAN RAM
    del model_train, model_val
    gc.collect()

    # train clfs modality
    train_encodings = enc_train[cfg.eval.modality_eval][0:int(cfg.eval.num_enc_clf)]
    train_labels = labels_train[0:int(cfg.eval.num_enc_clf)]
    clf_out_m = train_clfs_mimic(train_encodings, train_labels, clfs=cfg.eval.classifier_list, cfg=cfg,
                                 logger=trainer.logger.experiment)
    val_encodings = enc_val[cfg.eval.modality_eval]
    val_labels = labels_val
    out_numbers = eval_clfs_mimic(clf_out_m, val_encodings, val_labels, metrics=cfg.eval.metric_list)
    print(out_numbers)
    custom_log(cfg, out_numbers, trainer.logger.experiment, modality_name=cfg.eval.modality_eval)


if __name__ == "__main__":
    run_experiment()
