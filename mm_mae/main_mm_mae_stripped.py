import os

# Set the environment variable
os.environ["HYDRA_FULL_ERROR"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # todo check if we can improve here
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.model_summary import ModelSummary
from pytorch_lightning.callbacks import LearningRateMonitor
import wandb
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from hydra.utils import instantiate
from utils.config_setup import load_config, resolve_clf
from utils.generic import check_cfg_logic, get_data_loaders, WandbCheckpointCallback, EpochPrefixCallback
from utils.config_setup import get_encoders
from utils.config_setup import get_decoders
from config.MyMVWSLConfig import MyMVWSLConfig

# EDIT
import sys  
from team.data import get_set, get_loader
from team.utils import check_config, get_datetime, merge_configs, print_mem, load_mmmae_config_structure, load_split_pth
from omegaconf import OmegaConf
from team.utils_circular import LoggingWrapper
from torch import nn
from networks.st_mem import ST_MEM  # EDIT
from pytorch_lightning.callbacks import ModelCheckpoint  # EDIT
import random
import numpy as np
from pathlib import Path

def main(config_path, run_model_func, output_dir, train_data_path, encoder_path):

    # ensure training does not crash in the end due to missing dir
    assert os.path.isdir(output_dir), f"output directory does not exist: {output_dir}"

    cfg = load_mmmae_config_structure(config_path, output_dir, train_data_path, encoder_path)

    # Resolve and check the config
    print(OmegaConf.to_yaml(cfg, resolve=True))
    cfg = OmegaConf.structured(cfg)
    check_cfg_logic(cfg)
    
    cfg = check_config(cfg)  # need to manually call this method in order to transform the necessary strings into enum-items

    set_seeds(cfg.seed)

    model = build_model(cfg)

    assert model is not None
    summary = ModelSummary(model, max_depth=2)
    print(summary)

    set_seeds(cfg.seed)
    
    # wandb
    if cfg.log.wandb_local_instance:
        wandb.login(host=os.getenv("WANDB_LOCAL_URL"))
    elif not cfg.log.wandb_offline:
        # EDIT: comment out and pass because we are not using online mode i.e. don't need to log in 
        # wandb.login(host="https://api.wandb.ai")
        pass

    wandb_logger = WandbLogger(
        name=cfg.log.wandb_run_name,
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False),  # TODO: set True and fix
        project=cfg.log.wandb_project_name,
        group=cfg.log.wandb_group,
        offline=True,  # don't go online 
        entity=cfg.log.wandb_entity,
        save_dir=output_dir  # save to output dir (as suggested by challenge team)
    )

    team_logger = LoggingWrapper(wandb_logger, ["training", "validation"], cfg.training.binarization_threshold, logger_prefix="team_")
    
    set_seeds(cfg.seed)

    model.team_logger = team_logger
    
    lr_monitor = LearningRateMonitor(logging_interval="step")
    pretrain_epoch_cb = EpochPrefixCallback("pretrain")
    offline_eval_epoch_cb = EpochPrefixCallback("offline_eval")

    print_mem("starting training")
    
    ################################
    #########  LOAD MODEL  #########
    ################################

    set_seeds(cfg.seed)

    state_dict = load_split_pth(encoder_path)["state_dict"]
    model.load_state_dict(state_dict) 

    ###############################
    ########  FINE-TUNING  ########
    ###############################

    set_seeds(cfg.seed)

    # create new loaders for downstream training (as we need chagas labels in this step)
    train_dst, val_dst = get_set(
        data_path=train_data_path,
        limit=cfg.dataset.debug_limit, 
        mode=cfg.training.fine_tuning_data_mode, 
        target_sr=cfg.dataset.target_sr,
        preprocessing=cfg.dataset.preprocessing,
        int_norm=cfg.dataset.intensity_normalization, 
        standard_len=cfg.dataset.standard_len, 
        pad_mode=cfg.dataset.pad_mode,
        heartbeat_mode=cfg.dataset.heartbeat_mode,
        st_mem_processing=cfg.dataset.st_mem_processing,
        lazy=cfg.dataset.lazy,
        num_gpus=cfg.training.num_gpus)
    train_loader = get_loader(train_dst, batch_size=cfg.training.batch_size, balanced=True)
    val_loader = get_loader(val_dst, batch_size=cfg.training.batch_size)

    set_seeds(cfg.seed)
    
    # Offline Eval Phase
    if cfg.do_offline_eval:
        pl.seed_everything(cfg.offline_eval.seed, workers=True)
        # TODO: currently implemented only for vision modality
        print("Starting offline evaluation")
        if cfg.do_pretraining:
            print("Using in-memory encoders")
        elif cfg.offline_eval.checkpoint_init:
            print("Loading encoders from checkpoint.")
            assert os.path.exists(cfg.offline_eval.ckpt_path), f"Checkpoint not found: {cfg.offline_eval.ckpt_path}"
            state_dict = torch.load(cfg.offline_eval.ckpt_path, weights_only=False)["state_dict"]
            model.load_state_dict(state_dict, strict=False)
        else:
            print("Using randomly initialized encoders for supervised baseline.")


        clf = build_clf(cfg, model.encoders)
        clf.team_logger = team_logger

        team_logger.reset_step()  # team logger has been used during pre-training i.e. need to reset step 

        # Train the classifier
        trainer = pl.Trainer(
            max_epochs=1,
            devices=cfg.offline_evalmodule.oe_num_gpus,
            log_every_n_steps=cfg.log.wandb_log_freq,
            num_nodes=cfg.offline_evalmodule.oe_num_nodes,
            accelerator="gpu",
            strategy=cfg.training.strategy,
            logger=wandb_logger,
            deterministic=True,
            callbacks=[lr_monitor, offline_eval_epoch_cb],
        )
        trainer.fit(clf, train_dataloaders=train_loader, val_dataloaders=val_loader)

    torch.save(model.state_dict(), output_dir + "/model.pth")
    torch.save(clf.state_dict(), output_dir + "/clf.pth")

    wandb.finish()

# returns a fresh instance of mm_mae model
def build_model(cfg):
    enc_modules = nn.ModuleDict()
    dec_modules = nn.ModuleDict()
    # one enc/dec that is re-used for all leads
    if cfg.model.share_weights:  
        model = init_model(cfg)
        for mod in cfg.dataset.modality_names:
            enc_modules.add_module(mod, model)
            dec_modules.add_module(mod, None)  # ST_MEM instance contains enc and dec i.e. add None as dec
    # one enc/dec per lead
    else:   
        for mod in cfg.dataset.modality_names:
            enc_modules.add_module(mod, init_model(cfg))
            dec_modules.add_module(mod, None)  
    encoders = enc_modules
    decoders = dec_modules

    evaluator = instantiate(cfg.evalmodule)
    beta_annealing = (instantiate(cfg.annealingmodule) if cfg.annealingmodule is not None else None)
    
    # Init model
    model = instantiate(cfg.modelmodule, encoders, decoders, evaluator, beta_annealing)

    return model

def build_clf(cfg, encs):
    clf = instantiate(cfg.offline_evalmodule, oe_encoders = encs, oe_learning_rate = cfg.downstream_base_lr)
    return clf

def set_seeds(seed: int):
    pl.seed_everything(seed, workers=True)  

    random.seed(seed)                
    np.random.seed(seed)             
    torch.manual_seed(seed)          
    torch.cuda.manual_seed(seed)     
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.enabled = False

    os.environ["PYTHONHASHSEED"] = str(seed)

    print("SEEDS SET")


# initialize fresh st_mem instance with the vit parameters from cfg.encodermodule and cfg.decodermodule
def init_model(cfg):
    return ST_MEM(
        patch_size=cfg.encodermodule.patch_size,
        num_leads=1,
        embed_dim=cfg.encodermodule.emb_dim,
        depth=cfg.encodermodule.n_enc_layers,
        num_heads=cfg.encodermodule.n_enc_heads,
        decoder_embed_dim=cfg.decodermodule.emb_dim,
        decoder_depth=cfg.decodermodule.n_dec_layers,
        decoder_num_heads=cfg.decodermodule.n_dec_heads,
        mlp_ratio=cfg.encodermodule.mlp_ratio,
    )