# wrapper around lightning trainer class that enables us to perform pretraining and fine-tuning steps (each of which is carried out via the lightning trainer)

from team.utils import create_model, get_datetime
from team.data import get_set, get_loader
from pytorch_lightning import Trainer as LightningTrainer
from pytorch_lightning.loggers import WandbLogger
from datetime import datetime
import numpy as np
from omegaconf import OmegaConf
import sys
import torch

class Trainer():
    def __init__(self, config):
        self.config = config

    # modes:
    # - full_data mode (i.e. one run using full data)
    # - optuna mode (i.e. run cross validated optuna-study)
    #
    # model saving:
    # since only in optuna mode a validation set is created we sometimes "abuse" optuna mode by running with only one set of hyperparameters and n_trials=1
    # in order to perform a training run with validation metrics. since we might also want to evaluate the resulting model we also need to save in optuna mode.
    def train(self, device, train_data_path, save_path, optuna=False):
        return_values = []

        num_runs = self.config.search.cross_validation_folds if optuna else 1

        for i in range(num_runs):
            date_time = get_datetime()
            # relative path with respect to python scripts in "code" (since these are the entry points of any execution)
            config_dict = OmegaConf.to_container(self.config, resolve=True)
            wandb_logger = WandbLogger(
                project="ecg_chagas",
                name=self.config.model.type + "_" + date_time,
                save_dir="./wandb_logs",
                offline=True,
                config=config_dict
            )

            model = create_model(self.config)

            seed = i if optuna else None  # do not split training data in case we are performing a full_data run 

            # PRE-TRAINING
            train_set, val_set = get_set(
                data_path=train_data_path, 
                limit=self.config.data.debug_limit, 
                mode=self.config.pretraining.data_mode, 
                target_sr=self.config.data.target_sr,
                preprocessing=self.config.data.preprocessing,
                int_norm=self.config.data.intensity_normalization, 
                standard_len=self.config.data.standard_len, 
                pad_mode=self.config.data.pad_mode,
                heartbeat_mode=self.config.data.heartbeat_mode,
                st_mem_processing=self.config.data.st_mem_processing,
                split_seed=seed)

            train_loader = get_loader(train_set, batch_size=self.config.model.batch_size, balanced=True)
            val_loader = get_loader(val_set, batch_size=1)  
            model.prepare_pretraining()
            # use large int to effectively disable lightnings automatic logging (becuase we are doing custom logging)
            trainer = LightningTrainer(max_epochs=self.config.pretraining.epochs, accelerator=device, logger=wandb_logger, enable_progress_bar=True, log_every_n_steps=sys.maxsize)
            trainer.fit(model, train_loader, val_loader)
            model.save(save_path + "pretrained.pth")
            print("pre-training completed.")

            # keep memory usage in check
            del train_set, val_set
            del train_loader, val_loader, trainer
            torch.cuda.empty_cache()

            # FINE-TUNING
            train_set, val_set = get_set(
                data_path=train_data_path, 
                limit=self.config.data.debug_limit, 
                mode=self.config.fine_tuning.data_mode, 
                target_sr=self.config.data.target_sr,
                preprocessing=self.config.data.preprocessing,
                int_norm=self.config.data.intensity_normalization, 
                standard_len=self.config.data.standard_len, 
                pad_mode=self.config.data.pad_mode,
                heartbeat_mode=self.config.data.heartbeat_mode,
                st_mem_processing=self.config.data.st_mem_processing,
                split_seed=seed)

            train_loader = get_loader(train_set, batch_size=self.config.model.batch_size, balanced=True)
            val_loader = get_loader(val_set, batch_size=1)  
            model.prepare_tuning()
            trainer = LightningTrainer(max_epochs=self.config.fine_tuning.epochs, accelerator=device, logger=wandb_logger, enable_progress_bar=True, log_every_n_steps=sys.maxsize)
            trainer.fit(model, train_loader, val_loader)
            model.save(save_path + "tuned.pth")
            print("fine-tuning completed.")

            # keep memory usage in check
            del train_set, val_set
            del train_loader, val_loader, trainer
            torch.cuda.empty_cache()

            return_values.append(model.min_val_loss)
        return np.average(return_values)  # in case we are not performing an optuna run this will be an average of one value (i.e. no averaging will happen)