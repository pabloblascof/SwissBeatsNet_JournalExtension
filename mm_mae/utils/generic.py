import torch
import pytorch_lightning as pl
from transformers import DataCollatorForLanguageModeling, AutoTokenizer
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.utilities.rank_zero import rank_zero_info
import os

# EDIT
import sys
sys.path.append("/local/home/elucas/ecg_chagas/code/")  
sys.path.append("/cluster/home/elucas/ecg_chagas/code/")
sys.path.append("/users/elucas/ecg_chagas/code/") 




from team.utils_circular import check_tensor

# EDIT 
# function that turns outputs of ECGDataset into the objects mm_mae code requires
def prepare_data(batch, modality_names):
    signals, labels, src, _ = batch

    check_tensor(signals, expected_shape=[None, None, 12], finite=True, name="signals")  # [batch_dim, signal_len, 12 leads]
    check_tensor(labels, expected_shape=[None, None], finite=True, name="labels")  # [batch_dim, num_labels (= 1 or 6 depending on if we are using chagas or non_chagas)]
    check_tensor(src, expected_shape=[None], finite=True, name="source")  # [batch_dim]

    # index out the i-th lead (don't touch batch_dim or time_dim). 
    # this removes the lead_dimension so unsqueeze it back since patchify einsum operation expects it.
    # finally permute dims into the arrangement expected by st_mem code.
    mods = {name: signals[:, :, i].unsqueeze(2).permute(0, 2, 1) for i, name in enumerate(modality_names)}  
    batch_size = signals.shape[0]
    mod_masks = {name: torch.ones(batch_size, dtype=torch.bool, device=signals.device) for name in modality_names}

    check_tensor(mods[modality_names[0]], expected_shape=[None, 1, None], finite=True, name="modality")  # [batch_dim, 1 lead, signal_len]
    check_tensor(mod_masks[modality_names[0]], expected_shape=[None], finite=True, name="modality presence mask")  # [batch_dim]

    return mods, labels, mod_masks, src

# EDIT
# tansform data structures that st_mem enc forward method returns into data structures that mm_mae enc would have returned
def transform_outputs(st_mem_enc_out):
    # [batch_dim, num_leads * num_patches_that_survived_masking, embedding_dim]
    check_tensor(st_mem_enc_out[0], expected_shape=[None, None, None], finite=True, name="latent embeddings st_mem") 
    # [batch_dim, num_leads, num_patches]
    check_tensor(st_mem_enc_out[1], expected_shape=[None, 1, None], finite=True, name="patch level mask") 
    # [batch_dim, num_leads, num_patches]
    check_tensor(st_mem_enc_out[2], expected_shape=[None, 1, None], finite=True, name="backward indices st_mem") 

    features = st_mem_enc_out[0].permute(1, 0, 2)
    indices = st_mem_enc_out[2].squeeze(1).permute(1, 0)

    # [num_leads * num_patches_that_survived_masking, batch_dim, embedding_dim]
    check_tensor(features, expected_shape=[None, None, None], finite=True, name="latent embeddings mm_mae") 
    # [num_patches, batch_dim]
    check_tensor(indices, expected_shape=[None, None], finite=True, name="backward indices mm_mae") 

    e_m_out = {
            # st_mem returns [batch_dim, num_patches, embedding_len] but mm_mae enc would have 
            # returned [num_patches, batch_dim, embedding_len] i.e. emulate that order to make
            # following mm_mae code work
            "features": features,
            # st_mem returns ids_restore in format [batch_size, num_leads (= 1 for us), num_patches] 
            # i.e. transform into format that mm_mae code expects (= [num_patches, batch_size])
            "backward_indexes": indices
            }
    return e_m_out


_SUPPORTED_TEXT_MODALITIES = [
    "findings",
    "impression",
    "impression_plus_findings"
]

class EpochPrefixCallback(pl.Callback):
    def __init__(self, prefix):
        self.prefix = prefix

    def on_train_epoch_end(self, trainer, pl_module):
        trainer.logger.log_metrics({f"{self.prefix}/epoch": trainer.current_epoch}, step=trainer.global_step)

    def on_validation_epoch_end(self, trainer, pl_module):
        trainer.logger.log_metrics({f"{self.prefix}/val_epoch": trainer.current_epoch}, step=trainer.global_step)

def check_cfg_logic(cfg):
    # Config logic checks
    if cfg.do_pretraining and not cfg.offline_eval.checkpoint_init:
        raise ValueError(
            "Configuration error: 'do_pretraining' is True but 'offline_eval.checkpoint_init' is False.\n"
            "To proceed, either:\n"
            "  - Set 'do_pretraining' to False if you want to train a supervised baseline, or\n"
            "  - Enable 'offline_eval.checkpoint_init' if you want to pretrain and/or evaluate a trained encoder."
        )
    if not cfg.do_pretraining and not cfg.do_offline_eval:
        raise ValueError(
            "Configuration error: Both 'do_pretraining' and 'do_offline_eval' are set to False.\n"
            "At least one of these options must be True to run the experiment."
        )


class CustomDataCollatorForLanguageModeling(DataCollatorForLanguageModeling):

    def __call__(self, features, **kwargs):
        # Ensure features is a list of tuples
        if not all(isinstance(feature, tuple) for feature in features):
            raise ValueError("Expected features to be a list of tuples")

        samples_list, labels_list, avail_mods_list = zip(*features)

        # Extract the additional keys of sample (non-related to text)
        additional_keys = {
            key: [sample[key] for sample in samples_list]
            for key in samples_list[0]
            if not is_text_modality(key)
        }

        data = {}
        # Process text features with the super LM collator from transformers
        for tm in get_text_modalities(samples_list[0].keys()):
            text_features = [sample[tm] for sample in samples_list]
            collated_text = super().__call__(text_features, **kwargs)
            data[tm] = collated_text

        # Add the additional keys of sample to the batch
        for key, value in additional_keys.items():
            data[key] = torch.stack(value)

        # Convert to dictionary of tensors
        avail_mods_tensor = {
            key: torch.tensor([d[key] for d in avail_mods_list])
            for key in avail_mods_list[0]
        }
        # todo assert check dimension of padded text here
        return [data, torch.stack(labels_list), avail_mods_tensor]


def get_data_loaders(
    modality_names,
    train_dst,
    val_dst,
    batch_size,
    num_workers,
    tokenizer_module,
):
    if has_text_modality(modality_names):
        train_loader = torch.utils.data.DataLoader(
            train_dst,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=16,
            drop_last=True,
            collate_fn=CustomDataCollatorForLanguageModeling(
                tokenizer=AutoTokenizer.from_pretrained(
                    pretrained_model_name_or_path=tokenizer_module.pretrained_model_name
                ),
                mlm=True,
                mlm_probability=tokenizer_module.mlm_probability,
            ),
        )

        val_loader = torch.utils.data.DataLoader(
            val_dst,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=16,
            drop_last=False,
            collate_fn=CustomDataCollatorForLanguageModeling(
                tokenizer=AutoTokenizer.from_pretrained(
                    pretrained_model_name_or_path=tokenizer_module.pretrained_model_name
                ),
                mlm=True,
                mlm_probability=tokenizer_module.mlm_probability,  # todo check mask settings here
            ),
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            train_dst,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=16,
            persistent_workers=True,
            drop_last=True,
        )
        val_loader = torch.utils.data.DataLoader(
            val_dst,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=16,
            drop_last=False,
        )

    return train_loader, val_loader


def move_tensors_to_cpu(data):
    if isinstance(data, torch.Tensor):
        return data.cpu()
    elif isinstance(data, dict):
        return {k: move_tensors_to_cpu(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [move_tensors_to_cpu(item) for item in data]
    elif isinstance(data, tuple):
        return tuple(move_tensors_to_cpu(item) for item in data)
    elif isinstance(data, set):
        return {move_tensors_to_cpu(item) for item in data}
    else:
        return data


def get_supported_text_modalities():
    # LIST THAT DEFINES THE MODALITIES OF THE TEXT DOMAIN ACCEPTED BY THE MODEL
    # MODALITIES WITH DIFFERENT NAMES ARE NOT PROCESSED AS TEXT
    return _SUPPORTED_TEXT_MODALITIES.copy()


def has_text_modality(modality_names):
    return any(modality in _SUPPORTED_TEXT_MODALITIES for modality in modality_names)


def is_text_modality(modality):
    return any(m == modality for m in _SUPPORTED_TEXT_MODALITIES)


def get_text_modalities(modality_names):
    return [value for value in modality_names if value in _SUPPORTED_TEXT_MODALITIES]


def remove_text_modalities_from_list(modality_names):
    for tm in _SUPPORTED_TEXT_MODALITIES:
        if tm in modality_names:
            modality_names.remove(tm)

class WandbCheckpointCallback(Callback):
    def __init__(self, wandb_logger, monitor="val_loss", mode="min"):
        self.wandb_logger = wandb_logger
        self.monitor = monitor
        self.mode = mode
        self._checkpoint_callback = None  # defer construction

    @rank_zero_only
    def _get_checkpoint_dir(self):
        exp = self.wandb_logger.experiment  # triggers wandb.init() safely on rank 0
        dir_attr = exp.dir() if callable(exp.dir) else exp.dir
        checkpoint_dir = os.path.join(dir_attr, "checkpoints")
        rank_zero_info(f"Using checkpoint dir: {checkpoint_dir}")
        return checkpoint_dir

    def on_fit_start(self, trainer, pl_module):
        checkpoint_dir = self._get_checkpoint_dir()

        # broadcast to all ranks
        if torch.distributed.is_initialized():
            dir_list = [checkpoint_dir]
            torch.distributed.broadcast_object_list(dir_list, src=0)
            checkpoint_dir = dir_list[0]

        # create and register actual checkpoint callback
        self._checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            monitor=self.monitor,
            mode=self.mode,
            save_last=True,
            save_on_train_epoch_end=False,
        )
        trainer.callbacks.append(self._checkpoint_callback)

