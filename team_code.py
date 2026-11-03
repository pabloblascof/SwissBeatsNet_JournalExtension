#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Optional libraries, functions, and variables. You can change or remove them.
#
################################################################################

import os

# Submission build (PROVENANCE.md): the same settings as the Dockerfile's ENV lines, applied
# here too so that a run without the image environment still reaches the registered model.
os.environ.setdefault("SBN_SUBMISSION_MODE", "registered_folder")
os.environ.setdefault("SBN_REGISTERED_PINS", "configs/registered_model_pins_v2c.json")

from team.utils import setup, create_model, load_config, PROFILE, profile, load_split_pth, load_mmmae_config_structure
from team.utils_challenge import load_signals, load_header
from team.utils_circular import check_tensor, prob_to_class
from team.data import load_source, load_and_process_signal, load_sampling_rate
from team.trainer import Trainer
import torch
import os
import subprocess
import sys
import yaml
from team.st_mem_processing import build_st_mem_transforms

curr_dir = os.path.dirname(os.path.abspath(__file__))

# make mm_mae imports work
sys.path.append(curr_dir + "/mm_mae/")
# make `src.*` imports work regardless of the working directory (e.g. /challenge in the container)
if curr_dir not in sys.path:
    sys.path.insert(0, curr_dir)
from main_mm_mae_stripped import main, build_model, build_clf
from main_mm_mae import main as main_internal

# Journal-branch model path (PLAN.md S7.2/S7.4). A model folder that contains
# journal_manifest.json is served by src.container.journal_model; any other
# folder goes through the unchanged CinC anchor path below. Guarded so the
# anchor keeps working even if src/ is absent.
try:
    from src.container.journal_model import JournalModel, MANIFEST_NAME as JOURNAL_MANIFEST_NAME
except Exception as _journal_import_error:  # pragma: no cover
    JournalModel = None
    JOURNAL_MANIFEST_NAME = "journal_manifest.json"
    print(f"WARNING: journal model path unavailable ({_journal_import_error}); anchor path only", file=sys.stderr)

config_path = "chagas"  # relative to utils.py (since it is used there)
encoder_path = curr_dir + "/mm_mae/pretrained_enc/good_disco"

# D-053 opt-in flag (src/container/submission.py). Unset or empty -- the default --
# keeps every function below on the CinC anchor behaviour, and src.container.submission
# is never imported. "train_registered_recipe" (option b) or "registered_folder"
# (option c) route train_model / an empty load_model folder to the registered model.
SUBMISSION_MODE_ENV = "SBN_SUBMISSION_MODE"


def _submission_mode():
    if not os.environ.get(SUBMISSION_MODE_ENV, "").strip():
        return None
    from src.container import submission
    return submission.active_mode()  # validates the value; None for "anchor"

################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments for the functions.
#
################################################################################

# Train your models. This function is *required*. You should edit this function to add your code, but do *not* change the arguments
# of this function. If you do not train one of the models, then you can return None for the model.

# Train your model.
def train_model(data_folder, model_folder, verbose):
    mode = _submission_mode()
    if mode is not None:  # D-053 options (b)/(c); off by default
        from src.container import submission
        submission.train_model(mode, data_folder, model_folder, verbose)
        return load_model(model_folder, verbose=False)

    main(
        config_path=config_path,
        run_model_func=run_model,
        output_dir=model_folder,
        train_data_path=data_folder,
        encoder_path=encoder_path,  # do not have this to run full training
    )

    return load_model(model_folder, verbose=False)

# Train your model.
def train_model_parameterized(data_folder, model_folder, verbose, config_path_arg, enc_path_arg):
    main_internal(
        config_path=config_path_arg,
        run_model_func=run_model,
        output_dir=model_folder,
        train_data_path=data_folder,
        encoder_path= enc_path_arg,  # do not have this to run full training
    )

    return load_model(model_folder, verbose=False)

# Load your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function. If you do not train one of the models, then you can return None for the model.
def load_model(model_folder, verbose):

    # Submission build (PROVENANCE.md): only the registered model_v2c is served, verified
    # against its pins. Whatever the model folder holds (the copy train_model.py made, nothing,
    # or files from an earlier run such as model.pth) never selects another model.
    if _submission_mode() == "registered_folder":
        from src.container import submission
        return submission.load_registered_model(verbose)

    # Journal model folder -> JournalModel (see src/container/journal_model.py)
    if os.path.exists(os.path.join(model_folder, JOURNAL_MANIFEST_NAME)):
        if JournalModel is None:
            raise RuntimeError("model folder has a journal manifest but src.container could not be imported")
        if verbose:
            print(f"Loading journal model from {model_folder}...")
        return JournalModel.load(model_folder)

    # D-053 option (c), off by default: a folder with neither a journal manifest nor
    # model.pth (a fresh /challenge/model mount, run-only) -> the verified registered model.
    if _submission_mode() == "registered_folder":
        from src.container import submission
        registered = submission.load_for_registered_mode(model_folder, verbose)
        if registered is not None:
            return registered

    # Load config
    config = load_mmmae_config_structure(config_path, model_folder, "", "")

    model = build_model(config)
    model.load_state_dict(torch.load(model_folder + "/model.pth")) 

    clf = build_clf(config, model.encoders)
    clf.load_state_dict(torch.load(model_folder + "/clf.pth")) 

    return model, clf

# Run your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def run_model(record, model, verbose):

    # Journal model: probability = sigmoid(logit) (D-010), label from the manifest rule.
    # Batch mode when `record` is a list (run_model_optimized.py), single otherwise (run_model.py).
    if JournalModel is not None and isinstance(model, JournalModel):
        if isinstance(record, list):
            class_label, probability = model.predict_challenge(record)
            assert isinstance(probability[0], float)
            assert isinstance(class_label[0], int)
        else:
            class_label, probability = model.predict_record(record)
            assert isinstance(probability, float)
            assert isinstance(class_label, int)
        return class_label, probability  # DO NOT return tensor (else multiprocesing queue crashes)

    # sometimes the model arg is a tuple (enc/dec, clf) - in which case we only need the classifier 
    if isinstance(model, tuple):
            model = model[1]

    ################################################################
    ################################################################
    
    # the callenge tables that we use for model selection are 
    # created at the end of validation epochs
    # (we also create tables at end of train epochs but those tables
    # are not really used for model selection as they are on
    # train data which the model could have overfitted to)
    # and there model is in eval mode i.e. also put model in eval
    # mode in order to be able to reproduce performance that was
    # observed in challenge table (which we use for model selection)

    model.eval()

    ################################################################
    ################################################################




    # BATCH_MODE: in our modified run_model.py we use batching for performance reasons
    # SINGLE_RECORD_MODE: run_model still needs to support the case when record is not a list 
    batch_mode = True if isinstance(record, list) else False

    config = load_mmmae_config_structure(config_path, "", "", "")

    signal = None

    with profile("prepare"):
        if batch_mode: 
            signals = [prepare_data(r, config, model) for r in record]
            signal = torch.stack(signals)
        else:  
            signal = prepare_data(record, config, model)
            # all models expect a batch_dimension (since their training is always performed in a batched fashion)
            signal = signal.unsqueeze(0)  

    # classifier forward expects 4 inputs - but since only signals is needed we can add dummy data
    inputs = (
        signal,
        torch.zeros(1, 1),
        torch.zeros(1),
        None) 

    with torch.no_grad():
        with profile("forward"):
            probability = model.forward(inputs)[0]

    with profile("post"):
        if batch_mode:
            probability = [probability[i].item() for i in range(len(record))]   
            class_label = [prob_to_class(probability[i], config.training.binarization_threshold) for i in range(len(record))]
             
        else:
            probability = probability.squeeze(0).item()  
            class_label = prob_to_class(probability, config.training.binarization_threshold) 

    with profile("assert"):
            # we have written the incorrect types to txt files multiple times now (leading to bad challenge score) so catch that error early
            if batch_mode:
                assert isinstance(probability[0], float)
                assert isinstance(class_label[0], int)  
            else:
                assert isinstance(probability, float)
                assert isinstance(class_label, int)

    return class_label, probability  # DO NOT return tensor (else multiprocesing queue crashes)

# prepare one record
def prepare_data(record, config, model):
    header = load_header(record)
    sr = load_sampling_rate(header)

    # build st_mem transforms
    # TODO: maybe we can cache this in a global variable
    _, st_mem_transforms_eval = build_st_mem_transforms(config.dataset.st_mem_processing)

    # preprocess in the same way that training data was preprocessed
    signal = load_and_process_signal("", 
                                    record, 
                                    sr, 
                                    config.dataset.target_sr,
                                    config.dataset.preprocessing,
                                    config.dataset.intensity_normalization, 
                                    config.dataset.standard_len, 
                                    config.dataset.pad_mode,
                                    st_mem_transforms_eval,
                                    config.dataset.heartbeat_mode)

    # send to appropriate device 
    signal = signal.to(next(model.parameters()).device)

    return signal