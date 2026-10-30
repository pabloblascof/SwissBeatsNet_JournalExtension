import numpy as np
import torch
from team.models.resnet import ResNet
import os
from hydra import initialize, compose
from omegaconf import OmegaConf
from enum import Enum
from contextlib import contextmanager
import time
import multiprocessing
import math
from datetime import datetime
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
from omegaconf import OmegaConf
import glob
# make imports from mm_mae repo work
import sys
curr_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(curr_dir + "/../mm_mae/")
from utils.config_setup import load_config as prepare_hydra
import psutil

def setup():
    # set seeds
    seed = 1
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # determine device
    device = "gpu" if torch.cuda.is_available() else "cpu"

    print("setup complete") 

    return device

def create_model(config):
    # ceate model
    model = None
    if config.model.type == "resnet":
        model = ResNet(config)

    return model

# standardize the way we create date_time strings 
def get_datetime():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# these cluster CPUs have tons of cores but any given user does not get access to all of them i.e. only spawn a reasonable amount of procs 
def get_num_procs():

    ################################
    ################################
    
    # num_procs MUST NOT exceed 8 because in that case Docker container on 
    # workstation (and likely also on challenge setup as they have 16 VCPUs) crashes
    upper_bound = 8

    ################################
    ################################


    count = multiprocessing.cpu_count()
    if upper_bound != -1:
        return min(upper_bound, count)
    else:
        return count

# takes function and applies it to records in a multiprocessed way.
# note that func needs to take in pid, multiprocessing_queue and the data_split as its first arguments.
# even if func_args are the same for all processes create a separate instance of each func_arg for each process
# (else some of the func_args could become a point of contention across processes which could limit parallelism).
def multiprocess_records(records, func, func_args):
    num_procs = get_num_procs()

    print("spawning " + str(num_procs) + " processes ...")

    # compute amount of work for each process
    num_records = len(records)
    chunk_size = int(math.ceil(num_records / num_procs))
    chunk_sizes = [chunk_size] *  (num_procs - 1)  # the first n - 1 processes will all process the same amount of work
    chunk_sizes.append(num_records - chunk_size * (num_procs - 1))  # the last process will work on whatever is left
    assert sum(chunk_sizes) == num_records, "sum of chunk_sizes (" + str(chunk_sizes) + ") is not equal to num_records (" + str(num_records) + ")"

    # split data
    splits = []
    start = 0
    for size in chunk_sizes:
        splits.append(records[start : start + size])
        start += size

    q = multiprocessing.Queue()

    # create and start
    ps = []
    for pid in range(num_procs): 
        args = [pid, q, splits[pid]] + func_args[pid]
        args = tuple(args)
        p = multiprocessing.Process(target=func, args=args)
        ps.append(p)
        p.start()

    # collect and merge results
    computation_results = []
    for _ in range(num_procs):
        process_result = q.get()
        computation_results.append(process_result)
    
    # join
    for p in ps: p.join()

    return computation_results

# neurokit plotting is nice (can be invoked using the respective nk functions) since it contains all sorts  
# of useful info but sometimes we just want a simple 2D plot of a time series (which this function implements) 
def plot_signals(signals, titles, dpi=100):
    signals_np = [s.detach().cpu().numpy() for s in signals]
    seq_len = max(len(s) for s in signals_np)

    global_min = min(s.min() for s in signals_np)
    global_max = max(s.max() for s in signals_np)

    fig, axes = plt.subplots(
        len(signals_np), 1, 
        figsize=(12, 2.5 * len(signals_np)), 
        sharex=True
    )

    for i, (signal, ax) in enumerate(zip(signals_np, axes)):
        ax.plot(signal, color='#2c3e50', linewidth=1.5, alpha=0.9)
        ax.grid(True, which='both', linestyle='--', linewidth=0.4, alpha=0.5)
        ax.set_title(titles[i], fontsize=12, fontweight='bold', color='#34495e')
        ax.tick_params(axis='both', labelsize=8, colors='#7f8c8d')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylim(global_min, global_max)

    fig.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)

    return Image.open(buf)

# reassemble a .pth pretrained model that was split up into multiple chunks
def load_split_pth(directory):
    all_files = sorted(
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, f))
    )

    buffer = BytesIO()
    for part_path in all_files:
        with open(part_path, 'rb') as f:
            buffer.write(f.read())

    buffer.seek(0)
    return torch.load(buffer)

################################################################
#########################    CONFIG    #########################
################################################################

loaded_config = None  # cache for loaded config s.t. when run_model is invoked in a loop we do not re-invoke compose over and over again

# TEAM CONFIG
def load_config():
    with initialize(version_base=None, config_path="config"):
        global loaded_config
        if loaded_config == None:
            # load
            config = compose(config_name="main")  # load config using hydra
            # check
            config = check_config(config)
            # cache
            loaded_config = config
        return loaded_config

# used to load the config structure of mm_mae repo (which st_mem and mm_mae use)
def load_mmmae_config_structure(config_path, output_dir, train_data_path, encoder_path):

    prepare_hydra()  # registers the sub_config dataclasses s.t. hydra can load them

    with initialize(config_path="../mm_mae/config/", version_base=None):
        global loaded_config
        if loaded_config == None:
            # load
            config = compose(config_name=config_path)
            # overrides
            config.model.output_dir = output_dir
            config.model.encoder_path = encoder_path
            config.model.train_data_path = train_data_path
            # check
            config = check_config(config)
            # cache
            loaded_config = config
    return loaded_config

# converts strings to enum values (which will throw an error if a string in config takes an invalid value)
def check_config(config):
    # enum conversion (validates that string values in config take legal values)
    config["training"]["pretraining_data_mode"] = [DatasetMode(elem) for elem in config["training"]["pretraining_data_mode"]]  
    config["training"]["fine_tuning_data_mode"] = [DatasetMode(elem) for elem in config["training"]["fine_tuning_data_mode"]]  
    config["dataset"]["intensity_normalization"] = NormalizationMode(config["dataset"]["intensity_normalization"])
    config["dataset"]["pad_mode"] = PaddingMode(config["dataset"]["pad_mode"])
    config["dataset"]["heartbeat_mode"] = HeartbeatMode(config["dataset"]["heartbeat_mode"])    
    config["dataset"]["preprocessing"] = PreprocessingMode(config["dataset"]["preprocessing"])
    return config

def merge_configs(configs, titles):
    res = {}
    for i, c in enumerate(configs): 
        if not isinstance(c, dict):
            c = OmegaConf.to_container(c, resolve=True, throw_on_missing=True)
        res[titles[i]] = c

    return res

################################################################
#######################    PROFILING    ########################
################################################################

PROFILE = False

@contextmanager
def profile(name):
    if PROFILE:
        start = time.time()
        yield
        end = time.time()
        print(f"{name:<10} {(end - start)*1000:<10.0f}", end="", flush=True)  # in ms
    else:
        yield

################################################################
#########################    ENUMS    ##########################
################################################################

# using enums tp represent strings in config enables us to validate the
# strings in config at runtime allowing us to catch invalid string values

class PreprocessingMode(Enum):
    NONE = "none"
    TEAM = "team"
    ST_MEM = "st_mem"

class DatasetMode(Enum):
    # code_15 contains chagas labels and other cardiological labels
    CODE_15_NON_CHAGAS = "code_15_non_chagas"
    CODE_15_CHAGAS = "code_15_chagas"
    # other sources only contain chagas labels
    PTBXL_CHAGAS = "ptbxl_chagas"
    SAMITROP_CHAGAS = "samitrop_chagas"
    FALLBACK_CHAGAS = "fallback_chagas"
    # in mm_mae pretraining we want to:
    # - load signals from all sub_datasets
    # - non_chagas labels for online Linear Regression probing 
    # since only code_15 contains non_chagas labels we need a way to pad instances of the other sub_datasets
    # (else tensor creation fails due to inhomogeneous shapes) which is what the following modes are meant for
    PTBXL_PADDED = "ptbxl_padded"
    SAMITROP_PADDED = "samitrop_padded"
    FALLBACK_PADDED = "fallback_padded"

class NormalizationMode(Enum):
    STRETCH_PEAK = "stretch_peak"
    STANDARDIZE_REGULAR_LEADWISE = "standardize_regular_leadwise"
    STANDARDIZE_REGULAR_PATIENTWISE= "standardize_regular_patientwise"
    STANDARDIZE_ROBUST_LEADWISE= "standardize_robust_leadwise"
    STANDARDIZE_ROBUST_PATIENTWISE= "standardize_robust_patientwise"
    NO_NORMALIZATION = "no_normalization"

class PaddingMode(Enum):
    LEFT = "left"
    RIGHT = "right"
    SYMMETRIC = "symmetric"

class HeartbeatMode(Enum):
    MULTIPLE = "multiple"
    SINGLE_RANDOM = "single_random"
    SINGLE_AVERAGE = "single_average"

def print_mem(tag=""):
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / 1024**3  # in GB
    print(f"\n [{tag}] PID {os.getpid()} Memory Usage: {mem:.3f} GB \n", flush=True)