from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import os
import pandas as pd
import numpy as np
import torch
from team.utils_challenge import load_signals, load_header, load_label, find_records
from sklearn.model_selection import train_test_split
import neurokit2 as nk
from scipy.signal import resample
from scipy.stats import median_abs_deviation
from tqdm import tqdm
from team.utils import DatasetMode, NormalizationMode, PaddingMode, HeartbeatMode, PreprocessingMode, PROFILE, profile, get_num_procs, multiprocess_records, print_mem
from team.utils_circular import SOURCE_TO_INT, check_tensor
import copy
from team.st_mem_processing import build_st_mem_transforms
import psutil
import gc
import torch.distributed as dist
import queue
import random

class ECGDataset(Dataset):
    def __init__(self, 
                records, 
                labels, 
                sources, 
                sampling_rates,
                signals,
                data_path, 
                target_sr, 
                preprocessing, 
                int_norm, 
                standard_len, 
                pad_mode, 
                st_mem_transforms, 
                heartbeat_mode,
                lazy):
        self.records = records
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.sources = torch.tensor(sources, dtype=torch.int)
        self.sampling_rates = torch.tensor(sampling_rates, dtype=torch.int)
        self.signals = signals
        self.data_path = data_path
        self.target_sr = target_sr
        self.preprocessing = preprocessing
        self.int_norm = int_norm
        self.standard_len = standard_len
        self.pad_mode = pad_mode
        self.st_mem_transforms = st_mem_transforms
        self.heartbeat_mode = heartbeat_mode
        self.lazy = lazy

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        label = self.labels[idx]
        check_tensor(label, expected_shape=[None], finite=True)  # numbers of labels present in tensor depends on which sub-datasets were configured
        
        source = self.sources[idx]
        check_tensor(source, expected_shape=[], finite=True)  # one int per datapoint (hence the empty list)
        
        record = self.records[idx]

        if self.lazy:
            sr = self.sampling_rates[idx].item()
            signal = load_and_process_signal(self.data_path, record, sr, self.target_sr, self.preprocessing, self.int_norm, self.standard_len, self.pad_mode, self.st_mem_transforms, self.heartbeat_mode)
        else:
            signal = self.signals[idx]
        
        check_tensor(signal, expected_shape=[None, 12], finite=True)  # signal length (depends on config), 12 leads

        return signal, label, source, record

####################################################
################    DATA LOADING    ################
####################################################
    
# provide mode in case only certain data sources should be loaded.
def load_data(data_path, limit, mode, target_sr, preprocessing, int_norm, standard_len, pad_mode, heartbeat_mode, lazy, num_gpus):
    records = find_records_limit(data_path, folder_limit=limit)
    num_records = len(records)
    print("found " + str(num_records) + " records")

    num_procs = get_num_procs()
    multiplied_args = [copy.deepcopy([data_path, mode, target_sr, preprocessing, int_norm, standard_len, pad_mode, heartbeat_mode, lazy]) for _ in range(num_procs)]
    computation_results = multiprocess_records(records, _work, multiplied_args)

    records_list = []
    labels_list = []
    sources_list = []
    rates_list = []

    for recs, labels, sources, rates, _ in computation_results:
        records_list += recs
        labels_list += labels
        sources_list += sources    
        rates_list += rates

    num_signals = len(records_list)
    if not lazy:
        # figure out shape of final tensor
        signal_shape = computation_results[0][4].shape[1:] 

        # Allocate final tensor
        signals_tensor = torch.empty((num_signals, *signal_shape), dtype=torch.float32)

        # Second pass: fill tensor (no Python list of signal arrays!)
        offset = 0
        for _, _, _, _, signals_np in computation_results:
            n = signals_np.shape[0]
            arr_tensor = torch.from_numpy(signals_np)  # zero-copy from NumPy
            signals_tensor[offset : offset + n].copy_(arr_tensor)  # copy into preallocated tensor
            offset += n
    else:
        signals_tensor = [None] * num_signals
    

    print("loaded " + str(len(records_list)) + " records")
    print("label distribution: " + str(get_label_distribution(labels_list, mode)))

    return records_list, labels_list, sources_list, rates_list, signals_tensor

def _work(pid, q, record_names, data_path, mode, target_sr, preprocessing, int_norm, standard_len, pad_mode, heartbeat_mode, lazy):
    records_list = []
    labels_list = []
    sources_list = []
    rates_list = []

    max_records = len(record_names)

    # preallocate signals tensor (for now we are assuming that no signals will get filtered away)
    if not lazy:
        signal_shape = (standard_len, 12)  
        signals_tensor = torch.empty((max_records, *signal_shape), dtype=torch.float32)

    num_filled = 0  # actual number of signals loaded

    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path_1 = os.path.join(script_dir, 'code_15_exams_1.csv')
    file_path_2 = os.path.join(script_dir, 'code_15_exams_2.csv')
    df1 = pd.read_csv(file_path_1)
    df2 = pd.read_csv(file_path_2)
    other_labels = pd.concat([df1, df2], ignore_index=True)

    iterator = tqdm(record_names, desc="process_0 loading") if pid == 0 else record_names

    for record_name in iterator:
        if PROFILE: print()

        # === load metadata ===
        with profile("load"):
            record = os.path.join(data_path, record_name)
            header = load_header(record)
            src = load_source(header)
            int_src = SOURCE_TO_INT[src]
            sr = load_sampling_rate(header)

        # === load labels ===
        with profile("label"):
            label = load_label(record)
            all_labels = [label]
            if int_src == 0:
                exam_id = int(record.split("/")[-1].strip())
                row = other_labels[other_labels['exam_id'] == exam_id]
                row_labels = row[['1dAVb', 'RBBB', 'LBBB', 'SB', 'ST', 'AF']]
                other_labels_list = row_labels.iloc[0].tolist()
                all_labels += other_labels_list
            else:
                all_labels += [-1 for _ in range(6)]

            filtered_labels = _filter_labels(all_labels, int_src, mode)

        if filtered_labels is None:
            continue  # skip unqualified record

        # load signal if non-lazy mode is configured
        if not lazy:
            if preprocessing == PreprocessingMode.ST_MEM:
                raise NotImplementedError("st_mem preprocessing currently not supported in pre-loading mode")

            signal = load_and_process_signal(
                data_path, record_name, sr, target_sr,
                preprocessing, int_norm, standard_len, pad_mode,
                st_mem_transforms=None,
                heartbeat_mode=heartbeat_mode,
            )

            # write loaded signal into preallocated tensor
            signals_tensor[num_filled] = signal

        records_list.append(record_name)
        labels_list.append(filtered_labels)
        sources_list.append(int_src)
        rates_list.append(sr)
        num_filled += 1

    # if some signals were filterd out signals tensor has some unsused slots (which we trim away)
    if not lazy: 
        signals_tensor = signals_tensor[:num_filled]
        signals_np = signals_tensor.numpy()  
    else:
        signals_np = None

    q.put((records_list, labels_list, sources_list, rates_list, signals_np))

# method that loads, preprocesses and segments the signal at signal_path
# set optional parameter numpy to True if result should not be returned as a tensor but a numpy array
def load_and_process_signal(data_path, record_path, sr, target_sr, preprocessing, int_norm, standard_len, pad_mode, st_mem_transforms, heartbeat_mode, numpy=False):
    # LOAD
    record = os.path.join(data_path, record_path)
    signal, _ = load_signals(record)

    # RESAMPLE
    duration = len(signal) / sr  # lenght of the signal (in seconds)
    num_target_samples = int(duration * target_sr) 
    signal = resample(signal, num_target_samples)

    # PREPROCESSING / DATA AUGMENTATION
    # what preprocessing / augmentations we use depends on if we apply team_processing or st_mem processing (or neither)
    if signal.shape[0] > 50:  # both preprocessing methods crash in case signal is very short (i.e. in that case skip preprocessing)
        if preprocessing == PreprocessingMode.TEAM:
            signal = team_process(signal, target_sr, int_norm, standard_len, pad_mode)
        elif preprocessing == PreprocessingMode.ST_MEM:
            signal = st_mem_process(signal, st_mem_transforms)
    
    # TRIMMING / PADDING
    # this needs to happen regardless of whether we used team_process or st_mem process 
    # (else torch will try to collate tensors of mismatching shape and fail)
    if signal.shape[0] > standard_len:  # trim
            signal = signal[:standard_len, :]  
    elif signal.shape[0] < standard_len:   # pad
        missing = standard_len - signal.shape[0]    
        if pad_mode == PaddingMode.SYMMETRIC:
            left_pad = missing // 2  # in case of odd missing points left_pad will be one less than right_pad
            right_pad = missing - left_pad
            signal = np.concatenate([np.zeros((left_pad, 12)), signal, np.zeros((right_pad, 12))], axis=0) 
        elif pad_mode == PaddingMode.LEFT:
            signal = np.concatenate([np.zeros((missing, 12)), signal], axis=0)
        elif pad_mode == PaddingMode.RIGHT:
            signal = np.concatenate([signal, np.zeros((missing, 12))], axis=0)
    
    # HEARTBEAT SEGMENTATION
    if heartbeat_mode != HeartbeatMode.MULTIPLE:
        with profile("beats"):
            beats = _get_single_beats(signal, heartbeat_mode, sr)
            if beats == []: return signal  # in case we tried to segment but failed return the unsegmented signal
            signal = np.array(beats).transpose((1, 0))

    if not numpy:
        signal = torch.tensor(signal, dtype=torch.float32)
    
    return signal

# returns a list (per-lead) of single beats
# in case segmentation fails an empty list is returned
def _get_single_beats(signal, heartbeat_mode, target_sr):
    resampled_beats = []
    for lead_idx in range(signal.shape[1]):
        try:
            beats = nk.ecg_segment(signal[:, lead_idx], sampling_rate=target_sr)
        except (ValueError, ZeroDivisionError):
            print("WARNING: skipping datapoint due to failed segmentation")
            return []  

        beats = [beats[x]["Signal"].tolist() for x in beats]  # unpack signals

        # segmentations of ecg_segment can contain nans i.e. remove beats that contain nans (as they would corrupt the signal during resampling)
        beats = list(filter(lambda x: not np.isnan(x).any(), beats))

        if len(beats) == 0:
            print("WARNING: skipping datapoint because all beats contained nan(s)")
            return []

        if heartbeat_mode == HeartbeatMode.SINGLE_RANDOM:
            # TODO: sampling different index per lead is not ideal (but simple to implement) i.e. if time implement taking same index across leads
            idx = np.random.randint(len(beats))
            beat = beats[idx]
        elif heartbeat_mode == HeartbeatMode.SINGLE_AVERAGE:
            beat = np.array(beats).mean(axis=0)
        
        beat = resample(beat, target_sr)  # resample all beats to same number of samples (else numpy complains)
        resampled_beats.append(beat)
    return resampled_beats

# in case a valid folder_limit is provided we assume that folder contains 3 sub-folders for the 3 sub-datasets named "code_15", "ptbxl" and "samitrop"
def find_records_limit(folder, folder_limit, header_extension='.hea'):
    if folder_limit == -1:  # no limit: walk the whole folder-tree recursively
        all_records = list(find_records(folder, header_extension))
    else: # limit: walk each subfolder-tree and limit each tree after having walked it
        subfolder_names = ["code_15", "ptbxl", "samitrop"]
        all_records = []
        for subfolder in subfolder_names:
            subfolder_path = os.path.join(folder, subfolder)
            records = list(find_records(subfolder_path, header_extension))
            limited_records = records[:folder_limit]
            limited_records = [subfolder + "/" + r for r in limited_records]
            all_records += limited_records
    return all_records

def load_source(header):
    return header.split("# Source:")[1].split("\n")[0].strip()

def load_sampling_rate(header):
    sr =  header.split("\n")[0].split(" ")[2]
    
    # sanity checks to reduce probability of grabbing the wrong value
    try:
        sr = int(sr)
    except ValueError:
        raise ValueError(f"sampling rate '{sr}' is not a valid integer")
    
    return sr

# filter the labels of one datapoint based on the mode of the dataset
def _filter_labels(targets, source, mode):
    # FALLBACK SOURCE
    if source == -1:
        if DatasetMode.FALLBACK_PADDED in mode:
            return targets[1:]
        if DatasetMode.FALLBACK_CHAGAS in mode:
            return targets[:1]
        
    # NORMAL SOURCES
    elif source == 0:
        if DatasetMode.CODE_15_NON_CHAGAS in mode:
            return targets[1:]
        if DatasetMode.CODE_15_CHAGAS in mode:
            return targets[:1]
    elif source == 1:
        if DatasetMode.PTBXL_PADDED in mode:
            return targets[1:]
        if DatasetMode.PTBXL_CHAGAS in mode:
            return targets[:1]
    elif source == 2:
        if DatasetMode.SAMITROP_PADDED in mode:
            return targets[1:]
        if DatasetMode.SAMITROP_CHAGAS in mode:
            return targets[:1]
    
    # in case the provided source does not match with any provided mode we assume that user does not want
    # to use data from provided source (i.e. return None to tell caller that datapoint should be skipped)
    else:
        return None

def team_process(signal, target_sr, int_norm, standard_len, pad_mode):   
    # neurokit cleaning (filtering)
    for lead in range(signal.shape[1]):
        lead_signal = signal[:, lead]
        signal[:, lead] = nk.ecg_clean(lead_signal, sampling_rate=target_sr)  # neurokit can only process one lead at a time
    
    ###############################################
    ########    INTENSITY NORMALIZATION    ########
    ###############################################

    regular_modes = [NormalizationMode.STANDARDIZE_REGULAR_LEADWISE, NormalizationMode.STANDARDIZE_REGULAR_PATIENTWISE]
    robust_modes = [NormalizationMode.STANDARDIZE_ROBUST_LEADWISE, NormalizationMode.STANDARDIZE_ROBUST_PATIENTWISE]

    leadwise_modes = [NormalizationMode.STANDARDIZE_REGULAR_LEADWISE, NormalizationMode.STANDARDIZE_ROBUST_LEADWISE]
    patientwise_modes = [NormalizationMode.STANDARDIZE_REGULAR_PATIENTWISE, NormalizationMode.STANDARDIZE_ROBUST_PATIENTWISE]
    
    ################################################################
    # OPTION 1: STRETCH LARGEST PEAK TO 1 OR -1
    # - PRO:
    #   - ensures no valuess lie outside of [-1, 1]
    # - CON:
    #   - sensitive to outliers i.e. if an outlier-peak is present the rest of the signal will end up on a much smaller scale than other samples
    #   - mean and std of signal are not necessarily transformed to some standard values (e.g. 0 and 1) 
    if int_norm in [NormalizationMode.STRETCH_PEAK]:
        for lead in range(signal.shape[1]):
            m = np.max(np.abs(signal[:, lead]))
            if m != 0:  # if max is less than one we will divide by it thereby also scaling such a signal to [-1, 1]
                signal[:, lead] = signal[:, lead] / m
    ################################################################
    # OPTION 2: MIN-MAX 
    # i.e. for all samples in lead_signal substract min (this will shift data into the range [0, (max - min)])
    # and then divide by (max - min) after which data will by in [0, 1] 
    #
    # we have not yet implemented this normalization strategy because it will remove all negative information from the signal
    # which might not be a good idea as negative input values might be valuable in e.g. a forward pass involving convolutions
    # (but maybe at some point we will want to implement it and test it's performance in our setting)
    ################################################################
    # OPTION 3: STANDARDIZATION
    # - PRO:
    #   - standardizes mean and std to same values (0 and 1) across all datapoints
    # - CON:
    #   - regular mode is still sensitive to outliers (but robust mode less so)
    #   - does not ensure that no values peak outside of some standard range (e.g. [-1, 1])
    #
    # lead-wise standardization normalizes intensity differences across leads away (because all leads get scaled to the same scale)
    # in case we want to preserve scale differences (e.g. because we think they carry important information) use patient-wise normalization 
    elif int_norm in (regular_modes + robust_modes):
        # pick functions
        if int_norm in regular_modes:
            central_tendency = np.mean
            deviation = np.std
        elif int_norm in robust_modes:
            central_tendency = np.median
            deviation = median_abs_deviation
        
        # compute statistics
        subtrahends, divisors = [], []
        if int_norm in leadwise_modes:
            for lead in range(signal.shape[1]):
                subtrahends.append(central_tendency(signal[:, lead]))
                divisors.append(deviation(signal[:, lead]))
        elif int_norm in patientwise_modes:
            subtrahend = central_tendency(signal)
            divisor = deviation(signal)

            # assert that we have computed the mean and std over all leads of the patient
            # i.e. that we have not generated num_leads many lead_wise statistics
            assert len(subtrahend.shape) == 0
            assert len(divisor.shape) == 0

            subtrahends = [subtrahend] * signal.shape[1]
            divisors = [divisor] * signal.shape[1]
        
        # carry out normalization
        for lead in range(signal.shape[1]):
            signal[:, lead] = signal[:, lead] - subtrahends[lead]
            if divisors[lead] != 0:  
                signal[:, lead] = signal[:, lead] / divisors[lead]
            
    ################################################################
    # OPTION 4: NO NORMALIZATION
    # sure, we need to rm things such as low frequency wander from our signal but appyling intensity normalization might remove important information
    # i.e. maybe things like "stronger peaks" or "overall smaller scale of values" indicates important information about condition of patient's heart.
    # for example, this paper (https://arxiv.org/pdf/2311.04229) has seen a slight decrease in performance when applying normalization.
    elif int_norm in [NormalizationMode.NO_NORMALIZATION]: 
        pass
    ################################################################ 

    return signal

def st_mem_process(signal, st_mem_transforms):
    signal = signal.transpose(1, 0)  # st_mem transformation code expects signal in shape [leads, time]
    signal = st_mem_transforms(signal)  # apply transforms
    signal = signal.transpose(1, 0)  # subsequent team_code expects signal in shape [time, leads]
    return signal

###############################################################
################    SET AND LOADER CREATION    ################
###############################################################

# provide split_seed to obtain the data at data_path split into two sets.
def get_set(data_path, limit, mode, target_sr, preprocessing, int_norm, standard_len, pad_mode, heartbeat_mode, st_mem_processing, lazy, num_gpus, split_seed=None):    
    # load data with pre-processing
    records_list, labels_list, sources_list, rates_list, signals_tensor = load_data(data_path, limit, mode, target_sr, preprocessing, int_norm, standard_len, pad_mode, heartbeat_mode, lazy, num_gpus)

    # create st_mem transforms
    st_mem_transforms_train, st_mem_transforms_eval = build_st_mem_transforms(st_mem_processing)

    # split (if requested), print label distribution and return
    if split_seed == None:

        from sklearn.utils import shuffle as sk_shuffle
        records_list, labels_list, sources_list, rates_list, signals_tensor = sk_shuffle(
            records_list, labels_list, sources_list, rates_list, signals_tensor, random_state=42
        )
        

        full_set = ECGDataset(records_list, labels_list, sources_list, rates_list, signals_tensor, data_path, target_sr, preprocessing, int_norm, standard_len, pad_mode, st_mem_transforms_train, heartbeat_mode, lazy) 
        return full_set, ECGDataset([], [], [], [], [], data_path, target_sr, preprocessing, int_norm, standard_len, pad_mode, st_mem_transforms_eval, heartbeat_mode, lazy) 
    else:
        recs_train, recs_val, y_train, y_val, s_train, s_val, rates_train, rates_val, signals_train, signals_val = train_test_split(
            records_list, labels_list, sources_list, rates_list, signals_tensor, train_size=0.9, random_state=split_seed, shuffle=True
        )
        train_set = ECGDataset(recs_train, y_train, s_train, rates_train, signals_train, data_path, target_sr, preprocessing, int_norm, standard_len, pad_mode, st_mem_transforms_train, heartbeat_mode, lazy)  
        val_set = ECGDataset(recs_val, y_val, s_val, rates_val, signals_val, data_path, target_sr, preprocessing, int_norm, standard_len, pad_mode, st_mem_transforms_eval, heartbeat_mode, lazy)
        return train_set, val_set

# set balanced to True to obtain a data loader balanced over the chagas label
def get_loader(data_set, batch_size, balanced=False, drop_last=True):
    num_workers = get_num_procs()

    if balanced == True:
        # extract the chagas label from each label list
        primary_labels = [label[0] for label in data_set.labels]
        labels_np = np.array(primary_labels).astype(int)
        
        class_sample_counts = np.bincount(labels_np)
        class_weights = 1.0 / class_sample_counts

        sample_weights = class_weights[labels_np]

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
            generator=torch.Generator().manual_seed(1)
        )

        loader = DataLoader(
            data_set, 
            sampler=sampler, 
            batch_size=batch_size, 
            drop_last=drop_last, 
            num_workers=num_workers, 
            worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(1))
    else:
        loader = DataLoader(
            data_set, 
            batch_size=batch_size, 
            drop_last=drop_last, 
            num_workers=num_workers,
            worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(1))

    return loader   

def seed_worker(worker_id):
    worker_seed = 42 + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed) 

#############################################
################    OTHER    ################
#############################################

def get_label_distribution(labels, mode):
    if DatasetMode.CODE_15_NON_CHAGAS in mode:  # if this mode is present at least a subset of datapoints will not have a chagas label at position 0 
        label_dict = None  
    else:  # if we are in this block we can be sure that all datapoints will have a chagas label in the first position
        label_dict = {}
        for label in labels:
            chagas_label = label[0]  
            if chagas_label not in label_dict:
                label_dict[chagas_label] = 1
            else:
                label_dict[chagas_label] += 1
    return label_dict