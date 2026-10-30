# this code would fit more nicely into utils.py but circular import error forces us to create a separate utils file ...

import os
curr_file_path = os.path.dirname(os.path.abspath(__file__))
import torch
import wandb
import shutil
from team.multi_evaluation import multi_evaluate
from argparse import Namespace
import sys
sys.path.append(curr_file_path + "../")  # needed to find run_model.py
from run_model_optimized import run as invoke_model
import uuid
from pathlib import Path
import numpy as np
import torch.distributed as dist

def prob_to_class(prob, threshold):
    return int(prob > threshold)

################################################################
########################    LOGGING    #########################
################################################################

# wrapper around python dict that allows for fallback value
class FallbackDict(dict):
    def __init__(self, *args, default=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.default = default

    def __missing__(self, key):
        return self.default

# this dict exists in order to space efficiently (i.e. as an int and not a string) load source information 
SOURCE_TO_INT = FallbackDict({
                                "CODE-15%": 0,
                                "PTB-XL": 1,
                                "SaMi-Trop": 2,
                            }, default = -1  # fallback in case datapoint is not from one of our 3 sub-datasets (which challenge organizers might do)
                            )

class LoggingWrapper:
    # sometimes logger will be a logger is already being logged to i.e. in order to 
    # clearly demarcate the logs inserted by this logger one can supply a logger_prefix 
    def __init__(self, logger, step_types, threshold, logger_prefix=""):
        self.logger = logger
        self.step_types = step_types  # maybe some models contain exotic step_types (beyond training and validation)
        self.num_sources = 3  # this will ignore the fallback source - we only want to log information from the "normal" 3 sub-datasets
        self.threshold = threshold
        # the following data structures contain dicts per step_type and within these dicts a list per source
        self.step_preds = {} 
        self.step_labels = {}  # ground truth labels
        self.step_losses = {}
        # the step provided by lightning is messy i.e. create my own notion of steps (that we then have more control over)
        self.step = None
        self.reset_step()
        self.logger_prefix = logger_prefix
        # populate data structures with empty dict structures
        for i, _ in enumerate(self.step_types):
            self.reset_logging_data(i)

    def log(self, step_type_idx, downstream):
        # multi-gpu: only log from rank 0 (else we get duplicate logs)
        if self.not_rank_zero(): return

        step_type = self.step_types[step_type_idx] 

        ########  ALL SOURCES  ########
        logging_prefix = "downstream" if downstream else "pretrain"
        logging_prefix = self.logger_prefix + logging_prefix
        # loss
        all_losses = sum(self.step_losses[step_type].values(), [])  # sum is sort of like the fold operation from func_prog
        avg_loss = torch.stack(all_losses).mean()
        self.logger.log_metrics(
            {logging_prefix + "/" + step_type + "/loss": avg_loss.item()},
            step=self.step[step_type]
        )
        # histogram of predictions
        if downstream:
            # collect over all sources
            all_preds = sum(self.step_preds[step_type].values(), [])
            all_labels = sum(self.step_labels[step_type].values(), [])

            # label rates
            self._log_label_metrics(all_preds, all_labels, logging_prefix, step_type)

            # histogram of predictions
            self.logger.experiment.log({
                logging_prefix + "/" + step_type + "/predictions": 
                wandb.Histogram(torch.stack(all_preds).detach().cpu().numpy())
            })  # in principle it would be good to pass it a step here but then the hists are not logged to board anymore (so we omit step for now)

        ########  SINGLE SOURCES  ########
        if downstream:  # single source logging does not make sense during pretraining since in that case only code_15 is used
            for source in range(self.num_sources):
                extended_prefix = logging_prefix + "_" + str(source)

                filtered_preds = self.step_preds[step_type][source]
                filtered_labels = self.step_labels[step_type][source]
                filtered_losses = self.step_losses[step_type][source]

                if len(filtered_preds) == 0:
                    print(f"WARNING: {step_type} dataset does not contain instances of source {source}")
                    continue

                # loss
                self.logger.log_metrics(
                    {extended_prefix + "/" + step_type + "/loss": torch.stack(filtered_losses).mean().item()},
                    step=self.step[step_type]
                )

                # label rates
                self._log_label_metrics(filtered_preds, filtered_labels, extended_prefix, step_type)
            
                # histogram of predictions
                self.logger.experiment.log({
                    extended_prefix + "/" + step_type + "/predictions": 
                    wandb.Histogram(torch.stack(filtered_preds).detach().cpu().numpy())
                })  # in principle it would be good to pass it a step here but then the hists are not logged to board anymore (so we omit step for now)
        
        # only generate table in case we are in downstream training (since only then mmodel generates disease probabilities)
        if downstream:
            # merge single_source and mixed_source data (since multi_evaluate generates a row for all of them)
            mixed_and_single_labels = {key: value for key, value in self.step_labels[step_type].items()}
            mixed_and_single_labels[3] = all_labels
            mixed_and_single_labels = {  # extract items from tensors and wrap in np.array
                key: np.array([pred.item() for pred in preds])
                for key, preds in mixed_and_single_labels.items()
            }

            mixed_and_single_preds = {key: value for key, value in self.step_preds[step_type].items()}
            mixed_and_single_preds[3] = all_preds
            mixed_and_single_preds = {  # extract items from tensors and wrap in np.array
                key: np.array([pred.item() for pred in preds])
                for key, preds in mixed_and_single_preds.items()
            }

            mixed_and_single_binary_outputs = {  # apply prob_to_class to all predictions to get binary outputs
                key: np.array([prob_to_class(pred, self.threshold) for pred in preds])
                for key, preds in mixed_and_single_preds.items()
            }

            if step_type_idx == 1:  # only do for validation data
                self.log_table(mixed_and_single_labels=mixed_and_single_labels, 
                            mixed_and_single_preds=mixed_and_single_preds, 
                            mixed_and_single_binary_outputs=mixed_and_single_binary_outputs,
                            logging_prefix=logging_prefix,
                            step_type_idx=step_type_idx)
        
        self._increment_step(step_type_idx)

        return avg_loss
    
    # in case table should be generated on hold-out test data set test to True (in that case also provide model, run_model_func and train_data_path)
    def log_table(
            self, 
            mixed_and_single_labels=None, mixed_and_single_preds=None, mixed_and_single_binary_outputs=None, logging_prefix=None, step_type_idx=None,
            test=False, model=None, run_model_func=None, train_data_path=None): 

        # multi-gpu: only log from rank 0 (else we get duplicate logs)
        if self.not_rank_zero(): return
        
        if not test:
            in_memory_data = mixed_and_single_labels, mixed_and_single_preds, mixed_and_single_binary_outputs
            table = multi_evaluate(in_memory_data=in_memory_data)
        else:
            # invoke run model
            test_data_path = self._infer_path(train_data_path)

            # test data path exists on cluster and workstation but not in Docker container (in which case we don't need table generation anyway)
            if not os.path.exists(test_data_path):
                return
            else:
                unique_id = str(uuid.uuid4())  # add random id to dir to make it essentially impossible for parallel runs to write to the same tmp_dir
                tmp_dir_path = os.path.join(curr_file_path, f"tmp_log_{unique_id}")
                os.makedirs(tmp_dir_path, exist_ok=True)

                try:
                    # invoke model to create output txt files
                    args = Namespace(
                        data_folder=test_data_path,
                        output_folder=tmp_dir_path,
                    )
                    invoke_model(args, model, run_model_func)

                    # generate challenge table
                    table = multi_evaluate(test_data_path=test_data_path, output_path=tmp_dir_path)
                finally:
                    shutil.rmtree(tmp_dir_path)  # ensure cleanup even on error
            

        table_string = f"{table}"

        # generate title s.t. table ends up in correct spot in wandb board
        if test:
            title = "final_test_data"
        else:
            title = logging_prefix + ("_train_data" if step_type_idx == 0 else "_val_data")
        # log to wandb as plain text
        self.logger.experiment.log({
            title: wandb.Html(f"{table_string}")
        })
    
        # also print to console as a safety net (in case wandb logging does not work for some reason)
        print("\n" * 5)
        print(table_string)
        print("\n" * 5)

    def _infer_path(self, train_path):
        train_path = Path(train_path).resolve()
        test_path = train_path.parent / "test"
        return str(test_path)

    def _log_label_metrics(self, preds, labels, logging_prefix, step_type):
        # all correct predictions (of that class) over number of instances (of that class)
        logging_title = "accuracy"
        present_classes = list(set([label.item() for label in labels]))
        for class_idx in present_classes:  # only iterate over classes present (else we get division by zero)
            # get all preds and labels that correspond to the current label
            pred_count = 0
            label_count = 0
            for i in range(len(preds)):
                label = labels[i].item()
                pred_class = prob_to_class(preds[i].item(), self.threshold)

                if label == class_idx:  
                    label_count += 1  # we found a label of the class that we are interested in in this loop iteration
                    if pred_class == class_idx:  # we have found a correct prediction of the class that we are interested in in this loop iteration
                        pred_count += 1

            rate = pred_count / label_count
        
            self.logger.log_metrics(
                {logging_prefix + "/" + step_type + "/" + logging_title + "/class_" + str(class_idx): rate},
                step=self.step[step_type],
            )
            
    def store_logging_data(self, step_type_idx, sources, pred, labels, loss, downstream):
        check_tensor(loss, expected_shape=[None, 1], finite=True)    # [batch_dim, 1]
        check_tensor(sources, expected_shape=[None], finite=True)    # [batch_dim]
        
        # detach (for more efficient gathering)
        sources = sources.detach()
        loss = loss.detach()
        # GATHER (needed in mulit-gpu setting to obtain all chunks of valiation data on all gpus)
        sources = self._gather_tensor_across_gpus(sources)
        loss = self._gather_tensor_across_gpus(loss)

        step_type = self.step_types[step_type_idx]
        self._add(self.step_losses, step_type, sources, loss)

        if downstream:
            check_tensor(pred, expected_shape=[None, 1], finite=True)
            check_tensor(labels, expected_shape=[None, 1], finite=True)
            
            pred = pred.detach()
            labels = labels.detach()
            pred = self._gather_tensor_across_gpus(pred)
            labels = self._gather_tensor_across_gpus(labels)

            self._add(self.step_preds, step_type, sources, pred)
            self._add(self.step_labels, step_type, sources, labels)


    def _add(self, d, key, sources, val):
        # multi-gpu: we will only log from rank 0 (else we get duplicate logs) i.e. no need to store data on any other rank
        if self.not_rank_zero(): return

        # flatten away batch dimension 
        val = list(val.reshape(-1))
        # ensure that val is now structured the way you intended it to be (= a list of scalar tensors)
        assert all(isinstance(x, torch.Tensor) and x.ndim == 0 for x in val), "not all elements are scalar tensors"

        # detach (otherwise full graph is stored every time)
        val = [x.detach().cpu() for x in val]

        # insert data in the right place
        assert (len(val) == len(sources)), f"number of values {len(val)} does not match number of sources {len(sources)}"
        for i, source in enumerate(sources): 
            source_item = source.item()
            if source_item in d[key]:  # do not proceed in case code tries to add item with fallback source (since we are not logging information from this source)
                d[key][source_item].append(val[i])
        
    def reset_logging_data(self, step_type_idx):
        self._reset_one(self.step_preds, step_type_idx)
        self._reset_one(self.step_labels, step_type_idx)
        self._reset_one(self.step_losses, step_type_idx)

    def _reset_one(self, info_dict, step_type_idx):
        step_type = self.step_types[step_type_idx]
        type_dict = {}
        for source in range(self.num_sources):  
            type_dict[source] = []
        info_dict[step_type] = type_dict
    
    # not called by clients but used internally to create a notion of timesteps
    def _increment_step(self, step_type_idx):
        step_type = self.step_types[step_type_idx]
        self.step[step_type] += 1

    # should be called by clients when switching from pretraining to fine-tuning (to have plots start at 0 again)
    def reset_step(self):
        self.step = {step_type: 0 for step_type in self.step_types}

    def _gather_tensor_across_gpus(self, tensor, dim=0):
        if dist.is_available() and dist.is_initialized():
            gathered = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered, tensor)
            return torch.cat(gathered, dim=dim)
        else:
            return tensor

    # helper to figure out if current process is rank 0 or not
    def not_rank_zero(self):
        return dist.is_available() and dist.is_initialized() and dist.get_rank() != 0


################################################################
#######################    MINEFIELD    ########################
################################################################

# this function makes it easy to inject assertions as part of the "minefield of assertions" approach 
# the star forces all subsequent args to be passed as named args (which enforces good readability of calls to this function) 
def check_tensor(tensor, *, expected_shape=None, finite=False, name="tensor"):
    def fail(msg): 
        raise AssertionError(msg)

    tensor_shape = list(tensor.shape)
    
    if expected_shape is not None:
        # compare numbers of dimensions
        if len(expected_shape) != len(tensor_shape):
            fail(
                f"{name} number of dimensions mismatch: expected {len(expected_shape)}, got {len(tensor_shape)} "
                f"(tensor shape: {tensor_shape}, expected shape: {expected_shape})"
            )
        
        # compare sizes of each dimension
        for i, (actual, expected) in enumerate(zip(tensor_shape, expected_shape)):
            if expected is not None and actual != expected:
                fail(
                    f"{name} shape mismatch at dimension {i}: expected {expected}, got {actual} "
                    f"(tensor shape: {tensor_shape}, expected shape: {expected_shape})"
                )

    if finite:
        if not torch.isfinite(tensor).all():
            fail(f"{name} contains nan or inf")

################################################################
#######################    CRITERION    ########################
################################################################

# this class allows us to:
# - abstract away the reduction of the loss over all labels that we train on during pre-training
# - have a well-positioned place in the code to invoke some more tensor_assertions
class ReducedCriterion():
    def __init__(self, crit):
        self.crit = crit
    
    def compute(self, y_pred, y, downstream):
        num_classes = 1 if downstream else 6
        check_tensor(y_pred, expected_shape=[None, num_classes], finite=True)  # batch_dim, output(s) for positive class
        check_tensor(y, expected_shape=[None, num_classes], finite=True)  # batch_dim, label(s) of correct class

        loss = self.crit(y_pred, y)
        # in case of pre-training the loss will contain 6 items (one per non-chagas-label) i.e. reduce them to one item
        if not downstream:
            # without keepdim this operation will return [batch_size] but in case reduction is not applied loss will be
            # [batch_size, 1] after this if stmt i.e. use keepdim to always have loss be [batch_size, 1] after this if stmt
            loss = torch.sum(loss, dim=1, keepdim=True)  
        check_tensor(loss, expected_shape=[None, 1], finite=True)  # batch_dim, scalar loss per datapoint (hence no second dimension in list)
        
        return loss