# script that invokes the evaluation code provided by the challenge multiple times: 
# once for the mixed-dataset and once per sub-dataset (s.t. we get fine-grained evaluation insights)

# since this script is for our personal convenience (i.e. it will not be executed by the challenge team) 
# we do not have to worry about this script running inside of a docker container (and can e.g. use cluster paths)

import subprocess
import wandb
from multiprocessing import Pool
import os
import argparse
from evaluate_model_augmented import compute_metrics

curr_file_path = os.path.dirname(os.path.abspath(__file__))

def evaluate_dataset(args):
    test_data_path, output_path, dataset = args

    # create python cmd
    python_script = os.path.join(curr_file_path, "../evaluate_model_augmented.py")
    full_test_data_path = test_data_path if dataset == "mixed" else test_data_path + "/" + dataset
    full_output_path = output_path if dataset == "mixed" else output_path + "/" + dataset
    shell_command = (
        f"python {python_script} -d {full_test_data_path} -o {full_output_path}"
    )

    # if on workstation or cluster prepend the shell command with conda activation cmds
    # this is not needed in Docker container because there we are installing packages into pip and not a conda env
    in_docker = (infer_system() == "docker")
    if not in_docker:
        conda_env = "ecg"
        shell_command = f'eval "$(conda shell.bash hook)" && conda activate {conda_env} && ' + shell_command

    try:
        result = subprocess.run(["bash", "-c", shell_command], check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(e.stdout)
        print(e.stderr)
        raise

    stdout = result.stdout
    metrics_dict = _metrics_string_to_dict(stdout)
    return dataset, metrics_dict

def infer_system():
    system = None
    
    # probe paths
    cluster_path = "/cluster/home/elucas"
    workstation_path = "/local/home/elucas"

    if os.path.exists(cluster_path):
        system = "cluster"
    elif os.path.exists(workstation_path):
        system = "workstation"
    else:
        system = "docker"

    # very clearly print the inferred system (to catch inference errors early)
    print("SYSTEM=" + system)

    return system

def _metrics_string_to_dict(string):
    lines = string.splitlines()
    metrics = {}
    for line in lines:
        parts = line.split(":")
        if len(parts) < 2: continue  # evaluation script also returns empty lines
        name = parts[0]
        val = parts[1]
        val = -1 if (val == "nan") else float(val) 
        metrics[name] = val
    return metrics

# two options to run multi-dataset evaluation
# - pass in-memory data (true labels, probabilities, class decisions) to invoke metrics computation directly on this data
# - pass paths and invoke evaluation in the way the challenge organizers implemented it
def multi_evaluate(in_memory_data=None, test_data_path=None, output_path=None):
    ########  RUN EVALUATIONS  ########
    datasets = ["code_15", "ptbxl", "samitrop", "mixed"]

    if in_memory_data != None:
        mixed_and_single_labels, mixed_and_single_preds, mixed_and_single_binary_outputs = in_memory_data
        results = []
        for idx, dataset in enumerate(datasets):
            metrics_string = compute_metrics(mixed_and_single_labels[idx], mixed_and_single_preds[idx], mixed_and_single_binary_outputs[idx])
            metrics_dict = _metrics_string_to_dict(metrics_string)
            results.append((dataset, metrics_dict))
    else:
        args = [(test_data_path, output_path, d) for d in datasets]

        with Pool(processes=len(datasets)) as pool:
            results = pool.map(evaluate_dataset, args)

    metrics = {}
    for d, computed_metrics in results:
        metrics[d] = computed_metrics

    ########  ASSEMBLE RESULTS  ########
    metric_names = list(metrics["mixed"].keys())
    datasets = ["mixed", "code_15", "ptbxl", "samitrop"]

    header = [""] + metric_names
    separator = ["---"] * len(header)

    rows = []
    for ds in datasets:
        row = [ds]
        for metric in metric_names:
            val = metrics[ds].get(metric, -1)
            if isinstance(val, float):
                val_str = f"{val:.2f}"
            else:
                val_str = str(val)
            row.append(val_str)
        rows.append(row)

    lines = []
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(separator) + " |")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")

    markdown_table = "\n".join(lines)
    return markdown_table

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="multi_eval")
    parser.add_argument(
        "--test_data_path", "-d", type=str, required=True,
    )
    parser.add_argument(
        "--output_path", "-o", type=str, required=True,
    )
    args = parser.parse_args()

    print(multi_evaluate(args.test_data_path, args.output_path))