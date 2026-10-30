import os
from helper_code_patched import *

import torch  # EDIT
from tqdm import tqdm  # EDIT

# EDIT: using batching and running eval on the gpu (because eval was massively slow before)
# EDIT: pass a model object and not a path
def run(args, model, run_model_func):
    BATCH_SIZE = 256

    # ensure that model is on gpu
    model = model.to('cuda')

    records = find_records(args.data_folder)
    if len(records) == 0:
        raise Exception('No data were provided.')

    # create output folder
    os.makedirs(args.output_folder, exist_ok=True)

    # process records in batches
    binary_outputs = []
    probability_outputs = []
    processed_records = []

    for i in tqdm(range(0, len(records), BATCH_SIZE), desc="processing"):
        batch = records[i:i + BATCH_SIZE]
        batch_paths = [os.path.join(args.data_folder, r) for r in batch]

        binary_output, probability_output = run_model_func(batch_paths, model, False)

        binary_outputs.extend(binary_output)
        probability_outputs.extend(probability_output)
        processed_records.extend(batch)

    for i in range(len(processed_records)):
        curr_record = processed_records[i]
        output_path = os.path.join(args.output_folder, os.path.dirname(curr_record))
        os.makedirs(output_path, exist_ok=True)
        output_file = os.path.join(args.output_folder, curr_record + '.txt')
        save_outputs(output_file, curr_record,
                     torch.tensor(binary_outputs[i], dtype=torch.float32),
                     torch.tensor(probability_outputs[i], dtype=torch.float32))

    print("Done!")