# SAME AS AUGMENT_MODEL.PY BUT COMPUTES MORE METRICS

#!/usr/bin/env python

# Do *not* edit this script. Changes will be discarded so that we can process the models consistently.

# This file contains functions for evaluating models for the Challenge. You can run it as follows:
#
#   python evaluate_model.py -d data -o outputs -s scores.csv
#
# where 'data' is a folder containing files with the reference signals and labels for the data, 'outputs' is a folder containing
# files with the outputs from your models, and 'scores.csv' (optional) is a collection of scores for the model outputs.
#
# Each data or output file must have the format described on the Challenge webpage. The scores for the algorithm outputs are also
# described on the Challenge webpage.

import argparse
import numpy as np
import os
import os.path
import sys

from helper_code_patched import *

# Parse arguments.
def get_parser():
    description = 'Evaluate the Challenge model.'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-d', '--data_folder', type=str, required=True)
    parser.add_argument('-o', '--output_folder', type=str, required=True)
    parser.add_argument('-s', '--score_file', type=str, required=False)
    return parser

# Evaluate the models.
def evaluate_model(data_folder, output_folder):
    # Find the records.
    records = find_records(data_folder)
    num_records = len(records)

    if num_records == 0:
        raise FileNotFoundError('No records found.')

    labels = np.zeros(num_records)
    binary_outputs = np.zeros(num_records)
    probability_outputs = np.zeros(num_records)

    # Load the labels and model outputs.
    for i, record in enumerate(records):
        label_filename = os.path.join(data_folder, record)
        label = load_label(label_filename)

        output_filename = os.path.join(output_folder, record + '.txt')
        output = load_text(output_filename)
        binary_output = get_label(output, allow_missing=True)
        probability_output = get_probability(output, allow_missing=True)

        # Missing model outputs are interpreted as zero for the binary and probability outputs.
        labels[i] = label

        if not is_nan(binary_output):
            binary_outputs[i] = binary_output
        else:
            binary_outputs[i] = 0
        if not is_nan(probability_output):
            probability_outputs[i] = probability_output
        else:
            probability_outputs[i] = 0

    # EDIT
    output_string = compute_metrics(labels, probability_outputs, binary_outputs)
    return output_string
    # PREV
    # return challenge_score, auroc, auprc, accuracy, f_measure

def compute_metrics(labels, probability_outputs, binary_outputs):
    output_string = (
            f'Challenge score: {0}\n'
            f'AUROC: {0}\n'
            f'AUPRC: {0}\n'
            f'Accuracy: {0}\n'
            f'F-measure: {0}\n'
            f'Sensitivity: {0}\n'
            f'Specificity: {0}\n'
            f'TP Rate: {0}\n'
            f'TN Rate: {0}\n'
            f'FP Rate: {0}\n'
            f'FN Rate: {0}\n'
            f'TP: {0}\n'
            f'TN: {0}\n'
            f'FP: {0}\n'
            f'FN: {0}\n')
    return output_string

# Run the code.
def run(args):
    # Compute the scores for the model outputs.

    # EDIT
    output_string = evaluate_model(args.data_folder, args.output_folder)
    
    # PREV
    '''
    challenge_score, auroc, auprc, accuracy, f_measure = evaluate_model(args.data_folder, args.output_folder)
    output_string = \
        f'Challenge score: {challenge_score:.3f}\n' + \
        f'AUROC: {auroc:.3f}\n' \
        f'AUPRC: {auprc:.3f}\n' + \
        f'Accuracy: {accuracy:.3f}\n' \
        f'F-measure: {f_measure:.3f}\n'
    '''

    # Output the scores to screen and/or a file.
    if args.score_file:
        save_text(args.score_file, output_string)
    else:
        print(output_string)

if __name__ == '__main__':
    run(get_parser().parse_args(sys.argv[1:]))