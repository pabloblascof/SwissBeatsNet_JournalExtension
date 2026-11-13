# SwissBeatsNet — focus-issue entry 2: demographics-only baseline

This entry scores each record from **age and sex alone**, read from the record header. It reads no ECG
sample. It is a registered baseline for the paper's held-out evaluation, not an attempt at a better detector.
The paper's ECG model was entry F32 (commit `8981769` of this repository).

- **Model:** logistic regression on `[age, age², sex_male]`, standardised, `class_weight='balanced'`.
  - Fitted offline on every labelled training record of CODE-15%, SaMi-Trop and PTB-XL (366,185 records,
    any-positive labels).
  - Shipped as `model_v3/demographics.json` and pinned by sha256 in `configs/registered_model_pins_v3.json`.
- **Missing values:** a missing or unparseable age or sex takes the training-pool median.
- **Output:** `probability = sigmoid(logit)`. `label = logit > threshold_logit`, where the threshold is the 95th
  percentile of the training logits.
- **`train_model.py`** does not train. It verifies the pinned file and copies it into the model folder.
- **`run_model.py`** serves only the pinned model, whatever the model folder holds.
- **Runtime:** CPU only; well under a millisecond per record.

Standard commands:

    python train_model.py -d training_data -m model -v
    python run_model.py -d holdout_data -m model -o holdout_outputs -v
