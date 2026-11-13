#!/usr/bin/env python

# SwissBeatsNet, focus-issue entry 2: the demographics-only baseline (age and sex), registered in
# REGISTRATION_v3.md. It reads no ECG sample. See README.md and PROVENANCE.md.
#
# The model is a fixed logistic score fitted offline on the training data and shipped as
# model_v3/demographics.json, pinned by sha256 in configs/registered_model_pins_v3.json:
#     x      = [age, age^2, sex_male]          (sex_male: 1 Male, 0 Female)
#     x      = median-imputed where missing    (training-pool medians)
#     logit  = intercept + sum(coef * (x - mean) / scale)
#     output = (logit > threshold_logit, sigmoid(logit))
# train_model() does not train: it verifies the pinned file and copies it into the model folder.

import hashlib
import json
import math
import os
import shutil

import numpy as np

from helper_code import get_age, get_sex, load_header

HERE = os.path.dirname(os.path.abspath(__file__))
REGISTERED_DIR = os.path.join(HERE, "model_v3")
PINS = os.path.join(HERE, "configs", "registered_model_pins_v3.json")
MODEL_FILE = "demographics.json"


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(folder=REGISTERED_DIR):
    """Raise unless every pinned file in ``folder`` has its pinned sha256."""
    with open(PINS) as f:
        pins = json.load(f)["files"]
    for name, digest in pins.items():
        got = _sha256(os.path.join(folder, name))
        if got != digest:
            raise RuntimeError(f"{name}: sha256 {got} does not match the pin {digest}")
    return True


class DemographicsModel:
    def __init__(self, params):
        if params.get("columns") != ["age", "age_sq", "sex_male"]:
            raise ValueError(f"unexpected columns {params.get('columns')}")
        self.median = np.asarray(params["impute_median"], dtype=np.float64)
        self.mean = np.asarray(params["scaler_mean"], dtype=np.float64)
        self.scale = np.asarray(params["scaler_scale"], dtype=np.float64)
        self.coef = np.asarray(params["coef"], dtype=np.float64)
        self.intercept = float(params["intercept"])
        self.threshold = float(params["threshold_logit"])

    @staticmethod
    def features(age, sex):
        """``[age, age^2, sex_male]`` with NaN for anything missing or unparseable."""
        try:
            a = float(age)
        except (TypeError, ValueError):
            a = float("nan")
        if not math.isfinite(a):
            a = float("nan")
        s = str(sex).strip().lower() if sex is not None else ""
        m = 1.0 if s in ("male", "m") else 0.0 if s in ("female", "f") else float("nan")
        return np.array([a, a * a, m], dtype=np.float64)

    def logit(self, age, sex):
        x = self.features(age, sex)
        x = np.where(np.isfinite(x), x, self.median)
        return float(self.intercept + np.dot((x - self.mean) / self.scale, self.coef))

    def predict(self, age, sex):
        z = self.logit(age, sex)
        return int(z > self.threshold), float(1.0 / (1.0 + math.exp(-z)))


################################################################################
# Required functions.
################################################################################

def train_model(data_folder, model_folder, verbose):
    # No training: the registered model is fixed. Verify it and copy it into the model folder.
    verify()
    os.makedirs(model_folder, exist_ok=True)
    shutil.copyfile(os.path.join(REGISTERED_DIR, MODEL_FILE), os.path.join(model_folder, MODEL_FILE))
    verify(model_folder)
    if verbose:
        print("Registered demographics model verified and copied; no training performed.")


def load_model(model_folder, verbose):
    # Only the pinned registered model is ever served, whatever the model folder holds.
    verify()
    with open(os.path.join(REGISTERED_DIR, MODEL_FILE)) as f:
        model = DemographicsModel(json.load(f))
    if verbose:
        print("Loaded the registered demographics model (pins verified).")
    return model


def run_model(record, model, verbose):
    try:
        header = load_header(record)
        age, sex = get_age(header), get_sex(header)
    except Exception as exc:  # a malformed header still gets an output: the imputed score
        if verbose:
            print(f"{record}: header unreadable ({exc}); using imputed demographics")
        age, sex = None, None
    return model.predict(age, sex)
