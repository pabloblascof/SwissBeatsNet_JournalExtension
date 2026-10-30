# Challenge container for SwissBeatsNet: the registered single model model_v2c
# (REGISTRATION_v2.md, tag registration-v2). Built and run by the organisers with
# the standard train_model.py / run_model.py commands; see README.md.
#
# Base image: pinned by tag. To pin it by digest as well, resolve it with
#   docker pull nvcr.io/nvidia/pytorch:24.01-py3 && docker inspect --format='{{index .RepoDigests 0}}' nvcr.io/nvidia/pytorch:24.01-py3
# and replace the tag with `nvcr.io/nvidia/pytorch@sha256:<digest>`.
FROM nvcr.io/nvidia/pytorch:24.01-py3
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y python3.10-venv && apt-get clean && rm -rf /var/lib/apt/lists/*

## DO NOT EDIT these 3 lines.
RUN mkdir /challenge
COPY ./ /challenge
WORKDIR /challenge

# The registered model (REGISTRATION_v2.md section 5). It lives outside the
# /challenge/model mount, so a mounted folder cannot shadow it. train_model.py
# verifies it and copies it into the model folder; run_model.py on an empty
# model folder verifies and loads it in place. No training happens.
COPY ./model_v2c /challenge/registered_model
ENV SBN_SUBMISSION_MODE=registered_folder
ENV SBN_REGISTERED_PINS=configs/registered_model_pins_v2c.json

# Determinism and a quiet, writable runtime: hash seed fixed, cuBLAS workspace
# fixed (required by torch.use_deterministic_algorithms), matplotlib (imported
# transitively by team/utils.py) pointed at a writable cache, no .pyc writes.
ENV PYTHONHASHSEED=0 \
    CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    MPLCONFIGDIR=/tmp/sbn_mpl \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# install packages into pip (we have tried conda before but that failed on the challenge setup)
# --no-deps is NOT used because the base image's torch build must be replaced by
# the pinned torch==2.5.1 wheel and its own dependencies.
RUN pip install --upgrade pip && pip install -r requirements.txt

# Fail the build, not the run, if any byte of the registered model differs from
# its sha256 pins, or if the model path cannot be imported.
RUN SBN_REGISTERED_MODEL_DIR=/challenge/registered_model python -m src.container.submission verify
RUN python -c "import src.container.journal_model; print('journal model path importable')"
RUN python -c "import team_code; print('team_code importable')"
