# Challenge container for SwissBeatsNet, focus-issue entry 2: the registered demographics-only
# baseline model_v3 (REGISTRATION_v3.md). Base image and pinned requirements are those of entry F32. Built and run by the organisers with
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

# Fail the build, not the run, if any byte of the registered model differs from its sha256 pin.
RUN python -c "import team_code; team_code.verify(); print('model_v3 pins verified; team_code importable')"
