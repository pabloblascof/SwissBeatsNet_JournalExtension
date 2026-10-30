# SwissBeatsNet: Chagas disease detection from the 12-lead ECG

Post-Challenge evaluation entry for the George B. Moody PhysioNet Challenge 2025. This is one model, run once
against the sequestered test cohorts. It is the single model registered for the journal extension of our
Challenge paper, and it has the same shape as our Challenge model: a frozen per-lead masked-autoencoder ECG
encoder with per-lead classification heads.

## Model

**Encoder.** An ST-MEM ViT-Base masked autoencoder (12 layers, width 768, 12 heads), patch size 50. Each lead is
encoded separately as a 2,250-sample sequence at 250 Hz (9.0 s). It was trained with masked reconstruction at
mask ratio 0.9 and an MSE inter-lead alignment objective. It is frozen at inference.
- Self-supervised pretraining: 200 epochs on MIMIC-IV-ECG v1.0.
- Continued self-supervised training: 60 epochs on the Challenge training corpora (CODE-15%, SaMi-Trop and
  PTB-XL, 366,185 records).

**Head.** Twelve per-lead non-linear heads on the encoder's CLS token. Their logits are averaged into one logit.
The heads were trained for one epoch on the same 366,185 records (11,854 Chagas-positive).

**Outputs.**
- Probability: `sigmoid(logit)`.
- Binary label: 1 if the logit exceeds 1.3080, the 95th percentile of the training-record logits (top 5%).

**Preprocessing.** Resample to 250 Hz, then the Challenge model's signal cleaning and lead-wise standardisation.
The first 2,250 samples (9.0 s) are kept; shorter records are padded symmetrically. Age and sex are not used.

**Input handling** (`src/container/input_guard.py`). Records in the training formats pass through unchanged.
Otherwise:
- the 12 standard leads are selected and ordered by name;
- if only I, II and V1–V6 are present, the limb leads are derived;
- non-finite samples are interpolated within their lead.

**Failures.** A record that cannot be read or processed is scored `(0, 0.0)` with a warning, and the run
continues. A systematic failure stops the run with an error instead of writing all-zero outputs: either the first
20 records all fail, or more than 5% fail after 100 records.

## How the container runs

The trained weights are inside the image, so nothing is downloaded at build or run time.

- `model_v2c/` holds the model: the encoder as ten byte chunks of at most 48 MiB, the head, and a manifest. The
  Dockerfile copies it to `/challenge/registered_model`.
- `configs/registered_model_pins_v2c.json` pins the sha256 of each of its 13 files. The build runs
  `python -m src.container.submission verify` and **fails if any byte differs**.
- `SBN_SUBMISSION_MODE=registered_folder` is set in the image:
  - `train_model.py` does **not** train. It verifies the pinned files and copies them into the model folder.
  - `run_model.py` loads the model folder. If that folder is empty, it verifies and loads
    `/challenge/registered_model` instead. Either order of the two scripts therefore scores the same weights.
- The GPU is used when present; otherwise the model runs on the CPU.
  - Measured inference: about 0.25 s per record on 8 CPU threads, 0.10 s on one GPU.
  - Model load: about 3 s.

Standard Challenge commands:

    docker build -t swissbeatsnet .
    docker run -it -v <model>:/challenge/model -v <holdout_data>:/challenge/holdout_data \
        -v <holdout_outputs>:/challenge/holdout_outputs -v <training_data>:/challenge/training_data swissbeatsnet bash
    python train_model.py -d training_data -m model -v
    python run_model.py -d holdout_data -m model -o holdout_outputs -v

## Files

| Path | Role |
|---|---|
| `team_code.py` | the required `train_model` / `load_model` / `run_model` |
| `train_model.py`, `run_model.py`, `helper_code.py`, `evaluate_model.py`, `prepare_*_data.py` | the organisers' scripts (see below) |
| `helper_code_patched.py`, `run_model_optimized.py`, `evaluate_model_augmented.py` | our Challenge entry's helpers, imported by `team/` |
| `src/container/` | model loading, input checks, pinned-file verification |
| `src/data/`, `src/models/`, `team/`, `mm_mae/` | preprocessing, encoder and head code |

The organisers' scripts are the copies from our Challenge entry:
- `train_model.py`, `run_model.py` and `evaluate_model.py` import `helper_code_patched` instead of `helper_code`.
  That file differs from `helper_code.py` only in the scoring functions; record discovery and output writing
  are identical.
- The `prepare_*_data.py` copies also carry our entry's data-conversion changes. They are not used at run time.
- The organisers' unmodified `train_model.py`, `run_model.py` and `helper_code.py` give the same outputs
  (tested).
| `model_v2c/` | the trained model (weights and manifest) |
| `configs/registered_model_pins_v2c.json` | sha256 pins of the model files |
| `PROVENANCE.md` | the source commit and how this folder was assembled |
| `Dockerfile`, `requirements.txt` | the runtime environment |

Licences: `LICENSE` covers the organisers' scripts. `mm_mae/` is under `mm_mae/LICENSE` (GPL-3.0). The
encoder code in `mm_mae/networks/` (`st_mem.py`, `st_mem_vit.py`, `vit.py`) is adapted from ST-MEM and keeps its
licence, `st_mem/repo/LICENSE` (CC BY-NC 4.0).

## Authors

See `AUTHORS.txt`.
