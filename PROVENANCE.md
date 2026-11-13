# Provenance

- **Parent commit:** `8981769` (entry F32). This commit replaces the tree with the demographics-only baseline.
  - Kept unchanged: `Dockerfile` base image and `requirements.txt`, `helper_code.py`, `helper_code_patched.py`,
    `train_model.py`, `run_model.py`, `LICENSE`, `AUTHORS.txt`.
  - The Dockerfile drops the model_v2c lines and verifies the model_v3 pin at build time.
- **Source:** `model_v3/demographics.json` was produced by `scripts/container/fit_demographics_final.py` of the
  project repository (branch `journal_submission`). It uses the S3.2 baseline recipe
  `configs/baselines/demographics.yaml` and the frozen patient-disjoint splits, with every outer fold used for
  fitting.
- **Exactness:** the numpy formula in `team_code.py` reproduces the fitted model's scores on all 220 public records,
  with a largest probability difference of 1e-16 and 0 label differences.
- **Registration:** `REGISTRATION_v3.md` in the project repository.
