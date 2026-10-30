# Provenance

- Code: commit `346f523b20aabaaf21017f5aa3b8327994fc7e94` (tag `registration-v2`), exported with `git archive` and restricted to the runtime file set
  in `scripts/container/submission_v2c/FILES` of that repository. Every code file is byte-identical to that
  commit except eight, in which source comments, docstrings and three error-message strings that cited an
  internal project-rules file now say "project rule" instead: `src/run.py`, `src/models/anchor.py` and
  `src/data/{features_tokens,inventory,labels,mimic,signals,splits}.py`. Their syntax trees are otherwise
  identical, and the organisers' `train_model.py` and `run_model.py` give the same outputs on 220 public
  training records as the Docker dry run of the unedited tree (largest probability difference 4e-8; no label
  differs). The model files are unchanged and pinned.
- Model: `model_v2c/`, the 13 files pinned by sha256 in `configs/registered_model_pins_v2c.json`. The Docker build re-verifies them.
- Added from commit `88f076e762e8b98cbec09e076a4e91f52f6a9282`: `Dockerfile`, `README.md`, `AUTHORS.txt`, `.dockerignore`
  (`scripts/container/submission_v2c/`), and this file.
- Assembled by `scripts/container/package_submission_v2c.sh`. File times are set to the tagged commit's time,
  so rebuilding from the same two commits gives the same zip.
