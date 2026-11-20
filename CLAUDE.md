# CLAUDE.md

ML track reconstruction for CMS Phase-2 (HL-LHC, PU200). A sparse transformer is trained with an **object condensation (OC)** loss to group LST (Line Segment Tracking) **T3** objects, one point cloud per event, into tracks. Upstream: [SegmentLinking/transformer-oc](https://github.com/SegmentLinking/transformer-oc), originally by Aashay Arora (see `notes/tuna.md` for links to the CHEP26 talk and DP note).

## Direction (planned changes)

This repo is mid-redesign; the sections below describe the *current* state.

- Replace the object condensation loss with a simpler, more robust loss. Don't invest in OC/β/DBSCAN tuning.
- Use per-hit truth (`t3_hit_*_simIdx`, `t3_hit_*_layer`) instead of per-T3/per-MD truth wherever possible.
- Put the fraction of real vs fake hits into the loss, rather than real/fake track classification.
- Train on η/φ patches instead of whole events, so training fits on smaller GPUs.
- Training must run on CPU. Keep code device-agnostic; avoid new CUDA-only dependencies (`fastgraphcompute` goes away with OC).
- Possibly add a decoding step that merges a cluster of T3s into a hit list for a track candidate.
- Turn the repo into an installable Python package with proper package imports, replacing the flat scripts and `sys.path` hacks. New modules should be written to fit a package layout.

When writing new code, follow this direction even if the existing code doesn't yet.

## Layout

- `data/`: preprocessing (ROOT LSTNtuple to `.pt`). `t3_processing.py` is the main script, `ls_processing.py` is the LS variant, and column names live in `constants.py`. Everything under `data/` except `*.py`, `*.sh` and `*.sub` is gitignored.
- `src/`: training. `main.py` (Lightning entrypoint), `model.py` (`TransformerOCModel` and `TransformerLightningModule`), `dataset.py` (`PCDataset`), `utils/object_condensation.py` (OC loss, uses `fastgraphcompute`), `utils/hept_utils.py` (HEPT LSH attention, currently unused).
- `analysis/`: evaluation. `main.py` runs inference, clusters, and plots efficiency, fake rate and duplicates; `measures.py` holds the metrics. The other scripts are one-off studies.
- `config_*.yaml`: one config drives preprocessing paths, training and analysis. `config_pu200_dev.yaml` is a small local smoke-test config.
- `misc/nrp/`: Kubernetes YAMLs and Dockerfiles for the NRP cluster (namespace `cern-cms-gpu-tracking`).
- `notes/tuna.md`: running log of data provenance, environment and issues. Read it for background.

## Environment

Run everything in the Apptainer image (a copy of `aaarora/ml-tracking`, with torch 2.5.0+cu121 and Python 3.10):

```bash
apptainer exec --nv /ceph/users/atuna/ml-tracking.sif python3 <script> ...
```

- `--nv` is required: without it, `fastgraphcompute` (the OC loss) fails to import.
- The image lacks `statsmodels` (imported by `analysis/main.py`) and `pytest`.
- Local GPU here is a single Tesla V100 16 GB (device 0). `config_pu200.yaml` uses `gpus: [1]`, and `analysis/main.py` hardcodes `cuda:1`; both need changing to run locally.

## Data

- Raw LSTNtuples (ttbar PU200): `/ceph/users/atuna/work/cms_tracking_ml/data/output_pu200/output_pu200_*.root`
- Preprocessed (current): `/ceph/cms/store/user/jachisma/transformer-oc/output_pu200_t3_hits_v2/{train,val,test}/LSTNtuple_<name>.root.entry_<N>.pt` (3000/700/700 events), made by the transformer-datasets repo (see its README for the format). Per-hit truth is in `hit_idx` and one `hit_{k}_simIdxs` field per hit slot. In apptainer, bind it with `-B /ceph/cms,/ceph/cms_ech`.
- Older: `data/pu200_t3/{train,val}/graph_*.pt` and `data/aashay_pu200_t3/{train,val}` (Aashay's original, 5540 train / 1320 val).
- Each `graph_N.pt` is a `torch_geometric.data.Data` point cloud with no edges: `x` (`[N_T3, 40]` T3 features: 3 T3, 2 × 5 LS, 3 × 9 MD for the T3's three distinct MDs, inner to outer), `sim_index` (`[N_T3]`, target sim-track index, negative means fake/unmatched), `sim_features` (`[N_sim, 8]`), `md_layer` and `md_simIdx` (`[N_T3, 3]`), `hit_layer` and `hit_simIdx` (`[N_T3, 6]`, two hits per MD). An event has about 60k T3s. Graphs made before the duplicate middle MD was dropped have 49 features and 4 MD slots, and don't work with the current code.

## Common commands (run from the repo root)

```bash
# Preprocess one event (run from data/ because of the flat imports of constants.py)
cd data && apptainer exec /ceph/users/atuna/ml-tracking.sif python3 t3_processing.py --seed 42 --input ../../data/output_pu200/output_pu200_0.root --n_events 1 --output pu200_t3/

# Train (outputs to output/lightning_logs/version_N/)
apptainer exec --nv /ceph/users/atuna/ml-tracking.sif python3 src/main.py --config config_pu200_dev.yaml

# Evaluate (plots land in the lightning_logs version dir)
apptainer exec --nv /ceph/users/atuna/ml-tracking.sif python3 analysis/main.py --config config_pu200.yaml --output output/lightning_logs/version_N/ --n-events 100 --epsilon 0.05

# Unit tests (plain asserts; runs without pytest)
cd data && python3 test_t3_processing.py
```

## Gotchas

Several of these (flat imports, normalization, batch = 1 event, OC targets) describe the current code and are expected to change with the redesign above.

- **Imports are flat** (`from dataset import ...`, `from constants import ...`). `src/` scripts rely on their own directory being on `sys.path`, `analysis/main.py` does `sys.path.append('./src')`, and config data paths are relative. Run from the repo root (preprocessing from `data/`).
- **Feature normalization is computed from the training data.** Per-column min/max (after log-transforming `t3_pt`) over all training files is precomputed with `python3 src/normalization.py --data-dir <train dir> --output <file>` (the current dataset's is `normalization_train.json` next to its `train/`) and set as `normalization_file` in the configs; with `normalization_file: null`, `src/main.py` computes it from `normalization_n_files` training files instead. `src/main.py` passes it to `PCDataset`, and saves it as `<run dir>/normalization.json` and in the checkpoint hparams; `analysis/main.py` reuses that file, and falls back to the legacy hardcoded ranges in `src/constants.py` for older runs. Columns of `x` are looked up by name through `FEATURE_NAMES` in `src/constants.py`, which must match the dataset's layout (checked on load). The model's ΔR neighbor search uses the raw `graph.eta`/`graph.phi` attached by `PCDataset`. `analysis/measures.py` still un-normalizes pT/η and reads MD x/y from hardcoded columns.
- **Batch size is 1 event** (the PyG `DataLoader` default), with `accumulate_grad_batches` for an effective batch. Memory scales with `N_T3 × max_neighbors × hidden_dim`. Full-size configs have OOMed on a 16 GB GPU, so use `config_pu200_dev.yaml` (small `hidden_dim`, `max_neighbors`, subsets) locally.
- Attention is sparse over ΔR neighbors (`dr_threshold`, `max_neighbors`), not full self-attention.
- OC targets: `sim_index < 0` is mapped to `-1` (noise) in the training and validation steps.
- `analysis/main.py` rebuilds `TransformerOCModel` from the config but does not pass `max_neighbors`, so it falls back to the default (64). Keep the analysis config consistent with the training config, or pass the parameters explicitly.
- `debug: true` in a config disables early stopping, monitors `train_loss` for checkpoints and saves every epoch.
- Checkpoint state dicts are prefixed with `model.`. `analysis/main.py` strips the prefix when loading.

## Workflow

- Develop on a branch and PR into `SegmentLinking/transformer-oc` `main`. Don't commit or push without being asked.
- Don't commit data, `.root`, `.pt`, checkpoints or `output/`.
- Compare new ideas against the reproduced baseline on a fixed validation event set: tracking efficiency vs pT/η, fake rate, duplicate rate, and inference time per event.
