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
- Preprocessed: `data/pu200_t3/{train,val}/graph_*.pt` (regenerated locally) and `data/aashay_pu200_t3/{train,val}` (Aashay's original, 5540 train / 1320 val).
- Each `graph_N.pt` is a `torch_geometric.data.Data` point cloud with no edges: `x` (`[N_T3, 49]` T3 features), `sim_index` (`[N_T3]`, target sim-track index, negative means fake/unmatched), `sim_features` (`[N_sim, 8]`), `md_layer` and `md_simIdx` (`[N_T3, 4]`). An event has about 60k T3s.

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
- **Feature normalization is hardcoded.** `PCDataset.__getitem__` log-transforms `x[:, 0]` (pT) and then min/max-normalizes with tensors hardcoded for a 26- or 49-feature layout. `TransformerOCModel.build_neighbor_index` un-normalizes η and φ from columns 1 and 2 using hardcoded ranges. If you add, remove or reorder features in `t3_processing.py`, update both places.
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
