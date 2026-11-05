# CLAUDE.md

ML track reconstruction for CMS Phase-2 (HL-LHC, PU200). A sparse transformer is trained with an **object condensation (OC)** loss to group LST (Line Segment Tracking) **T3** objects, one point cloud per event, into tracks. Upstream: [SegmentLinking/transformer-oc](https://github.com/SegmentLinking/transformer-oc), originally by Aashay Arora. Datasets are made in a separate repo, [SegmentLinking/transformer-datasets](https://github.com/SegmentLinking/transformer-datasets).

## Direction (planned changes)

This repo is mid-redesign; the sections below describe the *current* state.

- Replace the object condensation loss with a simpler, more robust loss. Don't invest in OC/β/DBSCAN tuning.
- Use per-hit truth (`hit_simIdx`, `hit_layer`) instead of per-T3/per-MD truth wherever possible.
- Put the fraction of real vs fake hits into the loss, rather than real/fake track classification.
- Train on η/φ patches instead of whole events, so training fits on smaller GPUs.
- Training must run on CPU. Keep code device-agnostic; avoid new CUDA-only dependencies (`fastgraphcompute` goes away with OC).
- Possibly add a decoding step that merges a cluster of T3s into a hit list for a track candidate.
- Turn the repo into an installable Python package with proper package imports, replacing the flat scripts and `sys.path` hacks. New modules should be written to fit a package layout.

When writing new code, follow this direction even if the existing code doesn't yet.

## Layout

- `data/`: intentionally empty in git; put datasets here. Everything under `data/` except `README.md` is gitignored.
- `src/`: training. `main.py` (Lightning entrypoint), `model.py` (`TransformerOCModel` and `TransformerLightningModule`), `dataset.py` (`PCDataset`, plus the η/φ patch mask), `constants.py` (feature normalization ranges and η/φ columns), `helpers.py` (seeding, config loading), `test_dataset.py` (tests for the η/φ mask), `utils/object_condensation.py` (OC loss, uses `fastgraphcompute`), `utils/torch_geometric_interface.py` (PyG `batch` to `row_splits`), `utils/hept_utils.py` (HEPT LSH attention, currently unused).
- `analysis/`: evaluation. `main.py` runs inference, clusters, and plots efficiency, fake rate and duplicates; `measures.py` holds the metrics. The other scripts are one-off studies.
- `config_*.yaml`: one config drives data paths, training and analysis. `config_pu200.yaml` is the reference; `config_pu200.tuna.yaml` is a smaller local variant (`max_neighbors: 16`, `gpus: [0]`).
- `misc/nrp/`: Kubernetes YAMLs and Dockerfiles for the NRP cluster (namespace `cern-cms-gpu-tracking`).

## Environment

Run everything in the Apptainer image (a copy of `aaarora/ml-tracking`, with torch 2.5.0+cu121 and Python 3.10):

```bash
apptainer exec --nv /ceph/users/atuna/ml-tracking.sif python3 <script> ...
```

- `--nv` is required: without it, `fastgraphcompute` (the OC loss) fails to import.
- The image lacks `statsmodels` (imported by `analysis/main.py`) and `pytest`.
- Local GPU here is a single Tesla V100 16 GB (device 0). `config_pu200.yaml` and `config_pu100.yaml` use `gpus: [1]`, and `analysis/main.py` hardcodes `cuda:1`; both need changing to run locally.

## Data

- Graphs are made with transformer-datasets (`python -m transformer_datasets.t3_processing`), which documents the pipeline and the graph format. Configs read them from `data/<name>/{train,val}` (e.g. `data/pu200_t3/`, 4023 train / 977 val locally).
- Each `graph_N.pt` is a `torch_geometric.data.Data` point cloud with no edges: `x` (`[N_T3, 49]` T3 features), `sim_index` (`[N_T3]`, target sim-track index, negative means fake/unmatched), `sim_features` (`[N_sim, 8]`), `md_layer` and `md_simIdx` (`[N_T3, 4]`), `hit_layer` and `hit_simIdx` (`[N_T3, 6]`). An event has about 60k T3s.

## Common commands (run from the repo root)

```bash
# Train (outputs to output/lightning_logs/version_N/)
apptainer exec --nv /ceph/users/atuna/ml-tracking.sif python3 src/main.py --config config_pu200.tuna.yaml

# Evaluate (plots land in the lightning_logs version dir)
apptainer exec --nv /ceph/users/atuna/ml-tracking.sif python3 analysis/main.py --config config_pu200.yaml --output output/lightning_logs/version_N/ --n-events 100 --epsilon 0.05

# Unit tests (plain asserts; runs without pytest)
cd src && apptainer exec /ceph/users/atuna/ml-tracking.sif python3 test_dataset.py
```

## Gotchas

Several of these (flat imports, normalization, batch = 1 event, OC targets) describe the current code and are expected to change with the redesign above.

- **Imports are flat** (`from dataset import ...`, `from constants import ...`). `src/` scripts rely on their own directory being on `sys.path`, `src/__init__.py` and `analysis/main.py` do `sys.path.append('./src')`, and config data paths are relative. Run from the repo root.
- **Feature normalization is hardcoded, and tied to transformer-datasets.** `PCDataset.__getitem__` log-transforms `x[:, 0]` (pT) and then min/max-normalizes with ranges from `src/constants.py`, hardcoded for a 26- or 49-feature layout. `TransformerOCModel.build_neighbor_index` un-normalizes η and φ from columns 1 and 2 using hardcoded ranges. If transformer-datasets adds, removes or reorders features, update both places.
- **Batch size is 1 event** (the PyG `DataLoader` default), with `accumulate_grad_batches` for an effective batch. Memory scales with `N_T3 × max_neighbors × hidden_dim`. Full-size configs have OOMed on a 16 GB GPU, so use small `hidden_dim` and `max_neighbors` locally.
- Attention is sparse over ΔR neighbors (`dr_threshold`, `max_neighbors`), not full self-attention.
- OC targets: `sim_index < 0` is mapped to `-1` (noise) in the training and validation steps.
- `analysis/main.py` rebuilds `TransformerOCModel` from the config but does not pass `max_neighbors`, so it falls back to the default (64). Keep the analysis config consistent with the training config, or pass the parameters explicitly.
- `debug: true` in a config disables early stopping, monitors `train_loss` for checkpoints and saves every epoch.
- Checkpoint state dicts are prefixed with `model.`. `analysis/main.py` strips the prefix when loading.

## Workflow

- Develop on a branch and PR into `SegmentLinking/transformer-oc` `main`. Never run `git commit` or `git push`; the user does that.
- Don't commit data, `.root`, `.pt`, checkpoints or `output/`.
- Compare new ideas against the reproduced baseline on a fixed validation event set: tracking efficiency vs pT/η, fake rate, duplicate rate, and inference time per event.
