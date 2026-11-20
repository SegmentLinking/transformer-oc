# MDML: a directory for MD-level models

SimpleMDNet is a barebones network that groups outer-tracker mini-doublets (MDs) into track candidates. Its purpose is to be a real model, in the ballpark of competent, to build infrastructure around (training, inference, evaluation with `harness/`). Physics performance is not the goal yet: it will fail in dense regions and has known limits.

MDTransformer will be a more robust network that groups outer-tracker mini-doublets (MDs) into track candidates, and we will target performance to be better than T5 track reconstruction.

This file is the living design note: update it whenever a decision is made or changed.

## How it works: SimpleMDNet

1. Each MD goes through a feed-forward MLP on its own features (no neighbours, no attention).
2. The MLP predicts, per MD, whether it is fake and the kinematics of its sim: q/pT, η, cos φ, sin φ.
3. MDs with a high predicted fake probability are dropped.
4. DBSCAN clusters the remaining MDs in the predicted (q/pT, η, cos φ, sin φ) space; each cluster is a track candidate.

Input: the per-event `.pt` files of `transformer-datasets` `md_processing.py`, with the `md_*` arrays.

## How it works: MDTransformer

In progress: `md_transformer.py` is an outline, filled in one piece at a time. Settings are in `md_transformer.yaml`.

1. Training cuts random (η, φ) patches from each event. A patch has a margin around it: margin MDs are context for attention, but not in the loss or the predictions.
2. A transformer predicts, per MD, whether it is fake and the kinematics of its sim: q/pT, η, cos φ, sin φ (φ relative to the patch centre).
3. An MD attends to another only if their |Δη| and |Δφ| are within the window of their layer pair.
4. Inference covers the event with a patch grid (patches with margin 0); each grid patch gets its margin and goes through the network.
5. Decoding as for now: a fake cut, then DBSCAN on the predicted kinematics (`cluster_space: kinematics`).
6. Optional (`embed_dim > 0`): a learned per-MD embedding, trained with a contrastive loss to put MDs of one sim close; `cluster_space: embedding` runs DBSCAN on it instead. Off by default.

Open: the layer-pair windows come from a quantile of same-sim MD pairs (`window_quantile`). They reflect the physics, so they should be stable across events and runs; once checked, replace `window_quantile` with the per-layer-pair η/φ windows themselves in the yaml (keep `window_scale`).

Open: MDs with no sim have every sim feature at −999 in the `.pt` files. `MDInputs.from_mds` turns their sim kinematics into `nan`, so any accidental use shows up at once. Later (not now), `md_processing.py` could write `nan` for missing fields directly.

Open: test `absolute_phi: true` once performance approaches T5. The network is φ-symmetric by default, but the detector is not, even in simulation: modules are rectangles assembled into a periodic mechanical structure. The effect may be negligible. Real data adds more φ asymmetry (dead channels, misalignment).

Open: move away from DBSCAN soon, ideally to a decoder trained with the loss. DBSCAN on the embedding is especially awkward: the contrastive loss only compares MDs within one patch, so nothing ties the embeddings of different grid patches together, and unrelated tracks in distant patches can land close. The embedding stays in the code as a possibility, with `embed_dim: 0` by default. The long-term plan for MD ML includes a contrastive embedding.

## Pieces, bottom-up

Each is one small module with its own test.

1. **`data.py`:** a `MDsFromOneEvent` dataclass read from one `.pt` file, and the `.pt` paths of a dataset's
   directory (`train/` holds only training events, and so on).
2. **`simple_mdnet.py`:** The `SimpleMDNet` class owns the MDNet and the working point (`fake_cut`, `eps`); it trains the MDNet, chooses the working point, and turns `.pt` files into one harness `EventTrackCandidates` per event, with every field filled. Running the module is the end-to-end command: train, then choose the working point, into one run directory.
3. **Harness:** `--trackcands simple_mdnet` in `harness/performance.py`, calling `SimpleMDNet.trackcands`.
4. **`clustering.py`:** `find_mdclusters` (DBSCAN with `min_samples=MIN_MDS`), shared by the MD networks.
5. **`trackcands.py`:** `TrackCandidatesFromOneEvent` (a copy of the harness's `EventTrackCandidates`; how to de-duplicate them is open) and `trackcands_from_mdclusters`, shared by the MD networks.
6. **`md_transformer.py`:** `MDTransformerNet` (the algorithm and working point), `MDTransformerLightningModule` (training) and `MDTransformerModule` (the network). Independent of `simple_mdnet.py`.

## Decisions

| Date | Decision |
|---|---|
| 2026-11-17 | The model sees one MD at a time: an MLP on the MD's own features, predicting a fake logit and the sim's (q/pT, η, cos φ, sin φ). cos φ and sin φ instead of φ, so the regression has no wrap-around at ±π. |
| 2026-11-17 | Fake means `md_simIdx < 0` (not both hits from one sim) or a sim with pT < 0.8 GeV. Loss: BCE on fake over every MD, plus MSE on the sim kinematics of non-fake MDs only. |
| 2026-11-17 | Track candidates: DBSCAN with `min_samples=3` (so at least 3 MDs) on the predicted kinematics of MDs passing the fake cut, with no scaling of the axes. Score: the mean of 1 − p(fake) over the track candidate's MDs. |
| 2026-11-17 | Names say what an array is and where it comes from. `md_*` arrays have one entry per MD. Truth from the sim is `md_sim*` (`md_simIsFake`, `md_simKinematics`), network output is `md_pred*` (`md_predIsFakeLogit`, `md_predKinematics`), so a sim quantity never reads like a measurement (as LSTNtuple's `md_isFake` can). |
| 2026-11-17 | The tracking algorithm is the whole chain, not the training alone: train, choose the working point (`--fake-cut`, `--eps`), infer, evaluate. Each step is its own script and is developed and checked on its own, but the goal is one command that relaunches every step into one run directory (`output/mdml/run_<date>/`: checkpoints, track candidates, performance). Lesson from the T3 model, where the steps after training (DBSCAN eps, β cut) were separate and easy to forget. |
| 2026-11-17 | One class, `SimpleMDNet`, is the whole algorithm: it owns the MDNet and the working point, and returns track candidates in memory instead of writing them to disk. A saved `simple_mdnet.pt` holds the weights and the working point together. Track candidates get `trackcand_simIdx` from the `.pt` file's per-hit sims (`md_inner_simIdxs`, `md_outer_simIdxs`), which cover only hits in MDs and so are complete for these track candidates. Replaces the separate `model.py`, `train.py` and `infer.py`. |
| 2026-11-17 | Training runs on the GPU by default (`--device`), and stays device-agnostic so it also runs on CPU. |
| 2026-11-19 | The MD networks are independent: they never import from or mention each other. Code they share lives in `clustering.py` and `trackcands.py`. |
| 2026-11-19 | MDTransformer vocabulary: a *patch* is any (η, φ) subset of an event; its *margin* is context around it. Inference uses a *patch grid*: patches with margin 0 that cover the event. |
| 2026-11-19 | MDTransformer settings, including the run's paths, live in one config yaml (`--config` is the only argument); Lightning saves the hyperparameters as `hparams.yaml`. |

## Running

From the repo root, inside the Apptainer image (`--nv` for the GPU):

```bash
apptainer exec --nv /ceph/users/atuna/ml-tracking.sif python3 -m mdml.<module>
apptainer exec /ceph/users/atuna/ml-tracking.sif python3 -m pytest mdml/test_<module>.py
```
