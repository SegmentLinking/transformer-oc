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

1. In progress.

## Pieces, bottom-up

Each is one small module with its own test.

1. **`data.py`:** a `MDsFromOneEvent` dataclass read from one `.pt` file, and the `.pt` paths of a dataset's
   directory (`train/` holds only training events, and so on).
2. **`simple_mdnet.py`:** The `SimpleMDNet` class owns the MDNet and the working point (`fake_cut`, `eps`); it trains the MDNet, chooses the working point, and turns `.pt` files into one harness `EventTrackCandidates` per event, with every field filled. Running the module is the end-to-end command: train, then choose the working point, into one run directory.
3. **Harness:** `--trackcands simple_mdnet` in `harness/performance.py`, calling `SimpleMDNet.trackcands`.

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

## Running

From the repo root, inside the Apptainer image (`--nv` for the GPU):

```bash
apptainer exec --nv /ceph/users/atuna/ml-tracking.sif python3 -m mdml.<module>
apptainer exec /ceph/users/atuna/ml-tracking.sif python3 -m pytest mdml/test_<module>.py
```
