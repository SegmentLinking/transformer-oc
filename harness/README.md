# Evaluation harness

Common metrics for track candidates. Every method's track candidates go through the
same code and come out as the same numbers, so "A is better than B" means the same thing every time. We are comparing MD-based models against the T5 model.

This file is the living design note: update it whenever a decision is made or changed.

## Performance references

The current performance measurements for T5s and MDs built from sim indices are:

| Track candidates | Efficiency | Fake rate |
|---|---|---|
| All T5s | 0.7882 ± 0.0022 | 0.0188 |
| T5s after cleaning | 0.7875 ± 0.0022 | 0.0079 |
| MDs matched to sim tracks (≥1 layer) | 0.9191 ± 0.0015 | 0 |
| MDs matched to sim tracks (≥2 layers) | 0.8983 ± 0.0016 | 0 |
| MDs matched to sim tracks (≥3 layers) | 0.8823 ± 0.0017 | 0 |
| MDs matched to sim tracks (≥4 layers) | 0.8660 ± 0.0018 | 0 |
| MDs matched to sim tracks (≥5 layers) | 0.8429 ± 0.0020 | 0 |

This provides important references for performance we can target with a MD-based ML approach. We should be able to reach 78 percent efficiency, and we probably cannot exceed 92 percent efficiency.

## What goes in

- **Track candidates**, per event: each track candidate is a set of LST hit indices, its matched sim, a score, and reconstructed q/pT, η and φ. Each builder assigns the sim from its own source:
  - T5s: LST's `t5_simIdx`
  - MDs matched to sim: the LST `md_simIdx` which matches to all MDs
  - SimpleMDNet: Per-hit sim indices from the `.pt` file
- **Truth**, from the LSTNtuple of the same event: the sim track properties, for the efficiency denominator. Use ntuples made in LST's nominal mode (`all_sim_tracks` off): they store only the hard-scatter sims, as the first n_sims entries of the full sim list, while per-hit sim indices refer to the full list. So a sim index >= n_sims (the length of the `sim_*` branches) is pileup, and a track candidate's matched sim can be beyond `sim_*`.

## What comes out

- **Efficiency:** of the sim tracks we care about (the denominator), the fraction found by at least one track candidate. Plotted vs pT, eta, and phi.
- **Fake rate:** the fraction of track candidates matched to no sim.
- **Duplicate rate:** the fraction of all track candidates (the same denominator as the fake rate) that are matched to a hard-scatter sim another track candidate is also matched to. A track candidate matched to a pileup sim is neither a fake nor a duplicate.

All on a fixed set of events, as `[tag] key=value` logs and pdf plots.

## Pieces, bottom-up

Each is one small module with its own test, checked against the previous implementation on the same events.

1. **`pmatched.py`:** over a track candidate's unique hits, count the hits of each sim (a hit with two sims counts for both). The track candidate matches a sim with more than 75% of its hits, as LST. Given a map from hit index to sims. Not used by T5s or the ground truth, which carry their own sim; used by SimpleMDNet (`mdml/simple_mdnet.py`). Check: reproduced LST's `t5_simIdx`.
2. **`metrics.py`:** the efficiency denominator (LST's type 7: charged, pT ≥ 1 GeV, |η| ≤ 2.4, vertex near the beam spot, hard scatter), then efficiency, fake rate and duplicate rate from the matches.
3. **`trackcands.py`:** an `EventTrackCandidates` dataclass of per-event arrays (not one object per track candidate: an event has about 60k T5s), and functions that build it from T5s and from the ground truth. SimpleMDNet builds its own (`SimpleMDNet.trackcands`).
4. **`performance.py`:** the driver: the dataset's LSTNtuple files from its file list (`train.txt`, `val.txt` or `test.txt` from `md_processing.py`; never reshuffled), then track candidates (T5s, the ground truth from MDs matched to sims, or SimpleMDNet, by `--trackcands`) through 2–3, with plots.

Later: comparison at matched efficiency (tune score thresholds on the training set), and the decision rule.

## Decisions

| Date | Decision |
|---|---|
| 2026-11-14 | Track candidates are sets of hits, not MDs: a hit can be in several MDs, and T5s and models then share one path. Revisit if colleagues expect the MD model to report MDs. |
| 2026-11-14 | Closed 2026-11-15 (was open: whether LST's duplicate-cleaned T5s count). Decided when writing `performance.py`: all T5s by default, with `--drop-duplicate-cleaned` to try later (see the 2026-11-15 row). |
| 2026-11-14 | Fake and duplicate follow LST exactly (`write_lst_ntuple.cc`, `performance.cc`): the duplicate rate is over all track candidates. Check: reproduce `t5_isFake` and `t5_isDuplicate`. |
| 2026-11-14 | Closed (was open: which sims count for duplicates). Evaluate on nominal-mode ntuples. Matching and fakes use every sim, pileup included; duplicates count only stored sims (index < n_sims), which in nominal mode are the hard scatter. The code doesn't depend on the mode: on an all-sim ntuple every sim is stored, and the same code reproduces that file's flags. On 10 nominal events: 69863 / 69863 T5s agree with `t5_isFake`, `t5_isDuplicate` and `t5_simIdx`. |
| 2026-11-14 | Superseded 2026-11-17 (see that row). Hit truth comes from OT MDs only (pLS MDs skipped); hits with no sim are left out of the hit → sims map. |
| 2026-11-14 | Unmatched is written as `UNMATCHED = -999` (LST's value) but always tested as `sim < 0`, since LST code uses several negative values. |
| 2026-11-14 | Superseded 2026-11-17 (no matching in `performance.py`). Track candidates are matched one at a time (a Python loop); revisit if it's too slow. |
| 2026-11-14 | Matching must reproduce LST's `t5_simIdx` exactly, unless LST colleagues say it shouldn't. |
| 2026-11-14 | `best_sim` also reproduces LST's pMatched exactly: hits with no sim compete as one more "sim" (fraction 0 if they outnumber the best sim, or tie and appear first). Costs order dependence in that tie; never changes a match. On 10 events: 15534 / 15534 5-layer T5s agree with `t5_pMatched`. |
| 2026-11-14 | Rebuilt piece by piece with review; the previous version is kept outside the repo for comparison. |
| 2026-11-15 | A T5 track candidate is its 10 base hits (5 base MDs), as LST matches T5s; the extension-layer hits that `t5_hitIndices` appends are left out. Open: how to treat extension hits (asking colleagues). |
| 2026-11-15 | All T5s are track candidates by default (focus on efficiency and fake rate). `trackcands_from_t5s(drop_duplicate_cleaned=True)` drops those removed by LST's T5-T5 duplicate cleaning (`t5_isDupBitmask` bits 1, 2), to try later. |
| 2026-11-15 | Track candidate hits are a numpy int array, shape (n_trackcands, n_hits_max), padded with `NO_HIT = -999`, so every `EventTrackCandidates` field is numpy. Strip padding (`hit >= 0`) before matching: a padded slot would count as a hit with no sim. T5s use exactly 10 base hits, so no padding yet. |
| 2026-11-15 | The manifest records LSTNtuple file names, not paths. The same inputs exist as all-sim ntuples (training) and nominal ntuples (evaluation), with the same names and event order. The harness joins `lstntuple_file` with the nominal directory it's given. Checked: one file entry by entry against both all-sim copies (100 / 100 entries agree on the number of MDs, the number of T5s and the sum of T5 pT), and 84 / 84 manifest events in 28 finished nominal files have as many OT MDs as their `.pt` files. |
| 2026-11-15 | Ground-truth check of the harness: `trackcands_from_mds_matched_to_sim` makes one track candidate per stored sim from the hits of its OT MDs with `md_simIdx` equal to the sim (both hits from it), optionally requiring `min_layers` unique `md_layer` values. Expect no fakes, and efficiency equal to the fraction of denominator sims with such an MD. A check from hits matched to sims (no MDs) would need per-hit LSTNtuple branches (each OT hit's sims and layer by LST hit index), which the LSTNtuple doesn't have: a change to `write_lst_ntuple.cc`, to raise with colleagues. |
| 2026-11-16 | Datasets are made of whole LSTNtuple files (agreed with Jade): `md_processing.py` shuffles the name-sorted files with the seed and cuts 70 / 15 / 15% of files into train / val / test (30 / 7 / 7 of 44). Each dataset has a file list (`train.txt`, `val.txt`, `test.txt`: one LSTNtuple file name per line, replacing the manifest on 2026-11-17), and the harness evaluates every entry of the listed files (`--file-list`, `--n-files`). No tuning set: thresholds are tuned on train, since the datasets are large and more can be produced. |
| 2026-11-17 | Each track candidate builder assigns its track candidates' sims from its own source, so sources of truth are never mixed: T5s take LST's `t5_simIdx`, and each ground-truth track candidate the sim it was built from. `performance.py` no longer matches hits to sims, and `md_truth.py` (`hit_sims_from_mds`) is deleted: its map covered only hits in OT MDs, not every hit as a global hit → sims map (LST's `matchedSimTrkIdxsAndFracs`) would, and the name hid that. The harness's own matching reproduced `t5_simIdx` for 4,851,284 / 4,851,284 T5s (660 events), and on one val file both T5 and ground-truth totals are unchanged; the T5 run takes 11 s instead of 51 s. `pmatched.py` stays for the MD model, whose hit → sims map should come from `md_anchor_simIdxs` / `md_other_simIdxs` (kept in `MD_COLUMNS`). |
| 2026-11-17 | SimpleMDNet (`mdml/simple_mdnet.py`) is evaluated with `--trackcands simple_mdnet --simple-mdnet <simple_mdnet.pt>`. `performance.py` loads it once (on the CPU) and, per event, calls `SimpleMDNet.trackcands` on the event's `.pt` file from `md_processing.py`. The builder lives in `performance.py`, not `trackcands.py`, because `mdml` imports `harness.trackcands`. SimpleMDNet assigns its sims itself, with `pmatched.py` on a hit -> sims map built per track candidate from its own MDs (`.pt` `md_inner_simIdxs` / `md_outer_simIdxs`): complete for these track candidates, which are made only of those hits. This replaces the plan in the row above to use the LSTNtuple's `md_anchor_simIdxs` / `md_other_simIdxs`. First run, `epoch_4` with fake cut and eps each at 50% (chosen on val), one val file: efficiency 0.078, fake rate 0.829, duplicate rate 0.0001 (T5: 0.778, 0.018, 0.052). |
| 2026-11-18 | `trackcand_matching.py` is renamed `pmatched.py`, after LST's pMatched (the matched fraction, as in `t5_pMatched`), a name LST readers know. Its functions (`best_sim`, `matched_sim`) keep their names. |
| 2026-11-18 | `events.py` is folded into `performance.py`: its only function, `read_file_list`, was two lines and only `performance.py` calls it. Its tests stay in `test_harness.py`, which now imports `harness.performance` and so also needs `mdml`, `uproot` and `matplotlib`. The file lists hold LSTNtuple file names, not paths, for the reason in the 2026-11-15 manifest row. |

