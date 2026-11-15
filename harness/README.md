# Evaluation harness

Common metrics for track candidates. Every method's candidates (LST T5s now, the MD model later) go through the
same code and come out as the same numbers, so "A is better than B" means the same thing every time.

This file is the living design note: update it whenever a decision is made or changed.

## What goes in

- **Track candidates**, per event: each track candidate is a set of LST hit indices, a score, and reconstructed pT,
  η and φ.
- **Truth**, from the LSTNtuple of the same event: the sims each hit belongs to, and the sims' properties. Use
  ntuples made in LST's nominal mode (`all_sim_tracks` off): they store only the hard-scatter sims, as the first
  n_sims entries of the full sim list, while per-hit sim indices refer to the full list. So a sim index >= n_sims
  (the length of the `sim_*` branches) is pileup, and a track candidate's matched sim can be beyond `sim_*`.

## What comes out

- **Efficiency:** of the sims we care about (the denominator), the fraction found by at least one track candidate.
  Plotted vs pT, η, and the number of OT layers on which the sim has a true MD.
- **Fake rate:** the fraction of track candidates matched to no sim.
- **Duplicate rate:** the fraction of all track candidates (the same denominator as the fake rate) that are matched
  to a hard-scatter sim another track candidate is also matched to. A track candidate matched to a pileup sim is
  neither a fake nor a duplicate.

All on a fixed set of events, as `[tag] key=value` logs and pdf plots.

## Pieces, bottom-up

Each is one small module with its own test, checked against the previous implementation on the same events.

1. **`md_truth.py`:** which sims each OT hit belongs to, from the ntuple's per-hit MD branches (pLS MDs skipped).
2. **`trackcand_matching.py`:** over a track candidate's unique hits, count the hits of each sim (a hit with two
   sims counts for both). The track candidate matches a sim with more than 75% of its hits, as LST.
   Check: reproduce LST's `t5_simIdx`.
3. **`metrics.py`:** the efficiency denominator (LST's type 7: charged, pT ≥ 1 GeV, |η| ≤ 2.4, vertex near the
   beam spot, hard scatter), then efficiency, fake rate and duplicate rate from the matches.
4. **`trackcands.py`:** an `EventTrackCandidates` dataclass of per-event arrays (not one object per track
   candidate: an event has about 60k T5s), and functions that build it from T5s and, later, from a model's output.
5. **`events.py`:** which events form the train, validation, tuning and test sets, from the manifest. Never
   reshuffled.
6. **`score_t5.py`:** the first driver: T5s through 1–5, with plots.

Later: comparison at matched efficiency (tune score thresholds on the tuning set), and the decision rule.

## Decisions

| Date | Decision |
|---|---|
| 2026-11-14 | Track candidates are sets of hits, not MDs: a hit can be in several MDs, and T5s and models then share one path. Revisit if colleagues expect the MD model to report MDs. |
| 2026-11-14 | The T5 selection (whether LST's duplicate-cleaned T5s count) stays open: decide it head-on when writing `score_t5.py`. |
| 2026-11-14 | Fake and duplicate follow LST exactly (`write_lst_ntuple.cc`, `performance.cc`): the duplicate rate is over all track candidates. Check: reproduce `t5_isFake` and `t5_isDuplicate`. |
| 2026-11-14 | Closed (was open: which sims count for duplicates). Evaluate on nominal-mode ntuples. Matching and fakes use every sim, pileup included; duplicates count only stored sims (index < n_sims), which in nominal mode are the hard scatter. The code doesn't depend on the mode: on an all-sim ntuple every sim is stored, and the same code reproduces that file's flags. On 10 nominal events: 69863 / 69863 T5s agree with `t5_isFake`, `t5_isDuplicate` and `t5_simIdx`. |
| 2026-11-14 | Hit truth comes from OT MDs only (pLS MDs skipped); hits with no sim are left out of the hit → sims map. |
| 2026-11-14 | Unmatched is written as `UNMATCHED = -999` (LST's value) but always tested as `sim < 0`, since LST code uses several negative values. |
| 2026-11-14 | Track candidates are matched one at a time (a Python loop); revisit if it's too slow. |
| 2026-11-14 | Matching must reproduce LST's `t5_simIdx` exactly, unless LST colleagues say it shouldn't. |
| 2026-11-14 | `best_sim` also reproduces LST's pMatched exactly: hits with no sim compete as one more "sim" (fraction 0 if they outnumber the best sim, or tie and appear first). Costs order dependence in that tie; never changes a match. On 10 events: 15534 / 15534 5-layer T5s agree with `t5_pMatched`. |
| 2026-11-14 | Rebuilt piece by piece with review; the previous version is kept outside the repo for comparison. |
| 2026-11-15 | A T5 track candidate is its 10 base hits (5 base MDs), as LST matches T5s; the extension-layer hits that `t5_hitIndices` appends are left out. Open: how to treat extension hits (asking colleagues). |
| 2026-11-15 | All T5s are track candidates by default (focus on efficiency and fake rate). `trackcands_from_t5s(drop_duplicate_cleaned=True)` drops those removed by LST's T5-T5 duplicate cleaning (`t5_isDupBitmask` bits 1, 2), to try later. |
| 2026-11-15 | Track candidate hits are a numpy int array, shape (n_trackcands, n_hits_max), padded with `NO_HIT = -999`, so every `EventTrackCandidates` field is numpy. Strip padding (`hit >= 0`) before matching: a padded slot would count as a hit with no sim. T5s use exactly 10 base hits, so no padding yet. |

## Running

From the repo root, inside the Apptainer image: `python3 -m harness.<module>`.
