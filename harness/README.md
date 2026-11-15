# Evaluation harness

Common metrics for track candidates. Every method's candidates (LST T5s now, the MD model later) go through the
same code and come out as the same numbers, so "A is better than B" means the same thing every time.

This file is the living design note: update it whenever a decision is made or changed.

## What goes in

- **Candidates**, per event: each candidate is a set of LST hit indices, a score, and reconstructed pT, η and φ.
- **Truth**, from the LSTNtuple of the same event: the sims each hit belongs to, and the sims' properties.

## What comes out

- **Efficiency:** of the sims we care about (the denominator), the fraction found by at least one candidate. Plotted
  vs pT, η, and the number of OT layers on which the sim has a true MD.
- **Fake rate:** the fraction of candidates matched to no sim.
- **Duplicate rate:** of the matched candidates, the fraction whose sim is also found by another candidate.

All on a fixed set of events, as `[tag] key=value` logs and pdf plots.

## Pieces, bottom-up

Each is one small module with its own test, checked against the previous implementation on the same events.

1. **`md_truth.py`:** which sims each OT hit belongs to, from the ntuple's per-hit MD branches (pLS MDs skipped).
2. **`matching.py`:** over a candidate's unique hits, count the hits of each sim (a hit with two sims counts for
   both). The candidate matches a sim with more than 75% of its hits, as LST. Check: reproduce LST's `t5_simIdx`.
3. **`metrics.py`:** the efficiency denominator (LST's type 7: charged, pT ≥ 1 GeV, |η| ≤ 2.4, vertex near the
   beam spot, hard scatter), then efficiency, fake rate and duplicate rate from the matches.
4. **`candidates.py`:** an `EventCandidates` dataclass of per-event arrays (not one object per candidate: an
   event has about 60k T5s), and functions that build it from T5s and, later, from a model's output.
5. **`events.py`:** which events form the train, validation, tuning and test sets, from the manifest. Never
   reshuffled.
6. **`score_t5.py`:** the first driver: T5s through 1–5, with plots.

Later: comparison at matched efficiency (tune score thresholds on the tuning set), and the decision rule.

## Decisions

| Date | Decision |
|---|---|
| 2026-11-14 | Candidates are sets of hits, not MDs: a hit can be in several MDs, and T5s and models then share one path. Revisit if colleagues expect the MD model to report MDs. |
| 2026-11-14 | The T5 selection (whether LST's duplicate-cleaned T5s count) stays open: decide it head-on when writing `score_t5.py`. |
| 2026-11-14 | Duplicate rate is per candidate for now; confirm against the tracking group's definition when writing `metrics.py`. |
| 2026-11-14 | Hit truth comes from OT MDs only (pLS MDs skipped); hits with no sim are left out of the hit → sims map. |
| 2026-11-14 | Rebuilt piece by piece with review; the previous version is kept outside the repo for comparison. |

## Running

From the repo root, inside the Apptainer image: `python3 -m harness.<module>`.
