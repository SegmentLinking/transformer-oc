# harness/CLAUDE.md

Working rules for the evaluation harness. The design note, with what goes in and out, the pieces and the dated decisions, is `README.md`. Update its Decisions table whenever a decision is made or changed.

@README.md

## Rules

- **Reproduce LST exactly.** Matching, fakes and duplicates follow LST's code (`trkCore.cc`, `write_lst_ntuple.cc`,
  `performance.cc`). Check every change against LST's own flags on a nominal-mode ntuple (`all_sim_tracks` off):
  `t5_simIdx`, `t5_pMatched`, `t5_isFake` and `t5_isDuplicate` must agree for every T5. Differ from LST only when
  LST colleagues say we should, and record it in the Decisions table.
- **Sim indices beyond `sim_*` are pileup.** A nominal-mode ntuple stores only the first n_sims sims (hard
  scatter), but per-hit and per-T5 sim indices refer to the full list. Matching and fakes use every sim; only
  duplicates and the efficiency denominator are restricted to stored sims.
- **Test indices with `< 0`, never `== -999`.** LST code uses -1, -99, -999 and -9999. Write `UNMATCHED = -999`
  for an unmatched sim and `NO_HIT = -999` for hit padding, but always test with `< 0` (or keep with `>= 0`).
- **Say "track candidate", write `trackcand`.** In prose (docstrings, comments, READMEs) write "track candidate",
  never bare "candidate": in LST, "candidate" alone is ambiguous. In identifiers write `trackcand`
  (`n_trackcands_per_sim`, `trackcand_simIdx`); in CamelCase, spell it out (`EventTrackCandidates`). Not `tc`,
  which in the LSTNtuple means LST's own TC container.
- **Name per-track-candidate arrays like LST branches:** `trackcand_simIdx` after `t5_simIdx`,
  `trackcand_hitIndices` after `t5_hitIndices`.
- **Build piece by piece, bottom-up.** One small module at a time, each with its own `test_<module>.py`
  (pytest; `pytest.raises` for expected errors). Propose the outline (signatures and docstrings) first, then the
  bodies, and wait for review between pieces.

## Open questions for colleagues

Ask soon; don't sweep under the rug. When one is answered, record the answer in the README's Decisions table and
remove it here.

1. **`t5_pMatched` ignores extension hits.** It is computed from the 10 base hits (5 base MDs), while extended
   T5s store 12–14 hits in `t5_hitIndices`. Matching on all stored hits changes the matched sim for 0.6% of
   extended T5s (312 of 54329, 10 events). Is base-hit matching intended? Which does the tracking group use for
   T5 efficiency and fake rate?
2. **Hits of unsaved secondaries attributed to the parent sim?** 11% of prompt hard-scatter sims with pT ≥ 0.8
   GeV have an MD far off their helix (|Δφ| > 0.1 or |Δz| > 15 cm), often a second MD on a layer where the sim
   already has an on-helix one, and neutral sims (n, K0L, Λ) own MDs. Is CMSSW attributing hits of untracked
   secondaries (δ-rays, interaction products) to the parent TrackingParticle? Does LST's efficiency definition
   treat such hits specially?

## Running

From the repo root, inside the Apptainer image:

```bash
apptainer exec /ceph/users/atuna/ml-tracking.sif python3 -m pytest harness/test_<module>.py
```

Scripts that import `harness` from outside the repo (e.g. scratch checks) need
`apptainer exec --env PYTHONPATH=$PWD ...`.
