# harness/CLAUDE.md

Working rules for the evaluation harness. The design note, with what goes in and out, the pieces and the dated
decisions, is `README.md`. Update its Decisions table whenever a decision is made or changed.

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
  (plain asserts, runnable without pytest). Propose the outline (signatures and docstrings) first, then the
  bodies, and wait for review between pieces.

## Running

From the repo root, inside the Apptainer image:

```bash
apptainer exec /ceph/users/atuna/ml-tracking.sif python3 -m harness.test_<module>
```

Scripts that import `harness` from outside the repo (e.g. scratch checks) need
`apptainer exec --env PYTHONPATH=$PWD ...`.
