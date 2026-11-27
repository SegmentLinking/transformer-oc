# MDML TODO

Open items for the MD-based networks. Remove an item when it is done or decided; record decisions in
`README.md`.

## Network inputs

- Add one-hot encoded inputs for the MD's inner and outer isMacroPixel flags to `MD_FEATURES` (`data.py`).

## Decisions

- Decide how to de-duplicate `harness/trackcands.py` `EventTrackCandidates` and `mdml/trackcands.py`
  `TrackCandidatesFromOneEvent` (copied on 2026-11-19 to keep mdml free of harness types; two identical classes
  will drift).

## Questions for colleagues

Listed in `../harness/CLAUDE.md`, "Open questions for colleagues" (open since 2026-11-16):

- `t5_pMatched` ignores T5 extension hits: is base-hit matching intended?
- Are hits of unsaved secondaries (δ-rays, interaction products) attributed to the parent sim track?

## Understanding

- Digest the helix starting direction φ0 = φ_MD + asin(0.0057 · r[cm] · q/pT[GeV⁻¹]), with
  0.0057 = 0.3 × 3.8 T / 2 / 100 (asked 2026-11-22). It drives the "helix φ0" decoder space in
  `scratch/mdtx/decode/`.
- Revisit the circle-through-origin geometry behind `md_pt` / `md_dphichange` and `bend()` in
  `studies/md_extrapolation_demo.py` (asked 2026-11-12). The hard part was φ_hit = φ0 − t/2.
