# MDML TODO

Open items for the MD-based networks. When an item is done or decided, check it and move it to Completed; record
decisions in `README.md`.

## Network inputs

- [ ] Add one-hot encoded inputs for the MD's inner and outer isMacroPixel flags to `MD_FEATURES` (`data.py`).
- [ ] Add additional datasets to the training
  - [ ] ZpToMM_m6000
  - [ ] QCD
  - [ ] muonCube

## Training

- [ ] Make the choice of training patch centres configurable: a random MD (now), or drawn at random from a uniform
  distribution in η (|η| ≤ 2.8) and φ. Random MDs put more patches in dense regions. The inference patch grid is
  not random (evenly spaced centres), but it gives every η/φ region one patch, so on average uniformly drawn
  centres would cover the detector the way inference does.
- [ ] Vertex regression (2026-11-22): train with `sim_vz_weight` and `sim_vtxperp_weight` > 0 and measure how well
  the network regresses each MD's sim track production vertex (`[md_quality]` keys `vz_res_iqr`,
  `vtxperp_res_iqr`). One MD alone should resolve it poorly (the 2S strips have little z information), but
  attention across the patch may do better. Decide from the resolution whether vertex outputs help the decoder.

## Code

- [ ] Speed up `MDTransformerNet.predict`, about 2 s of the harness's 2.2 s per event on the V100
  (run_2026_11_19_224953, first validation file). About 1.3 s of it is building the 84 grid patches on the CPU,
  each scanning all ~50k MDs of the event; the forward pass is 0.45–0.8 s. Ideas: sort MDs into grid cells once,
  build patches on the GPU, a larger `patches_per_batch`.
- [ ] Rename SimpleMDNet to MDMLP (2026-11-20, once the harness and the first-pass MDTransformer are stable). This
  covers the class, `simple_mdnet.py`, `--trackcands simple_mdnet` and the saved `simple_mdnet.pt`.

## Decisions

- [ ] Decide how to de-duplicate `harness/trackcands.py` `EventTrackCandidates` and `mdml/trackcands.py`
  `TrackCandidatesFromOneEvent` (copied on 2026-11-19 to keep mdml free of harness types; two identical classes
  will drift).

## Questions for colleagues

Listed in `../harness/CLAUDE.md`, "Open questions for colleagues" (open since 2026-11-16):

- [ ] `t5_pMatched` ignores T5 extension hits: is base-hit matching intended?
- [ ] Are hits of unsaved secondaries (δ-rays, interaction products) attributed to the parent sim track?
- [ ] Where are muonCube datasets?
- [ ] What datasets are typical for training beyond ttbar, QCD, ZpToMM_m6000, muonCube?

## Understanding

- [ ] Digest the helix starting direction φ0 = φ_MD + asin(0.0057 · r[cm] · q/pT[GeV⁻¹]), with
  0.0057 = 0.3 × 3.8 T / 2 / 100 (asked 2026-11-22). It drives the "helix φ0" decoder space in
  `scratch/mdtx/decode/`.
- [ ] Revisit the circle-through-origin geometry behind `md_pt` / `md_dphichange` and `bend()` in
  `studies/md_extrapolation_demo.py` (asked 2026-11-12). The hard part was φ_hit = φ0 − t/2.

## Completed
