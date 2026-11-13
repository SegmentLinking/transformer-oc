# MDML: grouping outer-tracker MDs into track candidates

A running record of the plan, decisions, constraints and studies for an MD-based alternative to the T3-based
sparse-attention transformer in this repo. Keep it current: decisions get a date, open questions move to
"Decided" when settled, and superseded items move to "Gone stale" at the bottom.

Last updated: 2026-11-10

## Goal

Group outer-tracker mini-doublets (MDs) of one event into track candidates with a sparse-attention network, and
judge it by the tracking group's standard track-level metrics (efficiency, fake rate, duplicate rate) against
LST T5s, at efficiency tuned to match as a function of sim pT and η.

## Decided

| Date | Decision |
|---|---|
| 2026-11-10 | Input features gain the MD direction-vs-position angle (as a q/pT estimate) and η. |
| 2026-11-10 | Layer goes in as a learned embedding, not a float. No separate ±z flag (the sign of `z` carries it) and no barrel/endcap flag (the layer embedding covers it). |
| 2026-11-10 | Normalization is physically motivated where possible (e.g. η / 2.5), not dataset min/max. |
| 2026-11-10 | No φ-rotation or z-flip augmentation, and absolute φ stays an input: the real detector is not exactly φ- or z-symmetric. |
| 2026-11-10 | Build the neighbourhood as a pluggable component and try both global ΔR kNN and layer-aware kNN; pick by the metrics. |
| 2026-11-10 | Learned geometry bias in attention; about 6 attention layers to start. |
| 2026-11-10 | Grouping is not object condensation. Hybrid regression/embedding (see Design). Regression target is sim q/pT in GeV⁻¹ directly (±1.25 at 0.8 GeV), for interpretability. |
| 2026-11-10 | Track candidates need at least 3 MDs. |
| 2026-11-10 | Candidate making has an efficiency that is tunable vs pT and η (see Design). Comparisons are made at matched efficiency, then fake and duplicate rates are compared. |
| 2026-11-10 | Primary metrics: track-level efficiency, fake rate, duplicate rate, with **hit-level 75% sim matching** (not MD-level; MD-level was a mistake in the T3 work). Graph-level metrics are diagnostics only. |
| 2026-11-10 | Baseline: LST T5s. T4 + T5 later. No comparison against the T3 ML network for now. |
| 2026-11-10 | Align with LST standards wherever possible. Efficiency denominator is LST's (pT ≥ 1 GeV, see harness), with no reconstructability cut; add a breakdown by number of OT layers with a true MD. |
| 2026-11-10 | Event split of the 4500 available ttbar events: 2700 train, 600 validation (training loss / early stopping), 200 tuning (harness thresholds), 1000 test (reported numbers). Produce more ttbar if over-training shows up. |
| 2026-11-10 | Fake rate is the priority metric over duplicate rate. |
| 2026-11-10 | Efficiency denominator is hard scatter only; pileup efficiency is not a goal. Pileup truth may still be used in training if it helps. |
| 2026-11-10 | The T5 baseline is the extended T5s (up to 7 layers) as LST outputs them, matched on all stored hits (`t5_hitIndices`), not `t5_pMatched`. |
| 2026-11-10 | Provisional OT-only T5 baseline: `ot_cleaned` (T5s with `t5_isDupBitmask & 3 == 0`, i.e. surviving LST's T5-T5 cleaning, with or without a pixel seed). Quote mainly efficiency and fake rate; its duplicate rate (~81%) is not a fair target. Revisit when colleagues answer question 5. |
| 2026-11-10 | Sims with pT < `pt_min` (default 0.8 GeV, configurable) are treated as noise in training. Alternatives for later: leave them out of the loss, or use them as positives. |
| 2026-11-10 | Next topic after the current items: define the comparison harness in detail. |
| 2026-11-11 | q/pT refinement before the road (dynamic graph) is planned for the near future, but the first model starts without it: a static road from the MD's own q/pT. |
| 2026-11-12 | Neighbourhood centre on each layer: the helix with the MD's own q/pT (`-1 / md_pt`) from a vertex at (0, 0, 0) through the MD, followed to the layer. It shifts φ (the bend) and η (the longer curved path). It is the best extrapolation available at the start of training; refine later (q/pT refinement, z0 from MD pairs). `studies/md_extrapolation_demo.py` shows own, helix and oracle by default; `--detailed` adds the helix with the sim's q/pT and the helix from the sim's vertex. |

## Constraints

- Training must run on CPU too; keep code device-agnostic, no new CUDA-only dependencies. Relaxable (user,
  2026-11-10) if a GPU-only piece (e.g. block-sparse attention kernels) proves crucial; test when ready.
- Hardware: phi3 Tesla V100-PCIE 16 GB (free use) for development; NRP A100-SXM4 80 GB (one GPU) for larger
  runs. `kubectl` is not on phi3's PATH (a `~/.kube/config` exists), so NRP jobs are submitted by the user for
  now.
- Outer tracker only: pixel LS MDs (`md_isPLS`) are dropped, so absolute efficiency is not comparable to full LST.
- Event size is never a reason to drop a physics gain (user, 2026-11-10): an event can always be split into η/φ
  chunks with overlapping margins that fit on the GPU, and the chunks recombined into one event reconstruction.
  Whole events on the GPU are nicer, not required. Recombination needs a rule for candidates in the margins
  (e.g. keep a candidate only from the chunk whose core contains its fitted η/φ, then merge duplicates).

## Data

MD dataset from `transformer-datasets/src/transformer_datasets/md_processing.py`, one file per event:

- `x` (n_md, 8): `[layer, rt, cosphi, sinphi, z, drt, dphi, dz]`, positions of the inner hit (cm),
  dX = outer − inner.
- `sim_index` (n_md,): `md_simIdx`, `-999` if unmatched.
- `sim_features` (n_md, 10): `sim_pt, sim_eta, sim_phi, sim_q, sim_vx, sim_vy, sim_vz, sim_vtxperp, sim_event,
  sim_bunchCrossing` of the matched sim.
- `hit_index` (n_md, 2): LST hit index of the [inner, outer] hit, same index space as `t5_hitIndices`
  (added 2026-11-10).
- `hit_sim_pairs` (n_pairs, 2): unique [hit index, sim index] pairs over all hits of the kept MDs, no fraction
  cut; a hit with no sim has no pair (added 2026-11-10). Hit-level matching of a candidate: take its unique hit
  indices, select their pairs, count hits per sim.
- Needs an LSTNtuple with the per-hit MD branches (see MD truth); older ntuples no longer process.

Test ntuple with T4/T5 branches (100 events):
`transformer-datasets/samples/ttbar/LSTNtuple_101b722e-bc54-4136-b7c4-1204e1373674.root`.

Numbers from a few ttbar PU200 events (`transformer-datasets/samples/ttbar/train`):

- 40k–54k MDs per event (cf. ~60k T3s).
- 51–53% of MDs are "unmatched". This means *not both hits from one sim* (see MD truth below), so it includes
  MDs with one good hit.
- ~1.5k–2k sims with pT > 0.8 GeV per event, median 7 MDs each.

### MD truth: what the ntuple has, and the gap

- `md_simIdx`, `md_simIdxAll`, `md_simIdxAllFrac` come from LST's `matchedSimTrkIdxsAndFracs`
  (`trkCore.cc`), called with the default `matchfrac = 0.75` (`trkCore.h`). The function only returns sims with
  `percent_matched > matchfrac` (strict), so every `*_simIdxAll` branch lists only sims passing the 75% match,
  despite the name. An MD has 2 hits, so only 2/2 passes: `md_simIdxAllFrac` is always 1.0. Same pattern
  elsewhere: `ls_simIdxAllFrac` only 1.0 (3/4 fails the strict cut), `t3` only 0.833 and 1.0, `t5` only 0.8–1.0,
  while `t5_pMatched` (the max fraction, before the cut) also has 0.4–0.7.
- Lowering `matchfrac` at the MD call site would not fix it: each hit-sim combination counts only its single
  most common sim, ties go to whichever comes first, and a tie won by "no sim" (-1) is dropped. A mixed MD
  would list at most one of its sims, depending on hit order.
- There is no per-hit sim information and no MD hit index in the ntuple. **Hit-level matching of MD
  candidates is not possible yet.**
- Ntuple addition, 2026-11-10, in
  `CMSSW_20_1_0_pre5/src/RecoTracker/LSTCore/standalone/code/core/write_lst_ntuple.cc` (uncommitted). The test
  ntuple has been regenerated with it:
  - `md_anchor_hitIdx`, `md_other_hitIdx` (int): LST SoA hit index, the same index space as `t5_hitIndices`.
    For OT hits this equals the `ph2_*` index (`hitOrigIdx` is the identity for OT).
  - `md_anchor_simIdxs`, `md_other_simIdxs` (vector of int): all sims of the hit's sim hits
    (`ph2_simHitIdx` → `simhit_simTrkIdx`, `pix_*` for pixel hits), no fraction cut, empty if none.
  - Hit indices are also needed to de-duplicate hits shared between MDs of one candidate.
  - Validated on 3 events of the test ntuple (one-off script, not kept). Zero failures: every MD with
    `md_simIdx = s` has `s` in both hit lists; no MD with a sim common to both hits is unmatched; the same hit
    index always has the same coordinates and sims; no negative sims in the lists; every T5 hit index is an MD
    hit index. For 10-hit T5s, the best sim's share of hits recomputed from these branches matches `t5_pMatched`
    in 1929 of 1933 cases (1 event).
- OT MD composition (3 events, 144k OT MDs): true (both hits from one sim) 47%, two different sims 19%, one sim
  + one noise hit 11%, both hits noise 22%. About 6% of hits carry more than one sim.
- **T5s in this LST release are extended:** 5 base layers plus up to 2 layers taken from overlapping duplicate
  T5s (`tryExtendT5` / `ExtendT5FromDupT5ByMD` in `src/alpaka/Kernels.h`; `Params_T5::kLayers = 7`,
  `kHits = 14` in `interface/Common.h`; CMSSW commit `de7d518f960` "extend T5's after build").
  - 1 event: 1933 T5s with 5 layers, 6808 with 6, 241 with 7. `t5_hitIndices` and `t5_logicalLayers` always
    have 2 × `t5_nLayers` and `t5_nLayers` entries; no logical layer repeats within a T5.
  - `t5_pMatched` uses `getHitsFromT5` → `getMDsFromT5` (via the two T3s), i.e. only the 10 base hits, so it
    ignores extension hits. For extended T5s it disagrees with the purity of the full stored hit list in ~25%
    of cases.
  - pT5s use the extended layers (`PixelQuintuplet.h` reads `nLayers`), so downstream LST sees the full list.
  - Decided: the baseline is extended T5s on all stored hits.
- Hit-level truth is required for evaluation (decided 2026-11-10): MD-level matching is a poor stand-in, a main
  lesson of the T3 ML studies.
- T5s already carry hit-level matching: `t5_simIdxAllFrac` and `t5_pMatched` come in steps of 0.1 (10 hits),
  and `t5_hitIndices` is stored.

### Anchor vs inner hit

- In PS modules the anchor is the macro-pixel hit and the other hit is the strip hit. PS modules are mounted
  flipped half of the time, so the anchor is the inner hit for ~50% of MDs in barrel layers 1–3 (measured 0.49)
  and 68–75% in the endcap (mix of PS and 2S rings). In 2S modules both hits are strips and LST takes the anchor
  as the inner (lower) hit (measured 1.00 in layers 4–6). (Explanation from the user, 2026-11-10.)
- η feature: anchor η kept (decided 2026-11-10). Measured 68% |Δη| vs sim η on 3 events, true MDs with pT > 0.8:
  anchor is never worse than inner (by 0–5%). Both are dominated by the vertex z spread (η is computed from
  z = 0): 0.10 at layer 1, ~0.065 at layers 4–6, ~0.04 in the endcap.

### Layer geometry (measured from MD inner hits, 5 events, cm)

| Layer | rt min | rt median | rt max | \|z\| min | \|z\| median | \|z\| max |
|---|---|---|---|---|---|---|
| 1 | 21.6 | 24.6 | 27.9 | 0.1 | 37.8 | 119.9 |
| 2 | 34.6 | 37.0 | 39.8 | 0.1 | 43.7 | 119.9 |
| 3 | 49.8 | 52.2 | 54.8 | 0.1 | 43.3 | 119.9 |
| 4 | 66.8 | 67.4 | 70.6 | 2.2 | 57.1 | 115.2 |
| 5 | 84.1 | 84.7 | 87.8 | 2.2 | 57.1 | 115.2 |
| 6 | 106.4 | 106.9 | 110.1 | 2.2 | 56.9 | 115.2 |
| 7 | 23.7 | 52.8 | 104.9 | 128.6 | 130.3 | 133.4 |
| 8 | 23.7 | 53.8 | 104.9 | 152.5 | 154.1 | 157.2 |
| 9 | 33.2 | 62.4 | 104.9 | 182.7 | 184.4 | 187.6 |
| 10 | 33.2 | 61.6 | 104.9 | 219.0 | 220.7 | 223.9 |
| 11 | 33.2 | 64.0 | 104.9 | 262.4 | 264.1 | 267.2 |

### Layer transitions along sim tracks (`studies/md_layer_transitions.py`, plots in `md_plots/layer_transitions_*`, not in git)

Sim pT ≥ 0.8 GeV, 100 events, MD-level truth. MDs of a sim ordered by the inner hit's sqrt(rt² + z²).

- Same layer → same layer (overlapping modules) is 17–30% of transitions from every layer. Several MDs per layer
  per track is normal, so candidate building must not assume one MD per layer.
- Barrel L → L+1: 56–63%. Skips (L → L+2) are ≤ 3% in the barrel, up to 7% in the endcap (8 → 10).
- Barrel 1–5 → endcap 7: 8–15%. Barrel 6 → 7: 0.00. Endcap → barrel: ~0.
- Track ends: 68% after layer 6, 81% after layer 11, and 14% after layer 5. Some ends and skips will be MD
  inefficiency or MDs with only one matched hit, which MD-level truth does not see.
- Adjacency for a layer-aware neighbourhood (both directions): same layer; L ± 1; L ± 2 in the endcap;
  barrel 1–5 ↔ 7.

### Sibling distances (`studies/md_sibling_plots.py`, plots in `md_plots/`, not in git)

- At 0.8–1 GeV only ~half of an MD's siblings are within ΔR < 0.1; the farthest sibling's ΔR rank k has a
  median of ~500–1000. One hop cannot see a whole track.
- The next-layer sibling reaches ~90% at k ≈ 100 (0.8–1 GeV) and k ≈ 20–30 (3–4 GeV) in global ΔR rank.
- The T3 settings (`dr_threshold: 0.1`, `max_neighbors: 16`) are far too small for MDs.

### Neighbourhood size: global vs layer-aware kNN (`studies/md_neighbour_rank.py`, plots in `md_plots/kept_fraction_vs_k`, `md_plots/link_dr_cdf`, not in git)

50 events (old MD files, MD-level truth), hard-scatter sims with pT ≥ 0.8 GeV, ~20k links. A link goes from a
true MD to the same sim's nearest-ΔR MD on the next higher layer. Rank = k needed to keep the link, counting all
MDs (pileup included): global = all MDs, layer-aware = MDs on the target layer only. Inner-hit η/φ.

| Links kept | 0.8–1 GeV global | 0.8–1 GeV layer-aware | 1–2 GeV global | 1–2 GeV layer-aware | ≥ 2 GeV global | ≥ 2 GeV layer-aware |
|---|---|---|---|---|---|---|
| 90% | 153 | 27 | 74 | 13 | 34 | 7 |
| 99% | 1384 | 148 | 702 | 75 | 653 | 63 |

- Layer-aware needs ~6–10× smaller k per target layer. Its total cost is k × number of adjacent layers (3–5
  with the adjacency above), so it is still ~2–3× cheaper than global at the same link recall.
- Global kNN at 99% recall (k ~ 1000+) is not affordable; global k ≈ 128 keeps ~88% of 0.8–1 GeV links.
- ΔR of the links: 99% within 0.53 (0.8–1 GeV), 0.39 (1–2), 0.35 (≥ 2). The T3 `dr_threshold: 0.1` keeps only
  ~60% of 0.8–1 GeV links; use ≥ 0.5 or rely on k alone.
- barrel→endcap links have a ~5% tail at ΔR > 0.3 at every pT (also the 1% tail elsewhere): not curvature.
  Suspects: loopers or secondaries sharing a sim index, where "next layer" picks a far piece. To check.
- Per-link recall understates track connectivity: attention over ~6 layers can route around a missed link via
  overlap MDs (same layer) and skip links. The quantity that matters is whether each track stays connected in
  the neighbour graph; measure that next.

#### Curvature-aware layer kNN (2026-11-10, same script, 50 new-format events from one ntuple)

The query point on the target layer is shifted in φ to where a helix from the beam line with the source MD's q/pT
crosses that layer: φ(r) = φ0 − asin(k · r · q/pT), k = 2.99792458e-3 · 3.8 / 2 (sign checked against sim:
median |Δφ| between MDs of a track 0.15 → 0.004 rad with the true q/pT). Target radius: the layer's median rt
(barrel), or a straight line in r-z from the beam line (endcap). `layer_curv_md` uses the MD's own q/pT estimate,
`layer_curv_sim` the sim's true q/pT (oracle).

| Links kept, k per target layer | 0.8–1 GeV layer | 0.8–1 GeV curv_md | 0.8–1 GeV curv_sim | 1–2 GeV layer | 1–2 GeV curv_md | ≥ 2 GeV layer | ≥ 2 GeV curv_md |
|---|---|---|---|---|---|---|---|
| k = 1 | 10% | 52% | 53% | 19% | 59% | 46% | 58% |
| 90% | 27 | 7 | 6 | 13 | 5 | 7 | 6 |
| 99% | 137 | 120 | 77 | 76 | 72 | 53 | 65 |

- The shift collapses the core: at 0.8–1 GeV the nearest MD on the next layer (k = 1) is the right one for 52%
  of links, vs 10% without the shift. The MD's own q/pT estimate is almost as good as the truth.
- Curvature-aware layer kNN with k = 32 per layer keeps 97.9% of all links, k = 64 keeps 98.8%.
- The last ~1–2% are not helix-like: 79% of links with rank > 64 (97% with rank > 128) have ΔR > 0.3, mostly
  barrel 4→5 and 5→6, 1→2 and 2→3, and barrel→endcap. Larger k buys little for these. Explained in "The non-helix
  tail" below: displaced sims and MDs off their own sim's helix.
- After the shift, Δη dominates the residual (median 0.024 vs Δφ 0.0016 at ≥ 2 GeV), because η from z = 0 ignores
  the vertex z spread. A uniform η down-weighting of the metric (w = 0.05–0.5) does not help: w = 1 is best with
  the MD estimate.
- At ≥ 2 GeV the MD estimate's noise makes the 99% tail slightly worse than no shift (65 vs 53): the shift could
  be scaled by the estimate's uncertainty.

#### Curvature-aware road over all layers (`studies/md_road_rank.py`, plots `md_plots/road_*`, 2026-11-10)

Same 50 events. For each true MD, the road holds k MDs per layer on every layer a straight line in r-z from the
beam line through the MD crosses (±10% margin on the layer extents), each around the φ where the helix with the
MD's q/pT crosses that layer. "Whole track" = the nearest MD of the sim on every other layer of the sim is in the
road. hop = distance in the sim's own ordered layer list.

- Road size: 5.3 layers on average (own layer excluded). The road's layers contain all of the sim's other layers
  for 94–97% of MDs; 99.4% of hop-1 links are on a road layer.
- k per road layer needed for 90% of MDs to see their whole track:

  | pT | no shift | MD q/pT | true q/pT |
  |---|---|---|---|
  | 0.8–1 GeV | 298 | 70 | 47 |
  | 1–2 GeV | 150 | 43 | 29 |
  | ≥ 2 GeV | 51 | 47 | 33 |

  So k ≈ 64 per layer (~340 neighbours per MD) gives ~89% of 0.8–1 GeV MDs their whole track in one hop, and
  k ≈ 192 per layer (~1000 neighbours) ~95%. The 512–1024 range is exactly the whole-track road.
- 99% needs 400–1000+ per layer: the same non-helix tail as for next-layer links.
- By sim selection (2026-11-11, `--selection`; plots `md_plots/road_*_<selection>`). k per road layer for 90% /
  99% of MDs to see their whole track:

  | Selection | pT | no shift | MD q/pT | true q/pT | road complete |
  |---|---|---|---|---|---|
  | hard_scatter | 0.8–1 GeV | 298 / 746 | 70 / 1088 | 47 / 584 | 94.2% |
  | | 1–2 GeV | 150 / 408 | 43 / 475 | 29 / 326 | 97.1% |
  | | ≥ 2 GeV | 51 / 296 | 47 / 380 | 33 / 305 | 95.8% |
  | denominator (LST type 7, pT ≥ 0.8) | 0.8–1 GeV | 316 / 722 | 46 / 1019 | 28 / 541 | 95.1% |
  | | 1–2 GeV | 157 / 384 | 36 / 451 | 22 / 209 | 97.6% |
  | | ≥ 2 GeV | 49 / 217 | 46 / 305 | 31 / 217 | 96.0% |
  | denominator, on-helix MDs only | 0.8–1 GeV | 303 / 484 | 29 / 317 | 18 / 93 | 98.5% |
  | | 1–2 GeV | 150 / 309 | 28 / 226 | 17 / 77 | 99.1% |
  | | ≥ 2 GeV | 43 / 111 | 35 / 160 | 25 / 87 | 99.1% |

  - The denominator alone fixes the 90% level (displaced sims gone); the 99% tail only shrinks once off-helix MDs
    are dropped too.
  - For denominator tracks: k ≈ 32 per layer (~170 neighbours) gives 90% their whole track with the MD's own
    q/pT. 99% needs 160–320 per layer with the MD q/pT but ~90 with the true q/pT: at 99%, q/pT refinement is
    the lever, not larger k.
- Far hops need a better q/pT than the MD's own: at 0.8–1 GeV, hop 5+ needs k = 89 (90%) with the MD estimate vs
  45 with the true q/pT; at ≥ 2 GeV the MD estimate's noise makes far hops worse than no shift at the 99% level.
  Idea: refine q/pT before building the road (a few attention layers on the local curvature-aware neighbourhood
  regress q/pT, then the road is built from the refined value, i.e. a dynamic graph), and scale the shift or the
  road width by the q/pT uncertainty σ the physics head already predicts.

#### The non-helix tail (`studies/md_tail_causes.py`, 2026-11-11, same 50 events)

Links with curvature-aware (MD q/pT) rank > 64: 248 of 21363 (1.2%). Each MD is compared with its sim's prompt
helix (sim vz, η, φ, q/pT): "off helix" = |Δφ| > 0.1 or |Δz| > 15 cm (median on-helix residuals: 0.002, ~1 cm).

| Cause | Share of tail | Share of all links | Tail rate within cause |
|---|---|---|---|
| Displaced sim (vertex rt > 2.5 cm) | 34% | 22% | 1.8% |
| Prompt sim, source or target MD off its helix | 52% | 2.5% | 24% |
| Prompt, on helix, MD q/pT has the wrong charge | 12% | 4.3% | 3.3% |
| Prompt, on helix, rest | 2% | 71% | 0.03% |

- Off-helix MDs: 11% of prompt hard-scatter sims with pT ≥ 0.8 GeV have at least one. 78% of tail sources share
  their layer with another MD of the same sim. Neutral sims (n, K0L, Λ) also own MDs. Likely reading: CMSSW
  attributes hits of unsaved secondaries (δ-rays, hadronic-interaction products) to the parent TrackingParticle,
  plus kinks from elastic hadronic scattering. These are truth-label artefacts, not hits a track finder should
  attach (question 6 for colleagues).
- Wrong-charge cases are all fixed by the true q/pT (rank ≤ 64), and all have a second MD of the sim on the same
  layer: the refined-q/pT idea covers them.
- Displaced sims are outside the LST type-7 denominator anyway; only 19% of their tail links are fixed by the true
  q/pT, since the helix does not start at the beam line.
- Restricted to prompt, on-helix links of charged sims (what the denominator cares about), curvature-aware layer
  kNN needs far less:

  | pT | k for 90% | k for 99% (MD q/pT) | kept at k = 64 | k for 99% (true q/pT) |
  |---|---|---|---|---|
  | 0.8–1 GeV | 5 | 35 | 99.3% | 14 |
  | 1–2 GeV | 4 | 25 | 99.8% | 12 |
  | ≥ 2 GeV | 5 | 25 | 99.9% | 16 |

  vs 120 / 72 / 65 at 99% over all links. So k = 32–64 per layer is enough for denominator tracks, and the 99%
  road numbers above are inflated by the same artefacts (to recheck with this selection).
- Consequence for training truth (proposal): label off-helix MDs of a sim as noise (or drop them from the
  grouping target), like mixed MDs, so the network is not asked to attach hits no helix reaches. Evaluation stays
  hit-level against the ntuple truth.

#### Extrapolation walkthrough (`studies/md_extrapolation_demo.py`, 2026-11-12)

Standalone script on one LSTNtuple (reads nothing else from the repo): one random MD of interest per layer per
event (true OT MD of a prompt hard-scatter sim, pT ≥ 0.8 GeV), 100 MDs from 10 events, 517 (MD, layer) pairs with a
sibling. Per MD: one pdf with an x-y view and one (φ, η) panel per layer; `summary.pdf`; key=value log
(`md_plots/extrapolation/`, and `md_plots/extrapolation_detailed/` with `--detailed`, not in git).

- Checks: `md_pt` = rt_anchor · K / sin(`md_dphichange`) (signed: q/pT = −1 / `md_pt`), and `md_eta`, `md_phi` are
  the anchor hit's direction from the origin. Agree to < 1e-4 on every OT MD of 10 events.
- k = number of MDs on the layer within ΔR of the nearest sibling, from each centre:

  | Centre | What it uses | k median | k 90% |
  |---|---|---|---|
  | own | the MD's (η, φ) | 6 | 52 |
  | helix | helix with the MD's q/pT from (0, 0, 0) through the MD: φ bend and η path length | 1 | 11 |
  | helix sim q/pT (`--detailed`) | as helix, with the sim's q/pT | 1 | 8 |
  | helix sim vz (`--detailed`) | as helix, from the sim's vertex (0, 0, sim_vz) | 1 | 5 |
  | oracle | the sim's helix from its vertex, with its q/pT and η | 1 | 2 |

  Earlier centres, dropped the same day: φ bent with the MD's q/pT, η kept (1 / 12, ≈ helix), and φ bent with the
  sim's q/pT, η kept (1 / 9). The bullets below quote them as "φ shift".

- The φ shift from the MD's own q/pT is most of the win (52 → 12 at 90%), largest at low pT (0.8–1 GeV: 96 → 11).
  16 of 100 MDs have a poor q/pT estimate (> 30% off, some with the wrong sign, e.g. a layer-6 2S MD at +0.007 vs
  −0.77); the true q/pT gains little more on average (12 → 9), most in the barrel (90%: 26 → 13–15).
- After the φ shift, the remaining distance is mostly in η: the helix does not change the track's η, but `md_eta`
  is the hit position seen from the origin, `arcsinh(sinh η + z0 / r)`, which drifts with r when the vertex is not
  at z = 0. Example: a 1.45 GeV pion from vz = 2.7 cm, `md_eta` −0.80 on layer 1 to −0.90 on layer 6 (sim η −0.88).
  The longer curved path adds a smaller drift, ≈ tanh η · (s/r − 1).
- Correcting η for the path length alone (from the MD) barely helps (12 → 11; 56 pairs better, 36 worse); in that
  example it moves η by −0.018 on layer 6 against −0.081 for the oracle. The oracle's gain (→ 2) is the vertex z0,
  which one MD cannot measure (MD hits are mm apart in r; z resolution 1.5 mm PS, 5 cm 2S). z0 needs two MDs on
  different layers (attention or an MD-pair estimate); a fixed window instead needs to be longer in η than in φ.
- One oracle ingredient at a time, against helix (k 90% 11): the sim's vertex z helps more (→ 5; 144 pairs
  better, 35 worse) than the sim's q/pT (→ 8; 101 better, 30 worse). Both together, with the sim's η, give the
  oracle's 2. So z0 is the larger gap, as the η-drift bullet above says.
- One (MD, layer) pair has k ≈ 1000 from every centre, the oracle included: a sibling no helix reaches (likely
  the non-helix tail above; not checked).

#### Making large k (512–1024) work (user, 2026-11-10: "keep a very open mind", be cowboys)

- For next-layer links, curvature-aware k ≈ 32–64 per layer already saturates; more k there is wasted.
- Where large k could pay: one-hop context of the whole track. A curvature-aware *road*: k per layer on every
  layer along the predicted helix (not only adjacent ones), e.g. 11 layers × 32–64 = 350–700 neighbours, so each
  MD sees its whole candidate track in one hop. To check: rank of far-layer siblings with the shift (the
  extrapolation error grows with distance, so the q/pT resolution matters more).
- Memory at N ≈ 50k MDs, k = 1024, d = 64: gathered keys/values are ~13 GB per attention layer in fp32. Options:
  bf16; gradient checkpointing; chunking queries; η/φ chunks with overlap (see Constraints); the A100 80 GB;
  block-sparse attention that never gathers (sort MDs by layer/η/φ into blocks and attend block-to-block, e.g.
  PyTorch FlexAttention with a mask function; check its CPU support for the CPU-training constraint).

## Design

### Input features

- Existing 8 columns, with layer moved to an embedding (`nn.Embedding(12, d)`: a learned vector per layer,
  equivalent to one-hot followed by a linear layer).
- Done 2026-11-10 (`md_processing.py`, `x` columns 8 and 9), both LST's own MD estimates:
  - η = `md_eta` (anchor hit η).
  - q/pT = −sin(`md_dphichange`) / (k · anchor rt) in GeV⁻¹, k = 2.99792458e-3 · 3.8 / 2 (LST's `k2Rinv1GeVf`).
    `md_dphichange` is LST's α (direction-vs-position angle; in the endcap it is built from dphi and dz), and
    this is LST's `md_pt` inverted (finite at α = 0) with the sign flipped: LST's sign is opposite to sim q.
  - Measured on 2 events, true MDs of sims with pT > 0.8 GeV: q/pT correlation 0.90 with sim q/pT, median
    ratio 0.99, charge right for 95%; η correlation 0.99.
- Normalization: physical scales, e.g. η / 2.5, rt / 110 cm, z / 270 cm, q/pT in GeV⁻¹ as is. Scales for drt,
  dphi, dz to be set from the nominal sensor spacings.

### Neighbourhood (two variants, config switch)

Both produce `neighbor_idx`, `neighbor_mask` of shape (N, k), so the rest of the model is shared.

- **Global ΔR kNN** (current T3 code): needs k ≈ 128+ and `dr_threshold` ≈ 0.3 for good low-pT recall.
  Includes far layers, i.e. long lever arms.
- **Layer-aware kNN**: k nearest in ΔR on each adjacent layer (adjacency from the layer-transition study).
- Centre of the ΔR search on each layer (Decided 2026-11-12): the helix from (0, 0, 0) through the MD with its
  own q/pT, shifting φ and η (see "Extrapolation walkthrough").

Choose k and `dr_threshold` from an extended `studies/md_sibling_plots.py` (see Next steps), not by guessing.

### Attention with a learned geometry bias

Standard attention score: `s_ij = q_i · k_j / sqrt(d)`. With a geometry bias:
`s_ij = q_i · k_j / sqrt(d) + b_ij`, where `b_ij = MLP(Δη_ij, Δφ_ij, Δrt_ij, Δz_ij, layer_i, layer_j)` gives
one number per head. The network can learn, for example, "neighbours at the Δφ my curvature predicts matter".
Cost is a small MLP on N × k pairs per layer. About 6 layers to start.

### Grouping head: hybrid regression/embedding

Motivation: PCA of earlier tracking embeddings shows the top components correlate (R > 0.9) with sim η,
cos φ, sin φ, q/pT. Regress those directly, and leave free dimensions for whatever else is useful.

- Physics head per MD: mean μ = (η, cos φ, sin φ, q/pT) and an uncertainty σ for each, trained with a Gaussian
  negative log-likelihood against the sim values, for MDs matched to a sim with pT ≥ `pt_min`. σ lets the
  network say "my pT is still poor here", which matters given MD pT resolution.
- Free head per MD: f, a few dimensions (e.g. 8–16) with no target.
- Grouping, two variants:
  - **H1, distance-based:** cluster in the combined space [μ / σ, f]. Free dims trained with a contrastive loss
    over neighbour pairs only (no O(N²)).
  - **H2, edge head (preferred start):** for each neighbour pair, an MLP on
    `(h_i, h_j, Δgeom, (μ_i − μ_j) / sqrt(σ_i² + σ_j²), |f_i − f_j|)` with BCE target "same sim, pT ≥ pt_min".
    Plus a per-MD noise head.
- Each candidate gets track parameters: the inverse-variance-weighted mean of its MDs' μ. Used for duplicate
  merging, for the tunable efficiency below, and for pT/η resolution plots against T5.

### MD clusters to track candidates, with tunable efficiency

Two stages, so efficiency can be set per (pT, η) bin, as LST does with its η-binned T5 DNN working points:

1. **Loose building** (high efficiency, not tuned per bin): drop MDs with a high noise score, keep edges above a
   loose threshold, take connected components, split components with incompatible MDs on one layer
   (highest-score walk), keep candidates with ≥ 3 MDs. Its efficiency is the ceiling for stage 2.
2. **Candidate selection:** each candidate gets a quality score (to start: from its edge scores and size; later
   a learned candidate classifier with target = hit-level matched fraction) and its predicted (pT, η). A
   threshold map c(pT̂, η̂), binned in the predicted parameters, keeps candidates with score > c.
   - Tune c on tuning events so efficiency vs sim pT and η matches a target curve (e.g. T5's), then measure
     fake and duplicate rates on separate test events.
   - Scanning c per bin gives efficiency-vs-fake-rate curves per bin.
   - Caveat: efficiency is defined in sim bins but c is applied in predicted bins, so pT/η resolution causes bin
     migration. Check how well the tuned efficiency lands on target.
3. Output: the MD list and its de-duplicated hit list.

### Evaluation

- Track level (primary): efficiency, fake rate, duplicate rate vs pT and η, using hit-level 75% matching.
  Baseline T5 on the same events with the same matching, compared at matched efficiency.
- Graph level (diagnostic): neighbourhood edge recall, edge and noise ROC (TPR vs FPR).
- Inference time per event.

## Evaluation harness (agreed 2026-11-10, follow-ups in Open questions)

Purpose: one fixed ruler that every model and every baseline goes through, so intermediate design choices can be
decided by its numbers.

1. **Candidate format (common to all methods):** per event, a list of candidates, each with a set of LST hit
   indices, a quality score, and reconstructed (pT, η, φ).
   - MD model: hits from `hit_index` of its MDs; score and parameters from the network.
   - T5: `t5_hitIndices` (extended, all hits), score `t5_dnnScore`, `t5_pt`/`t5_eta`/`t5_phi`.
2. **Hit truth:** hit → sims from `hit_sim_pairs` (MD dataset) or the `md_*_simIdxs` branches. Every T5 hit is an
   MD hit (checked), so one table serves both.
3. **Matching:** over a candidate's unique hits, a hit counts for each of its sims (same as the max over LST's
   permutations). The candidate matches sim s if hits(s) / n_hits > 0.75 (strict, as LST). Validate by
   reproducing `t5_pMatched` on 5-layer T5s.
4. **Efficiency denominator:** LST's `getDenomSimTrkType` == 7 (`trkCore.cc`): q ≠ 0, pT ≥ 1 GeV,
   |η| ≤ 2.4, vertex rt ≤ 2.5 cm, |vz| ≤ 30 cm, bunch crossing 0, event 0 (hard scatter). Efficiency vs pT drops
   the pT cut; efficiency vs η keeps it. No reconstructability cut. Extra plot: efficiency vs the number of OT
   layers on which the sim has a true MD.
5. **Efficiency:** fraction of denominator sims matched by at least one candidate.
   **Fake rate:** fraction of candidates matched to no sim (any sim, pileup and low pT included).
   **Duplicate rate:** fraction of matched candidates whose sim is matched by more than one candidate (per
   candidate; to be confirmed against the tracking group's definition).
   Fake and duplicate rates are binned in the candidate's reconstructed pT and η.
6. **Event sets:** from the 4500 ttbar events: 2700 train, 600 validation, 200 tuning (thresholds, working
   points), 1000 test (reported numbers). Fixed once, listed by file and entry in the repo, and never reshuffled.
   Tuning and test events are never used in training.
7. **Comparison at matched efficiency:** tune each method's score thresholds per (pT, η) bin on the tuning set to
   a common target efficiency curve (default: T5's at its default working point), then report fake and duplicate
   rates on the test set. Also scan thresholds to draw efficiency vs fake rate per bin for each method.
8. **Implementation status** (`harness/` package in this repo, run as `python3 -m harness.<module>` from the
   repo root inside the Apptainer image):
   - Done 2026-11-10: `matching.py` (hit-level matcher), `ntuple.py` (hit truth from LSTNtuple MD branches),
     `test_matching.py`, `validate_t5_matching.py`.
   - Validation on 10 events of the test ntuple: matched sim agrees with LST's `t5_simIdx` for all 15534
     5-layer T5s; the best-sim fraction agrees with `t5_pMatched` for 15520, and the 14 differences are all at
     fractions ≤ 0.5 (LST skips hit-sim combinations whose most common entry is "no sim"). Matching extended T5s
     on all stored hits changes the matched sim for 312 of 54329 (0.6%).
   - Also validated (10 events): matching every T5 on its first 10 stored hits (the base hits; extensions are
     appended after them) reproduces `t5_simIdx` for all 69863 T5s, and the set of sims found by any T5 equals
     the sims with a non-empty `sim_t5IdxAll` (10615 sims).
   - Done 2026-11-10: `metrics.py` (LST type-7 denominator, matched sims, fake, duplicate, OT layers with a true
     MD), `test_metrics.py`, `t5_metrics.py` (efficiency, fake and duplicate rate plots for three T5 selections).
   - **T5 selection is open** (see Questions for colleagues). The ntuple stores every T5, including those LST's
     duplicate cleaning removed (`t5_isDupBitmask`: bits 1/2 = first/second T5-T5 cleaning pass, bit 4 =
     cross-cleaned against a pT5/pT3 sharing ≥ 4 OT hits, `TrackCandidate.h`). Selections in `t5_metrics.py`:
     `all`; `ot_cleaned` (bits 1 and 2 unset); `track_candidate` (no bit set and not `partOfPT5`, i.e. LST's
     T5 track candidates, what LST's own "T5" fake/duplicate plots use).
   - Next: decide the T5 selection, then the event-split manifest, then matched-efficiency tuning.
9. **Decision rule:** at matched efficiency, A beats B if its integrated fake rate is lower beyond statistical
   uncertainty. Duplicate rate is second: it decides when fake rates are compatible, and a large duplicate-rate
   increase is flagged for discussion. Inference time per event is reported and breaks remaining ties.

## Questions for colleagues (ask soon; don't sweep under the rug)

Recorded 2026-11-10. Each needs an answer from the LST / tracking group before we rely on our interpretation.

1. **`t5_pMatched` ignores extension hits.** It is computed from `getHitsFromT5` (the 5 base MDs, 10 hits),
   while extended T5s store 12–14 hits in `t5_hitIndices`. For extended T5s, matching on all stored hits changes
   the matched sim for 0.6% (312 of 54329, 10 events). Is base-hit matching intended? Which does the tracking
   group use for T5 efficiency and fake rate?
2. **`t5_pMatched` disagrees with a direct hit count in 14 of 15534 5-layer T5s** (10 events), always at
   fractions ≤ 0.5. Cause, from reading `matchedSimTrkIdxsAndFracs` (`trkCore.cc`): hit-sim permutations whose
   most common entry is "no sim" are skipped before `pmatched` is updated. It never changes a 75% decision, but
   is it intended?
3. **`*_simIdxAll` branches are cut at > 75%** despite comments saying "(> 0%)" (`write_lst_ntuple.cc`).
   Fix the comments, or the cut?
4. **Duplicate rate:** per candidate or per sim? (LST's `efficiency/src/performance.cc` uses per candidate, over
   all candidates including fakes; `t5_isDuplicate` = matched to a sim with more than one matched T5. We follow
   that; confirm it is what the group quotes.)
5. **Which T5s are the OT-only baseline?** LST's "T5" plots use T5 track candidates, which exclude every T5
   absorbed into or cross-cleaned by a pT5/pT3, so their efficiency is low (~5% in a 5-event smoke test). T5s
   surviving only the T5-T5 cleaning have ~80% efficiency but a ~80% duplicate rate. What does the group
   consider the fair OT-only T5 reference for an algorithm that never sees pixel seeds?
   Provisional answer (2026-11-10): `ot_cleaned`. On 300 events: efficiency 78.2%, fake rate 1.0%, duplicate
   rate 81% (`all`: 78.3% / 2.3% / 97%; `track_candidate`: 4.1% / 2.8% / 1.8%).

6. **Hits of unsaved secondaries attributed to the parent sim?** 11% of prompt hard-scatter sims with pT ≥ 0.8
   GeV have an MD far off their helix (|Δφ| > 0.1 or |Δz| > 15 cm), often a second MD on a layer where the sim
   already has an on-helix one, and neutral sims (n, K0L, Λ) own MDs. Is this CMSSW attributing hits of untracked
   secondaries (δ-rays, interaction products) to the parent TrackingParticle? Does LST's efficiency definition
   treat such hits specially? (Recorded 2026-11-11.)

## Open questions

- Proposed (2026-11-10): train only on "true" MDs (both hits from one sim), treat mixed MDs as noise, and so no
  regression target for mixed MDs. Evaluation stays hit-level regardless. Studies once per-hit truth exists:
  how many fake MDs share a true hit with a true MD ("combinatorial partners"), and how often a sim has a hit on
  a layer but no true MD there (coverage lost by dropping mixed MDs).
- Follow-up: produce more ttbar beyond the 4500 events; later fold in di-jet and Z' → μμ samples.
- ~~Event-split manifest.~~ Done 2026-11-11: `md_processing.py` writes `source_path`/`source_entry` into each file
  and `<output>/manifest.txt` (key=value, one line per event); `harness/manifest.py` reads it.
- **Split mismatch (2026-11-11).** The Decided split is 2700 train / 600 validation / 200 tuning / 1000 test,
  but the first 4400-event production used a two-way 80/20 split (3527 train / 873 val). Fix (user, 2026-11-11):
  `md_processing.py` takes `--train --val --tune --test` fractions (default 0.6 / 0.15 / 0.05 / 0.2, must sum
  to 1) and cuts exact set sizes from one seeded shuffle; one directory per set. On 4400 events the defaults give
  2640 / 660 / 220 / 880. Production rerun 2026-11-11: all 4400 files written, no errors.
- Regression target for φ: sim φ at the vertex (network learns the bend) vs φ at a reference radius.

## Next steps

1. ~~Regenerate the MD dataset from ntuples with the per-hit branches.~~ Done 2026-11-11: 4400 events (44 files ×
   100, one file failed upstream) in `transformer-datasets/samples/ttbar/{train,val}`, `--seed 42`, with
   `manifest.txt`. All files match the manifest, have 10 columns and finite x; median 45.7k MDs per event (26.6k–
   69.9k), 48% true MDs. The split still differs from the Decided one (Open questions).
2. Define and freeze the evaluation harness, validated first on T5s alone. 2026-11-11: `harness/t5_metrics.py`
   reads events through the manifest (`--manifest`, `--set`). T5 baseline on the 873 val events (41787
   denominator sims): `ot_cleaned` efficiency 78.6%, fake rate 1.0%, duplicate rate 81.0% (`all`: 78.7% / 2.3% /
   96.7%; `track_candidate`: 4.2% / 2.9% / 1.8%); plots in `md_plots/t5_val/`. Consistent with the earlier 300
   events (that val set is superseded). Test set (880 events, 43470 denominator sims, `md_plots/t5_test/`):
   `ot_cleaned` 78.7% / 1.0% / 81.0% (`all`: 78.9% / 2.3% / 96.7%; `track_candidate`: 4.2% / 2.9% / 1.9%).
   Ceiling of perfect MD grouping (`studies/md_ceiling.py`, `md_plots/ceiling_test/`): denominator sims with a
   true MD on ≥ 3 / 4 / 5 OT layers: 88.1% / 86.5% / 84.1%, vs 78.7% for `ot_cleaned` T5s. In 20 test events,
   of the sims below 3 layers, about half have no hit in any OT MD (hyperon decays, early interactions), 40% have
   hits on only 1–2 layers, and ~7% have hits on ≥ 3 layers but too few true MDs (the cost of dropping mixed
   MDs is small). 2026-11-12: candidate input format for models, `harness/candidates.py` (per event: MD indices
   + offsets, score, pT, η, φ; hits via the event file's `hit_index`). Check on one test event: grouping each
   sim's true MDs gives 5879 candidates, all matched to their own sim. Still to do: a script scoring model
   candidate files through the manifest, and the perfect-grouping check against the ceiling.
3. ~~Next-layer sibling rank per target layer (layer-aware k) vs global k.~~ Done 2026-11-10
   (`studies/md_neighbour_rank.py`), plus curvature-aware variants and the all-layer road
   (`studies/md_road_rank.py`). Non-helix tail explained 2026-11-11 (displaced sims, off-helix MDs). Road for denominator sims done
   2026-11-11. Next: road with a refined q/pT.
4. ~~Add α (as q/pT estimate) and η to the MD dataset.~~ Done 2026-11-10.
5. Model: shared encoder + both neighbourhoods + H2 head + two-stage candidate making; then H1 as a variant.

## Gone stale

Kept briefly for context; delete when no longer useful.

- 2026-11-10: "`md_simIdxAll` / `md_simIdxAllFrac` give hit-level truth for MDs." Wrong: they list only sims
  matching both hits (fraction always 1.0). See MD truth.
- 2026-11-10: "51–53% of MDs are unmatched", read as mostly noise. With per-hit truth: only 22% of OT MDs
  have two noise hits; the rest of the unmatched ones have at least one sim hit.
- 2026-11-10: "Check MD-level vs hit-level T5 purity, as a test of MD-level evaluation." Dropped: evaluation is
  hit-level, full stop.
- 2026-11-10: "Scale the curvature feature to 1 at pT = `pt_min`." Replaced by using q/pT in GeV⁻¹ directly.
- 2026-11-10: "Study: truth layer-transition matrix." Done; see Layer transitions.
