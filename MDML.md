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
| 2026-11-10 | Sims with pT < `pt_min` (default 0.8 GeV, configurable) are treated as noise in training. Alternatives for later: leave them out of the loss, or use them as positives. |
| 2026-11-10 | Next topic after the current items: define the comparison harness in detail. |

## Constraints

- Training must run on CPU too; keep code device-agnostic, no new CUDA-only dependencies.
- Hardware: phi3 Tesla V100-PCIE 16 GB (free use) for development; NRP A100-SXM4 80 GB (one GPU) for larger
  runs. `kubectl` is not on phi3's PATH (a `~/.kube/config` exists), so NRP jobs are submitted by the user for
  now.
- Outer tracker only: pixel LS MDs (`md_isPLS`) are dropped, so absolute efficiency is not comparable to full LST.

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

### Layer transitions along sim tracks (`md_layer_transitions.py`, plots in `md_plots/layer_transitions_*`)

Sim pT ≥ 0.8 GeV, 100 events, MD-level truth. MDs of a sim ordered by the inner hit's sqrt(rt² + z²).

- Same layer → same layer (overlapping modules) is 17–30% of transitions from every layer. Several MDs per layer
  per track is normal, so candidate building must not assume one MD per layer.
- Barrel L → L+1: 56–63%. Skips (L → L+2) are ≤ 3% in the barrel, up to 7% in the endcap (8 → 10).
- Barrel 1–5 → endcap 7: 8–15%. Barrel 6 → 7: 0.00. Endcap → barrel: ~0.
- Track ends: 68% after layer 6, 81% after layer 11, and 14% after layer 5. Some ends and skips will be MD
  inefficiency or MDs with only one matched hit, which MD-level truth does not see.
- Adjacency for a layer-aware neighbourhood (both directions): same layer; L ± 1; L ± 2 in the endcap;
  barrel 1–5 ↔ 7.

### Sibling distances (`md_sibling_plots.py`, plots in `md_plots/`)

- At 0.8–1 GeV only ~half of an MD's siblings are within ΔR < 0.1; the farthest sibling's ΔR rank k has a
  median of ~500–1000. One hop cannot see a whole track.
- The next-layer sibling reaches ~90% at k ≈ 100 (0.8–1 GeV) and k ≈ 20–30 (3–4 GeV) in global ΔR rank.
- The T3 settings (`dr_threshold: 0.1`, `max_neighbors: 16`) are far too small for MDs.

## Design

### Input features

- Existing 8 columns, with layer moved to an embedding (`nn.Embedding(12, d)`: a learned vector per layer,
  equivalent to one-hot followed by a linear layer).
- η of the inner hit.
- Direction-vs-position angle α: the angle in the transverse plane between the MD direction (outer − inner) and
  the inner hit position. α ≈ r / (2R) with R = pT / (0.3 B), so 2α / (0.3 B rt) is a per-MD q/pT estimate in
  GeV⁻¹ (sign convention to be checked against sim q). Short lever arm, so expect poor resolution; attention to
  siblings should improve it.
- Normalization: physical scales, e.g. η / 2.5, rt / 110 cm, z / 270 cm, q/pT in GeV⁻¹ as is. Scales for drt,
  dphi, dz to be set from the nominal sensor spacings.

### Neighbourhood (two variants, config switch)

Both produce `neighbor_idx`, `neighbor_mask` of shape (N, k), so the rest of the model is shared.

- **Global ΔR kNN** (current T3 code): needs k ≈ 128+ and `dr_threshold` ≈ 0.3 for good low-pT recall.
  Includes far layers, i.e. long lever arms.
- **Layer-aware kNN**: k nearest in ΔR on each adjacent layer (adjacency from the layer-transition study).

Choose k and `dr_threshold` from an extended `md_sibling_plots.py` (see Next steps), not by guessing.

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
   - Next: denominator and metrics, then the event-split manifest, then matched-efficiency tuning.
9. **Decision rule:** at matched efficiency, A beats B if its integrated fake rate is lower beyond statistical
   uncertainty. Duplicate rate is second: it decides when fake rates are compatible, and a large duplicate-rate
   increase is flagged for discussion. Inference time per event is reported and breaks remaining ties.

## Open questions

- Proposed (2026-11-10): train only on "true" MDs (both hits from one sim), treat mixed MDs as noise, and so no
  regression target for mixed MDs. Evaluation stays hit-level regardless. Studies once per-hit truth exists:
  how many fake MDs share a true hit with a true MD ("combinatorial partners"), and how often a sim has a hit on
  a layer but no true MD there (coverage lost by dropping mixed MDs).
- Follow-up: confirm the duplicate-rate definition (per candidate vs per sim) with the tracking group.
- Follow-up: produce more ttbar beyond the 4500 events; later fold in di-jet and Z' → μμ samples.
- Event-split manifest: MD dataset files (`event_N.pt`) don't record their source ntuple file and entry, which the
  harness needs to join model candidates with ntuple truth and T5s. Proposal: a manifest in the repo listing
  (ntuple path, entry, set) for the 4500 ttbar events, and `md_processing.py` writes the source into each file.
  Needs the location of the 4500-event ntuples.
- Regression target for φ: sim φ at the vertex (network learns the bend) vs φ at a reference radius.

## Next steps

1. Regenerate the MD dataset from ntuples with the per-hit branches (`md_processing.py` done 2026-11-10).
2. Define and freeze the evaluation harness, validated first on T5s alone.
3. Extend `md_sibling_plots.py`: next-layer sibling rank per target layer (layer-aware k) vs global k.
4. Add α (as q/pT estimate) and η to the MD dataset.
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
