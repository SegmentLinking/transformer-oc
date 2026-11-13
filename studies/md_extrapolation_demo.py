"""
Walk through the pT-extrapolated (curvature-aware) neighbourhood of single MDs, one MD at a time.

For each MD of interest (a true outer-tracker MD of a prompt hard-scatter sim with pT >= PT_MIN):
  1. Check LST's MD estimates: md_pt = rt * K2RINV_1GEV / sin(md_dphichange), with rt the anchor hit's, and
     md_eta, md_phi the direction of the anchor hit (x, y, z) seen from the origin.
  2. Find its sim and its siblings: the other true MDs of that sim (md_simIdx: both hits from the sim).
  3. On every layer the track could cross (a straight line in r-z from the origin through the MD) and every layer
     with a sibling, place neighbourhood centers in (eta, phi):
         own:               the MD's own (eta, phi).
         helix:             using only what one MD gives: the helix with the MD's q/pT estimate, q/pT = -1 / md_pt,
                            from a vertex assumed at (0, 0, 0) through the MD, followed to where it reaches the
                            layer. Moves phi by the bend, and eta for the longer curved path.
         helix sim q/pT:    as helix, with the sim's true q/pT: helix vs this isolates a poor MD q/pT estimate.
         helix sim vz:      as helix, from the sim's true vertex (0, 0, sim_vz): helix vs this isolates the vertex
                            z, which moves eta.
         oracle:            the sim's own helix, from its vertex (0, 0, sim_vz) with its true q/pT and eta, followed
                            to where it reaches the layer.
     By default the plots and summary show own, helix and oracle; --detailed adds the two one-ingredient oracles.
     A helix from the beam line has phi(r) = phi0 - asin(K2RINV_1GEV * r * q/pT), so going from the MD's radius to
     the layer's radius moves phi by
     -(asin(K2RINV_1GEV * r_layer * q/pT) - asin(K2RINV_1GEV * md_anchor_rt * q/pT)).
  4. From each center, measure dR to the nearest sibling on the layer, and k: the number of MDs on the layer
     within that dR (k = 1: the sibling is the nearest MD on the layer). k is the number of neighbours per layer a
     model needs to see that sibling. The MD's own layer is skipped.

The --n_mds MDs of interest are drawn at random from one event (--tree_entry), so the layers appear in proportion
to their candidates; with the default 100, all 11 layers (flat and tilted barrel, endcap, PS and 2S modules)
almost always appear. Positions are anchor hits, as in md_eta and md_phi.
Self-contained: reads one LSTNtuple and nothing else from the repo.

Output: one pdf, --output.
    Page 1, summary: k and dR to the nearest sibling per center, over all MDs of interest and layers.
    Then one page per MD, in layer order. First panel: x-y view of the sim's MDs, with the straight line (own phi)
        and the helices with the MD's and the sim's q/pT. Then one panel per layer: the layer's MDs near the centers
        in (phi, eta), siblings highlighted, and a circle around each center reaching the nearest sibling, so it
        holds k MDs.
Logs (key=value): [check] per event, [md] per MD of interest, [layer] per (MD, layer), [summary] per category.

Usage (from the repo root):
    apptainer exec /ceph/users/atuna/ml-tracking.sif python3 studies/md_extrapolation_demo.py \
        --input ../transformer-datasets/samples/ttbar/LSTNtuple_<X>.root \
        --output ../md_plots/md_extrapolation.pdf
"""

import os
import random
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from pprint import pprint

import awkward as ak
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import uproot

LAYERS = range(1, 12)
BARREL_LAYERS = range(1, 7)
ENDCAP_LAYERS = range(7, 12)
# Curvature constant (speed of light * B / 2) in GeV^-1 cm^-1, as LST's k2Rinv1GeVf
K2RINV_1GEV = 2.99792458e-3 * 3.8 / 2
# Sims of interest: LST's type-7 denominator cuts, with a lower pT cut (as the road studies)
PT_MIN = 0.8
ABS_ETA_MAX = 2.4
VERTEX_RT_MAX = 2.5
VERTEX_ABS_Z_MAX = 30.0
# Margin on a layer's extent when deciding whether the straight line in r-z crosses it
ROAD_MARGIN = 0.1
# Barrel detId side (bits 18-19, LST's ModulesSoA.h): 1 = tilted -z, 2 = tilted +z, 3 = flat (center)
FLAT_SIDES = {3}
# md_type: 1 = PS, 0 = 2S
PS_TYPE = 1
# Neighbourhood centers (see the module docstring): shown by default, and with --detailed. All are computed and
# logged per layer either way.
SALIENT_CENTERS = ["own", "helix", "oracle"]
DETAILED_CENTERS = ["own", "helix", "helix_sim_qoverpt", "helix_sim_vz", "oracle"]
# Short names in plot titles and legends
CENTER_LABELS = {
    "own": "own",
    "helix": "helix",
    "helix_sim_qoverpt": "sim q/pT",
    "helix_sim_vz": "sim vz",
    "oracle": "oracle",
}
# Pairs of centers compared in the [summary] log, when both are shown: (center, reference)
COMPARISONS = [
    ("helix", "own"),
    ("helix_sim_qoverpt", "helix"),
    ("helix_sim_vz", "helix"),
    ("oracle", "helix"),
]
REGIONS = ["barrel_flat", "barrel_tilted", "endcap"]
PT_BINS = [(0.8, 1.0), (1.0, 2.0), (2.0, np.inf)]
# Padding (in eta and phi) of each layer panel around the siblings, centers and circles
WINDOW = 0.1
# Slots of the dataviz skill's validated categorical palette
COLORS = {
    "own": "#2a78d6",
    "helix": "#e34948",
    "helix_sim_qoverpt": "#e87ba4",
    "helix_sim_vz": "#eb6834",
    "oracle": "#6250d6",
    "sibling": "#1baf7a",
    "md": "#eda100",
}
BACKGROUND_COLOR = "#a0a0a0"
LINESTYLES = {
    "own": "--",
    "helix": "-",
    "helix_sim_qoverpt": ":",
    "helix_sim_vz": (0, (5, 1, 1, 1, 1, 1)),
    "oracle": "-.",
}
# Log-spaced bin edges at half-integers, so each bin holds whole values of k
K_BINS = np.concatenate([[0.5], np.unique(np.round(np.logspace(0, 4, 33))) + 0.5])
DR_BINS = np.logspace(-4, 0, 33)

MD_COLUMNS = [
    "md_isPLS",
    "md_pt",
    "md_eta",
    "md_phi",
    "md_dphichange",
    "md_anchor_x",
    "md_anchor_y",
    "md_anchor_z",
    "md_type",
    "md_layer",
    "md_detId",
    "md_simIdx",
]
SIM_COLUMNS = [
    "sim_pt",
    "sim_eta",
    "sim_q",
    "sim_pdgId",
    "sim_event",
    "sim_bunchCrossing",
    "sim_vx",
    "sim_vy",
    "sim_vz",
]

GEOMETRY = {
    1: {"rt_median": 25.0, "abs_z_max": 119.9},
    2: {"rt_median": 37.4, "abs_z_max": 119.9},
    3: {"rt_median": 52.4, "abs_z_max": 119.9},
    4: {"rt_median": 67.4, "abs_z_max": 115.2},
    5: {"rt_median": 84.7, "abs_z_max": 115.2},
    6: {"rt_median": 106.9, "abs_z_max": 115.2},
    7: {"abs_z_median": 130.3, "rt_min": 23.7, "rt_max": 104.9},
    8: {"abs_z_median": 154.1, "rt_min": 23.7, "rt_max": 104.9},
    9: {"abs_z_median": 184.4, "rt_min": 33.2, "rt_max": 104.9},
    10: {"abs_z_median": 220.7, "rt_min": 33.2, "rt_max": 104.9},
    11: {"abs_z_median": 264.1, "rt_min": 33.2, "rt_max": 104.9},
}


@dataclass
class MDOfInterest:
    """One MD we extrapolate from: its own estimates, its sim, and the other MDs of that sim."""

    tree_entry: int
    i_md: int
    md_layer: int
    # barrel_flat, barrel_tilted or endcap
    md_region: str
    # PS or 2S
    md_module: str
    md_anchor_rt: float
    md_anchor_z: float
    md_eta: float
    md_phi: float
    md_pt: float
    # q/pT = -1 / md_pt, as in the module docstring
    md_qoverpt: float
    sim_qoverpt: float
    sim_pt: float
    sim_eta: float
    sim_vz: float
    sim_pdgId: int
    # Indices of the sim's other (non-pLS) MDs, on any layer
    md_siblings: list[int]


def main() -> None:
    args = parse_args()
    if args.dump_geometry:
        dump_geometry(input_path=args.input)
        return
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    rng = random.Random(args.seed)
    centers = DETAILED_CENTERS if args.detailed else SALIENT_CENTERS
    with uproot.open(args.input) as root_file:
        mds, sims = read_event(tree=root_file["tree"], tree_entry=args.tree_entry)
    check_lst_estimates(mds=mds, tree_entry=args.tree_entry)

    # Study every MD first, so the summary can be the first page
    studied = []
    records = []
    for i_md in pick_mds(mds=mds, sims=sims, rng=rng, n_mds=args.n_mds):
        md, layer_records = study_md(mds=mds, sims=sims, i_md=i_md)
        studied.append((md, layer_records))
        records.extend(layer_records)
    log_summary(records=records, centers=centers)

    with PdfPages(args.output) as pdf:
        plot_summary(records=records, centers=centers, pdf=pdf)
        for md, layer_records in studied:
            plot_md(
                mds=mds,
                md=md,
                layer_records=layer_records,
                centers=centers,
                pdf=pdf,
            )
    print(f"[done] output={args.output} n_pages={1 + len(studied)}")


def parse_args() -> Namespace:
    parser = ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--input", type=str, required=True, help="One LSTNtuple")
    parser.add_argument(
        "--tree_entry",
        type=int,
        default=0,
        help="Entry (event) of --input to draw the MDs of interest from",
    )
    parser.add_argument("--n_mds", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Also show the helix with the sim's q/pT and the helix from the sim's vertex",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="md_extrapolation.pdf",
        help="The pdf to write",
    )
    parser.add_argument(
        "--dump_geometry",
        action="store_true",
        help="Only print GEOMETRY, measured over all events of --input, to paste into this script",
    )
    return parser.parse_args()


def dump_geometry(input_path: str) -> None:
    """
    Print GEOMETRY, the extent of each layer from the anchor hits of all OT MDs over all events of the
    LSTNtuple, as a Python constant to paste at the top of this script. Only the entries the script uses: barrel
    layers, rt_median and abs_z_max; endcap layers, abs_z_median, rt_min and rt_max.
    """
    rt_on_layer = {layer: [] for layer in LAYERS}
    abs_z_on_layer = {layer: [] for layer in LAYERS}
    with uproot.open(input_path) as root_file:
        tree = root_file["tree"]
        n_events = tree.num_entries
        for tree_entry in range(n_events):
            mds, _ = read_event(tree=tree, tree_entry=tree_entry)
            for layer in LAYERS:
                on_layer = (mds["md_layer"] == layer) & ~mds["md_isPLS"]
                rt_on_layer[layer].append(mds["md_anchor_rt"][on_layer])
                abs_z_on_layer[layer].append(np.abs(mds["md_anchor_z"][on_layer]))

    print("# Extent of each layer [cm], from the anchor hits of all OT MDs")
    print(f"# in {n_events} events of {os.path.basename(input_path)} (--dump_geometry)")
    geometry = {}
    for layer in LAYERS:
        rt = np.concatenate(rt_on_layer[layer])
        abs_z = np.concatenate(abs_z_on_layer[layer])
        # float() so numpy scalars print as plain numbers, not np.float64(...)
        if layer in BARREL_LAYERS:
            geometry[layer] = {
                "rt_median": round(float(np.median(rt)), 1),
                "abs_z_max": round(float(abs_z.max()), 1),
            }
        else:
            geometry[layer] = {
                "abs_z_median": round(float(np.median(abs_z)), 1),
                "rt_min": round(float(rt.min()), 1),
                "rt_max": round(float(rt.max()), 1),
            }
    print("GEOMETRY = ", end="")
    pprint(geometry, sort_dicts=False)


def read_event(
    tree: uproot.TTree,
    tree_entry: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    MD and sim branches of one event, as numpy arrays.

    mds also gets md_anchor_rt (the anchor hit's transverse radius) and tree_entry.
    """
    arrays = tree.arrays(
        MD_COLUMNS + SIM_COLUMNS,
        entry_start=tree_entry,
        entry_stop=tree_entry + 1,
    )
    # tree.arrays loaded a single entry, so it is the first and only one in arrays
    only_entry = 0
    mds = {col: ak.to_numpy(arrays[col][only_entry]) for col in MD_COLUMNS}
    sims = {col: ak.to_numpy(arrays[col][only_entry]) for col in SIM_COLUMNS}
    mds["md_anchor_rt"] = np.hypot(mds["md_anchor_x"], mds["md_anchor_y"])
    mds["tree_entry"] = tree_entry
    return mds, sims


def check_lst_estimates(
    mds: dict[str, np.ndarray],
    tree_entry: int,
) -> None:
    """
    Recompute md_pt, md_eta and md_phi from the anchor hit and md_dphichange for every OT MD, and log the largest
    difference.
    """
    ot = ~mds["md_isPLS"]
    rt = mds["md_anchor_rt"][ot]
    # Compared as 1 / pT, which stays finite where md_dphichange = 0 makes md_pt infinite
    inverse_pt = np.sin(mds["md_dphichange"][ot]) / (rt * K2RINV_1GEV)
    eta = np.arcsinh(mds["md_anchor_z"][ot] / rt)
    phi = np.arctan2(mds["md_anchor_y"][ot], mds["md_anchor_x"][ot])
    inverse_pt_diff = np.abs(inverse_pt - 1.0 / mds["md_pt"][ot])
    eta_diff = np.abs(eta - mds["md_eta"][ot])
    phi_diff = np.abs(wrap_phi(phi=phi - mds["md_phi"][ot]))
    print(
        f"[check] tree_entry={tree_entry} n_ot_md={ot.sum()} max_diff_inverse_pt={inverse_pt_diff.max():.2e} "
        f"max_diff_eta={eta_diff.max():.2e} max_diff_phi={phi_diff.max():.2e} "
        f"frac_negative_md_pt={(mds['md_pt'][ot] < 0).mean():.3f}"
    )


def pick_mds(
    mds: dict[str, np.ndarray],
    sims: dict[str, np.ndarray],
    rng: random.Random,
    n_mds: int,
) -> list[int]:
    """
    n_mds MDs of interest (or all candidates, if fewer), drawn at random among the true OT MDs of a selected sim
    with at least one sibling on another layer, sorted by layer.
    """
    layers_of_sim = {}
    for i_md in range(len(mds["md_layer"])):
        simidx = mds["md_simIdx"][i_md]
        if mds["md_isPLS"][i_md] or unmatched(simidx=simidx):
            continue
        layers_of_sim.setdefault(simidx, set()).add(mds["md_layer"][i_md])

    two_layers = 2
    candidates = []
    for i_md in range(len(mds["md_layer"])):
        simidx = mds["md_simIdx"][i_md]
        if (
            mds["md_isPLS"][i_md]
            or unmatched(simidx=simidx)
            or not is_selected_sim(sims=sims, simidx=simidx)
        ):
            continue
        if len(layers_of_sim[simidx]) < two_layers:
            continue
        candidates.append(i_md)

    chosen = rng.sample(candidates, min(n_mds, len(candidates)))
    return sorted(chosen, key=lambda i_md: mds["md_layer"][i_md])


def unmatched(simidx: int) -> bool:
    """True for an MD without a matched sim (md_simIdx is negative)."""
    return simidx < 0


def is_selected_sim(
    sims: dict[str, np.ndarray],
    simidx: int,
) -> bool:
    """LST's type-7 denominator cuts on sim simidx, with pT >= PT_MIN."""
    return (
        sims["sim_q"][simidx] != 0
        and sims["sim_pt"][simidx] >= PT_MIN
        and abs(sims["sim_eta"][simidx]) <= ABS_ETA_MAX
        and np.hypot(sims["sim_vx"][simidx], sims["sim_vy"][simidx]) <= VERTEX_RT_MAX
        and abs(sims["sim_vz"][simidx]) <= VERTEX_ABS_Z_MAX
        and sims["sim_event"][simidx] == 0
        and sims["sim_bunchCrossing"][simidx] == 0
    )


def study_md(
    mds: dict[str, np.ndarray],
    sims: dict[str, np.ndarray],
    i_md: int,
) -> tuple[MDOfInterest, list[dict]]:
    """
    The five centers, and dR and k from each, on every layer for one MD of interest.

    md: the MD of interest, its sim and its siblings. layer_records: one per layer, with the centers
    (center_eta, center_phi), the layer's siblings and, per center, dr_<center> and k_<center> (None without a
    sibling on the layer).
    """
    simidx = mds["md_simIdx"][i_md]
    siblings = []
    for i_sibling in np.flatnonzero(mds["md_simIdx"] == simidx):
        if i_sibling != i_md and not mds["md_isPLS"][i_sibling]:
            siblings.append(i_sibling)
    md = MDOfInterest(
        tree_entry=mds["tree_entry"],
        i_md=i_md,
        md_layer=mds["md_layer"][i_md],
        md_region=region(mds=mds, i_md=i_md),
        md_module="PS" if mds["md_type"][i_md] == PS_TYPE else "2S",
        md_anchor_rt=mds["md_anchor_rt"][i_md],
        md_anchor_z=mds["md_anchor_z"][i_md],
        md_eta=mds["md_eta"][i_md],
        md_phi=mds["md_phi"][i_md],
        md_pt=mds["md_pt"][i_md],
        md_qoverpt=-1.0 / mds["md_pt"][i_md],
        sim_qoverpt=sims["sim_q"][simidx] / sims["sim_pt"][simidx],
        sim_pt=sims["sim_pt"][simidx],
        sim_eta=sims["sim_eta"][simidx],
        sim_vz=sims["sim_vz"][simidx],
        sim_pdgId=sims["sim_pdgId"][simidx],
        md_siblings=siblings,
    )
    road_layers = projected_layers(
        md_anchor_rt=md.md_anchor_rt,
        md_anchor_z=md.md_anchor_z,
        own_layer=md.md_layer,
    )
    print(
        f"[md] tree_entry={md.tree_entry} i_md={i_md} md_layer={md.md_layer} region={md.md_region} "
        f"module={md.md_module} md_pt={md.md_pt:.3f} md_qoverpt={md.md_qoverpt:+.4f} "
        f"sim_qoverpt={md.sim_qoverpt:+.4f} sim_pt={md.sim_pt:.3f} "
        f"sim_eta={md.sim_eta:+.3f} sim_vz={md.sim_vz:+.2f} sim_pdgId={md.sim_pdgId} "
        f"n_siblings={len(siblings)} road_layers={','.join(str(layer) for layer in road_layers)}"
    )

    sibling_layers = {mds["md_layer"][i_sibling] for i_sibling in siblings}
    sibling_layers -= {md.md_layer}
    layer_records = []
    for target_layer in sorted(set(road_layers) | sibling_layers):
        on_layer = np.flatnonzero((mds["md_layer"] == target_layer) & ~mds["md_isPLS"])
        siblings_here = [
            i_sibling
            for i_sibling in siblings
            if mds["md_layer"][i_sibling] == target_layer
        ]
        center_eta, center_phi = layer_centers(
            target_layer=target_layer,
            md=md,
        )
        record = {
            "md_info": md,
            "layer": target_layer,
            "in_road": target_layer in road_layers,
            "r_target": target_radius(
                target_layer=target_layer,
                md_anchor_rt=md.md_anchor_rt,
                md_anchor_z=md.md_anchor_z,
            ),
            "n_md_on_layer": len(on_layer),
            "siblings": siblings_here,
            "center_eta": center_eta,
            "center_phi": center_phi,
        }
        for name in DETAILED_CENTERS:
            if len(siblings_here) == 0:
                record[f"dr_{name}"] = None
                record[f"k_{name}"] = None
                continue
            record[f"dr_{name}"], record[f"k_{name}"] = nearest_sibling_k(
                center_eta=center_eta[name],
                center_phi=center_phi[name],
                mds=mds,
                siblings=siblings_here,
                on_layer=on_layer,
            )
        log_layer(record=record)
        layer_records.append(record)
    return md, layer_records


def region(
    mds: dict[str, np.ndarray],
    i_md: int,
) -> str:
    """
    barrel_flat, barrel_tilted or endcap, from md_layer and the barrel detId side bits (as LST's
    ModuleMethods.h).
    """
    if mds["md_layer"][i_md] in ENDCAP_LAYERS:
        return "endcap"
    side = (mds["md_detId"][i_md] & (3 << 18)) >> 18
    return "barrel_flat" if side in FLAT_SIDES else "barrel_tilted"


def projected_layers(
    md_anchor_rt: float,
    md_anchor_z: float,
    own_layer: int,
) -> list[int]:
    """Layers a straight line in r-z from the origin through the MD crosses, within ROAD_MARGIN of their extent."""
    # Slope of the line in an r-z plot with r on the vertical axis: rt = abs_r_over_z * |z|
    abs_r_over_z = abs(md_anchor_rt / md_anchor_z)
    road_layers = []
    for layer in LAYERS:
        if layer == own_layer:
            continue
        geo = GEOMETRY[layer]
        if layer in BARREL_LAYERS:
            # |z| where the line reaches the layer's radius
            z_at_crossing = geo["rt_median"] / abs_r_over_z
            crosses = z_at_crossing <= geo["abs_z_max"] * (1 + ROAD_MARGIN)
        else:
            # rt where the line reaches the layer's |z|
            r_at_crossing = abs_r_over_z * geo["abs_z_median"]
            crosses = (
                geo["rt_min"] * (1 - ROAD_MARGIN)
                <= r_at_crossing
                <= geo["rt_max"] * (1 + ROAD_MARGIN)
            )
        if crosses:
            road_layers.append(layer)
    return road_layers


def layer_centers(
    target_layer: int,
    md: MDOfInterest,
) -> tuple[dict[str, float], dict[str, float]]:
    """
    (center_eta, center_phi) of all DETAILED_CENTERS on the target layer, see the module docstring.

    own stays at the MD's (eta, phi). The three helix centers follow a helix from (0, 0, vz) through the MD, each
    with its own q/pT and vz. The oracle follows the sim's helix from its vertex, with the sim's eta.
    """
    center_eta = {"own": md.md_eta}
    center_phi = {"own": md.md_phi}
    # Per helix center through the MD: (q/pT, vz)
    helix_inputs = {
        "helix": (md.md_qoverpt, 0.0),
        "helix_sim_qoverpt": (md.sim_qoverpt, 0.0),
        "helix_sim_vz": (md.md_qoverpt, md.sim_vz),
    }
    for name, (qoverpt, vz) in helix_inputs.items():
        # z = vz + s * sinh(eta) at the MD gives the helix's sinh(eta)
        md_path_length = path_length(rt=md.md_anchor_rt, qoverpt=qoverpt)
        center_eta[name], center_phi[name] = helix_center(
            target_layer=target_layer,
            qoverpt=qoverpt,
            vz=vz,
            sinh_eta=(md.md_anchor_z - vz) / md_path_length,
            md=md,
        )
    center_eta["oracle"], center_phi["oracle"] = helix_center(
        target_layer=target_layer,
        qoverpt=md.sim_qoverpt,
        vz=md.sim_vz,
        sinh_eta=np.sinh(md.sim_eta),
        md=md,
    )
    return center_eta, center_phi


def helix_center(
    target_layer: int,
    qoverpt: float,
    vz: float,
    sinh_eta: float,
    md: MDOfInterest,
) -> tuple[float, float]:
    """
    (eta, phi) where a helix from (0, 0, vz) with q/pT and eta reaches the target layer. eta is seen from the
    origin, as md_eta; phi is the MD's, moved by the bend from the MD's radius to the layer's radius.
    """
    helix_rt, helix_z = helix_crossing(
        target_layer=target_layer,
        qoverpt=qoverpt,
        vz=vz,
        sinh_eta=sinh_eta,
        md_anchor_z=md.md_anchor_z,
    )
    phi_shift = -(
        bend(rt=helix_rt, qoverpt=qoverpt) - bend(rt=md.md_anchor_rt, qoverpt=qoverpt)
    )
    return np.arcsinh(helix_z / helix_rt), wrap_phi(phi=md.md_phi + phi_shift)


def target_radius(
    target_layer: int,
    md_anchor_rt: float,
    md_anchor_z: float,
) -> float:
    """
    Radius at which a straight line in r-z from the origin through the MD reaches the target layer: barrel, the
    layer's median rt; endcap, the radius where the line reaches the layer's median |z|.
    """
    if target_layer in BARREL_LAYERS:
        return GEOMETRY[target_layer]["rt_median"]
    return md_anchor_rt * GEOMETRY[target_layer]["abs_z_median"] / abs(md_anchor_z)


def helix_crossing(
    target_layer: int,
    qoverpt: float,
    vz: float,
    sinh_eta: float,
    md_anchor_z: float,
) -> tuple[float, float]:
    """
    (rt, z) where a helix from (0, 0, vz) with q/pT and eta reaches the target layer: barrel, at the layer's
    median rt; endcap, at the layer's median |z| on the MD's side.

    Along the helix, z = vz + s * sinh(eta), with s the path length in x-y (see path_length).
    """
    if target_layer in BARREL_LAYERS:
        rt = GEOMETRY[target_layer]["rt_median"]
        return rt, vz + path_length(rt=rt, qoverpt=qoverpt) * sinh_eta
    helix_z = np.sign(md_anchor_z) * GEOMETRY[target_layer]["abs_z_median"]
    helix_s = (helix_z - vz) / sinh_eta
    return radius_after(helix_s=helix_s, qoverpt=qoverpt), helix_z


def path_length(
    rt: float,
    qoverpt: float,
) -> float:
    """
    Path length in x-y of a helix from the beam line to radius rt: a circle of radius R reaches rt after
    s = 2R * asin(rt / 2R), with 1 / 2R = K2RINV_1GEV * |q/pT|. A straight line (q/pT = 0) has s = rt.
    """
    curvature = K2RINV_1GEV * abs(qoverpt)
    if curvature == 0:
        return rt
    return np.arcsin(min(curvature * rt, 1.0)) / curvature


def radius_after(
    helix_s: float,
    qoverpt: float,
) -> float:
    """
    Radius reached after path length helix_s in x-y, the inverse of path_length: rt = 2R * sin(s / 2R), with s =
    helix_s.
    """
    curvature = K2RINV_1GEV * abs(qoverpt)
    if curvature == 0:
        return helix_s
    return np.sin(curvature * helix_s) / curvature


def bend(
    rt: float | np.ndarray,
    qoverpt: float,
) -> float | np.ndarray:
    """phi0 - phi(rt) of a helix from the beam line, asin(K2RINV_1GEV * rt * q/pT), clipped to a half turn."""
    return np.arcsin(np.clip(K2RINV_1GEV * rt * qoverpt, -1, 1))


def nearest_sibling_k(
    center_eta: float,
    center_phi: float,
    mds: dict[str, np.ndarray],
    siblings: list[int],
    on_layer: np.ndarray,
) -> tuple[float, int]:
    """
    (dR, k) from one center: dR to the nearest sibling on the layer, and k, the number of MDs on the layer at
    least that close to the center (the sibling included, so k >= 1).
    """
    dr_siblings = []
    for i_sibling in siblings:
        dr_siblings.append(
            delta_r(
                eta_a=center_eta,
                phi_a=center_phi,
                eta_b=mds["md_eta"][i_sibling],
                phi_b=mds["md_phi"][i_sibling],
            )
        )
    radius = min(dr_siblings)
    dr_all = delta_r(
        eta_a=center_eta,
        phi_a=center_phi,
        eta_b=mds["md_eta"][on_layer],
        phi_b=mds["md_phi"][on_layer],
    )
    return radius, int(np.sum(dr_all <= radius))


def log_layer(record: dict) -> None:
    """One [layer] line: per center, its shift from the MD (dphi, deta), dR to the nearest sibling and k."""
    md = record["md_info"]
    fields = []
    for name in DETAILED_CENTERS:
        dphi = wrap_phi(phi=record["center_phi"][name] - md.md_phi)
        deta = record["center_eta"][name] - md.md_eta
        fields.append(
            f"dphi_{name}={dphi:+.4f} deta_{name}={deta:+.4f} "
            f"dr_{name}={fmt(value=record[f'dr_{name}'])} k_{name}={record[f'k_{name}']}"
        )
    print(
        f"[layer] tree_entry={md.tree_entry} i_md={md.i_md} md_layer={md.md_layer} "
        f"layer={record['layer']} in_road={record['in_road']} r_target={record['r_target']:.1f} "
        f"n_md_on_layer={record['n_md_on_layer']} n_siblings={len(record['siblings'])} "
        + " ".join(fields)
    )


def wrap_phi(phi: float | np.ndarray) -> float | np.ndarray:
    """phi wrapped into [-pi, pi)."""
    return (phi + np.pi) % (2 * np.pi) - np.pi


def delta_r(
    eta_a: float | np.ndarray,
    phi_a: float | np.ndarray,
    eta_b: float | np.ndarray,
    phi_b: float | np.ndarray,
) -> float | np.ndarray:
    return np.hypot(eta_a - eta_b, wrap_phi(phi=phi_a - phi_b))


def fmt(value: float | None) -> str:
    return "None" if value is None else f"{value:.4f}"


def plot_md(
    mds: dict[str, np.ndarray],
    md: MDOfInterest,
    layer_records: list[dict],
    centers: list[str],
    pdf: PdfPages,
) -> None:
    """One page per MD of interest: the x-y view, then one panel per layer."""
    n_panels = 1 + len(layer_records)
    n_cols = 4
    n_rows = (n_panels + n_cols - 1) // n_cols
    fig, axs = plt.subplots(
        nrows=n_rows, ncols=n_cols, figsize=(4.2 * n_cols, 4.2 * n_rows)
    )
    axs = axs.ravel()
    plot_transverse(ax=axs[0], mds=mds, md=md)
    for ax, record in zip(axs[1:], layer_records):
        plot_layer(ax=ax, mds=mds, md=md, record=record, centers=centers)
    for ax in axs[n_panels:]:
        ax.axis("off")
    axs[1].legend(fontsize="x-small", loc="upper left")
    fig.suptitle(
        f"MD of interest: tree entry {md.tree_entry}, layer {md.md_layer}, {md.md_region}, {md.md_module}"
        f" | md_pt={md.md_pt:.2f} GeV, q/pT estimate {md.md_qoverpt:+.3f}\n"
        f"Sim: pdgId {md.sim_pdgId}, pT {md.sim_pt:.2f} GeV, eta {md.sim_eta:+.2f}, "
        f"q/pT {md.sim_qoverpt:+.3f} | {len(md.md_siblings)} sibling MDs",
        fontsize=11,
    )
    fig.tight_layout()
    pdf.savefig(fig, dpi=150)  # dpi of the rasterized background MDs only
    plt.close(fig)


def plot_transverse(
    ax: plt.Axes,
    mds: dict[str, np.ndarray],
    md: MDOfInterest,
) -> None:
    """
    x-y view: the sim's MDs, the straight line (own phi) and helices from the beam line through the MD. In x-y, the
    helix with the sim's q/pT is the oracle's track.
    """
    r_max = max(mds["md_anchor_rt"][i_md] for i_md in md.md_siblings + [md.i_md]) * 1.05
    rt = np.linspace(0, r_max, 200)
    for label, qoverpt, name in [
        ("own: phi constant", 0.0, "own"),
        ("helix, MD q/pT", md.md_qoverpt, "helix"),
        ("oracle, sim q/pT", md.sim_qoverpt, "oracle"),
    ]:
        phi = md.md_phi - (
            bend(rt=rt, qoverpt=qoverpt) - bend(rt=md.md_anchor_rt, qoverpt=qoverpt)
        )
        ax.plot(
            rt * np.cos(phi),
            rt * np.sin(phi),
            color=COLORS[name],
            linestyle=LINESTYLES[name],
            linewidth=1.5,
            label=label,
        )
    ax.scatter(
        [mds["md_anchor_x"][i_sibling] for i_sibling in md.md_siblings],
        [mds["md_anchor_y"][i_sibling] for i_sibling in md.md_siblings],
        s=25,
        color=COLORS["sibling"],
        edgecolor="white",
        zorder=3,
        label="siblings",
    )
    ax.scatter(
        mds["md_anchor_x"][md.i_md],
        mds["md_anchor_y"][md.i_md],
        s=90,
        marker="*",
        color=COLORS["md"],
        edgecolor="black",
        zorder=4,
        label="MD of interest",
    )
    ax.scatter(0, 0, s=15, color="black", marker="+")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x [cm]")
    ax.set_ylabel("y [cm]")
    ax.set_title("Transverse view", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize="x-small", loc="best")


def plot_layer(
    ax: plt.Axes,
    mds: dict[str, np.ndarray],
    md: MDOfInterest,
    record: dict,
    centers: list[str],
) -> None:
    """MDs of one layer in (phi, eta), with the centers and the circles reaching the nearest sibling."""
    # phi is drawn as md_phi + (wrapped difference to md_phi), so a panel near phi = ±pi does not split in two;
    # its axis may then run slightly beyond ±pi
    center_x = {}
    for name in centers:
        center_x[name] = md.md_phi + wrap_phi(
            phi=record["center_phi"][name] - md.md_phi
        )
    sibling_x = []
    sibling_y = []
    for i_sibling in record["siblings"]:
        sibling_x.append(md.md_phi + wrap_phi(phi=mds["md_phi"][i_sibling] - md.md_phi))
        sibling_y.append(mds["md_eta"][i_sibling])

    # A square window around the centers, siblings and circles
    xs = list(center_x.values()) + sibling_x
    ys = [record["center_eta"][name] for name in centers] + sibling_y
    radii = [record[f"dr_{name}"] or 0.0 for name in centers]
    middle_x = 0.5 * (min(xs) + max(xs))
    middle_y = 0.5 * (min(ys) + max(ys))
    half = 0.5 * max(max(xs) - min(xs), max(ys) - min(ys)) + max(radii) + WINDOW

    on_layer = (mds["md_layer"] == record["layer"]) & ~mds["md_isPLS"]
    bkg_x = md.md_phi + wrap_phi(phi=mds["md_phi"][on_layer] - md.md_phi)
    bkg_y = mds["md_eta"][on_layer]
    in_window = (np.abs(bkg_x - middle_x) < half) & (np.abs(bkg_y - middle_y) < half)
    ax.scatter(
        bkg_x[in_window],
        bkg_y[in_window],
        s=3,
        color=BACKGROUND_COLOR,
        rasterized=True,
        label="MDs on layer",
    )
    ax.scatter(
        sibling_x,
        sibling_y,
        s=40,
        color=COLORS["sibling"],
        edgecolor="black",
        zorder=3,
        label="siblings",
    )
    for name in centers:
        center = (center_x[name], record["center_eta"][name])
        ax.scatter(
            *center,
            s=60,
            marker="x",
            color=COLORS[name],
            zorder=4,
            label=f"center, {CENTER_LABELS[name]}",
        )
        if record[f"dr_{name}"] is not None:
            ax.add_patch(
                plt.Circle(
                    center,
                    record[f"dr_{name}"],
                    fill=False,
                    color=COLORS[name],
                    linestyle=LINESTYLES[name],
                    linewidth=1.5,
                )
            )
    ax.set_xlim(middle_x - half, middle_x + half)
    ax.set_ylim(middle_y - half, middle_y + half)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$\phi$")
    ax.set_ylabel(r"$\eta$")
    ax.grid(alpha=0.3)

    where = "barrel" if record["layer"] in BARREL_LAYERS else "endcap"
    road = "" if record["in_road"] else ", not in road"
    if len(record["siblings"]) == 0:
        result = "no sibling"
    else:
        k_per_center = []
        for name in centers:
            k_per_center.append(f"{CENTER_LABELS[name]} {record[f'k_{name}']}")
        result = "k: " + ", ".join(k_per_center)
    helix_dphi = wrap_phi(phi=record["center_phi"]["helix"] - md.md_phi)
    helix_deta = record["center_eta"]["helix"] - md.md_eta
    ax.set_title(
        f"Layer {record['layer']} ({where}{road})\n{result}\n"
        f"r = {record['r_target']:.0f} cm, helix dphi {helix_dphi:+.3f} deta {helix_deta:+.3f}",
        fontsize=10,
    )


def log_summary(
    records: list[dict],
    centers: list[str],
) -> None:
    """
    Median and 90% quantile of k per center, and how often each COMPARISONS center beats its reference (when both
    are in centers), per region of the MD of interest, per sim pT bin and overall.
    """
    with_sibling = [rec for rec in records if len(rec["siblings"]) > 0]
    groups = []
    for name in REGIONS:
        groups.append(
            (
                f"region={name}",
                [rec for rec in with_sibling if rec["md_info"].md_region == name],
            )
        )
    for low, high in PT_BINS:
        groups.append(
            (
                f"sim_pt={low}-{high}",
                [rec for rec in with_sibling if low <= rec["md_info"].sim_pt < high],
            )
        )
    groups.append(("all", with_sibling))

    for label, group in groups:
        if len(group) == 0:
            continue
        fields = []
        for name in centers:
            knn = [rec[f"k_{name}"] for rec in group]
            fields.append(
                f"k_{name}_median={np.median(knn):.0f} k_{name}_q90={np.quantile(knn, 0.9):.0f}"
            )
        for name, reference in COMPARISONS:
            if name not in centers or reference not in centers:
                continue
            n_better = sum(
                1 for rec in group if rec[f"k_{name}"] < rec[f"k_{reference}"]
            )
            n_worse = sum(
                1 for rec in group if rec[f"k_{name}"] > rec[f"k_{reference}"]
            )
            fields.append(
                f"n_{name}_better_than_{reference}={n_better} "
                f"n_{name}_worse_than_{reference}={n_worse}"
            )
        print(f"[summary] {label} n_layers={len(group)} " + " ".join(fields))


def plot_summary(
    records: list[dict],
    centers: list[str],
    pdf: PdfPages,
) -> None:
    """The summary page: k and dR to the nearest sibling per center, and k vs own."""
    with_sibling = [rec for rec in records if len(rec["siblings"]) > 0]
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 4.6))
    for name in centers:
        label = CENTER_LABELS[name]
        axs[0].hist(
            [rec[f"k_{name}"] for rec in with_sibling],
            bins=K_BINS,
            histtype="step",
            linewidth=2,
            color=COLORS[name],
            linestyle=LINESTYLES[name],
            label=label,
        )
        axs[1].hist(
            [max(rec[f"dr_{name}"], DR_BINS[0]) for rec in with_sibling],
            bins=DR_BINS,
            histtype="step",
            linewidth=2,
            color=COLORS[name],
            linestyle=LINESTYLES[name],
            label=label,
        )
    axs[0].set_xlabel("k: MDs on the layer within dR of the nearest sibling")
    axs[1].set_xlabel("dR from the center to the nearest sibling")
    for ax in axs[:2]:
        ax.set_xscale("log")
        ax.set_ylabel("(MD of interest, layer) pairs")
        ax.grid(alpha=0.3)
        ax.legend(title="Center", fontsize="small")

    for name in centers:
        if name == "own":
            continue
        axs[2].scatter(
            [rec["k_own"] for rec in with_sibling],
            [rec[f"k_{name}"] for rec in with_sibling],
            s=10,
            color=COLORS[name],
            alpha=0.6,
            label=CENTER_LABELS[name],
        )
    axs[2].plot([1, 1e4], [1, 1e4], color="black", linewidth=1, linestyle=":")
    axs[2].set_xscale("log")
    axs[2].set_yscale("log")
    axs[2].set_xlabel("k, own (eta, phi)")
    axs[2].set_ylabel("k, shifted center")
    axs[2].legend(title="Center", fontsize="small")
    axs[2].set_title("Below the diagonal: the shift helps", fontsize=10)
    axs[2].grid(alpha=0.3)

    n_mds = len({(rec["md_info"].tree_entry, rec["md_info"].i_md) for rec in records})
    fig.suptitle(
        f"pT-extrapolated vs plain (eta, phi) neighbourhood: {n_mds} MDs of interest, "
        f"{len(with_sibling)} (MD, layer) pairs with a sibling",
        fontsize=11,
    )
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


if __name__ == "__main__":
    main()
