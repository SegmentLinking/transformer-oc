"""
Tests for mdml, one section per module, bottom-up.

From the repo root:
apptainer exec /ceph/users/atuna/ml-tracking.sif python3 -m pytest mdml/test_mdml.py
"""

import math
import os

import numpy as np
import torch

from mdml.clustering import NO_MDCLUSTER, find_mdclusters
from mdml.data import (
    MD_FEATURES,
    SIM_FEATURES,
    MDsFromOneEvent,
    event_paths,
    read_event,
)
from mdml.simple_mdnet import (
    N_INPUTS,
    N_LAYERS,
    SIM_KINEMATICS,
    MDNet,
    SimpleMDNet,
    loss_terms,
    md_inputs,
    md_targets,
)
from mdml.trackcands import trackcands_from_mdclusters

# data.py


def test_read_event(tmp_path):
    # Two MDs, as md_processing.py writes them
    path = os.path.join(tmp_path, "a.root.entry_0.pt")
    torch.save(
        {
            "md_features": torch.zeros(2, len(MD_FEATURES)),
            "md_simIdx": torch.tensor([4, -999]),
            "md_simFeatures": torch.zeros(2, len(SIM_FEATURES)),
            "md_hitIdx": torch.tensor([[10, 11], [12, 13]]),
            "md_inner_simIdxs": torch.tensor([[4], [-999]], dtype=torch.int32),
            "md_outer_simIdxs": torch.tensor([[4], [7]], dtype=torch.int32),
            "lstntuple_file": "a.root",
            "lstntuple_entry": 0,
        },
        path,
    )
    event = read_event(path=path)
    assert event.md_simIdx.tolist() == [4, -999]
    assert event.md_hitIdx.tolist() == [[10, 11], [12, 13]]
    assert event.md_inner_simIdxs.tolist() == [[4], [-999]]
    assert event.md_outer_simIdxs.tolist() == [[4], [7]]
    assert event.md_features.shape == (2, len(MD_FEATURES))
    assert event.lstntuple_file == "a.root"
    assert event.lstntuple_entry == 0


def test_event_paths(tmp_path):
    # Sorted by LSTNtuple name, then by entry as a number (2 before 10); non-.pt files are left out
    for name in [
        "b.root.entry_10.pt",
        "b.root.entry_2.pt",
        "a.root.entry_0.pt",
        "notes.txt",
    ]:
        open(os.path.join(tmp_path, name), "w").close()
    paths = event_paths(pt_dir=str(tmp_path))
    assert [os.path.basename(path) for path in paths] == [
        "a.root.entry_0.pt",
        "b.root.entry_2.pt",
        "b.root.entry_10.pt",
    ]


# simple_mdnet.py


UNMATCHED = -999


def make_mds(
    md_simIdx: list[int],
    sim_rows: list[dict[str, float]],
) -> MDsFromOneEvent:
    """MDs with zero features except layer 1, and the given sim features (others 0)."""
    n_md = len(md_simIdx)
    md_features = torch.zeros(n_md, len(MD_FEATURES))
    md_features[:, MD_FEATURES.index("layer")] = 1
    md_simFeatures = torch.zeros(n_md, len(SIM_FEATURES))
    for i_md, row in enumerate(sim_rows):
        for name, value in row.items():
            md_simFeatures[i_md, SIM_FEATURES.index(name)] = value
    return MDsFromOneEvent(
        md_features=md_features,
        md_simIdx=torch.tensor(md_simIdx),
        md_simFeatures=md_simFeatures,
        md_hitIdx=torch.zeros(n_md, 2, dtype=torch.int64),
        md_inner_simIdxs=torch.full((n_md, 1), UNMATCHED, dtype=torch.int32),
        md_outer_simIdxs=torch.full((n_md, 1), UNMATCHED, dtype=torch.int32),
        lstntuple_file="a.root",
        lstntuple_entry=0,
    )


def test_md_inputs():
    # MD 0: layer 1 at rt = 55 cm, z = -135 cm; MD 1: layer 11 with dphi = 0.005
    md_features = torch.zeros(2, len(MD_FEATURES))
    md_features[:, MD_FEATURES.index("layer")] = torch.tensor([1.0, 11.0])
    md_features[0, MD_FEATURES.index("rt")] = 55.0
    md_features[0, MD_FEATURES.index("z")] = -135.0
    md_features[1, MD_FEATURES.index("dphi")] = 0.005
    inputs = md_inputs(md_features=md_features)
    assert inputs.shape == (2, N_INPUTS)
    # Layer one-hot in the first N_LAYERS columns
    assert inputs[0, :N_LAYERS].tolist() == [1.0] + [0.0] * (N_LAYERS - 1)
    assert inputs[1, :N_LAYERS].tolist() == [0.0] * (N_LAYERS - 1) + [1.0]
    # Then the scaled features, in FEATURE_SCALES order: rt, cosphi, sinphi, z, drt, dphi, ...
    assert torch.allclose(inputs[0, N_LAYERS], torch.tensor(0.5))
    assert torch.allclose(inputs[0, N_LAYERS + 3], torch.tensor(-0.5))
    assert torch.allclose(inputs[1, N_LAYERS + 5], torch.tensor(0.5))


def test_md_targets():
    # MD 0: matched, sim pT 2 GeV, q -1; MD 1: matched to a 0.5 GeV sim; MD 2: unmatched
    mds = make_mds(
        md_simIdx=[3, 5, UNMATCHED],
        sim_rows=[
            {"sim_pt": 2.0, "sim_eta": 1.2, "sim_phi": math.pi / 2, "sim_q": -1.0},
            {"sim_pt": 0.5, "sim_eta": 0.0, "sim_phi": 0.0, "sim_q": 1.0},
            {name: UNMATCHED for name in SIM_FEATURES},
        ],
    )
    md_simIsFake, md_simKinematics = md_targets(mds=mds, pt_min=0.8)
    assert md_simIsFake.tolist() == [False, True, True]
    assert md_simKinematics.shape == (3, len(SIM_KINEMATICS))
    # q/pT, eta, cos phi, sin phi
    assert torch.allclose(
        input=md_simKinematics[0],
        other=torch.tensor([-0.5, 1.2, 0.0, 1.0]),
        atol=1e-6,
    )


def test_mdnet_shapes():
    md_predIsFakeLogit, md_predKinematics = MDNet()(torch.zeros(7, N_INPUTS))
    assert md_predIsFakeLogit.shape == (7,)
    assert md_predKinematics.shape == (7, len(SIM_KINEMATICS))


def test_loss_terms_ignores_fake_kinematics():
    # MD 1 is fake: its predicted kinematics are far off, but don't enter the MSE
    md_predIsFakeLogit = torch.tensor([0.0, 0.0])
    md_predKinematics = torch.tensor(
        [[1.0, 1.0, 1.0, 1.0], [100.0, 100.0, 100.0, 100.0]]
    )
    md_simIsFake = torch.tensor([False, True])
    md_simKinematics = torch.tensor([[1.0, 1.0, 1.0, 3.0], [0.0, 0.0, 0.0, 0.0]])
    loss_bce, loss_mse = loss_terms(
        md_predIsFakeLogit=md_predIsFakeLogit,
        md_predKinematics=md_predKinematics,
        md_simIsFake=md_simIsFake,
        md_simKinematics=md_simKinematics,
    )
    # Logit 0 is p = 0.5 for both MDs: BCE = ln 2
    assert torch.allclose(loss_bce, torch.tensor(math.log(2.0)))
    # One of the 4 kinematics off by 2: MSE = 4 / 4
    assert torch.allclose(loss_mse, torch.tensor(1.0))


def test_find_mdclusters():
    # MDs 0-2 close together, MDs 3-5 close together elsewhere, MD 6 alone (noise), MD 7 next to MD 0 but not kept
    md_predKinematics = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.01, 0.0, 0.0, 0.0],
            [0.0, 0.01, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [1.01, 1.0, 0.0, 0.0],
            [1.0, 1.01, 0.0, 0.0],
            [5.0, 5.0, 0.0, 0.0],
            [0.0, 0.0, 0.01, 0.0],
        ]
    )
    md_isKept = np.array([True] * 7 + [False])
    md_mdclusterIdx = find_mdclusters(
        md_predKinematics=md_predKinematics,
        md_isKept=md_isKept,
        eps=0.1,
    )
    assert md_mdclusterIdx.tolist() == [0, 0, 0, 1, 1, 1, NO_MDCLUSTER, NO_MDCLUSTER]


def test_trackcands_from_mdclusters():
    # MD cluster 0, MDs 0-2: unique hits 10-14, hit 11 shared by MDs 0 and 1. Hits 10-13 have sim 7 (hit 13 also
    # sim 9), hit 14 sim 8: 4 of 5 hits is 80% > 75%, so sim 7.
    # MD cluster 1, MDs 3-5: hits 20-25, each of a different sim, so unmatched. MD 6 is in no MD cluster.
    mds = make_mds(md_simIdx=[UNMATCHED] * 7, sim_rows=[])
    mds.md_hitIdx = torch.tensor(
        [[10, 11], [11, 12], [13, 14], [20, 21], [22, 23], [24, 25], [30, 31]]
    )
    pad = UNMATCHED
    mds.md_inner_simIdxs = torch.tensor(
        [[7, pad], [7, pad], [7, 9], [1, pad], [3, pad], [5, pad], [7, pad]],
        dtype=torch.int32,
    )
    mds.md_outer_simIdxs = torch.tensor(
        [[7, pad], [7, pad], [8, pad], [2, pad], [4, pad], [6, pad], [7, pad]],
        dtype=torch.int32,
    )
    trackcands = trackcands_from_mdclusters(
        mds=mds,
        md_mdclusterIdx=np.array([0, 0, 0, 1, 1, 1, NO_MDCLUSTER]),
        md_predProbFake=np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.9]),
        md_predKinematics=np.array(
            [[0.5, 1.0, 0.0, 1.0]] * 3 + [[-0.25, -1.0, 1.0, 0.0]] * 3 + [[0.0] * 4]
        ),
    )
    assert trackcands.trackcand_hitIndices.tolist() == [
        [10, 11, 12, 13, 14, -999],
        [20, 21, 22, 23, 24, 25],
    ]
    assert trackcands.trackcand_simIdx.tolist() == [7, UNMATCHED]
    # Mean of 1 - p(fake); pT = 1 / |q/pT|; phi = atan2(sin phi, cos phi)
    assert np.allclose(trackcands.trackcand_score, [0.8, 1.0])
    assert np.allclose(trackcands.trackcand_pt, [2.0, 4.0])
    assert np.allclose(trackcands.trackcand_eta, [1.0, -1.0])
    assert np.allclose(trackcands.trackcand_phi, [math.pi / 2, 0.0])


def test_save_and_load(tmp_path):
    simple_mdnet = SimpleMDNet(device="cpu", pt_min=0.7)
    simple_mdnet.fake_cut = 0.6
    simple_mdnet.eps = 0.08
    path = os.path.join(tmp_path, "simple_mdnet.pt")
    simple_mdnet.save(path=path)
    loaded = SimpleMDNet.load(path=path, device="cpu")
    assert (loaded.fake_cut, loaded.eps, loaded.pt_min) == (0.6, 0.08, 0.7)
    inputs = torch.rand(5, N_INPUTS)
    assert torch.equal(loaded.mdnet(inputs)[1], simple_mdnet.mdnet(inputs)[1])
