"""
Read per-event MD files created by md_processing.py
"""

import glob
import os
from dataclasses import dataclass

import torch

MD_FEATURES = [
    "layer",
    "rt",
    "cosphi",
    "sinphi",
    "z",
    "drt",
    "dphi",
    "dz",
    "eta",
    "qoverpt",
]
SIM_FEATURES = [
    "sim_pt",
    "sim_eta",
    "sim_phi",
    "sim_q",
    "sim_vx",
    "sim_vy",
    "sim_vz",
    "sim_vtxperp",
    "sim_event",
    "sim_bunchCrossing",
]


@dataclass
class MDsFromOneEvent:
    """One event's MDs: entry i of every md_* field belongs to MD i."""

    md_features: torch.Tensor      # (n_md, len(MD_FEATURES))
    md_simIdx: torch.Tensor        # (n_md,), < 0 if no sim
    md_simFeatures: torch.Tensor   # (n_md, len(SIM_FEATURES))
    md_hitIdx: torch.Tensor        # (n_md, 2): LST hit index of the [inner, outer] hit
    md_inner_simIdxs: torch.Tensor # (n_md, n_max_simIdxs): sims of the inner hit, padded
    md_outer_simIdxs: torch.Tensor # (n_md, n_max_simIdxs): same for the outer hit
    lstntuple_file: str
    lstntuple_entry: int


def read_event(path: str) -> MDsFromOneEvent:
    """The event in the .pt file at path."""
    data = torch.load(path, weights_only=True)
    return MDsFromOneEvent(
        md_features=data["md_features"],
        md_simIdx=data["md_simIdx"],
        md_simFeatures=data["md_simFeatures"],
        md_hitIdx=data["md_hitIdx"],
        md_inner_simIdxs=data["md_inner_simIdxs"],
        md_outer_simIdxs=data["md_outer_simIdxs"],
        lstntuple_file=data["lstntuple_file"],
        lstntuple_entry=data["lstntuple_entry"],
    )


def event_paths(pt_dir: str) -> list[str]:
    """Every .pt file in pt_dir"""
    paths = glob.glob(os.path.join(pt_dir, "*.pt"))
    return sorted(paths, key=lstntuple_and_entry)


def lstntuple_and_entry(path: str) -> tuple[str, int]:
    """The LSTNtuple name and entry N"""
    # .../LSTNtuple_X.root.entry_7.pt -> ("LSTNtuple_X.root", 7)
    bname, entry = os.path.basename(path).removesuffix(".pt").split(".entry_")
    return bname, int(entry)
