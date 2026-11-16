"""
LSTNtuple branch names the harness reads, in one place.
"""

SIM_COLUMNS = [
    "sim_q",
    "sim_pt",
    "sim_eta",
    "sim_phi",
    "sim_vtxperp",
    "sim_vz",
    "sim_bunchCrossing",
    "sim_event",
]
MD_COLUMNS = [
    "md_isPLS",
    "md_layer",
    "md_simIdx",
    "md_anchor_hitIdx",
    "md_other_hitIdx",
    "md_anchor_simIdxs",
    "md_other_simIdxs",
]
T5_COLUMNS = [
    "t5_hitIndices",
    "t5_dnnScore",
    "t5_pt",
    "t5_eta",
    "t5_phi",
    "t5_isDupBitmask",
]
# Everything the harness reads, for every event
COLUMNS = SIM_COLUMNS + MD_COLUMNS + T5_COLUMNS
