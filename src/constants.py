import math

# Range from which random eta/phi mask centers are drawn
ETA_MIN, ETA_MAX = -2.6, 2.6
PHI_MIN, PHI_MAX = -math.pi, math.pi

# Unit conversions
BYTES_PER_GB = 1e9

# Column layout of the node features x, as written by transformer-datasets
# (transformer_datasets/constants.py, t3_processing.py): T3 variables, then the LS variables of
# each of the T3's 2 LSs, then the MD variables of each of its 3 distinct MDs, inner to outer.
# Columns are looked up by name. PCDataset checks that x has len(FEATURE_NAMES) columns.
T3_VARS = ["t3_pt", "t3_eta", "t3_phi"]
LS_VARS = ["ls_dPhis", "ls_dPhiChanges", "ls_dAlphaInners", "ls_dAlphaOuters", "ls_dAlphaInnerOuters"]
MD_VARS = ["md_anchor_x", "md_anchor_y", "md_anchor_z", "md_other_x", "md_other_y", "md_other_z",
           "md_dphi", "md_dphichange", "md_dz"]
N_T3_LSS = 2
N_T3_MDS = 3
FEATURE_NAMES = (
    T3_VARS
    + [f"ls{i}_{v.removeprefix('ls_')}" for i in range(N_T3_LSS) for v in LS_VARS]
    + [f"md{i}_{v.removeprefix('md_')}" for i in range(N_T3_MDS) for v in MD_VARS]
)
ETA_FEATURE = "t3_eta"
PHI_FEATURE = "t3_phi"
T3_ETA_COLUMN = FEATURE_NAMES.index(ETA_FEATURE)
T3_PHI_COLUMN = FEATURE_NAMES.index(PHI_FEATURE)
# Features that are log-transformed before min/max normalization
LOG_FEATURES = ["t3_pt"]

# Legacy normalization, used only when PCDataset is given no normalization (e.g. analysis of
# checkpoints trained before the normalization was computed from the training data).
# Feature 0 (pT) is log-transformed before normalization.
# T3 graphs (data/t3_processing.py): 40 features
# [pt, eta, phi | LS0 (5) | LS1 (5) | MD0 (9) | MD1 (9) | MD2 (9)], MDs inner to outer
T3_FEATURE_MIN = [-3.4627, -2.4507, -3.1416, -0.2362, -0.7272, -0.8202, -0.8768, -0.5988,
                  -0.2704, -0.8894, -1.0590, -1.3935, -0.7862, -73.8012, -72.4212, -187.5750,
                  -73.8019, -77.1076, -187.8705, -0.0231, -0.5393, -5.0250, -93.8255, -93.9099,
                  -223.8540, -97.9344, -93.9060, -224.1495, -0.0111, -0.8231, -5.0255, -110.0160,
                  -110.0150, -267.2350, -110.1943, -110.1950, -267.6350, -0.0046, -0.9416, -5.0259]
T3_FEATURE_MAX = [7.5964, 2.4535, 3.1416, 0.2434, 0.7304, 0.7744, 0.8728, 0.5682,
                  0.2755, 0.8922, 1.1500, 1.5453, 0.8093, 73.7524, 70.4150, 187.5750,
                  73.7676, 71.4665, 187.8705, 0.0228, 0.5575, 5.0250, 93.8869, 93.8406,
                  223.8540, 96.1440, 96.9953, 224.1495, 0.0097, 0.8098, 5.0253, 110.0128,
                  110.0150, 267.2350, 110.1814, 110.1950, 267.6350, 0.0050, 0.9323, 5.0260]

# Padding of the hit_{k}_simIdxs fields of the graphs (UNMATCHED in transformer_datasets/constants.py)
HIT_UNMATCHED = -999
