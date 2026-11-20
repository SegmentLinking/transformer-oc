import os
import re
import torch
from torch_geometric.data import Dataset
import glob

from constants import FEATURE_NAMES, ETA_FEATURE, PHI_FEATURE, T3_FEATURE_MIN, T3_FEATURE_MAX
from constants import ETA_MIN, ETA_MAX, PHI_MIN, PHI_MAX
from normalization import check_feature_count, check_normalization, log_transform

# Graph fields that are NOT one row per T3, and so are not cut by the eta/phi mask.
# Every other tensor field must have one row per T3 and is cut (see apply_eta_phi_mask).
NON_T3_FIELDS = ["sim_features"]


def legacy_normalization():
    """The hardcoded normalization from src/constants.py, for checkpoints trained before it was computed from data."""
    return {"names": list(FEATURE_NAMES), "min": T3_FEATURE_MIN, "max": T3_FEATURE_MAX, "log": ["t3_pt"]}


def natural_sort_key(path):
    """Sort key so that numbers in file names sort numerically: entry_2 before entry_10, graph_2 before graph_10."""
    return [int(tok) if tok.isdigit() else tok for tok in re.split(r"(\d+)", os.path.basename(path))]


def list_graph_files(data_dir):
    """All .pt graph files in data_dir, in natural order (e.g. LSTNtuple_<name>.root.entry_<N>.pt, or graph_<N>.pt)."""
    return sorted(glob.glob(os.path.join(data_dir, "*.pt")), key=natural_sort_key)


class PCDataset(Dataset):
    def __init__(
        self,
        data_dir,
        subset=None,
        eta_mask_half_width=None,
        phi_mask_half_width=None,
        eta_mask_center=0.0,
        phi_mask_center=0.0,
        normalization=None,
    ):
        """
        normalization: dict from normalization.compute_normalization (per-feature min/max and log columns).
        None falls back to the legacy hardcoded ranges in src/constants.py.
        """
        self.data_dir = data_dir
        if normalization is None:
            normalization = legacy_normalization()
        check_normalization(normalization, FEATURE_NAMES)
        self.normalization = normalization
        self.log_features = list(normalization["log"])
        self.feature_min = torch.tensor(normalization["min"], dtype=torch.float32)
        self.feature_max = torch.tensor(normalization["max"], dtype=torch.float32)
        self.eta_column = FEATURE_NAMES.index(ETA_FEATURE)
        self.phi_column = FEATURE_NAMES.index(PHI_FEATURE)
        self.subset = subset if subset is not None else float('inf')

        # eta/phi mask: A center of None means it is drawn at random for every event.
        # A half width of None means no cut in that coordinate, and the center must then be 0.
        validate_mask_args(
            eta_center=eta_mask_center,
            phi_center=phi_mask_center,
            eta_half_width=eta_mask_half_width,
            phi_half_width=phi_mask_half_width,
        )
        self.eta_mask_center = eta_mask_center
        self.phi_mask_center = phi_mask_center
        self.eta_mask_half_width = eta_mask_half_width
        self.phi_mask_half_width = phi_mask_half_width

        self.graph_files = list_graph_files(data_dir)

        if len(self.graph_files) == 0:
            raise ValueError(f"No graph files found in {data_dir}")

    def __len__(self):
        return min(len(self.graph_files), self.subset)

    def __getitem__(self, idx):
        graph_path = self.graph_files[idx]
        graph = torch.load(graph_path, map_location='cpu', weights_only=False)
        check_feature_count(graph.x, FEATURE_NAMES, graph_path)

        # Raw (unnormalized) eta/phi of every T3, for the model's Delta R neighbor search
        graph.eta = graph.x[:, self.eta_column].clone()
        graph.phi = graph.x[:, self.phi_column].clone()

        self.apply_eta_phi_mask(graph)

        x = log_transform(graph.x, FEATURE_NAMES, self.log_features)
        graph.x = (x - self.feature_min) / (self.feature_max - self.feature_min + 1e-8)

        graph.filename = os.path.basename(graph_path)
        return graph

    def apply_eta_phi_mask(self, graph):
        """
        Cut all per-T3 fields of `graph` (every tensor field except NON_T3_FIELDS) in place to
        the eta/phi mask region, and attach the centers used as graph.eta_mask_center and
        graph.phi_mask_center.

        With no half widths (full event), the centers are attached and nothing is cut.
        """
        # torch RNG, so DataLoader workers draw different centers
        eta_center = self.eta_mask_center
        phi_center = self.phi_mask_center
        if eta_center is None:
            eta_center = ETA_MIN + (ETA_MAX - ETA_MIN) * torch.rand(()).item()
        if phi_center is None:
            phi_center = PHI_MIN + (PHI_MAX - PHI_MIN) * torch.rand(()).item()

        graph.eta_mask_center = torch.tensor(eta_center)
        graph.phi_mask_center = torch.tensor(phi_center)

        if self.eta_mask_half_width is None and self.phi_mask_half_width is None:
            return

        mask = eta_phi_mask(
            eta=graph.x[:, self.eta_column],
            phi=graph.x[:, self.phi_column],
            eta_center=eta_center,
            phi_center=phi_center,
            eta_half_width=self.eta_mask_half_width,
            phi_half_width=self.phi_mask_half_width,
        )

        n_t3 = graph.x.shape[0]
        per_t3 = []
        for field, value in graph:
            if field in NON_T3_FIELDS or not torch.is_tensor(value) or value.dim() == 0:
                continue
            if value.shape[0] != n_t3:
                raise ValueError(
                    f"Graph field {field} has {value.shape[0]} rows but there are {n_t3} T3s. "
                    f"If it is not a per-T3 field, add it to NON_T3_FIELDS in dataset.py."
                )
            per_t3.append(field)
        for field in per_t3:
            graph[field] = graph[field][mask]

    def get_feature_dims(self):
        sample = self[0]
        return {
            'node_features_dim': sample.num_node_features,
            'sim_features_dim': sample.sim_features.shape[1] if hasattr(sample, 'sim_features') else None
        }


def eta_phi_mask(eta, phi, eta_center, phi_center, eta_half_width, phi_half_width):
    """
    Boolean mask selecting entries inside an eta/phi region.

    The region is |eta - eta_center| <= eta_half_width and |dphi| <= phi_half_width,
    where dphi is wrapped into [-pi, pi]. A half width of None means no cut in that
    coordinate. Boundaries are inclusive.
    """
    mask = torch.ones_like(eta, dtype=torch.bool)
    if eta_half_width is not None:
        mask &= (eta - eta_center).abs() <= eta_half_width
    if phi_half_width is not None:
        dphi = torch.atan2(torch.sin(phi - phi_center), torch.cos(phi - phi_center))
        mask &= dphi.abs() <= phi_half_width
    return mask


def validate_mask_args(eta_center, phi_center, eta_half_width, phi_half_width):
    """
    Raise ValueError unless each coordinate's (half_width, center) is a valid mask mode:
      - half_width None, center 0: no cut in this coordinate
      - half_width set, center None: random center
      - half_width set, center set: fixed center
    """
    if eta_half_width is None:
        if eta_center != 0:
            raise ValueError(f"eta_mask_center must be 0 when eta_mask_half_width is None, got {eta_center}")
    elif eta_half_width <= 0:
        raise ValueError(f"eta_mask_half_width must be positive, got {eta_half_width}")

    if phi_half_width is None:
        if phi_center != 0:
            raise ValueError(f"phi_mask_center must be 0 when phi_mask_half_width is None, got {phi_center}")
    elif phi_half_width <= 0:
        raise ValueError(f"phi_mask_half_width must be positive, got {phi_half_width}")


