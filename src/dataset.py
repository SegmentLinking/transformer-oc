import os
import torch
from torch_geometric.data import Dataset
import glob

from constants import LS_FEATURE_MIN, LS_FEATURE_MAX, T3_FEATURE_MIN, T3_FEATURE_MAX
from constants import T3_ETA_COLUMN, T3_PHI_COLUMN, ETA_MIN, ETA_MAX, PHI_MIN, PHI_MAX

# Graph fields with one row per T3, which are cut by the eta/phi mask.
# sim_features (one row per sim particle) is deliberately not cut.
PER_T3_FIELDS = ["x", "sim_index", "md_layer", "md_simIdx", "hit_layer", "hit_simIdx"]


class PCDataset(Dataset):
    def __init__(
        self,
        data_dir,
        subset=None,
        eta_mask_half_width=None,
        phi_mask_half_width=None,
        eta_mask_center=0.0,
        phi_mask_center=0.0,
    ):
        self.data_dir = data_dir
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

        self.graph_files = glob.glob(os.path.join(data_dir, "graph_*.pt"))
        self.graph_files.sort()

        if len(self.graph_files) == 0:
            raise ValueError(f"No graph files found in {data_dir}")

    def __len__(self):
        return min(len(self.graph_files), self.subset)

    def __getitem__(self, idx):
        graph_path = self.graph_files[idx]
        graph = torch.load(graph_path, map_location='cpu', weights_only=False)

        if self.eta_mask_half_width is not None or self.phi_mask_half_width is not None:
            self.apply_eta_phi_mask(graph)

        graph.x[:, 0] = torch.log(graph.x[:, 0] + 1e-6)

        if graph.x.shape[1] == len(LS_FEATURE_MIN):
            min_vals = torch.tensor(LS_FEATURE_MIN)
            max_vals = torch.tensor(LS_FEATURE_MAX)
        else:
            min_vals = torch.tensor(T3_FEATURE_MIN)
            max_vals = torch.tensor(T3_FEATURE_MAX)

        graph.x = (graph.x - min_vals) / (max_vals - min_vals + 1e-8)

        return graph

    def apply_eta_phi_mask(self, graph):
        """Cut all per-T3 fields of `graph` in place to the eta/phi mask region."""
        # torch RNG, so DataLoader workers draw different centers
        eta_center = self.eta_mask_center
        phi_center = self.phi_mask_center
        if eta_center is None:
            eta_center = ETA_MIN + (ETA_MAX - ETA_MIN) * torch.rand(()).item()
        if phi_center is None:
            phi_center = PHI_MIN + (PHI_MAX - PHI_MIN) * torch.rand(()).item()

        mask = eta_phi_mask(
            eta=graph.x[:, T3_ETA_COLUMN],
            phi=graph.x[:, T3_PHI_COLUMN],
            eta_center=eta_center,
            phi_center=phi_center,
            eta_half_width=self.eta_mask_half_width,
            phi_half_width=self.phi_mask_half_width,
        )

        for field in PER_T3_FIELDS:
            if field in graph:
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


