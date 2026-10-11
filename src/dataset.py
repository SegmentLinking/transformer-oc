import os
import torch
from torch_geometric.data import Dataset
import glob

from constants import LS_FEATURE_MIN, LS_FEATURE_MAX, T3_FEATURE_MIN, T3_FEATURE_MAX

class PCDataset(Dataset):
    def __init__(self, data_dir, subset=None):
        self.data_dir = data_dir
        self.subset = subset if subset is not None else float('inf')

        self.graph_files = glob.glob(os.path.join(data_dir, "graph_*.pt"))
        self.graph_files.sort()
        
        if len(self.graph_files) == 0:
            raise ValueError(f"No graph files found in {data_dir}")
        
    def __len__(self):
        return min(len(self.graph_files), self.subset)

    def __getitem__(self, idx):
        graph_path = self.graph_files[idx]
        graph = torch.load(graph_path, map_location='cpu', weights_only=False)
        
        graph.x[:, 0] = torch.log(graph.x[:, 0] + 1e-6)
        
        if graph.x.shape[1] == len(LS_FEATURE_MIN):
            min_vals = torch.tensor(LS_FEATURE_MIN)
            max_vals = torch.tensor(LS_FEATURE_MAX)
        else:
            min_vals = torch.tensor(T3_FEATURE_MIN)
            max_vals = torch.tensor(T3_FEATURE_MAX)

        graph.x = (graph.x - min_vals) / (max_vals - min_vals + 1e-8)
        
        return graph
    
    def get_feature_dims(self):
        sample = self[0]
        return {
            'node_features_dim': sample.num_node_features,
            'sim_features_dim': sample.sim_features.shape[1] if hasattr(sample, 'sim_features') else None
        }