"""
test_dataloader.py - a short script to explain PyG dataloader to myself

You can run me like:
> apptainer exec --nv /ceph/users/atuna/ml-tracking.sif python3 test_dataloader.py
"""
import torch
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
import pytorch_lightning as pl

FNAMES = [
    "/ceph/users/atuna/work/cms_tracking_ml/transformer-oc/data/pu200_t3/train/graph_1000.pt",
    "/ceph/users/atuna/work/cms_tracking_ml/transformer-oc/data/pu200_t3/train/graph_1001.pt"
]

def main():
    dm = ParticleTrackingDataModule()
    for data in dm.train_dataloader():
        print("Start of a new data batch")
        print(data)
        # print("data.x:", data.x)
        # print("data.sim_index:", data.sim_index)
        # print("data.sim_features:", data.sim_features)
        # print("data.md_layer:", data.md_layer)
        # print("data.md_simIdx:", data.md_simIdx)
        print("data.batch:", data.batch)
        print("data.ptr:", data.ptr)
        print("End of the data batch\n")


class PCDataset(Dataset):
    def __init__(self):
        pass
        
    def __len__(self):
        return len(FNAMES)

    def __getitem__(self, idx):
        print(f"PCDataset loading {idx}")
        return torch.load(FNAMES[idx], weights_only=False)


class ParticleTrackingDataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()

    def train_dataloader(self):
        return DataLoader(PCDataset(), shuffle=False)

    def val_dataloader(self):
        return DataLoader(PCDataset(), shuffle=False)



if __name__ == "__main__":
    main()
