"""
MDTransformerNet: a tracking algorithm from outer-tracker MDs to track candidates.

From repo root, train and choose the working point like:
apptainer exec --nv ml-tracking.sif python3 -m mdml.md_transformer --config config.yaml
"""
import argparse
import torch.nn as nn
import pytorch_lightning as pl


def main() -> None:
    args = parse_args()
    raise NotImplementedError()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, help="Path to config yaml")
    return parser.parse_args()


class MDTransformerNet:

    def __init__(self):
        raise NotImplementedError()


class MDTransformerLightningModule(pl.LightningModule):

    def __init__(self):
        raise NotImplementedError()


class MDTransformerModule(nn.Module):

    def __init__(self):
        raise NotImplementedError()


if __name__ == "__main__":
    main()
