"""Export a trained TransformerOCModel to TorchScript, for LST (standalone libtorch, CMSSW PyTorchAlpaka).

The exported module takes the *raw* 49 T3 features (as written by data/t3_processing.py, i.e. t3_pt not
logged, no min-max scaling) and applies the same scaling as src/dataset.py itself. Methods:

    forward(x_raw[N,49])                                 -> (beta[N,1], H[N,latent])   full model
    forward_net(x_raw[N,49], idx[N,k] int64, mask[N,k] bool) -> (beta, H)              neighbours supplied by caller
    scale(x_raw[N,49])                                   -> x[N,49]                    dataset.py scaling
    neighbors(x_raw[N,49])                               -> (idx[N,k] int64, mask[N,k] bool)  ΔR neighbour search

All inputs are one event (a single graph). Run inference under torch.no_grad() / torch::NoGradGuard.

usage (after `source pyenv-cmssw/activate.sh`):
    python3 export/export_torchscript.py --checkpoint <ckpt> --hparams <hparams.yaml> --output <file.pt> \
        [--validate-dir <dir with graph_*.pt>] [--n-validate 3]
"""
import argparse
import glob
import os
import sys
from typing import Tuple

import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from model import TransformerOCModel  # noqa: E402

# Min-max scaling of the 49-feature graphs, copied from src/dataset.py (PCDataset.__getitem__).
# --validate-dir checks the exported scaling against PCDataset itself, so a copy mistake shows up there.
MIN_VALS = [-3.4627, -2.4507, -3.1416, -0.2362, -0.7272, -0.8202, -0.8768, -0.5988,
            -0.2704, -0.8894, -1.0590, -1.3935, -0.7862, -73.8012, -72.4212, -187.5750,
            -73.8019, -77.1076, -187.8705, -0.0231, -0.5393, -5.0250, -93.8255, -93.9099,
            -223.8540, -97.9344, -93.9060, -224.1495, -0.0111, -0.8231, -5.0255, -93.8255,
            -93.9099, -223.8540, -97.9344, -93.9060, -224.1495, -0.0111, -0.8231, -5.0255,
            -110.0160, -110.0150, -267.2350, -110.1943, -110.1950, -267.6350, -0.0046,
            -0.9416, -5.0259]
MAX_VALS = [7.5964, 2.4535, 3.1416, 0.2434, 0.7304, 0.7744, 0.8728, 0.5682,
            0.2755, 0.8922, 1.1500, 1.5453, 0.8093, 73.7524, 70.4150, 187.5750,
            73.7676, 71.4665, 187.8705, 0.0228, 0.5575, 5.0250, 93.8869, 93.8406,
            223.8540, 96.1440, 96.9953, 224.1495, 0.0097, 0.8098, 5.0253, 93.8869,
            93.8406, 223.8540, 96.1440, 96.9953, 224.1495, 0.0097, 0.8098, 5.0253,
            110.0128, 110.0150, 267.2350, 110.1814, 110.1950, 267.6350, 0.0050,
            0.9323, 5.0260]


class ScriptableOCModel(TransformerOCModel):
    """TransformerOCModel with the two changes TorchScript needs; the computation is unchanged.

    - build_neighbor_index without its @torch.no_grad() decorator (not scriptable); callers run under no_grad.
    - F.normalize(p=2.0) instead of p=2 (TorchScript requires a float). Keep `network` in sync with
      TransformerOCModel.forward.
    """

    build_neighbor_index = TransformerOCModel.build_neighbor_index.__wrapped__

    def network(self, x_raw: torch.Tensor, neighbor_idx: torch.Tensor, neighbor_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.input_proj(x_raw)
        for layer in self.transformer_layers:
            x = layer(x, neighbor_idx, neighbor_mask)
        coords_latent = self.latent_head(x)
        coords_latent = F.normalize(coords_latent, p=2.0, dim=-1)
        eps = 1e-6
        beta = self.beta_head(x).sigmoid()
        beta = torch.clamp(beta, eps, 1 - eps)
        return beta, coords_latent

    def forward(self, x_raw: torch.Tensor, batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        neighbor_idx, neighbor_mask = self.build_neighbor_index(x_raw, batch)
        return self.network(x_raw, neighbor_idx, neighbor_mask)


class ExportedOC(torch.nn.Module):
    """Raw-feature interface around ScriptableOCModel (one event per call)."""

    def __init__(self, model: ScriptableOCModel):
        super().__init__()
        self.model = model
        self.register_buffer("min_vals", torch.tensor(MIN_VALS))
        self.register_buffer("max_vals", torch.tensor(MAX_VALS))

    @torch.jit.export
    def scale(self, x_raw: torch.Tensor) -> torch.Tensor:
        # same operations, in the same order, as src/dataset.py
        x = x_raw.clone()
        x[:, 0] = torch.log(x[:, 0] + 1e-6)
        return (x - self.min_vals) / (self.max_vals - self.min_vals + 1e-8)

    @torch.jit.export
    def neighbors(self, x_raw: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.scale(x_raw)
        batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        return self.model.build_neighbor_index(x, batch)

    @torch.jit.export
    def forward_net(self, x_raw: torch.Tensor, neighbor_idx: torch.Tensor, neighbor_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.model.network(self.scale(x_raw), neighbor_idx, neighbor_mask)

    def forward(self, x_raw: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.scale(x_raw)
        batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        return self.model(x, batch)


def load_model(checkpoint, hparams):
    hp = yaml.safe_load(open(hparams))
    kwargs = dict(input_dim=hp["input_dim"], hidden_dim=hp["hidden_dim"], num_layers=hp["num_layers"],
                  num_heads=hp["nhead"], latent_dim=hp["latent_dim"], dropout=hp.get("dropout", 0.1),
                  dr_threshold=hp["dr_threshold"], max_neighbors=hp.get("max_neighbors", 64))
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = state.get("state_dict", state)
    state = {k.replace("model.", "", 1): v for k, v in state.items()}
    eager = TransformerOCModel(**kwargs)
    eager.load_state_dict(state, strict=True)
    scriptable = ScriptableOCModel(**kwargs)
    scriptable.load_state_dict(state, strict=True)
    return eager.eval(), scriptable.eval(), hp


def validate(scripted, eager, validate_dir, n_graphs, devices):
    """Compare the TorchScript module with the eager model fed by PCDataset (the training data path)."""
    from dataset import PCDataset
    files = sorted(glob.glob(os.path.join(validate_dir, "graph_*.pt")))[:n_graphs]
    dataset = PCDataset(validate_dir, subset=None)
    ok = True
    for path in files:
        x_raw = torch.load(path, map_location="cpu", weights_only=False).x.float()
        x_ref = dataset[dataset.graph_files.index(path)].x.float()
        for dev in devices:
            e, s = eager.to(dev), scripted.to(dev)
            xr, xs = x_raw.to(dev), x_ref.to(dev)
            with torch.no_grad():
                ref = e(xs, torch.zeros(xs.shape[0], dtype=torch.long, device=dev))
                d_scale = (s.scale(xr) - xs).abs().max().item()
                idx_ref, mask_ref = e.build_neighbor_index(xs, torch.zeros(xs.shape[0], dtype=torch.long, device=dev))
                idx, mask = s.neighbors(xr)
                same_nb = bool(torch.equal(idx, idx_ref) and torch.equal(mask, mask_ref))
                b_full, h_full = s(xr)
                b_net, h_net = s.forward_net(xr, idx_ref, mask_ref)
            d = {
                "full B": (b_full - ref["B"]).abs().max().item(), "full H": (h_full - ref["H"]).abs().max().item(),
                "net B": (b_net - ref["B"]).abs().max().item(), "net H": (h_net - ref["H"]).abs().max().item(),
            }
            # CPU must be bit-identical. On GPU, log() can differ from the CPU-scaled reference in the last bit,
            # which moves beta by up to ~1e-4 (observed 1.5e-4) and H by ~1e-6; neighbours must still be identical.
            if dev == "cpu":
                good = d_scale == 0.0 and same_nb and max(d.values()) == 0.0
            else:
                good = d_scale < 1e-6 and same_nb and max(d["full B"], d["net B"]) < 1e-3 and max(d["full H"], d["net H"]) < 1e-4
            ok &= good
            print(f"{os.path.basename(path)} N={x_raw.shape[0]:6d} {dev:4s} scale diff={d_scale:.1e} "
                  f"neighbours identical={same_nb} " + " ".join(f"{k}={v:.1e}" for k, v in d.items())
                  + ("  OK" if good else "  FAIL"))
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--hparams", required=True)
    ap.add_argument("--output", required=True, help="TorchScript file to write (.pt)")
    ap.add_argument("--validate-dir", default=None, help="directory with graph_*.pt (49-feature graphs)")
    ap.add_argument("--n-validate", type=int, default=3)
    args = ap.parse_args()

    eager, scriptable, hp = load_model(args.checkpoint, args.hparams)
    scripted = torch.jit.script(ExportedOC(scriptable).eval())
    scripted.save(args.output)
    print(f"wrote {args.output} (torch {torch.__version__}; dr_threshold={hp['dr_threshold']}, "
          f"max_neighbors={hp.get('max_neighbors', 64)}, latent_dim={hp['latent_dim']})")

    if args.validate_dir:
        loaded = torch.jit.load(args.output, map_location="cpu")
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        ok = validate(loaded, eager, args.validate_dir, args.n_validate, devices)
        print("VALIDATION", "OK" if ok else "FAILED")
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
