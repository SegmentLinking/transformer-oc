"""Write test inputs and reference outputs for export/cpp/run_torchscript.cc.

Takes the raw features of one graph (optionally only the first --n rows), runs the exported TorchScript module
on CPU, and writes little-endian binary files:
    <prefix>_x.bin    int64 N, then float32 x_raw[N,49]
    <prefix>_ref.bin  int64 N, int64 latent_dim, then float32 beta[N], float32 H[N,latent_dim]

usage: python3 export/make_cpp_testdata.py --model <file.pt> --graph <graph_X.pt> --out-prefix <prefix> [--n 20000]
"""
import argparse
import ctypes
import struct

import torch


def raw_bytes(t):
    t = t.contiguous().float().cpu()
    return ctypes.string_at(t.data_ptr(), t.numel() * 4)  # CMSSW torch has no NumPy support


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--graph", required=True)
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--n", type=int, default=0, help="use only the first N T3s (0 = all)")
    args = ap.parse_args()

    x = torch.load(args.graph, map_location="cpu", weights_only=False).x.float()
    if args.n > 0:
        x = x[: args.n]
    module = torch.jit.load(args.model, map_location="cpu")
    with torch.no_grad():
        beta, H = module(x)
    n, latent = H.shape
    with open(f"{args.out_prefix}_x.bin", "wb") as f:
        f.write(struct.pack("<q", n))
        f.write(raw_bytes(x))
    with open(f"{args.out_prefix}_ref.bin", "wb") as f:
        f.write(struct.pack("<qq", n, latent))
        f.write(raw_bytes(beta.reshape(-1)))
        f.write(raw_bytes(H))
    print(f"wrote {args.out_prefix}_x.bin and _ref.bin: N={n}, latent_dim={latent}")


if __name__ == "__main__":
    main()
