"""
Per-feature min/max normalization of the node features x, computed from the training data.

A normalization is a dict {"names": [...], "min": [...], "max": [...], "log": [...]}, one entry
per column of x in names/min/max. Columns listed in "log" are log-transformed before the min/max
are taken and before normalizing. Keeping the names alongside the numbers means a normalization
can be checked against a dataset instead of trusting that the column count happens to match.
"""
import argparse
import json
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

LOG_EPSILON = 1e-6

# Thread-count variables of the numerical libraries, set to 1 in normalization workers
SINGLE_THREAD_ENV = ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]


def log_transform(x: torch.Tensor, names: list[str], log_features: list[str]) -> torch.Tensor:
    """Copy of x with the columns named in log_features replaced by log(x + LOG_EPSILON)."""
    x = x.clone()
    for name in log_features:
        col = names.index(name)
        x[:, col] = torch.log(x[:, col] + LOG_EPSILON)
    return x


def unnormalize(x: torch.Tensor, normalization: dict) -> torch.Tensor:
    """Inverse of PCDataset's normalization: raw features from normalized x (log columns exponentiated back)."""
    names = normalization["names"]
    feature_min = torch.tensor(normalization["min"], dtype=x.dtype, device=x.device)
    feature_max = torch.tensor(normalization["max"], dtype=x.dtype, device=x.device)
    x = x * (feature_max - feature_min + 1e-8) + feature_min
    for name in normalization["log"]:
        col = names.index(name)
        x[:, col] = torch.exp(x[:, col]) - LOG_EPSILON
    return x


def compute_normalization(
    files: list[str],
    names: list[str],
    log_features: list[str],
    n_files: int | None,
    n_workers: int = 1,
) -> dict:
    """
    Per-column min/max of x (after the log transform) over n_files of the given graph files.

    The files are picked evenly spaced over the list (all of them when n_files is None or at
    least len(files)), so that a sorted list of many LSTNtuples is sampled across files rather
    than from the first few. The choice depends only on the list and n_files.
    n_workers > 1 reads the files in that many processes; the result does not depend on it.
    """
    if not files:
        raise ValueError("No graph files to compute the feature normalization from")
    if n_files is None or n_files >= len(files):
        chosen = list(files)
    else:
        chosen = [files[i] for i in np.linspace(0, len(files) - 1, n_files).round().astype(int)]

    args = [(path, names, log_features) for path in chosen]
    if n_workers > 1:
        # spawn, not fork: forking after torch has started its thread pool can deadlock.
        # Workers must be single-threaded from the moment they import numpy/torch: otherwise each
        # starts OpenBLAS/OpenMP pools of one thread per core, which hits per-user thread limits
        # (2048 on the UCSD nodes). Spawned workers inherit os.environ at start, so set it around the pool.
        saved = {var: os.environ.get(var) for var in SINGLE_THREAD_ENV}
        os.environ.update({var: "1" for var in SINGLE_THREAD_ENV})
        try:
            with ProcessPoolExecutor(
                max_workers=n_workers,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=torch.set_num_threads,
                initargs=(1,),
            ) as pool:
                per_file = list(pool.map(_file_min_max, args, chunksize=8))
        finally:
            for var, value in saved.items():
                if value is None:
                    os.environ.pop(var, None)
                else:
                    os.environ[var] = value
    else:
        per_file = [_file_min_max(a) for a in args]

    lo = torch.stack([file_lo for file_lo, _ in per_file]).min(dim=0).values
    hi = torch.stack([file_hi for _, file_hi in per_file]).max(dim=0).values

    return {
        "names": list(names),
        "min": lo.tolist(),
        "max": hi.tolist(),
        "log": list(log_features),
        "files": [str(p) for p in chosen],
    }


def _file_min_max(args: tuple[str, list[str], list[str]]) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-column min and max of x (after the log transform) of one graph file."""
    path, names, log_features = args
    x = torch.load(path, map_location="cpu", weights_only=False).x
    check_feature_count(x, names, path)
    x = log_transform(x, names, log_features)
    return x.min(dim=0).values, x.max(dim=0).values


def check_feature_count(x: torch.Tensor, names: list[str], source: str) -> None:
    if x.shape[1] != len(names):
        raise ValueError(
            f"{source}: x has {x.shape[1]} columns but {len(names)} feature names are defined "
            f"(FEATURE_NAMES in src/constants.py). Update them to match the dataset's column layout."
        )


def check_normalization(normalization: dict, names: list[str]) -> None:
    """Raise ValueError unless the normalization is for exactly these feature names."""
    if normalization["names"] != list(names):
        raise ValueError(
            f"Normalization is for features {normalization['names']}, but the dataset has {list(names)}"
        )
    for key in ("min", "max"):
        if len(normalization[key]) != len(names):
            raise ValueError(f"Normalization has {len(normalization[key])} '{key}' values for {len(names)} features")


def save_normalization(normalization: dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(normalization, f, indent=2)


def load_normalization(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute the feature normalization of a training set once, for normalization_file in a config",
    )
    parser.add_argument("--data-dir", required=True, help="Directory of training graph files (.pt)")
    parser.add_argument("--output", required=True, help="Path of the normalization JSON to write")
    parser.add_argument("--n-files", type=int, default=None, help="Number of files, evenly spaced (default: all)")
    parser.add_argument("--n-workers", type=int, default=8, help="Processes reading files in parallel")
    args = parser.parse_args()

    # Flat imports, as elsewhere in src/ (run as python3 src/normalization.py)
    from constants import FEATURE_NAMES, LOG_FEATURES
    from dataset import list_graph_files

    files = list_graph_files(args.data_dir)
    n_files = len(files) if args.n_files is None else min(args.n_files, len(files))
    print(f"[info] computing normalization from {n_files} of {len(files)} files in {args.data_dir}", flush=True)
    normalization = compute_normalization(
        files,
        names=FEATURE_NAMES,
        log_features=LOG_FEATURES,
        n_files=args.n_files,
        n_workers=args.n_workers,
    )
    # Write to a temporary name and rename, so an interrupted run doesn't leave a file that looks finished
    tmp = args.output + ".tmp"
    save_normalization(normalization, tmp)
    os.replace(tmp, args.output)
    print(f"[done] output={args.output} n_files={len(normalization['files'])}")


if __name__ == "__main__":
    main()
