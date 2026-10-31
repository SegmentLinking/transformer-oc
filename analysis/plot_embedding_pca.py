"""
plot_embedding_pca.py - what the T3 embedding of a model encodes: a PCA of the
embedding vs the T3 input features and the truth (sim) features, and linear probes

Runs the model on the events in --data-dir, and collects the embedding of every T3
(up to --n-t3s T3s, so memory stays bounded) together with its raw input features.

PCA: a PCA of the embeddings finds the directions of largest variance. For each of
the first --n-pca components, and for each of the 49 T3 features in x, a 2D
histogram of the component vs the feature shows whether that direction of the
embedding tracks something physical.

Linear probes (--do-linear-probes): a linear fit predicts a quantity (1/pT, eta,
cos(phi), sin(phi)) from the full embedding. This finds the quantity even when it
is spread over many components, none of which looks like it on its own. The fit
uses the T3s of the even events, is tested on the odd events, and has a small
ridge penalty (--ridge).

Selections:
  --beta-cut: only T3s with beta > cut are used, like the T3s kept for
              clustering. --n-t3s then counts only those.
  --truthmatch: the PCA is still fit on all T3s, like the model is trained on all
              T3s. Then a second set of up to --n-t3s T3s is collected, with only
              T3s matched to a stored sim particle (0 <= sim_index <
              len(sim_features), about 1% of T3s), and projected onto the same
              components. The plots show these, also vs their sim features (the
              stored ones, plus sim_q/pt, the signed curvature, and
              cos/sin(sim_phi), which don't wrap around), and with
              --do-linear-probes, more probes predict their sim quantities.

Output, in --output-dir:
  explained_variance.pdf: fraction of embedding variance in each component
  correlation.pdf: Pearson correlation of each component with each feature. This
                   only catches linear relationships (e.g. not phi wrap-around), so
                   it is a guide to the 2D histograms, not a replacement for them.
  pca_<k>.pdf: one page per feature, with the 2D histogram of component k vs it
  linear_probes.pdf: one page per probed quantity, with the 2D histogram of its
                     true vs predicted value, for the test T3s
  linear_probes_sim.pdf: the same for the sim quantities, with --truthmatch

Axes are cut to the --quantile and 1 - --quantile quantiles, so long tails don't
squash the bulk into one bin. pT (T3 and sim) is plotted as log10(pT).

You can run me like:
> apptainer exec --nv /ceph/users/atuna/ml-tracking.sif python3 analysis/plot_embedding_pca.py \
    --model output/lightning_logs/version_118/final_model.ckpt \
    --data-dir data/pu200_t3/val --n-t3s 1000000 --n-pca 8 --output-dir pca_plots/
or, for linear probes of truth-matched T3s passing a beta cut, without the PCA pages:
> apptainer exec --nv /ceph/users/atuna/ml-tracking.sif python3 analysis/plot_embedding_pca.py \
    --model output/lightning_logs/version_118/final_model.ckpt \
    --data-dir data/pu200_t3/val --n-t3s 100000 --n-pca 0 --truthmatch \
    --beta-cut 0.005 --do-linear-probes --output-dir probe_plots/
"""

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
import numpy as np
import torch
from torch_geometric.data import Data

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(REPO_ROOT, "src"))
from dataset import PCDataset
from model import TransformerOCModel

rcParams.update(
    {
        "font.size": 16,
        "figure.figsize": (8, 8),
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "axes.grid": True,
        "axes.grid.which": "both",
        "grid.linewidth": 0.5,
        "grid.alpha": 0.1,
        "grid.color": "gray",
        "figure.subplot.left": 0.15,
        "figure.subplot.bottom": 0.09,
        "figure.subplot.right": 0.97,
        "figure.subplot.top": 0.95,
    }
)

# Names of the 49 columns of x, in the order built by data/t3_processing.py:
# the T3, then its two LSs, then the four MDs of those LSs (LS 0's outer MD and
# LS 1's inner MD are the same MD, so MD 1 and MD 2 repeat each other)
# pT is plotted as log10, since it has a long tail (and the model sees log pT too)
T3_NAMES = ["log10(t3_pt)", "t3_eta", "t3_phi"]
LS_VARS = ["dPhi", "dPhiChange", "dAlphaInner", "dAlphaOuter", "dAlphaInnerOuter"]
MD_VARS = [
    "anchor_x",
    "anchor_y",
    "anchor_z",
    "other_x",
    "other_y",
    "other_z",
    "dphi",
    "dphichange",
    "dz",
]
LS_NAMES = [f"ls{i}_{var}" for i in range(2) for var in LS_VARS]
MD_NAMES = [f"md{i}_{var}" for i in range(4) for var in MD_VARS]
FEATURE_NAMES = T3_NAMES + LS_NAMES + MD_NAMES

# Names of the 8 columns of sim_features (SIM_VARS in data/constants.py), and of
# the plotted sim features: sim_pt as log10, followed by the derived sim_q/pt,
# cos(sim_phi) and sin(sim_phi)
SIM_VARS = [
    "sim_pt",
    "sim_eta",
    "sim_phi",
    "sim_q",
    "sim_vx",
    "sim_vy",
    "sim_vz",
    "sim_vtxperp",
]
SIM_NAMES = (
    ["log10(sim_pt)"] + SIM_VARS[1:] + ["sim_q/pt", "cos(sim_phi)", "sin(sim_phi)"]
)

# Keys of the model output: the embedding coordinates, and beta (absent in models
# without a beta output)
EMBEDDING_KEY = "H"
BETA_KEY = "B"

# The Lightning module stores the network as self.model, which prefixes the keys
# of the checkpoint state dict
CHECKPOINT_PREFIX = "model."

BAR_COLOR = "#7e03a8"
LINE_COLOR = "black"
CUTOFF_COLOR = "gray"
TITLE_FONTSIZE = 14
COUNTS_LABEL = "T3s"

# Colormaps: T3 counts in the 2D histograms, and correlations in [-1, 1]
CMAP_YIELDS = "rainbow"
CMAP_CORRELATION = "RdBu_r"

# 2D histogram bins with fewer T3s than this are left empty (white). Bins hold
# integer counts, so 1 hides exactly the bins with no T3s.
CMIN = 1

# A constant feature gets an axis of this half width around its value
CONSTANT_RANGE_HALF_WIDTH = 0.5

# The explained variance plot shows at least this many components, so the drop-off
# after the plotted ones is visible
VARIANCE_PLOT_MIN_COMPONENTS = 20
# Room above a cumulative fraction of 1
VARIANCE_PLOT_YMAX = 1.05

# The correlation plot is wide, to fit one column per T3 feature, with room at the
# bottom for the rotated feature names
CORRELATION_FIGSIZE = (16, 6)
CORRELATION_SUBPLOT_LEFT = 0.08
CORRELATION_SUBPLOT_BOTTOM = 0.28
CORRELATION_XTICK_FONTSIZE = 9
CORRELATION_YTICK_FONTSIZE = 10

# The linear probe plots show R^2 as text in the top left, in axes coordinates
PROBE_TEXT_X = 0.05
PROBE_TEXT_Y = 0.95


def main() -> None:
    args = options()
    os.makedirs(args.output_dir, exist_ok=True)
    model, hparams = load_model(path=args.model, device=args.device)
    print(
        f"model={args.model} loss={hparams.get('loss', 'unknown')} "
        f"latent_dim={hparams['latent_dim']}"
    )

    # The PCA is fit on all T3s (passing the beta cut, if any)
    embeddings, features, names, events = collect(
        model=model,
        data_dir=args.data_dir,
        n_events=args.n_events,
        n_t3s=args.n_t3s,
        truthmatch=False,
        beta_cut=args.beta_cut,
        device=args.device,
        seed=args.seed,
    )
    mean, basis, explained = pca(embeddings=embeddings, n_pca=args.n_pca)
    if args.do_linear_probes:
        run_linear_probes(
            embeddings=embeddings,
            targets=t3_probe_targets(features=features),
            events=events,
            ridge=args.ridge,
            quantile=args.quantile,
            n_bins=args.n_bins,
            logz=args.logz,
            path=os.path.join(args.output_dir, "linear_probes.pdf"),
        )

    # The plots show only matched T3s, projected onto the same components
    if args.truthmatch:
        embeddings, features, names, events = collect(
            model=model,
            data_dir=args.data_dir,
            n_events=args.n_events,
            n_t3s=args.n_t3s,
            truthmatch=True,
            beta_cut=args.beta_cut,
            device=args.device,
            seed=args.seed,
        )
        if args.do_linear_probes:
            run_linear_probes(
                embeddings=embeddings,
                targets=sim_probe_targets(features=features, names=names),
                events=events,
                ridge=args.ridge,
                quantile=args.quantile,
                n_bins=args.n_bins,
                logz=args.logz,
                path=os.path.join(args.output_dir, "linear_probes_sim.pdf"),
            )

    write_pca_plots(
        components=project(embeddings=embeddings, mean=mean, basis=basis),
        explained=explained,
        features=features,
        names=names,
        quantile=args.quantile,
        n_bins=args.n_bins,
        logz=args.logz,
        output_dir=args.output_dir,
    )


def collect(
    model: TransformerOCModel,
    data_dir: str,
    n_events: int | None,
    n_t3s: int,
    truthmatch: bool,
    beta_cut: float | None,
    device: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    """
    Embeddings [N, latent_dim] and features [N, n_features] of up to n_t3s T3s,
    reading events in order until n_t3s is reached, the names of the features,
    and the event index [N] of each T3.
    The last event read is subsampled at random, so it doesn't contribute only its
    first T3s.
    """
    dataset = PCDataset(data_dir, subset=n_events)
    generator = torch.Generator().manual_seed(seed)
    names = FEATURE_NAMES + SIM_NAMES if truthmatch else list(FEATURE_NAMES)
    embeddings = []
    features = []
    events = []
    n_collected = 0
    for i_event in range(len(dataset)):
        if n_collected >= n_t3s:
            break
        coords, x, n_t3 = event_t3s(
            model=model,
            dataset=dataset,
            i_event=i_event,
            truthmatch=truthmatch,
            beta_cut=beta_cut,
            device=device,
        )
        n_candidates = len(x)
        coords, x = subsample(
            coords=coords,
            x=x,
            n_keep=n_t3s - n_collected,
            generator=generator,
        )
        embeddings.append(coords.numpy())
        features.append(x.numpy())
        events.append(np.full(len(x), i_event))
        n_collected += len(x)
        print(
            f"event={i_event} file={os.path.basename(dataset.graph_files[i_event])} "
            f"n_t3={n_t3} candidates={n_candidates} kept={len(x)} total={n_collected}"
        )
    if n_collected == 0:
        raise ValueError(
            "No T3s collected. Check --data-dir, --n-events and --beta-cut."
        )
    return (
        np.concatenate(embeddings),
        np.concatenate(features),
        names,
        np.concatenate(events),
    )


def event_t3s(
    model: TransformerOCModel,
    dataset: PCDataset,
    i_event: int,
    truthmatch: bool,
    beta_cut: float | None,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """
    Embeddings and features of the T3s of one event which pass the selection, and
    the number of T3s in the event before it.

    The features are the raw input features (with pT as log10(pT)). With
    truthmatch, only T3s matched to a stored sim particle are kept, and their sim
    features are appended. With beta_cut, only T3s with beta > beta_cut are kept.
    """
    # The model takes normalized features, but the plots show raw ones
    graph = dataset[i_event]
    raw = torch.load(
        dataset.graph_files[i_event],
        map_location="cpu",
        weights_only=False,
    )
    coords, beta = embed(model=model, graph=graph, device=device)
    x = raw.x
    if x.shape[1] != len(FEATURE_NAMES):
        raise ValueError(f"Expected {len(FEATURE_NAMES)} T3 features, got {x.shape[1]}")
    pt_column = FEATURE_NAMES.index("log10(t3_pt)")
    x[:, pt_column] = torch.log10(x[:, pt_column])
    n_t3 = len(x)

    passes = torch.ones(n_t3, dtype=torch.bool)
    if beta_cut is not None:
        if beta is None:
            raise ValueError("--beta-cut needs a model with a beta output")
        passes = beta > beta_cut
    if truthmatch:
        matched, sim = sim_per_t3(
            sim_index=raw.sim_index,
            sim_features=raw.sim_features,
        )
        coords = coords[matched]
        x = torch.cat([x[matched], sim], dim=1)
        passes = passes[matched]
    return coords[passes], x[passes], n_t3


def subsample(
    coords: torch.Tensor,
    x: torch.Tensor,
    n_keep: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """At most n_keep T3s, chosen at random if there are more."""
    if len(x) <= n_keep:
        return coords, x
    keep = torch.randperm(len(x), generator=generator)[:n_keep]
    return coords[keep], x[keep]


def load_model(
    path: str,
    device: str,
) -> tuple[TransformerOCModel, dict]:
    """Rebuild the model from the hyperparameters saved in a Lightning checkpoint."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    hparams = checkpoint["hyper_parameters"]
    model = TransformerOCModel(
        input_dim=hparams["input_dim"],
        hidden_dim=hparams["hidden_dim"],
        num_layers=hparams["num_layers"],
        num_heads=hparams["num_heads"],
        latent_dim=hparams["latent_dim"],
        dropout=hparams["dropout"],
        dr_threshold=hparams["dr_threshold"],
        max_neighbors=hparams["max_neighbors"],
        # Older checkpoints (e.g. Aashay's OC runs) predate this option, and the
        # model then always normalized
        normalize_embeddings=hparams.get("normalize_embeddings", True),
    )
    state_dict = {
        key.removeprefix(CHECKPOINT_PREFIX): value
        for key, value in checkpoint["state_dict"].items()
    }
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, hparams


@torch.no_grad()
def embed(
    model: TransformerOCModel,
    graph: Data,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Embedding coordinates [N, latent_dim] and beta [N] of each T3 (None if the model
    has no beta), on the cpu. GPU memory grows with the number of T3s (and the
    model's max_neighbors), so an event too big for the GPU is embedded on the cpu
    instead, which is slower but gives the same result.
    """
    try:
        coords, beta = embed_on(model=model, graph=graph, device=device)
        return coords.cpu(), None if beta is None else beta.cpu()
    except torch.OutOfMemoryError:
        print(f"file={graph.filename} n_t3={len(graph.x)} out of GPU memory, using cpu")
    torch.cuda.empty_cache()
    model.to("cpu")
    coords, beta = embed_on(model=model, graph=graph, device="cpu")
    model.to(device)
    return coords, beta


@torch.no_grad()
def embed_on(
    model: TransformerOCModel,
    graph: Data,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Embedding coordinates and beta of each T3, on the given device."""
    x = graph.x.to(device)
    batch = torch.zeros(len(x), dtype=torch.long, device=device)
    out = model(x, batch)
    beta = out[BETA_KEY].float().flatten() if BETA_KEY in out else None
    return out[EMBEDDING_KEY].float(), beta


def sim_per_t3(
    sim_index: torch.Tensor,
    sim_features: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Which T3s are matched to a stored sim particle [N_T3], i.e. their sim_index is a
    row of sim_features (not a fake, or a particle which isn't stored), and the sim
    features [N_matched, len(SIM_NAMES)] of those particles, with sim_pt as log10 and
    sim_q/pt, cos(sim_phi) and sin(sim_phi) appended.
    """
    matched = (sim_index >= 0) & (sim_index < len(sim_features))
    sim = sim_features[sim_index[matched].long()]
    pt = sim[:, SIM_VARS.index("sim_pt")]
    q = sim[:, SIM_VARS.index("sim_q")]
    phi = sim[:, SIM_VARS.index("sim_phi")]
    derived = torch.stack([q / pt, torch.cos(phi), torch.sin(phi)], dim=1)
    sim = torch.cat([sim, derived], dim=1)
    sim[:, SIM_VARS.index("sim_pt")] = torch.log10(pt)
    return matched, sim


def pca(
    embeddings: np.ndarray,
    n_pca: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit a PCA to the embeddings: their mean [latent_dim], the first n_pca principal
    components as columns [latent_dim, n_pca], and the fraction of variance
    explained by every component.
    """
    # float64, since the covariance sums over ~1e6 T3s
    mean = embeddings.mean(axis=0, dtype=np.float64)
    centered = embeddings.astype(np.float64) - mean
    covariance = centered.T @ centered / (len(centered) - 1)
    # eigh returns eigenvalues in ascending order
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    return mean, eigenvectors[:, :n_pca], eigenvalues / eigenvalues.sum()


def project(
    embeddings: np.ndarray,
    mean: np.ndarray,
    basis: np.ndarray,
) -> np.ndarray:
    """Projections [N, n_pca] of the embeddings onto the fitted components."""
    return ((embeddings - mean) @ basis).astype(np.float32)


def correlate(
    components: np.ndarray,
    features: np.ndarray,
) -> np.ndarray:
    """Pearson correlation [n_pca, n_features] of each component with each feature."""
    n_pca = components.shape[1]
    full = np.corrcoef(components, features, rowvar=False)
    # A constant feature has no correlation with anything
    return np.nan_to_num(full[:n_pca, n_pca:])


def quantile_edges(
    values: np.ndarray,
    quantile: float,
    n_bins: int,
) -> np.ndarray:
    """n_bins + 1 bin edges spanning the [quantile, 1 - quantile] range of values."""
    low, high = np.quantile(values, [quantile, 1 - quantile])
    if low == high:
        low, high = low - CONSTANT_RANGE_HALF_WIDTH, high + CONSTANT_RANGE_HALF_WIDTH
    return np.linspace(low, high, n_bins + 1)


def write_pca_plots(
    components: np.ndarray,
    explained: np.ndarray,
    features: np.ndarray,
    names: list[str],
    quantile: float,
    n_bins: int,
    logz: bool,
    output_dir: str,
) -> None:
    """The explained variance, the correlations, and the pca_<k>.pdf pages."""
    n_pca = components.shape[1]
    for k, fraction in enumerate(explained[:n_pca]):
        print(
            f"pca={k} explained_variance={fraction:.4f} "
            f"cumulative={explained[: k + 1].sum():.4f}"
        )
    fig = plot_explained_variance(explained=explained, n_pca=n_pca)
    save(fig=fig, path=os.path.join(output_dir, "explained_variance.pdf"))
    if n_pca == 0:
        return

    correlation = correlate(components=components, features=features)
    fig = plot_correlation(correlation=correlation, names=names)
    save(fig=fig, path=os.path.join(output_dir, "correlation.pdf"))

    # Feature axis ranges and bin edges are shared by every component
    feature_edges = [
        quantile_edges(values=features[:, j], quantile=quantile, n_bins=n_bins)
        for j in range(features.shape[1])
    ]
    for k in range(n_pca):
        component_edges = quantile_edges(
            values=components[:, k],
            quantile=quantile,
            n_bins=n_bins,
        )
        path = os.path.join(output_dir, f"pca_{k}.pdf")
        with PdfPages(path) as pdf:
            for j, name in enumerate(names):
                fig = plot_component_vs_feature(
                    feature=features[:, j],
                    component=components[:, k],
                    feature_edges=feature_edges[j],
                    component_edges=component_edges,
                    feature_name=name,
                    component_index=k,
                    explained=explained[k],
                    correlation=correlation[k, j],
                    logz=logz,
                )
                pdf.savefig(fig)
                plt.close(fig)
        print(f"Wrote {path}")


def draw_hist2d(
    fig: Figure,
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    logz: bool,
) -> None:
    """2D histogram of T3 counts, with empty bins left white, and its colorbar."""
    _, _, _, image = ax.hist2d(
        x,
        y,
        bins=[x_edges, y_edges],
        norm=LogNorm() if logz else None,
        cmin=CMIN,
        cmap=CMAP_YIELDS,
    )
    fig.colorbar(image, ax=ax, label=COUNTS_LABEL)


def plot_component_vs_feature(
    feature: np.ndarray,
    component: np.ndarray,
    feature_edges: np.ndarray,
    component_edges: np.ndarray,
    feature_name: str,
    component_index: int,
    explained: float,
    correlation: float,
    logz: bool,
) -> Figure:
    fig, ax = plt.subplots()
    draw_hist2d(
        fig=fig,
        ax=ax,
        x=feature,
        y=component,
        x_edges=feature_edges,
        y_edges=component_edges,
        logz=logz,
    )
    ax.set_xlabel(feature_name)
    ax.set_ylabel(f"PCA component {component_index}")
    ax.set_title(
        f"PCA {component_index} ({100 * explained:.1f}% of variance), "
        f"correlation = {correlation:+.2f}",
        fontsize=TITLE_FONTSIZE,
    )
    return fig


def run_linear_probes(
    embeddings: np.ndarray,
    targets: dict[str, np.ndarray],
    events: np.ndarray,
    ridge: float,
    quantile: float,
    n_bins: int,
    logz: bool,
    path: str,
) -> None:
    """
    Fit a linear probe of each target (name: values [N]) on the T3s of the even
    events, and plot its true vs predicted value for the T3s of the odd events,
    one page per target.
    """
    train = events % 2 == 0
    if train.all() or not train.any():
        raise ValueError(
            "--do-linear-probes needs at least two events, to train on one and test "
            "on the other. Raise --n-t3s or --n-events."
        )
    with PdfPages(path) as pdf:
        for name, target in targets.items():
            prediction = linear_probe(
                embeddings=embeddings,
                target=target,
                train=train,
                ridge=ridge,
            )
            r2_train = r_squared(target=target[train], prediction=prediction[train])
            r2_test = r_squared(
                target=target[~train],
                prediction=prediction[~train],
            )
            print(
                f"probe={name} n_train={train.sum()} n_test={(~train).sum()} "
                f"r2_train={r2_train:.4f} r2_test={r2_test:.4f}"
            )
            fig = plot_probe(
                target=target[~train],
                prediction=prediction[~train],
                edges=quantile_edges(
                    values=target[~train],
                    quantile=quantile,
                    n_bins=n_bins,
                ),
                name=name,
                r2_train=r2_train,
                r2_test=r2_test,
                logz=logz,
            )
            pdf.savefig(fig)
            plt.close(fig)
    print(f"Wrote {path}")


def t3_probe_targets(
    features: np.ndarray,
) -> dict[str, np.ndarray]:
    """The T3 quantities predicted by the linear probes, by name."""
    log_pt = features[:, FEATURE_NAMES.index("log10(t3_pt)")]
    eta = features[:, FEATURE_NAMES.index("t3_eta")]
    phi = features[:, FEATURE_NAMES.index("t3_phi")]
    return {
        "1/t3_pt": 10**-log_pt,
        "t3_eta": eta,
        "cos(t3_phi)": np.cos(phi),
        "sin(t3_phi)": np.sin(phi),
    }


def sim_probe_targets(
    features: np.ndarray,
    names: list[str],
) -> dict[str, np.ndarray]:
    """
    The sim quantities predicted by the linear probes, by name, from the sim
    columns of truth-matched T3s.
    """
    pt = 10 ** features[:, names.index("log10(sim_pt)")]
    eta = features[:, names.index("sim_eta")]
    phi = features[:, names.index("sim_phi")]
    return {
        "sim_eta": eta,
        "sim_phi": phi,
        "cos(sim_phi)": np.cos(phi),
        "sin(sim_phi)": np.sin(phi),
        "sim_pt": pt,
        "1/sim_pt": 1 / pt,
        "sim_q/pt": features[:, names.index("sim_q/pt")],
    }


def linear_probe(
    embeddings: np.ndarray,
    target: np.ndarray,
    train: np.ndarray,
    ridge: float,
) -> np.ndarray:
    """
    Least-squares fit of target = w . embedding + b on the train T3s, with a ridge
    penalty ridge * |w|^2, and its prediction for every T3. The penalty is added
    as extra rows sqrt(ridge) * I of the design matrix, which leave the offset b
    unpenalized, so lstsq solves both the plain (ridge = 0) and the ridge fit.

    A small ridge matters: the embedding has some nearly empty directions, and
    without a penalty the fit puts huge weights on them, so the few test T3s
    slightly off them get wild predictions (seen with ~1k matched T3s).
    """
    design = np.hstack([embeddings, np.ones((len(embeddings), 1))]).astype(np.float64)
    n_weights = embeddings.shape[1]
    penalty = np.sqrt(ridge) * np.eye(n_weights, n_weights + 1)
    weights, _, _, _ = np.linalg.lstsq(
        np.vstack([design[train], penalty]),
        np.concatenate([target[train], np.zeros(n_weights)]),
        rcond=None,
    )
    return design @ weights


def r_squared(
    target: np.ndarray,
    prediction: np.ndarray,
) -> float:
    """Fraction of the variance of target explained by prediction."""
    residual = np.sum((target - prediction) ** 2)
    total = np.sum((target - target.mean()) ** 2)
    return 1 - residual / total


def plot_probe(
    target: np.ndarray,
    prediction: np.ndarray,
    edges: np.ndarray,
    name: str,
    r2_train: float,
    r2_test: float,
    logz: bool,
) -> Figure:
    fig, ax = plt.subplots()
    draw_hist2d(
        fig=fig,
        ax=ax,
        x=target,
        y=prediction,
        x_edges=edges,
        y_edges=edges,
        logz=logz,
    )
    # The line of perfect prediction
    ax.plot(edges[[0, -1]], edges[[0, -1]], color=LINE_COLOR, linestyle="--")
    ax.text(
        PROBE_TEXT_X,
        PROBE_TEXT_Y,
        f"R$^2$ test = {r2_test:.3f}\nR$^2$ train = {r2_train:.3f}",
        transform=ax.transAxes,
        verticalalignment="top",
    )
    ax.set_xlabel(f"true {name}")
    ax.set_ylabel(f"predicted {name}")
    ax.set_title(f"Linear probe of {name}, test events", fontsize=TITLE_FONTSIZE)
    return fig


def plot_explained_variance(
    explained: np.ndarray,
    n_pca: int,
) -> Figure:
    fig, ax = plt.subplots()
    n_shown = min(len(explained), max(n_pca, VARIANCE_PLOT_MIN_COMPONENTS))
    ax.bar(
        np.arange(n_shown),
        explained[:n_shown],
        color=BAR_COLOR,
    )
    ax.plot(
        np.arange(n_shown),
        np.cumsum(explained[:n_shown]),
        color=LINE_COLOR,
        marker="o",
        label="Cumulative",
    )
    # Between the last plotted component and the first unplotted one
    ax.axvline(
        n_pca - 0.5, color=CUTOFF_COLOR, linestyle="--", label=f"--n-pca = {n_pca}"
    )
    ax.set_xlabel("PCA component")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylabel("Fraction of embedding variance")
    ax.set_ylim(0, VARIANCE_PLOT_YMAX)
    ax.legend(loc="center right")
    return fig


def plot_correlation(
    correlation: np.ndarray,
    names: list[str],
) -> Figure:
    n_pca, n_features = correlation.shape
    fig, ax = plt.subplots(figsize=CORRELATION_FIGSIZE)
    fig.subplots_adjust(
        left=CORRELATION_SUBPLOT_LEFT,
        bottom=CORRELATION_SUBPLOT_BOTTOM,
    )
    # One cell per (component, feature), centered on integer ticks
    image = ax.pcolormesh(
        np.arange(n_features + 1) - 0.5,
        np.arange(n_pca + 1) - 0.5,
        correlation,
        cmap=CMAP_CORRELATION,
        vmin=-1,
        vmax=1,
    )
    fig.colorbar(image, ax=ax, label="Pearson correlation")
    ax.set_xticks(np.arange(n_features))
    ax.set_xticklabels(names, rotation=90, fontsize=CORRELATION_XTICK_FONTSIZE)
    ax.set_yticks(np.arange(n_pca))
    ax.set_yticklabels(
        [f"PCA {k}" for k in range(n_pca)],
        fontsize=CORRELATION_YTICK_FONTSIZE,
    )
    # PCA 0 at the top
    ax.invert_yaxis()
    ax.minorticks_off()
    ax.grid(False)
    return fig


def save(
    fig: Figure,
    path: str,
) -> None:
    fig.savefig(path)
    plt.close(fig)
    print(f"Wrote {path}")


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PCA and linear probes of the T3 embedding, vs T3 and sim features",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Lightning checkpoint (.ckpt)",
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Graphs dir",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for the output PDFs",
    )
    parser.add_argument(
        "--n-events",
        type=int,
        default=None,
        help="Max events to read (default: all)",
    )
    parser.add_argument(
        "--n-t3s",
        type=int,
        default=1_000_000,
        help="Max T3s to collect, for the PCA fit and for the plots",
    )
    parser.add_argument(
        "--n-t3s-limit",
        type=int,
        default=1_000_000,
        help="Refuse to run with --n-t3s above this, to protect memory",
    )
    parser.add_argument(
        "--n-pca",
        type=int,
        default=8,
        help="PCA components to plot (0: only the explained variance)",
    )
    parser.add_argument(
        "--n-bins",
        type=int,
        default=100,
        help="Bins per axis of the 2D histograms",
    )
    parser.add_argument(
        "--quantile",
        type=float,
        default=1e-4,
        help="Cut each axis to [q, 1 - q] quantiles",
    )
    parser.add_argument(
        "--truthmatch",
        action="store_true",
        help="Plot only T3s matched to a stored sim particle, also vs sim features",
    )
    parser.add_argument(
        "--beta-cut",
        type=float,
        default=None,
        help="Use only T3s with beta > cut (default: no cut)",
    )
    parser.add_argument(
        "--do-linear-probes",
        action="store_true",
        help="Fit linear probes of T3 (and with --truthmatch, sim) quantities",
    )
    parser.add_argument(
        "--ridge",
        type=float,
        default=1e-6,
        help="Ridge penalty of the linear probes",
    )
    parser.add_argument(
        "--logz",
        action="store_true",
        help="Log scale for the 2D histogram counts",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for subsampling the last event",
    )
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    parser.add_argument(
        "--device",
        default=default_device,
        help="cuda or cpu",
    )
    args = parser.parse_args()
    if args.ridge < 0:
        parser.error(f"--ridge must be non-negative, got {args.ridge}")
    if args.n_pca < 0:
        parser.error(f"--n-pca must be non-negative, got {args.n_pca}")
    if args.n_t3s <= 0:
        parser.error(f"--n-t3s must be positive, got {args.n_t3s}")
    if args.n_t3s > args.n_t3s_limit:
        parser.error(
            f"--n-t3s {args.n_t3s} is above --n-t3s-limit {args.n_t3s_limit}. "
            "The PCA fit holds up to --n-t3s T3s in memory, with a few float64 "
            "copies of their embeddings: a few GB per million T3s. "
            "If that fits, raise --n-t3s-limit."
        )
    return args


if __name__ == "__main__":
    main()
