"""
Evaluate a trained model: run inference and DBSCAN clustering on every event of a dataset split, match
the clusters to sim tracks with LST-style hit matching (hit_matching.py), and plot efficiency, fake rate,
duplicate rate and purity vs pT and eta, optionally against LST's T5s / track candidates of the same events.

Run from the repo root, e.g.
  apptainer exec --nv -B /ceph /ceph/users/atuna/ml-tracking.sif python3 analysis/main.py \
      --config config_pu200_mask.yaml --output output/lightning_logs/version_N/ --plot-output <dir> --t5
"""
import os
# Cap the BLAS/OpenMP threads before NumPy is imported: on many-core nodes (256 here) OpenBLAS can hang
# with its default of one thread per core. Override by setting the variables in the environment.
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "8")
import re
from collections import defaultdict
from glob import glob
import numpy as np
from argparse import ArgumentParser
import pathlib

import time

import torch
from torch_geometric.loader import DataLoader

import uproot

import matplotlib.pyplot as plt
import mplhep as hep
hep.style.use(hep.style.CMS)

from scipy.stats import beta as beta_dist

import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(_HERE, '..', 'src'))

from dataset import PCDataset
from helpers import load_config
from hit_matching import select_sims
from normalization import load_normalization
from measures import (calculate_tracking_metrics, run_inference_and_clustering, plot_cluster_size_histogram,
                      plot_layer_span_histogram, plot_layer_span_2d, t3_beta_hit_fraction_hist,
                      plot_beta_vs_hit_fraction, plot_cluster_hit_fraction)

# Graph file of entry N of LSTNtuple_<name>.root, as written by transformer-datasets
GRAPH_FILE = re.compile(r"^(?P<ntuple>.+\.root)\.entry_(?P<entry>\d+)\.pt$")
DEFAULT_LSTNTUPLE_DIR = "/ceph/users/atuna/CMSSW_20_1_0_pre2/src/ttbar_lstntuple"
LST_LABELS = {"t5": "LST T5s", "tc": "LST track candidates"}


def lstntuple_events(graph_files, lstntuple_dir):
    """(LSTNtuple path, entry) of every graph file, from its name LSTNtuple_<name>.root.entry_<N>.pt."""
    events = []
    for path in graph_files:
        m = GRAPH_FILE.match(os.path.basename(path))
        if m is None:
            raise ValueError(f"Can't tell the LSTNtuple entry of {path}: expected LSTNtuple_<name>.root.entry_<N>.pt")
        events.append((os.path.join(lstntuple_dir, m["ntuple"]), int(m["entry"])))
    return events


def load_lst_objects(events, prefix):
    """
    LST objects (prefix 't5' or 'tc') and sim tracks of the given (LSTNtuple path, entry) events, in order:
    one dict of NumPy arrays per event, with pt, eta, pMatched and simIdx of the objects and sim_features
    (same columns as the graphs' sim_features). pMatched is LST's hit-based match fraction and simIdx its
    best sim track (set only if pMatched > 0.75).
    """
    obj_branches = [f"{prefix}_{v}" for v in ("pt", "eta", "pMatched", "simIdx")]
    sim_branches = ["sim_pt", "sim_eta", "sim_phi", "sim_q", "sim_vx", "sim_vy", "sim_vz", "sim_vtxperp"]
    by_file = defaultdict(list)
    for i, (path, entry) in enumerate(events):
        by_file[path].append((entry, i))
    out = [None] * len(events)
    for path, entries in by_file.items():
        with uproot.open(path) as f:
            tree = f["tree"]
            for entry, i in entries:
                a = tree.arrays(obj_branches + sim_branches, entry_start=entry, entry_stop=entry + 1)
                ev = {v: np.asarray(a[f"{prefix}_{v}"][0]) for v in ("pt", "eta", "pMatched", "simIdx")}
                ev["sim_features"] = np.stack([np.asarray(a[b][0], dtype=np.float64) for b in sim_branches], axis=1)
                out[i] = ev
    return out


def lst_metric_counts(event, pt_bins, eta_bins, purity_threshold=0.75, eta_cut=None):
    """
    Histograms for LST objects of one event with the same definitions as calculate_tracking_metrics:
    an object is matched to its best sim track if pMatched > purity_threshold; efficiency and duplicate
    rate are over all selected sim tracks, fake rate over objects with 0.8 < pT < 100 GeV.
    """
    sim = event["sim_features"]
    n_sim = len(sim)
    matched = (event["simIdx"] >= 0) & (event["pMatched"] > purity_threshold)
    sims = event["simIdx"][matched]
    n_matched = np.bincount(sims[sims < n_sim], minlength=n_sim)
    sel = select_sims(sim)
    pt_max = pt_bins[-1]
    sim_pts, sim_etas = np.clip(sim[sel, 0], None, pt_max), sim[sel, 1]
    reco, dup = n_matched[sel] > 0, n_matched[sel] > 1
    acc = (sim_etas >= eta_cut[0]) & (sim_etas <= eta_cut[1]) if eta_cut is not None else np.ones(len(sim_etas), bool)

    in_pt = (event["pt"] > 0.8) & (event["pt"] < 100)
    obj_pts, obj_etas = np.clip(event["pt"][in_pt], None, pt_max), event["eta"][in_pt]
    fake = ~matched[in_pt]
    h = lambda values, bins: np.histogram(values, bins=bins)[0]
    return {
        "eff_num_pt": h(sim_pts[acc & reco], pt_bins), "eff_den_pt": h(sim_pts[acc], pt_bins),
        "eff_num_eta": h(sim_etas[reco], eta_bins), "eff_den_eta": h(sim_etas, eta_bins),
        "dup_num_pt": h(sim_pts[acc & dup], pt_bins), "dup_num_eta": h(sim_etas[dup], eta_bins),
        "fake_num_pt": h(obj_pts[fake], pt_bins), "fake_den_pt": h(obj_pts, pt_bins),
        "fake_num_eta": h(obj_etas[fake], eta_bins), "fake_den_eta": h(obj_etas, eta_bins),
    }


def calculate_lst_metrics(lst_events, pt_bins, eta_bins, purity_threshold=0.75, eta_cut=None):
    """Efficiency, fake rate and duplicate rate (with errors) of LST objects, summed over events (see lst_metric_counts)."""
    totals = defaultdict(float)
    for event in lst_events:
        for key, value in lst_metric_counts(event, pt_bins, eta_bins, purity_threshold, eta_cut).items():
            totals[key] = totals[key] + value
    out = {}
    for var in ("pt", "eta"):
        for name, num, den in (("efficiency", "eff_num", "eff_den"), ("duplicate_rate", "dup_num", "eff_den"),
                               ("fake_rate", "fake_num", "fake_den")):
            k, n = totals[f"{num}_{var}"], totals[f"{den}_{var}"]
            out[f"{name}_{var}"] = safe_divide(k, n)
            out[f"{name}_err_{var}"] = clopper_pearson_error(k, n)
    out["totals"] = dict(totals)
    return out


def safe_divide(num, den):
    num = np.asarray(num, dtype=float)
    return np.divide(num, den, out=np.zeros_like(num), where=np.asarray(den) > 0)


def clopper_pearson_error(k, N, cl=0.95):
    """Asymmetric Clopper-Pearson errors [low, high] of k / N (zero where N = 0)."""
    k = np.asarray(k, dtype=float)
    N = np.asarray(N, dtype=float)
    valid = N > 0
    _N = np.where(valid, N, 1)
    eff = np.where(valid, k / _N, 0)
    alpha = 1 - cl
    with np.errstate(invalid="ignore"):
        low = np.where(k > 0, beta_dist.ppf(alpha / 2, k, _N - k + 1), 0.0)
        high = np.where(k < _N, beta_dist.ppf(1 - alpha / 2, k + 1, _N - k), 1.0)
    low = np.where(valid, low, 0)
    high = np.where(valid, high, 0)
    return [eff - low, high - eff]


def resolve_checkpoint(output):
    """(checkpoint path, run dir) from --output: a checkpoint file, or a run dir (final_model.ckpt, else the last checkpoint)."""
    if output.endswith(".ckpt") or output.endswith(".pt"):
        return output, str(pathlib.Path(output).parent.parent if pathlib.Path(output).parent.name == "checkpoints"
                           else pathlib.Path(output).parent)
    ckpt = os.path.join(output, "final_model.ckpt")
    if not os.path.exists(ckpt):
        ckpt = sorted(glob(os.path.join(output, "checkpoints", "*.ckpt")))[-1]
    return ckpt, output


def find_normalization(hparams, run_dir):
    """The normalization the model was trained with: checkpoint hparams, else <run dir>/normalization.json, else None (legacy)."""
    if hparams.get("normalization"):
        print("Using the feature normalization from the checkpoint hparams")
        return hparams["normalization"]
    path = os.path.join(run_dir, "normalization.json")
    if os.path.exists(path):
        print(f"Loading feature normalization from {path}")
        return load_normalization(path)
    print("No normalization in the checkpoint or run dir: using the legacy hardcoded normalization and neighbor search")
    return None


def build_model(hparams, config, input_dim):
    """TransformerOCModel with the checkpoint's hparams, falling back to the config for older checkpoints."""
    from model import TransformerOCModel
    if "input_dim" in hparams and hparams["input_dim"] != input_dim:
        raise ValueError(f"Checkpoint was trained with {hparams['input_dim']} features, the dataset has {input_dim}")
    sources = {
        "hidden_dim": config.get("hidden_dim", config.get("d_model", 128)),
        "num_layers": config["num_layers"],
        "num_heads": config["nhead"],
        "latent_dim": config["latent_dim"],
        "dropout": config.get("dropout", 0.1),
        "dr_threshold": config.get("dr_threshold", 0.2),
        "max_neighbors": config.get("max_neighbors", 64),
        "normalize_embeddings": config.get("normalize_embeddings", True),
    }
    kwargs = {}
    for key, from_config in sources.items():
        if key in hparams:
            kwargs[key] = hparams[key]
        else:
            print(f"  {key} not in checkpoint hparams, using the config's: {from_config}")
            kwargs[key] = from_config
    print(f"Model: {kwargs}")
    return TransformerOCModel(input_dim=input_dim, **kwargs)


def plot_efficiency_fake_rate(pt_bins, eta_bins, metrics_pt, metrics_eta, output_path, epsilon):
    """Plot efficiency and fake rate."""
    pt_centers = (pt_bins[:-1] + pt_bins[1:]) / 2
    eta_centers = (eta_bins[:-1] + eta_bins[1:]) / 2

    plots = [
        ('efficiency', 'pt', pt_centers, metrics_pt['efficiency'], metrics_pt['efficiency_err'], 'Simulated track $p_T$ [GeV]', 'Efficiency', 'blue', (0, 1.1)),
        ('fake_rate', 'pt', pt_centers, metrics_pt['fake_rate'], metrics_pt['fake_rate_err'], 'Reconstructed track $p_T$ [GeV]', 'Fake Rate', 'red', None),
        ('efficiency', 'eta', eta_centers, metrics_eta['efficiency'], metrics_eta['efficiency_err'], r'Simulated track $\eta$', 'Efficiency', 'blue', (0, 1.1)),
        ('fake_rate', 'eta', eta_centers, metrics_eta['fake_rate'], metrics_eta['fake_rate_err'], r'Reconstructed track $\eta$', 'Fake Rate', 'red', None),
    ]

    for metric_name, var_name, centers, y_data, y_err, xlabel, ylabel, color, ylim in plots:
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.errorbar(centers, y_data, yerr=y_err, fmt='o', color=color, label='Transformer OC', capsize=3)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if ylim is not None:
            ax.set_ylim(ylim)

        hep.cms.label("Preliminary", data=False, rlabel=r"Phase-2 (14 TeV)", ax=ax)
        ax.text(0.95, 0.95, r"$t\bar{t}$, $\langle\mu\rangle=200$", transform=ax.transAxes, fontsize=20, verticalalignment='top', horizontalalignment='right')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{output_path}/{metric_name}_{var_name}_eps_{epsilon:.3f}.png", dpi=300, bbox_inches='tight')
        plt.close()


def plot_purity_duplicate(pt_bins, eta_bins, metrics_pt, metrics_eta, output_path, epsilon):
    """Plot purity and duplicate rate."""
    pt_centers = (pt_bins[:-1] + pt_bins[1:]) / 2
    eta_centers = (eta_bins[:-1] + eta_bins[1:]) / 2

    plots = [
        ('purity', 'pt', pt_centers, metrics_pt['purity'], metrics_pt['purity_err'], 'Simulated track $p_T$ [GeV]', 'Purity', 'green', (0.95, 1.0)),
        ('duplicate_rate', 'pt', pt_centers, metrics_pt['duplicate_rate'], metrics_pt['duplicate_rate_err'], 'Simulated track $p_T$ [GeV]', 'Duplicate Rate', 'purple', (0, 0.2)),
        ('purity', 'eta', eta_centers, metrics_eta['purity'], metrics_eta['purity_err'], r'Simulated track $\eta$', 'Purity', 'green', (0.95, 1.0)),
        ('duplicate_rate', 'eta', eta_centers, metrics_eta['duplicate_rate'], metrics_eta['duplicate_rate_err'], r'Simulated track $\eta$', 'Duplicate Rate', 'purple', (0, 0.2)),
    ]

    for metric_name, var_name, centers, y_data, y_err, xlabel, ylabel, color, ylim in plots:
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.errorbar(centers, y_data, yerr=y_err, fmt='o', color=color, label='Transformer OC', capsize=3)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if ylim is not None:
            ax.set_ylim(ylim)

        hep.cms.label("Preliminary", data=False, rlabel=r"Phase-2 (14 TeV)", ax=ax)
        ax.text(0.95, 0.95, r"$t\bar{t}$, $\langle\mu\rangle=200$", transform=ax.transAxes, fontsize=20, verticalalignment='top', horizontalalignment='right')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{output_path}/{metric_name}_{var_name}_eps_{epsilon:.3f}.png", dpi=300, bbox_inches='tight')
        plt.close()


def plot_t5_comparison(pt_bins, eta_bins, model_metrics_pt, model_metrics_eta, t5_metrics, output_path, epsilon,
                       metrics=('efficiency', 'fake_rate', 'duplicate_rate'), prefix='comparison', lst_label='LST T5s'):
    """Plot comparison of transformer and LST (T5 or track candidate) efficiency, fake rate, and duplicate rate.
    Only the metrics listed in `metrics` are plotted; files are saved as {prefix}_{metric}_{var}_eps_*.png."""
    pt_centers = (pt_bins[:-1] + pt_bins[1:]) / 2
    eta_centers = (eta_bins[:-1] + eta_bins[1:]) / 2

    layer_cut = r"($\geq$ 4 layers)"

    plots = [
        ('efficiency', 'pt', pt_centers, model_metrics_pt['efficiency'], model_metrics_pt['efficiency_err'],
         t5_metrics.get('efficiency_pt'), t5_metrics.get('efficiency_err_pt'), 'Simulated track $p_T$ [GeV]', rf'Efficiency {layer_cut}', (0, 1.19)),
        ('fake_rate', 'pt', pt_centers, model_metrics_pt['fake_rate'], model_metrics_pt['fake_rate_err'],
         t5_metrics.get('fake_rate_pt'), t5_metrics.get('fake_rate_err_pt'), 'Reconstructed track $p_T$ [GeV]', rf'Fake rate {layer_cut}', (0, 0.3)),
        ('duplicate_rate', 'pt', pt_centers, model_metrics_pt['duplicate_rate'], model_metrics_pt['duplicate_rate_err'],
         t5_metrics.get('duplicate_rate_pt'), t5_metrics.get('duplicate_rate_err_pt'), 'Simulated track $p_T$ [GeV]', rf'Duplicate rate {layer_cut}', (0, 0.2)),
        ('efficiency', 'eta', eta_centers, model_metrics_eta['efficiency'], model_metrics_eta['efficiency_err'],
         t5_metrics.get('efficiency_eta'), t5_metrics.get('efficiency_err_eta'), r'Simulated track $\eta$', rf'Efficiency {layer_cut}', (0, 1.19)),
        ('fake_rate', 'eta', eta_centers, model_metrics_eta['fake_rate'], model_metrics_eta['fake_rate_err'],
         t5_metrics.get('fake_rate_eta'), t5_metrics.get('fake_rate_err_eta'), r'Reconstructed track $\eta$', rf'Fake rate {layer_cut}', (0, 0.3)),
        ('duplicate_rate', 'eta', eta_centers, model_metrics_eta['duplicate_rate'], model_metrics_eta['duplicate_rate_err'],
         t5_metrics.get('duplicate_rate_eta'), t5_metrics.get('duplicate_rate_err_eta'), r'Simulated track $\eta$', rf'Duplicate rate {layer_cut}', (0, 0.2)),
    ]

    for metric_name, var_name, centers, m_y, m_err, t_y, t_err, xlabel, ylabel, ylim in plots:
        if metric_name not in metrics:
            continue
        if metric_name == 'duplicate_rate':
            ylim = (0, max(ylim[1], 1.15 * max(np.max(m_y), np.max(t_y))))
        fig, ax = plt.subplots(figsize=(12, 8))

        ax.errorbar(centers, m_y, yerr=m_err, fmt='o', color='blue', label='Transformer OC', capsize=3, markersize=5)
        ax.errorbar(centers, t_y, yerr=t_err, fmt='s', color='red', label=lst_label, capsize=3, markersize=5)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if ylim is not None:
            ax.set_ylim(ylim)

        hep.cms.label("Preliminary", data=False, rlabel=r"Phase-2 (14 TeV)", ax=ax)
        ax.text(0.95, 0.95, r"$t\bar{t}$, $\langle\mu\rangle=200$", transform=ax.transAxes, fontsize=20, verticalalignment='top', horizontalalignment='right')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{output_path}/{prefix}_{metric_name}_{var_name}_eps_{epsilon:.3f}.png", dpi=300, bbox_inches='tight')
        plt.close()


def main():
    argparser = ArgumentParser()
    argparser.add_argument('--config', type=str, default='./config.yaml', help='Path to the config file (YAML)')
    argparser.add_argument('--output', type=str, required=True, help='Run dir (output/lightning_logs/version_N) or checkpoint file')
    argparser.add_argument('--plot-output', type=str, default=None, help='Plot output folder (default: the run dir)')
    argparser.add_argument('--split', choices=['train', 'val', 'test'], default='test', help='Dataset split to evaluate (config <split>_data_dir)')
    argparser.add_argument('--patch', action='store_true',
                           help="Evaluate on the config's eta/phi patches (as validation: random centers fixed per event) "
                                "instead of whole events; only sim tracks inside the patch count for the efficiency")
    argparser.add_argument('--epsilon', type=float, default=None, help='DBSCAN epsilon parameter (overrides config)')
    argparser.add_argument('--n-events', type=int, default=-1, help='Number of events to process (-1 for all)')
    argparser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu', help='Torch device for inference')
    argparser.add_argument('--t5', action='store_true', help='Compare with LST T5s of the same events')
    argparser.add_argument('--tc', action='store_true', help='Compare with LST track candidates of the same events')
    argparser.add_argument('--lstntuple-dir', type=str, default=None,
                           help=f"Directory of the LSTNtuples the graphs were made from (default: config lstntuple_dir, else {DEFAULT_LSTNTUPLE_DIR})")
    args = argparser.parse_args()

    config = load_config(args.config)

    ckpt_path, run_dir = resolve_checkpoint(args.output)
    plot_output = args.plot_output or run_dir
    os.makedirs(plot_output, exist_ok=True)
    print(f"Loading model from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    hparams = dict(checkpoint.get('hyper_parameters', {}))
    normalization = find_normalization(hparams, run_dir)
    raw_eta_phi = normalization is not None

    mask_kwargs = {}
    if args.patch:
        if args.t5 or args.tc:
            raise ValueError("--patch evaluates part of each event: it can't be compared with LST's whole-event T5s/TCs")
        mask_kwargs = dict(
            eta_mask_half_width=config.get('eta_mask_half_width'),
            phi_mask_half_width=config.get('phi_mask_half_width'),
            eta_mask_center=config.get('eta_mask_center', 0.0),
            phi_mask_center=config.get('phi_mask_center', 0.0),
            center_seed=config.get('val_mask_seed', config['seed']),
        )
        print(f"Evaluating on eta/phi patches: {mask_kwargs}")

    subset = args.n_events if args.n_events > 0 else None
    data_dir = config[f'{args.split}_data_dir']
    dataset = PCDataset(data_dir, subset=subset, normalization=normalization, **mask_kwargs)
    # Without a normalization the dataset uses the legacy one; un-normalizing for the metrics needs it too
    normalization = dataset.normalization
    data_loader = DataLoader(dataset, shuffle=False, drop_last=False, num_workers=2)
    print(f"Loaded {len(dataset)} events from {data_dir}")

    device = torch.device(args.device)
    use_fp16 = device.type == 'cuda'  # Only use FP16 on CUDA
    print(f"Using device: {device}, FP16: {use_fp16}")

    model = build_model(hparams, config, input_dim=dataset[0].num_node_features)
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    if any(key.startswith('model.') for key in state_dict.keys()):
        state_dict = {key.replace('model.', '', 1): value for key, value in state_dict.items() if key.startswith('model.')}
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    del checkpoint

    min_samples = config.get('dbscan_min_samples', 2)
    purity_threshold = config.get('purity_threshold', 0.75)
    beta_threshold = config.get('beta_threshold', None)
    min_layers = config.get('min_layers', 4)
    epsilon = args.epsilon if args.epsilon is not None else config.get('dbscan_eps', 0.05)
    eta_cut = config.get('eta_cut', (-2.5, 2.5))

    pt_bins = np.linspace(config.get('pt_bins_start', 0.5), config.get('pt_bins_end', 2.0), config.get('pt_bins_count', 50))
    eta_bins = np.linspace(config.get('eta_bins_start', -4), config.get('eta_bins_end', 4), config.get('eta_bins_count', 50))
    beta_bins = np.logspace(-6, 0, 61)

    print(f"PT bins: {len(pt_bins)-1} bins from {pt_bins[0]:.2f} to {pt_bins[-1]:.2f} GeV")
    print(f"Eta bins: {len(eta_bins)-1} bins from {eta_bins[0]:.2f} to {eta_bins[-1]:.2f}")
    print(f"Processing {len(data_loader)} events with epsilon = {epsilon}")
    if beta_threshold is not None:
        print(f"Using beta threshold = {beta_threshold} (rejecting T3s with beta < {beta_threshold})")
    print(f"Clusters with >= {min_layers} layers are matched to a sim track if > {purity_threshold} of their distinct hits are on it")

    sums = defaultdict(float)
    all_cluster_sizes, all_cluster_data, all_cluster_layer_spans, all_cluster_fracs = [], [], [], []
    beta_frac_counts = None
    timings = []

    for i, data in enumerate(data_loader):
        if i % 10 == 0:
            print(f"Processing event {i+1}/{len(data_loader)}")

        start_time = time.time()
        cluster, beta = run_inference_and_clustering(data, model, device, eps=epsilon, min_samples=min_samples, use_fp16=use_fp16,
                                                     beta_threshold=beta_threshold, raw_eta_phi=raw_eta_phi)
        timings.append(time.time() - start_time)

        sim_region = None
        if args.patch:
            sim_region = (data.eta_mask_center.item(), data.phi_mask_center.item(),
                          mask_kwargs['eta_mask_half_width'], mask_kwargs['phi_mask_half_width'])
        metrics = calculate_tracking_metrics(data, cluster.labels_, pt_bins, eta_bins, normalization,
                                             purity_threshold=purity_threshold, eta_cut=eta_cut,
                                             min_layers=min_layers, sim_region=sim_region)
        for key, value in metrics.items():
            if isinstance(value, np.ndarray):
                sums[key] = sums[key] + value
        all_cluster_sizes.extend(metrics['cluster_sizes'])
        all_cluster_data.extend(metrics['cluster_data'])
        all_cluster_layer_spans.extend(metrics['cluster_layer_spans'])
        all_cluster_fracs.extend(metrics['cluster_hit_fractions'])

        counts = t3_beta_hit_fraction_hist(data, beta, beta_bins)
        beta_frac_counts = counts if beta_frac_counts is None else beta_frac_counts + counts

    print(f"Average inference and clustering time per event: {np.mean(timings):.4f} seconds")

    metrics_pt, metrics_eta = {}, {}
    for var, out in (('pt', metrics_pt), ('eta', metrics_eta)):
        eff_num, eff_den = sums[f'efficiency_numerator_{var}'], sums[f'efficiency_denominator_{var}']
        fake_num, fake_den = sums[f'fake_rate_numerator_{var}'], sums[f'total_tracks_{var}']
        dup_num = sums[f'duplicate_rate_numerator_{var}']
        purity_sum, purity_count = sums[f'purity_sum_{var}'], sums[f'purity_count_{var}']
        purity = safe_divide(purity_sum, purity_count)
        purity_err = np.where(purity_count > 0, np.sqrt(purity * (1 - purity) / np.maximum(purity_count, 1)), 0)
        out.update({
            'efficiency': safe_divide(eff_num, eff_den),
            'fake_rate': safe_divide(fake_num, fake_den),
            'purity': purity,
            'duplicate_rate': safe_divide(dup_num, eff_den),
            'efficiency_err': clopper_pearson_error(eff_num, eff_den),
            'fake_rate_err': clopper_pearson_error(fake_num, fake_den),
            'purity_err': [purity_err, purity_err],
            'duplicate_rate_err': clopper_pearson_error(dup_num, eff_den),
        })

    plot_efficiency_fake_rate(pt_bins, eta_bins, metrics_pt, metrics_eta, plot_output, epsilon)
    plot_purity_duplicate(pt_bins, eta_bins, metrics_pt, metrics_eta, plot_output, epsilon)
    plot_beta_vs_hit_fraction(beta_frac_counts, beta_bins, plot_output, epsilon)
    if all_cluster_fracs:
        plot_cluster_hit_fraction(all_cluster_fracs, purity_threshold, plot_output, epsilon)

    lst_metrics = {}
    lst_prefixes = [p for p, wanted in (('t5', args.t5), ('tc', args.tc)) if wanted]
    if lst_prefixes:
        lstntuple_dir = args.lstntuple_dir or config.get('lstntuple_dir', DEFAULT_LSTNTUPLE_DIR)
        events = lstntuple_events(dataset.graph_files[:len(dataset)], lstntuple_dir)
        if purity_threshold < 0.75:
            print("Warning: LST sets simIdx only for pMatched > 0.75, so LST objects use 0.75 for lower purity thresholds")
        for prefix in lst_prefixes:
            print(f"\nLoading {LST_LABELS[prefix]} of the same {len(events)} events from {lstntuple_dir}")
            lst_events = load_lst_objects(events, prefix)
            lst_metrics[prefix] = calculate_lst_metrics(lst_events, pt_bins, eta_bins, purity_threshold=purity_threshold, eta_cut=eta_cut)
            plot_t5_comparison(pt_bins, eta_bins, metrics_pt, metrics_eta, lst_metrics[prefix], plot_output, epsilon,
                               prefix=f'comparison_{prefix}', lst_label=LST_LABELS[prefix])

    if all_cluster_sizes:
        plot_cluster_size_histogram(all_cluster_sizes, pt_bins, eta_bins, plot_output, epsilon, cluster_data=all_cluster_data)
        print(f"\nCluster size statistics:")
        print(f"  Total clusters: {len(all_cluster_sizes)}")
        print(f"  Mean cluster size: {np.mean(all_cluster_sizes):.2f}")
        print(f"  Median cluster size: {np.median(all_cluster_sizes):.0f}")
        print(f"  Min/Max cluster size: {min(all_cluster_sizes)}/{max(all_cluster_sizes)}")

    if all_cluster_layer_spans:
        plot_layer_span_histogram(all_cluster_layer_spans, plot_output, epsilon)
        plot_layer_span_2d(all_cluster_data, all_cluster_layer_spans, pt_bins, eta_bins, plot_output, epsilon)
        print(f"\nLayer span statistics (clusters with >= {min_layers} layers):")
        print(f"  Mean layer span: {np.mean(all_cluster_layer_spans):.2f}")
        print(f"  Median layer span: {np.median(all_cluster_layer_spans):.0f}")
        print(f"  Min/Max layer span: {min(all_cluster_layer_spans)}/{max(all_cluster_layer_spans)}")

    def ratio(num, den):
        den = np.sum(sums[den]) if isinstance(den, str) else den
        return np.sum(sums[num]) / den if den > 0 else float('nan')

    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY")
    print("="*60)
    print(f"Epsilon: {epsilon}")
    print(f"Purity Threshold: {purity_threshold} (LST-style hit matching)")
    if beta_threshold is not None:
        print(f"Beta Threshold: {beta_threshold}")
    print(f"Events processed: {len(data_loader)} ({args.split}{', patches' if args.patch else ''})")
    print(f"Model: {ckpt_path}")
    print("\nTransformer OC (integrated over the pT bins):")
    print(f"  Efficiency: {ratio('efficiency_numerator_pt', 'efficiency_denominator_pt'):.3f}")
    print(f"  Fake Rate: {ratio('fake_rate_numerator_pt', 'total_tracks_pt'):.3f}")
    print(f"  Duplicate Rate: {ratio('duplicate_rate_numerator_pt', 'efficiency_denominator_pt'):.3f}")
    print(f"  Purity: {ratio('purity_sum_pt', 'purity_count_pt'):.3f}")
    for prefix, m in lst_metrics.items():
        t = m['totals']
        print(f"\n{LST_LABELS[prefix]} (integrated over the pT bins):")
        print(f"  Efficiency: {np.sum(t['eff_num_pt']) / max(np.sum(t['eff_den_pt']), 1):.3f}")
        print(f"  Fake Rate: {np.sum(t['fake_num_pt']) / max(np.sum(t['fake_den_pt']), 1):.3f}")
        print(f"  Duplicate Rate: {np.sum(t['dup_num_pt']) / max(np.sum(t['eff_den_pt']), 1):.3f}")

    print(f"\nPlots saved to: {plot_output}/")


if __name__ == "__main__":
    main()
