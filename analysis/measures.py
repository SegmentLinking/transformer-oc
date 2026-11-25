import os
import sys

# Flat imports from analysis/ and src/, wherever this module is imported from
_HERE = os.path.dirname(os.path.abspath(__file__))
for _path in (_HERE, os.path.join(_HERE, "..", "src")):
    if _path not in sys.path:
        sys.path.append(_path)

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import torch
from sklearn.cluster import DBSCAN

hep.style.use(hep.style.CMS)

from profiler import InferenceProfiler, print_gpu_info, profile_memory
from constants import N_T3_MDS
from dataset import eta_phi_mask
from hit_matching import match_clusters, select_sims
from normalization import unnormalize
from utils.hit_targets import hit_candidates, hit_targets


def to_numpy(t):
    """Tensor -> NumPy copy. Uses DLPack because CMSSW's torch is built without NumPy support,
    so tensor.numpy() fails there."""
    return np.from_dlpack(t.detach().cpu().contiguous()).copy()


@torch.no_grad()
def run_inference_and_clustering(
    data,
    model,
    device,
    eps=0.2,
    min_samples=2,
    use_fp16=False,
    beta_threshold=None,
    profiler=None,
    enable_profiling=False,
    raw_eta_phi=False,
):
    # raw_eta_phi: pass the dataset's raw eta/phi to the model's neighbor search, as training does
    # since the normalization moved to normalization.json. False keeps the legacy neighbor search
    # that older checkpoints were trained with.
    # Create profiler if requested and not provided
    if profiler is None and enable_profiling:
        profiler = InferenceProfiler(device=device if device != "cpu" else "cuda:0")

    # Data transfer
    if profiler:
        with profiler.profile_section("data_transfer"):
            x = data.x.to(device)
            batch = data.batch.to(device)
    else:
        x = data.x.to(device)
        batch = data.batch.to(device)
    coords = {"eta": data.eta.to(device), "phi": data.phi.to(device)} if raw_eta_phi else {}

    model.eval()

    # Model inference
    if profiler:
        with profiler.profile_section("model_forward"):
            if use_fp16 and device != "cpu":
                with torch.amp.autocast("cuda"):
                    out = model(x, batch, **coords)
                X = out["H"].float()
                beta = out["B"].float()
            else:
                out = model(x, batch, **coords)
                X = out["H"]
                beta = out["B"]
    else:
        if use_fp16 and device != "cpu":
            with torch.amp.autocast("cuda"):
                out = model(x, batch, **coords)
            X = out["H"].float()
            beta = out["B"].float()
        else:
            out = model(x, batch, **coords)
            X = out["H"]
            beta = out["B"]

    # Transfer to CPU
    if profiler:
        with profiler.profile_section("gpu_to_cpu_transfer"):
            X_cpu = to_numpy(X)
            beta_cpu = to_numpy(beta)
    else:
        X_cpu = to_numpy(X)
        beta_cpu = to_numpy(beta)

    # Clustering
    if profiler:
        with profiler.profile_section("clustering"):
            if beta_threshold is not None:
                beta_mask = beta_cpu.flatten() > beta_threshold
                X_filtered = X_cpu[beta_mask]

                cluster = DBSCAN(eps=eps, min_samples=min_samples).fit(X_filtered)

                full_labels = np.full(len(X_cpu), -1, dtype=int)
                full_labels[beta_mask] = cluster.labels_

                class ClusterResult:
                    def __init__(self, labels):
                        self.labels_ = labels

                cluster_result = ClusterResult(full_labels)
            else:
                cluster_result = DBSCAN(eps=eps, min_samples=min_samples).fit(X_cpu)
    else:
        if beta_threshold is not None:
            beta_mask = beta_cpu.flatten() > beta_threshold
            X_filtered = X_cpu[beta_mask]

            cluster = DBSCAN(eps=eps, min_samples=min_samples).fit(X_filtered)

            full_labels = np.full(len(X_cpu), -1, dtype=int)
            full_labels[beta_mask] = cluster.labels_

            class ClusterResult:
                def __init__(self, labels):
                    self.labels_ = labels

            cluster_result = ClusterResult(full_labels)
        else:
            cluster_result = DBSCAN(eps=eps, min_samples=min_samples).fit(X_cpu)

    if profiler:
        return cluster_result, beta_cpu, profiler
    else:
        return cluster_result, beta_cpu


def calculate_tracking_metrics(
    data,
    cluster_labels,
    pt_bins,
    eta_bins,
    normalization,
    purity_threshold=0.75,
    eta_cut=None,
    min_layers=4,
    sim_region=None,
):
    """
    Efficiency, fake rate, duplicate rate and purity of the clusters of one event, with LST-style hit
    matching (hit_matching.match_clusters): a cluster is matched to its best sim track if more than
    purity_threshold of its distinct hits have that sim track. Clusters with fewer than min_layers
    distinct hit layers are dropped.

    Efficiency and duplicate rate are over all selected sim tracks of the event (hit_matching.select_sims),
    whether or not any T3 matches them. sim_region: None, or (eta_center, phi_center, eta_half_width,
    phi_half_width) to only count sim tracks with (eta, phi) in that patch (evaluation on eta/phi patches).

    normalization: the dataset's (PCDataset.normalization), to recover raw T3 pT, eta and MD positions.
    Reco track pT/eta are those of the cluster's T3 with the innermost MD.
    """
    labels = np.asarray(cluster_labels)
    names = normalization["names"]
    x_raw = to_numpy(unnormalize(data.x.float(), normalization))
    t3_pt = x_raw[:, names.index("t3_pt")]
    t3_eta = x_raw[:, names.index("t3_eta")]
    md_r = np.stack([np.hypot(x_raw[:, names.index(f"md{i}_anchor_x")], x_raw[:, names.index(f"md{i}_anchor_y")])
                     for i in range(N_T3_MDS)], axis=1)
    t3_r = md_r.min(axis=1)

    m = match_clusters(labels, to_numpy(data.hit_idx), to_numpy(hit_candidates(data)), to_numpy(data.hit_layer))

    # innermost T3 of each cluster, in the order of m["label"] (sorted labels)
    in_cluster = np.where(labels >= 0)[0]
    order = in_cluster[np.lexsort((t3_r[in_cluster], labels[in_cluster]))]
    inner = order[np.r_[True, labels[order][1:] != labels[order][:-1]]] if len(order) else order
    reco_pt = t3_pt[inner].astype(float)
    reco_eta = t3_eta[inner].astype(float)

    keep = m["n_layers"] >= min_layers
    matched = keep & (m["sim"] >= 0) & (m["frac"] > purity_threshold)
    fake = keep & ~matched
    in_pt = (reco_pt > 0.8) & (reco_pt < 100)

    pt_max = pt_bins[-1]
    real_pts, real_etas, real_purities = np.clip(reco_pt[matched & in_pt], None, pt_max), reco_eta[matched & in_pt], m["frac"][matched & in_pt]
    fake_pts, fake_etas = np.clip(reco_pt[fake & in_pt], None, pt_max), reco_eta[fake & in_pt]

    # Denominator: all selected sim tracks of the (signal) event
    sim_features = to_numpy(data.sim_features)
    n_sim = len(sim_features)
    sim_sel = select_sims(sim_features)
    if sim_region is not None:
        eta_c, phi_c, eta_hw, phi_hw = sim_region
        sim_sel &= to_numpy(eta_phi_mask(torch.from_numpy(sim_features[:, 1]), torch.from_numpy(sim_features[:, 2]),
                                         eta_c, phi_c, eta_hw, phi_hw))
    matched_sims = m["sim"][matched]
    n_matched = np.bincount(matched_sims[matched_sims < n_sim], minlength=n_sim)
    sim_pts = np.clip(sim_features[sim_sel, 0], None, pt_max)
    sim_etas = sim_features[sim_sel, 1]
    sim_reco = n_matched[sim_sel] > 0
    sim_dup = n_matched[sim_sel] > 1
    if eta_cut is not None:
        in_acc = (sim_etas >= eta_cut[0]) & (sim_etas <= eta_cut[1])
    else:
        in_acc = np.ones(len(sim_etas), dtype=bool)

    hist = lambda values, bins, weights=None: np.histogram(values, bins=bins, weights=weights)[0]
    hist_real_pt, hist_real_eta = hist(real_pts, pt_bins), hist(real_etas, eta_bins)
    hist_fake_pt, hist_fake_eta = hist(fake_pts, pt_bins), hist(fake_etas, eta_bins)

    return {
        "efficiency_numerator_pt": hist(sim_pts[in_acc & sim_reco], pt_bins),
        "efficiency_numerator_eta": hist(sim_etas[sim_reco], eta_bins),
        "efficiency_denominator_pt": hist(sim_pts[in_acc], pt_bins),
        "efficiency_denominator_eta": hist(sim_etas, eta_bins),
        "fake_rate_numerator_pt": hist_fake_pt,
        "fake_rate_numerator_eta": hist_fake_eta,
        "total_tracks_pt": hist_real_pt + hist_fake_pt,
        "total_tracks_eta": hist_real_eta + hist_fake_eta,
        "purity_sum_pt": hist(real_pts, pt_bins, real_purities),
        "purity_sum_eta": hist(real_etas, eta_bins, real_purities),
        "purity_count_pt": hist_real_pt,
        "purity_count_eta": hist_real_eta,
        "duplicate_rate_numerator_pt": hist(sim_pts[in_acc & sim_dup], pt_bins),
        "duplicate_rate_numerator_eta": hist(sim_etas[sim_dup], eta_bins),
        "cluster_sizes": m["n_t3"].tolist(),
        # per cluster with >= min_layers layers: (reco pT, reco eta, size), layer span, hit fraction
        "cluster_data": list(zip(reco_pt[keep], reco_eta[keep], m["n_t3"][keep])),
        "cluster_layer_spans": m["n_layers"][keep].tolist(),
        "cluster_hit_fractions": m["frac"][keep].tolist(),
    }


def t3_beta_hit_fraction_hist(data, beta, beta_bins):
    """
    Counts (H + 1, len(beta_bins) - 1) of the T3s' beta, per T3 hit fraction k / H (row k): the fraction
    of the T3's hits on its best sim track, as used by the hit-fraction loss (hit_fraction_loss).
    """
    cand = hit_candidates(data)
    n_hits = cand.shape[1]
    _, frac = hit_targets(cand, 1)
    k = np.rint(to_numpy(frac) * n_hits).astype(int)
    beta = np.clip(np.asarray(beta).flatten(), beta_bins[0], beta_bins[-1])
    counts = np.zeros((n_hits + 1, len(beta_bins) - 1))
    for i in range(n_hits + 1):
        counts[i] = np.histogram(beta[k == i], bins=beta_bins)[0]
    return counts


def plot_beta_vs_hit_fraction(counts, beta_bins, output_path, epsilon):
    """Normalized beta distributions of the T3s, one per T3 hit fraction (from t3_beta_hit_fraction_hist)."""
    n_hits = counts.shape[0] - 1
    fig, ax = plt.subplots(figsize=(12, 8))
    cmap = plt.get_cmap("viridis")
    for k in range(n_hits + 1):
        total = counts[k].sum()
        if total == 0:
            continue
        ax.stairs(counts[k] / total, beta_bins, color=cmap(k / n_hits), linewidth=2,
                  label=f"{k}/{n_hits} hits matched ({int(total)} T3s)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel("Fraction of T3s")
    ax.legend(fontsize=14, loc="upper left")
    hep.cms.label("Preliminary", data=False, rlabel=r"Phase-2 (14 TeV)", ax=ax)
    plt.tight_layout()
    plt.savefig(f"{output_path}/beta_vs_hit_fraction_eps_{epsilon:.3f}.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_cluster_hit_fraction(fractions, purity_threshold, output_path, epsilon):
    """Distribution of the clusters' best-sim hit fraction, with the matching threshold."""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.hist(fractions, bins=np.linspace(0, 1, 41), edgecolor="black", alpha=0.7, color="steelblue", label="Transformer OC")
    ax.axvline(purity_threshold, color="red", linestyle="--", label=f"Matching threshold ({purity_threshold})")
    ax.set_xlabel("Fraction of cluster hits on best sim track")
    ax.set_ylabel("Number of clusters")
    ax.set_yscale("log")
    ax.legend(fontsize=18, loc="upper left")
    hep.cms.label("Preliminary", data=False, rlabel=r"Phase-2 (14 TeV)", ax=ax)
    plt.tight_layout()
    plt.savefig(f"{output_path}/cluster_hit_fraction_eps_{epsilon:.3f}.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_performance_histograms(
    pt_bins, eta_bins, metrics_pt, metrics_eta, output_path, epsilon
):
    pt_centers = (pt_bins[:-1] + pt_bins[1:]) / 2
    eta_centers = (eta_bins[:-1] + eta_bins[1:]) / 2

    plots = [
        (
            "efficiency",
            "pt",
            pt_centers,
            metrics_pt["efficiency"],
            metrics_pt["efficiency_err"],
            "Simulated track $p_T$ [GeV]",
            "Efficiency",
            "blue",
            (0, 1.1),
        ),
        (
            "fake_rate",
            "pt",
            pt_centers,
            metrics_pt["fake_rate"],
            metrics_pt["fake_rate_err"],
            "Reconstructed track $p_T$ [GeV]",
            "Fake Rate",
            "red",
            None,
        ),
        (
            "purity",
            "pt",
            pt_centers,
            metrics_pt["purity"],
            metrics_pt["purity_err"],
            "Simulated track $p_T$ [GeV]",
            "Purity",
            "green",
            (0.95, 1.0),
        ),
        (
            "duplicate_rate",
            "pt",
            pt_centers,
            metrics_pt["duplicate_rate"],
            metrics_pt["duplicate_rate_err"],
            "Simulated track $p_T$ [GeV]",
            "Duplicate Rate",
            "purple",
            (0, max(0.5, np.max(metrics_pt["duplicate_rate"]) * 1.2)),
        ),
        (
            "efficiency",
            "eta",
            eta_centers,
            metrics_eta["efficiency"],
            metrics_eta["efficiency_err"],
            "Simulated track $η$",
            "Efficiency",
            "blue",
            (0, 1.1),
        ),
        (
            "fake_rate",
            "eta",
            eta_centers,
            metrics_eta["fake_rate"],
            metrics_eta["fake_rate_err"],
            "Reconstructed track $η$",
            "Fake Rate",
            "red",
            None,
        ),
        (
            "purity",
            "eta",
            eta_centers,
            metrics_eta["purity"],
            metrics_eta["purity_err"],
            "Simulated track $η$",
            "Purity",
            "green",
            (0.95, 1.0),
        ),
        (
            "duplicate_rate",
            "eta",
            eta_centers,
            metrics_eta["duplicate_rate"],
            metrics_eta["duplicate_rate_err"],
            "Simulated track $η$",
            "Duplicate Rate",
            "purple",
            (0, max(0.5, np.max(metrics_eta["duplicate_rate"]) * 1.2)),
        ),
    ]

    for (
        metric_name,
        var_name,
        centers,
        y_data,
        y_err,
        xlabel,
        ylabel,
        color,
        ylim,
    ) in plots:
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.errorbar(
            centers,
            y_data,
            yerr=y_err,
            fmt="o",
            color=color,
            label="Transformer OC",
            capsize=3,
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if ylim is not None:
            ax.set_ylim(ylim)

        hep.cms.label(
            "Preliminary",
            data=False,
            rlabel=r"Phase-2 (14 TeV)",
            ax=ax,
        )
        ax.text(0.95, 0.95, r"$t\bar{t}$, $\langle\mu\rangle=200$", transform=ax.transAxes, fontsize=20, verticalalignment='top', horizontalalignment='right')
        ax.legend(fontsize=18, loc='upper right', bbox_to_anchor=(0.95, 0.88))

        plt.tight_layout()
        plt.savefig(
            f"{output_path}/{metric_name}_{var_name}_eps_{epsilon:.3f}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()


def plot_cluster_size_histogram(
    cluster_sizes, pt_bins, eta_bins, output_path, epsilon, cluster_data=None
):
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    cluster_sizes = np.array(cluster_sizes)
    bins = np.arange(0, min(30, cluster_sizes.max() + 2), 1) - 0.5
    ax.hist(
        cluster_sizes,
        bins=bins,
        edgecolor="black",
        alpha=0.7,
        color="steelblue",
        label="Transformer OC",
    )

    ax.set_xlabel("Number of T3s per Cluster")
    ax.set_ylabel("Number of Clusters")
    ax.grid(True, alpha=0.3)
    hep.cms.label(
        "Preliminary",
        data=False,
        rlabel=r"Phase-2 (14 TeV)",
        ax=ax,
    )
    ax.text(0.95, 0.95, r"$t\bar{t}$, $\langle\mu\rangle=200$", transform=ax.transAxes, fontsize=20, verticalalignment='top', horizontalalignment='right')
    ax.legend(fontsize=18, loc='upper right', bbox_to_anchor=(0.95, 0.88))

    plt.tight_layout()
    plt.savefig(
        f"{output_path}/cluster_size_overall_eps_{epsilon:.3f}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    if cluster_data is not None and len(cluster_data) > 0:
        cluster_data_array = np.array(cluster_data)
        pts = cluster_data_array[:, 0]
        etas = cluster_data_array[:, 1]
        sizes = cluster_data_array[:, 2]

        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        size_sum, xedges, yedges = np.histogram2d(
            pts, etas, bins=[pt_bins, eta_bins], weights=sizes
        )
        size_count, _, _ = np.histogram2d(pts, etas, bins=[pt_bins, eta_bins])

        mean_size = np.divide(
            size_sum, size_count, out=np.zeros_like(size_sum), where=size_count > 0
        )

        im = ax.pcolormesh(eta_bins, pt_bins, mean_size, cmap="viridis", shading="auto")
        cbar = plt.colorbar(im, ax=ax)

        ax.set_xlabel("Simulated track $η$")
        ax.set_ylabel("Simulated track $p_T$ [GeV]")
        ax.grid(True, alpha=0.3, color="white", linewidth=0.5)
        hep.cms.label(
            "Preliminary",
            data=False,
            rlabel=r"Phase-2 (14 TeV)",
            ax=ax,
        )
        ax.text(0.95, 0.95, r"$t\bar{t}$, $\langle\mu\rangle=200$", transform=ax.transAxes, fontsize=20, verticalalignment='top', horizontalalignment='right')

        plt.tight_layout()
        plt.savefig(
            f"{output_path}/cluster_size_2d_eps_{epsilon:.3f}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()


def plot_layer_span_histogram(cluster_layer_spans, output_path, epsilon):
    if cluster_layer_spans is None or len(cluster_layer_spans) == 0:
        print("No layer span data available to plot.")
        return

    fig, ax = plt.subplots()

    cluster_layer_spans = np.array(cluster_layer_spans)
    bins = np.arange(cluster_layer_spans.min() - 1, cluster_layer_spans.max() + 2, 1) - 0.5
    ax.hist(
        cluster_layer_spans,
        bins=bins,
        density=True,
        edgecolor="black",
        alpha=0.7,
        color="coral",
        label="Transformer OC",
    )

    ax.set_xlabel("Number of unique layers per cluster")
    ax.set_ylabel("Density")
    ax.set_ylim(0, 0.7)
    ax.grid(True, alpha=0.3)
    hep.cms.label(
        "Preliminary",
        data=False,
        rlabel=r"Phase-2 (14 TeV)",
        ax=ax,
    )
    ax.text(0.95, 0.95, r"$t\bar{t}$, $\langle\mu\rangle=200$", transform=ax.transAxes, fontsize=20, verticalalignment='top', horizontalalignment='right')
    ax.legend(fontsize=18, loc='upper right', bbox_to_anchor=(0.95, 0.88))

    plt.tight_layout()
    plt.savefig(
        f"{output_path}/cluster_layer_span_eps_{epsilon:.3f}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def plot_layer_span_2d(
    cluster_data, cluster_layer_spans, pt_bins, eta_bins, output_path, epsilon
):
    if cluster_data is None or cluster_layer_spans is None or len(cluster_data) == 0:
        print("No cluster data or layer span data available for 2D plot.")
        return

    cluster_data_array = np.array(cluster_data)
    pts = cluster_data_array[:, 0]
    etas = cluster_data_array[:, 1]
    layer_spans = np.array(cluster_layer_spans)

    if len(pts) != len(layer_spans):
        print(
            f"Warning: Mismatch in data lengths (pts: {len(pts)}, layer_spans: {len(layer_spans)})"
        )
        min_len = min(len(pts), len(layer_spans))
        pts = pts[:min_len]
        etas = etas[:min_len]
        layer_spans = layer_spans[:min_len]

    fig, ax = plt.subplots()

    layer_sum, xedges, yedges = np.histogram2d(
        pts, etas, bins=[pt_bins, eta_bins], weights=layer_spans
    )
    layer_count, _, _ = np.histogram2d(pts, etas, bins=[pt_bins, eta_bins])

    mean_layer_span = np.divide(
        layer_sum, layer_count, out=np.zeros_like(layer_sum), where=layer_count > 0
    )

    im = ax.pcolormesh(
        eta_bins, pt_bins, mean_layer_span, cmap="plasma", shading="auto"
    )
    cbar = plt.colorbar(im, ax=ax)

    ax.set_xlabel("Simulated track $η$")
    ax.set_ylabel("Simulated track $p_T$ [GeV]")
    ax.grid(True, alpha=0.3, color="white", linewidth=0.5)
    hep.cms.label(
        "Preliminary",
        data=False,
        rlabel=r"Phase-2 (14 TeV)",
        ax=ax,
    )
    ax.text(0.95, 0.95, r"$t\bar{t}$, $\langle\mu\rangle=200$", transform=ax.transAxes, fontsize=20, verticalalignment='top', horizontalalignment='right')

    plt.savefig(
        f"{output_path}/layer_span_2d_eps_{epsilon:.3f}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
