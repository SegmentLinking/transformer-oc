"""
Evaluate track candidates' performance on one dataset: efficiency, fake rate and duplicate rate, as key=value logs and pdf
plots. The track candidates are LST T5s, or a ground-truth check of the harness from MDs matched to sims
(--trackcands).

The dataset's LSTNtuple files come from its file list (val.txt from md_processing.py), and are read from a directory
of nominal-mode LSTNtuples.
Every event of each file is evaluated.

From the repo root:
apptainer exec /ceph/users/atuna/ml-tracking.sif python3 -m harness.evaluate \
    --file-list ../transformer-datasets/samples/ttbar/val.txt \
    --lstntuple-dir ../transformer-datasets/samples/ttbar/nominal \
    --trackcands t5 \
    --output output/evaluate/val
"""

import argparse
import os
import time
from dataclasses import dataclass

import awkward as ak
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib import rcParams
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import uproot

from harness.constants import COLUMNS, SIM_COLUMNS
from harness.events import read_file_list
from harness.metrics import (
    DENOMINATOR_PT_MIN,
    is_denominator,
    is_duplicate,
    n_trackcands_per_sim,
)
from harness.trackcands import (
    trackcands_from_mds_matched_to_sim,
    trackcands_from_t5s,
)

# Where the track candidates come from (--trackcands)
TRACKCAND_SOURCES = ["t5", "mds_matched_to_sim"]
# Log-spaced from 0.1 to 50 GeV (27 bins), for plots with a log pT axis
PT_BINS = np.geomspace(0.1, 50.0, 28)
ETA_BINS = np.linspace(-2.4, 2.4, 25)
PHI_BINS = np.linspace(-np.pi, np.pi, 25)
# Plot style
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
# Fixed y ranges, so plots of different track candidates can be compared by eye; rates on a log axis
EFFICIENCY_Y_RANGE = (0.0, 1.02)
RATE_Y_RANGE = (1e-3, 2.0)


@dataclass
class Performance:
    """
    Sim tracks (sims) and track candidates, as per-event arrays.

    The sim_* fields hold the sims in the efficiency denominator without its pT cut (pt_min=0), so efficiency vs
    pT can show the turn-on; apply DENOMINATOR_PT_MIN for everything else. The trackcand_* fields hold every track
    candidate.
    """

    sim_pt: np.ndarray
    sim_eta: np.ndarray
    sim_phi: np.ndarray
    sim_isFound: np.ndarray
    trackcand_pt: np.ndarray
    trackcand_eta: np.ndarray
    trackcand_phi: np.ndarray
    trackcand_isFake: np.ndarray
    trackcand_isDuplicate: np.ndarray


def main() -> None:
    args = parse_args()
    lstntuple_files = read_file_list(path=args.file_list)
    if args.n_files is not None:
        lstntuple_files = lstntuple_files[: args.n_files]
    all_performances = evaluate_events(
        lstntuple_files=lstntuple_files,
        lstntuple_dir=args.lstntuple_dir,
        trackcand_source=args.trackcands,
        drop_duplicate_cleaned=args.drop_duplicate_cleaned,
        min_layers=args.min_layers,
    )
    perf = concatenate_performances(all_performances=all_performances)
    log_totals(perf=perf, n_events=len(all_performances))
    os.makedirs(args.output, exist_ok=True)
    pdf_path = os.path.join(args.output, f"performance_{args.trackcands}.pdf")
    with PdfPages(pdf_path) as pdf:
        print(f"[plot] pdf_path={pdf_path}")
        plot_performance(
            file_list=args.file_list,
            perf=perf,
            pdf=pdf,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate track candidates' performance on one dataset."
    )
    parser.add_argument(
        "--file-list",
        type=str,
        required=True,
        help="the dataset's LSTNtuple file list from md_processing.py, e.g. val.txt",
    )
    parser.add_argument(
        "--lstntuple-dir", required=True, help="directory of nominal-mode LSTNtuples"
    )
    parser.add_argument(
        "--n-files",
        type=int,
        default=None,
        help="evaluate only the first N LSTNtuple files of the dataset",
    )
    parser.add_argument(
        "--trackcands",
        choices=TRACKCAND_SOURCES,
        default="t5",
        help="where the track candidates come from",
    )
    parser.add_argument(
        "--drop-duplicate-cleaned",
        action="store_true",
        help="t5 only: drop T5s removed by LST's T5-T5 duplicate cleaning",
    )
    parser.add_argument(
        "--min-layers",
        type=int,
        default=0,
        help="mds_matched_to_sim only: keep sims whose MDs span at least this many layers",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="directory for the plots, written as performance_<trackcands>.pdf",
    )
    return parser.parse_args()


def evaluate_events(
    lstntuple_files: list[str],
    lstntuple_dir: str,
    trackcand_source: str,
    drop_duplicate_cleaned: bool,
    min_layers: int,
) -> list[Performance]:
    """
    Evaluate every event of each LSTNtuple file, opening each file once and reading its entries one at a time.

    Logs one [event] line per event.
    """
    all_perfs = []
    for lstntuple_file in lstntuple_files:
        with uproot.open(os.path.join(lstntuple_dir, lstntuple_file)) as root_file:
            tree = root_file["tree"]
            for entry in range(tree.num_entries):
                start = time.time()
                event = read_event(
                    tree=tree,
                    entry=entry,
                    columns=COLUMNS,
                )
                perf = evaluate_event(
                    event=event,
                    trackcand_source=trackcand_source,
                    drop_duplicate_cleaned=drop_duplicate_cleaned,
                    min_layers=min_layers,
                )
                all_perfs.append(perf)
                print(
                    f"[event] lstntuple_file={lstntuple_file} lstntuple_entry={entry}"
                    f" n_denominator={len(perf.sim_pt)} n_found={np.sum(perf.sim_isFound)}"
                    f" n_trackcands={len(perf.trackcand_pt)} n_fake={np.sum(perf.trackcand_isFake)}"
                    f" n_duplicate={np.sum(perf.trackcand_isDuplicate)} seconds={time.time() - start:.1f}"
                )
    return all_perfs


def read_event(
    tree: uproot.TTree,
    entry: int,
    columns: list[str],
) -> ak.Record:
    """
    One entry of an LSTNtuple tree as one event (an ak.Record).

    tree.arrays returns an ak.Array with one element per entry in [entry_start, entry_stop); reading exactly one
    entry and taking element [0] gives that event.
    """
    events = tree.arrays(
        columns,
        entry_start=entry,
        entry_stop=entry + 1,
    )
    return events[0]


def evaluate_event(
    event: ak.Record,
    trackcand_source: str,
    drop_duplicate_cleaned: bool,
    min_layers: int,
) -> Performance:
    """
    One event: build its track candidates, each with its sim from its source, and flag found sims, fakes and
    duplicates.
    """
    if trackcand_source == "t5":
        trackcands = trackcands_from_t5s(
            event=event,
            drop_duplicate_cleaned=drop_duplicate_cleaned,
        )
    else:
        trackcands = trackcands_from_mds_matched_to_sim(
            event=event,
            min_layers=min_layers,
        )
    trackcand_simIdx = trackcands.trackcand_simIdx
    sims = {column: ak.to_numpy(event[column]) for column in SIM_COLUMNS}
    n_trackcands_of_sim = n_trackcands_per_sim(
        trackcand_simIdx=trackcand_simIdx,
        n_sims=len(sims["sim_pt"]),
    )
    # Without the pT cut, so efficiency vs pT shows the turn-on
    in_denominator = is_denominator(sims=sims, pt_min=0.0)
    return Performance(
        sim_pt=sims["sim_pt"][in_denominator],
        sim_eta=sims["sim_eta"][in_denominator],
        sim_phi=sims["sim_phi"][in_denominator],
        sim_isFound=(n_trackcands_of_sim > 0)[in_denominator],
        trackcand_pt=trackcands.trackcand_pt,
        trackcand_eta=trackcands.trackcand_eta,
        trackcand_phi=trackcands.trackcand_phi,
        trackcand_isFake=trackcand_simIdx < 0,
        trackcand_isDuplicate=is_duplicate(
            trackcand_simIdx=trackcand_simIdx,
            n_trackcands_of_sim=n_trackcands_of_sim,
        ),
    )


def concatenate_performances(all_performances: list[Performance]) -> Performance:
    """All events' performances in one Performance, each field concatenated over events."""
    return Performance(
        sim_pt=np.concatenate([perf.sim_pt for perf in all_performances]),
        sim_eta=np.concatenate([perf.sim_eta for perf in all_performances]),
        sim_phi=np.concatenate([perf.sim_phi for perf in all_performances]),
        sim_isFound=np.concatenate(
            [perf.sim_isFound for perf in all_performances]
        ),
        trackcand_pt=np.concatenate(
            [perf.trackcand_pt for perf in all_performances]
        ),
        trackcand_eta=np.concatenate(
            [perf.trackcand_eta for perf in all_performances]
        ),
        trackcand_phi=np.concatenate(
            [perf.trackcand_phi for perf in all_performances]
        ),
        trackcand_isFake=np.concatenate(
            [perf.trackcand_isFake for perf in all_performances]
        ),
        trackcand_isDuplicate=np.concatenate(
            [perf.trackcand_isDuplicate for perf in all_performances]
        ),
    )


def log_totals(
    perf: Performance,
    n_events: int,
) -> None:
    """One [total] line: counts and rates over all events, with the denominator's pT cut (DENOMINATOR_PT_MIN)."""
    is_found = perf.sim_isFound[perf.sim_pt >= DENOMINATOR_PT_MIN]
    efficiency, efficiency_unc = fraction_and_uncertainty(
        n_selected=np.sum(is_found),
        n_total=len(is_found),
    )
    fake_rate, fake_rate_unc = fraction_and_uncertainty(
        n_selected=np.sum(perf.trackcand_isFake),
        n_total=len(perf.trackcand_isFake),
    )
    duplicate_rate, duplicate_rate_unc = fraction_and_uncertainty(
        n_selected=np.sum(perf.trackcand_isDuplicate),
        n_total=len(perf.trackcand_isDuplicate),
    )
    print(
        f"[total] n_events={n_events} n_denominator={len(is_found)} n_found={np.sum(is_found)}"
        f" efficiency={efficiency:.4f} efficiency_unc={efficiency_unc:.4f}"
        f" n_trackcands={len(perf.trackcand_isFake)} n_fake={np.sum(perf.trackcand_isFake)}"
        f" fake_rate={fake_rate:.4f} fake_rate_unc={fake_rate_unc:.4f}"
        f" n_duplicate={np.sum(perf.trackcand_isDuplicate)}"
        f" duplicate_rate={duplicate_rate:.4f} duplicate_rate_unc={duplicate_rate_unc:.4f}"
    )


def plot_performance(
    file_list: str,
    perf: Performance,
    pdf: PdfPages,
) -> None:
    """
    Efficiency vs sim pT (no pT cut), eta and phi; fake and duplicate rate vs track candidate pT, eta and phi.

    All in one pdf, one plot per page.
    """
    has_pt_cut = perf.sim_pt >= DENOMINATOR_PT_MIN
    pt_cut_label = rf"($p_\mathrm{{T}} \geq {DENOMINATOR_PT_MIN:g}$ GeV)"
    plot_fraction(
        file_list=file_list,
        values=perf.sim_pt,
        is_selected=perf.sim_isFound,
        bins=PT_BINS,
        xlabel=r"Sim. $p_\mathrm{T}$ [GeV]",
        ylabel="Efficiency",
        y_range=EFFICIENCY_Y_RANGE,
        pdf=pdf,
        log_x=True,
    )
    for values, bins, xlabel in [
        (perf.sim_eta, ETA_BINS, rf"Sim. $\eta$ {pt_cut_label}"),
        (perf.sim_phi, PHI_BINS, rf"Sim. $\phi$ {pt_cut_label}"),
    ]:
        plot_fraction(
            file_list=file_list,
            values=values[has_pt_cut],
            is_selected=perf.sim_isFound[has_pt_cut],
            bins=bins,
            xlabel=xlabel,
            ylabel="Efficiency",
            y_range=EFFICIENCY_Y_RANGE,
            pdf=pdf,
        )
    for ylabel, is_flagged in [
        ("Fake rate", perf.trackcand_isFake),
        ("Duplicate rate", perf.trackcand_isDuplicate),
    ]:
        plot_fraction(
            file_list=file_list,
            values=perf.trackcand_pt,
            is_selected=is_flagged,
            bins=PT_BINS,
            xlabel=r"Track candidate $p_\mathrm{T}$ [GeV]",
            ylabel=ylabel,
            y_range=RATE_Y_RANGE,
            pdf=pdf,
            log_x=True,
            log_y=True,
        )
        for values, bins, xlabel in [
            (perf.trackcand_eta, ETA_BINS, r"Track candidate $\eta$"),
            (perf.trackcand_phi, PHI_BINS, r"Track candidate $\phi$"),
        ]:
            plot_fraction(
                file_list=file_list,
                values=values,
                is_selected=is_flagged,
                bins=bins,
                xlabel=xlabel,
                ylabel=ylabel,
                y_range=RATE_Y_RANGE,
                pdf=pdf,
                log_y=True,
            )
    print(f"[plot] pages={pdf.get_pagecount()}")


def plot_fraction(
    file_list: str,
    values: np.ndarray,
    is_selected: np.ndarray,
    bins: np.ndarray,
    xlabel: str,
    ylabel: str,
    y_range: tuple[float, float],
    pdf: PdfPages,
    log_x: bool = False,
    log_y: bool = False,
) -> None:
    """
    Per bin of values, the fraction with is_selected (e.g. efficiency: sims, is found), as one page of pdf, with
    the y-axis fixed to y_range.

    Error bars are from fraction_and_uncertainty; empty bins are not drawn, nor are zero fractions on a log
    y-axis. With log_x, the x-axis is logarithmic and each point sits at its bin's geometric center,
    sqrt(low * high), which looks centered on that axis.
    """
    n_total, _ = np.histogram(values, bins=bins)
    n_selected, _ = np.histogram(values[is_selected], bins=bins)
    fraction, uncertainty = fraction_and_uncertainty(
        n_selected=n_selected,
        n_total=n_total,
    )
    bin_lows, bin_highs = bins[:-1], bins[1:]
    if log_x:
        bin_centers = np.sqrt(bin_lows * bin_highs)
    else:
        bin_centers = (bin_lows + bin_highs) / 2
    fig, ax = plt.subplots()
    ax.errorbar(
        bin_centers,
        fraction,
        # Horizontal bars from the center to each bin edge
        xerr=[bin_centers - bin_lows, bin_highs - bin_centers],
        yerr=uncertainty,
        fmt="o",
        markersize=3,
    )
    if log_x:
        ax.semilogx()
        # Tick labels as plain numbers (0.1, 1, 10), not powers of ten
        ax.xaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:g}"))
    if log_y:
        ax.semilogy()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(y_range)
    ax.text(0.05, 1.01, dataset_name(file_list), transform=ax.transAxes)
    pdf.savefig(fig)
    plt.close(fig)


def fraction_and_uncertainty(
    n_selected: np.ndarray,
    n_total: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    The fraction f = k / n and its binomial uncertainty sigma_f = sqrt(f * (1 - f) / n), for k = n_selected and
    n = n_total (numbers or arrays).

    This simple form gives sigma_f = 0 at f = 0 or f = 1. Where n = 0, both are nan (not drawn in plots).
    """
    # Silence the division warnings for n = 0, which give nan
    with np.errstate(divide="ignore", invalid="ignore"):
        fraction = n_selected / n_total
        uncertainty = np.sqrt(fraction * (1 - fraction) / n_total)
    return fraction, uncertainty


def dataset_name(file_list: str) -> str:
    bname = os.path.basename(file_list)
    if bname == "train.txt":
        return "Training"
    if bname == "val.txt":
        return "Validation"
    if bname == "test.txt":
        return "Test"
    raise ValueError(f"Unknown dataset file: {file_list}")


if __name__ == "__main__":
    main()
