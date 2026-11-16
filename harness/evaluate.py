"""
Evaluate track candidates on one event set: efficiency, fake rate and duplicate rate, as key=value logs and pdf
plots. The track candidates are LST T5s, or a ground-truth check of the harness from MDs matched to sims
(--trackcands).

Events come from the manifest; their LSTNtuples are read from a directory of nominal-mode LSTNtuples.

From the repo root:
apptainer exec /ceph/users/atuna/ml-tracking.sif python3 -m harness.evaluate \
    --manifest ../transformer-datasets/samples/ttbar/manifest.txt \
    --lstntuple-dir ../transformer-datasets/samples/ttbar \
    --event-set val \
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
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import uproot

from harness.events import (
    EVENT_SETS,
    ManifestEvent,
    entries_by_lstntuple,
    events_in_set,
    read_manifest,
)
from harness.md_truth import HIT_SIMS_FROM_MDS_COLUMNS, hit_sims_from_mds
from harness.metrics import (
    DENOMINATOR_PT_MIN,
    SIM_COLUMNS,
    is_denominator,
    is_duplicate,
    n_trackcands_per_sim,
)
from harness.trackcand_matching import matched_sim
from harness.trackcands import (
    MDS_MATCHED_TO_SIM_COLUMNS,
    T5_COLUMNS,
    EventTrackCandidates,
    trackcands_from_mds_matched_to_sim,
    trackcands_from_t5s,
)

# Where the track candidates come from, and the extra columns each needs
COLUMNS_OF_TRACKCAND_SOURCE = {
    "t5": T5_COLUMNS,
    "mds_matched_to_sim": MDS_MATCHED_TO_SIM_COLUMNS,
}
# Columns for matching and metrics, whatever the track candidates
COLUMNS = HIT_SIMS_FROM_MDS_COLUMNS + SIM_COLUMNS
# Log-spaced from 0.1 to 50 GeV (27 bins), for plots with a log pT axis
PT_BINS = np.geomspace(0.1, 50.0, 28)
ETA_BINS = np.linspace(-2.4, 2.4, 25)


@dataclass
class EventEvaluation:
    """
    One event's evaluated sims and track candidates, as per-event arrays.

    The sim_* fields hold the sims in the efficiency denominator without its pT cut (pt_min=0), so efficiency vs
    pT can show the turn-on; apply DENOMINATOR_PT_MIN for everything else. The trackcand_* fields hold every track
    candidate.
    """

    sim_pt: np.ndarray
    sim_eta: np.ndarray
    sim_isFound: np.ndarray
    trackcand_pt: np.ndarray
    trackcand_eta: np.ndarray
    trackcand_isFake: np.ndarray
    trackcand_isDuplicate: np.ndarray


def main() -> None:
    args = parse_args()
    manifest_events = read_manifest(path=args.manifest)
    events = events_in_set(events=manifest_events, event_set=args.event_set)
    if args.n_events is not None:
        events = events[: args.n_events]
    all_evaluations = evaluate_events(
        events=events,
        lstntuple_dir=args.lstntuple_dir,
        trackcand_source=args.trackcands,
        drop_duplicate_cleaned=args.drop_duplicate_cleaned,
        min_layers=args.min_layers,
    )
    evaluation = concatenate_evaluations(all_evaluations=all_evaluations)
    log_totals(evaluation=evaluation, n_events=len(all_evaluations))
    os.makedirs(args.output, exist_ok=True)
    plot_evaluation(
        evaluation=evaluation,
        path=os.path.join(args.output, f"evaluation_{args.trackcands}.pdf"),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate track candidates on one event set."
    )
    parser.add_argument(
        "--manifest", required=True, help="manifest.txt from md_processing.py"
    )
    parser.add_argument(
        "--lstntuple-dir", required=True, help="directory of nominal-mode LSTNtuples"
    )
    parser.add_argument("--event-set", required=True, choices=EVENT_SETS)
    parser.add_argument(
        "--n-events",
        type=int,
        default=None,
        help="evaluate only the first N events of the set",
    )
    parser.add_argument(
        "--trackcands",
        choices=list(COLUMNS_OF_TRACKCAND_SOURCE),
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
        help="directory for the plots, written as evaluation_<trackcands>.pdf",
    )
    return parser.parse_args()


def evaluate_events(
    events: list[ManifestEvent],
    lstntuple_dir: str,
    trackcand_source: str,
    drop_duplicate_cleaned: bool,
    min_layers: int,
) -> list[EventEvaluation]:
    """
    Evaluate every event, opening each LSTNtuple once and reading its entries one at a time.

    Logs one [event] line per event.
    """
    all_evaluations = []
    for lstntuple_file, entries in entries_by_lstntuple(events=events).items():
        with uproot.open(os.path.join(lstntuple_dir, lstntuple_file)) as root_file:
            tree = root_file["tree"]
            for entry in entries:
                start = time.time()
                event = read_event(
                    tree=tree,
                    entry=entry,
                    columns=COLUMNS + COLUMNS_OF_TRACKCAND_SOURCE[trackcand_source],
                )
                evaluation = evaluate_event(
                    event=event,
                    trackcand_source=trackcand_source,
                    drop_duplicate_cleaned=drop_duplicate_cleaned,
                    min_layers=min_layers,
                )
                all_evaluations.append(evaluation)
                print(
                    f"[event] lstntuple_file={lstntuple_file} lstntuple_entry={entry}"
                    f" n_denominator={len(evaluation.sim_pt)} n_found={np.sum(evaluation.sim_isFound)}"
                    f" n_trackcands={len(evaluation.trackcand_pt)} n_fake={np.sum(evaluation.trackcand_isFake)}"
                    f" n_duplicate={np.sum(evaluation.trackcand_isDuplicate)} seconds={time.time() - start:.1f}"
                )
    return all_evaluations


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
) -> EventEvaluation:
    """
    One event: build its track candidates, match each to a sim, and flag found sims, fakes and duplicates.
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
    hit_sims = hit_sims_from_mds(event=event)
    trackcand_simIdx = match_trackcands(trackcands=trackcands, hit_sims=hit_sims)
    sims = {column: ak.to_numpy(event[column]) for column in SIM_COLUMNS}
    n_trackcands_of_sim = n_trackcands_per_sim(
        trackcand_simIdx=trackcand_simIdx,
        n_sims=len(sims["sim_pt"]),
    )
    # Without the pT cut, so efficiency vs pT shows the turn-on
    in_denominator = is_denominator(sims=sims, pt_min=0.0)
    return EventEvaluation(
        sim_pt=sims["sim_pt"][in_denominator],
        sim_eta=sims["sim_eta"][in_denominator],
        sim_isFound=(n_trackcands_of_sim > 0)[in_denominator],
        trackcand_pt=trackcands.trackcand_pt,
        trackcand_eta=trackcands.trackcand_eta,
        trackcand_isFake=trackcand_simIdx < 0,
        trackcand_isDuplicate=is_duplicate(
            trackcand_simIdx=trackcand_simIdx,
            n_trackcands_of_sim=n_trackcands_of_sim,
        ),
    )


def match_trackcands(
    trackcands: EventTrackCandidates,
    hit_sims: dict[int, list[int]],
) -> np.ndarray:
    """Each track candidate's matched sim index, shape (n_trackcands,), with NO_HIT padding stripped first."""
    # dtype=int keeps an event with no track candidates an int array
    return np.array(
        [
            matched_sim(hits=hits[hits >= 0].tolist(), hit_sims=hit_sims)
            for hits in trackcands.trackcand_hitIndices
        ],
        dtype=int,
    )


def concatenate_evaluations(all_evaluations: list[EventEvaluation]) -> EventEvaluation:
    """All events' evaluations in one EventEvaluation, each field concatenated over events."""
    return EventEvaluation(
        sim_pt=np.concatenate([evaluation.sim_pt for evaluation in all_evaluations]),
        sim_eta=np.concatenate([evaluation.sim_eta for evaluation in all_evaluations]),
        sim_isFound=np.concatenate(
            [evaluation.sim_isFound for evaluation in all_evaluations]
        ),
        trackcand_pt=np.concatenate(
            [evaluation.trackcand_pt for evaluation in all_evaluations]
        ),
        trackcand_eta=np.concatenate(
            [evaluation.trackcand_eta for evaluation in all_evaluations]
        ),
        trackcand_isFake=np.concatenate(
            [evaluation.trackcand_isFake for evaluation in all_evaluations]
        ),
        trackcand_isDuplicate=np.concatenate(
            [evaluation.trackcand_isDuplicate for evaluation in all_evaluations]
        ),
    )


def log_totals(
    evaluation: EventEvaluation,
    n_events: int,
) -> None:
    """One [total] line: counts and rates over all events, with the denominator's pT cut (DENOMINATOR_PT_MIN)."""
    is_found = evaluation.sim_isFound[evaluation.sim_pt >= DENOMINATOR_PT_MIN]
    efficiency, efficiency_unc = fraction_and_uncertainty(
        n_selected=np.sum(is_found),
        n_total=len(is_found),
    )
    fake_rate, fake_rate_unc = fraction_and_uncertainty(
        n_selected=np.sum(evaluation.trackcand_isFake),
        n_total=len(evaluation.trackcand_isFake),
    )
    duplicate_rate, duplicate_rate_unc = fraction_and_uncertainty(
        n_selected=np.sum(evaluation.trackcand_isDuplicate),
        n_total=len(evaluation.trackcand_isDuplicate),
    )
    print(
        f"[total] n_events={n_events} n_denominator={len(is_found)} n_found={np.sum(is_found)}"
        f" efficiency={efficiency:.4f} efficiency_unc={efficiency_unc:.4f}"
        f" n_trackcands={len(evaluation.trackcand_isFake)} n_fake={np.sum(evaluation.trackcand_isFake)}"
        f" fake_rate={fake_rate:.4f} fake_rate_unc={fake_rate_unc:.4f}"
        f" n_duplicate={np.sum(evaluation.trackcand_isDuplicate)}"
        f" duplicate_rate={duplicate_rate:.4f} duplicate_rate_unc={duplicate_rate_unc:.4f}"
    )


def plot_evaluation(
    evaluation: EventEvaluation,
    path: str,
) -> None:
    """
    Efficiency vs sim pT (no pT cut) and vs sim eta; fake and duplicate rate vs track candidate pT and eta.

    All in one pdf at path, one plot per page.
    """
    has_pt_cut = evaluation.sim_pt >= DENOMINATOR_PT_MIN
    with PdfPages(path) as pdf:
        plot_fraction(
            values=evaluation.sim_pt,
            is_selected=evaluation.sim_isFound,
            bins=PT_BINS,
            xlabel="sim pT [GeV]",
            ylabel="efficiency",
            pdf=pdf,
            log_x=True,
        )
        plot_fraction(
            values=evaluation.sim_eta[has_pt_cut],
            is_selected=evaluation.sim_isFound[has_pt_cut],
            bins=ETA_BINS,
            xlabel=f"sim eta (pT >= {DENOMINATOR_PT_MIN} GeV)",
            ylabel="efficiency",
            pdf=pdf,
        )
        for flag_name, is_flagged in [
            ("fake_rate", evaluation.trackcand_isFake),
            ("duplicate_rate", evaluation.trackcand_isDuplicate),
        ]:
            plot_fraction(
                values=evaluation.trackcand_pt,
                is_selected=is_flagged,
                bins=PT_BINS,
                xlabel="track candidate pT [GeV]",
                ylabel=flag_name.replace("_", " "),
                pdf=pdf,
                log_x=True,
            )
            plot_fraction(
                values=evaluation.trackcand_eta,
                is_selected=is_flagged,
                bins=ETA_BINS,
                xlabel="track candidate eta",
                ylabel=flag_name.replace("_", " "),
                pdf=pdf,
            )
    print(f"[plot] path={path}")


def plot_fraction(
    values: np.ndarray,
    is_selected: np.ndarray,
    bins: np.ndarray,
    xlabel: str,
    ylabel: str,
    pdf: PdfPages,
    log_x: bool = False,
) -> None:
    """
    Per bin of values, the fraction with is_selected (e.g. efficiency: sims, is found), as one page of pdf.

    Error bars are from fraction_and_uncertainty; empty bins are not drawn. With log_x, the x-axis is logarithmic
    and each point sits at its bin's geometric center, sqrt(low * high), which looks centered on that axis.
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
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(bottom=0.0)
    ax.grid(alpha=0.3)
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


if __name__ == "__main__":
    main()
