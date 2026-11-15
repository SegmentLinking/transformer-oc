"""
Which events form the train, validation, tuning and test sets, from the manifest written by transformer-datasets'
md_processing.py (<output>/manifest.txt). Never reshuffled: the manifest is the split.

One line per event, as key=value:
    event_file=event_0.pt event_set=train lstntuple_file=LSTNtuple_X.root lstntuple_entry=0

The manifest records the LSTNtuple's file name, not its directory: the same input exists as an all-sim LSTNtuple
(for training) and a nominal one (for evaluation), with the same name and event order. The harness reads the
nominal one, joining the file name with a directory it is given.
"""

from dataclasses import dataclass

EVENT_SETS = ["train", "val", "tune", "test"]


@dataclass
class ManifestEvent:
    """One manifest line: the event's dataset file, its set, and where it is in the LSTNtuples."""

    event_file: str
    event_set: str
    lstntuple_file: str
    lstntuple_entry: int


def read_manifest(path: str) -> list[ManifestEvent]:
    """Every event in the manifest, in manifest order."""
    events = []
    with open(path) as manifest_file:
        for line in manifest_file:
            # "key=value key=value ..." -> {key: value}
            fields = dict(item.split("=", 1) for item in line.split())
            events.append(
                ManifestEvent(
                    event_file=fields["event_file"],
                    event_set=fields["event_set"],
                    lstntuple_file=fields["lstntuple_file"],
                    lstntuple_entry=int(fields["lstntuple_entry"]),
                )
            )
    return events


def events_in_set(
    events: list[ManifestEvent],
    event_set: str,
) -> list[ManifestEvent]:
    """The events of one set (one of EVENT_SETS), in manifest order. Raises ValueError for an unknown set."""
    if event_set not in EVENT_SETS:
        raise ValueError(f"unknown event_set={event_set}, expected one of {EVENT_SETS}")
    return [event for event in events if event.event_set == event_set]


def entries_by_lstntuple(events: list[ManifestEvent]) -> dict[str, list[int]]:
    """The sorted entries to read from each LSTNtuple file name, files in order of first appearance."""
    entries = {}
    for event in events:
        if event.lstntuple_file not in entries:
            entries[event.lstntuple_file] = []
        entries[event.lstntuple_file].append(event.lstntuple_entry)
    return {
        lstntuple_file: sorted(file_entries)
        for lstntuple_file, file_entries in entries.items()
    }
