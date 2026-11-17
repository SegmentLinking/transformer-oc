"""
Tests for harness/events.py.

From the repo root:
apptainer exec /ceph/users/atuna/ml-tracking.sif python3 -m pytest harness/test_events.py
"""

import os
import tempfile

import pytest

from harness.events import (
    ManifestEvent,
    entries_by_lstntuple,
    events_in_set,
    read_manifest,
)

# Five events from two LSTNtuples; b.root appears first, and its entries are out of order
MANIFEST_LINES = [
    "event_file=event_0.pt event_set=train lstntuple_file=b.root lstntuple_entry=7",
    "event_file=event_1.pt event_set=test lstntuple_file=a.root lstntuple_entry=0",
    "event_file=event_2.pt event_set=test lstntuple_file=b.root lstntuple_entry=2",
    "event_file=event_3.pt event_set=val lstntuple_file=a.root lstntuple_entry=1",
    "event_file=event_4.pt event_set=test lstntuple_file=b.root lstntuple_entry=5",
]


def read_test_manifest() -> list[ManifestEvent]:
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "manifest.txt")
        with open(path, "w") as manifest_file:
            manifest_file.write("\n".join(MANIFEST_LINES) + "\n")
        return read_manifest(path=path)


def test_read_manifest():
    events = read_test_manifest()
    assert len(events) == 5
    assert events[0] == ManifestEvent(
        event_file="event_0.pt",
        event_set="train",
        lstntuple_file="b.root",
        lstntuple_entry=7,
    )
    assert [event.event_file for event in events] == [
        "event_0.pt",
        "event_1.pt",
        "event_2.pt",
        "event_3.pt",
        "event_4.pt",
    ]


def test_events_in_set():
    events = read_test_manifest()
    in_test_set = events_in_set(events=events, event_set="test")
    assert [event.event_file for event in in_test_set] == [
        "event_1.pt",
        "event_2.pt",
        "event_4.pt",
    ]
    # A known set with no events in this manifest
    assert events_in_set(events=events, event_set="tune") == []


def test_events_in_set_rejects_unknown_set():
    events = read_test_manifest()
    with pytest.raises(ValueError):
        events_in_set(events=events, event_set="validation")


def test_entries_by_lstntuple():
    events = read_test_manifest()
    # Files in order of first appearance (b.root first), entries sorted
    assert entries_by_lstntuple(events=events) == {
        "b.root": [2, 5, 7],
        "a.root": [0, 1],
    }
    assert list(entries_by_lstntuple(events=events)) == ["b.root", "a.root"]
