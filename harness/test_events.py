"""
Tests for harness/events.py.

From the repo root:
apptainer exec /ceph/users/atuna/ml-tracking.sif python3 -m pytest harness/test_events.py
"""

import os

from harness.events import read_file_list


def test_read_file_list(tmp_path):
    # Files keep list order
    path = os.path.join(tmp_path, "val.txt")
    with open(path, "w") as list_file:
        list_file.write("b.root\na.root\n")
    assert read_file_list(path=path) == ["b.root", "a.root"]


def test_read_file_list_empty(tmp_path):
    # A dataset with no files has an empty list
    path = os.path.join(tmp_path, "test.txt")
    open(path, "w").close()
    assert read_file_list(path=path) == []
