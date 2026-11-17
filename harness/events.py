"""
Which LSTNtuple files form a dataset, from the file lists of md_processing.py.

md_processing.py writes train.txt, val.txt and test.txt, one LSTNtuple file name per line, e.g.
    LSTNtuple_X.root

The lists record the LSTNtuple's file name, not its directory: the same input exists as an all-sim LSTNtuple
(for training) and a nominal one (for evaluation), with the same name and event order. The harness reads the
nominal one, joining the file name with a directory it is given.
"""


def read_file_list(path: str) -> list[str]:
    """The LSTNtuple file names listed at path, in order."""
    with open(path) as list_file:
        return list_file.read().split()
