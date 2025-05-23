"""Utility functions for loading and saving Python objects."""

import pickle
from pathlib import Path


def load_pickle(path: Path):
    """Load a Python object from a pickle file.

    Args:
        path (Path): Path to the pickle file.

    Returns:
        Any: The loaded Python object.
    """
    with path.open("rb") as f_in:
        return pickle.load(f_in)
