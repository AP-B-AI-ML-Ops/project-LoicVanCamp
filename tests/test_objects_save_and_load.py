"""Unit test for verifying pickle save and load functionality using load_pickle utility."""

import pickle

from train.scripts.utils import load_pickle


def test_load_pickle_roundtrip(tmp_path):
    """Test that an object saved with pickle can be loaded back identically using load_pickle.

    Args:
        tmp_path (pathlib.Path): Temporary directory provided by pytest for file operations.
    """
    obj = {"a": 1, "b": 2}
    file = tmp_path / "test.pkl"
    with file.open("wb") as f:
        pickle.dump(obj, f)
    loaded = load_pickle(file)
    assert loaded == obj
