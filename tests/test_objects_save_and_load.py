import pickle
from pathlib import Path
from train.scripts.utils import load_pickle


def test_load_pickle_roundtrip(tmp_path):
    obj = {"a": 1, "b": 2}
    file = tmp_path / "test.pkl"
    with file.open("wb") as f:
        pickle.dump(obj, f)
    loaded = load_pickle(file)
    assert loaded == obj
