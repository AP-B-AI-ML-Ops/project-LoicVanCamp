import pickle
from pathlib import Path


def load_pickle(path: Path):
    with path.open("rb") as f_in:
        return pickle.load(f_in)
