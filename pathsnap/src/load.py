import pandas as pd, os
ROOT = "data/raw"

def list_trips(split):
    return sorted(os.listdir(f"{ROOT}/{split}"))

def load_trip(split, tid):
    b = f"{ROOT}/{split}/{tid}/{tid}"
    trk = pd.read_csv(b + ".track", sep=r"\s+", header=None, names=["lon", "lat", "t"])
    trk = trk.dropna().sort_values("t").reset_index(drop=True)
    nod = pd.read_csv(b + ".nodes", sep=r"\s+", header=None, names=["lon", "lat"])
    arc = pd.read_csv(b + ".arcs", sep=r"\s+", header=None, names=["u", "v"])
    rte = pd.read_csv(b + ".route", header=None)[0].values
    return trk, nod, arc, rte
