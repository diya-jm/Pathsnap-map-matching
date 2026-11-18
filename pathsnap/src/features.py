import numpy as np, pandas as pd
from shapely import STRtree, points, linestrings
from pyproj import Transformer
from src.load import load_trip, list_trips

TF = Transformer.from_crs(4326, 32632, always_xy=True)  # UTM 32N: lon 6-12 E
R = 60.0    # candidate radius (m), tune using miss rate on train
BUF = 300.0 # bbox buffer when cropping the road network (m)
W = 50      # max forward jump along the route when labelling

def project(trk, nod):
    tx, ty = TF.transform(trk.lon.values, trk.lat.values)
    nx, ny = TF.transform(nod.lon.values, nod.lat.values)
    return np.c_[tx, ty], np.c_[nx, ny]

def crop_arcs(P, N, arc):
    lo, hi = P.min(0) - BUF, P.max(0) + BUF
    inb = ((N >= lo) & (N <= hi)).all(1)
    return np.where(inb[arc.u.values] & inb[arc.v.values])[0]

def candidates(P, N, arc, ids, radius=R):
    A, B = N[arc.u.values[ids]], N[arc.v.values[ids]]
    tree = STRtree(linestrings(np.stack([A, B], axis=1)))
    pi, si = tree.query(points(P), predicate="dwithin", distance=radius)
    a, b = A[si], B[si]
    d = b - a
    L2 = np.maximum((d ** 2).sum(1), 1e-9)
    t = np.clip(((P[pi] - a) * d).sum(1) / L2, 0, 1)
    proj = a + t[:, None] * d
    return pd.DataFrame(dict(
        pt=pi, arc_id=ids[si], u=arc.u.values[ids[si]], v=arc.v.values[ids[si]],
        dist=np.linalg.norm(P[pi] - proj, axis=1),
        t=t, alen=np.sqrt(L2), abr=np.arctan2(d[:, 0], d[:, 1])))

def add_features(df, P, T):
    n = len(P)
    i0 = np.clip(np.arange(n) - 1, 0, n - 1)
    i1 = np.clip(np.arange(n) + 1, 0, n - 1)
    v = P[i1] - P[i0]
    dt = np.maximum(T[i1] - T[i0], 1e-3)
    heading = np.arctan2(v[:, 0], v[:, 1])
    speed = np.linalg.norm(v, axis=1) / dt
    gap = np.r_[0, np.diff(T)]
    pt = df["pt"].values
    df["speed"], df["gap"] = speed[pt], gap[pt]
    dh = df["abr"].values - heading[pt]
    df["cos_h"] = np.cos(dh)
    df["dh_deg"] = np.abs(np.degrees(np.arctan2(np.sin(dh), np.cos(dh))))
    g = df.groupby("pt")["dist"]
    df["n_cand"] = g.transform("size")
    df["rank"] = g.rank(method="first")
    df["d_minus_best"] = df["dist"] - g.transform("min")
    return df

def add_labels(df, rte):
    pos = {}
    for k, a in enumerate(rte):
        pos.setdefault(int(a), []).append(k)
    df = df.sort_values(["pt", "dist"]).reset_index(drop=True)
    y = np.zeros(len(df), int)
    j = 0
    for p, g in df.groupby("pt", sort=True):
        for idx, a in zip(g.index, g["arc_id"].values):
            ks = pos.get(int(a))
            if not ks:
                continue
            k = next((k for k in ks if j <= k <= j + W), None)
            if k is not None:
                y[idx] = 1
                j = k
                break   # sorted by distance, so first hit is the nearest route arc
    df["y"] = y
    return df

def build_trip(split, tid):
    trk, nod, arc, rte = load_trip(split, tid)
    P, N = project(trk, nod)
    ids = crop_arcs(P, N, arc)
    df = candidates(P, N, arc, ids)
    df = add_features(df, P, trk.t.values)
    df = add_labels(df, rte)
    df["trip"] = tid
    has = df.groupby("pt")["y"].max().sum()
    stats = dict(trip=tid, n_pts=len(P), arcs_kept=len(ids),
                 mean_cand=len(df) / len(P),
                 pts_no_cand=len(P) - df["pt"].nunique(),
                 miss_rate=1 - has / len(P))
    return df, stats

def build_split(split):
    frames, stats = [], []
    for tid in list_trips(split):
        df, s = build_trip(split, tid)
        frames.append(df); stats.append(s)
        print(tid, s)
    pd.concat(frames).to_pickle(f"results/feat_{split}.pkl")
    pd.DataFrame(stats).to_csv(f"results/prep_stats_{split}.csv", index=False)

if __name__ == "__main__":
    import sys
    build_split(sys.argv[1])
