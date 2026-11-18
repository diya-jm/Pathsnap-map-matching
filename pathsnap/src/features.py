import numpy as np, pandas as pd
from shapely import STRtree, points, linestrings
from pyproj import Transformer
from src.load import load_trip, list_trips
from src.reference import labels

R = 60.0    # candidate radius (m), tune using miss rate on train
BUF = 300.0 # bbox buffer when cropping the road network (m)
W = 50      # max forward jump along the route when labelling
FEATURE_COLUMNS = [
    "dist", "t", "alen", "speed", "gap", "cos_h", "dh_deg",
    "n_cand", "rank", "d_minus_best",
]

def project(trk, nod):
    lon0 = float(trk.lon.median())
    lat0 = float(trk.lat.median())
    zone = int(np.floor((lon0 + 180) / 6)) + 1
    zone = min(max(zone, 1), 60)
    epsg = (32600 if lat0 >= 0 else 32700) + zone
    tf = Transformer.from_crs(4326, epsg, always_xy=True)
    tx, ty = tf.transform(trk.lon.values, trk.lat.values)
    nx, ny = tf.transform(nod.lon.values, nod.lat.values)
    P, N = np.c_[tx, ty], np.c_[nx, ny]
    if not np.isfinite(P).all() or not np.isfinite(N).all():
        raise ValueError(f"UTM projection EPSG:{epsg} produced non-finite coordinates")
    return P, N

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
    df = df.reset_index(drop=True)
    df["y"] = labels(df, rte, max_forward=W)
    return df

def validate_trip(trk, nod, P, N, df, tid):
    """Check coordinate, feature, label, and candidate dimensions per trip."""
    if P.shape != (len(trk), 2) or N.shape != (len(nod), 2):
        raise ValueError(f"{tid}: projected coordinate dimensions do not match input rows")
    if not np.isfinite(P).all() or not np.isfinite(N).all():
        raise ValueError(f"{tid}: projected coordinates contain non-finite values")
    if df[FEATURE_COLUMNS].shape != (len(df), len(FEATURE_COLUMNS)):
        raise ValueError(f"{tid}: feature matrix has unexpected dimensions")
    values = df[FEATURE_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        bad = np.argwhere(~np.isfinite(values))[0]
        raise ValueError(
            f"{tid}: non-finite feature at candidate row {bad[0]}, "
            f"column {FEATURE_COLUMNS[bad[1]]}"
        )
    if len(df["y"]) != len(values) or not df["y"].isin([0, 1]).all():
        raise ValueError(f"{tid}: candidate feature and label dimensions do not match")
    if len(df) and not df["pt"].between(0, len(P) - 1).all():
        raise ValueError(f"{tid}: candidate point index is out of bounds")

def build_trip(split, tid):
    trk, nod, arc, rte = load_trip(split, tid)
    P, N = project(trk, nod)
    ids = crop_arcs(P, N, arc)
    df = candidates(P, N, arc, ids)
    df = add_features(df, P, trk.t.values)
    df = add_labels(df, rte)
    df["trip"] = tid
    validate_trip(trk, nod, P, N, df, tid)
    labeled_pts = int(df.groupby("pt")["y"].max().sum()) if len(df) else 0
    candidate_misses = len(P) - labeled_pts
    pts_no_cand = len(P) - df["pt"].nunique()
    stats = dict(trip=tid, n_pts=len(P), arcs_kept=len(ids),
                 mean_cand=len(df) / len(P),
                 pts_no_cand=pts_no_cand,
                 candidate_miss_pts=candidate_misses,
                 miss_rate=candidate_misses / len(P))
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
