"""Trip-local road geometry and prediction. Reference files are never opened here."""
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

K = 8
FEATURES = ["distance_m", "excess_distance_m", "heading_cosine", "speed_mps",
            "arc_length_m", "fraction_along", "gap_seconds", "heading_available"]

def read_trip(folder):
    def read(ext, dtype):
        return np.loadtxt(folder / (folder.name + "." + ext), dtype=dtype, ndmin=2)
    gps, nodes, arcs = read("track", float), read("nodes", float), read("arcs", int)
    if not np.isfinite(gps).all() or not np.isfinite(nodes).all():
        raise ValueError("Non-finite coordinates or times")
    if arcs.min() < 0 or arcs.max() >= len(nodes):
        raise ValueError("Arc endpoint outside node table")
    scale = np.array([np.cos(np.radians(nodes[:,1].mean())), 1]) * 111195
    origin = nodes.mean(axis=0)
    xy, road = (gps[:,:2]-origin)*scale, (nodes-origin)*scale
    a = road[arcs[:,0]]
    v = road[arcs[:,1]] - a
    return dict(gps=gps, nodes=nodes, arcs=arcs, xy=xy, a=a, v=v,
                length=np.linalg.norm(v, axis=1))

def project(trip, points, ids):
    starts, vectors = trip["a"][ids], trip["v"][ids]
    offset = points - starts
    fraction = np.clip((offset*vectors).sum(-1) /
                       np.maximum(trip["length"][ids]**2, 1e-9), 0, 1)
    distance = np.linalg.norm(offset-fraction[...,None]*vectors, axis=-1)
    return distance, fraction

def candidates(t):
    # Index points along every directed arc so long roads remain discoverable.
    count = np.maximum(1, np.ceil(t["length"]/25).astype(int))
    ids = np.repeat(np.arange(len(count)), count)
    local = np.arange(len(ids))-np.repeat(np.cumsum(count)-count, count)
    samples = t["a"][ids]+t["v"][ids]*((local+.5)/count[ids])[:,None]
    tree = cKDTree(samples)
    xy, gps = t["xy"], t["gps"]
    width = min(K, len(count))
    _, neighbors = tree.query(xy, k=min(64,len(ids)))
    neighbors = np.asarray(neighbors).reshape(len(xy),-1)
    pool = ids[neighbors]
    d, _ = project(t, xy[:,None,:], pool)
    order = np.argsort(d,axis=1,kind="stable")
    ranked = np.take_along_axis(pool,order,axis=1)
    chosen = np.empty((len(xy),width),int)
    for i,row in enumerate(ranked):
        unique = list(dict.fromkeys(row))
        if len(unique)<width:
            all_ids = np.arange(len(count))
            distances,_ = project(t,xy[i],all_ids)
            unique = np.argsort(distances,kind="stable")
        chosen[i] = unique[:width]
    dist, fraction = project(t, xy[:,None,:], chosen)
    # Centered heading suppresses jitter; the whole trip is available at inference.
    index = np.arange(len(xy))
    left,right = np.maximum(index-2,0),np.minimum(index+2,len(xy)-1)
    delta = xy[right]-xy[left]
    motion = np.linalg.norm(delta,axis=1)
    dt = gps[right,2]-gps[left,2]
    valid = (dt>0)&(dt<=15)&(motion>=3)
    cosine = (delta[:,None,:]*t["v"][chosen]).sum(-1)/np.maximum(motion[:,None]*t["length"][chosen],1e-9)
    broadcast = lambda arr: np.broadcast_to(arr[:,None],dist.shape)
    x = np.stack([dist,dist-dist.min(1,keepdims=True),cosine*valid[:,None],
                  broadcast(np.clip(motion/np.maximum(dt,1),0,80)),t["length"][chosen],
                  fraction,broadcast(np.r_[0,np.diff(gps[:,2])]),broadcast(valid)],axis=-1)
    return chosen, x.astype("float32")

def decode(t, ids, probability):
    return viterbi(ids, t["arcs"][ids], t["gps"][:,2], probability)

def viterbi(ids, states, times, probability):
    """Viterbi with soft disconnected transitions and resets across time gaps."""
    emission = np.log(np.clip(probability,1e-9,1))
    same = ids[:-1,:,None] == ids[1:,None,:]
    connected = states[:-1,:,1,None] == states[1:,None,:,0]
    costs = np.where(same | connected,0.,3.)
    n,k = ids.shape
    back = np.zeros((n,k),int)
    out = np.empty(n,int)
    score = emission[0].copy()
    gaps = np.r_[True,(np.diff(times)>15)|(np.diff(times)<=0)]
    for i in range(1,n):
        step = score[:,None] - costs[i-1]
        back[i] = np.argmax(step,axis=0)
        if gaps[i]:
            back[i] = np.argmax(score)
            score = emission[i].copy()
        else:
            score = emission[i]+step[back[i],np.arange(k)]
    out[-1] = np.argmax(score)
    for i in range(n-1,0,-1):
        out[i-1] = back[i,out[i]]
    return ids[np.arange(n),out]

def predict(model,t):
    ids,x = candidates(t)
    p = model.predict_proba(x.reshape(-1,x.shape[-1]))[:,1].reshape(ids.shape)
    return decode(t,ids,p)

