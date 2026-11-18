"""Reference-only logic used by training and by the server AFTER prediction."""
import numpy as np
from pathsnap import project

def labels(folder, t):
    route = np.loadtxt(folder/(folder.name+".route"),dtype=int,ndmin=1)
    if not len(route) or route.min()<0 or route.max()>=len(t["arcs"]):
        raise ValueError("Invalid reference route")
    # Monotone sequence alignment. Stay on, or advance along, the supplied route.
    # Prefix minima compute the cheapest preceding position in O(route length).
    n,m = len(t["gps"]),len(route)
    back = np.zeros((n,m),np.int32)
    state = np.arange(m)
    start = 0.05 * state
    for i,point in enumerate(t["xy"]):
        distance,_ = project(t,point,route)
        if i:
            minima = np.minimum.accumulate(cost)
            back[i] = np.maximum.accumulate(np.where(cost==minima,state,0))
            start = minima
        cost = start + distance
    j = np.argmin(cost+0.05*(m-1-state))
    label = np.empty(n,int)
    for i in range(n-1,-1,-1):
        label[i] = route[j]
        j = back[i,j]
    return route,label

def continuity(arcs, pred):
    return np.mean((pred[1:]==pred[:-1])|(arcs[pred[:-1],1]==arcs[pred[1:],0])) if len(pred)>1 else 1.

def sequence_score(pred, route):
    from difflib import SequenceMatcher
    compact=lambda x: x[np.r_[True,x[1:]!=x[:-1]]].tolist()
    return SequenceMatcher(None,compact(pred),compact(route),autojunk=False).ratio()

