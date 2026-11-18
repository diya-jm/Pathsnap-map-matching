from pathlib import Path
import csv
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from pathsnap import read_trip,candidates
from reference import labels
from review2_data import build as build_review2

ROOT=Path("data/Student folder")
if not ROOT.exists():
    ROOT=Path("pathsnap/data/raw")
CACHE=Path(".cache")
RESULTS=Path("results")

def prepare():
    RESULTS.mkdir(exist_ok=True)
    rows=[]
    for split in ["train","student_test"]:
        target=CACHE/split
        target.mkdir(exist_ok=True, parents=True)
        for folder in sorted((ROOT/split).iterdir()):
            t=read_trip(folder)
            ids,x=candidates(t)
            route,y=labels(folder,t)
            np.savez_compressed(target/(folder.name+".npz"),ids=ids,x=x,y=y,route=route,
                                edges=t["arcs"][ids],times=t["gps"][:,2])
            gps,nodes,arcs=t["gps"],t["nodes"],t["arcs"]
            gaps=np.diff(gps[:,2])
            graph=coo_matrix((np.ones(len(arcs)),arcs.T),shape=(len(nodes),len(nodes)))
            rows.append(dict(split=split,trip=folder.name,observations=len(gps),
                nodes=len(nodes),arcs=len(arcs),route_arcs=len(route),max_gap_s=gaps.max(initial=0),
                gaps_over_15=int((gaps>15).sum()),nonpositive_steps=int((gaps<=0).sum()),
                one_second_steps=int((gaps==1).sum()),nonfinite=int((~np.isfinite(gps)).sum()),
                min_lon=gps[:,0].min(),max_lon=gps[:,0].max(),min_lat=gps[:,1].min(),
                max_lat=gps[:,1].max(),isolated_nodes=len(nodes)-len(np.unique(arcs)),
                weak_components=connected_components(graph,directed=False,return_labels=False),
                candidate_misses=int((~(ids==y[:,None]).any(1)).sum())))
    with (RESULTS/"inventory.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
    build_review2()
if __name__=="__main__":
    prepare()

