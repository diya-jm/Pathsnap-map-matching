"""Laptop 1. Set PATHSNAP_TRIPS to the directory holding the ten trip folders."""
import logging
import os
from pathlib import Path
from time import perf_counter
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pathsnap import read_trip,predict
from reference import labels,continuity,sequence_score

app=FastAPI(title="PathSnap")
app.add_middleware(CORSMiddleware,allow_origins=["*"],allow_methods=["GET"],allow_headers=[])
root=Path(os.environ.get("PATHSNAP_TRIPS","data/Student folder/student_test"))
model=joblib.load("artifacts/model.joblib")
logger=logging.getLogger("uvicorn.error")

def folders():
    if not root.is_dir():
        raise HTTPException(503,"Trip directory is unavailable")
    return {p.name:p for p in root.iterdir() if p.is_dir() and (p/(p.name+".track")).is_file()}

@app.get("/trips")
def trips():
    return {"trips":sorted(folders())}

@app.get("/match/{trip_id}")
def match(trip_id:str):
    folder=folders().get(trip_id)
    if folder is None:
        raise HTTPException(404,"Unknown trip")
    start=perf_counter()
    t=read_trip(folder)
    pred=predict(model,t)
    metrics={"continuity":float(continuity(t["arcs"],pred))}
    if (folder/(folder.name+".route")).is_file():
        route,y=labels(folder,t)
        metrics.update(top1=float(np.mean(y==pred)),sequence=sequence_score(pred,route))
    unique=pred[np.r_[True,pred[1:]!=pred[:-1]]]
    segments=t["nodes"][t["arcs"][unique]]
    seconds=perf_counter()-start
    logger.info("Matched trip %s | observations=%d | elapsed=%.3fs | scores=%s",trip_id,len(pred),seconds,metrics)
    return dict(trip=trip_id,observations=len(pred),seconds=seconds,metrics=metrics,
                gps=t["gps"][:,:2].tolist(),segments=segments.tolist(),predicted_arc_ids=pred.tolist())

