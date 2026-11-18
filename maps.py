from pathlib import Path
import folium
from branca.element import Element
import joblib
import numpy as np
from prepare import ROOT
from pathsnap import read_trip,predict
from reference import labels

def build(split,trip):
    folder=ROOT/split/trip
    t=read_trip(folder)
    pred=predict(joblib.load("artifacts/model.joblib"),t)
    route,_=labels(folder,t)
    gps=t["gps"][:,:2]
    m=folium.Map(tiles=None,location=gps.mean(0)[::-1].tolist(),prefer_canvas=True)
    m.default_js=[("leaflet","vendor/leaflet.js")]
    m.default_css=[("leaflet_css","vendor/leaflet.css")]
    lo,hi=gps.min(0),gps.max(0)
    margin=np.maximum((hi-lo)*.08,.001)
    segments=t["nodes"][t["arcs"]]
    keep=((segments.max(1)>=lo-margin)&(segments.min(1)<=hi+margin)).all(1)
    def roads(name,geometry,color,weight,opacity):
        group=folium.FeatureGroup(name=name)
        folium.PolyLine(geometry[:,:,::-1].tolist(),color=color,weight=weight,opacity=opacity).add_to(group)
        group.add_to(m)
    roads("Road network",segments[keep],"#7d8f84",1,.55)
    roads("Reference route",segments[route],"#9450a0",7,.8)
    roads("Predicted route",segments[np.unique(pred)],"#11724f",3,1)
    layer=folium.FeatureGroup(name="GPS observations")
    for latlon in gps[:,::-1]:
        folium.CircleMarker(latlon.tolist(),radius=1.4,color="#dd762f",weight=1,
                            fill=True,fill_opacity=.9).add_to(layer)
    layer.add_to(m)
    m.fit_bounds([lo[::-1].tolist(),hi[::-1].tolist()])
    folium.LayerControl(collapsed=False).add_to(m)
    m.get_root().html.add_child(Element('<div style="position:fixed;bottom:20px;left:16px;z-index:999;background:white;padding:12px;font:14px Arial"><b>Trip '+trip+'</b><br><span style="color:#9450a0">Purple · Reference</span><br><span style="color:#11724f">Green · Prediction</span><br><span style="color:#bc5618">Orange · GPS</span></div>'))
    m.save(str(Path("maps")/(split+"_"+trip+".html")))

if __name__=="__main__":
    build("student_test","00000007")

