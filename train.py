"""Run five baselines, group-wise validation and Gaussian-process Bayesian search."""
import csv
import json
from pathlib import Path
from time import perf_counter
import joblib
import numpy as np
from scipy.stats import norm
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, log_loss
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from pathsnap import viterbi, FEATURES
from reference import sequence_score
from prepare import CACHE, RESULTS

SEED=138
MODELS={
    "Logistic":make_pipeline(StandardScaler(),LogisticRegression(max_iter=300)),
    "GaussianNB":make_pipeline(StandardScaler(),GaussianNB()),
    "DecisionTree":DecisionTreeClassifier(max_depth=9,min_samples_leaf=20,random_state=SEED),
    "RandomForest":RandomForestClassifier(n_estimators=40,max_depth=12,min_samples_leaf=5,n_jobs=2,random_state=SEED),
    "HistBoost":HistGradientBoostingClassifier(max_iter=90,learning_rate=.1,random_state=SEED,early_stopping=False)}
SEARCH={"Logistic":("logisticregression__C",np.geomspace(.001,100,61)),
        "GaussianNB":("gaussiannb__var_smoothing",np.geomspace(1e-11,1e-2,61)),
        "DecisionTree":("max_depth",np.arange(3,24)),
        "RandomForest":("max_depth",np.arange(4,25)),
        "HistBoost":("learning_rate",np.geomspace(.02,.3,61))}

def save(name,rows):
    with (RESULTS/name).open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=rows[0]); w.writeheader(); w.writerows(rows)

def load(split):
    trips=[]
    for file in sorted((CACHE/split).glob("*.npz")):
        with np.load(file) as data:
            t=dict(data)
        t["name"]=file.stem
        trips.append(t)
    return trips

def arrays(trips):
    rng=np.random.default_rng(SEED)
    xs,ys=[],[]
    for t in trips:
        idx=np.sort(rng.choice(len(t["x"]),min(250,len(t["x"])),replace=False))
        x=t["x"][idx]
        xs.append(x.reshape(-1,x.shape[-1]))
        ys.append((t["ids"][idx]==t["y"][idx,None]).ravel())
    return np.concatenate(xs),np.concatenate(ys)

def evaluate(model,trips,decoder=False):
    cm=np.zeros((2,2),int)
    total=hits=misses=loss=elapsed=0
    rows=[]
    for t in trips:
        x,ids,y=t["x"],t["ids"],t["y"]
        truth=ids==y[:,None]
        start=perf_counter()
        prob=model.predict_proba(x.reshape(-1,x.shape[-1]))[:,1].reshape(ids.shape)
        ix=np.argmax(prob,axis=1)
        pred=ids[np.arange(len(y)),ix]
        elapsed+=perf_counter()-start
        cm+=confusion_matrix(truth.ravel(),(prob>=.5).ravel(),labels=[False,True])
        loss+=log_loss(truth.ravel(),prob.ravel(),labels=[False,True],normalize=False)
        total+=len(y); hits+=int((pred==y).sum()); misses+=int((~truth.any(1)).sum())
        def score(p):
            index=np.argmax(ids==p[:,None],axis=1)
            e=t["edges"][np.arange(len(y)),index]
            continuity=np.mean((p[1:]==p[:-1])|(e[:-1,1]==e[1:,0])) if len(y)>1 else 1.
            return [float(np.mean(p==y)),float(continuity),float(sequence_score(p,t["route"]))]
        raw=score(pred)
        row=dict(trip=t["name"],n=len(y),raw_top1=raw[0],raw_continuity=raw[1],raw_sequence=raw[2])
        if decoder:
            start=perf_counter()
            path=viterbi(ids,t["edges"],t["times"],prob)
            row["decode_s"]=perf_counter()-start
            row["hmm_top1"],row["hmm_continuity"],row["hmm_sequence"]=score(path)
        rows.append(row)
    tn,fp,fn,tp=map(int,cm.ravel())
    f1=lambda a,b,c: 2*a/max(1,2*a+b+c)
    return dict(macro_f1=(f1(tp,fp,fn)+f1(tn,fp,fn))/2,precision=tp/max(tp+fp,1),
           recall=tp/max(tp+fn,1),accuracy=(tp+tn)/cm.sum(),top1=hits/total,
           miss_rate=misses/total,log_loss=loss/cm.sum(),continuity=np.mean([r["raw_continuity"] for r in rows]),
           sequence=np.mean([r["raw_sequence"] for r in rows]),inference_ms_per_obs=1000*elapsed/total,
           tn=tn,fp=fp,fn=fn,tp=tp),rows

def main():
    train,test=load("train"),load("student_test")
    assert len(train)==80 and len(test)==10
    if any(t["x"].shape[-1] != len(FEATURES) for t in train + test):
        raise ValueError(f"Prepared caches must contain the {len(FEATURES)} required features; rerun `python prepare.py`.")
    assignment_path = RESULTS / "review2_fold_assignments.csv"
    if not assignment_path.exists():
        raise FileNotFoundError("Run `python prepare.py` first to create the shared Review 2 fold assignments.")
    fold_by_trip = {}
    with assignment_path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            trip_id = row["trip"]
            if trip_id in fold_by_trip:
                raise ValueError(f"Trip {trip_id} appears more than once in the fold manifest")
            fold_by_trip[trip_id] = int(row["validation_fold"])
    train_names = {t["name"] for t in train}
    test_names = {t["name"] for t in test}
    if set(fold_by_trip) != train_names or set(fold_by_trip) & test_names:
        raise ValueError("Review 2 fold manifest must cover training trips only, exactly once")
    folds=[]
    for fold_id in range(1,4):
        val=[t for t in train if fold_by_trip[t["name"]] == fold_id]
        fit=[t for t in train if fold_by_trip[t["name"]] != fold_id]
        fit_names={t["name"] for t in fit}
        val_names={t["name"] for t in val}
        if not val or fit_names & val_names:
            raise ValueError(f"Invalid trip-grouped fold {fold_id}")
        folds.append((arrays(fit),val))
    all_x,all_y=arrays(train)
    summary=[]; fold_rows=[]
    def cross(model,label,trial=False):
        records=[]
        for i,((x,y),val) in enumerate(folds):
            m=clone(model).fit(x,y)
            if trial:
                hits=count=0
                cm=np.zeros((2,2),int)
                for t in val:
                    features=t["x"]
                    ids=t["ids"]
                    prob=m.predict_proba(features.reshape(-1,features.shape[-1]))[:,1].reshape(ids.shape)
                    hits+=int(np.sum(ids[np.arange(len(ids)),prob.argmax(axis=1)]==t["y"]))
                    count+=len(ids)
                    cm+=confusion_matrix((ids==t["y"][:,None]).ravel(),(prob>=.5).ravel(),labels=[False,True])
                tn,fp,fn,tp=cm.ravel()
                metrics=dict(top1=hits/count,macro_f1=tp/max(1,2*tp+fp+fn)+tn/max(1,2*tn+fp+fn))
            else:
                metrics,_=evaluate(m,val)
            records.append(metrics)
            fold_rows.append(dict(model=label,fold=i+1,**metrics))
        return np.mean([r["top1"] for r in records]),np.mean([r["macro_f1"] for r in records])
    def experiment(name,model):
        cv,f1=cross(model,name)
        start=perf_counter()
        model.fit(all_x,all_y)
        seconds=perf_counter()-start
        metrics,_=evaluate(model,test)
        training,_=evaluate(model,train)
        summary.append(dict(model=name,cv_top1=cv,cv_macro_f1=f1,train_top1=training["top1"],
                       train_macro_f1=training["macro_f1"],fit_s=seconds,**metrics))
        save("models.csv",summary); save("folds.csv",fold_rows)
        print(name,cv,metrics["top1"],flush=True)
    for name,model in MODELS.items():
        experiment(name,model)
    best=max(summary,key=lambda r:r["cv_top1"])["model"]
    key,grid=SEARCH[best]
    loc=np.linspace(0,1,len(grid))
    chosen=[]; objective=[]; trials=[]
    for i in range(8):
        if i<3:
            j=[0,len(grid)//2,len(grid)-1][i]
        else:
            gp=GaussianProcessRegressor(kernel=Matern(length_scale=.25,nu=2.5),alpha=.005,
                                        optimizer=None,normalize_y=True).fit(loc[chosen,None],objective)
            mu,sd=gp.predict(loc[:,None],return_std=True)
            delta=mu-max(objective); z=delta/np.maximum(sd,1e-12)
            acquisition=delta*norm.cdf(z)+sd*norm.pdf(z)
            acquisition[chosen]=-np.inf
            j=int(np.argmax(acquisition))
        value=grid[j].item()
        m=clone(MODELS[best]).set_params(**{key:value})
        start=perf_counter()
        score,f1=cross(m,"trial_"+str(i+1),trial=True)
        chosen.append(j); objective.append(score)
        trials.append(dict(trial=i+1,parameter=key,value=value,cv_top1=score,cv_macro_f1=f1,seconds=perf_counter()-start))
        save("bayesian_trials.csv",trials)
    value=grid[chosen[np.argmax(objective)]].item()
    tuned=clone(MODELS[best]).set_params(**{key:value})
    experiment(best+"_tuned",tuned)
    _,rows=evaluate(tuned,test,True); save("decoder.csv",rows)
    Path("artifacts").mkdir(exist_ok=True)
    joblib.dump(tuned,"artifacts/model.joblib",compress=3)
    Path("artifacts/config.json").write_text(json.dumps(dict(model=best,parameter=key,value=value,
       seed=SEED,search_min=grid.min().item(),search_max=grid.max().item(),candidates=8,
       gap_seconds=15,disconnection_penalty=3,features=FEATURES,n_features=len(FEATURES),
       validation_folds=3),indent=2))
if __name__=="__main__":
    main()

