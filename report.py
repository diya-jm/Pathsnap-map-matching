"""Regenerate the consolidated PDF from the development experiment outputs."""
import csv
import json
from io import BytesIO
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import joblib
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,PageBreak
from PIL import Image as Picture
from pathsnap import read_trip, predict
from reference import labels

def read(name):
    with open("results/"+name,newline="") as f:
        return list(csv.DictReader(f))
def pct(x):return f"{float(x)*100:.2f}%"
def num(x):return f"{float(x):.4f}"
def avg(rows,col):return np.mean([float(r[col]) for r in rows])
def pooled(rows,col):return np.average([float(r[col]) for r in rows],weights=[int(r["n"]) for r in rows])

def raw_root():
    preferred=Path("data/Student folder")
    return preferred if preferred.exists() else Path("pathsnap/data/raw")

def trajectory_figure():
    fig,axes=plt.subplots(1,2,figsize=(8,3.7))
    examples=[("train","00000054"),("student_test","00000077")]
    for ax,(split,trip_id) in zip(axes,examples):
        folder=raw_root()/split/trip_id
        t=read_trip(folder);gps=t["gps"][:,:2]
        ax.plot(gps[:,0],gps[:,1],color="#26795d",linewidth=1.25)
        ax.scatter(gps[0,0],gps[0,1],s=20,color="#26795d",label="start",zorder=3)
        ax.scatter(gps[-1,0],gps[-1,1],s=20,color="#c47530",label="end",zorder=3)
        ax.set_title(f"{split}: trip {trip_id}",fontsize=9)
        ax.set_xlabel("Longitude");ax.set_ylabel("Latitude")
        ax.set_aspect(1/np.cos(np.radians(gps[:,1].mean())),adjustable="datalim")
        ax.grid(alpha=.2);ax.legend(fontsize=7,loc="best")
    fig.suptitle("GPS trajectories (training and student-test partitions)",fontsize=11)
    return fig

def network_overview_figure():
    trip_id="00000033";folder=raw_root()/"train"/trip_id
    t=read_trip(folder);gps=t["gps"][:,:2];segments=t["nodes"][t["arcs"]]
    route,_=labels(folder,t)
    fig,ax=plt.subplots(figsize=(8,3.8))
    ax.add_collection(LineCollection(segments,colors="#c8d3cc",linewidths=.35,zorder=1))
    ax.add_collection(LineCollection(t["nodes"][t["arcs"][route]],colors="#9450a0",linewidths=1.6,zorder=2))
    ax.plot(gps[:,0],gps[:,1],color="#c47530",linewidth=.8,zorder=3)
    ax.set_xlim(gps[:,0].min()-.003,gps[:,0].max()+.003)
    ax.set_ylim(gps[:,1].min()-.003,gps[:,1].max()+.003)
    ax.set_aspect(1/np.cos(np.radians(gps[:,1].mean())),adjustable="box")
    ax.set_title(f"Training trip {trip_id}: supplied road network and aligned reference route")
    ax.set_xlabel("Longitude");ax.set_ylabel("Latitude")
    ax.legend(handles=[Line2D([0],[0],color="#c8d3cc",lw=2,label="Road network"),
                       Line2D([0],[0],color="#9450a0",lw=2,label="Reference route"),
                       Line2D([0],[0],color="#c47530",lw=2,label="GPS trajectory")],fontsize=8)
    ax.grid(alpha=.15)
    return fig

def mismatch_network_figure(padding=12):
    trip_id="00000077";folder=raw_root()/"student_test"/trip_id
    t=read_trip(folder);pred=predict(joblib.load("artifacts/model.joblib"),t)
    _,aligned=labels(folder,t);gps=t["gps"][:,:2]
    bad=np.flatnonzero(pred!=aligned)
    if len(bad):
        runs=np.split(bad,np.flatnonzero(np.diff(bad)>1)+1)
        run=max(runs,key=len);lo=max(0,int(run[0])-padding);hi=min(len(gps),int(run[-1])+padding+1)
    else:
        lo,hi=max(0,len(gps)//2-20),min(len(gps),len(gps)//2+20)
    window=gps[lo:hi];nodes=t["nodes"];arcs=t["arcs"]
    # Plot in a local metre coordinate system so the close-up has the correct
    # aspect ratio and does not collapse into a thin strip in degree units.
    lon0=float(window[:,0].mean());lat0=float(window[:,1].mean())
    def metres(xy):
        out=np.array(xy,copy=True,dtype=float)
        out[...,0]=(out[...,0]-lon0)*111320*np.cos(np.radians(lat0))
        out[...,1]=(out[...,1]-lat0)*111320
        return out
    window=metres(window);nodes=metres(nodes)
    network=nodes[arcs]
    margin_x=max((window[:,0].max()-window[:,0].min())*.25,.001)
    margin_y=max((window[:,1].max()-window[:,1].min())*.25,.001)
    x0,x1=window[:,0].min()-margin_x,window[:,0].max()+margin_x
    y0,y1=window[:,1].min()-margin_y,window[:,1].max()+margin_y
    keep=((network[:,:,0].min(1)<=x1)&(network[:,:,0].max(1)>=x0)&
          (network[:,:,1].min(1)<=y1)&(network[:,:,1].max(1)>=y0))
    fig,ax=plt.subplots(figsize=(8,4.2))
    ax.add_collection(LineCollection(network[keep],colors="#c8d3cc",linewidths=.45,zorder=1))
    ref_seg=nodes[arcs[aligned[lo:hi]]]
    pred_seg=nodes[arcs[pred[lo:hi]]]
    ax.add_collection(LineCollection(ref_seg,colors="#9450a0",linewidths=2.4,alpha=.85,zorder=2))
    ax.add_collection(LineCollection(pred_seg,colors="#16734f",linewidths=1.35,alpha=.95,zorder=3))
    local_bad=pred[lo:hi]!=aligned[lo:hi]
    ax.scatter(window[:,0],window[:,1],s=7,color="#d7792b",zorder=4)
    if local_bad.any():
        ax.scatter(window[local_bad,0],window[local_bad,1],s=22,marker="x",color="#b32929",zorder=5)
    ax.set_xlim(x0,x1);ax.set_ylim(y0,y1)
    ax.set_aspect("equal",adjustable="box")
    ax.set_xlabel("East (m)");ax.set_ylabel("North (m)")
    ax.set_title(f"Trip {trip_id}: road context near a prediction/reference disagreement (points {lo}-{hi-1})")
    handles=[Line2D([0],[0],color="#c8d3cc",lw=2,label="Road network"),
             Line2D([0],[0],color="#9450a0",lw=2,label="Aligned reference"),
             Line2D([0],[0],color="#16734f",lw=2,label="Predicted road"),
             Line2D([0],[0],marker="x",color="#b32929",lw=0,label="Disagreement")]
    ax.legend(handles=handles,fontsize=8,loc="best");ax.grid(alpha=.15)
    return fig,lo,hi,int(local_bad.sum()),int(len(window))

def network_summary(inv):
    rows=[]
    for split in ("train","student_test"):
        trips=[r for r in inv if r["split"]==split]
        components=[int(r["weak_components"]) for r in trips]
        rows.append([split,f"{sum(int(r['nodes']) for r in trips):,}",
            f"{sum(int(r['arcs']) for r in trips):,}",
            f"{sum(int(r['isolated_nodes']) for r in trips):,}",
            f"{np.mean(components):.1f} (range {min(components)}-{max(components)})"])
    return rows

def main():
    inv=read("inventory.csv"); models=read("models.csv"); trials=read("bayesian_trials.csv")
    dec=read("decoder.csv"); cfg=json.loads(Path("artifacts/config.json").read_text())
    baseline=next(r for r in models if r["model"]==cfg["model"]); tuned=models[-1]
    story=[]
    title=ParagraphStyle("title",fontName="Helvetica-Bold",fontSize=25,leading=29,textColor="#153d31",spaceAfter=14)
    h=ParagraphStyle("section",fontName="Helvetica-Bold",fontSize=13,leading=16,textColor="#1b5844",spaceBefore=9,spaceAfter=5)
    body=ParagraphStyle("body",fontName="Helvetica",fontSize=10,leading=14,spaceAfter=7,textColor="#253c34")
    def p(s):story.append(Paragraph(s,body))
    def heading(s):story.append(Paragraph(s,h))
    def page(s):
        if story:story.append(PageBreak())
        story.append(Paragraph(s,title))
    def table(rows,widths=None,fontsize=8):
        t=Table(rows,colWidths=widths,hAlign="LEFT",repeatRows=1)
        t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),"#214c3c"),("TEXTCOLOR",(0,0),(-1,0),colors.white),
             ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("FONTSIZE",(0,0),(-1,-1),fontsize),
             ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#edf3ee")]),
             ("VALIGN",(0,0),(-1,-1),"MIDDLE"),("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5)]))
        story.extend([t,Spacer(1,8)])
    def figure(fig,height):
        fig.tight_layout()
        buf=BytesIO();fig.savefig(buf,format="png",dpi=145,bbox_inches="tight");plt.close(fig);buf.seek(0)
        im=Image(buf);s=min(490/im.imageWidth,height/im.imageHeight)
        im.drawWidth=im.imageWidth*s;im.drawHeight=im.imageHeight*s;story.append(im)
    def photo(path,maxh):
        if not Path(path).is_file():return
        with Picture.open(path) as pic:w,hh=pic.size
        scale=min(490/w,maxh/hh);story.append(Image(path,width=w*scale,height=hh*scale))
    page("PathSnap | Reconstructing vehicle routes")
    p("Machine Learning Hackathon / UE24CS352A / PES University")
    heading("Development result")
    p(f"Across the ten student-test trips, the tuned HistBoost classifier achieved {pct(tuned['top1'])} point Top-1 and Viterbi decoding achieved {pct(pooled(dec,'hmm_top1'))}. Mean per-trip continuity changed from {pct(avg(dec,'raw_continuity'))} to {pct(avg(dec,'hmm_continuity'))}. These are development results, not scores on a hidden evaluator partition.")
    heading("Data inventory and partition")
    rows=[["Partition","Trips","GPS fixes","Missing GPS values","Gaps >15 s","1 s intervals"]]
    for split in ["train","student_test"]:
        sub=[r for r in inv if r["split"]==split]
        total=lambda k:sum(int(r[k]) for r in sub)
        rows.append([split,len(sub),f"{total('observations'):,}",total("nonfinite"),total("gaps_over_15"),pct(total("one_second_steps")/(total("observations")-len(sub)))])
    table(rows)
    p("The raw inventory has 80 training trips and 10 student-test trips. All listed tracks, node tables, arc tables and routes were readable; track coordinates and times were finite, and timestamps increased. Student-test routes are available for development diagnostics only. The hidden evaluator trip IDs were not present in this checkout and are not used in these analyses.")
    heading("Road-network statistics by partition")
    net_rows=[["Partition","Nodes","Directed arcs","Isolated nodes","Weak components per trip"]]
    net_rows.extend(network_summary(inv));table(net_rows)
    p("Node and arc totals sum trip-local networks, so the same physical location may be counted more than once across trips. Weak components are computed per trip with direction ignored, then summarized by the mean and range. No isolated nodes were found. Per-trip extents and component counts are in results/inventory.csv.")
    figure(trajectory_figure(),205)
    p("GPS trajectories: training trip 00000054 and student-test trip 00000077. These are plotted in longitude and latitude; the latter is also the weakest student-test trip by point Top-1.")
    figure(network_overview_figure(),220)
    p("Road-network visualization for training trip 00000033. Grey lines are the supplied directed road network, purple lines show the point-aligned reference route, and orange shows the GPS trajectory. This is a local graph plot, separate from the interactive Folium map.")
    heading("Sampling and error concentration")
    p("The training partition has 16,395 candidate misses (7.86% of its 208,417 fixes); the student-test partition has 62 (0.32% of 19,160). Training trip 00000054 contributes 10,551 misses and trip 00000033 contributes 3,889. Trip 00000054 is held out in fold 1 and trip 00000033 in fold 3, so these difficult trips contribute to low held-out fold scores. In particular, HistBoost fold Top-1 is 62.39%, 87.75% and 82.95% in folds 1-3. This miss-rate and trip-composition difference helps explain why student-test Top-1 is higher; it does not prove that this is the only cause. Each fit samples at most 250 GPS observations per training trip and retains their candidate rows; evaluation uses every validation and student-test observation.")
    page("From GPS points to road candidates")
    heading("Reference alignment")
    p("Each track row gives longitude, latitude and time. An arc identifies a directed pair of node rows. A route is an ordered list of traversed arc IDs without timestamps. Reference alignment uses dynamic programming to choose a nondecreasing sequence of route positions minimizing total point-to-segment distance, with a small endpoint bias. A point may stay on a segment or skip ahead. This yields approximate per-point labels. Repeated roads, close parallel segments and long stationary periods can leave the alignment ambiguous.")
    heading("Candidates and features")
    p("Coordinates use a trip-local equirectangular approximation in metres. The longitude scale uses mean latitude. A spatial index samples each directed segment at intervals of at most 25 m. For each GPS point, 64 nearby samples propose roads, which are reranked by exact point-to-segment distance. Up to eight distinct directed roads are retained. If the pool contains too few roads, the search examines all arcs. The sampled search can miss a close arc in dense networks.")
    table([["Feature columns","What they tell the model"],
        ["dist; d_minus_best","How far the GPS fix is from this road, and how much farther it is than the closest candidate."],
        ["t; alen","Where the projected fix falls along the segment, and the segment's length."],
        ["speed; gap","Approximate GPS speed and time since the previous observation."],
        ["cos_h; dh_deg","Agreement between GPS movement direction and the directed road, as cosine and degrees."],
        ["n_cand; rank","How many nearby candidate roads were found and this road's distance rank."]])
    p("These are the ten columns in X, in order: dist, t, alen, speed, gap, cos_h, dh_deg, n_cand, rank, d_minus_best. A centred five-observation window provides heading and approximate speed. Heading is suppressed below 3 m displacement or when the window exceeds 15 s; an unavailable heading is encoded as cosine 0 and angular difference 90 degrees. The candidate count uses distinct roads in the nearby indexed search pool. All learned scaling is fitted inside each training fold. Candidate generation reads track and network files only.")
    heading("Experimental protocol")
    p("Three group-aware folds hold out entire trips from the 80-trip training set. The fitting cap applies in each training fold, using seed 138. All ten student-test trips are evaluated separately. Candidate-level precision, recall and macro F1 use a 0.5 probability threshold. Observation Top-1 uses the largest score among each point's candidates, with misses counted as errors. The primary selection metric is mean fold Top-1 because the deployed system must pick one road per observation.")
    p("Continuity counts consecutive predictions on the same arc or with matching directed endpoints. Skipping a short connecting arc between fixes can lower continuity even when the vehicle travelled a valid path. Sequence similarity uses matching blocks from Python SequenceMatcher on consecutive-deduplicated predicted and reference arc sequences. Continuity and sequence results are averaged per trip.")
    page("3.1 Consolidated five-model comparison")
    baseline_models=models[:5]
    cv_ranked=sorted(baseline_models,key=lambda r:(float(r["cv_top1"]),float(r["cv_macro_f1"])),reverse=True)
    selected=cv_ranked[0];runner=cv_ranked[1]
    cv_gap=float(selected["cv_top1"])-float(runner["cv_top1"])
    families={"Logistic":"Linear probabilistic","GaussianNB":"Generative probabilistic",
              "DecisionTree":"Single decision tree","RandomForest":"Bagged tree ensemble",
              "HistBoost":"Gradient boosting"}
    configurations={
      "Logistic":"StandardScaler; C=1; L2; max_iter=300",
      "GaussianNB":"StandardScaler; var_smoothing=1e-9",
      "DecisionTree":"Gini; max_depth=9; min_samples_leaf=20",
      "RandomForest":"40 trees; sqrt features; depth=12; leaf=5",
      "HistBoost":"90 iterations; learning_rate=0.1; max_leaves=31; no early stop"}
    small=ParagraphStyle("smallcell",fontName="Helvetica",fontSize=6.7,leading=8)
    headers=[Paragraph(s.replace(" ","<br/>"),small) for s in
             ["Model","Algorithm family","Key configuration","Mean CV Top-1","CV Macro F1",
              "Student Top-1","Route continuity","Student Macro F1","Log loss","Inference ms/fix"]]
    model_rows=[headers]
    for r in baseline_models:
        model_rows.append([Paragraph(r["model"],small),Paragraph(families[r["model"]],small),
            Paragraph(configurations[r["model"]],small),pct(r["cv_top1"]),num(r["cv_macro_f1"]),
            pct(r["top1"]),pct(r["continuity"]),num(r["macro_f1"]),num(r["log_loss"]),
            f"{float(r['inference_ms_per_obs']):.6f}"])
    widths=[42,52,90,42,42,44,44,46,38,42]
    table(model_rows,widths,fontsize=6.5)
    p("Selection rule: choose the highest arithmetic mean of observation Top-1 across the three trip-grouped validation folds. Student-test metrics are reported after selection and are not used to choose the baseline. If CV Top-1 ties, CV macro F1 is the tie-breaker. HistBoost is selected. RandomForest is runner-up: mean CV Top-1 is 77.6967% versus 77.5898%, a gap of 0.10696 percentage points. The gap is small relative to the fold spread; the ranking is not evidence of statistical significance.")
    p("CV metrics are the unweighted mean across three folds; student-test metrics pool the ten trips' candidate/observation results as defined in train.py. Inference time is predict_proba time per GPS fix on this machine and excludes loading, feature preparation, HMM decoding and network transport.")
    fig,axes=plt.subplots(1,3,figsize=(8.0,3.5));names=[r["model"] for r in baseline_models];ix=np.arange(len(names));w=.34
    axes[0].bar(ix-w/2,[float(r["cv_top1"]) for r in baseline_models],w,label="Grouped CV",color="#9bb8a7")
    axes[0].bar(ix+w/2,[float(r["top1"]) for r in baseline_models],w,label="Student test",color="#26795d")
    axes[0].set_title("Top-1 accuracy");axes[0].set_ylabel("Accuracy");axes[0].set_ylim(0,1);axes[0].legend(fontsize=7)
    axes[1].bar(ix-w/2,[float(r["cv_macro_f1"]) for r in baseline_models],w,label="Grouped CV",color="#9bb8a7")
    axes[1].bar(ix+w/2,[float(r["macro_f1"]) for r in baseline_models],w,label="Student test",color="#26795d")
    axes[1].set_title("Macro F1");axes[1].set_ylabel("Macro F1");axes[1].set_ylim(0,1);axes[1].legend(fontsize=7)
    axes[2].bar(ix,[float(r["log_loss"]) for r in baseline_models],color="#496a83")
    axes[2].set_title("Student-test log loss");axes[2].set_ylabel("Log loss (lower is better)")
    for ax in axes:
        ax.set_xticks(ix,names,rotation=45,ha="right");ax.grid(axis="y",alpha=.2)
    figure(fig,225)
    page("3.1 Fit and validation diagnostics")
    fold_rows=read("folds.csv")
    table([["Model","Precision","Recall","Fit (s)","Train Top-1","Mean CV Top-1","Fold Top-1 range"]]+
      [[r["model"],num(r["precision"]),num(r["recall"]),f"{float(r['fit_s']):.2f}",pct(r["train_top1"]),pct(r["cv_top1"]),
        f"{min(float(a['top1']) for a in fold_rows if a['model']==r['model'])*100:.2f}-{max(float(a['top1']) for a in fold_rows if a['model']==r['model'])*100:.2f}%"] for r in baseline_models])
    p("Precision and recall are candidate-level positive-class measures at probability threshold 0.5. The fold range is the minimum-to-maximum held-out Top-1. The mean CV Top-1 values are 76.29% (Logistic), 75.94% (GaussianNB), 76.86% (DecisionTree), 77.59% (RandomForest), and 77.70% (HistBoost). The lowest fold scores are around 61-62%; folds containing trip 00000054 and trip 00000033 have much higher miss rates than the student-test trips.")
    page("3.2 Model-by-model analysis")
    notes={
      "Logistic":("Logistic Regression","A linear probabilistic baseline is useful for checking whether distance, heading and motion cues combine additively. StandardScaler is fitted inside each fold; C=1.0 applies L2 regularisation, and max_iter=300 is the initial solver limit. The objective is regularised binary cross-entropy (logistic loss). Strengths are speed, a compact model and smooth scores. Its linear boundary cannot represent many interactions such as a distance threshold that changes with heading.","Train/CV Top-1 are 76.30%/76.29%, with essentially no train-validation gap; this suggests limited fit rather than memorization. Student test: Top-1 85.25%, macro F1 0.8996, log loss 0.1211, inference 0.000576 ms/fix."),
      "GaussianNB":("Gaussian Naive Bayes","A fast generative baseline models each feature with a class-conditional Gaussian and combines features under conditional independence. StandardScaler is used; var_smoothing=1e-9 is the initial numerical stabilizer. Its objective is the Gaussian likelihood and resulting Bayes posterior, not a discriminative cross-entropy fit. Strengths are speed and high positive recall. Correlated features such as dist and d_minus_best violate independence and can produce poorly calibrated scores.","Train/CV Top-1 are 76.00%/75.94%, showing little train-validation gap and weak candidate ranking. Student test: Top-1 84.42%, macro F1 0.8092, log loss 0.5102, inference 0.001593 ms/fix. The large log loss and many false positives are evidence of poor probability quality."),
      "DecisionTree":("Decision Tree","A shallow tree can express nonlinear thresholds and feature interactions while remaining inspectable. Initial settings are Gini impurity, max_depth=9 and min_samples_leaf=20. Each split greedily reduces Gini impurity; leaf probabilities are class frequencies. Strengths are simple rules and low inference cost. A single tree is sensitive to sample changes and can fit trip-specific patterns.","Train/CV Top-1 are 80.57%/76.86%, a 3.71 percentage-point gap consistent with overfitting. Student test: Top-1 86.85%, macro F1 0.9213, log loss 0.0988, inference 0.000536 ms/fix."),
      "RandomForest":("Random Forest","Bootstrap aggregation of 40 Gini trees (max_depth=12, min_samples_leaf=5, max_features=sqrt, n_jobs=2) reduces the variance of a single tree. The objective is per-tree Gini impurity reduction; class probabilities are averaged across trees. Strengths are robust nonlinear interactions and stronger probabilities than the single tree. Limits are larger model size and slower inference than linear or shallow-tree baselines.","Train/CV Top-1 are 81.38%/77.59%, a 3.79 percentage-point gap consistent with some overfitting. Student test: Top-1 87.15%, macro F1 0.9238, log loss 0.0816, inference 0.015854 ms/fix."),
      "HistBoost":("Histogram Gradient Boosting","Boosted shallow trees can correct residual errors across distance, heading and motion features. Initial settings are 90 iterations, learning_rate=0.1, max_leaf_nodes=31 and early_stopping=False. The objective is binary log loss. Strengths are the best mean grouped-CV Top-1 among the five baselines, fast inference, and useful probability quality. Limits include sensitivity to hyperparameters and a train-validation gap.","Train/CV Top-1 are 81.63%/77.70%, a 3.93 percentage-point gap that is evidence of a generalization gap. Student test: Top-1 87.22%, macro F1 0.9251, log loss 0.0805, inference 0.008546 ms/fix. Its CV Top-1 lead over RandomForest is only 0.107 percentage points.")}
    for r in baseline_models:
        name,why,results_text=notes[r["model"]]
        heading(name);p(why);p(results_text)
    page("Bayesian optimisation")
    p("The search varied the boosting learning rate from 0.02 to 0.30 on a logarithmic grid. This explores the size of each corrective step with the tree budget held fixed. Three initial evaluations sampled the endpoints and centre. A Gaussian-process surrogate with a Matern kernel selected five further untried values by expected improvement on mean grouped-CV Top-1.")
    table([["Trial","Learning rate","CV Top-1","CV macro F1","Elapsed (s)"]]+[[r["trial"],num(r["value"]),pct(r["cv_top1"]),num(r["cv_macro_f1"]),f"{float(r['seconds']):.1f}"] for r in trials])
    p(f"Search cost totalled {sum(float(r['seconds']) for r in trials):.1f} s. The selected learning rate was {cfg['value']:.5f}. The final model was refitted on observations sampled across all 80 training trips with the same cap and seed. This is a narrow one-parameter search: max iterations, leaf count, minimum leaf size and L2 regularisation were held fixed, so it does not identify a joint optimum.")
    table([["Configuration","CV Top-1","CV macro F1","Test Top-1","Test macro F1","Log loss","ms/fix"]]+
      [[name,pct(r["cv_top1"]),num(r["cv_macro_f1"]),pct(r["top1"]),num(r["macro_f1"]),num(r["log_loss"]),num(r["inference_ms_per_obs"])] for name,r in [("Baseline",baseline),("Tuned",tuned)]])
    p(f"Tuning changed mean CV Top-1 from {pct(baseline['cv_top1'])} to {pct(tuned['cv_top1'])} (+0.089 percentage points), while CV macro F1 fell from {num(baseline['cv_macro_f1'])} to {num(tuned['cv_macro_f1'])}. Student-test Top-1 rose only 0.047 percentage points, macro F1 fell slightly, log loss improved from 0.08051 to 0.08027, and inference time changed from 0.008546 to 0.008626 ms/fix. The changes are small compared with the fold-to-fold Top-1 spread (about 25.4 points for this model); no significance test was run, so the apparent gains may be noise and should not be presented as a reliable improvement.")
    page("Sequential route reconstruction")
    p("The hidden state is one of the eight directed road candidates at a GPS fix. Each emission is the log of the classifier's positive-class score, clipped away from zero. Same-arc and shared-directed-endpoint transitions receive score 0; other transitions receive -3. The -3 value is a soft disconnection penalty (relative factor exp(-3), about 0.050), not a forbidden transition; the candidate sampler may omit an intermediate road. A time gap greater than 15 seconds resets the chain so evidence before a long outage does not constrain the next segment.")
    p("The 15-second reset matches the gap threshold used in the data audit: 337 of 208,417 training intervals and 15 of 19,160 student-test intervals exceed it. The -3 penalty was chosen as a moderate soft penalty to preserve a path through candidate-sampling omissions. These settings were chosen using development data only; the repository records the final settings but no decoder-parameter grid-search log, so they should be described as heuristic choices rather than a proven optimum. The student-test candidate-miss rate is "+pct(tuned["miss_rate"])+".")
    p("The recurrence is score(j,state) = emission(j,state) + max over previous states of [score(i) + transition(i,j)]. This is Viterbi-style max-score decoding, but not a fully normalized probabilistic HMM: candidate positive probabilities are not normalized across states, transition values are compatibility penalties rather than normalized transition probabilities, and no explicit start-state distribution is used. The resulting path is the maximum of these unnormalized scores.")
    p("The active candidate generator fills up to eight states per fix by falling back to the nearest roads in the network when the 60 m proposal pool is too small. Thus a point can lack its aligned reference road while still having model states. An empty or invalid road graph fails the request; there is no explicit null-road state.")
    table([["Trip","Raw Top-1","HMM Top-1","Raw cont.","HMM cont.","Raw seq.","HMM seq."]]+[[r["trip"]]+[pct(r[k]) for k in ["raw_top1","hmm_top1","raw_continuity","hmm_continuity","raw_sequence","hmm_sequence"]] for r in dec])
    p(f"Observation-weighted Top-1 rose from {pct(pooled(dec,'raw_top1'))} to {pct(pooled(dec,'hmm_top1'))}. Mean trip sequence similarity changed from {pct(avg(dec,'raw_sequence'))} to {pct(avg(dec,'hmm_sequence'))}. Trip 00000077 remained the weakest pointwise case despite high decoded continuity. A connected trajectory can still follow the wrong road.")
    trip7=next(r for r in dec if r["trip"]=="00000007")
    p(f"The per-trip decoder CSV gives trip 00000007 HMM sequence similarity as {pct(trip7['hmm_sequence'])}. The older one-computer UI capture shows 72.5%, so that stale image is excluded from this PDF; the reproducible CSV value is used here.")
    p("On trip 00000056, Viterbi increased point accuracy but reduced sequence similarity. The first metric weights time spent on each road, while the second compares road order after collapsing repeats. An overall gain therefore provides limited evidence about recovery of short, briefly traversed segments.")
    page("6. Geographical inspection")
    fig,lo,hi,disagreements,window_points=mismatch_network_figure(padding=3)
    figure(fig,325)
    p(f"Zoomed local road-network view for student-test trip 00000077, points {lo}-{hi-1}, around the longest contiguous prediction/reference disagreement. {disagreements} of {window_points} fixes in this close-up are mismatches. Trip 00000077 has the weakest decoded student-test point Top-1 ({pct(next(r['hmm_top1'] for r in dec if r['trip']=='00000077'))}). The view distinguishes the aligned reference, predicted directed roads, GPS fixes and nearby network.")
    p("The interactive Folium export at maps/student_test_00000007.html uses tiles=None and draws the supplied trip network locally. Its grey canvas is intentional; no external tile provider is requested, so this is not a failed tile load. The development screenshot evidence/folium.png is trip 00000007 and does not show the disagreement close-up above.")
    page("7. API, UI and deployment")
    p("The API provides GET /trips and GET /match/{trip_id}. Laptop 1 hosts the model and trip data; Laptop 2 serves the UI, connects to Laptop 1's LAN address and displays the returned route. The API predicts first and reads the route reference afterward for optional evaluation metrics. The trip list is loaded dynamically from the folder passed to serve.py, so the evaluator list appears only when the organizer-provided evaluator folder is configured.")
    table([["Component","Available in this repository","Verification status"],
      ["API","api.py, serve.py; model scoring, Viterbi and request logs","Local rehearsal recorded for 10 student-test trips"],
      ["UI","ui/index.html; dynamic /trips list and API-address field","Local browser/API rehearsal only"],
      ["Evaluator trips","Organizer-provided folder is supplied at deployment","Not in checkout; IDs unverified"],
      ["Laptop 1 receiving-terminal capture","evidence/api-laptop1.png","Not present"],
      ["Laptop 2 UI captures","evidence/ (trips 77, 91)","LAN URL; successful requests"]])
    p("The two Laptop2 captures show successful route requests to 10.0.14.176:8000 for student-test trips 00000077 and 00000091. The UI reports rounded metrics, so the detailed comparison in this report uses results/decoder.csv. The Laptop1 receiving-terminal capture and organizer evaluator-trip IDs are not in this checkout; those deployment requirements remain unverified.")
    shot_style=ParagraphStyle("shot_caption",fontName="Helvetica",fontSize=8,leading=10,textColor="#253c34",alignment=1)
    shot_row=[]
    for trip_id in ("00000077","00000091"):
        path=f"evidence/ui-laptop2-trip{trip_id}.jpeg"
        if Path(path).is_file():
            with Picture.open(path) as pic: w,hh=pic.size
            im=Image(path,width=200,height=200*hh/w)
            shot_row.append([im,Paragraph(f"Laptop2 UI connected to 10.0.14.176:8000 — trip {trip_id}",shot_style)])
    if len(shot_row)==2:
        shots=Table([[item[0] for item in shot_row],[item[1] for item in shot_row]],colWidths=[242,242],hAlign="LEFT")
        shots.setStyle(TableStyle([("VALIGN",(0,0),(-1,-1),"TOP"),("ALIGN",(0,0),(-1,-1),"CENTER"),("LEFTPADDING",(0,0),(-1,-1),2),("RIGHTPADDING",(0,0),(-1,-1),2)]))
        story.extend([shots,Spacer(1,8)])
    heading("Repository components")
    table([["Deliverable","Source"],
      ["Preprocessing and features","prepare.py, pathsnap.py, reference.py, review2_data.py"],
      ["Five-model training and tuning","train.py"],
      ["Evaluation and HMM comparison","evaluate() in train.py; results/folds.csv and results/decoder.csv"],
      ["API and UI","api.py, serve.py, ui/index.html"],
      ["Requirements and instructions","requirements.txt, README.md"],
      ["Geographical maps and PDF","maps.py, report.py"]])
    heading("Reproducibility and AI disclosure")
    p("AI assistance was used in implementation, experiments and drafting. From the repository root, regenerate this report with python report.py after prepare.py and train.py.")
    Path("report").mkdir(exist_ok=True)
    doc=SimpleDocTemplate("report/PathSnap.pdf",pagesize=A4,leftMargin=49,rightMargin=49,topMargin=50,bottomMargin=42,title="PathSnap | Technical report")
    def footer(canvas,doc):
        canvas.setFont("Helvetica",8);canvas.setFillColor("#456255");canvas.drawString(49,24,"PATHSNAP  /  TECHNICAL REPORT");canvas.drawRightString(546,24,str(doc.page))
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
if __name__=="__main__":
    main()

