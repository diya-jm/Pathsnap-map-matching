"""Regenerate the consolidated PDF from the development experiment outputs."""
import csv
import json
from io import BytesIO
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,PageBreak
from PIL import Image as Picture

def read(name):
    with open("results/"+name,newline="") as f:
        return list(csv.DictReader(f))
def pct(x):return f"{float(x)*100:.2f}%"
def num(x):return f"{float(x):.4f}"
def avg(rows,col):return np.mean([float(r[col]) for r in rows])
def pooled(rows,col):return np.average([float(r[col]) for r in rows],weights=[int(r["n"]) for r in rows])

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
    def table(rows,widths=None):
        t=Table(rows,colWidths=widths,hAlign="LEFT",repeatRows=1)
        t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),"#214c3c"),("TEXTCOLOR",(0,0),(-1,0),colors.white),
             ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("FONTSIZE",(0,0),(-1,-1),8),
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
    p(f"Across 19,160 observations from ten student-test trips, the tuned classifier placed {pct(tuned['top1'])} on the aligned reference segment. Viterbi decoding raised this to {pct(pooled(dec,'hmm_top1'))}. Mean trip continuity increased from {pct(avg(dec,'raw_continuity'))} to {pct(avg(dec,'hmm_continuity'))}. These scores describe the released development trips.")
    heading("Data inventory")
    rows=[["Partition","Trips","GPS fixes","Missing GPS values","Gaps >15 s","1 s intervals"]]
    for split in ["train","student_test"]:
        sub=[r for r in inv if r["split"]==split]
        total=lambda k:sum(int(r[k]) for r in sub)
        rows.append([split,len(sub),f"{total('observations'):,}",total("nonfinite"),total("gaps_over_15"),pct(total("one_second_steps")/(total("observations")-len(sub)))])
    table(rows)
    p("Every trip had readable track, nodes, arcs and route tables. GPS coordinates and timestamps were finite, timestamps increased throughout, and the loaders validated arc endpoints and reference IDs. Network component counts, isolated nodes, geographical extents and gap statistics for each trip are recorded in results/inventory.csv.")
    fig,ax=plt.subplots(figsize=(7.3,2.7))
    for split,color in [("train","#26775b"),("student_test","#c77a32")]:
        vals=sorted(int(r["observations"]) for r in inv if r["split"]==split)
        ax.plot(np.linspace(0,100,len(vals)),vals,label=split,color=color)
    ax.set(xlabel="Trip percentile within partition",ylabel="GPS observations",yscale="log");ax.legend();figure(fig,185)
    heading("Sampling and error concentration")
    p("Training trip 00000054 alone accounts for 10,551 of the 16,395 training points whose aligned reference segment was absent from the candidate set. Trip 00000033 adds 3,889. Each training trip therefore contributes at most 250 randomly sampled observations per fit. Evaluation uses every observation, including candidate misses.")
    page("From GPS points to road candidates")
    heading("Reference alignment")
    p("Each track row gives longitude, latitude and time. An arc identifies a directed pair of node rows. A route is an ordered list of traversed arc IDs without timestamps. Reference alignment uses dynamic programming to choose a nondecreasing sequence of route positions minimizing total point-to-segment distance, with a small endpoint bias. A point may stay on a segment or skip ahead. This yields approximate per-point labels. Repeated roads, close parallel segments and long stationary periods can leave the alignment ambiguous.")
    heading("Candidates and features")
    p("Coordinates use a trip-local equirectangular approximation in metres. The longitude scale uses mean latitude. A spatial index samples each directed segment at intervals of at most 25 m. For each GPS point, 64 nearby samples propose roads, which are reranked by exact point-to-segment distance. Up to eight distinct directed roads are retained. If the pool contains too few roads, the search examines all arcs. The sampled search can miss a close arc in dense networks.")
    table([["Feature","Reason"],["Segment distance; excess over nearest","Separate plausible roads from distant alternatives."],
        ["Heading cosine; heading available","Distinguish road direction when motion provides evidence."],
        ["Speed; time gap","Identify stationary periods or unreliable motion estimates."],
        ["Arc length; fraction along arc","Locate the observation relative to junctions and endpoints."]])
    p("A centred five-observation window provides heading and approximate speed for a complete recorded trip. Heading is suppressed below 3 m displacement or when the window exceeds 15 s. All learned scaling is fitted inside each training fold. The candidate generator reads track and network files only.")
    heading("Experimental protocol")
    p("Three group-aware folds hold out entire trips from the 80-trip training set. The fitting cap applies in each training fold, using seed 138. All ten student-test trips are evaluated separately. Candidate-level precision, recall and macro F1 use a 0.5 probability threshold. Observation Top-1 uses the largest score among each point's candidates, with misses counted as errors. The primary selection metric is mean fold Top-1 because the deployed system must pick one road per observation.")
    p("Continuity counts consecutive predictions on the same arc or with matching directed endpoints. Skipping a short connecting arc between fixes can lower continuity even when the vehicle travelled a valid path. Sequence similarity uses matching blocks from Python SequenceMatcher on consecutive-deduplicated predicted and reference arc sequences. Continuity and sequence results are averaged per trip.")
    page("Five models on the same trips")
    table([["Model","CV F1","Test F1","Top-1","Continuity","Log loss","ms/fix"]]+[[r["model"],num(r["cv_macro_f1"]),num(r["macro_f1"]),pct(r["top1"]),pct(r["continuity"]),num(r["log_loss"]),num(r["inference_ms_per_obs"])] for r in models[:-1]])
    fig,axes=plt.subplots(2,1,figsize=(7.4,4.6))
    names=[r["model"] for r in models[:-1]];ix=np.arange(5)
    for offset,k,label,c in [(-.25,"cv_macro_f1","CV macro F1","#a3bdb0"),(0,"macro_f1","Test macro F1","#26795d"),(.25,"top1","Test Top-1","#c47530")]:
        axes[0].bar(ix+offset,[float(r[k]) for r in models[:-1]],.25,label=label,color=c)
    axes[0].set_ylim(0,1);axes[0].legend(ncol=3,fontsize=8);axes[0].set_xticks(ix,names)
    axes[1].bar(ix,[float(r["log_loss"]) for r in models[:-1]],color="#496a83")
    axes[1].set(ylabel="Candidate log loss",xticks=ix,xticklabels=names);figure(fig,285)
    p("Times above cover candidate scoring and selection on this machine. Feature extraction, route alignment, decoding and network transport are excluded. Lower log loss rewards confident correct candidate probabilities. Baseline random forest had the lowest student-test log loss, while histogram boosting had the highest cross-validated Top-1.")
    table([["Model","Precision","Recall","Fit (s)","Train Top-1","CV Top-1","Fold range"]]+[[r["model"],num(r["precision"]),num(r["recall"]),f"{float(r['fit_s']):.2f}",pct(r["train_top1"]),pct(r["cv_top1"]),"/".join(pct(fn([float(a["top1"]) for a in read("folds.csv") if a["model"]==r["model"]])) for fn in (min,max))] for r in models[:-1]])
    p("Precision and recall refer to the positive class (the aligned reference arc). The fold range gives minimum and maximum held-out Top-1. Training Top-1 is scored on all fixes from fitting trips, including observations outside the training subsample.")
    page("Model interpretations")
    notes={
      "Logistic":("Logistic regression","L2-regularized logistic loss; C=1, up to 300 iterations, standardized features.","A fast additive baseline measures how much linear distance and heading evidence can explain. Its lower CV Top-1 than the tree models suggests it misses interactions near junctions."),
      "GaussianNB":("Gaussian naive Bayes","Per-class Gaussian likelihoods and empirical class priors; variance smoothing 1e-9 after standardization.","The highest positive recall coincides with many false positives. Dependence between distance and excess distance challenges the independence assumption and can make probabilities overconfident."),
      "DecisionTree":("Decision tree","Greedy Gini impurity reduction; maximum depth 9 and at least 20 examples per leaf.","Splits can express different distance rules for moving and stationary vehicles. The training-to-CV gap is consistent with fitting local patterns that generalize less well to new trips."),
      "RandomForest":("Random forest","40 bootstrap trees, Gini impurity, maximum depth 12, minimum leaf size 5, square-root feature sampling.","Averaging leaf probabilities reduces reliance on individual splits. Test log loss improved over the single tree, at greater fitting and prediction cost."),
      "HistBoost":("Histogram gradient boosting","90 boosting iterations, binary log loss, initial learning rate 0.1, up to 31 leaves, early stopping disabled.","Successive trees can correct errors involving several features. It achieved the highest baseline mean CV Top-1, narrowly ahead of the forest; this ranking does not establish a statistically significant difference.")}
    for r in models[:-1]:
        name,conf,desc=notes[r["model"]];heading(name);p(conf+" "+desc)
        p(f"Train / CV Top-1: {pct(r['train_top1'])} / {pct(r['cv_top1'])}. Student-test Top-1: {pct(r['top1'])}; macro F1: {num(r['macro_f1'])}; log loss: {num(r['log_loss'])}.")
    page("Bayesian optimisation")
    p("The search varied the boosting learning rate from 0.02 to 0.30 on a logarithmic grid. This explores the size of each corrective step with the tree budget held fixed. Three initial evaluations sampled the endpoints and centre. A Gaussian-process surrogate with a Matern kernel selected five further untried values by expected improvement on mean grouped-CV Top-1.")
    table([["Trial","Learning rate","CV Top-1","CV macro F1","Elapsed (s)"]]+[[r["trial"],num(r["value"]),pct(r["cv_top1"]),num(r["cv_macro_f1"]),f"{float(r['seconds']):.1f}"] for r in trials])
    p(f"Search cost totalled {sum(float(r['seconds']) for r in trials):.1f} s. The selected learning rate was {cfg['value']:.5f}. The final model was refitted on observations sampled across all 80 training trips with the same cap and seed.")
    table([["Configuration","CV macro F1","Test Top-1","Test macro F1","Log loss","ms/fix"]]+[[name,num(r["cv_macro_f1"]),pct(r["top1"]),num(r["macro_f1"]),num(r["log_loss"]),num(r["inference_ms_per_obs"])] for name,r in [("Baseline",baseline),("Tuned",tuned)]])
    p("Tuning produced a small increase in student-test Top-1 and a larger proportional reduction in log loss. More reliable score magnitudes can matter to the decoder, which combines log scores across observations. These experiments do not isolate calibration as the cause of decoder gains.")
    page("Sequential route reconstruction")
    p("The hidden state is a candidate directed arc at the current fix. Its emission score is the log of the classifier's positive-class probability, clipped above zero. A transition receives zero log penalty for the same arc or a shared directed endpoint, and -3 otherwise. These are unnormalized compatibility scores. Disconnected moves stay possible when sampling skips a road. At gaps above 15 seconds, the chain restarts. Backpointers recover the maximum-score sequence.")
    p("At each fix j, score(j) = log emission(j) + max over previous states i of [score(i) + transition(i,j)]. The decoder cannot select a reference road absent from the candidate set. The student-test candidate miss rate was "+pct(tuned["miss_rate"])+".")
    table([["Trip","Raw Top-1","HMM Top-1","Raw cont.","HMM cont.","Raw seq.","HMM seq."]]+[[r["trip"]]+[pct(r[k]) for k in ["raw_top1","hmm_top1","raw_continuity","hmm_continuity","raw_sequence","hmm_sequence"]] for r in dec])
    p(f"Observation-weighted Top-1 rose from {pct(pooled(dec,'raw_top1'))} to {pct(pooled(dec,'hmm_top1'))}. Mean trip sequence similarity changed from {pct(avg(dec,'raw_sequence'))} to {pct(avg(dec,'hmm_sequence'))}. Trip 00000077 remained the weakest pointwise case despite high decoded continuity. A connected trajectory can still follow the wrong road.")
    p("On trip 00000056, Viterbi increased point accuracy but reduced sequence similarity. The first metric weights time spent on each road, while the second compares road order after collapsing repeats. An overall gain therefore provides limited evidence about recovery of short, briefly traversed segments.")
    page("Geographical inspection")
    p("Student-test trip 00000007. The exported Folium view overlays observed GPS, the road network, the supplied reference route and predicted segments. The interactive map in maps/ allows each layer to be toggled independently.")
    photo("evidence/folium.png",365)
    p("The reference may sit behind the predicted road where the routes coincide. Wider purple strokes keep both routes visible. Orange GPS points show deviations from the network. Geographic coordinates determine map placement; the background road layer comes directly from the supplied trip network.")
    page("API and evaluator interface")
    p("Laptop 2 sends the chosen trip ID to GET /match/{trip_id}. Laptop 1 reads the track and network, computes candidates, scores them and decodes the route. Only then does the server open the route reference to calculate aggregate scores. The response contains GPS geometry, predictions, elapsed time and metrics. The UI draws coordinates locally.")
    photo("evidence/ui-laptop2.png" if Path("evidence/ui-laptop2.png").exists() else "evidence/ui-local.png",310)
    if Path("evidence/ui-laptop2.png").exists():p("Evaluator UI on Laptop 2.")
    else:p("Development rehearsal: browser and API on one computer, student-test trip 00000007.")
    if Path("evidence/api-laptop1.png").exists():
        photo("evidence/api-laptop1.png",195);p("Receiving API terminal on Laptop 1.")
    else:p("The two-laptop demonstration and its receiving-terminal screenshot are pending. The README gives the commands and screenshot filenames for the final report rebuild.")
    if Path("results/api_rehearsal.csv").exists():
        a=read("api_rehearsal.csv");p(f"Local HTTP rehearsal: {len(a)} distinct development trips returned predictions whose point accuracy matched the saved offline run.")
    heading("Reproducibility and assistance")
    p("Run prepare.py, train.py, maps.py and report.py as described in the README. The repository stores the per-trip inventory, fold results, search history and trained model. AI assistance was used to implement the pipeline, run experiments and draft this analysis. Team review and the live demonstration are required before submission.")
    Path("report").mkdir(exist_ok=True)
    doc=SimpleDocTemplate("report/PathSnap.pdf",pagesize=A4,leftMargin=49,rightMargin=49,topMargin=50,bottomMargin=42,title="PathSnap | Technical report")
    def footer(canvas,doc):
        canvas.setFont("Helvetica",8);canvas.setFillColor("#456255");canvas.drawString(49,24,"PATHSNAP  /  TECHNICAL REPORT");canvas.drawRightString(546,24,str(doc.page))
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
if __name__=="__main__":
    main()

