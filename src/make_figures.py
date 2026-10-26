"""make_figures.py — paper figures from results/. Run after run_experiment."""
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd

TEAL="#028090"; SEA="#00A896"; AMBER="#F3A712"; RED="#C94C4C"; GREEN="#5E8C61"
NAVY="#14323B"; MUTE="#5A7177"; INK="#1A1A1A"; GRID="#D9E6E6"
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":11,"axes.edgecolor":MUTE,
    "axes.labelcolor":INK,"text.color":INK,"xtick.color":INK,"ytick.color":INK,
    "axes.grid":True,"grid.color":GRID,"grid.linewidth":0.8,"axes.axisbelow":True,"figure.dpi":150})

R = os.path.join(os.path.dirname(__file__), "..", "results")
F = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(F, exist_ok=True)
S = pd.read_csv(f"{R}/summary.csv"); byd = pd.read_csv(f"{R}/by_difficulty.csv")
conf = pd.read_csv(f"{R}/flag_by_type.csv")
present = [t for t in ["assoc_threshold","featureleak","leaklint","dataleaks"]
           if t in set(S.tool)]

# FIG 1 — detection vs typing
fig, ax = plt.subplots(figsize=(7.4,4.2))
ok = S[S.status=="ok"].set_index("tool").reindex(present).reset_index()
x=np.arange(len(ok)); w=0.38
b1=ax.bar(x-w/2, ok.detect_recall, w, label="Detection recall (T1)", color=TEAL)
b2=ax.bar(x+w/2, ok.type_separation, w, label="Type separation (T2)", color=AMBER)
ax.axhline(0.33, ls="--", lw=1, color=MUTE)
ax.text(len(ok)-0.5,0.35,"chance (3 types)",fontsize=8,color=MUTE,ha="right")
for b in list(b1)+list(b2):
    h=b.get_height()
    if not np.isnan(h): ax.text(b.get_x()+b.get_width()/2,h+0.015,f"{h:.2f}",ha="center",fontsize=8.5)
ax.set_xticks(x); ax.set_xticklabels(ok.tool); ax.set_ylim(0,1.12); ax.set_ylabel("Score")
ax.set_title("Leakage tools detect informative missingness but cannot type it",
             fontsize=12.5,color=NAVY,weight="bold",pad=12)
ax.legend(frameon=False,fontsize=9.5,loc="upper right")
for s in ["top","right"]: ax.spines[s].set_visible(False)
plt.tight_layout(); plt.savefig(f"{F}/fig1_detect_vs_type.png",bbox_inches="tight"); plt.close()

# FIG 2 — flag by type (the collapse)
tools2=[t for t in ["assoc_threshold","featureleak"] if t in set(conf.tool)]
fig, axes = plt.subplots(1,len(tools2),figsize=(4.3*len(tools2),3.6),squeeze=False)
for ax,tool in zip(axes[0],tools2):
    sub=conf[conf.tool==tool]; types=["L","K","S","N"]; frac=[]
    for ty in types:
        tot=sub[sub.gt_type==ty]["count"].sum()
        fl=sub[(sub.gt_type==ty)&(sub.flagged==True)]["count"].sum()
        frac.append(fl/tot if tot else 0)
    bars=ax.bar(types,frac,color=[GREEN,RED,RED,MUTE])
    for b,f in zip(bars,frac): ax.text(b.get_x()+b.get_width()/2,f+0.02,f"{f:.0%}",ha="center",fontsize=9)
    ax.set_ylim(0,1.15); ax.set_title(tool,fontsize=11,color=NAVY,weight="bold")
    ax.set_xlabel("ground-truth type")
    for s in ["top","right"]: ax.spines[s].set_visible(False)
axes[0][0].set_ylabel("flagged rate")
fig.text(0.5,1.02,"Keep (L, green) and remove (K/S, red) get the identical verdict",
         ha="center",fontsize=12.5,color=NAVY,weight="bold")
plt.tight_layout(rect=[0,0,1,0.95]); plt.savefig(f"{F}/fig2_flag_by_type.png",bbox_inches="tight"); plt.close()

# FIG 3 — detection vs difficulty
fig, ax = plt.subplots(figsize=(7.0,4.0))
cmap={"assoc_threshold":TEAL,"featureleak":SEA,"leaklint":RED,"dataleaks":AMBER}
for tool in present:
    d=byd[byd.tool==tool].sort_values("diff")
    ax.plot(d["diff"].values,d["detect_recall"].values,"-o",color=cmap.get(tool,MUTE),label=tool,lw=2,ms=6)
ax.set_xlabel("injected mask–target association strength"); ax.set_ylabel("detection recall")
ax.set_ylim(-0.05,1.08); ax.set_xticks([0.2,0.3,0.4,0.5])
ax.set_title("Detection improves with strength; typing never does",fontsize=12.5,color=NAVY,weight="bold",pad=10)
ax.legend(frameon=False,fontsize=9.5,loc="center right")
for s in ["top","right"]: ax.spines[s].set_visible(False)
plt.tight_layout(); plt.savefig(f"{F}/fig3_detect_vs_difficulty.png",bbox_inches="tight"); plt.close()
print("figures written to", F)
