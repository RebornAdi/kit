"""
final_figures.py — Build the project's final figures from final_*.csv files.
Run AFTER final_eval on each dataset you have. Figures:
  figF1 — separator vs baseline across datasets (bar)
  figF2 — per-dataset confusion matrices (grid)
"""
import os, glob, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd

TEAL="#028090"; GREEN="#5E8C61"; RED="#C94C4C"; MUTE="#5A7177"; NAVY="#14323B"; INK="#1A1A1A"; GRID="#D9E6E6"
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":11,"axes.edgecolor":MUTE,
    "axes.labelcolor":INK,"text.color":INK,"xtick.color":INK,"ytick.color":INK,
    "axes.grid":True,"grid.color":GRID,"grid.linewidth":0.8,"axes.axisbelow":True,"figure.dpi":150})

R = os.path.join(os.path.dirname(__file__), "..", "results")
F = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(F, exist_ok=True)

summ = sorted(glob.glob(f"{R}/final_*.csv"))
summ = [s for s in summ if "confusion" not in s]
if not summ:
    raise SystemExit("no final_*.csv found — run final_eval first")
D = pd.concat([pd.read_csv(s) for s in summ], ignore_index=True)
order = [d for d in ["synthetic","adult","mimic","gmc"] if d in set(D.dataset)]
D = D.set_index("dataset").reindex(order).reset_index()

# FIG F1 — separator vs baseline
fig, ax = plt.subplots(figsize=(6.6,4.1))
x=np.arange(len(D)); w=0.38
ax.bar(x-w/2, D.baseline_acc, w, label="Baseline ceiling", color=MUTE)
ax.bar(x+w/2, D.separator_acc, w, label="Our separator", color=TEAL)
ax.axhline(0.5, ls="--", lw=1, color=RED)
for i,(b,sv) in enumerate(zip(D.baseline_acc,D.separator_acc)):
    ax.text(i-w/2,b+0.012,f"{b:.2f}",ha="center",fontsize=9)
    ax.text(i+w/2,sv+0.012,f"{sv:.2f}",ha="center",fontsize=9,weight="bold")
ax.set_xticks(x); ax.set_xticklabels([d.capitalize() for d in D.dataset])
ax.set_ylim(0,0.8); ax.set_ylabel("balanced typing accuracy (held-out)")
ax.set_title("Separator beats the baseline ceiling on synthetic and real data",
             fontsize=12.5,color=NAVY,weight="bold",pad=10)
ax.legend(frameon=False,fontsize=9.5,loc="upper right")
for s in ["top","right"]: ax.spines[s].set_visible(False)
plt.tight_layout(); plt.savefig(f"{F}/figF1_separator_vs_baseline.png",bbox_inches="tight"); plt.close()

# FIG F2 — confusion matrices
confs = {os.path.basename(p).replace("final_confusion_","").replace(".csv",""): p
         for p in glob.glob(f"{R}/final_confusion_*.csv")}
confs = {k:confs[k] for k in order if k in confs}
if confs:
    n=len(confs)
    fig, axes = plt.subplots(1,n,figsize=(3.6*n,3.4),squeeze=False)
    for ax,(name,p) in zip(axes[0],confs.items()):
        C=pd.read_csv(p,index_col=0).reindex(index=["L","K","S","N"],columns=["L","K","S","N"]).fillna(0).astype(int)
        Cn=C.div(C.sum(axis=1).replace(0,1),axis=0)
        im=ax.imshow(Cn.values,cmap="BuGn",vmin=0,vmax=1)
        ax.set_xticks(range(4)); ax.set_xticklabels(["L","K","S","N"])
        ax.set_yticks(range(4)); ax.set_yticklabels(["L","K","S","N"])
        ax.set_title(name.capitalize(),fontsize=11,color=NAVY,weight="bold")
        ax.set_xlabel("predicted")
        for i in range(4):
            for j in range(4):
                v=C.values[i,j]
                ax.text(j,i,f"{v}",ha="center",va="center",
                        color="white" if Cn.values[i,j]>0.5 else INK,fontsize=10,weight="bold")
    axes[0][0].set_ylabel("true type")
    plt.tight_layout(); plt.savefig(f"{F}/figF2_confusions.png",bbox_inches="tight"); plt.close()

print("final figures written to", F)
