"""make_separator_figures.py — figures for the separator results."""
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd

TEAL="#028090"; AMBER="#F3A712"; RED="#C94C4C"; GREEN="#5E8C61"
NAVY="#14323B"; MUTE="#5A7177"; INK="#1A1A1A"; GRID="#D9E6E6"
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":11,"axes.edgecolor":MUTE,
    "axes.labelcolor":INK,"text.color":INK,"xtick.color":INK,"ytick.color":INK,
    "axes.grid":True,"grid.color":GRID,"grid.linewidth":0.8,"axes.axisbelow":True,"figure.dpi":150})

R = os.path.join(os.path.dirname(__file__), "..", "results")
F = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(F, exist_ok=True)
S = pd.read_csv(f"{R}/separator_summary.csv")
C = pd.read_csv(f"{R}/separator_confusion.csv", index_col=0)

# FIG A — balanced typing accuracy: baseline vs separator
fig, ax = plt.subplots(figsize=(6.2,4.0))
order = {"baseline_ceiling":"Baselines\n(cannot type)",
         "separator_rule":"Separator\n(rule-based)",
         "separator_supervised":"Separator\n(supervised)"}
S = S[S.method.isin(order)].copy()
S["lab"] = S.method.map(order)
colors = {"baseline_ceiling":MUTE,"separator_rule":TEAL,"separator_supervised":GREEN}
bars = ax.bar(S.lab, S.balanced_acc, color=[colors[m] for m in S.method], width=0.6)
ax.axhline(0.5, ls="--", lw=1, color=RED); ax.text(len(S)-0.5,0.515,"chance / baseline ceiling",fontsize=8,color=RED,ha="right")
for b,v in zip(bars,S.balanced_acc):
    ax.text(b.get_x()+b.get_width()/2, v+0.012, f"{v:.2f}", ha="center", fontsize=11, weight="bold")
ax.set_ylim(0,0.8); ax.set_ylabel("balanced typing accuracy")
ax.set_title("Our stability signal lifts typing above the baseline ceiling",
             fontsize=12.5, color=NAVY, weight="bold", pad=10)
for s in ["top","right"]: ax.spines[s].set_visible(False)
plt.tight_layout(); plt.savefig(f"{F}/figA_separator_accuracy.png", bbox_inches="tight"); plt.close()

# FIG B — confusion matrix (where it works, where it doesn't)
fig, ax = plt.subplots(figsize=(5.2,4.4))
M = C.reindex(index=["L","K","S","N"], columns=["L","K","S","N"]).fillna(0).astype(int)
Mn = M.div(M.sum(axis=1).replace(0,1), axis=0)
im = ax.imshow(Mn.values, cmap="BuGn", vmin=0, vmax=1)
ax.set_xticks(range(4)); ax.set_xticklabels(["L","K","S","N"])
ax.set_yticks(range(4)); ax.set_yticklabels(["L","K","S","N"])
ax.set_xlabel("predicted type"); ax.set_ylabel("true type")
for i in range(4):
    for j in range(4):
        v = M.values[i,j]; frac = Mn.values[i,j]
        ax.text(j,i,f"{v}", ha="center", va="center",
                color="white" if frac>0.5 else INK, fontsize=12, weight="bold")
ax.set_title("Separator confusion: S recovered, K collapses into L",
             fontsize=11.5, color=NAVY, weight="bold", pad=10)
plt.colorbar(im, fraction=0.046, pad=0.04, label="row-normalized")
plt.tight_layout(); plt.savefig(f"{F}/figB_separator_confusion.png", bbox_inches="tight"); plt.close()
print("separator figures written to", F)