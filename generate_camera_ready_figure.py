import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parent
OUT = BASE / "figures"
OUT.mkdir(parents=True, exist_ok=True)
with open(BASE / "results" / "results.json", encoding="utf-8") as f:
    r = json.load(f)

plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":8.2,
                     "axes.titlesize":9.2, "axes.labelsize":8.2,
                     "axes.spines.top":False, "axes.spines.right":False})
blue, orange, green, red, gray = "#2864A8", "#E6812F", "#3A8D5D", "#C84C4C", "#8B98A5"

fig, ax = plt.subplots(1, 3, figsize=(7.15, 2.45))

# (a) mitigation
labels = ["Baseline", "Log target", "Weighted"]
keys = ["baseline_rf", "log_target_rf", "weighted_rf"]
x = np.arange(3); w=.34
ax[0].bar(x-w/2, [r[k]["MAE"] for k in keys], w, color=blue, label="All test products")
ax[0].bar(x+w/2, [r[k]["premium_MAE"] for k in keys], w, color=orange, label="Premium (≥€80)")
ax[0].set(xticks=x, xticklabels=labels, ylabel="MAE (€)", title="(a) High-price mitigation")
ax[0].tick_params(axis="x", rotation=20); ax[0].legend(frameon=False, fontsize=6.5)

# (b) ablation
means=[r["cv_with_brand"]["MAE"]["mean"],r["cv_without_brand"]["MAE"]["mean"]]
errs=[r["cv_with_brand"]["MAE"]["std"],r["cv_without_brand"]["MAE"]["std"]]
bars=ax[1].bar(["With brand","Without brand"],means,yerr=errs,capsize=3,color=[green,gray])
ax[1].set(ylabel="Five-fold MAE (€)", title="(b) Brand ablation", ylim=(0,12))
for b,v in zip(bars,means): ax[1].text(b.get_x()+b.get_width()/2,v+.25,f"{v:.2f}",ha="center",fontsize=7)

# (c) generalization
vals=[r["baseline_rf"]["MAE"],r["cv_with_brand"]["MAE"]["mean"],r["leave_one_brand_out_pooled"]["MAE"]]
bars=ax[2].bar(["Held-out\nproducts","Five-fold\nCV","Unseen\nbrands"],vals,color=[blue,green,red])
ax[2].set(ylabel="MAE (€)", title="(c) Generalization tests", ylim=(0,16))
for b,v in zip(bars,vals): ax[2].text(b.get_x()+b.get_width()/2,v+.3,f"{v:.2f}",ha="center",fontsize=7)

fig.tight_layout(w_pad=1.2)
fig.savefig(OUT/"fig6_robustness_experiments.png",dpi=320,bbox_inches="tight",facecolor="white")
fig.savefig(OUT/"fig6_robustness_experiments.pdf",bbox_inches="tight",facecolor="white")
print("generated",OUT/"fig6_robustness_experiments.png")
