import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "inverted_table_tennis_rubber_prices_cleaned.csv"
OUT = BASE / "figures"
SEED = 42
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 9.5,
    "axes.labelsize": 8.5, "legend.fontsize": 7.5, "figure.dpi": 140,
    "axes.spines.top": False, "axes.spines.right": False,
})
BLUE, ORANGE, GREEN, RED, PURPLE = "#2864A8", "#E6812F", "#3A8D5D", "#C84C4C", "#7656A5"


def nums(v):
    return [float(x) for x in re.findall(r"\d+(?:\.\d+)?", str(v))]


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=320, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


df = pd.read_csv(DATA)
df["Price"] = df["Price_EUR"]
df["HardnessValue"] = df["Hardness_ESN_numeric"]
df["MeanSpongeThickness"] = df["Mean_Sponge_Thickness_mm"]

numeric = ["Speed", "Spin", "Control", "Gears", "HardnessValue", "Tackiness", "MeanSpongeThickness", "Overall"]
categorical = ["Brand"]
X, y = df[numeric + categorical], df["Price"]
preprocess = ColumnTransformer([
    ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
])
models = {
    "Linear": LinearRegression(),
    "Decision tree": DecisionTreeRegressor(max_depth=6, min_samples_leaf=4, random_state=SEED),
    "Random forest": RandomForestRegressor(n_estimators=500, min_samples_leaf=2, random_state=SEED, n_jobs=1),
    "Gradient boosting": GradientBoostingRegressor(n_estimators=200, learning_rate=0.03, max_depth=2, random_state=SEED),
    "MLP": MLPRegressor(hidden_layer_sizes=(32, 16), activation="relu", solver="adam", alpha=0.01,
                        max_iter=3000, early_stopping=True, validation_fraction=0.15,
                        n_iter_no_change=80, random_state=SEED),
}
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.20, random_state=SEED)
results, predictions, fitted = {}, {}, {}
for name, model in models.items():
    pipe = Pipeline([("preprocess", preprocess), ("model", model)])
    pipe.fit(Xtr, ytr)
    pred = pipe.predict(Xte)
    results[name] = [r2_score(yte, pred), mean_squared_error(yte, pred) ** 0.5, mean_absolute_error(yte, pred)]
    predictions[name], fitted[name] = pred, pipe

# Figure 1: dataset overview
fig, ax = plt.subplots(1, 2, figsize=(7.1, 2.65), gridspec_kw={"width_ratios": [1, 1.35]})
ax[0].hist(y, bins=np.arange(0, 151, 10), color=BLUE, edgecolor="white", linewidth=.7)
ax[0].axvline(y.mean(), color=RED, ls="--", lw=1.3, label=f"Mean = €{y.mean():.2f}")
ax[0].axvline(y.median(), color=ORANGE, ls=":", lw=1.5, label=f"Median = €{y.median():.2f}")
ax[0].set(xlabel="Listed price (€)", ylabel="Number of products", title="(a) Price distribution")
ax[0].legend(frameon=False)
stats = df.groupby("Brand")["Price"].agg(["median", "count"]).sort_values("median")
yy = np.arange(len(stats))
ax[1].barh(yy, stats["median"], color=GREEN, alpha=.9)
ax[1].set_yticks(yy, stats.index)
ax[1].set(xlabel="Median listed price (€)", title="(b) Brand medians (sample count shown)")
for i, (v, n) in enumerate(zip(stats["median"], stats["count"])):
    ax[1].text(v + 1, i, f"n={int(n)}", va="center", fontsize=6.8)
ax[1].set_xlim(0, stats["median"].max() + 18)
fig.tight_layout(w_pad=1.2)
save(fig, "fig1_dataset_overview")

# Figure 2: test performance
names = list(results)
display_names = ["Linear", "Decision\ntree", "Random\nforest", "Gradient\nboosting", "MLP"]
r2 = [results[n][0] for n in names]
rmse = [results[n][1] for n in names]
mae = [results[n][2] for n in names]
fig, ax = plt.subplots(1, 2, figsize=(7.1, 2.35))
colors = ["#9AA5B1", ORANGE, BLUE, GREEN, PURPLE]
bars = ax[0].bar(display_names, r2, color=colors)
ax[0].set(ylabel="$R^2$ (higher is better)", title="(a) Explained variance", ylim=(0, 1.0))
ax[0].tick_params(axis="x", rotation=0)
for b, v in zip(bars, r2): ax[0].text(b.get_x()+b.get_width()/2, v+.02, f"{v:.3f}", ha="center", fontsize=6.8)
x = np.arange(len(names)); w = .36
ax[1].bar(x-w/2, rmse, w, label="RMSE", color=RED)
ax[1].bar(x+w/2, mae, w, label="MAE", color=BLUE)
ax[1].set(ylabel="Error (€; lower is better)", title="(b) Prediction errors", xticks=x, xticklabels=display_names)
ax[1].tick_params(axis="x", rotation=0); ax[1].legend(frameon=False)
fig.tight_layout(w_pad=1.5)
save(fig, "fig2_model_performance")

# Figure 3: random-forest predictions and residuals
pred = predictions["Random forest"]
resid = yte.to_numpy() - pred
lo, hi = min(yte.min(), pred.min()), max(yte.max(), pred.max())
fig, ax = plt.subplots(1, 2, figsize=(7.1, 2.55))
ax[0].scatter(yte, pred, s=24, alpha=.72, color=BLUE, edgecolor="white", linewidth=.35)
ax[0].plot([lo, hi], [lo, hi], "--", color=RED, lw=1.2, label="Perfect prediction")
ax[0].set(xlabel="Actual price (€)", ylabel="Predicted price (€)", title="(a) Actual vs. predicted")
ax[0].legend(frameon=False)
ax[1].scatter(pred, resid, s=24, alpha=.72, color=GREEN, edgecolor="white", linewidth=.35)
ax[1].axhline(0, ls="--", color=RED, lw=1.2)
ax[1].set(xlabel="Predicted price (€)", ylabel="Residual: actual − predicted (€)", title="(b) Residual plot")
fig.tight_layout(w_pad=1.5)
save(fig, "fig3_random_forest_predictions")

# Figure 4: grouped random-forest feature importance
rf_pipe = fitted["Random forest"]
encoded = rf_pipe.named_steps["preprocess"].get_feature_names_out()
imp = rf_pipe.named_steps["model"].feature_importances_
grouped = {"Speed": 0, "Spin": 0, "Control": 0, "Gears": 0, "Hardness": 0,
           "Tackiness": 0, "Sponge thickness": 0, "Overall": 0, "Brand (all indicators)": 0}
mapping = {"HardnessValue": "Hardness", "MeanSpongeThickness": "Sponge thickness"}
for feature, value in zip(encoded, imp):
    raw = feature.split("__", 1)[-1]
    if feature.startswith("cat__"):
        grouped["Brand (all indicators)"] += value
    else:
        grouped[mapping.get(raw, raw)] += value
order = sorted(grouped, key=grouped.get)
fig, ax = plt.subplots(figsize=(3.55, 2.8))
ax.barh(order, [grouped[k] for k in order], color=[BLUE if "Brand" not in k else ORANGE for k in order])
ax.set(xlabel="Decrease-in-impurity importance", title="Grouped random-forest feature importance")
ax.set_xlim(0, max(grouped.values()) * 1.15)
fig.tight_layout()
save(fig, "fig4_feature_importance")

# Figure 5: study workflow
fig, ax = plt.subplots(figsize=(7.1, 1.65)); ax.set_xlim(0, 10); ax.set_ylim(0, 2); ax.axis("off")
labels = ["Raw data\n464 records", "Cleaning\n376 valid prices", "Preprocessing\nscale + encode", "Five models\ntrain + validate", "Evaluation\n$R^2$, RMSE, MAE"]
xs = [.15, 2.15, 4.15, 6.15, 8.15]
box_colors = ["#EAF1F8", "#FDF0E4", "#E8F4EC", "#F0ECF8", "#FBEAEA"]
for i, (x0, label) in enumerate(zip(xs, labels)):
    box = FancyBboxPatch((x0, .55), 1.7, .9, boxstyle="round,pad=0.04,rounding_size=0.08",
                         ec="#4B5563", fc=box_colors[i], lw=1)
    ax.add_patch(box); ax.text(x0+.85, 1.0, label, ha="center", va="center", fontsize=8)
    if i < len(xs)-1:
        ax.add_patch(FancyArrowPatch((x0+1.72, 1), (xs[i+1]-.05, 1), arrowstyle="-|>", mutation_scale=10, color="#4B5563"))
ax.text(5, 1.78, "Experimental workflow", ha="center", va="center", fontsize=10, weight="bold")
save(fig, "fig5_workflow")

print(f"Generated {len(list(OUT.glob('*.png')))} PNG and {len(list(OUT.glob('*.pdf')))} PDF figures in {OUT}")
