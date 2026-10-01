import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, LeaveOneGroupOut, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "inverted_table_tennis_rubber_prices_cleaned.csv"
OUT = BASE / "results"
SEED = 42
OUT.mkdir(parents=True, exist_ok=True)

def load_data():
    df = pd.read_csv(DATA)
    df["Price"] = df["Price_EUR"]
    df["HardnessValue"] = df["Hardness_ESN_numeric"]
    df["MeanSpongeThickness"] = df["Mean_Sponge_Thickness_mm"]
    return df

NUMERIC = ["Speed", "Spin", "Control", "Gears", "HardnessValue", "Tackiness", "MeanSpongeThickness", "Overall"]

def make_pipeline(use_brand=True):
    transforms = [("num", Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ]), NUMERIC)]
    if use_brand:
        transforms.append(("brand", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["Brand"]))
    prep = ColumnTransformer(transforms)
    rf = RandomForestRegressor(n_estimators=500, min_samples_leaf=2, random_state=SEED, n_jobs=1)
    return Pipeline([("preprocess", prep), ("model", rf)])

def metrics(y, pred):
    y, pred = np.asarray(y), np.asarray(pred)
    premium = y >= 80
    return {
        "R2": float(r2_score(y, pred)),
        "RMSE": float(mean_squared_error(y, pred) ** 0.5),
        "MAE": float(mean_absolute_error(y, pred)),
        "bias_all_actual_minus_pred": float(np.mean(y - pred)),
        "premium_n": int(premium.sum()),
        "premium_MAE": float(mean_absolute_error(y[premium], pred[premium])) if premium.any() else None,
        "premium_bias_actual_minus_pred": float(np.mean(y[premium] - pred[premium])) if premium.any() else None,
    }

df = load_data()
X = df[NUMERIC + ["Brand"]]
y = df["Price"].to_numpy()
train_idx, test_idx = train_test_split(np.arange(len(df)), test_size=.20, random_state=SEED)
Xtr, Xte, ytr, yte = X.iloc[train_idx], X.iloc[test_idx], y[train_idx], y[test_idx]

results = {}

# Baseline RF.
base = make_pipeline(True)
base.fit(Xtr, ytr)
base_pred = base.predict(Xte)
results["baseline_rf"] = metrics(yte, base_pred)

# Log-target mitigation; inverse transform predictions to euros.
log_model = make_pipeline(True)
log_model.fit(Xtr, np.log1p(ytr))
log_pred = np.expm1(log_model.predict(Xte))
results["log_target_rf"] = metrics(yte, log_pred)

# Weighted RF: premium training records receive three times the base weight.
weighted = make_pipeline(True)
weights = np.where(ytr >= 80, 3.0, 1.0)
weighted.fit(Xtr, ytr, model__sample_weight=weights)
weighted_pred = weighted.predict(Xte)
results["weighted_rf"] = metrics(yte, weighted_pred)

# Brand ablation on identical held-out split.
no_brand = make_pipeline(False)
no_brand.fit(Xtr, ytr)
no_brand_pred = no_brand.predict(Xte)
results["no_brand_rf"] = metrics(yte, no_brand_pred)

# Five-fold CV ablation, with preprocessing refitted inside each fold.
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
for label, use_brand in [("cv_with_brand", True), ("cv_without_brand", False)]:
    fold = []
    for tr, va in kf.split(X):
        model = make_pipeline(use_brand)
        model.fit(X.iloc[tr], y[tr])
        fold.append(metrics(y[va], model.predict(X.iloc[va])))
    results[label] = {
        k: {"mean": float(np.mean([f[k] for f in fold])), "std": float(np.std([f[k] for f in fold]))}
        for k in ["R2", "RMSE", "MAE"]
    }

# Leave-one-brand-out predictions: each brand is entirely unseen during training.
logo = LeaveOneGroupOut()
logo_pred = np.empty_like(y, dtype=float)
per_brand = []
for tr, va in logo.split(X, y, groups=df["Brand"]):
    model = make_pipeline(True)
    model.fit(X.iloc[tr], y[tr])
    p = model.predict(X.iloc[va])
    logo_pred[va] = p
    per_brand.append({"Brand": df.loc[va[0], "Brand"], "n": int(len(va)), "MAE": float(mean_absolute_error(y[va], p)), "Bias": float(np.mean(y[va] - p))})
results["leave_one_brand_out_pooled"] = metrics(y, logo_pred)
results["leave_one_brand_out_macro_MAE"] = float(np.mean([x["MAE"] for x in per_brand]))

pd.DataFrame({
    "Rubber Name": df.loc[test_idx, "Rubber Name"].to_numpy(),
    "Brand": df.loc[test_idx, "Brand"].to_numpy(),
    "ActualPrice": yte,
    "BaselineRF": base_pred,
    "LogTargetRF": log_pred,
    "WeightedRF": weighted_pred,
    "NoBrandRF": no_brand_pred,
}).to_csv(OUT / "heldout_predictions.csv", index=False)
pd.DataFrame(per_brand).sort_values("MAE", ascending=False).to_csv(OUT / "leave_one_brand_out_by_brand.csv", index=False)
with open(OUT / "results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
print(json.dumps(results, indent=2))
