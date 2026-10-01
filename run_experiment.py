import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "inverted_table_tennis_rubber_prices_cleaned.csv"
OUT = BASE / "results"
SEED = 42
OUT.mkdir(parents=True, exist_ok=True)


df = pd.read_csv(DATA)
df["Price"] = df["Price_EUR"]
df["HardnessValue"] = df["Hardness_ESN_numeric"]
df["MeanSpongeThickness"] = df["Mean_Sponge_Thickness_mm"]

numeric = ["Speed", "Spin", "Control", "Gears", "HardnessValue", "Tackiness", "MeanSpongeThickness", "Overall"]
categorical = ["Brand"]
X = df[numeric + categorical]
y = df["Price"]

preprocess = ColumnTransformer([
    ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
])

models = {
    "Linear regression": LinearRegression(),
    "Decision tree": DecisionTreeRegressor(max_depth=6, min_samples_leaf=4, random_state=SEED),
    "Random forest": RandomForestRegressor(n_estimators=500, min_samples_leaf=2, random_state=SEED, n_jobs=1),
    "Gradient boosting": GradientBoostingRegressor(n_estimators=200, learning_rate=0.03, max_depth=2, random_state=SEED),
    "MLP neural network": MLPRegressor(hidden_layer_sizes=(32, 16), activation="relu", solver="adam", alpha=0.01, max_iter=3000, early_stopping=True, validation_fraction=0.15, n_iter_no_change=80, random_state=SEED),
}

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=SEED)
cv = KFold(n_splits=5, shuffle=True, random_state=SEED)

print(f"n={len(df)}, train={len(X_train)}, test={len(X_test)}, brands={df.Brand.nunique()}")
holdout_rows = []
cv_rows = []
for name, model in models.items():
    pipe = Pipeline([("preprocess", preprocess), ("model", model)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    holdout = (r2_score(y_test, pred), mean_squared_error(y_test, pred) ** 0.5, mean_absolute_error(y_test, pred))
    scores = cross_validate(pipe, X, y, cv=cv, scoring={"r2": "r2", "mae": "neg_mean_absolute_error", "mse": "neg_mean_squared_error"}, n_jobs=1)
    cv_r2 = scores["test_r2"]
    cv_mae = -scores["test_mae"]
    cv_rmse = np.sqrt(-scores["test_mse"])
    print(name)
    print(" holdout R2 RMSE MAE", *(f"{v:.3f}" for v in holdout))
    print(" cv R2", f"{cv_r2.mean():.3f} +/- {cv_r2.std():.3f}", "RMSE", f"{cv_rmse.mean():.3f} +/- {cv_rmse.std():.3f}", "MAE", f"{cv_mae.mean():.3f} +/- {cv_mae.std():.3f}")
    holdout_rows.append({"Model": name, "R2": holdout[0], "RMSE_EUR": holdout[1], "MAE_EUR": holdout[2]})
    cv_rows.append({
        "Model": name,
        "R2_mean": cv_r2.mean(), "R2_std": cv_r2.std(),
        "RMSE_EUR_mean": cv_rmse.mean(), "RMSE_EUR_std": cv_rmse.std(),
        "MAE_EUR_mean": cv_mae.mean(), "MAE_EUR_std": cv_mae.std(),
    })

pd.DataFrame(holdout_rows).to_csv(OUT / "model_comparison_holdout.csv", index=False)
pd.DataFrame(cv_rows).to_csv(OUT / "model_comparison_cross_validation.csv", index=False)
with open(OUT / "model_comparison.json", "w", encoding="utf-8") as f:
    json.dump({"holdout": holdout_rows, "cross_validation": cv_rows}, f, indent=2)
