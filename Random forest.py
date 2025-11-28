# -*- coding: utf-8 -*-
"""
Created on Fri Nov 28 13:33:34 2025

@author: masci
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.pipeline import Pipeline

# === Load and prepare data ===
df["Quarter"] = pd.PeriodIndex(df["Quarter"], freq="Q")
df = df.sort_values("Quarter")

# Compute year-over-year GDP growth (%)
df["GDP_yoy"] = (df["GDP"] / df["GDP"].shift(4) - 1) * 100

# Target: GDP_yoy two quarters ahead
h = 2
df["target_tph"] = df["GDP_yoy"].shift(-h)

# Lag predictors by 1 quarter
predictors = ["Inflation", "Government_Consumption", "GDP",
              "Private consumption", "Exchange rate to USD"]
for col in predictors:
    df[f"{col}_lag1"] = df[col].shift(1)

Xcols = [f"{col}_lag1" for col in predictors]
df = df.dropna(subset=Xcols + ["target_tph"]).copy()

# === Hyperparameter tuning using early data ===
train_tune = df[df["Quarter"] < pd.Period("2005Q1", "Q")]
X_tune, y_tune = train_tune[Xcols].values, train_tune["target_tph"].values

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("rf", RandomForestRegressor(random_state=0, n_jobs=-1))
])

param_grid = {
    "rf__n_estimators": [100, 200],
    "rf__max_depth": [None, 6, 12]
}

tscv = TimeSeriesSplit(n_splits=5)
gscv = GridSearchCV(pipe, param_grid, cv=tscv, scoring="neg_mean_squared_error")
gscv.fit(X_tune, y_tune)

best_params = gscv.best_params_
print("Best RF parameters:", best_params)
n_estimators = best_params["rf__n_estimators"]
max_depth = best_params["rf__max_depth"]

# === Define forecasting function ===
def rf_forecast(df, Xcols, start_oos, rolling=False, window_size=40):
    records = []
    for t in df.loc[df["Quarter"] >= start_oos, "Quarter"]:
        test = df[df["Quarter"] == t]
        if rolling:
            end_idx = df.index[df["Quarter"] == t][0]
            start_idx = max(0, end_idx - window_size)
            train = df.iloc[start_idx:end_idx]
        else:
            train = df[df["Quarter"] < t]
        if train.empty or test.empty:
            continue

        Xtr, ytr = train[Xcols].values, train["target_tph"].values
        Xte, yte = test[Xcols].values, test["target_tph"].values

        scaler = StandardScaler().fit(Xtr)
        Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)

        mdl = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=0,
            n_jobs=-1
        )
        mdl.fit(Xtr_s, ytr)
        pred = mdl.predict(Xte_s)[0]

        records.append({
            "Quarter": t,
            "y_true": yte[0],
            "y_hat": pred
        })
    return pd.DataFrame(records)

# === Run both models ===
start_oos = pd.Period("1985Q1", "Q")
print("\nRunning expanding window Random Forest...")
res_expanding = rf_forecast(df, Xcols, start_oos, rolling=False)
print("Running rolling window Random Forest...")
res_rolling = rf_forecast(df, Xcols, start_oos, rolling=True, window_size=40)

# === Plot comparison with Riksbank ===
df["Quarter_ts"] = df["Quarter"].dt.to_timestamp()
res_expanding_plot = res_expanding.copy()
res_expanding_plot["Quarter"] = res_expanding_plot["Quarter"].dt.to_timestamp()
res_rolling_plot = res_rolling.copy()
res_rolling_plot["Quarter"] = res_rolling_plot["Quarter"].dt.to_timestamp()

actual = df[["Quarter_ts", "GDP_yoy", "GDP forecast by riksbank"]].dropna()

plt.figure(figsize=(12, 6))
plt.plot(actual["Quarter_ts"], actual["GDP_yoy"],
         label="Actual GDP Growth", color="black", linewidth=2)
plt.plot(res_expanding_plot["Quarter"], res_expanding_plot["y_hat"],
         label="Random Forest (Expanding Window)", color="orange", linewidth=1.8)
plt.plot(res_rolling_plot["Quarter"], res_rolling_plot["y_hat"],
         label="Random Forest (Rolling Window)", color="green", linewidth=1.8)
plt.plot(actual["Quarter_ts"], actual["GDP forecast by riksbank"],
         label="Riksbank Forecast", color="red", linestyle="--", linewidth=2)

plt.axvspan(pd.to_datetime("2020-04-01"), pd.to_datetime("2021-06-30"),
            color="red", alpha=0.1, label="COVID shock")
plt.axvspan(pd.to_datetime("2021-07-01"), actual["Quarter_ts"].max(),
            color="green", alpha=0.1, label="Recovery")

plt.title("Sweden GDP Growth Forecasts: Random Forest vs Riksbank Benchmark", fontsize=13)
plt.xlabel("Quarter")
plt.ylabel("Real GDP Growth (y/y %)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()

# === Evaluate forecast accuracy ===
def rmse(y, yh):
    return np.sqrt(np.mean((y - yh)**2))

def smape(y, yh):
    return 100 * np.mean(2 * np.abs(y - yh) / (np.abs(y) + np.abs(yh) + 1e-6))

actual_series = df.set_index("Quarter")["GDP_yoy"]
riksbank_series = df.set_index("Quarter")["GDP forecast by riksbank"].dropna()

res_expanding_eval = res_expanding.set_index("Quarter").join(actual_series.rename("actual")).dropna()
res_rolling_eval = res_rolling.set_index("Quarter").join(actual_series.rename("actual")).dropna()
riksbank_eval = riksbank_series.to_frame("riksbank").join(actual_series.rename("actual")).dropna()

metrics = pd.DataFrame({
    "Model": ["Random Forest (Expanding)", "Random Forest (Rolling)", "Riksbank Benchmark"],
    "RMSE": [
        rmse(res_expanding_eval["actual"], res_expanding_eval["y_hat"]),
        rmse(res_rolling_eval["actual"], res_rolling_eval["y_hat"]),
        rmse(riksbank_eval["actual"], riksbank_eval["riksbank"])
    ],
    "sMAPE": [
        smape(res_expanding_eval["actual"], res_expanding_eval["y_hat"]),
        smape(res_rolling_eval["actual"], res_rolling_eval["y_hat"]),
        smape(riksbank_eval["actual"], riksbank_eval["riksbank"])
    ]
})

print("\n=== Forecast Accuracy Comparison ===")
print(metrics.to_string(index=False, formatters={
    "RMSE": "{:.3f}".format,
    "sMAPE": "{:.2f}".format
}))

# === Optional: save results ===
res_expanding.to_csv("RandomForest_expanding_results.csv", index=False)
res_rolling.to_csv("RandomForest_rolling_results.csv", index=False)
