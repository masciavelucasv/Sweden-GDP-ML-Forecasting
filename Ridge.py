# -*- coding: utf-8 -*-
"""
Created on Fri Nov 28 13:33:47 2025

@author: masci
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, RidgeCV
import matplotlib.pyplot as plt

# === Load and prepare data ===
df["Quarter"] = pd.PeriodIndex(df["Quarter"], freq="Q")
df = df.sort_values("Quarter")

# Compute y/y GDP growth (%)
df["GDP_yoy"] = (df["GDP"] / df["GDP"].shift(4) - 1) * 100

# Target: GDP_yoy two quarters ahead
h = 2
df["target_tph"] = df["GDP_yoy"].shift(-h)

# Lag predictors by 1 quarter (real-time setup)
predictors = ["Inflation","Government_Consumption","GDP",
              "Private consumption","Exchange rate to USD"]
for col in predictors:
    df[f"{col}_lag1"] = df[col].shift(1)

Xcols = [f"{col}_lag1" for col in predictors]
df = df.dropna(subset=Xcols+["target_tph"]).copy()

# === Function to compute forecast ===
def ridge_forecast(df, Xcols, start_oos, rolling=False, window_size=40):
    records = []
    alphas = np.linspace(0.01, 50, 50)
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

        ridgecv = RidgeCV(alphas=alphas, scoring="neg_mean_squared_error", cv=5)
        ridgecv.fit(Xtr_s, ytr)

        ridge = Ridge(alpha=ridgecv.alpha_)
        ridge.fit(Xtr_s, ytr)
        pred = ridge.predict(Xte_s)[0]

        records.append({
            "Quarter": t,
            "y_true": yte[0],
            "y_hat": pred,
            "alpha": ridgecv.alpha_
        })
    return pd.DataFrame(records)

# === Run both models ===
start_oos = pd.Period("1985Q1","Q")

print("Running expanding window Ridge...")
res_expanding = ridge_forecast(df, Xcols, start_oos, rolling=False)

print("Running rolling window Ridge...")
res_rolling = ridge_forecast(df, Xcols, start_oos, rolling=True, window_size=40)

# === Plot comparison with Riksbank ===
df["Quarter_ts"] = df["Quarter"].dt.to_timestamp()
res_expanding_plot = res_expanding.copy()
res_expanding_plot["Quarter"] = res_expanding_plot["Quarter"].dt.to_timestamp()
res_rolling_plot = res_rolling.copy()
res_rolling_plot["Quarter"] = res_rolling_plot["Quarter"].dt.to_timestamp()

actual = df[["Quarter_ts", "GDP_yoy", "GDP forecast by riksbank"]].dropna()

plt.figure(figsize=(12,6))
plt.plot(actual["Quarter_ts"], actual["GDP_yoy"], 
         label="Actual GDP Growth", color="black", linewidth=2)
plt.plot(res_expanding_plot["Quarter"], res_expanding_plot["y_hat"], 
         label="Ridge (Expanding Window)", color="orange", linewidth=1.8)
plt.plot(res_rolling_plot["Quarter"], res_rolling_plot["y_hat"], 
         label="Ridge (Rolling Window)", color="green", linewidth=1.8)
plt.plot(actual["Quarter_ts"], actual["GDP forecast by riksbank"], 
         label="Riksbank Forecast", color="red", linestyle="--", linewidth=2)

plt.axvspan(pd.to_datetime("2020-04-01"), pd.to_datetime("2021-06-30"), color="red", alpha=0.1, label="COVID shock")
plt.axvspan(pd.to_datetime("2021-07-01"), actual["Quarter_ts"].max(), color="green", alpha=0.1, label="Recovery")

plt.title("Sweden GDP Growth Forecasts: Ridge Regression vs Riksbank Benchmark", fontsize=13)
plt.xlabel("Quarter")
plt.ylabel("Real GDP Growth (y/y %)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()

# === Save results (optional) ===
res_expanding.to_csv("Ridge_expanding_results.csv", index=False)
res_rolling.to_csv("Ridge_rolling_results.csv", index=False)

# === Evaluate model accuracy ===

def rmse(y, yh):
    return np.sqrt(np.mean((y - yh)**2))

def smape(y, yh):
    return 100 * np.mean(2 * np.abs(y - yh) / (np.abs(y) + np.abs(yh) + 1e-6))

# Align forecasts with actual GDP values
actual_series = df.set_index("Quarter")["GDP_yoy"]
riksbank_series = df.set_index("Quarter")["GDP forecast by riksbank"].dropna()

# Merge actuals with each forecast
res_expanding_eval = res_expanding.set_index("Quarter").join(actual_series.rename("actual")).dropna()
res_rolling_eval = res_rolling.set_index("Quarter").join(actual_series.rename("actual")).dropna()
riksbank_eval = riksbank_series.to_frame("riksbank").join(actual_series.rename("actual")).dropna()

# Compute metrics
metrics = pd.DataFrame({
    "Model": ["Ridge (Expanding)", "Ridge (Rolling)", "Riksbank Benchmark"],
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
