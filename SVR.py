# -*- coding: utf-8 -*-
"""
Created on Fri Nov 28 13:34:07 2025

@author: masci
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, GridSearchCV

#Load and prepare data
df.set_index(df["Quarter"], inplace=True)
df.drop(columns='Quarter', inplace=True)
sdf = (df - df.mean()) / df.std()
sdf['Growth_Rate'] = sdf['GDP'].pct_change() * 100
X = sdf[sdf.columns.difference(['Growth_Rate'])]
X = X.dropna().copy()
y = sdf['Growth_Rate'].dropna()
X = X.loc[y.index]
if '1982-Q1' in X.index:
    X = X.drop('1982-Q1')
    y = y.drop('1982-Q1')
sdf.dropna(inplace=True)

#Kernel Comparison: Poly(2), Poly(3), RBF
tscv = TimeSeriesSplit(n_splits=5)
kernels = {
    'Poly (degree=2)': SVR(kernel='poly', degree=2),
    'Poly (degree=3)': SVR(kernel='poly', degree=3),
    'RBF': SVR(kernel='rbf')
}

print(" Comparing kernels using 5-fold time-series CV (RMSE):")
for name, model in kernels.items():
    scores = cross_val_score(model, X, y, cv=tscv, scoring='neg_mean_squared_error')
    rmse = np.sqrt(-scores.mean())
    print(f"{name:20s} → RMSE: {rmse:.4f}")

#Hyperparameter tuning for best kernel (RBF expected best)
svr = SVR(kernel='rbf')
svr_params = {
    'C': [0.1, 1, 10, 50],
    'epsilon': [0.01, 0.1, 0.5],
    'gamma': ['scale', 'auto']}
grid = GridSearchCV(svr, svr_params, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1)
grid.fit(X.loc[:'2017-Q4'], y.loc[:'2017-Q4'])
best_svr = grid.best_estimator_
print("\nBest SVR (RBF) parameters:", grid.best_params_)

#Expanding Window Forecast
forecast_horizon = 2
train_end = X.index.get_loc('2017-Q4')
test_start = train_end + 1
test_end = X.index.get_loc('2024-Q4')
pred_svr, actuals, test_dates = [], [], []
for i in range(test_start, test_end + 1 - forecast_horizon):
    X_train = X.iloc[:i]
    y_train = y.iloc[:i]
    X_test = X.iloc[i + forecast_horizon - 1:i + forecast_horizon]
    y_test = y.iloc[i + forecast_horizon - 1:i + forecast_horizon]
    best_svr.fit(X_train, y_train)
    y_pred = best_svr.predict(X_test)[0]
    actuals.append(y_test.values[0])
    pred_svr.append(y_pred)
    test_dates.append(y.index[i + forecast_horizon - 1])
results_svr = pd.DataFrame({
    'Date': test_dates,
    'Actual': actuals,
    'SVR_Pred': pred_svr}).set_index('Date')

#Add Riksbank benchmark
results_svr['Riksbank'] = df.loc[results_svr.index, 'GDP forecast by riksbank']

#Evaluation
def smape(y_true, y_pred):
    return 100 / len(y_true) * np.sum(
        2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))
for col in ['SVR_Pred', 'Riksbank']:
    rmse = sqrt(mean_squared_error(results_svr['Actual'], results_svr[col]))
    smape_val = smape(results_svr['Actual'], results_svr[col])
    print(f"{col:10s} → RMSE: {rmse:.4f}, sMAPE: {smape_val:.2f}%")
svr_rmse = sqrt(mean_squared_error(results_svr['Actual'], results_svr['SVR_Pred']))
bm_rmse = sqrt(mean_squared_error(results_svr['Actual'], results_svr['Riksbank']))
improvement = (1 - svr_rmse / bm_rmse) * 100
print(f"\nSVR improves RMSE by {improvement:.2f}% over the Riksbank benchmark.\n")

#Plot Results
results_svr.index = pd.PeriodIndex(results_svr.index.astype(str), freq='Q').to_timestamp(how='end')
plt.figure(figsize=(11,5))
plt.plot(results_svr.index, results_svr['Actual'], 'k-', linewidth=2, label='Actual GDP Growth')
plt.plot(results_svr.index, results_svr['SVR_Pred'], 'g--', linewidth=2, label='SVR (RBF) Forecast')
plt.plot(results_svr.index, results_svr['Riksbank'], 'r:', linewidth=2, label='Riksbank Forecast')
plt.title('Sweden Real GDP Growth – SVR (RBF) Expanding Window Forecast (2 Quarters Ahead)')
plt.xlabel('Quarter')
plt.ylabel('GDP Growth (%)')
plt.grid(True)
plt.legend()
labels = results_svr.index.to_period('Q').astype(str)
step = max(1, len(labels)//12)
plt.xticks(results_svr.index[::step], labels[::step], rotation=45)
plt.ylim(-30, 30)
plt.tight_layout()
plt.show()
