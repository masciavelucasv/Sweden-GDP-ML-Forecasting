# Machine Learning Forecasts of Swedish GDP Growth (1985–2024)
The analysis replicates and extends the methodological approach of Yoon (2021), applying it to a small open economy and a longer time horizon, including the COVID-19 shock and the recovery period.

---

## 📌 Project Overview

The forecasting problem is to predict Sweden's real GDP growth **two quarters ahead** using macroeconomic indicators such as:

- Inflation
- Government and private consumption
- Real GDP
- SEK/USD exchange rate

All predictors are lagged by one quarter to mimic a real-time central banking forecasting environment.

The Random Forest model is estimated under two training strategies:

| Strategy | Description |
|----------|-------------|
| **Expanding Window** | Uses all data available up to each forecast point |
| **Rolling Window (10 years)** | Uses only the most recent 40 quarters, improving adaptability during structural breaks |

Both variants are compared against the **Riksbank's own GDP growth projections**.

> ⚠️ **Note:** The dataset from Statistics Sweden (SCB), FRED, and the Riksbank is not included due to licensing.

---

## 🧠 What the Script Does

`random_forest_gdp_forecast.py` performs the full forecasting workflow:

<details>
<summary><strong>1. Load and preprocess the data</strong></summary>

- Converts quarters to real time-series objects
- Computes year-over-year GDP growth
- Creates the two-quarters-ahead target variable
- Lags predictors to simulate real-time forecasting

</details>

<details>
<summary><strong>2. Tune the Random Forest</strong></summary>

The script performs hyperparameter tuning using early data (before 2005), with:

- `n_estimators` ∈ {100, 200}
- `max_depth` ∈ {None, 6, 12}
- Time-series cross-validation (`TimeSeriesSplit`)

This avoids look-ahead bias.

</details>

<details>
<summary><strong>3. Generate forecasts</strong></summary>

The script produces:
- Expanding-window forecasts
- Rolling-window (40-quarter) forecasts

Both forecast sequences are saved as CSV files in `/results`.

</details>

<details>
<summary><strong>4. Plot the forecasts</strong></summary>

A single plot compares:
- Actual GDP growth
- Random Forest (Expanding)
- Random Forest (Rolling)
- Riksbank forecast

COVID-19 (2020–2021) and the recovery period are shaded for clarity.

</details>

<details>
<summary><strong>5. Evaluate forecasting accuracy</strong></summary>

Using two metrics: **RMSE** and **sMAPE**

The script prints a table comparing:
- Random Forest (Expanding)
- Random Forest (Rolling)
- Riksbank Benchmark

These metrics were used in the empirical research project.

</details>

---

### 3. Run the forecasting script

```bash
python code/random_forest_gdp_forecast.py
```

This will:
- Train the RF models
- Generate forecasts
- Output accuracy metrics
- Produce comparison plots
- Save both expanding and rolling window forecast CSV files

---

## 📊 Key Insights from the Study

| Finding | Detail |
|--------|--------|
| 🏆 Best during COVID-19 | Rolling window RF adapts exceptionally well during the shock |
| 📈 Best post-recovery | Expanding window RF provides smoother forecasts after 2021 |
| ✅ vs Riksbank | Both RF variants perform competitively or better than the benchmark |
| 🌍 Broader implication | ML models can provide valuable real-time insights for small open economies |

---

---

## ✨ Acknowledgements

This repository supports the empirical study **"Adaptive Machine Learning Forecasts of Swedish GDP Growth (1985–2024)"**.

The code was developed by **Luca Masciavè** as part of an academic research project in macroeconomic forecasting.

