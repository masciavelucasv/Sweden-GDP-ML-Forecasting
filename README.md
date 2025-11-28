# Machine Learning Forecasts of Swedish GDP Growth (1985–2024)

This repository contains the code and materials for an empirical forecasting study on Sweden’s real GDP growth.
The project evaluates whether modern machine learning methods—specifically Random Forests with expanding and rolling windows—can outperform the Riksbank’s official GDP forecasts.

The analysis replicates and extends the methodological approach of Yoon (2021), applying it to a small open economy and a longer time horizon, including the COVID-19 shock and the recovery period.

📌 Project Overview

The forecasting problem is to predict Sweden’s real GDP growth two quarters ahead using macroeconomic indicators such as:

Inflation

Government and private consumption

Real GDP

SEK/USD exchange rate

All predictors are lagged by one quarter to mimic a real-time central banking forecasting environment.

The Random Forest model is estimated under two training strategies:

Expanding Window – Uses all data available up to each forecast point.

Rolling Window (10 years) – Uses only the most recent 40 quarters, improving adaptability during structural breaks.

Both variants are compared against the Riksbank’s own GDP growth projections.


Note: The dataset from Statistics Sweden (SCB), FRED, and the Riksbank is not included due to licensing.

🧠 What the Script Does

random_forest_gdp_forecast.py performs the full forecasting workflow:

1. Load and preprocess the data

Converts quarters to real time-series objects

Computes year-over-year GDP growth

Creates the two-quarters-ahead target variable

Lags predictors to simulate real-time forecasting

2. Tune the Random Forest

The script performs hyperparameter tuning using early data (before 2005), with:

n_estimators ∈ {100, 200}

max_depth ∈ {None, 6, 12}

Time-series cross-validation (TimeSeriesSplit)

This avoids look-ahead bias.

3. Generate forecasts

The script produces:

Expanding-window forecasts

Rolling-window (40-quarter) forecasts

Both forecast sequences are saved as CSV files in /results.

4. Plot the forecasts

A single plot compares:

Actual GDP growth

Random Forest (Expanding)

Random Forest (Rolling)

Riksbank forecast

COVID-19 (2020–2021) and the recovery period are shaded for clarity.

5. Evaluate forecasting accuracy

Using two metrics:

RMSE

sMAPE

The script prints a table comparing:

Random Forest (Expanding)

Random Forest (Rolling)

Riksbank Benchmark

These metrics were used in the empirical research project.

🚀 How to Run the Script
1. Install dependencies

Create a virtual environment (recommended) and install:

pip install pandas numpy matplotlib scikit-learn

2. Place the dataset

Download the combined dataset from SCB, FRED, and the Riksbank and place it here:

data/Cleaned_ERA_xlsx_Sheet.csv

3. Run the forecasting script
python code/random_forest_gdp_forecast.py


This will:

Train the RF models

Generate forecasts

Output accuracy metrics

Produce comparison plots

Save both expanding and rolling window forecast CSV files

📊 Key Insights from the Study

Rolling window Random Forest adapts exceptionally well during the COVID-19 shock.

Expanding window Random Forest provides smoother forecasts during the post-2021 recovery.

Both RF variants perform competitively or better than the Riksbank benchmark.

Machine learning models can provide valuable real-time insights for small open economies like Sweden.

📚 Data Sources

Statistics Sweden (SCB): Quarterly GDP, inflation, consumption

FRED: SEK/USD exchange rate

Sveriges Riksbank: Official GDP forecasts

Due to licensing restrictions, these datasets are not redistributed in the repository.

✨ Acknowledgements

This repository supports the empirical study “Adaptive Machine Learning Forecasts of Swedish GDP Growth (1985–2024)”.

The code was developed by Luca Masciavè as part of an academic research project in macroeconomic forecasting.

