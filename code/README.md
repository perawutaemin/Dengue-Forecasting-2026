# Experiment code

Run from the **repository root**:

```bash
pip install -r requirements.txt
```

## Paper pipeline (matches Focus manuscript)

### 1. SARIMA baseline (walk-forward)

```bash
python code/sarima_walkforward_clean.py
```

- Tune: train 2021–2022 → validate 2023  
- Production: train 2022–2023 → forecast 2024  
- Output: `results/sarima/`

### 2. XGBoost Thailand Self-lag

```bash
python code/xgboost_thailand_self.py
```

Uses lags with Pearson |r| ≥ 0.6 from `dataset/correlation/04_thailand_selflag_correlation_2022_2023.csv`.  
Output: `results/thailand_self/`

Optional self-lag pruning path:

```bash
python code/xgboost_thailand_self_pruning.py
```

### 3. Cross-country prune + retune (Lao / Korea / Guatemala and the full 16-country screen)

```bash
python code/run_16countries_prune_vs_sarima.py
```

Steps:
1. Correlation screen vs Thailand (2022–2023)  
2. Backward prune weather features by XGBoost gain importance  
3. Small hyperparameter grid (val 2023) → production 2024  
4. Compare with SARIMA baseline (`dengue_final_results/analysis_8_sarima_walkforward/predictions_2024.csv`)

Output: `results/prune_vs_sarima/`

Paper countries of interest:
- Lao PeopleS Democratic Republic → count_lag_52 + TAVG_lag_10  
- South Korea → count_lag_50 + PRCP_lag_51  
- Guatemala → count_lag_52 + TMAX_lag_9  

### 4. (Optional) Legacy hyperparameter sweep

```bash
python code/xgboost_hyperparam_tuning.py
```

## Notes

- Paths are relative to the **repo root**.
- Korea dengue counts are loaded from `dataset/clean_data/Dengue_data/normal_data/Korea_Dengue.csv` (no native timelag file).
- Korea weather prefers `dataset/clean_data/Weather_data/South Korea_Weather_Weekly_2019_2024.csv`, with cleaned fallback `South_Korea_Weather_Weekly_2019_2024_cleaned.csv`.
