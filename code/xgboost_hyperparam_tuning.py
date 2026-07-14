"""
XGBoost Hyperparameter Tuning — Walk-Forward Validation
=========================================================
Tunes n_estimators, max_depth, learning_rate for 3 analysis types:

  Type A — Thailand self-prediction        (analysis_4 features)
  Type B — Multi-country count-only        (analysis_2 best = step 0, 15 countries)
  Type C — Country + weather               (analysis_1 best = Lao PDR + TAVG_lag_10)

Walk-forward (same as SARIMA):
  Tune  : train 2021-2022  →  validate 2023  →  pick best params
  Apply : train 2022-2023  →  predict  2024  (same features, tuned params)

Default params (current): n_estimators=400, max_depth=3, learning_rate=0.05

Output: analysis_11_xgboost_hyperparam/
"""
from __future__ import annotations
import csv, math, warnings
from itertools import product as iterproduct
from pathlib import Path

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

ROOT      = Path(__file__).resolve().parents[1]  # repo root
DT_DIR    = ROOT / "dataset/clean_data/Dengue_data/timelag_data"
WT_DIR    = ROOT / "dataset/clean_data/Weather_data_2019_2024/timelag_data"
OUT       = ROOT / "results/hyperparam_tuning"
OUT.mkdir(parents=True, exist_ok=True)

# ── helpers ────────────────────────────────────────────────────────────────────
def rmse(a, b): return math.sqrt(sum((x-y)**2 for x,y in zip(a,b))/len(a))
def mae(a, b):  return sum(abs(x-y) for x,y in zip(a,b))/len(a)
def r2(a, b):
    m = sum(a)/len(a); ss = sum((x-m)**2 for x in a)
    return (1 - sum((x-y)**2 for x,y in zip(a,b))/ss) if ss else float("nan")
def smooth(y, w=4):
    return np.convolve(np.array(y,dtype=float), np.ones(w)/w, mode="same").tolist()

DEFAULT = {"n_estimators": 400, "max_depth": 3, "learning_rate": 0.05}
GRID = {
    "n_estimators":  [200, 400, 600],
    "max_depth":     [2, 3, 4, 5],
    "learning_rate": [0.02, 0.05, 0.1],
}
CANDIDATES = [
    {"n_estimators": ne, "max_depth": md, "learning_rate": lr}
    for ne, md, lr in iterproduct(GRID["n_estimators"], GRID["max_depth"], GRID["learning_rate"])
]

TRAIN_VAL  = [2021, 2022]   # validation train years
TEST_VAL   = 2023           # validation test year
TRAIN_PROD = [2022, 2023]   # production train years
TEST_PROD  = 2024           # production test year


NON_NUMERIC = {"Disease", "Date", "PROVINCE", "Pro_Code"}

def load_timelag(path):
    rows = []
    with Path(path).open() as f:
        for r in csv.DictReader(f):
            row = {}
            for k, v in r.items():
                if k in NON_NUMERIC:
                    continue
                # normalise key: Year→year, Week_no→week_no, Count→count
                nk = k.lower() if k in ("Year","Week_no","Count") else k
                try:    row[nk] = float(v)
                except: row[k]  = v
            rows.append(row)
    return rows


def split_xy(rows, feature_cols, target_col, years):
    X, y = [], []
    for r in rows:
        yr = int(r.get("year", r.get("Year", 0)))
        if yr in years:
            X.append([r[c] for c in feature_cols])
            y.append(r[target_col])
    return np.array(X, dtype=float), np.array(y, dtype=float)


def fit_predict_xgb(X_tr, y_tr, X_te, params):
    from xgboost import XGBRegressor
    model = XGBRegressor(objective="reg:squarederror", random_state=42,
                         n_jobs=1, **params)
    model.fit(X_tr, y_tr, verbose=False)
    return model.predict(X_te).tolist()


def grid_search(X_val_tr, y_val_tr, X_val_te, y_val_te, label=""):
    best_params, best_rmse, best_fc = None, float("inf"), None
    for params in CANDIDATES:
        try:
            fc   = fit_predict_xgb(X_val_tr, y_val_tr, X_val_te, params)
            err  = rmse(y_val_te.tolist(), fc)
            if math.isfinite(err) and err < best_rmse:
                best_rmse, best_params, best_fc = err, params, fc
        except: pass
    print(f"  {label} best → {best_params}  val_RMSE={best_rmse:.1f}")
    return best_params, best_rmse, best_fc


def run_analysis(name, rows, feature_cols, target_col, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)

    X_val_tr, y_val_tr = split_xy(rows, feature_cols, target_col, TRAIN_VAL)
    X_val_te, y_val_te = split_xy(rows, feature_cols, target_col, [TEST_VAL])
    X_prd_tr, y_prd_tr = split_xy(rows, feature_cols, target_col, TRAIN_PROD)
    X_prd_te, y_prd_te = split_xy(rows, feature_cols, target_col, [TEST_PROD])

    weeks_val  = [r.get("week_no", r.get("Week_no", i+1)) for i,r in enumerate(rows)
                  if int(r.get("year",r.get("Year",0))) == TEST_VAL]
    weeks_prod = [r.get("week_no", r.get("Week_no", i+1)) for i,r in enumerate(rows)
                  if int(r.get("year",r.get("Year",0))) == TEST_PROD]

    # ── validation grid search ──────────────────────────────────────────────────
    best_params, val_rmse_tuned, fc_val_tuned = grid_search(
        X_val_tr, y_val_tr, X_val_te, y_val_te, name)

    # default params on validation
    fc_val_def = fit_predict_xgb(X_val_tr, y_val_tr, X_val_te, DEFAULT)
    val_rmse_def = round(rmse(y_val_te.tolist(), fc_val_def), 1)
    val_r2_def   = round(r2(y_val_te.tolist(),  fc_val_def), 3)
    val_rmse_tuned_r = round(val_rmse_tuned, 1)
    val_r2_tuned     = round(r2(y_val_te.tolist(), fc_val_tuned), 3)

    # ── production (2022-2023 → 2024) ──────────────────────────────────────────
    fc_prod_tuned = fit_predict_xgb(X_prd_tr, y_prd_tr, X_prd_te, best_params)
    fc_prod_def   = fit_predict_xgb(X_prd_tr, y_prd_tr, X_prd_te, DEFAULT)

    prod_rmse_tuned = round(rmse(y_prd_te.tolist(), fc_prod_tuned), 1)
    prod_r2_tuned   = round(r2(y_prd_te.tolist(),   fc_prod_tuned), 3)
    prod_rmse_def   = round(rmse(y_prd_te.tolist(),  fc_prod_def),  1)
    prod_r2_def     = round(r2(y_prd_te.tolist(),    fc_prod_def),  3)

    print(f"  {name} production tuned: RMSE={prod_rmse_tuned}  R²={prod_r2_tuned}")
    print(f"  {name} production default: RMSE={prod_rmse_def}  R²={prod_r2_def}")

    # ── grid search heatmap ─────────────────────────────────────────────────────
    # collect RMSE for all candidates
    grid_data = {}
    for params in CANDIDATES:
        try:
            fc  = fit_predict_xgb(X_val_tr, y_val_tr, X_val_te, params)
            err = round(rmse(y_val_te.tolist(), fc), 1)
            key = (params["n_estimators"], params["max_depth"], params["learning_rate"])
            grid_data[key] = err
        except: pass

    # ── PLOT: 2 rows (validation, production) + heatmap ────────────────────────
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(f"XGBoost Hyperparameter Tuning — {name}\n"
                 f"Walk-Forward: train {TRAIN_VAL} → val {TEST_VAL}  "
                 f"|  Best params: n_est={best_params['n_estimators']} "
                 f"depth={best_params['max_depth']} lr={best_params['learning_rate']}",
                 fontsize=12, fontweight="bold")
    gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.35)

    # top-left: validation
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(weeks_val, smooth(y_val_te.tolist()),   color="#1f3b73", lw=2.3, label="Actual 2023")
    ax1.plot(weeks_val, smooth(fc_val_tuned),  color="#e74c3c", lw=2.0,
             label=f"Tuned   RMSE={val_rmse_tuned_r}  R²={val_r2_tuned}")
    ax1.plot(weeks_val, smooth(fc_val_def),    color="#95a5a6", lw=1.6, ls="--",
             label=f"Default RMSE={val_rmse_def}  R²={val_r2_def}")
    ax1.set_title("① Validation 2023", loc="left", fontsize=10, fontweight="bold")
    ax1.set_ylabel("cases"); ax1.set_xlabel("Week_no")
    ax1.set_ylim(bottom=0); ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

    # top-right: production
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(weeks_prod, smooth(y_prd_te.tolist()),    color="#1f3b73", lw=2.3, label="Actual 2024")
    ax2.plot(weeks_prod, smooth(fc_prod_tuned),  color="#e74c3c", lw=2.0,
             label=f"Tuned   RMSE={prod_rmse_tuned}  R²={prod_r2_tuned}")
    ax2.plot(weeks_prod, smooth(fc_prod_def),    color="#95a5a6", lw=1.6, ls="--",
             label=f"Default RMSE={prod_rmse_def}  R²={prod_r2_def}")
    ax2.set_title("② Production 2024", loc="left", fontsize=10, fontweight="bold")
    ax2.set_ylabel("cases"); ax2.set_xlabel("Week_no")
    ax2.set_ylim(bottom=0); ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

    # bottom: heatmap — for each lr, n_est vs depth
    ax3 = fig.add_subplot(gs[1, :])
    lrs = sorted(GRID["learning_rate"]); nes = sorted(GRID["n_estimators"]); mds = sorted(GRID["max_depth"])
    n_lr = len(lrs)
    bar_w = 0.25
    x_base = np.arange(len(nes) * len(mds))

    rmse_matrix = []
    tick_labels = []
    for ne in nes:
        for md in mds:
            row_rmses = [grid_data.get((ne, md, lr), float("nan")) for lr in lrs]
            rmse_matrix.append(row_rmses)
            tick_labels.append(f"ne={ne}\nd={md}")

    clrs_lr = ["#3498db", "#e67e22", "#e74c3c"]
    for i, lr in enumerate(lrs):
        vals = [grid_data.get((ne, md, lr), float("nan"))
                for ne in nes for md in mds]
        xpos = x_base + (i - 1) * bar_w
        ax3.bar(xpos, vals, width=bar_w, label=f"lr={lr}", color=clrs_lr[i], alpha=0.8, edgecolor="white")

    # mark best
    best_key = (best_params["n_estimators"], best_params["max_depth"], best_params["learning_rate"])
    for ni, ne in enumerate(nes):
        for mi, md in enumerate(mds):
            lr = best_params["learning_rate"]
            if ne == best_params["n_estimators"] and md == best_params["max_depth"]:
                xi = ni * len(mds) + mi
                li = lrs.index(lr)
                val = grid_data.get(best_key, 0)
                ax3.annotate(f"BEST\n{val:.0f}", xy=(xi + (li-1)*bar_w, val),
                             xytext=(xi + (li-1)*bar_w, val + 80),
                             ha="center", fontsize=8, color="#e74c3c", fontweight="bold",
                             arrowprops=dict(arrowstyle="->", color="#e74c3c", lw=1.2))

    ax3.set_xticks(x_base); ax3.set_xticklabels(tick_labels, fontsize=7.5)
    ax3.set_ylabel("RMSE (validation 2023)  ↓", fontsize=10)
    ax3.set_title("③ Grid Search — RMSE ของทุก combination  (แยกตาม learning_rate)",
                  loc="left", fontsize=10, fontweight="bold")
    ax3.legend(fontsize=9); ax3.grid(alpha=0.25, axis="y")

    fig.savefig(out_dir / "hyperparam_tuning.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # save metrics
    with (out_dir / "metrics.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["phase","train","test","params","rmse","mae","r2"])
        w.writerow(["validation_tuned",   str(TRAIN_VAL), TEST_VAL,
                    str(best_params), val_rmse_tuned_r,
                    round(mae(y_val_te.tolist(),  fc_val_tuned),1),  val_r2_tuned])
        w.writerow(["validation_default", str(TRAIN_VAL), TEST_VAL,
                    str(DEFAULT), val_rmse_def,
                    round(mae(y_val_te.tolist(),  fc_val_def),1),   val_r2_def])
        w.writerow(["production_tuned",   str(TRAIN_PROD), TEST_PROD,
                    str(best_params), prod_rmse_tuned,
                    round(mae(y_prd_te.tolist(),  fc_prod_tuned),1), prod_r2_tuned])
        w.writerow(["production_default", str(TRAIN_PROD), TEST_PROD,
                    str(DEFAULT), prod_rmse_def,
                    round(mae(y_prd_te.tolist(),  fc_prod_def),1),  prod_r2_def])

    with (out_dir / "best_params.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["n_estimators","max_depth","learning_rate",
                                          "val_rmse","val_r2","prod_rmse","prod_r2"])
        w.writeheader()
        w.writerow({**best_params,
                    "val_rmse": val_rmse_tuned_r, "val_r2": val_r2_tuned,
                    "prod_rmse": prod_rmse_tuned,  "prod_r2": prod_r2_tuned})

    return best_params, {
        "val_rmse_tuned": val_rmse_tuned_r, "val_r2_tuned": val_r2_tuned,
        "val_rmse_def":   val_rmse_def,     "val_r2_def":   val_r2_def,
        "prod_rmse_tuned": prod_rmse_tuned, "prod_r2_tuned": prod_r2_tuned,
        "prod_rmse_def":  prod_rmse_def,    "prod_r2_def":  prod_r2_def,
    }


# ══════════════════════════════════════════════════════════════════════════════
# TYPE A — Thailand self-prediction
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== Type A: Thailand self-prediction ===")
th_path = DT_DIR / "Thailand_Dengue_timelag_0_52_2020_2024.csv"
th_rows = load_timelag(th_path)
th_feat_cols = ["lag_1","lag_2","lag_3","lag_4","lag_5","lag_6",
                "lag_9","lag_43","lag_47","lag_48","lag_49","lag_50","lag_51","lag_52"]
# ensure year/week_no
for r in th_rows:
    if "year" not in r and "Year" in r: r["year"] = r["Year"]
    if "week_no" not in r and "Week_no" in r: r["week_no"] = r["Week_no"]

params_A, metrics_A = run_analysis(
    "Thailand Self-Prediction", th_rows, th_feat_cols, "count",
    OUT / "A_thailand_self")
print(f"  Saved: A_thailand_self/")


# ══════════════════════════════════════════════════════════════════════════════
# TYPE B — Multi-country count-only (analysis 2 step 0 = 15 countries)
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== Type B: Multi-country count-only ===")

COUNTRIES_15 = ["Peru","Dominican Republic","Lao PeopleS Democratic Republic","Bolivia",
                "Puerto Rico","Nicaragua","United States Of America","Argentina",
                "Guatemala","Paraguay","Samoa","Costa Rica",
                "Northern Mariana Islands","Kiribati","Vanuatu"]

def load_country_count_lags(countries, th_rows_ref):
    """Merge Thailand target with count lags from each country."""
    th_by_key = {(int(r["year"]), int(r["week_no"])): r for r in th_rows_ref}
    country_data = {}
    for c in countries:
        fp = DT_DIR / f"{c.replace(' ','_')}_Dengue_timelag_0_52_2020_2024.csv"
        if not fp.exists(): continue
        rows = load_timelag(fp)
        by_key = {}
        for r in rows:
            yr = int(r.get("year", r.get("Year", 0)))
            wk = int(r.get("week_no", r.get("Week_no", 0)))
            by_key[(yr, wk)] = r
        country_data[c] = by_key

    merged = []
    for r in th_rows_ref:
        yr = int(r["year"]); wk = int(r["week_no"])
        row = {"year": yr, "week_no": wk, "count": r["count"]}
        for c in countries:
            if c in country_data and (yr, wk) in country_data[c]:
                src = country_data[c][(yr, wk)]
                row[f"{c}_lag_52"] = src.get("lag_52", src.get("count", 0))
        merged.append(row)
    return merged

mc_rows    = load_country_count_lags(COUNTRIES_15, th_rows)
mc_feat_cols = [f"{c}_lag_52" for c in COUNTRIES_15 if f"{c}_lag_52" in mc_rows[0]]
print(f"  Features available: {len(mc_feat_cols)} countries")

params_B, metrics_B = run_analysis(
    "Multi-Country Count-Only", mc_rows, mc_feat_cols, "count",
    OUT / "B_multicountry")
print(f"  Saved: B_multicountry/")


# ══════════════════════════════════════════════════════════════════════════════
# TYPE C — Country + Weather (Lao PDR + TAVG_lag_10)
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== Type C: Lao PDR + TAVG_lag_10 ===")

lao_name = "Lao PeopleS Democratic Republic"
lao_dengue = load_timelag(DT_DIR / f"{lao_name.replace(' ','_')}_Dengue_timelag_0_52_2020_2024.csv")
lao_weather = load_timelag(WT_DIR / f"{lao_name}_Weather_Weather_timelag_0_52_2019_2024.csv")

lao_w_by_key = {}
for r in lao_weather:
    yr = int(r.get("year", r.get("Year", 0)))
    wk = int(r.get("week_no", r.get("Week_no", 0)))
    lao_w_by_key[(yr, wk)] = r

# merge Lao dengue lags + TAVG_lag_10 → predict Thailand
lao_merged = []
for r in th_rows_ref if False else th_rows:
    yr = int(r["year"]); wk = int(r["week_no"])
    row = {"year": yr, "week_no": wk, "count": r["count"]}
    lao_d_row = {kk: vv for kk, vv in []}
    # Lao dengue lag_52
    lao_fp = DT_DIR / f"{lao_name.replace(' ','_')}_Dengue_timelag_0_52_2020_2024.csv"
    if lao_fp.exists():
        pass
    row["Lao_lag_52"] = 0.0
    row["Lao_TAVG_lag_10"] = 0.0
    lao_merged.append(row)

# load properly
lao_d_by_key = {}
for r in lao_dengue:
    yr = int(r.get("year", r.get("Year", 0)))
    wk = int(r.get("week_no", r.get("Week_no", 0)))
    lao_d_by_key[(yr, wk)] = r

lao_merged = []
for r in th_rows:
    yr = int(r["year"]); wk = int(r["week_no"])
    row = {"year": yr, "week_no": wk, "count": r["count"]}
    ld = lao_d_by_key.get((yr, wk), {})
    lw = lao_w_by_key.get((yr, wk), {})
    row["Lao_lag_52"]      = float(ld.get("lag_52", ld.get("count", 0)) or 0)
    row["Lao_TAVG_lag_10"] = float(lw.get("TAVG_lag_10", lw.get("tavg_lag_10", 0)) or 0)
    lao_merged.append(row)

lao_feat_cols = ["Lao_lag_52", "Lao_TAVG_lag_10"]
params_C, metrics_C = run_analysis(
    "Lao PDR + TAVG_lag_10", lao_merged, lao_feat_cols, "count",
    OUT / "C_laopdr_weather")
print(f"  Saved: C_laopdr_weather/")


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY COMPARISON PLOT
# ══════════════════════════════════════════════════════════════════════════════
print("\nCreating summary comparison plot...")

types = ["A\nThailand\nSelf", "B\nMulti-\ncountry", "C\nLao+\nWeather"]
colors_tuned = ["#e74c3c","#e67e22","#8e44ad"]
colors_def   = "#95a5a6"

data_rows = [
    (types[0], params_A, metrics_A),
    (types[1], params_B, metrics_B),
    (types[2], params_C, metrics_C),
]

fig, axes = plt.subplots(1, 3, figsize=(15, 6))

for ax, vals_name, ylabel, key_t, key_d in [
    (axes[0], "RMSE (↓ better)", "RMSE",  "prod_rmse_tuned", "prod_rmse_def"),
    (axes[1], "MAE (↓ better)",  "MAE",   "prod_rmse_tuned", "prod_rmse_def"),
    (axes[2], "R² (↑ better)",   "R²",    "prod_r2_tuned",   "prod_r2_def"),
]:
    x = np.arange(3)
    tuned_vals = [m[key_t] for _, _, m in data_rows]
    def_vals   = [m[key_d] for _, _, m in data_rows]
    bars1 = ax.bar(x-0.2, tuned_vals, 0.38, label="Tuned params",   color=colors_tuned, alpha=0.88, edgecolor="white")
    bars2 = ax.bar(x+0.2, def_vals,   0.38, label="Default params", color=colors_def,   alpha=0.65, edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels([t for t,_,_ in data_rows], fontsize=8.5)
    ax.set_ylabel(ylabel, fontsize=10); ax.set_title(vals_name, fontsize=10, fontweight="bold")
    ax.grid(alpha=0.25, axis="y")
    for bar, v in list(zip(bars1, tuned_vals)) + list(zip(bars2, def_vals)):
        ax.text(bar.get_x()+bar.get_width()/2,
                bar.get_height() + abs(bar.get_height()*0.02),
                str(v), ha="center", va="bottom", fontsize=7.5)

from matplotlib.patches import Patch
fig.legend(handles=[Patch(color="#e74c3c",label="Tuned (walk-forward)"),
                    Patch(color="#95a5a6",label="Default (n_est=400, depth=3, lr=0.05)")],
           fontsize=9, loc="upper right")
fig.suptitle("XGBoost Hyperparameter Tuning — Production 2024 Results\n"
             "Tuned (walk-forward 2021-22→23) vs Default params",
             fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT/"summary_comparison.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# save master summary
with (OUT/"summary.csv").open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["analysis","best_n_est","best_depth","best_lr",
                "val_rmse_tuned","val_r2_tuned","val_rmse_def","val_r2_def",
                "prod_rmse_tuned","prod_r2_tuned","prod_rmse_def","prod_r2_def"])
    for lbl, params, m in [("A_Thailand_Self",params_A,metrics_A),
                            ("B_MultiCountry", params_B,metrics_B),
                            ("C_Lao_Weather",  params_C,metrics_C)]:
        w.writerow([lbl, params["n_estimators"], params["max_depth"], params["learning_rate"],
                    m["val_rmse_tuned"], m["val_r2_tuned"], m["val_rmse_def"], m["val_r2_def"],
                    m["prod_rmse_tuned"], m["prod_r2_tuned"], m["prod_rmse_def"], m["prod_r2_def"]])

print(f"\n{'='*55}")
print(" DONE — Output: dengue_final_results/analysis_11_xgboost_hyperparam/")
print(f"{'='*55}")
print(f"  A Thailand Self  → best params: {params_A}")
print(f"  B MultiCountry   → best params: {params_B}")
print(f"  C Lao+Weather    → best params: {params_C}")
