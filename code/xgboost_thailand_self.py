"""
Thailand dengue self-prediction with XGBoost (univariate, lag features only).

- Features : Thailand self-lag columns where Pearson corr >= 0.6
             with the 2022-2023 training target (lags 1-52).
- Train     : 2022-2023
- Test      : 2024
- Output    : analysis_7_xgboost_thailand_self/
"""
from __future__ import annotations

import csv
import math
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]  # repo root

TIMELAG_CSV = ROOT / "dataset/clean_data/Dengue_data/timelag_data/Thailand_Dengue_timelag_0_52_2020_2024.csv"
CORR_CSV = ROOT / "dataset/correlation/04_thailand_selflag_correlation_2022_2023.csv"

OUT_DIR = ROOT / "results/thailand_self"
PLOTS_DIR = OUT_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_YEARS = (2022, 2023)
TEST_YEAR   = 2024
CORR_THRESH = 0.6
SMOOTH_WIN  = 5

XGB_PARAMS = dict(
    n_estimators=400, max_depth=3, learning_rate=0.05,
    subsample=0.9, colsample_bytree=0.9,
    objective="reg:squarederror", random_state=42,
)


# ── helpers ────────────────────────────────────────────────────────────────────
def rmse(a, b): return math.sqrt(sum((x-y)**2 for x,y in zip(a,b))/len(a))
def mae(a, b):  return sum(abs(x-y) for x,y in zip(a,b))/len(a)
def r2(a, b):
    m=sum(a)/len(a); ss=sum((x-m)**2 for x in a); se=sum((x-y)**2 for x,y in zip(a,b))
    return 1-se/ss if ss else float("nan")

def smooth(y, win=SMOOTH_WIN):
    arr=np.asarray(y,dtype=float)
    if arr.size==0 or win<=1: return arr.tolist()
    pad=win//2
    return np.convolve(np.pad(arr,(pad,pad),mode="edge"),np.ones(win)/win,mode="valid").tolist()


def load_selected_lags():
    lags=[]
    with CORR_CSV.open() as f:
        for r in csv.DictReader(f):
            if r["pass_0.6"].strip().lower()=="true":
                lags.append(int(r["self_lag"]))
    return sorted(lags)


def load_data(lags):
    """Return dict[(year,week)] -> (count, [feat_values])."""
    data={}
    with TIMELAG_CSV.open() as f:
        for r in csv.DictReader(f):
            y,w=int(r["Year"]),int(r["Week_no"])
            try:
                count=float(r["Count"])
                feats=[float(r[f"lag_{l}"]) for l in lags]
            except (ValueError,KeyError):
                continue
            data[(y,w)]=(count,feats)
    return data


def main():
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from xgboost import XGBRegressor

    # ── features ──────────────────────────────────────────────────────────────
    lags = load_selected_lags()
    print(f"Selected lags (corr ≥ {CORR_THRESH}): {lags}  ({len(lags)} features)")

    data = load_data(lags)

    tr_keys = sorted(k for k in data if k[0] in TRAIN_YEARS)
    te_keys = sorted((k for k in data if k[0]==TEST_YEAR), key=lambda x: x[1])

    X_tr = [data[k][1] for k in tr_keys]
    y_tr = [data[k][0] for k in tr_keys]
    X_te = [data[k][1] for k in te_keys]
    y_te = [data[k][0] for k in te_keys]
    weeks = [k[1] for k in te_keys]

    # ── train & predict ───────────────────────────────────────────────────────
    model = XGBRegressor(**XGB_PARAMS)
    model.fit(X_tr, y_tr)
    y_pred = [float(v) for v in model.predict(X_te)]

    metrics = {
        "rmse": round(rmse(y_te, y_pred), 2),
        "mae":  round(mae(y_te, y_pred), 2),
        "r2":   round(r2(y_te, y_pred), 4),
        "n_features": len(lags),
        "lags": "|".join(map(str, lags)),
        "train_years": f"{min(TRAIN_YEARS)}-{max(TRAIN_YEARS)}",
        "test_year":  TEST_YEAR,
    }
    print(f"RMSE {metrics['rmse']}  MAE {metrics['mae']}  R2 {metrics['r2']}")

    # ── save metrics ──────────────────────────────────────────────────────────
    with (OUT_DIR / "metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        w.writeheader(); w.writerow(metrics)

    # ── save predictions ──────────────────────────────────────────────────────
    with (OUT_DIR / "predictions_2024.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["week_no", "actual", "predicted"])
        for i in range(len(weeks)):
            w.writerow([weeks[i], round(y_te[i],1), round(y_pred[i],1)])

    # ── feature importance ────────────────────────────────────────────────────
    with (OUT_DIR / "feature_importance.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["lag", "importance"])
        for lag, imp in sorted(zip(lags, model.feature_importances_),
                               key=lambda x: -x[1]):
            w.writerow([lag, round(float(imp), 6)])

    # ── plots (Actual vs Predicted only) ─────────────────────────────────────
    for tag, transform in (("actual_vs_predicted", lambda v: v),
                           ("actual_vs_predicted_smooth", smooth)):
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(weeks, transform(y_te),   label="Actual",    linewidth=2.4, color="#1f3b73")
        ax.plot(weeks, transform(y_pred), label="Predicted", linewidth=2.0, color="#e67e22")
        ax.set_title(
            f"Thailand dengue 2024 — XGBoost self-prediction  |  "
            f"RMSE {metrics['rmse']:.0f}  MAE {metrics['mae']:.0f}  R2 {metrics['r2']:.2f}",
            fontsize=11, loc="left")
        ax.set_xlabel("Week_no"); ax.set_ylabel("cases")
        ax.legend(loc="upper right"); ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / f"{tag}.png", dpi=150)
        plt.close(fig)

    print("Done →", OUT_DIR.relative_to(WORKSPACE))


if __name__ == "__main__":
    main()
