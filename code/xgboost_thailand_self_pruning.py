"""
Thailand XGBoost self-prediction — iterative feature pruning.

Mirrors the approach in analysis_1 (run_final_pruning.py) but uses only
Thailand's own self-lag features (corr >= 0.6, 2022-2023).

Pruning steps:
  step 0  = all selected lags
  step k  = remove the lag with lowest XGBoost importance, re-evaluate
  ...
  last    = single lag remaining

Records RMSE / MAE / R2 at every step, marks the best step (*BEST*).
Train 2022-2023, Test 2024.

Output: analysis_9_xgboost_selflag_pruning/
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

OUT_DIR   = ROOT / "results/thailand_self_pruning"
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
    m  = sum(a)/len(a)
    ss = sum((x-m)**2 for x in a)
    se = sum((x-y)**2 for x,y in zip(a,b))
    return 1 - se/ss if ss else float("nan")

def smooth(y, win=SMOOTH_WIN):
    arr = np.asarray(y, dtype=float)
    pad = win // 2
    return np.convolve(np.pad(arr, (pad, pad), mode="edge"),
                       np.ones(win)/win, mode="valid").tolist()


def load_selected_lags():
    lags = []
    with CORR_CSV.open() as f:
        for r in csv.DictReader(f):
            if r["pass_0.6"].strip().lower() == "true":
                lags.append(int(r["self_lag"]))
    return sorted(lags)


def load_data(lags):
    data = {}
    with TIMELAG_CSV.open() as f:
        for r in csv.DictReader(f):
            y, w = int(r["Year"]), int(r["Week_no"])
            try:
                count = float(r["Count"])
                feats = {l: float(r[f"lag_{l}"]) for l in lags}
            except (ValueError, KeyError):
                continue
            data[(y, w)] = (count, feats)
    return data


def build_xy(keys, data, lags):
    X, y = [], []
    for k in keys:
        if k not in data:
            continue
        count, feats = data[k]
        X.append([feats[l] for l in lags])
        y.append(count)
    return X, y


def main():
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from xgboost import XGBRegressor

    all_lags = load_selected_lags()
    print(f"Starting lags ({len(all_lags)}): {all_lags}")

    data = load_data(all_lags)
    tr_keys = sorted(k for k in data if k[0] in TRAIN_YEARS)
    te_keys = sorted((k for k in data if k[0] == TEST_YEAR), key=lambda x: x[1])
    y_actual = [data[k][0] for k in te_keys]
    weeks    = [k[1] for k in te_keys]

    cur_lags   = list(all_lags)
    step_rows  = []
    preds_dict = {"week_no": weeks, "actual": y_actual}
    best       = None
    step       = 0

    # ── pruning loop ───────────────────────────────────────────────────────────
    while True:
        X_tr, y_tr = build_xy(tr_keys, data, cur_lags)
        X_te, _    = build_xy(te_keys, data, cur_lags)

        model = XGBRegressor(**XGB_PARAMS)
        model.fit(X_tr, y_tr)
        y_pred = [float(v) for v in model.predict(X_te)]

        row = {
            "step":         step,
            "n_lags":       len(cur_lags),
            "lags":         "|".join(f"lag_{l}" for l in cur_lags),
            "rmse":         round(rmse(y_actual, y_pred), 2),
            "mae":          round(mae(y_actual, y_pred), 2),
            "r2":           round(r2(y_actual, y_pred), 4),
            "removed_next": "",
        }
        step_rows.append(row)
        col_key = f"step{step}_{'|'.join(map(str,cur_lags))}"
        preds_dict[col_key] = [round(v, 1) for v in y_pred]

        if best is None or row["rmse"] < best["rmse"]:
            best = dict(row); best["pred_key"] = col_key

        if len(cur_lags) == 1:
            break

        # remove least-important lag
        imp = model.feature_importances_
        idx = int(np.argmin(imp))
        removed = cur_lags[idx]
        step_rows[-1]["removed_next"] = f"lag_{removed}"
        cur_lags.pop(idx)
        step += 1

    # ── save pruning steps ─────────────────────────────────────────────────────
    with (OUT_DIR / "pruning_steps.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(step_rows[0].keys()))
        w.writeheader(); w.writerows(step_rows)

    # ── save predictions ───────────────────────────────────────────────────────
    pred_cols = list(preds_dict.keys())
    with (OUT_DIR / "predictions_2024.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(pred_cols)
        for i in range(len(weeks)):
            w.writerow([preds_dict[c][i] for c in pred_cols])

    # ── per-step subplot ───────────────────────────────────────────────────────
    n = len(step_rows)
    fig, axes = plt.subplots(n, 1, figsize=(12, 3.0*n), sharex=True)
    if n == 1: axes = [axes]

    for ax, r in zip(axes, step_rows):
        col  = f"step{r['step']}_{'|'.join(r['lags'].replace('lag_','').split('|'))}"
        # rebuild key the same way as stored
        lags_in_step = [int(x) for x in r["lags"].replace("lag_","").split("|")]
        col = f"step{r['step']}_{'|'.join(map(str,lags_in_step))}"
        yp   = preds_dict[col]
        star = "  *BEST*" if r["step"] == best["step"] else ""
        ax.plot(weeks, smooth(y_actual), label="Actual",    linewidth=2.4, color="#1f3b73")
        ax.plot(weeks, smooth(yp),       label="Predicted", linewidth=2.0, color="#e67e22")
        ax.set_title(
            f"{r['lags']}  |  RMSE {r['rmse']:.0f}  MAE {r['mae']:.0f}  R2 {r['r2']:.2f}{star}",
            fontsize=9, loc="left")
        ax.set_ylabel("cases"); ax.legend(loc="upper right", fontsize=8); ax.grid(alpha=0.3)

    axes[-1].set_xlabel("Week_no")
    fig.suptitle("Thailand dengue 2024 — XGBoost self-lag pruning", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    fig.savefig(PLOTS_DIR / "actual_vs_predicted_per_step.png", dpi=150)
    plt.close(fig)

    # ── best-step standalone plot ──────────────────────────────────────────────
    for tag, tf in (("actual_vs_predicted_best", lambda v: v),
                    ("actual_vs_predicted_best_smooth", smooth)):
        yp_best = preds_dict[best["pred_key"]]
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(weeks, tf(y_actual), label="Actual",    linewidth=2.4, color="#1f3b73")
        ax.plot(weeks, tf(yp_best),  label="Predicted", linewidth=2.0, color="#e67e22")
        ax.set_title(
            f"Thailand dengue 2024 — XGBoost self-lag *BEST*  ({best['lags']})  |  "
            f"RMSE {best['rmse']:.0f}  MAE {best['mae']:.0f}  R2 {best['r2']:.2f}",
            fontsize=11, loc="left")
        ax.set_xlabel("Week_no"); ax.set_ylabel("cases")
        ax.legend(loc="upper right"); ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / f"{tag}.png", dpi=150)
        plt.close(fig)

    # ── summary ────────────────────────────────────────────────────────────────
    print(f"\n{'step':>5}{'n_lags':>8}{'RMSE':>9}{'MAE':>9}{'R2':>8}  lags")
    for r in step_rows:
        star = " *BEST*" if r["step"] == best["step"] else ""
        print(f"{r['step']:>5}{r['n_lags']:>8}{r['rmse']:>9}{r['mae']:>9}{r['r2']:>8}{star}  {r['lags']}")
    print(f"\nDone → {OUT_DIR.relative_to(WORKSPACE)}")


if __name__ == "__main__":
    main()
