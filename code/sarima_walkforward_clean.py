"""
SARIMA Walk-Forward Validation — Clean Version
================================================
Box-Jenkins methodology, 4 steps:

  Step 1  Seasonal pattern  →  m = 52
  Step 2  ADF Test          →  d = 1, D = 1
  Step 3  ACF / PACF        →  p, q, P, Q (initial guess)
  Step 4  Grid Search       →  train 2021-2022, validate 2023  →  best params
  Final   Production        →  train 2022-2023, predict 2024  (same params)

Output: analysis_9_sarima_walkforward/
"""
from __future__ import annotations
import csv, math, warnings
from itertools import product
from pathlib import Path

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings("ignore")

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).resolve().parents[1]  # repo root
THAILAND_CSV = ROOT / "dataset/clean_data/Dengue_data/normal_data/Thailand_Dengue.csv"
OUT          = ROOT / "results/sarima"
OUT.mkdir(parents=True, exist_ok=True)

SEASONAL_M = 52
SMOOTH_W   = 4

# ── helpers ────────────────────────────────────────────────────────────────────
def load():
    d = {}
    with THAILAND_CSV.open() as f:
        for r in csv.DictReader(f):
            d[(int(r["Year"]), int(r["Week_no"]))] = float(r["Count"])
    return d

def get(data, years):
    ks = sorted(k for k in data if k[0] in years)
    return [data[k] for k in ks]

def smooth(y, w=SMOOTH_W):
    return np.convolve(np.array(y, dtype=float), np.ones(w)/w, mode="same").tolist()

def rmse(a, b): return math.sqrt(sum((x-y)**2 for x,y in zip(a,b))/len(a))
def r2(a, b):
    m = sum(a)/len(a); ss = sum((x-m)**2 for x in a)
    se = sum((x-y)**2 for x,y in zip(a,b))
    return (1 - se/ss) if ss else float("nan")

# ── load ───────────────────────────────────────────────────────────────────────
data = load()
years_all = sorted({k[0] for k in data})

# per-year totals for seasonality bar
week_avgs = {}
for (yr, wk), v in data.items():
    if 2013 <= yr <= 2022:
        week_avgs.setdefault(wk, []).append(v)
avg_by_week = {wk: np.mean(vs) for wk, vs in week_avgs.items()}
weeks_sorted = sorted(avg_by_week); avg_vals = [avg_by_week[w] for w in weeks_sorted]

# series for ADF / ACF / PACF (2013-2022)
y_hist = np.array(get(data, range(2013, 2023)), dtype=float)

# series for validation & production
y_val_train = get(data, [2021, 2022])
y_prod_train = get(data, [2022, 2023])
weeks23 = sorted(k[1] for k in data if k[0] == 2023)
weeks24 = sorted(k[1] for k in data if k[0] == 2024)
y_2023  = [data[(2023, w)] for w in weeks23]
y_2024  = [data[(2024, w)] for w in weeks24]

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Seasonal Pattern
# ═══════════════════════════════════════════════════════════════════════════════
print("Step 1: Seasonal pattern  →  m=52")

fig1, ax = plt.subplots(figsize=(11, 4))
ax.bar(weeks_sorted, avg_vals, color="#3498db", alpha=0.75, width=0.9)
peak_wk = weeks_sorted[np.argmax(avg_vals)]
ax.axvline(peak_wk, color="#e74c3c", lw=1.5, ls="--", label=f"Peak week {peak_wk}")
ax.set_xlabel("Week_no", fontsize=11); ax.set_ylabel("Avg cases (2013-2022)", fontsize=11)
ax.set_title("Step 1 — Average Weekly Pattern (2013–2022)\n"
             "→ Clear annual cycle  →  Seasonal period  m = 52", fontsize=11, fontweight="bold")
ax.legend(fontsize=9); ax.grid(alpha=0.3, axis="y")
fig1.tight_layout()
fig1.savefig(OUT/"step1_seasonal_pattern.png", dpi=150)
plt.close(fig1)
print("  Saved: step1_seasonal_pattern.png")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — ADF Test
# ═══════════════════════════════════════════════════════════════════════════════
from statsmodels.tsa.stattools import adfuller, acf, pacf

print("\nStep 2: ADF Tests")

def run_adf(series, label):
    r   = adfuller(series, autolag="AIC")
    ok  = r[1] < 0.05
    sta = "Stationary ✓" if ok else "Non-stationary ✗"
    print(f"  {label:42s}  p={r[1]:.4f}  {sta}")
    return {"series": label, "adf_stat": round(r[0],3), "p_value": round(r[1],4),
            "stationary": ok, "result": sta}

y_d1   = np.diff(y_hist)
y_sd   = np.array([y_hist[i]-y_hist[i-SEASONAL_M] for i in range(SEASONAL_M, len(y_hist))])
y_both = np.diff(y_sd)

adf_rows = [
    run_adf(y_hist,  "Original (2013-2022)"),
    run_adf(y_d1,    "After d=1  (1st difference)"),
    run_adf(y_sd,    "After D=1  (seasonal diff m=52)"),
    run_adf(y_both,  "After d=1 + D=1  (both)"),
]

with (OUT/"step2_adf_test.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["series","adf_stat","p_value","stationary","result"])
    w.writeheader(); w.writerows(adf_rows)
print("  Saved: step2_adf_test.csv")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 — ACF / PACF
# ═══════════════════════════════════════════════════════════════════════════════
print("\nStep 3: ACF / PACF")

MAX_LAG  = 80
CI       = 1.96 / math.sqrt(len(y_both))
acf_v    = acf( y_both, nlags=MAX_LAG)
pacf_v   = pacf(y_both, nlags=MAX_LAG)
lags     = np.arange(MAX_LAG+1)

sig_acf  = [i for i,v in enumerate(acf_v)  if i > 0 and abs(v) > CI]
sig_pacf = [i for i,v in enumerate(pacf_v) if i > 0 and abs(v) > CI]

q_guess = max((l for l in sig_acf  if l <= 4), default=1)
p_guess = max((l for l in sig_pacf if l <= 4), default=1)
Q_guess = 1 if abs(acf_v[SEASONAL_M])  > CI else 0
P_guess = 1 if abs(pacf_v[SEASONAL_M]) > CI else 0
print(f"  ACF/PACF suggests: p={p_guess}, q={q_guess}, P={P_guess}, Q={Q_guess}")

fig3, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
for ax, vals, color, name, guess, pq in [
    (axes[0], acf_v,  "#27ae60", "ACF",  f"→ q={q_guess}  Q={Q_guess}", "q, Q"),
    (axes[1], pacf_v, "#e74c3c", "PACF", f"→ p={p_guess}  P={P_guess}", "p, P"),
]:
    ax.bar(lags, vals, color=color, alpha=0.7, width=0.8)
    ax.axhline( CI, color="red", lw=1.3, ls="--", label=f"±95% CI (±{CI:.3f})")
    ax.axhline(-CI, color="red", lw=1.3, ls="--")
    ax.axvline(SEASONAL_M, color="orange", lw=1.8, ls=":", label=f"Lag {SEASONAL_M} (seasonal)")
    sig = sig_acf if name == "ACF" else sig_pacf
    ax.set_title(f"Step 3 — {name}  (stationary series d=1, D=1)\n"
                 f"Significant lags: {sig[:8]}…   {guess}", loc="left", fontsize=10, fontweight="bold")
    ax.set_ylabel(name); ax.legend(fontsize=8, loc="upper right"); ax.grid(alpha=0.3)
axes[1].set_xlabel("Lag (weeks)", fontsize=10)
fig3.suptitle("Step 3 — ACF & PACF  →  Initial Guess for p, q, P, Q",
              fontsize=12, fontweight="bold")
fig3.tight_layout()
fig3.savefig(OUT/"step3_acf_pacf.png", dpi=150)
plt.close(fig3)
print("  Saved: step3_acf_pacf.png")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Grid Search  (train 2021-2022, validate 2023)
# ═══════════════════════════════════════════════════════════════════════════════
from statsmodels.tsa.statespace.sarimax import SARIMAX

print("\nStep 4: Grid Search  (train 2021-2022 → validate 2023) ...")

SEA_ORDERS = [(0,1,0),(0,1,1),(1,1,0),(1,1,1)]
candidates = list(product(range(4), range(2), range(4), SEA_ORDERS))

grid_results = []
best = None
for p, d, q, (P, D, Q) in candidates:
    try:
        res = SARIMAX(y_val_train, order=(p,d,q), seasonal_order=(P,D,Q,SEASONAL_M),
                      enforce_stationarity=False, enforce_invertibility=False
                      ).fit(disp=False, maxiter=60, method="lbfgs")
        fc  = [float(v) for v in res.forecast(len(y_2023))]
        err = rmse(y_2023, fc)
        if not math.isfinite(err): continue
        row = {"rank": 0, "label": f"({p},{d},{q})({P},{D},{Q})",
               "order": (p,d,q), "sorder": (P,D,Q,SEASONAL_M),
               "rmse": round(err,1), "r2": round(r2(y_2023,fc),3), "fc": fc}
        grid_results.append(row)
        if best is None or err < best["rmse"]:
            best = row
            print(f"  New best  SARIMA{row['label']}  RMSE={err:.1f}")
    except: pass

grid_results.sort(key=lambda x: x["rmse"])
for i, row in enumerate(grid_results): row["rank"] = i+1

print(f"\n  Best → SARIMA{best['label']}  RMSE={best['rmse']}  R2={best['r2']}")

# save grid top 10
with (OUT/"step4_grid_search_top10.csv").open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["rank","label","rmse","r2"])
    for row in grid_results[:10]:
        w.writerow([row["rank"], row["label"], row["rmse"], row["r2"]])

# grid plot: horizontal bar top 15 + predicted lines
TOP15 = grid_results[:15]
SEL   = [grid_results[i] for i in [0,1,4,14,49] if i < len(grid_results)]
for row in SEL:
    if "fc" not in row:  # already have fc from grid search
        pass

clr15 = ["#e74c3c"]+["#e67e22"]*4+["#95a5a6"]*10
LCOLORS = ["#e74c3c","#e67e22","#f1c40f","#27ae60","#9b59b6"]
LSTYLES = ["-","--","-.",":","-"]

fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={"width_ratios":[1,1.2]})

# left: RMSE bar
ax4a.barh(range(15), [r["rmse"] for r in TOP15], color=clr15, height=0.72, edgecolor="white")
ax4a.set_yticks(range(15))
ax4a.set_yticklabels([f"#{r['rank']}  {r['label']}" for r in TOP15], fontsize=8.5)
ax4a.invert_yaxis()
ax4a.set_xlabel("RMSE  (validation 2023)  ↓ ยิ่งน้อยยิ่งดี", fontsize=9)
ax4a.set_title("Step 4 — RMSE top 15 candidates", loc="left", fontsize=10, fontweight="bold")
ax4a.axvline(grid_results[0]["rmse"], color="#e74c3c", lw=1.2, ls="--", alpha=0.7)
for i,(row,c) in enumerate(zip(TOP15, clr15)):
    ax4a.text(row["rmse"]+5, i, str(row["rmse"]), va="center", fontsize=7.5,
              color=c if c!="#95a5a6" else "#555")
ax4a.set_xlim(left=min(r["rmse"] for r in TOP15)-80)
ax4a.grid(alpha=0.25, axis="x")

# right: actual vs predicted
ax4b.plot(weeks23, smooth(y_2023), color="#1f3b73", lw=2.5, label="Actual 2023", zorder=10)
for i,row in enumerate(SEL):
    ax4b.plot(weeks23, smooth(row["fc"]),
              color=LCOLORS[i], lw=2.0 if i==0 else 1.5, ls=LSTYLES[i],
              label=f"Rank #{row['rank']}  {row['label']}  RMSE={row['rmse']}  R²={row['r2']}")
ax4b.set_title("Actual vs Predicted — validation 2023", loc="left", fontsize=10, fontweight="bold")
ax4b.set_xlabel("Week_no", fontsize=10); ax4b.set_ylabel("cases", fontsize=10)
ax4b.set_ylim(bottom=0); ax4b.legend(fontsize=8.5, loc="upper left"); ax4b.grid(alpha=0.3)

fig4.suptitle(f"Step 4 — Grid Search  (train 2021–2022 → validate 2023)\n"
              f"Best: SARIMA{best['label']}  RMSE={best['rmse']}  R²={best['r2']}",
              fontsize=11, fontweight="bold")
fig4.tight_layout()
fig4.savefig(OUT/"step4_grid_search.png", dpi=150)
plt.close(fig4)
print("  Saved: step4_grid_search.png")

# ═══════════════════════════════════════════════════════════════════════════════
# FINAL — Production  (train 2022-2023, predict 2024, same params)
# ═══════════════════════════════════════════════════════════════════════════════
print("\nFinal: Production  (train 2022-2023 → predict 2024) ...")

best_order  = best["order"]
best_sorder = best["sorder"]

res_prod = SARIMAX(y_prod_train, order=best_order, seasonal_order=best_sorder,
                   enforce_stationarity=False, enforce_invertibility=False
                   ).fit(disp=False, maxiter=60, method="lbfgs")
fc24 = [float(v) for v in res_prod.forecast(len(y_2024))]

prod_rmse = round(rmse(y_2024, fc24), 1)
prod_r2   = round(r2(y_2024, fc24),   3)
print(f"  Production RMSE={prod_rmse}  R²={prod_r2}")

# save predictions
with (OUT/"predictions_2023_validation.csv").open("w", newline="") as f:
    w = csv.writer(f); w.writerow(["week_no","actual","predicted"])
    for i,wk in enumerate(weeks23): w.writerow([wk, round(y_2023[i],1), round(best["fc"][i],1)])

with (OUT/"predictions_2024.csv").open("w", newline="") as f:
    w = csv.writer(f); w.writerow(["week_no","actual","predicted"])
    for i,wk in enumerate(weeks24): w.writerow([wk, round(y_2024[i],1), round(fc24[i],1)])

with (OUT/"metrics.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["phase","train","test","rmse","r2","param"])
    w.writeheader()
    w.writerow({"phase":"Validation","train":"2021-2022","test":"2023",
                "rmse":best["rmse"],"r2":best["r2"],"param":f"SARIMA{best['label']}"})
    w.writerow({"phase":"Production","train":"2022-2023","test":"2024",
                "rmse":prod_rmse,"r2":prod_r2,"param":f"SARIMA{best['label']}"})

# ─── final plot: both phases together ──────────────────────────────────────────
y2021 = get(data, [2021]); y2022 = get(data, [2022])
idx21 = list(range(1, len(y2021)+1))
idx22 = list(range(idx21[-1]+1, idx21[-1]+1+len(y2022)))
idx23 = list(range(idx22[-1]+1, idx22[-1]+1+len(y_2023)))
idx24 = list(range(idx23[-1]+1, idx23[-1]+1+len(y_2024)))
all_idx = idx21+idx22+idx23+idx24
all_act = y2021+y2022+y_2023+y_2024

fig5, ax5 = plt.subplots(figsize=(15, 6))
ax5.axvspan(idx21[0],    idx22[-1]+0.5,  alpha=0.07, color="#3498db")
ax5.axvspan(idx22[0],    idx23[-1]+0.5,  alpha=0.07, color="#e67e22")
ax5.axvspan(idx23[0]-0.5, idx23[-1]+0.5, alpha=0.10, color="#e74c3c")
ax5.axvspan(idx24[0]-0.5, idx24[-1],     alpha=0.10, color="#8e44ad")
for xi in [idx22[-1]+0.5, idx23[-1]+0.5]:
    ax5.axvline(xi, color="#555", lw=1.3, ls="--")

ax5.plot(all_idx, smooth(all_act, w=4), color="#1f3b73", lw=2.3, label="Actual", zorder=10)
ax5.plot(idx23, smooth(best["fc"], w=4), color="#e74c3c", lw=2.2,
         label=f"Predicted 2023 (validation)  RMSE={best['rmse']}  R²={best['r2']}")
ax5.plot(idx24, smooth(fc24, w=4), color="#8e44ad", lw=2.2,
         label=f"Predicted 2024 (production)   RMSE={prod_rmse}  R²={prod_r2}")

mid_x = (idx23[-1]+idx24[0])//2
ax5.annotate("", xy=(idx24[0]+2, 2800), xytext=(idx23[-1]-2, 2800),
             arrowprops=dict(arrowstyle="<->", color="#555", lw=1.5))
ax5.text(mid_x, 3050, f"Same parameter\nSARIMA{best['label']}",
         ha="center", fontsize=8.5, color="#333",
         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#aaa"))

for xpos, lbl, col in [
    (np.mean(idx21+idx22), "TRAIN (val)\n2021-2022", "#2980b9"),
    (np.mean(idx22+idx23), "TRAIN (prod)\n2022-2023", "#d35400"),
    (np.mean(idx23),       "TEST 2023",  "#c0392b"),
    (np.mean(idx24),       "TEST 2024",  "#6c3483"),
]:
    ax5.text(xpos, -350, lbl, ha="center", fontsize=8.5, color=col, fontweight="bold")

tick_p = [idx21[0], idx22[0], idx23[0], idx24[0], idx24[-1]]
tick_l = ["W1 2021","W1 2022","W1 2023","W1 2024","W52 2024"]
ax5.set_xticks(tick_p); ax5.set_xticklabels(tick_l, fontsize=9)
ax5.set_ylabel("cases", fontsize=11); ax5.set_ylim(bottom=-500)
ax5.set_title(f"SARIMA Walk-Forward Validation — Final Result\n"
              f"Parameter: SARIMA{best['label']}   "
              f"Validation RMSE={best['rmse']} R²={best['r2']}   "
              f"Production RMSE={prod_rmse} R²={prod_r2}",
              fontsize=11, fontweight="bold")
ax5.legend(fontsize=9, loc="upper left", framealpha=0.93); ax5.grid(alpha=0.3)
fig5.tight_layout()
fig5.savefig(OUT/"final_walkforward.png", dpi=150)
plt.close(fig5)
print("  Saved: final_walkforward.png")

# ── summary ────────────────────────────────────────────────────────────────────
print(f"""
══════════════════════════════════════════════
 SARIMA Walk-Forward Validation — Summary
══════════════════════════════════════════════
 Best parameter : SARIMA{best['label']}
   m=52   d=1 D=1   p={best_order[0]} q={best_order[2]}   P={best_sorder[0]} Q={best_sorder[2]}

 Validation  (2021-22 → 2023) : RMSE={best['rmse']}  R²={best['r2']}
 Production  (2022-23 → 2024) : RMSE={prod_rmse}  R²={prod_r2}

 Output → {OUT.relative_to(WORKSPACE)}
══════════════════════════════════════════════
""")
