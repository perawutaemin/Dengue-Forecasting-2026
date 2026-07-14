"""
Per-country XGB for all 16 Multi-Country countries:
  1) Screen count & weather features with |corr| >= 0.7 vs Thailand (2022-2023)
  2) Iteratively prune by XGB feature_importance (drop lowest weather first)
  3) Compare best pruned model vs SARIMA baseline on 2024

Train 2022-2023 → Test 2024.
"""
from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]  # repo root
OUT = ROOT / "results/prune_vs_sarima"
DENGUE_TL = ROOT / "dataset/clean_data/Dengue_data/timelag_data"
DENGUE_N = ROOT / "dataset/clean_data/Dengue_data/normal_data"
W2019_2024 = ROOT / "dataset/clean_data/Weather_data_2019_2024"
WDIR = ROOT / "dataset/clean_data/Weather_data"
SARIMA_PRED = ROOT / "dengue_final_results/analysis_8_sarima_walkforward/predictions_2024.csv"

SCREEN = OUT / "screening"
COUNTRIES_DIR = OUT / "countries"
SUMMARY = OUT / "summary"
SVG = OUT / "svg"
for d in (SCREEN, COUNTRIES_DIR, SUMMARY, SVG):
    d.mkdir(parents=True, exist_ok=True)

COUNTRIES = [
    "Peru",
    "Dominican Republic",
    "Lao PeopleS Democratic Republic",
    "Bolivia",
    "South Korea",
    "Puerto Rico",
    "Nicaragua",
    "United States Of America",
    "Argentina",
    "Guatemala",
    "Paraguay",
    "Samoa",
    "Costa Rica",
    "Northern Mariana Islands",
    "Kiribati",
    "Vanuatu",
]

# dengue file stem overrides (Korea has no timelag file)
DENGUE_STEM = {
    "South Korea": None,  # build from Korea_Dengue.csv
}
WEATHER_NAME = {
    "South Korea": "South Korea",
}

CORR_YEARS = (2022, 2023)
TRAIN_YEARS = (2022, 2023)
TEST_YEAR = 2024
MAX_LAG = 52
MIN_COUNT_LAG = 2
CORR_THR = 0.7
WEATHER_THR = 0.6  # keep weather gate softer than country count
WEATHER_VARS = ("TMAX", "TMIN", "TAVG", "PRCP")

XGB_PARAMS = dict(
    n_estimators=400,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    objective="reg:squarederror",
    random_state=42,
)

# light retune on best feature set (val 2023 → prod 2024)
TUNE_GRID = [
    dict(n_estimators=n, max_depth=d, learning_rate=lr, subsample=0.9, colsample_bytree=0.9,
         objective="reg:squarederror", random_state=42)
    for n in (200, 400, 600)
    for d in (2, 3)
    for lr in (0.02, 0.05, 0.1)
]


def slug(c: str) -> str:
    return c.replace(" ", "_").lower().replace("'", "")


def rmse(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)) / len(a))


def mae(a, b):
    return sum(abs(x - y) for x, y in zip(a, b)) / len(a)


def pearson(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if len(x) < 3:
        return float("nan")
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def smooth(y, win=5):
    arr = np.asarray(y, float)
    pad = win // 2
    return np.convolve(np.pad(arr, (pad, pad), mode="edge"), np.ones(win) / win, mode="valid")


def chron_keys(years):
    return [(y, w) for y in years for w in range(1, 53)]


def fill_series(series: dict, keys: list):
    """Neighbor-average fill for missing weeks (prev/next available)."""
    vals = []
    for k in keys:
        v = series.get(k)
        vals.append(float(v) if v is not None and str(v).strip() != "" else np.nan)
    arr = np.asarray(vals, float)
    n = len(arr)
    for i in range(n):
        if not np.isnan(arr[i]):
            continue
        lo = i - 1
        while lo >= 0 and np.isnan(arr[lo]):
            lo -= 1
        hi = i + 1
        while hi < n and np.isnan(arr[hi]):
            hi += 1
        if lo >= 0 and hi < n:
            arr[i] = 0.5 * (arr[lo] + arr[hi])
        elif lo >= 0:
            arr[i] = arr[lo]
        elif hi < n:
            arr[i] = arr[hi]
        else:
            arr[i] = 0.0
    return {k: float(arr[i]) for i, k in enumerate(keys)}


def load_thailand():
    fp = DENGUE_N / "Thailand_Dengue.csv"
    out = {}
    with fp.open() as f:
        for r in csv.DictReader(f):
            out[(int(r["Year"]), int(r["Week_no"]))] = float(r["Count"])
    return out


def load_country_count_raw(country: str) -> dict:
    """Weekly count keyed by (year, week)."""
    if country == "South Korea":
        fp = DENGUE_N / "Korea_Dengue.csv"
        out = {}
        with fp.open() as f:
            for r in csv.DictReader(f):
                out[(int(r["Year"]), int(r["Week_no"]))] = float(r["Count"])
        return out

    stem = country.replace(" ", "_")
    fp = DENGUE_TL / f"{stem}_Dengue_timelag_0_52_2020_2024.csv"
    if not fp.exists():
        # fallback normal_data
        fp2 = DENGUE_N / f"{stem}_Dengue.csv"
        out = {}
        with fp2.open() as f:
            for r in csv.DictReader(f):
                out[(int(r["Year"]), int(r["Week_no"]))] = float(r["Count"])
        return out
    out = {}
    with fp.open() as f:
        for r in csv.DictReader(f):
            out[(int(r["Year"]), int(r["Week_no"]))] = float(r["Count"])
    return out


def find_weather_file(country: str) -> Path | None:
    name = WEATHER_NAME.get(country, country)
    for base in (W2019_2024, WDIR):
        for suffix in ("_Weather_Weekly_2019_2024.csv", "_Weather_Weekly_2019_2023.csv"):
            fp = base / f"{name}{suffix}"
            if fp.exists():
                return fp
    # Korea cleaned fallback
    if country == "South Korea":
        cleaned = WDIR / "South_Korea_Weather_Weekly_2019_2024_cleaned.csv"
        if cleaned.exists():
            return cleaned
    return None


def load_weather_raw(country: str) -> dict:
    """{(year,week): {TMAX,TMIN,TAVG,PRCP}} with NaN for missing."""
    fp = find_weather_file(country)
    if fp is None:
        return {}
    out = {}
    with fp.open() as f:
        for r in csv.DictReader(f):
            y, w = int(float(r["Year"])), int(float(r["Week_no"]))
            row = {}
            for v in WEATHER_VARS:
                s = (r.get(v) or "").strip()
                row[v] = float(s) if s and s.lower() not in ("nan", "none", "na") else np.nan
            out[(y, w)] = row
    return out


def lag_map(series: dict, lag: int, keys: list) -> dict:
    """Map target key -> value from series at key shifted back by lag weeks in chron order."""
    # Build full chronology covering needed range
    years = sorted({y for y, _ in series} | {y for y, _ in keys})
    if not years:
        return {}
    full = chron_keys(range(min(years) - 1, max(years) + 1))
    # index of each key
    idx = {k: i for i, k in enumerate(full)}
    filled = fill_series(series, full)
    out = {}
    for k in keys:
        if k not in idx:
            continue
        j = idx[k] - lag
        if j < 0:
            continue
        src = full[j]
        out[k] = filled[src]
    return out


def best_lag_corr(thai: dict, series: dict, years, min_lag: int, max_lag: int):
    keys = chron_keys(years)
    y = [thai.get(k, np.nan) for k in keys]
    best = (None, float("nan"))
    for lag in range(min_lag, max_lag + 1):
        xm = lag_map(series, lag, keys)
        pairs = [(xm[k], thai[k]) for k in keys if k in xm and k in thai]
        if len(pairs) < 20:
            continue
        xs, ys = zip(*pairs)
        c = pearson(xs, ys)
        if np.isnan(c):
            continue
        if best[0] is None or abs(c) > abs(best[1]):
            best = (lag, c)
    return best


def screen_country(country: str, thai: dict):
    count_raw = load_country_count_raw(country)
    count_lag, count_corr = best_lag_corr(thai, count_raw, CORR_YEARS, MIN_COUNT_LAG, MAX_LAG)

    weather_raw = load_weather_raw(country)
    weather_hits = []
    weather_file = find_weather_file(country)
    for var in WEATHER_VARS:
        series = {k: row[var] for k, row in weather_raw.items()}
        # drop all-nan
        if not series or all(np.isnan(v) if isinstance(v, float) else False for v in series.values()):
            continue
        # convert nan to missing for fill_series
        series2 = {k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in series.items()}
        lag, corr = best_lag_corr(thai, series2, CORR_YEARS, 0, MAX_LAG)
        if lag is None or np.isnan(corr):
            continue
        weather_hits.append({"variable": var, "lag": lag, "corr": corr, "pass": abs(corr) >= WEATHER_THR})

    count_pass = count_lag is not None and not np.isnan(count_corr) and abs(count_corr) >= CORR_THR
    weather_pass = [w for w in weather_hits if w["pass"]]
    keep = count_pass or len(weather_pass) > 0

    return {
        "country": country,
        "weather_file": weather_file.name if weather_file else "",
        "count_lag": count_lag,
        "count_corr": count_corr,
        "count_pass": count_pass,
        "weather_all": weather_hits,
        "weather_pass": weather_pass,
        "keep": keep,
        "count_raw": count_raw,
        "weather_raw": weather_raw,
    }


def build_feature_maps(spec, feature_list, keys):
    """feature_list: list of (name, kind, var_or_None, lag)"""
    maps = []
    names = []
    for name, kind, var, lag in feature_list:
        if kind == "count":
            series = spec["count_raw"]
            m = lag_map(series, lag, keys)
        else:
            series = {
                k: (None if (isinstance(row[var], float) and np.isnan(row[var])) else row[var])
                for k, row in spec["weather_raw"].items()
                if var in row
            }
            m = lag_map(series, lag, keys)
        maps.append(m)
        names.append(name)
    return names, maps


def build_xy(keys, target, maps):
    X, y, used = [], [], []
    for k in keys:
        if k not in target:
            continue
        if any(k not in m for m in maps):
            continue
        X.append([m[k] for m in maps])
        y.append(target[k])
        used.append(k)
    return np.asarray(X, float), np.asarray(y, float), used


def fit_predict(Xtr, ytr, Xte, params=None):
    p = dict(params or XGB_PARAMS)
    model = XGBRegressor(**p)
    model.fit(Xtr, ytr)
    return model, list(map(float, model.predict(Xte)))


def prune_country(spec, thai):
    """Return pruning_steps, best_row, predictions dict, feature lists."""
    features = []
    if spec["count_pass"]:
        features.append(
            (f"count_lag_{spec['count_lag']}", "count", None, int(spec["count_lag"]))
        )
    for w in spec["weather_pass"]:
        features.append(
            (f"{w['variable']}_lag_{w['lag']}", "weather", w["variable"], int(w["lag"]))
        )

    if not features:
        return [], None, None, []

    keys_tr = chron_keys(TRAIN_YEARS)
    keys_te = chron_keys([TEST_YEAR])
    y_actual_full = [thai[k] for k in keys_te if k in thai]
    weeks_full = [k[1] for k in keys_te if k in thai]

    cur = list(features)
    steps = []
    preds = {"week_no": weeks_full, "actual": y_actual_full}
    best = None
    step = 0

    while cur:
        names, maps = build_feature_maps(spec, cur, keys_tr + keys_te)
        Xtr, ytr, _ = build_xy(keys_tr, thai, maps)
        Xte, yte, used_te = build_xy(keys_te, thai, maps)
        if len(Xtr) < 20 or len(Xte) < 10:
            break
        model, yp = fit_predict(Xtr, ytr, Xte)
        # align yp to full weeks if some missing
        pred_by_week = {used_te[i][1]: yp[i] for i in range(len(used_te))}
        yp_aligned = [pred_by_week.get(w, float("nan")) for w in weeks_full]
        # metrics on non-nan
        pairs = [(a, p) for a, p in zip(y_actual_full, yp_aligned) if not np.isnan(p)]
        a_m, p_m = zip(*pairs) if pairs else ([], [])
        row = {
            "step": step,
            "n_features": len(cur),
            "features": "|".join(f[0] for f in cur),
            "feature_kinds": "|".join(f[1] for f in cur),
            "rmse": round(rmse(a_m, p_m), 2) if pairs else float("nan"),
            "mae": round(mae(a_m, p_m), 2) if pairs else float("nan"),
            "n_test": len(pairs),
            "removed_next": "",
            "importances": "",
        }
        imps = list(map(float, model.feature_importances_))
        row["importances"] = "|".join(f"{n}:{imp:.4f}" for n, imp in zip(names, imps))
        steps.append(row)
        preds[f"step{step}"] = [None if np.isnan(v) else round(v, 1) for v in yp_aligned]

        if best is None or row["rmse"] < best["rmse"]:
            best = dict(row)
            best["feature_list"] = list(cur)
            best["pred"] = yp_aligned
            best["imps"] = list(zip(names, imps))

        if len(cur) == 1:
            break

        # Drop lowest-importance weather first; if no weather left, drop lowest any
        weather_idx = [i for i, f in enumerate(cur) if f[1] == "weather"]
        if weather_idx:
            local = [(i, imps[i]) for i in weather_idx]
            drop_i = min(local, key=lambda t: t[1])[0]
        else:
            drop_i = int(np.argmin(imps))
        row["removed_next"] = cur[drop_i][0]
        steps[-1]["removed_next"] = row["removed_next"]
        del cur[drop_i]
        step += 1

    return steps, best, preds, features


def retune_best(spec, best, thai):
    """Small grid on best features: tune on 2023 (train 2021-2022), apply 2022-2023→2024."""
    if best is None:
        return None
    feats = best["feature_list"]
    keys_tune_tr = chron_keys([2021, 2022])
    keys_tune_va = chron_keys([2023])
    keys_prod_tr = chron_keys([2022, 2023])
    keys_prod_te = chron_keys([2024])

    names, maps = build_feature_maps(spec, feats, chron_keys(range(2021, 2025)))
    Xtr, ytr, _ = build_xy(keys_tune_tr, thai, maps)
    Xva, yva, _ = build_xy(keys_tune_va, thai, maps)
    if len(Xtr) < 20 or len(Xva) < 10:
        return None

    best_p, best_val = None, float("inf")
    for p in TUNE_GRID:
        model, yp = fit_predict(Xtr, ytr, Xva, p)
        r = rmse(yva, yp)
        if r < best_val:
            best_val, best_p = r, p

    Xtr2, ytr2, _ = build_xy(keys_prod_tr, thai, maps)
    Xte2, yte2, used = build_xy(keys_prod_te, thai, maps)
    model, yp = fit_predict(Xtr2, ytr2, Xte2, best_p)
    weeks = [k[1] for k in used]
    return {
        "params": best_p,
        "val_rmse_2023": round(best_val, 2),
        "rmse": round(rmse(yte2, yp), 2),
        "mae": round(mae(yte2, yp), 2),
        "weeks": weeks,
        "actual": list(map(float, yte2)),
        "pred": yp,
        "features": "|".join(f[0] for f in feats),
        "n_features": len(feats),
        "imps": list(zip(names, map(float, model.feature_importances_))),
    }


def load_sarima():
    rows = list(csv.DictReader(SARIMA_PRED.open()))
    weeks = [int(float(r["week_no"])) for r in rows]
    actual = [float(r["actual"]) for r in rows]
    pred = [float(r["predicted"]) for r in rows]
    return weeks, actual, pred


def save_both(fig, png_path: Path):
    import matplotlib.pyplot as plt

    png_path = Path(png_path)
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    fig.savefig(png_path.with_suffix(".svg"), format="svg", bbox_inches="tight")
    fig.savefig(SVG / png_path.name.replace(".png", ".svg"), format="svg", bbox_inches="tight")
    plt.close(fig)


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["axes.unicode_minus"] = True

    thai = load_thailand()
    sar_weeks, sar_actual, sar_pred = load_sarima()
    sar_rmse = rmse(sar_actual, sar_pred)
    sar_mae = mae(sar_actual, sar_pred)
    print(f"SARIMA baseline 2024: RMSE={sar_rmse:.1f} MAE={sar_mae:.1f}")

    screening_rows = []
    weather_detail_rows = []
    summary_rows = []

    for country in COUNTRIES:
        print(f"\n=== {country} ===")
        spec = screen_country(country, thai)
        cdir = COUNTRIES_DIR / slug(country)
        cdir.mkdir(parents=True, exist_ok=True)
        (cdir / "plots").mkdir(exist_ok=True)

        screening_rows.append(
            {
                "country": country,
                "keep": spec["keep"],
                "count_lag": spec["count_lag"],
                "count_corr": None if spec["count_corr"] is None or (isinstance(spec["count_corr"], float) and np.isnan(spec["count_corr"])) else round(float(spec["count_corr"]), 6),
                "count_pass": spec["count_pass"],
                "n_weather_pass": len(spec["weather_pass"]),
                "weather_pass": "|".join(f"{w['variable']}_lag{w['lag']}({w['corr']:.3f})" for w in spec["weather_pass"]),
                "weather_file": spec["weather_file"],
            }
        )
        for w in spec["weather_all"]:
            weather_detail_rows.append(
                {
                    "country": country,
                    "variable": w["variable"],
                    "best_lag": w["lag"],
                    "corr": round(w["corr"], 6),
                    "pass_0.6": w["pass"],
                }
            )

        if not spec["keep"]:
            print("  SKIP: no count/weather with |r|>=0.6")
            continue

        steps, best, preds, init_feats = prune_country(spec, thai)
        if best is None:
            print("  SKIP: could not build features")
            continue

        # save pruning steps
        with (cdir / "pruning_steps.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(steps[0].keys()))
            w.writeheader()
            w.writerows(steps)

        with (cdir / "predictions_pruning.csv").open("w", newline="") as f:
            cols = list(preds.keys())
            wr = csv.writer(f)
            wr.writerow(cols)
            for i in range(len(preds["week_no"])):
                wr.writerow([preds[c][i] for c in cols])

        tuned = retune_best(spec, best, thai)

        # choose final = tuned if better else prune-best
        if tuned and tuned["rmse"] <= best["rmse"]:
            final = {
                "source": "pruned+tuned",
                "rmse": tuned["rmse"],
                "mae": tuned["mae"],
                "features": tuned["features"],
                "n_features": tuned["n_features"],
                "params": tuned["params"],
                "val_rmse_2023": tuned["val_rmse_2023"],
                "weeks": tuned["weeks"],
                "actual": tuned["actual"],
                "pred": tuned["pred"],
                "imps": tuned["imps"],
            }
        else:
            final = {
                "source": "pruned",
                "rmse": best["rmse"],
                "mae": best["mae"],
                "features": best["features"],
                "n_features": best["n_features"],
                "params": XGB_PARAMS,
                "val_rmse_2023": "",
                "weeks": preds["week_no"],
                "actual": preds["actual"],
                "pred": best["pred"],
                "imps": best["imps"],
            }

        # save final predictions + importance
        with (cdir / "predictions_best.csv").open("w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["week_no", "actual", "predicted"])
            for w, a, p in zip(final["weeks"], final["actual"], final["pred"]):
                wr.writerow([w, a, None if p is None or (isinstance(p, float) and np.isnan(p)) else round(float(p), 1)])

        with (cdir / "feature_importance_best.csv").open("w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["feature", "importance"])
            for n, imp in final["imps"]:
                wr.writerow([n, round(imp, 6)])

        with (cdir / "best_config.csv").open("w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["key", "value"])
            for k in ("source", "rmse", "mae", "features", "n_features", "val_rmse_2023"):
                wr.writerow([k, final[k]])
            p = final["params"]
            wr.writerow(["n_estimators", p.get("n_estimators")])
            wr.writerow(["max_depth", p.get("max_depth")])
            wr.writerow(["learning_rate", p.get("learning_rate")])

        beat = final["rmse"] < sar_rmse
        summary_rows.append(
            {
                "country": country,
                "count_lag": spec["count_lag"],
                "count_corr": round(float(spec["count_corr"]), 4) if spec["count_pass"] else "",
                "weather_features": "|".join(f"{w['variable']}_lag_{w['lag']}" for w in spec["weather_pass"]),
                "best_features": final["features"],
                "n_features": final["n_features"],
                "source": final["source"],
                "rmse": final["rmse"],
                "mae": final["mae"],
                "sarima_rmse": round(sar_rmse, 2),
                "sarima_mae": round(sar_mae, 2),
                "delta_rmse_vs_sarima": round(final["rmse"] - sar_rmse, 2),
                "beats_sarima": beat,
                "n_estimators": final["params"].get("n_estimators"),
                "max_depth": final["params"].get("max_depth"),
                "learning_rate": final["params"].get("learning_rate"),
            }
        )
        print(
            f"  best RMSE={final['rmse']:.1f} MAE={final['mae']:.1f} "
            f"feats={final['features']} | vs SARIMA {sar_rmse:.1f} "
            f"{'✓ BEAT' if beat else '✗ worse'}"
        )

        # per-country plot: actual vs best vs SARIMA
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.plot(final["weeks"], smooth(final["actual"]), label="Actual", color="#1f3b73", lw=2.4)
        ax.plot(final["weeks"], smooth(final["pred"]), label=f"XGB best (RMSE {final['rmse']:.0f})", color="#e67e22", lw=2.0)
        ax.plot(sar_weeks, smooth(sar_pred), label=f"SARIMA (RMSE {sar_rmse:.0f})", color="#2ca02c", lw=1.8, alpha=0.85)
        ax.set_title(f"Thailand 2024 — {country}\n{final['features']}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Week"); ax.set_ylabel("Dengue Cases")
        ax.legend(loc="upper right"); ax.grid(alpha=0.3)
        fig.tight_layout()
        save_both(fig, cdir / "plots" / "actual_vs_xgb_vs_sarima.png")

        # pruning path plot
        if len(steps) > 1:
            fig, ax = plt.subplots(figsize=(9, 4.5))
            xs = [r["step"] for r in steps]
            ys = [r["rmse"] for r in steps]
            ax.plot(xs, ys, marker="o", color="#e67e22", lw=2)
            ax.axhline(sar_rmse, color="#2ca02c", ls="--", label=f"SARIMA {sar_rmse:.0f}")
            ax.axhline(best["rmse"], color="#e67e22", ls=":", alpha=0.5)
            best_step = next(r["step"] for r in steps if r["rmse"] == best["rmse"] and r["features"] == best["features"])
            ax.scatter([best_step], [best["rmse"]], s=120, zorder=5, color="red", label=f"best prune {best['rmse']:.0f}")
            ax.set_xlabel("Pruning step (drop lowest-importance weather)")
            ax.set_ylabel("RMSE 2024")
            ax.set_title(f"Feature pruning path — {country}")
            ax.legend(); ax.grid(alpha=0.3)
            fig.tight_layout()
            save_both(fig, cdir / "plots" / "pruning_path.png")

    # --- screening CSVs ---
    with (SCREEN / "country_screening.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(screening_rows[0].keys()))
        w.writeheader(); w.writerows(screening_rows)
    if weather_detail_rows:
        with (SCREEN / "weather_correlation_detail.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(weather_detail_rows[0].keys()))
            w.writeheader(); w.writerows(weather_detail_rows)

    # --- summary ---
    summary_rows.sort(key=lambda r: r["rmse"])
    with (SUMMARY / "all_countries_vs_sarima.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader(); w.writerows(summary_rows)

    # SARIMA row for reference
    with (SUMMARY / "baseline_sarima.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "rmse", "mae"])
        w.writerow(["SARIMA", round(sar_rmse, 2), round(sar_mae, 2)])

    n_beat = sum(1 for r in summary_rows if r["beats_sarima"])
    print(f"\n=== DONE: {n_beat}/{len(summary_rows)} countries beat SARIMA ===")
    for r in summary_rows:
        mark = "✓" if r["beats_sarima"] else " "
        print(f"  {mark} {r['country']:<32} RMSE={r['rmse']:8.1f}  Δ={r['delta_rmse_vs_sarima']:+8.1f}  {r['best_features']}")

    # bar chart
    fig, ax = plt.subplots(figsize=(12, 6.5))
    labels = [r["country"].replace("PeopleS Democratic Republic", "PDR") for r in summary_rows] + ["SARIMA"]
    vals = [r["rmse"] for r in summary_rows] + [sar_rmse]
    colors = ["#2ca02c" if r["beats_sarima"] else "#d62728" for r in summary_rows] + ["#1f77b4"]
    ypos = np.arange(len(labels))
    ax.barh(ypos, vals, color=colors, alpha=0.9)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.axvline(sar_rmse, color="#1f77b4", ls="--", alpha=0.7)
    ax.set_xlabel("RMSE (Thailand 2024)")
    ax.set_title(f"16 countries: best pruned XGB vs SARIMA\n(green = beats SARIMA; {n_beat}/{len(summary_rows)} beat)", fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    save_both(fig, SUMMARY / "rmse_bar_vs_sarima.png")

    # overlay top-5 beaters + SARIMA
    top = [r for r in summary_rows if r["beats_sarima"]][:5]
    if not top:
        top = summary_rows[:5]
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.plot(sar_weeks, smooth(sar_actual), label="Actual", color="#1f3b73", lw=2.6)
    ax.plot(sar_weeks, smooth(sar_pred), label=f"SARIMA ({sar_rmse:.0f})", color="#1f77b4", lw=2.0, ls="--")
    cmap = ["#e67e22", "#d62728", "#9467bd", "#8c564b", "#17becf"]
    for i, r in enumerate(top):
        cdir = COUNTRIES_DIR / slug(r["country"])
        rows = list(csv.DictReader((cdir / "predictions_best.csv").open()))
        weeks = [int(float(x["week_no"])) for x in rows]
        pred = [float(x["predicted"]) for x in rows]
        short = r["country"].replace("PeopleS Democratic Republic", "PDR")
        ax.plot(weeks, smooth(pred), label=f"{short} ({r['rmse']:.0f})", color=cmap[i % len(cmap)], lw=1.8)
    ax.set_title("Top models vs SARIMA — Thailand dengue 2024", fontweight="bold")
    ax.set_xlabel("Week"); ax.set_ylabel("Cases")
    ax.legend(loc="upper right", fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    save_both(fig, SUMMARY / "overlay_top_vs_sarima.png")

    # README
    readme = OUT / "README.md"
    lines = [
        "# 16 Countries — Feature Pruning vs SARIMA",
        "",
        "Per-country XGBoost using dengue count and/or weather features with **|corr| ≥ 0.7**",
        "vs Thailand (screening years **2022–2023**), then iterative **feature_importance** pruning,",
        "compared to **SARIMA** baseline on Thailand **2024**.",
        "",
        "## Protocol",
        "- Count lag screen: lag 2–52; Weather: TMAX/TMIN/TAVG/PRCP, lag 0–52",
        "- Keep country if count **or** any weather passes 0.6",
        "- Prune: drop lowest-importance **weather** feature each step; pick lowest RMSE",
        "- Retune small grid on best feature set (val 2023 → prod 2024)",
        "- Train prod: 2022–2023 → Test: 2024",
        "",
        f"## SARIMA baseline",
        f"- RMSE **{sar_rmse:.1f}**, MAE **{sar_mae:.1f}**",
        f"- Countries beating SARIMA: **{n_beat}/{len(summary_rows)}**",
        "",
        "## Ranking (best RMSE)",
        "",
        "| Rank | Country | RMSE | MAE | Δ vs SARIMA | Features |",
        "|-----:|---------|-----:|----:|------------:|----------|",
    ]
    for i, r in enumerate(summary_rows, 1):
        lines.append(
            f"| {i} | {r['country']} | {r['rmse']} | {r['mae']} | {r['delta_rmse_vs_sarima']:+.1f} | `{r['best_features']}` |"
        )
    lines += [
        "",
        "## Folders",
        "- `screening/` — correlation gates",
        "- `countries/<slug>/` — pruning steps, predictions, plots",
        "- `summary/` — master comparison + charts",
    ]
    readme.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {readme}")


if __name__ == "__main__":
    main()
