"""
Second benchmark: same 3-tier pipeline, run on REAL Indian data (10 cities, 2000-2024).
No pressure/humidity in this dataset -> features are rain, temp, wind gusts instead.

Run:  python india_benchmark.py --csv india_2000_2024_daily_weather.csv
      (no --csv = SYNTHETIC dry run, do not quote)
"""
import argparse, os, json, time
import numpy as np, pandas as pd
from sklearn.ensemble import IsolationForest, HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

SEED = 42
FEATS = ["rain_mm", "rain_3d", "temp_max", "temp_min", "gust_max"]
N_TREES, MAX_SAMPLES, W = 50, 256, 6


def load(csv):
    d = pd.read_csv(csv, parse_dates=["date"])
    d["rain_mm"] = d["rain_sum"].fillna(d["precipitation_sum"])
    d = d.dropna(subset=["rain_mm", "temperature_2m_max", "temperature_2m_min", "wind_gusts_10m_max"])
    return d.sort_values(["city", "date"]).reset_index(drop=True)


def synth(n_city=10, n=3500):
    rng = np.random.default_rng(SEED); out = []
    for c in range(n_city):
        d = pd.date_range("2000-01-01", periods=n); doy = d.dayofyear.values
        storm = rng.random(n) < 0.012
        rain = (rng.random(n) < 0.2) * rng.gamma(0.6, 4, n) + storm * rng.uniform(30, 100, n)
        tmax = 30 + 8 * np.sin(doy / 365 * 2 * np.pi) + rng.normal(0, 3, n)
        gust = 20 + rng.gamma(1.5, 8, n) + storm * rng.uniform(15, 40, n)
        out.append(pd.DataFrame(dict(city=f"C{c}", date=d, rain_mm=rain, temp_max=tmax,
                                     temp_min=tmax - rng.uniform(5, 12, n), gust_max=gust)))
    return pd.concat(out, ignore_index=True)


def feats(df):
    df = df.rename(columns={"temperature_2m_max": "temp_max", "temperature_2m_min": "temp_min",
                             "wind_gusts_10m_max": "gust_max"})
    g = df.groupby("city")
    df["rain_3d"] = g["rain_mm"].transform(lambda s: s.rolling(3, min_periods=1).sum())
    thr = df["rain_mm"].quantile(0.99)
    df["y_now"] = (df["rain_mm"] >= thr).astype(int)
    df["y_next"] = df.groupby("city")["y_now"].shift(-1)
    df = df.dropna(subset=["y_next"]).copy(); df["y_next"] = df["y_next"].astype(int)
    return df, thr


def op(y, s, thr):
    p = s >= thr; tp = (p & (y == 1)).sum(); fp = (p & (y == 0)).sum(); fn = (~p & (y == 1)).sum()
    r, pr = tp / max(tp + fn, 1), tp / max(tp + fp, 1)
    return dict(recall=r, precision=pr, f1=2 * pr * r / max(pr + r, 1e-9))


def tier1(df, cutoff, R):
    tr, te = df[df.date <= cutoff], df[df.date > cutoff]
    normal = tr[(tr.y_now == 0) & (tr.y_next == 0)]
    clf = IsolationForest(n_estimators=N_TREES, max_samples=MAX_SAMPLES, random_state=SEED).fit(normal[FEATS].values)
    s_te = -clf.score_samples(te[FEATS].values)
    for tag, y in (("now", te.y_now.values), ("next", te.y_next.values)):
        R[f"t1_{tag}_auc"] = roc_auc_score(y, s_te); R[f"t1_{tag}_ap"] = average_precision_score(y, s_te)
        R[f"t1_{tag}_prev"] = float(y.mean())
        R[f"t1_{tag}_q10"] = op(y, s_te, np.quantile(-clf.score_samples(normal[FEATS].values), 0.90))


def tier2(df, cutoff, R):
    import tensorflow as tf
    tf.keras.utils.set_random_seed(SEED)
    d = df.copy(); d["rain_mm"] = np.log1p(d["rain_mm"]); d["rain_3d"] = np.log1p(d["rain_3d"])
    tr = d.date <= cutoff
    mu, sd = d.loc[tr, FEATS].mean(), d.loc[tr, FEATS].std() + 1e-6
    d[FEATS] = (d[FEATS] - mu) / sd
    Xs, ys, dts = [], [], []
    for _, g in d.groupby("city"):
        a = g[FEATS].values.astype(np.float32)
        if len(a) < W: continue
        Xs.append(np.lib.stride_tricks.sliding_window_view(a, (W, len(FEATS)))[:, 0])
        ys.append(g.y_next.values[W - 1:]); dts.append(g.date.values[W - 1:])
    X, y, dt = np.concatenate(Xs), np.concatenate(ys).astype(int), np.concatenate(dts)
    trm = dt <= np.datetime64(cutoff)
    m = tf.keras.Sequential([tf.keras.layers.Input((W, len(FEATS))), tf.keras.layers.Conv1D(8, 3, activation="relu"),
                             tf.keras.layers.GlobalAveragePooling1D(), tf.keras.layers.Dense(8, activation="relu"),
                             tf.keras.layers.Dense(1, activation="sigmoid")])
    m.compile("adam", "binary_crossentropy")
    w1 = float(np.sqrt((y[trm] == 0).sum() / max((y[trm] == 1).sum(), 1)))
    m.fit(X[trm], y[trm], epochs=8, batch_size=256, class_weight={0: 1.0, 1: w1}, verbose=0)
    sub = np.random.default_rng(SEED).choice((~trm).sum(), min(20000, (~trm).sum()), replace=False)
    Xe, ye = X[~trm][sub], y[~trm][sub]
    p = m.predict(Xe, batch_size=4096, verbose=0).ravel()
    R["t2_auc"], R["t2_ap"], R["t2_prev"] = roc_auc_score(ye, p), average_precision_score(ye, p), float(ye.mean())
    R["t2_q5"] = op(ye, p, np.quantile(p, 0.95))


def tier3(df, cutoff, R):
    d = df.copy(); d["_n"] = d.groupby("date")["rain_mm"].transform("count"); d = d[d["_n"] > 1].copy()
    ctx = []
    for c in FEATS:
        d["net_" + c] = (d.groupby("date")[c].transform("sum") - d[c]) / (d["_n"] - 1); ctx.append("net_" + c)
    tr, te = d.date <= cutoff, d.date > cutoff
    ytr, yte = d.loc[tr, "y_next"].values, d.loc[te, "y_next"].values
    w = np.where(ytr == 1, np.sqrt((ytr == 0).sum() / max((ytr == 1).sum(), 1)), 1.0)
    m = HistGradientBoostingClassifier(max_depth=4, max_iter=150, learning_rate=0.1, random_state=SEED)
    m.fit(d.loc[tr, FEATS + ctx].values, ytr, sample_weight=w)
    p = m.predict_proba(d.loc[te, FEATS + ctx].values)[:, 1]
    R["t3_auc"], R["t3_ap"], R["t3_prev"] = roc_auc_score(yte, p), average_precision_score(yte, p), float(yte.mean())
    R["t3_q5"] = op(yte, p, np.quantile(p, 0.95))
    R["t3_cities"] = int(d.city.nunique())


def report(R, thr, n, synthetic, has2, has3):
    f = lambda x: f"{x['recall']*100:.1f}% / {x['precision']*100:.1f}% / {x['f1']:.2f}"
    rows = [
        ("**Tier 1: Isolation Forest (slave)**", "", ""),
        ("Same-day extreme rain: ROC-AUC", f"{R['t1_now_auc']:.3f} (AP {R['t1_now_ap']:.3f} vs base {R['t1_now_prev']:.3f})", "measured"),
        ("Next-day extreme rain: ROC-AUC", f"{R['t1_next_auc']:.3f} (AP {R['t1_next_ap']:.3f} vs base {R['t1_next_prev']:.3f})", "measured"),
        ("  recall/precision/F1 @10% alert rate", f(R["t1_next_q10"]), "measured"),
    ]
    if has2:
        rows += [("**Tier 2: 1D-CNN, float (master)**", "", ""),
                 ("Next-day extreme rain: ROC-AUC", f"{R['t2_auc']:.3f} (AP {R['t2_ap']:.3f} vs base {R['t2_prev']:.3f})", "measured"),
                 ("  recall/precision/F1 @5% alert rate", f(R["t2_q5"]), "measured")]
    if has3:
        rows += [("**Tier 3: cloud, multi-city gradient boosting**", "", ""),
                 ("Cities in network", f"{R['t3_cities']}", "measured"),
                 ("Next-day extreme rain: ROC-AUC", f"{R['t3_auc']:.3f} (AP {R['t3_ap']:.3f} vs base {R['t3_prev']:.3f})", "measured"),
                 ("  recall/precision/F1 @5% alert rate", f(R["t3_q5"]), "measured")]
    md = ["| Metric | Value | Source |", "|---|---|---|"] + [f"| {a} | {b} | {c} |" for a, b, c in rows]
    head = "**SYNTHETIC - DO NOT QUOTE**\n\n" if synthetic else ""
    note = (f"\nDataset: {n:,} city-days, 2000-2024 daily weather, 10 Indian cities. NO pressure/humidity in this "
            f"source -> features are rain_mm, rain_3d, temp_max, temp_min, gust_max. Extreme = rain >= {thr:.1f} mm (99th pct).\n")
    out = head + "\n".join(md) + note
    open("india_results.md", "w").write(out); json.dump(R, open("india_results.json", "w"), indent=1, default=float)
    print("\n" + out)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--csv"); ap.add_argument("--skip-cnn", action="store_true")
    a = ap.parse_args()
    synthetic = not (a.csv and os.path.exists(a.csv))
    if synthetic: print("!! SYNTHETIC dry run - numbers are NOT quotable.")
    df, thr = feats(synth() if synthetic else load(a.csv))
    cutoff = df.date.quantile(0.7)
    R = {}; tier1(df, cutoff, R)
    has2 = has3 = False
    if not a.skip_cnn:
        try: tier2(df, cutoff, R); has2 = True
        except Exception as e: print(f"[Tier 2 skipped] {type(e).__name__}: {e}")
    try: tier3(df, cutoff, R); has3 = True
    except Exception as e: print(f"[Tier 3 skipped] {type(e).__name__}: {e}")
    report(R, thr, len(df), synthetic, has2, has3)


if __name__ == "__main__":
    main()