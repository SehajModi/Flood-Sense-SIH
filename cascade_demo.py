"""
Cascade demo: how many extreme-rain days survive each tier, and how many false alarms get filtered out.
Task: next-day extreme rain (early warning). Needs sih_tinyml_benchmark.py in the same folder.

Run:  python cascade_demo.py --csv weatherAUS.csv
      (no --csv = SYNTHETIC dry run, do not quote)

Method (no test-set peeking):
  train  = first 55% of dates      -> fit all three tiers
  val    = next 15%                -> pick each tier's threshold so it keeps r1/r2/r3 of the true events
  test   = last 30%                -> the funnel table below is measured here
"""
import argparse, os, json
import numpy as np, pandas as pd
from sklearn.ensemble import IsolationForest, HistGradientBoostingClassifier
from sih_tinyml_benchmark import load_weather, synth, make_features, FEATS, N_TREES, MAX_SAMPLES, SEED


def pick_thr(s, y, target):
    """Highest threshold that still keeps >= target of the true events (flag if score >= thr)."""
    pos = np.sort(s[y == 1])[::-1]
    if len(pos) == 0: return -np.inf
    return pos[int(np.ceil(target * len(pos))) - 1]


def tier1(df, tr):
    normal = df[tr & (df.y_now == 0) & (df.y_next == 0)]
    clf = IsolationForest(n_estimators=N_TREES, max_samples=MAX_SAMPLES, random_state=SEED).fit(normal[FEATS].values)
    return pd.Series(-clf.score_samples(df[FEATS].values), index=df.index)


def tier2(df, tr, W=6):
    import tensorflow as tf
    tf.random.set_seed(SEED)
    d = df.copy()
    for c in ("rain_mm", "rain_3d"): d[c] = np.log1p(d[c].clip(lower=0))
    mu, sd = d.loc[tr, FEATS].mean(), d.loc[tr, FEATS].std() + 1e-6
    d[FEATS] = (d[FEATS] - mu) / sd
    Xs, idx = [], []
    for _, g in d.groupby("Location"):
        a = g[FEATS].values.astype(np.float32)
        if len(a) < W: continue
        Xs.append(np.lib.stride_tricks.sliding_window_view(a, (W, len(FEATS)))[:, 0]); idx.append(g.index.values[W - 1:])
    X, idx = np.concatenate(Xs), np.concatenate(idx)
    y, trm = df.loc[idx, "y_next"].values, tr.loc[idx].values
    m = tf.keras.Sequential([tf.keras.layers.Input((W, len(FEATS))),
                             tf.keras.layers.Conv1D(8, 3, activation="relu"),
                             tf.keras.layers.GlobalAveragePooling1D(),
                             tf.keras.layers.Dense(8, activation="relu"),
                             tf.keras.layers.Dense(1, activation="sigmoid")])
    m.compile("adam", "binary_crossentropy")
    w1 = float(np.sqrt((y[trm] == 0).sum() / max((y[trm] == 1).sum(), 1)))
    m.fit(X[trm], y[trm], epochs=8, batch_size=256, class_weight={0: 1.0, 1: w1}, verbose=0)
    return pd.Series(m.predict(X, batch_size=4096, verbose=0).ravel(), index=idx)


def tier3(df, tr):
    d = df.copy(); d["raining"] = (d["rain_mm"] > 1).astype(float)
    d["_n"] = d.groupby("Date")["rain_mm"].transform("count"); d = d[d["_n"] > 1].copy()
    ctx = []
    for c in ["rain_mm", "rain_3d", "dP_9to3", "humidity", "temp", "raining"]:
        d["net_" + c] = (d.groupby("Date")[c].transform("sum") - d[c]) / (d["_n"] - 1); ctx.append("net_" + c)
    d["net_n"] = d["_n"] - 1; ctx.append("net_n")
    t = tr.loc[d.index].values; y = d.y_next.values
    w = np.where(y[t] == 1, np.sqrt((y[t] == 0).sum() / max((y[t] == 1).sum(), 1)), 1.0)
    m = HistGradientBoostingClassifier(max_depth=4, max_iter=150, learning_rate=0.1, random_state=SEED)
    m.fit(d.loc[t, FEATS + ctx].values, y[t], sample_weight=w)
    return pd.Series(m.predict_proba(d[FEATS + ctx].values)[:, 1], index=d.index)


def stage(name, mask, y, n_total):
    P, fl = int((y == 1).sum()), int(mask.sum()); tp = int((mask & (y == 1)).sum()); fp = fl - tp
    rec, prec = tp / max(P, 1), tp / max(fl, 1)
    return dict(stage=name, flagged=fl, flagged_pct=100 * fl / n_total, caught=tp, P=P, recall=rec,
                precision=prec, f1=2 * prec * rec / max(prec + rec, 1e-9), fa_per_1000=1000 * fp / n_total)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv"); ap.add_argument("--r1", type=float, default=0.95)
    ap.add_argument("--r2", type=float, default=0.90); ap.add_argument("--r3", type=float, default=0.90)
    a = ap.parse_args()
    synthetic = not (a.csv and os.path.exists(a.csv))
    if synthetic: print("!! SYNTHETIC dry run - numbers are NOT quotable.")
    df, thr_mm = make_features(synth() if synthetic else load_weather(a.csv))
    c_tr, c_val = df.Date.quantile(0.55), df.Date.quantile(0.70)
    tr, va, te = df.Date <= c_tr, (df.Date > c_tr) & (df.Date <= c_val), df.Date > c_val

    S = pd.DataFrame({"s1": tier1(df, tr)})
    try: S["s2"] = tier2(df, tr)
    except Exception as e: raise SystemExit(f"Tier 2 needs tensorflow: {type(e).__name__}: {e}")
    S["s3"] = tier3(df, tr)
    S["y"] = df["y_next"]; S = S.dropna()
    V, T = S[va.loc[S.index]], S[te.loc[S.index]]
    yV, yT, cols = V.y.values.astype(int), T.y.values.astype(int), ["s1", "s2", "s3"]

    thr, vpass = [], np.ones(len(V), bool)
    for c, r in zip(cols, (a.r1, a.r2, a.r3)):
        t = pick_thr(V[c].values[vpass], yV[vpass], r); thr.append(t); vpass &= V[c].values >= t
    v_rec = (vpass & (yV == 1)).sum() / max((yV == 1).sum(), 1)
    t3alone = pick_thr(V.s3.values, yV, v_rec)

    n, tpass = len(T), np.ones(len(T), bool)
    rows = [stage("All days (no filter)", tpass.copy(), yT, n)]
    for c, t, nm in zip(cols, thr, ("Tier 1: slave IF", "+ Tier 2: master CNN confirms", "+ Tier 3: cloud confirms")):
        tpass &= T[c].values >= t; rows.append(stage(nm, tpass.copy(), yT, n))
    rows.append(stage("Ref: Tier 3 alone, same recall", T.s3.values >= t3alone, yT, n))

    hdr = "| Stage | Days flagged | Events caught | Recall | Precision | F1 | False alarms /1000 days |\n|---|---|---|---|---|---|---|\n"
    body = "".join(f"| {r['stage']} | {r['flagged']:,} ({r['flagged_pct']:.1f}%) | {r['caught']}/{r['P']} | {r['recall']*100:.1f}% | "
                   f"{r['precision']*100:.1f}% | {r['f1']:.2f} | {r['fa_per_1000']:.1f} |\n" for r in rows)
    out = ("**SYNTHETIC - DO NOT QUOTE**\n\n" if synthetic else "") + hdr + body + (
        f"\nTest rows: {n:,}; extreme = Rainfall >= {thr_mm:.1f} mm next day. Thresholds set on validation to keep "
        f"{a.r1:.0%}/{a.r2:.0%}/{a.r3:.0%} of events at tiers 1/2/3 (float CNN, not INT8).\n")
    open("cascade_results.md", "w").write(out); json.dump(rows, open("cascade_results.json", "w"), indent=1, default=float)
    print("\n" + out)

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        R = rows[:4]; lab = ["All days", "Tier 1", "+Tier 2", "+Tier 3"]
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].bar(lab, [r["fa_per_1000"] for r in R], color="#c0392b"); ax[0].set_title("False alarms per 1000 days"); ax[0].set_yscale("log")
        ax[1].bar(lab, [r["recall"] * 100 for r in R], color="#2471a3"); ax[1].set_title("Extreme-rain days caught (%)"); ax[1].set_ylim(0, 105)
        plt.tight_layout(); plt.savefig("cascade_funnel.png", dpi=160); print("Saved cascade_funnel.png")
    except Exception as e: print(f"[plot skipped] {e}")


if __name__ == "__main__":
    main()
