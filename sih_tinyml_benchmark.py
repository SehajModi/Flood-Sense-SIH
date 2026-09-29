"""
SIH 2026 - TinyML benchmark
  Tier 1 (slave node)  : Isolation Forest, exported to a real C header (if_model.h)
  Tier 2 (master node) : tiny 1D-CNN, INT8 TFLite (optional, needs tensorflow)
  Tier 3 (cloud)       : multi-station fusion vs single-station, same model class (no TF needed)

Outputs: results.md (slide table), results.json, if_model.h, master_cnn_int8.tflite
"""
import argparse, json, time, os
import numpy as np, pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, average_precision_score

SEED = 42
FEATS = ["rain_mm", "rain_3d", "dP_9to3", "temp", "humidity"]
N_TREES, MAX_SAMPLES = 50, 256          # 256 samples -> max depth 8


# ------------------------------------------------------------------ data
def load_weather(csv):
    need = ["Location", "Date", "Rainfall", "Pressure9am", "Pressure3pm", "Temp3pm", "Humidity3pm"]
    df = pd.read_csv(csv, parse_dates=["Date"])[need].dropna()
    return df.sort_values(["Location", "Date"]).reset_index(drop=True)


def synth(n_loc=6, n=2500):
    rng = np.random.default_rng(SEED); out = []
    for l in range(n_loc):
        d = pd.date_range("2010-01-01", periods=n); doy = d.dayofyear.values
        storm = rng.random(n) < 0.012
        rain = (rng.random(n) < 0.22) * rng.gamma(0.6, 5, n) + storm * rng.uniform(35, 120, n)
        p9 = 1015 + 6 * np.sin(doy / 365 * 2 * np.pi) + rng.normal(0, 3, n)
        p3 = p9 + rng.normal(0, 1.5, n) - storm * rng.uniform(4, 9, n)
        temp = 20 + 8 * np.sin(doy / 365 * 2 * np.pi) + rng.normal(0, 3, n)
        hum = np.clip(50 + rng.normal(0, 15, n) + storm * 25 + (rain > 0) * 10, 5, 100)
        out.append(pd.DataFrame(dict(Location=f"S{l}", Date=d, Rainfall=rain, Pressure9am=p9,
                                     Pressure3pm=p3, Temp3pm=temp, Humidity3pm=hum)))
    return pd.concat(out, ignore_index=True)


def make_features(df):
    g = df.groupby("Location")
    df["rain_mm"] = df["Rainfall"]
    df["rain_3d"] = g["Rainfall"].transform(lambda s: s.rolling(3, min_periods=1).sum())
    df["dP_9to3"] = df["Pressure3pm"] - df["Pressure9am"]     # in-day pressure change (proxy for gradient)
    df["temp"], df["humidity"] = df["Temp3pm"], df["Humidity3pm"]
    thr = df["Rainfall"].quantile(0.99)                       # "extreme rain day" = top 1%
    df["y_now"] = (df["Rainfall"] >= thr).astype(int)
    df["y_next"] = df.groupby("Location")["y_now"].shift(-1)
    df = df.dropna(subset=["y_next"]).copy()
    df["y_next"] = df["y_next"].astype(int)
    return df, thr


# ------------------------------------------------------------------ IF -> flat C arrays
def c_factor(n):
    n = np.asarray(n, float); out = np.zeros_like(n)
    out[n == 2] = 1; m = n > 2
    out[m] = 2 * (np.log(n[m] - 1) + np.euler_gamma) - 2 * (n[m] - 1) / n[m]
    return out


def flatten_forest(clf):
    feat, thr, right, offs, depths = [], [], [], [0], []
    for est in clf.estimators_:
        t = est.tree_; n = t.node_count; depth = np.zeros(n, int)
        for i in range(n):
            if t.children_left[i] != -1:
                assert t.children_left[i] == i + 1, "unexpected node order"
                depth[i + 1] = depth[t.children_right[i]] = depth[i] + 1
        for i in range(n):
            if t.children_left[i] == -1:      # leaf: store path-length contribution
                feat.append(-1); right.append(0)
                thr.append(depth[i] + c_factor([t.n_node_samples[i]])[0])
            else:
                feat.append(int(t.feature[i])); thr.append(float(t.threshold[i]))
                right.append(int(t.children_right[i]))
        offs.append(len(feat)); depths.append(int(depth.max()))
    return dict(feat=np.array(feat, np.int8), thr=np.array(thr, np.float32),
                right=np.array(right, np.uint16), offs=offs, max_depth=max(depths),
                norm=float(len(clf.estimators_) * c_factor([clf.max_samples_])[0]))


def flat_score(F, X):
    out = np.empty(len(X))
    for k, x in enumerate(X.astype(np.float32)):
        s = 0.0
        for t in range(len(F["offs"]) - 1):
            b, i = F["offs"][t], 0
            while F["feat"][b + i] >= 0:
                i = i + 1 if x[F["feat"][b + i]] <= F["thr"][b + i] else int(F["right"][b + i])
            s += F["thr"][b + i]
        out[k] = 2 ** (-s / F["norm"])
    return out


def write_c_header(F, path):
    arr = lambda a, fmt: ", ".join(fmt(v) for v in a)
    src = f"""// Auto-generated Isolation Forest ({len(F['offs'])-1} trees). No heap, no libs except math.h.
#pragma once
#include <stdint.h>
#include <math.h>
#define IF_N_TREES {len(F['offs'])-1}
#define IF_N_NODES {len(F['feat'])}
static const uint16_t IF_OFFSET[IF_N_TREES+1] = {{{arr(F['offs'], str)}}};
static const int8_t   IF_FEAT[IF_N_NODES]  = {{{arr(F['feat'], str)}}};
static const float    IF_THR[IF_N_NODES]   = {{{arr(F['thr'], lambda v: f'{v:.6g}f')}}};
static const uint16_t IF_RIGHT[IF_N_NODES] = {{{arr(F['right'], str)}}};
#define IF_NORM {F['norm']:.6f}f
// features order: {FEATS}. Returns anomaly score in (0,1]; higher = more anomalous.
static inline float if_score(const float *x) {{
  float s = 0;
  for (int t = 0; t < IF_N_TREES; t++) {{
    uint16_t b = IF_OFFSET[t], i = 0;
    while (IF_FEAT[b + i] >= 0)
      i = (x[IF_FEAT[b + i]] <= IF_THR[b + i]) ? i + 1 : IF_RIGHT[b + i];
    s += IF_THR[b + i];
  }}
  return powf(2.0f, -s / IF_NORM);
}}
"""
    open(path, "w").write(src)


def op_point(y, s, thr):
    p = s >= thr
    tp = (p & (y == 1)).sum(); fp = (p & (y == 0)).sum()
    fn = (~p & (y == 1)).sum(); tn = (~p & (y == 0)).sum()
    rec, prec = tp / max(tp + fn, 1), tp / max(tp + fp, 1)
    return dict(recall=rec, precision=prec, f1=2 * prec * rec / max(prec + rec, 1e-9), fpr=fp / max(fp + tn, 1))


# ------------------------------------------------------------------ Tier 1
def tier1(df, cutoff, R):
    tr, te = df[df.Date <= cutoff], df[df.Date > cutoff]
    normal = tr[(tr.y_now == 0) & (tr.y_next == 0)]
    clf = IsolationForest(n_estimators=N_TREES, max_samples=MAX_SAMPLES, random_state=SEED).fit(normal[FEATS].values)
    s_tr, s_te = -clf.score_samples(normal[FEATS].values), -clf.score_samples(te[FEATS].values)
    R["t1_train_rows"], R["t1_test_rows"] = len(normal), len(te)

    F = flatten_forest(clf)
    idx = np.random.default_rng(SEED).choice(len(te), 300, replace=False)
    err = np.abs(flat_score(F, te[FEATS].values[idx]) - s_te[idx]).max()
    R["c_export_max_abs_err"] = float(err)
    write_c_header(F, "if_model.h")

    R["t1_nodes"] = len(F["feat"]); R["t1_depth"] = F["max_depth"]
    R["t1_flash_kb_est"] = len(F["feat"]) * 7 / 1024          # int8 + float32 + uint16 per node
    R["t1_cmp_per_inf"] = N_TREES * F["max_depth"]
    R["t1_mcu_us_est"] = R["t1_cmp_per_inf"] * 50 / 240       # 50 cycles/compare @240 MHz - ESTIMATE
    X = te[FEATS].values[:5000]; t0 = time.perf_counter()
    for _ in range(3): clf.score_samples(X)
    R["t1_host_us_per_sample"] = (time.perf_counter() - t0) / (3 * len(X)) * 1e6

    for task, y in (("now", te.y_now.values), ("next", te.y_next.values)):
        R[f"t1_{task}_auc"] = roc_auc_score(y, s_te)
        R[f"t1_{task}_ap"], R[f"t1_{task}_prev"] = average_precision_score(y, s_te), y.mean()
        for q in (0.01, 0.05, 0.10):
            R[f"t1_{task}_q{int(q*100)}"] = op_point(y, s_te, np.quantile(s_tr, 1 - q))
    return s_te


# ------------------------------------------------------------------ Tier 2
def tier2(df, cutoff, R, W=6):
    import tensorflow as tf
    tf.random.set_seed(SEED)
    d = df.copy()
    for c in ("rain_mm", "rain_3d"): d[c] = np.log1p(d[c].clip(lower=0))
    tr_mask = d.Date <= cutoff
    mu, sd = d.loc[tr_mask, FEATS].mean(), d.loc[tr_mask, FEATS].std() + 1e-6
    d[FEATS] = (d[FEATS] - mu) / sd
    Xs, ys, ds = [], [], []
    for _, g in d.groupby("Location"):
        a = g[FEATS].values.astype(np.float32)
        if len(a) < W: continue
        w = np.lib.stride_tricks.sliding_window_view(a, (W, len(FEATS)))[:, 0]
        Xs.append(w); ys.append(g.y_next.values[W - 1:]); ds.append(g.Date.values[W - 1:])
    X, y, dt = np.concatenate(Xs), np.concatenate(ys), np.concatenate(ds)
    trm = dt <= np.datetime64(cutoff)
    Xtr, ytr, Xte, yte = X[trm], y[trm], X[~trm], y[~trm]
    m = tf.keras.Sequential([
        tf.keras.layers.Input((W, len(FEATS))),
        tf.keras.layers.Conv1D(8, 3, activation="relu"),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(8, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")])
    m.compile("adam", "binary_crossentropy")
    w1 = float(np.sqrt((ytr == 0).sum() / max((ytr == 1).sum(), 1)))
    m.fit(Xtr, ytr, epochs=8, batch_size=256, class_weight={0: 1.0, 1: w1}, verbose=0)
    R["t2_params"] = int(m.count_params())
    sub = np.random.default_rng(SEED).choice(len(Xte), min(20000, len(Xte)), replace=False)
    Xe, ye = Xte[sub], yte[sub]
    pf = m.predict(Xe, verbose=0).ravel()
    R["t2_float_auc"] = roc_auc_score(ye, pf)

    conv = tf.lite.TFLiteConverter.from_keras_model(m)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = lambda: ([Xtr[i:i + 1]] for i in range(0, min(len(Xtr), 500)))
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = conv.inference_output_type = tf.int8
    tfl = conv.convert(); open("master_cnn_int8.tflite", "wb").write(tfl)
    R["t2_tflite_kb"] = len(tfl) / 1024
    it = tf.lite.Interpreter(model_content=tfl); it.allocate_tensors()
    i_d, o_d = it.get_input_details()[0], it.get_output_details()[0]
    R["t2_tensor_ram_kb_upper"] = sum(int(np.prod(t["shape"])) * np.dtype(t["dtype"]).itemsize
                                       for t in it.get_tensor_details()) / 1024
    si, zi = i_d["quantization"]; so, zo = o_d["quantization"]
    pq = np.empty(len(Xe)); t0 = time.perf_counter()
    for k in range(len(Xe)):
        it.set_tensor(i_d["index"], np.clip(np.round(Xe[k:k + 1] / si + zi), -128, 127).astype(np.int8))
        it.invoke()
        pq[k] = (it.get_tensor(o_d["index"]).astype(np.float32)[0, 0] - zo) * so
    R["t2_host_us_per_sample"] = (time.perf_counter() - t0) / len(Xe) * 1e6
    R["t2_int8_auc"], R["t2_int8_ap"], R["t2_prev"] = roc_auc_score(ye, pq), average_precision_score(ye, pq), ye.mean()
    thr = np.quantile(pq, 0.95)
    R["t2_q5"] = op_point(ye, pq, thr)


# ------------------------------------------------------------------ Tier 3 (cloud, multi-station)
def tier3(df, cutoff, R):
    """Cloud tier: does seeing OTHER stations on the same day improve next-day extreme-rain prediction?
    Same model class (gradient boosting) with and without network context, so the gain is from fusion."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    d = df.copy()
    d["raining"] = (d["rain_mm"] > 1).astype(float)
    d["_n"] = d.groupby("Date")["rain_mm"].transform("count")
    d = d[d["_n"] > 1].copy()
    ctx = []
    for c in ["rain_mm", "rain_3d", "dP_9to3", "humidity", "temp", "raining"]:
        tot = d.groupby("Date")[c].transform("sum")
        d["net_" + c] = (tot - d[c]) / (d["_n"] - 1)        # mean over the OTHER stations that day
        ctx.append("net_" + c)
    d["net_n"] = d["_n"] - 1; ctx.append("net_n")
    tr, te = d[d.Date <= cutoff], d[d.Date > cutoff]
    ytr, yte = tr.y_next.values, te.y_next.values
    w = np.where(ytr == 1, np.sqrt((ytr == 0).sum() / max((ytr == 1).sum(), 1)), 1.0)

    def run(cols):
        m = HistGradientBoostingClassifier(max_depth=4, max_iter=150, learning_rate=0.1, random_state=SEED)
        m.fit(tr[cols].values, ytr, sample_weight=w)
        pr = m.predict_proba(te[cols].values)[:, 1]
        return dict(auc=roc_auc_score(yte, pr), ap=average_precision_score(yte, pr),
                    q5=op_point(yte, pr, np.quantile(pr, 0.95)))
    R["t3_single"], R["t3_multi"] = run(FEATS), run(FEATS + ctx)
    R["t3_stations"], R["t3_test_rows"], R["t3_prev"] = int(d.Location.nunique()), len(te), float(yte.mean())


# ------------------------------------------------------------------ report
def report(R, thr_mm, ntot, synthetic, cnn_ok):
    f = lambda d: f"{d['recall']*100:.1f}% / {d['precision']*100:.1f}% / {d['f1']:.2f}"
    rows = [
        ("**Tier 1: Isolation Forest (slave node)**", "", ""),
        ("Config", f"{N_TREES} trees, max depth {R['t1_depth']}, 5 features", "measured"),
        ("Total split/leaf nodes", f"{R['t1_nodes']:,}", "measured"),
        ("Flash (model)", f"~{R['t1_flash_kb_est']:.0f} KB", "estimate: 7 B/node"),
        ("RAM at inference", "20 B features + <100 B locals, no heap", "by design"),
        ("Compares per inference (worst case)", f"{R['t1_cmp_per_inf']}", "measured"),
        ("Latency, laptop CPU", f"{R['t1_host_us_per_sample']:.0f} us/sample (batched)", "measured, NOT MCU"),
        ("Latency, ESP32-S3 @240MHz", f"~{R['t1_mcu_us_est']:.0f} us", "ESTIMATE, verify on HW"),
        ("C export matches sklearn", f"max abs score error {R['c_export_max_abs_err']:.1e}", "measured"),
        ("Same-day extreme rain: ROC-AUC", f"{R['t1_now_auc']:.3f}  (AP {R['t1_now_ap']:.3f} vs base {R['t1_now_prev']:.3f})", "measured"),
        ("  recall/precision/F1 @5% alert rate", f(R["t1_now_q5"]), "measured"),
        ("  recall/precision/F1 @10% alert rate", f(R["t1_now_q10"]), "measured"),
        ("Next-day extreme rain: ROC-AUC", f"{R['t1_next_auc']:.3f}  (AP {R['t1_next_ap']:.3f} vs base {R['t1_next_prev']:.3f})", "measured"),
        ("  recall/precision/F1 @10% alert rate", f(R["t1_next_q10"]), "measured"),
    ]
    if cnn_ok:
        rows += [
            ("**Tier 2: tiny 1D-CNN, INT8 (master node)**", "", ""),
            ("Parameters", f"{R['t2_params']}", "measured"),
            ("TFLite INT8 size", f"{R['t2_tflite_kb']:.1f} KB", "measured"),
            ("Tensor RAM (upper bound)", f"{R['t2_tensor_ram_kb_upper']:.1f} KB", "measured, no reuse"),
            ("Latency, laptop CPU", f"{R['t2_host_us_per_sample']:.0f} us/sample", "measured, NOT MCU"),
            ("Next-day extreme rain: ROC-AUC (INT8)", f"{R['t2_int8_auc']:.3f}  (AP {R['t2_int8_ap']:.3f} vs base {R['t2_prev']:.3f})", "measured"),
            ("  recall/precision/F1 @5% alert rate", f(R["t2_q5"]), "measured"),
        ]
    if "t3_multi" in R:
        a, b = R["t3_single"], R["t3_multi"]
        rows += [
            ("**Tier 3: cloud, multi-station fusion (AWS tier)**", "", ""),
            ("Stations in network", f"{R['t3_stations']}", "measured"),
            ("Next-day, single-station GBM: ROC-AUC", f"{a['auc']:.3f}  (AP {a['ap']:.3f} vs base {R['t3_prev']:.3f})", "measured"),
            ("  recall/precision/F1 @5% alert rate", f(a["q5"]), "measured"),
            ("Next-day, multi-station cloud GBM: ROC-AUC", f"{b['auc']:.3f}  (AP {b['ap']:.3f} vs base {R['t3_prev']:.3f})", "measured"),
            ("  recall/precision/F1 @5% alert rate", f(b["q5"]), "measured"),
            ("Gain from network fusion (ROC-AUC)", f"{b['auc'] - a['auc']:+.3f}", "measured"),
        ]
    md = ["| Metric | Value | Source |", "|---|---|---|"] + [f"| {a} | {b} | {c} |" for a, b, c in rows]
    head = ("**SYNTHETIC DATA - DO NOT QUOTE THESE NUMBERS**\n\n" if synthetic else "")
    note = (f"\nDataset rows used: {ntot:,}. 'Extreme rain' = Rainfall >= {thr_mm:.1f} mm (99th pct). "
            f"Train on the first 70% of dates (normal days only for IF), test on the last 30%. "
            f"Daily data: 9am->3pm pressure change is the pressure-gradient proxy.\n")
    out = head + "\n".join(md) + note
    open("results.md", "w").write(out); json.dump(R, open("results.json", "w"), indent=1, default=float)
    print("\n" + out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv"); ap.add_argument("--skip-cnn", action="store_true")
    a = ap.parse_args()
    synthetic = not (a.csv and os.path.exists(a.csv))
    if synthetic: print("!! No CSV given/found -> SYNTHETIC dry run. Numbers are NOT quotable.")
    df = synth() if synthetic else load_weather(a.csv)
    df, thr_mm = make_features(df)
    cutoff = df["Date"].quantile(0.7)
    R = {}
    tier1(df, cutoff, R)
    cnn_ok = False
    if not a.skip_cnn:
        try: tier2(df, cutoff, R); cnn_ok = True
        except Exception as e: print(f"[Tier 2 skipped] {type(e).__name__}: {e}")
    try: tier3(df, cutoff, R)
    except Exception as e: print(f"[Tier 3 skipped] {type(e).__name__}: {e}")
    report(R, thr_mm, len(df), synthetic, cnn_ok)


if __name__ == "__main__":
    main()