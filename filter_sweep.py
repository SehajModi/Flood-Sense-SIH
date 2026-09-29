"""
Filter-count sweep for the master-node 1D-CNN (Conv1D layer size), so the choice is measured, not guessed.
Needs sih_tinyml_benchmark.py in the same folder.

Run:  python filter_sweep.py --csv weatherAUS.csv            (about 5-10 min, 5 seeds x 7 sizes)
      python filter_sweep.py --csv weatherAUS.csv --seeds 3  (faster)
      (no --csv = SYNTHETIC dry run, do not quote)

Method: train 55% of dates / validation 15% / test 30%. Each size is trained with several random seeds
(training is noisy, so one run can't rank sizes). Sizes are ranked on VALIDATION AUC; test is reported, not used to choose.
Rule: pick the SMALLEST size whose validation AUC is within one std-dev of the best.
"""
import argparse, os, json
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from sih_tinyml_benchmark import load_weather, synth, make_features, FEATS, SEED

W = 6
SIZES = [1, 2, 4, 8, 16, 32, 64]


def build_windows(df, c_tr, c_val):
    d = df.copy()
    for c in ("rain_mm", "rain_3d"): d[c] = np.log1p(d[c].clip(lower=0))
    tr = d.Date <= c_tr
    mu, sd = d.loc[tr, FEATS].mean(), d.loc[tr, FEATS].std() + 1e-6
    d[FEATS] = (d[FEATS] - mu) / sd
    Xs, ys, dts = [], [], []
    for _, g in d.groupby("Location"):
        a = g[FEATS].values.astype(np.float32)
        if len(a) < W: continue
        Xs.append(np.lib.stride_tricks.sliding_window_view(a, (W, len(FEATS)))[:, 0])
        ys.append(g.y_next.values[W - 1:]); dts.append(g.Date.values[W - 1:])
    X, y, dt = np.concatenate(Xs), np.concatenate(ys).astype(int), np.concatenate(dts)
    a, b = np.datetime64(c_tr), np.datetime64(c_val)
    return X, y, dt <= a, (dt > a) & (dt <= b), dt > b


def make_model(tf, k):
    return tf.keras.Sequential([
        tf.keras.layers.Input((W, len(FEATS))),
        tf.keras.layers.Conv1D(k, 3, activation="relu"),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(8, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")])


def tflite_kb(tf, m, Xrep):
    try:
        c = tf.lite.TFLiteConverter.from_keras_model(m)
        c.optimizations = [tf.lite.Optimize.DEFAULT]
        c.representative_dataset = lambda: ([Xrep[i:i + 1]] for i in range(0, min(len(Xrep), 300)))
        c.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        c.inference_input_type = c.inference_output_type = tf.int8
        return len(c.convert()) / 1024
    except Exception as e:
        print(f"  [tflite size skipped: {type(e).__name__}]"); return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv"); ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args()
    import tensorflow as tf
    synthetic = not (a.csv and os.path.exists(a.csv))
    if synthetic: print("!! SYNTHETIC dry run - numbers are NOT quotable.")
    df, _ = make_features(synth() if synthetic else load_weather(a.csv))
    c_tr, c_val = df.Date.quantile(0.55), df.Date.quantile(0.70)
    X, y, trm, vam, tem = build_windows(df, c_tr, c_val)
    Xtr, ytr, Xva, yva, Xte, yte = X[trm], y[trm], X[vam], y[vam], X[tem], y[tem]
    cw = {0: 1.0, 1: float(np.sqrt((ytr == 0).sum() / max((ytr == 1).sum(), 1)))}
    print(f"windows: train {len(Xtr):,} (events {ytr.sum()}), val {len(Xva):,} ({yva.sum()}), test {len(Xte):,} ({yte.sum()})")

    rows = []
    for k in SIZES:
        va, te, tap = [], [], []
        for s in range(a.seeds):
            tf.keras.backend.clear_session(); tf.keras.utils.set_random_seed(SEED + s)
            m = make_model(tf, k); m.compile("adam", "binary_crossentropy")
            m.fit(Xtr, ytr, epochs=8, batch_size=256, class_weight=cw, verbose=0)
            pv, pt = m.predict(Xva, batch_size=4096, verbose=0).ravel(), m.predict(Xte, batch_size=4096, verbose=0).ravel()
            va.append(roc_auc_score(yva, pv)); te.append(roc_auc_score(yte, pt)); tap.append(average_precision_score(yte, pt))
        r = dict(filters=k, params=int(m.count_params()), tflite_kb=tflite_kb(tf, m, Xtr),
                 val_auc=float(np.mean(va)), val_std=float(np.std(va)), test_auc=float(np.mean(te)),
                 test_std=float(np.std(te)), test_ap=float(np.mean(tap)))
        rows.append(r); print(f"  {k:>3} filters: val {r['val_auc']:.3f}±{r['val_std']:.3f}  test {r['test_auc']:.3f}±{r['test_std']:.3f}  ({r['params']} params)")

    best = max(rows, key=lambda r: r["val_auc"])
    chosen = min((r for r in rows if r["val_auc"] >= best["val_auc"] - best["val_std"]), key=lambda r: r["filters"])
    hdr = "| Filters | Params | INT8 size (KB) | Val AUC | Test AUC | Test AP |\n|---|---|---|---|---|---|\n"
    body = "".join(f"| {r['filters']}{' ←' if r['filters'] == chosen['filters'] else ''} | {r['params']} | {r['tflite_kb']:.1f} | "
                   f"{r['val_auc']:.3f} ± {r['val_std']:.3f} | {r['test_auc']:.3f} ± {r['test_std']:.3f} | {r['test_ap']:.3f} |\n" for r in rows)
    verdict = (f"\nBest on validation: {best['filters']} filters. Smallest within one std-dev of best (rule): **{chosen['filters']} filters**. "
               f"Filters=8 test AUC {next(r for r in rows if r['filters'] == 8)['test_auc']:.3f} vs best-val size "
               f"{best['test_auc']:.3f}. Mean ± std over {a.seeds} seeds.\n")
    out = ("**SYNTHETIC - DO NOT QUOTE**\n\n" if synthetic else "") + hdr + body + verdict
    open("sweep_results.md", "w").write(out); json.dump(rows, open("sweep_results.json", "w"), indent=1)
    print("\n" + out)

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        ks = [r["filters"] for r in rows]
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
        ax[0].errorbar(ks, [r["val_auc"] for r in rows], [r["val_std"] for r in rows], marker="o", label="validation", capsize=3)
        ax[0].errorbar(ks, [r["test_auc"] for r in rows], [r["test_std"] for r in rows], marker="s", label="test", capsize=3)
        ax[0].axvline(8, color="grey", ls="--", lw=1, label="our choice (8)"); ax[0].set_xscale("log", base=2); ax[0].set_xticks(ks); ax[0].set_xticklabels(ks)
        ax[0].set_xlabel("Conv1D filters"); ax[0].set_ylabel("Next-day ROC-AUC"); ax[0].legend(frameon=False); ax[0].set_title("Accuracy vs filters", loc="left", fontweight="bold")
        ax[1].bar([str(k) for k in ks], [r["tflite_kb"] for r in rows], color="#2E86C1"); ax[1].set_xlabel("Conv1D filters"); ax[1].set_ylabel("INT8 model size (KB)")
        ax[1].set_title("Cost vs filters", loc="left", fontweight="bold")
        for s in ("top", "right"): ax[0].spines[s].set_visible(False); ax[1].spines[s].set_visible(False)
        if synthetic: fig.text(0.5, 0.5, "SYNTHETIC - DO NOT QUOTE", fontsize=36, color="red", alpha=0.25, ha="center", rotation=20)
        fig.tight_layout(); fig.savefig("filter_sweep.png", dpi=200); print("saved filter_sweep.png")
    except Exception as e: print(f"[plot skipped] {e}")


if __name__ == "__main__":
    main()
