"""
Charts for the India benchmark (india_benchmark.py). Run after that script:
    python make_visuals_india.py
Outputs: india_viz_dashboard.png + india_viz_1/2/3.png
"""
import json, os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

COL = {"t1": "#E67E22", "t2": "#2E86C1", "t3": "#27AE60", "line": "#7F8C8D"}
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
R = json.load(open("india_results.json"))
SYNTH = open("india_results.md").read().startswith("**SYNTHETIC")
NAMES = ["Tier 1 · Slave\nIsolation Forest", "Tier 2 · Master\n1D-CNN", "Tier 3 · Cloud\nGradient boosting"]


def bar_labels(ax, bars, texts, dy=0.0, **kw):
    for b, t in zip(bars, texts):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + dy, t, ha="center", va="bottom", fontweight="bold", **kw)


def p_ladder(ax):
    v = [R["t1_next_auc"], R["t2_auc"], R["t3_auc"]]
    b = ax.bar(NAMES, v, color=[COL["t1"], COL["t2"], COL["t3"]], width=0.6)
    ax.set_ylim(0.5, 1.0); ax.axhline(0.5, color=COL["line"], ls="--", lw=1, label="chance level (0.5)")
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    bar_labels(ax, b, [f"{x:.3f}" for x in v], dy=0.008, fontsize=12)
    ax.set_ylabel("Next-day ROC-AUC"); ax.set_title("Real Indian data: 10 cities, 2000-2024", loc="left", fontweight="bold")


def p_precision(ax):
    q = [R["t1_next_q10"], R["t2_q5"], R["t3_q5"]]
    rate = ["@10% alert rate", "@5% alert rate", "@5% alert rate"]
    rec, prec = [x["recall"] * 100 for x in q], [x["precision"] * 100 for x in q]
    x = range(3); w = 0.35
    b1 = ax.bar([i - w / 2 for i in x], rec, w, label="Recall (events caught)", color="#2E86C1")
    b2 = ax.bar([i + w / 2 for i in x], prec, w, label="Precision (alerts that were real)", color="#E67E22")
    bar_labels(ax, b1, [f"{v:.0f}%" for v in rec], dy=1, fontsize=10)
    bar_labels(ax, b2, [f"{v:.0f}%" for v in prec], dy=1, fontsize=10)
    ax.set_xticks(list(x)); ax.set_xticklabels([f"Tier {i+1}\n{r}" for i, r in enumerate(rate)])
    ax.set_ylim(0, 100); ax.legend(frameon=False, fontsize=9); ax.set_ylabel("%")
    ax.set_title("Recall vs precision at each tier's alert rate", loc="left", fontweight="bold")


def p_lift(ax):
    v = [R["t1_next_ap"] / R["t1_next_prev"], R["t2_ap"] / R["t2_prev"], R["t3_ap"] / R["t3_prev"]]
    b = ax.bar(NAMES, v, color=[COL["t1"], COL["t2"], COL["t3"]], width=0.6)
    ax.axhline(1, color=COL["line"], ls="--", lw=1, label="random guessing = 1x"); ax.legend(loc="upper left", frameon=False, fontsize=9)
    bar_labels(ax, b, [f"{x:.0f}x" for x in v], dy=max(v) * 0.01, fontsize=12)
    ax.set_ylim(0, max(v) * 1.15); ax.set_ylabel("Precision lift over random")
    ax.set_title("How much better than random", loc="left", fontweight="bold")


PANELS = [("india_viz_1_accuracy.png", p_ladder), ("india_viz_2_recall_precision.png", p_precision), ("india_viz_3_lift.png", p_lift)]
FOOT = (f"Data: 10 Indian cities, 2000-2024 daily weather (rain, temp, wind gusts only - no pressure/humidity "
        f"in this source). Extreme = next-day rain in top 1% of days. Cloud tier uses {R.get('t3_cities','?')}-city context.")


def stamp(fig):
    if SYNTH: fig.text(0.5, 0.5, "SYNTHETIC - DO NOT QUOTE", fontsize=40, color="red", alpha=0.25, ha="center", va="center", rotation=25)


for name, fn in PANELS:
    fig, ax = plt.subplots(figsize=(8, 4.8)); fn(ax)
    fig.text(0.01, 0.01, FOOT, fontsize=7, color="#555", wrap=True); stamp(fig)
    fig.tight_layout(rect=(0, 0.06, 1, 1)); fig.savefig(name, dpi=200); plt.close(fig); print("saved", name)

fig, axs = plt.subplots(1, 3, figsize=(18, 5))
for ax, (_, fn) in zip(axs, PANELS): fn(ax)
fig.suptitle("Isolation Forest -> CNN -> Cloud, tested on real Indian weather", fontsize=15, fontweight="bold")
fig.text(0.01, 0.01, FOOT, fontsize=8, color="#555", wrap=True); stamp(fig)
fig.tight_layout(rect=(0, 0.06, 1, 0.94)); fig.savefig("india_viz_dashboard.png", dpi=200)
print("saved india_viz_dashboard.png")