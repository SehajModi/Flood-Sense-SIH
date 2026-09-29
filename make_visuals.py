"""
Turn results.json + cascade_results.json into slide-ready charts.
Outputs: viz_dashboard.png (all 4 panels) + viz_1..viz_4 (one chart per slide)
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COL = {"all": "#95A5A6", "t1": "#E67E22", "t2": "#2E86C1", "t3": "#27AE60", "line": "#7F8C8D"}
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})


def load(p):
    return json.load(open(p)) if os.path.exists(p) else None


R, CS = load("results.json"), load("cascade_results.json")
SYNTH = any(os.path.exists(f) and open(f).read().startswith("**SYNTHETIC") for f in ("results.md", "cascade_results.md"))


def bar_labels(ax, bars, texts, dy=0.0, **kw):
    for b, t in zip(bars, texts):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + dy, t, ha="center", va="bottom", fontweight="bold", **kw)


# ---------------------------------------------------------------- panels
def p_ladder(ax):
    size = [f"~{R['t1_flash_kb_est']:.0f} KB flash (est.)", f"{R['t2_tflite_kb']:.1f} KB model", "no size limit"]
    names = [f"Tier 1 · Slave\nIsolation Forest\n{size[0]}", f"Tier 2 · Master\n1D-CNN (INT8)\n{size[1]}",
             f"Tier 3 · Cloud\nGradient boosting\n{size[2]}"]
    v = [R["t1_next_auc"], R["t2_int8_auc"], R["t3_multi"]["auc"]]
    b = ax.bar(names, v, color=[COL["t1"], COL["t2"], COL["t3"]], width=0.6)
    ax.set_ylim(0.5, 1.0); ax.axhline(0.5, color=COL["line"], ls="--", lw=1, label="chance level (0.5)")
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    bar_labels(ax, b, [f"{x:.3f}" for x in v], dy=0.006, fontsize=12)
    ax.set_ylabel("Next-day ROC-AUC (higher = better)")
    ax.set_title("More compute → better prediction", loc="left", fontweight="bold")


def p_funnel(ax):
    rows = CS[:4]; lab = ["All days", "Tier 1 flags", "+ Tier 2 confirms", "+ Tier 3 confirms"]
    y = list(range(len(rows)))[::-1]
    b = ax.barh(y, [r["flagged_pct"] for r in rows], color=[COL["all"], COL["t1"], COL["t2"], COL["t3"]], height=0.6)
    ax.set_yticks(y); ax.set_yticklabels(lab); ax.set_xlim(0, 118)
    for bi, r in zip(b, rows):
        ax.text(bi.get_width() + 1.5, bi.get_y() + bi.get_height() / 2,
                f"{r['flagged_pct']:.0f}% of days  |  {r['recall']*100:.0f}% of events caught", va="center", fontsize=10)
    ax.set_xlabel("% of days that raise an alert"); ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_title("Alerts shrink, real events stay caught", loc="left", fontweight="bold")


def p_false(ax):
    rows = CS[:4]; lab = ["All days", "Tier 1", "+ Tier 2", "+ Tier 3"]
    v = [r["fa_per_1000"] for r in rows]
    b = ax.bar(lab, v, color=[COL["all"], COL["t1"], COL["t2"], COL["t3"]], width=0.6)
    bar_labels(ax, b, [f"{x:.0f}" for x in v], dy=max(v) * 0.01, fontsize=12)
    if len(CS) > 4:
        ax.axhline(CS[4]["fa_per_1000"], color=COL["line"], ls="--", lw=1, label="cloud model run alone")
        ax.legend(loc="upper right", frameon=False, fontsize=9)
    ax.set_ylim(0, max(v) * 1.15); ax.set_ylabel("False alarms per 1000 station-days")
    ax.set_title(f"{v[0] / v[3]:.1f}× fewer false alarms", loc="left", fontweight="bold")


def p_lift(ax):
    names = ["Tier 1\nIsolation Forest", "Tier 2\n1D-CNN", "Tier 3\nCloud"]
    v = [R["t1_next_ap"] / R["t1_next_prev"], R["t2_int8_ap"] / R["t2_prev"], R["t3_multi"]["ap"] / R["t3_prev"]]
    b = ax.bar(names, v, color=[COL["t1"], COL["t2"], COL["t3"]], width=0.6)
    ax.axhline(1, color=COL["line"], ls="--", lw=1, label="random guessing = 1×"); ax.legend(loc="upper left", frameon=False, fontsize=9)
    bar_labels(ax, b, [f"{x:.0f}×" for x in v], dy=max(v) * 0.01, fontsize=12)
    ax.set_ylim(0, max(v) * 1.15); ax.set_ylabel("Precision lift over random")
    ax.set_title("How much better than random", loc="left", fontweight="bold")


PANELS = [("viz_1_accuracy_ladder.png", p_ladder, R), ("viz_2_cascade_funnel.png", p_funnel, CS),
          ("viz_3_false_alarms.png", p_false, CS), ("viz_4_lift.png", p_lift, R)]


def footer():
    t = "Data: weatherAUS (45 stations), next-day extreme rain = top 1% of rainfall days. Test = last 30% of dates."
    if R and "t3_single" in R and "t3_multi" in R:
        t += f" Multi-station fusion vs single-station: {R['t3_multi']['auc'] - R['t3_single']['auc']:+.3f} AUC (no gain)."
    return t


def stamp(fig):
    if SYNTH:
        fig.text(0.5, 0.5, "SYNTHETIC - DO NOT QUOTE", fontsize=44, color="red", alpha=0.25, ha="center", va="center", rotation=25)


def safe(ax, fn):
    try: fn(ax)
    except (KeyError, TypeError) as e:
        ax.axis("off"); ax.text(0.5, 0.5, f"data missing: {e}\n(run both scripts first)", ha="center", va="center")


def main():
    if not (R or CS): raise SystemExit("No results.json / cascade_results.json here. Run the two benchmark scripts first.")
    for name, fn, src in PANELS:
        fig, ax = plt.subplots(figsize=(8, 4.8))
        safe(ax, fn) if src else (ax.axis("off"), ax.text(0.5, 0.5, "results file missing", ha="center"))
        fig.text(0.01, 0.01, footer(), fontsize=7, color="#555"); stamp(fig)
        fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(name, dpi=200); plt.close(fig); print("saved", name)
    fig, axs = plt.subplots(2, 2, figsize=(16, 9))
    for ax, (_, fn, src) in zip(axs.ravel(), PANELS):
        safe(ax, fn) if src else ax.axis("off")
    fig.suptitle("Three-tier edge-to-cloud rainfall alerting: measured results", fontsize=16, fontweight="bold")
    fig.text(0.01, 0.005, footer(), fontsize=8, color="#555"); stamp(fig)
    fig.tight_layout(rect=(0, 0.02, 1, 0.96)); fig.savefig("viz_dashboard.png", dpi=200); print("saved viz_dashboard.png")


if __name__ == "__main__":
    main()
