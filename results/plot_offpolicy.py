"""
Professional plot of off-policy cartridge training on LongHealth.
Styled to match the Cartridges paper (Figure 5).
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Load data ────────────────────────────────────────────────────────
with open("/Users/csuda/cartridges-workspace/results/off_policy.json") as f:
    data = json.load(f)

steps = np.array([e["optimizer_step"] for e in data["evals"]])
scores = np.array([e["scores"]["score"] for e in data["evals"]]) * 100

# ── Moving average for smoothed trend ────────────────────────────────
def moving_avg(x, y, window=3):
    """Simple moving average, padded at edges."""
    smoothed = np.copy(y).astype(float)
    for i in range(len(y)):
        lo = max(0, i - window // 2)
        hi = min(len(y), i + window // 2 + 1)
        smoothed[i] = np.mean(y[lo:hi])
    return smoothed

scores_smooth = moving_avg(steps, scores, window=3)

# ── Plot ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 12,
    "axes.linewidth": 0.8,
    "axes.edgecolor": "#333333",
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "figure.facecolor": "white",
})

fig, ax = plt.subplots(1, 1, figsize=(9, 5.5))

color_main = "#2166AC"
color_fill = "#92C5DE"

# Smoothed trend line
ax.plot(steps, scores_smooth, color=color_main, linewidth=2.5, zorder=3,
        label="Off-Policy (p = 2048)")

# Shaded region: raw ↔ smoothed
ax.fill_between(steps,
                np.minimum(scores, scores_smooth),
                np.maximum(scores, scores_smooth),
                color=color_fill, alpha=0.3, zorder=2)

# Raw data points
ax.scatter(steps, scores, color=color_main, s=50, zorder=5,
           edgecolors="white", linewidth=1.0)

# Baseline: no cartridge
baseline = scores[0]
ax.axhline(y=baseline, color="#B2182B", linestyle="--", linewidth=1.3, alpha=0.6,
           label=f"No Cartridge ({baseline:.0f}%)", zorder=1)

# Best point
best_idx = np.argmax(scores)
ax.annotate(
    f"{scores[best_idx]:.0f}%",
    xy=(steps[best_idx], scores[best_idx]),
    xytext=(steps[best_idx] + 200, scores[best_idx] + 3.5),
    fontsize=11, color=color_main, fontweight="bold",
    arrowprops=dict(arrowstyle="->", color=color_main, lw=1.0,
                    connectionstyle="arc3,rad=-0.2"),
    bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
              edgecolor=color_main, alpha=0.95, linewidth=0.8),
)

# Epoch boundary
epoch1_end = 1347  # 43117 iters / 32 accum ≈ 1347 optimizer steps
ax.axvline(x=epoch1_end, color="#999999", linestyle=":", linewidth=1.0, alpha=0.5)
ax.text(epoch1_end + 30, 27, "Epoch 1 / 2", fontsize=9, color="#888888",
        rotation=90, va="bottom")

# Axes
ax.set_xlabel("Optimizer Steps", fontsize=13, labelpad=6)
ax.set_ylabel("LongHealth Accuracy (%)", fontsize=13, labelpad=6)
ax.set_title(
    "Off-Policy Cartridge Training — LongHealth",
    fontsize=15, fontweight="bold", pad=45
)

# Subtitle — below main title, above token axis
fig.text(0.5, 0.94, "Llama 3.2 3B  ·  p = 2048  ·  lr = 0.02  ·  batch = 32  ·  2 epochs",
         ha="center", fontsize=10, color="#666666")

ax.set_ylim(25, 65)
ax.set_xlim(-80, 2850)
ax.set_yticks(np.arange(25, 66, 5))

ax.legend(loc="lower right", frameon=True, framealpha=0.95, edgecolor="#cccccc",
          fontsize=11, borderpad=0.6)
ax.grid(True, which="both", axis="y", linestyle="-")
ax.grid(True, which="both", axis="x", linestyle="-")

# Top x-axis: tokens
ax2 = ax.twiny()
ax2.set_xlim(ax.get_xlim())
token_ticks = [0, 500, 1000, 1500, 2000, 2500]
ax2.set_xticks(token_ticks)
ax2.set_xticklabels([f"{int(t * 32 * 2048 / 1e6)}M" for t in token_ticks],
                     fontsize=10)
ax2.set_xlabel("Training Tokens", fontsize=11, labelpad=8, color="#888888")
ax2.tick_params(axis="x", colors="#888888", length=4)
ax2.spines["top"].set_color("#cccccc")

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("/Users/csuda/cartridges-workspace/results/offpolicy_longhealth.png",
            dpi=200, bbox_inches="tight", facecolor="white")
print("Saved: results/offpolicy_longhealth.png")
plt.close()
