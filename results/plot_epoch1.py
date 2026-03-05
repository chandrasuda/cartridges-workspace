"""
Off-policy cartridge training — Epoch 1 only.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Load data ────────────────────────────────────────────────────────
with open("/Users/csuda/cartridges-workspace/results/off_policy.json") as f:
    data = json.load(f)

EPOCH1_MAX_STEP = 1347  # 43117 iters / 32 accum

evals = [e for e in data["evals"] if e["optimizer_step"] <= EPOCH1_MAX_STEP]
steps = np.array([e["optimizer_step"] for e in evals])
tokens_m = np.array([e["total_tokens"] / 1e6 for e in evals])
scores = np.array([e["scores"]["score"] for e in evals]) * 100

# ── Moving average ───────────────────────────────────────────────────
def moving_avg(y, window=3):
    smoothed = np.copy(y).astype(float)
    for i in range(len(y)):
        lo = max(0, i - window // 2)
        hi = min(len(y), i + window // 2 + 1)
        smoothed[i] = np.mean(y[lo:hi])
    return smoothed

scores_smooth = moving_avg(scores, window=3)

# ── Plot ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 12,
    "axes.linewidth": 0.8,
    "axes.edgecolor": "#333333",
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "figure.facecolor": "white",
})

fig, ax = plt.subplots(1, 1, figsize=(9, 5.5))

color_main = "#2166AC"
color_fill = "#92C5DE"

# Smoothed trend
ax.plot(steps, scores_smooth, color=color_main, linewidth=2.5, zorder=3,
        label="Off-Policy (p = 2048)")

# Shaded band
ax.fill_between(steps,
                np.minimum(scores, scores_smooth),
                np.maximum(scores, scores_smooth),
                color=color_fill, alpha=0.3, zorder=2)

# Data points
ax.scatter(steps, scores, color=color_main, s=55, zorder=5,
           edgecolors="white", linewidth=1.0)

# Baseline
baseline = scores[0]
ax.axhline(y=baseline, color="#B2182B", linestyle="--", linewidth=1.3, alpha=0.6,
           label=f"No Cartridge ({baseline:.0f}%)", zorder=1)

# Best point
best_idx = np.argmax(scores)
ax.annotate(
    f"{scores[best_idx]:.0f}%",
    xy=(steps[best_idx], scores[best_idx]),
    xytext=(steps[best_idx] + 80, scores[best_idx] + 3),
    fontsize=11, color=color_main, fontweight="bold",
    arrowprops=dict(arrowstyle="->", color=color_main, lw=1.0,
                    connectionstyle="arc3,rad=-0.2"),
    bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
              edgecolor=color_main, alpha=0.95, linewidth=0.8),
)

# Axes
ax.set_xlabel("Optimizer Steps", fontsize=13, labelpad=6)
ax.set_ylabel("LongHealth Accuracy (%)", fontsize=13, labelpad=6)
ax.set_title(
    "Off-Policy Cartridge Training — Epoch 1",
    fontsize=15, fontweight="bold", pad=35
)
fig.text(0.5, 0.935,
         "Llama 3.2 3B  ·  p = 2048  ·  lr = 0.02  ·  batch = 32  ·  1 epoch (88M tokens)",
         ha="center", fontsize=10, color="#666666")

ax.set_ylim(25, 65)
ax.set_xlim(-30, 1400)
ax.set_yticks(np.arange(25, 66, 5))

ax.legend(loc="lower right", frameon=True, framealpha=0.95, edgecolor="#cccccc",
          fontsize=11, borderpad=0.6)
ax.grid(True, axis="both", linestyle="-")

# Top x-axis: tokens in millions
ax2 = ax.twiny()
ax2.set_xlim(ax.get_xlim())
token_ticks_steps = [0, 250, 500, 750, 1000, 1250]
ax2.set_xticks(token_ticks_steps)
ax2.set_xticklabels([f"{int(t * 32 * 2048 / 1e6)}M" for t in token_ticks_steps],
                     fontsize=10)
ax2.set_xlabel("Training Tokens", fontsize=11, labelpad=8, color="#888888")
ax2.tick_params(axis="x", colors="#888888", length=4)
ax2.spines["top"].set_color("#cccccc")

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig("/Users/csuda/cartridges-workspace/results/offpolicy_epoch1.png",
            dpi=200, bbox_inches="tight", facecolor="white")
print("Saved: results/offpolicy_epoch1.png")
plt.close()
