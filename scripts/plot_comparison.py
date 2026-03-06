"""
Plot comparison curves: Off-Policy vs On-Policy vs Hybrid.

Reads eval_scores.json files produced by the training scripts and
generates a matplotlib figure suitable for a Google Doc.

Usage:
    python plot_comparison.py                             # uses ./results/
    python plot_comparison.py --off off.json --on on.json # explicit paths
    python plot_comparison.py --xaxis tokens              # total tokens on x
    python plot_comparison.py --xaxis steps               # optimizer steps on x
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams.update({"font.size": 13, "font.family": "serif"})

COLORS = {
    "off_policy": "#2196F3",  # blue
    "on_policy": "#F44336",   # red
    "hybrid": "#9C27B0",      # purple
}
LABELS = {
    "off_policy": "Off-Policy (Paper Baseline)",
    "on_policy": "On-Policy (veRL + Tokasaurus)",
    "hybrid": "Hybrid (Warm-start → On-Policy)",
}


def load_eval_log(path: str) -> dict:
    """Load a single eval_scores.json."""
    with open(path) as f:
        data = json.load(f)
    # Support both {"method": ..., "evals": [...]} and bare [...]
    if isinstance(data, list):
        return {"method": "unknown", "evals": data}
    return data


def extract_curve(data: dict, score_key: str | None = None):
    """Extract (x_steps, x_tokens, y_scores) from eval log."""
    evals = data["evals"]
    steps, tokens, scores = [], [], []
    for entry in evals:
        s = entry["scores"]
        # Auto-detect score key
        if score_key is None:
            score_key = next(
                (k for k in s if "score" in k.lower()), list(s.keys())[0]
            )
        if score_key in s:
            steps.append(entry["optimizer_step"])
            tokens.append(entry.get("total_tokens", entry["optimizer_step"] * 32 * 2048))
            scores.append(s[score_key] * 100)  # convert to %
    return steps, tokens, scores, score_key


def plot(
    curves: dict[str, dict],
    xaxis: str = "steps",
    title: str = "LongHealth Accuracy: Off-Policy vs On-Policy",
    out: str = "comparison.png",
):
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for method, data in curves.items():
        steps, tokens, scores, key = extract_curve(data)
        x = steps if xaxis == "steps" else tokens
        color = COLORS.get(method, "#666")
        label = LABELS.get(method, method)
        ax.plot(x, scores, "o-", color=color, label=label, linewidth=2, markersize=5)

    ax.set_xlabel(
        "Optimizer Steps" if xaxis == "steps" else "Total Tokens Processed"
    )
    ax.set_ylabel("LongHealth Accuracy (%)")
    ax.set_title(title)
    ax.legend(loc="lower right", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    if xaxis == "tokens":
        ax.xaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M")
        )

    plt.tight_layout()
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved plot to {out}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot cartridge training comparison")
    parser.add_argument("--off", type=str, help="Path to off-policy eval_scores.json")
    parser.add_argument("--on", type=str, help="Path to on-policy eval_scores.json")
    parser.add_argument("--hybrid", type=str, help="Path to hybrid eval_scores.json")
    parser.add_argument(
        "--xaxis",
        choices=["steps", "tokens"],
        default="steps",
        help="X-axis: optimizer steps or total tokens",
    )
    parser.add_argument("--out", default="comparison.png", help="Output file path")
    parser.add_argument("--title", default=None, help="Plot title")
    args = parser.parse_args()

    curves = {}
    if args.off:
        curves["off_policy"] = load_eval_log(args.off)
    if args.on:
        curves["on_policy"] = load_eval_log(args.on)
    if args.hybrid:
        curves["hybrid"] = load_eval_log(args.hybrid)

    # Auto-discover from ./results/ if nothing specified
    if not curves:
        results = Path("results")
        for p in results.glob("**/eval_scores.json"):
            data = load_eval_log(str(p))
            method = data.get("method", p.parent.name)
            curves[method] = data
            print(f"Found: {method} → {p}")

    if not curves:
        print("No eval data found. Pass --off/--on/--hybrid or place files in ./results/")
        return

    title = args.title or "LongHealth Accuracy: Off-Policy vs On-Policy"
    plot(curves, xaxis=args.xaxis, title=title, out=args.out)


if __name__ == "__main__":
    main()
