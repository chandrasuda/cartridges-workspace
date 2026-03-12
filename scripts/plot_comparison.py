"""
Plot comparison curves: Off-Policy vs On-Policy vs Hybrid.

Reads results JSON files from training scripts and generates a matplotlib figure.

Usage:
    python plot_comparison.py --onpolicy results/onpolicy.json --offpolicy results/offpolicy.json
    python plot_comparison.py --xaxis tokens  # total tokens on x-axis
    python plot_comparison.py --xaxis steps   # optimizer steps on x-axis
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
    "off_policy": "Off-Policy (Teacher generates)",
    "on_policy": "On-Policy (Student generates)",
    "hybrid": "Hybrid (Warm-start → On-Policy)",
}


def load_results(path: str) -> dict:
    """Load results JSON file, supporting both local and Modal formats."""
    with open(path) as f:
        data = json.load(f)
    return data


def extract_curve_local(data: dict):
    """Extract (steps, tokens, scores) from local training results format."""
    # Local format: {"eval_results": [...], "token_history": [...], "config": {...}}
    if "token_history" in data:
        steps = [e["step"] for e in data["token_history"]]
        tokens = [e["tokens"] for e in data["token_history"]]
        scores = [e["accuracy"] for e in data["token_history"]]
        return steps, tokens, scores
    
    # Also support eval_results directly
    if "eval_results" in data:
        steps = [e["step"] for e in data["eval_results"]]
        tokens = [e.get("tokens", e["step"] * 100000) for e in data["eval_results"]]
        scores = [e["accuracy"] for e in data["eval_results"]]
        return steps, tokens, scores
    
    # Modal format: {"evals": [...]} or bare list
    evals = data.get("evals", data if isinstance(data, list) else [])
    steps, tokens, scores = [], [], []
    for entry in evals:
        if "optimizer_step" in entry:
            steps.append(entry["optimizer_step"])
            tokens.append(entry.get("total_tokens", entry["optimizer_step"] * 32 * 2048))
            # Find score key
            s = entry.get("scores", entry)
            score_key = next((k for k in s if "score" in k.lower() or "accuracy" in k.lower()), None)
            if score_key:
                score = s[score_key]
                scores.append(score * 100 if score < 1 else score)
    
    return steps, tokens, scores


def plot(
    curves: dict[str, tuple],
    xaxis: str = "tokens",
    title: str = "LongHealth Accuracy: Off-Policy vs On-Policy",
    out: str = "comparison.png",
):
    fig, ax = plt.subplots(figsize=(10, 6))

    for method, (steps, tokens, scores) in curves.items():
        x = steps if xaxis == "steps" else tokens
        color = COLORS.get(method, "#666")
        label = LABELS.get(method, method)
        ax.plot(x, scores, "o-", color=color, label=label, linewidth=2.5, markersize=7)

    ax.set_xlabel(
        "Training Steps" if xaxis == "steps" else "Total Tokens Processed",
        fontsize=14
    )
    ax.set_ylabel("LongHealth Accuracy (%)", fontsize=14)
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.legend(loc="lower right", framealpha=0.9, fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    if xaxis == "tokens":
        ax.xaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1e3:.0f}K")
        )

    plt.tight_layout()
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved plot to {out}")
    
    # Print summary table
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for method, (steps, tokens, scores) in curves.items():
        print(f"\n{LABELS.get(method, method)}:")
        print(f"  Steps: {min(steps)} → {max(steps)}")
        print(f"  Tokens: {min(tokens):,} → {max(tokens):,}")
        print(f"  Accuracy: {scores[0]:.1f}% → {scores[-1]:.1f}% (Δ{scores[-1]-scores[0]:+.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Plot cartridge training comparison")
    parser.add_argument("--onpolicy", type=str, help="Path to on-policy results JSON")
    parser.add_argument("--offpolicy", type=str, help="Path to off-policy results JSON")
    parser.add_argument("--hybrid", type=str, help="Path to hybrid results JSON")
    parser.add_argument(
        "--xaxis",
        choices=["steps", "tokens"],
        default="tokens",
        help="X-axis: training steps or total tokens (default: tokens)",
    )
    parser.add_argument("--output", "-o", default="comparison.png", help="Output file path")
    parser.add_argument("--title", default=None, help="Plot title")
    args = parser.parse_args()

    curves = {}
    
    if args.onpolicy:
        data = load_results(args.onpolicy)
        curves["on_policy"] = extract_curve_local(data)
        print(f"Loaded on-policy: {args.onpolicy}")
    
    if args.offpolicy:
        data = load_results(args.offpolicy)
        curves["off_policy"] = extract_curve_local(data)
        print(f"Loaded off-policy: {args.offpolicy}")
    
    if args.hybrid:
        data = load_results(args.hybrid)
        curves["hybrid"] = extract_curve_local(data)
        print(f"Loaded hybrid: {args.hybrid}")

    # Auto-discover from local_checkpoints if nothing specified
    if not curves:
        checkpoints = Path("local_checkpoints")
        for results_file in checkpoints.glob("**/onpolicy_results.json"):
            data = load_results(str(results_file))
            curves["on_policy"] = extract_curve_local(data)
            print(f"Found on-policy: {results_file}")
        for results_file in checkpoints.glob("**/offpolicy_results.json"):
            data = load_results(str(results_file))
            curves["off_policy"] = extract_curve_local(data)
            print(f"Found off-policy: {results_file}")

    if not curves:
        print("No results found. Run training scripts first or pass --onpolicy/--offpolicy paths.")
        return

    title = args.title or "On-Policy vs Off-Policy Cartridge Training"
    plot(curves, xaxis=args.xaxis, title=title, out=args.output)


if __name__ == "__main__":
    main()
