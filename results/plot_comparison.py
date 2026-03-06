#!/usr/bin/env python3
"""
Plot on-policy vs off-policy accuracy curves on the same graph.

X-axis: total tokens seen
Y-axis: LongHealth accuracy (%)

Reads:
  - results/off_policy.json        (from off-policy run)
  - results/onpolicy_eval.json     (download from Modal volume after training)
  - results/onpolicy_train_logs.txt (copy of training logs for token counts)

Usage:
    # After training finishes, download eval results:
    modal volume get onpolicy-results onpolicy_eval.json results/onpolicy_eval.json
    
    # Save training logs (copy from Modal dashboard or modal app logs):
    # modal app logs <app-id> > results/onpolicy_train_logs.txt
    
    python results/plot_comparison.py
"""

import json
import re
import os
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = os.path.dirname(os.path.abspath(__file__))


def load_offpolicy():
    """Load off-policy eval results."""
    with open(os.path.join(RESULTS_DIR, "off_policy.json")) as f:
        data = json.load(f)
    
    tokens = []
    accs = []
    for e in data["evals"]:
        tokens.append(e["total_tokens"])
        accs.append(e["scores"]["score"] * 100)  # Convert to percentage
    return tokens, accs


def load_onpolicy(log_file=None):
    """Load on-policy eval results + compute token counts from logs."""
    eval_path = os.path.join(RESULTS_DIR, "onpolicy_eval.json")
    if not os.path.exists(eval_path):
        print(f"⚠ {eval_path} not found — run training + eval first")
        return None, None, None

    with open(eval_path) as f:
        eval_data = json.load(f)

    # Try to get token counts from training logs
    tokens_per_step = {}
    log_path = log_file or os.path.join(RESULTS_DIR, "onpolicy_train_logs.txt")
    if os.path.exists(log_path):
        with open(log_path) as f:
            log_text = f.read()
        # Extract step -> total_num_tokens from log lines
        for match in re.finditer(r"step:(\d+).*?total_num_tokens:(\d+)", log_text):
            step = int(match.group(1))
            tokens = int(match.group(2))
            tokens_per_step[step] = tokens

    # Compute cumulative tokens per eval step
    eval_steps = sorted(e["step"] for e in eval_data["evals"])
    cumulative_tokens = {}
    if tokens_per_step:
        running_total = 0
        max_step = max(tokens_per_step.keys()) if tokens_per_step else 0
        for s in range(1, max_step + 1):
            running_total += tokens_per_step.get(s, 7800)  # fallback avg
            if s in eval_steps:
                cumulative_tokens[s] = running_total
    else:
        # Fallback: estimate ~7800 tokens per step
        print("⚠ No training logs found, estimating 7800 tokens/step")
        for s in eval_steps:
            cumulative_tokens[s] = s * 7800

    tokens = []
    accs = []
    for e in sorted(eval_data["evals"], key=lambda x: x["step"]):
        tokens.append(cumulative_tokens.get(e["step"], e["step"] * 7800))
        accs.append(e["accuracy"])

    baseline = eval_data.get("baseline", {}).get("accuracy", None)
    return tokens, accs, baseline


def main():
    # Load data
    off_tokens, off_accs = load_offpolicy()
    on_result = load_onpolicy()

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Off-policy curve
    ax.plot(np.array(off_tokens) / 1e6, off_accs, 'o-', color='#2196F3',
            linewidth=2, markersize=6, label='Off-policy (paper)', zorder=3)

    # On-policy curve
    if on_result and on_result[0] is not None:
        on_tokens, on_accs, baseline = on_result
        ax.plot(np.array(on_tokens) / 1e6, on_accs, 's-', color='#FF5722',
                linewidth=2, markersize=6, label='On-policy (ours)', zorder=3)

        # Baseline
        if baseline is not None:
            ax.axhline(y=baseline, color='gray', linestyle='--', linewidth=1,
                       label=f'Baseline (no cartridge): {baseline:.1f}%', zorder=1)
    else:
        print("On-policy results not available yet")

    # Off-policy baseline (step 0)
    if off_accs:
        ax.axhline(y=off_accs[0], color='lightblue', linestyle=':', linewidth=1,
                   label=f'Off-policy baseline: {off_accs[0]:.1f}%', zorder=1)

    ax.set_xlabel('Total Tokens (millions)', fontsize=13)
    ax.set_ylabel('LongHealth Accuracy (%)', fontsize=13)
    ax.set_title('On-Policy vs Off-Policy Cartridge Distillation', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 70)

    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "onpolicy_vs_offpolicy.png")
    plt.savefig(out_path, dpi=150)
    print(f"✓ Saved {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
