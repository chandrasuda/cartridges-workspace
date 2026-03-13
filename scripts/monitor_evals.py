#!/usr/bin/env python3
"""
Monitor Modal app logs and capture eval results to a file.

Usage:
    python scripts/monitor_evals.py <app_id> <output_file>
    
Example:
    python scripts/monitor_evals.py ap-ErOHAA8D2SgUC3eyULzeRN offpolicy_evals.json
"""

import subprocess
import sys
import json
import re
import time
from datetime import datetime

def monitor_logs(app_id: str, output_file: str):
    """Monitor logs and extract eval scores."""
    
    results = {
        "app_id": app_id,
        "evals": [],
        "last_updated": None
    }
    
    # Load existing results if file exists
    try:
        with open(output_file, 'r') as f:
            results = json.load(f)
    except:
        pass
    
    seen_steps = {e['step'] for e in results['evals']}
    
    print(f"Monitoring {app_id} -> {output_file}")
    print(f"Already have evals for steps: {sorted(seen_steps)}")
    print("Press Ctrl+C to stop\n")
    
    # Pattern to match eval scores
    # Example: {'generate_longhealth_p10/score': np.float64(0.28)}
    score_pattern = re.compile(r"step=(\d+).*score.*?(\d+\.\d+)")
    score_pattern2 = re.compile(r"score.*?np\.float64\((\d+\.\d+)\)")
    step_pattern = re.compile(r"step[=\-]?(\d+)", re.IGNORECASE)
    
    try:
        proc = subprocess.Popen(
            ["modal", "app", "logs", app_id],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        current_step = None
        for line in proc.stdout:
            # Track current step from various log formats
            step_match = re.search(r"optimizer_step=(\d+)", line)
            if step_match:
                current_step = int(step_match.group(1))
            
            step_match2 = re.search(r"\[step=(\d+)\]", line)
            if step_match2:
                current_step = int(step_match2.group(1))
            
            # Look for score output
            if "score" in line.lower() and "np.float" in line:
                score_match = score_pattern2.search(line)
                if score_match and current_step is not None:
                    score = float(score_match.group(1))
                    
                    if current_step not in seen_steps:
                        eval_result = {
                            "step": current_step,
                            "accuracy": round(score * 100, 1),
                            "timestamp": datetime.now().isoformat()
                        }
                        results['evals'].append(eval_result)
                        results['last_updated'] = datetime.now().isoformat()
                        seen_steps.add(current_step)
                        
                        # Save immediately
                        with open(output_file, 'w') as f:
                            json.dump(results, f, indent=2)
                        
                        print(f"✓ Step {current_step}: {score*100:.1f}% - saved to {output_file}")
            
            # Also print progress
            if "optimizer_step=" in line or "Generating" in line:
                # Clean up the line for display
                clean = re.sub(r'[⠀-⣿]', '', line).strip()
                if clean:
                    print(f"  {clean[:100]}")
                    
    except KeyboardInterrupt:
        print(f"\nStopped. Results saved to {output_file}")
        proc.terminate()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python scripts/monitor_evals.py <app_id> <output_file>")
        sys.exit(1)
    
    monitor_logs(sys.argv[1], sys.argv[2])
