#!/usr/bin/env python3
"""
Quick status check for overnight Modal training jobs.

Usage:
    python scripts/check_overnight_status.py
"""

import subprocess
import json
import sys

def main():
    print("🔍 Checking Modal job status...\n")
    
    # List running apps
    result = subprocess.run(["modal", "app", "list", "--json"], capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Failed to get app list: {result.stderr}")
        return 1
    
    try:
        apps = json.loads(result.stdout)
    except json.JSONDecodeError:
        # Fallback to text parsing
        print("📋 Running Modal Apps:")
        print("-" * 50)
        result = subprocess.run(["modal", "app", "list"], capture_output=True, text=True)
        print(result.stdout)
        return 0
    
    # Filter to our comparison jobs
    comparison_apps = [a for a in apps if 'compare' in a.get('description', '').lower() 
                       or 'policy' in a.get('description', '').lower()]
    
    if not comparison_apps:
        print("✅ No comparison jobs currently running!")
        print("\nTo check results:")
        print("  python scripts/plot_comparison.py")
        return 0
    
    print("📋 Comparison Jobs:")
    print("-" * 50)
    for app in comparison_apps:
        state = app.get('state', 'unknown')
        emoji = "🏃" if 'running' in state.lower() or 'detached' in state.lower() else "⏹️"
        print(f"{emoji} {app.get('app_id', 'N/A')}")
        print(f"   Name: {app.get('description', 'N/A')}")
        print(f"   State: {state}")
        print(f"   Tasks: {app.get('tasks', 'N/A')}")
        print(f"   Created: {app.get('created_at', 'N/A')}")
        print()
    
    # Try to peek at results volume
    print("\n📦 Checking results volume...")
    result = subprocess.run(
        ["modal", "volume", "ls", "comparison-results:/"],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        print(result.stdout)
    else:
        print("  (Volume not accessible yet or empty)")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
