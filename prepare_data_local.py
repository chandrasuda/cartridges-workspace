#!/usr/bin/env python3
"""
Prepare on-policy training data from local HF shards.

Reads the already-downloaded parquet files, extracts user-turn prompts,
and saves them in veRL's expected format.

Usage:
    python prepare_data_local.py
"""

import json
import random
from pathlib import Path
from glob import glob

import pandas as pd
import requests


# ---- patient name → patient_id mapping (for metadata) ----
PATIENT_NAMES = {}


def _load_patient_mapping():
    global PATIENT_NAMES
    if PATIENT_NAMES:
        return
    url = "https://raw.githubusercontent.com/kbressem/LongHealth/refs/heads/main/data/benchmark_v5.json"
    print(f"Downloading LongHealth patient mapping from {url} ...")
    data = requests.get(url, timeout=30).json()
    for pid, patient in data.items():
        PATIENT_NAMES[patient["name"]] = pid
    print(f"  Loaded {len(PATIENT_NAMES)} patient names")


def _extract_patient_id(prompt_text: str) -> str:
    _load_patient_mapping()
    for name, pid in PATIENT_NAMES.items():
        if name in prompt_text:
            return pid
    return "unknown"


# ---- Extract prompts from local parquet files ----
def extract_prompts_from_local_shard(shard_dir: Path) -> list[dict]:
    parquet_files = sorted(glob(str(shard_dir / "*.parquet")))
    if not parquet_files:
        raise ValueError(f"No parquet files in {shard_dir}")

    prompts = []
    for pf in parquet_files:
        print(f"  Reading {Path(pf).name} ...")
        df = pd.read_parquet(pf)

        for _, row in df.iterrows():
            messages = row.get("messages", None)
            if messages is None or len(messages) == 0:
                continue

            # Find first user message
            user_content = None
            for msg in messages:
                if isinstance(msg, dict) and msg.get("role") == "user":
                    user_content = msg["content"]
                    break

            if not user_content:
                continue

            # Try user question first, fall back to system_prompt for patient matching
            patient_id = _extract_patient_id(user_content)
            if patient_id == "unknown":
                sys_prompt = str(row.get("system_prompt", ""))
                patient_id = _extract_patient_id(sys_prompt)

            prompts.append({
                "prompt": [{"role": "user", "content": user_content}],
                "data_source": "longhealth_synthesized",
                "patient_id": patient_id,
                # Stash patient_id inside extra_info — veRL preserves this through the pipeline
                "extra_info": {"patient_id": patient_id},
                "reward_model": {"ground_truth": "", "style": "rule"},
            })

    return prompts


def main():
    data_root = Path("/Users/csuda/cartridges-workspace/data/hf_shards")
    out_dir = Path("/Users/csuda/cartridges-workspace/processed_data")
    out_dir.mkdir(parents=True, exist_ok=True)

    shard_dirs = sorted(data_root.glob("shard_*"))
    print(f"Found {len(shard_dirs)} shards: {[s.name for s in shard_dirs]}\n")

    all_prompts = []
    for shard_dir in shard_dirs:
        print(f"Processing {shard_dir.name} ...")
        prompts = extract_prompts_from_local_shard(shard_dir)
        print(f"  → {len(prompts)} prompts\n")
        all_prompts.extend(prompts)

    print(f"Total prompts: {len(all_prompts)}")

    # Val = small random subset (real eval is LongHealth benchmark, not val loss)
    random.seed(42)
    val_prompts = random.sample(all_prompts, min(500, len(all_prompts)))

    train_df = pd.DataFrame(all_prompts)
    val_df = pd.DataFrame(val_prompts)

    train_path = out_dir / "train.parquet"
    val_path = out_dir / "val.parquet"

    train_df.to_parquet(train_path)
    val_df.to_parquet(val_path)

    print(f"\n✓ {train_path}  ({len(train_df)} rows)")
    print(f"✓ {val_path}  ({len(val_df)} rows)")
    print(f"\nSample row:\n{json.dumps(all_prompts[0], indent=2)}")


if __name__ == "__main__":
    main()
