#!/usr/bin/env python3
"""Download all 196K QA conversations from HuggingFace to local disk."""
import os
import requests
import pyarrow.parquet as pq
import io
import pandas as pd
from huggingface_hub import HfApi

OUT_DIR = "/Users/csuda/cartridges-workspace/data/hf_shards"

SHARDS = [
    "hazyresearch/m07d11_longhealth_synthesize_llama-3.2-3b_p10_n65536-0",
    "hazyresearch/m07d11_longhealth_synthesize_llama-3.2-3b_p10_n65536-1",
    "hazyresearch/m07d11_longhealth_synthesize_llama-3.2-3b_p10_n65536-2",
]

api = HfApi()
total_rows = 0

for shard_idx, repo_id in enumerate(SHARDS):
    shard_name = f"shard_{shard_idx}"
    shard_dir = os.path.join(OUT_DIR, shard_name)
    os.makedirs(shard_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Shard {shard_idx}: {repo_id}")
    print(f"{'='*60}")

    repo_files = api.list_repo_files(repo_id, repo_type="dataset")
    parquet_files = sorted(f for f in repo_files if f.startswith("data/") and f.endswith(".parquet"))
    print(f"  {len(parquet_files)} parquet files")

    shard_rows = 0
    for pf in parquet_files:
        local_path = os.path.join(shard_dir, os.path.basename(pf))

        if os.path.exists(local_path):
            existing = pd.read_parquet(local_path)
            print(f"  ✓ {os.path.basename(pf)} already exists ({len(existing)} rows)")
            shard_rows += len(existing)
            continue

        url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{pf}"
        print(f"  Downloading {os.path.basename(pf)}...", end=" ", flush=True)
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()

        # Save raw parquet
        with open(local_path, "wb") as f:
            f.write(resp.content)

        n = len(pq.read_table(local_path))
        shard_rows += n
        print(f"{n} rows")

    total_rows += shard_rows
    print(f"  Shard {shard_idx} total: {shard_rows} rows")

print(f"\n{'='*60}")
print(f"DONE: {total_rows} total conversations across {len(SHARDS)} shards")
print(f"Saved to: {OUT_DIR}")
print(f"{'='*60}")
