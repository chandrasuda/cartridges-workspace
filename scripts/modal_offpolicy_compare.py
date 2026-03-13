"""
Off-policy cartridge training on Modal — Runs EXACT official cartridges code.

This runs the official longhealth_train.py from HazyResearch with:
- NUM_TOKENS=512 (our cache size)
- MODEL=llama (Llama-3.2-3B-Instruct)

Their code handles everything: truncation, packing, evaluation, etc.

Usage:
    modal run --detach scripts/modal_offpolicy_compare.py
"""

import modal

WORKSPACE_VERSION = "v73-run-exact-official-code"
GPU = "A100-80GB"
TIMEOUT_HOURS = 24

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04",
        add_python="3.12",
    )
    .apt_install("git")
    .env({
        "CUDA_HOME": "/usr/local/cuda",
        # These are the ONLY changes from their defaults:
        "NUM_TOKENS": "512",  # Cache size (their default is 2048)
        "MODEL": "llama",     # Use Llama-3.2-3B (their code supports this)
        "CARTRIDGES_DIR": "/opt/cartridges",
        "CARTRIDGES_OUTPUT_DIR": "/results",
        "WANDB_MODE": "disabled",  # We don't need wandb
    })
    .pip_install("torch==2.6.0", "packaging", "numpy")
    .run_commands(
        "pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl",
    )
    .run_commands(
        f"echo '{WORKSPACE_VERSION}' && "
        "git clone --depth 1 https://github.com/HazyResearch/cartridges.git /opt/cartridges"
    )
    .run_commands(
        "git clone --depth 1 -b geoff/cartridges https://github.com/ScalingIntelligence/tokasaurus.git /opt/tokasaurus"
    )
    .run_commands(
        "pip install -e /opt/cartridges && pip install -e /opt/tokasaurus"
    )
    .pip_install(
        "requests", "transformers==4.53.0", "pandas", "pyarrow", "huggingface_hub", "pydrantic",
    )
)

results_volume = modal.Volume.from_name("comparison-results", create_if_missing=True)
app = modal.App("offpolicy-compare", image=image)


@app.function(
    gpu=GPU,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=TIMEOUT_HOURS * 3600,
    min_containers=0,
    max_containers=1,
    scaledown_window=600,
    volumes={"/results": results_volume},
)
def train():
    """
    Run the EXACT official longhealth_train.py from cartridges repo.
    
    Their config (from longhealth_train.py):
    - lr=2e-2 (0.02)
    - epochs=2
    - global_batch_size=32
    - packed_seq_length=2048
    - packing_mode="truncate"
    - generate_eval_every_n_steps=128
    - save_every_n_steps=512
    
    We only change via env vars:
    - NUM_TOKENS=512 (cache size)
    - MODEL=llama
    """
    import subprocess
    import sys
    import os
    
    print("=" * 70)
    print("RUNNING OFFICIAL CARTRIDGES longhealth_train.py")
    print("=" * 70)
    print("Environment (our only changes):")
    print(f"  NUM_TOKENS={os.environ.get('NUM_TOKENS')}")
    print(f"  MODEL={os.environ.get('MODEL')}")
    print("")
    print("Their config (unchanged):")
    print("  lr=0.02")
    print("  epochs=2") 
    print("  global_batch_size=32")
    print("  packed_seq_length=2048")
    print("  packing_mode=truncate")
    print("  generate_eval_every_n_steps=128")
    print("=" * 70)
    
    # Run their EXACT script
    cmd = [
        sys.executable,
        "/opt/cartridges/examples/benchmarks/longhealth/longhealth_train.py",
    ]
    
    result = subprocess.run(cmd, stdout=sys.stdout, stderr=sys.stderr)
    
    # Copy results to our volume for comparison
    import shutil
    import json
    results_dir = os.environ.get("CARTRIDGES_OUTPUT_DIR", "/results")
    
    # Find the output directory (their code creates timestamped dirs)
    for item in os.listdir(results_dir):
        item_path = os.path.join(results_dir, item)
        if os.path.isdir(item_path) and "longhealth" in item:
            # Copy to our expected location
            os.makedirs("/results/offpolicy", exist_ok=True)
            for f in os.listdir(item_path):
                src = os.path.join(item_path, f)
                dst = os.path.join("/results/offpolicy", f)
                if os.path.isfile(src):
                    shutil.copy2(src, dst)
                elif os.path.isdir(src):
                    if os.path.exists(dst):
                        shutil.rmtree(dst)
                    shutil.copytree(src, dst)
            print(f"Copied results from {item_path} to /results/offpolicy")
    
    results_volume.commit()
    return result.returncode


@app.local_entrypoint()
def main():
    print("=" * 70)
    print("OFFICIAL OFF-POLICY TRAINING")
    print("Running exact cartridges longhealth_train.py")
    print("=" * 70)
    
    exit_code = train.remote()
    print(f"Training finished with exit code: {exit_code}")
