"""
Run the OFFICIAL cartridges longhealth_train.py from HazyResearch.

This uses their code exactly as published - no modifications.
Their config:
  - LR = 0.02
  - global_batch_size = 32
  - epochs = 2
  - HF datasets with pre-computed teacher logprobs

Usage:
    modal run --detach scripts/modal_official_offpolicy.py
"""

import modal

WORKSPACE_VERSION = "v70-official"
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
        "CARTRIDGES_DIR": "/opt/cartridges",
        "CARTRIDGES_OUTPUT_DIR": "/results",
        "NUM_TOKENS": "512",  # Their default is 2048, but we use 512 for comparison
        "MODEL": "llama",
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
        "requests", "transformers==4.53.0", "pandas", "pyarrow", "huggingface_hub",
        "pydrantic", "wandb",
    )
)

results_volume = modal.Volume.from_name("comparison-results", create_if_missing=True)
app = modal.App("official-offpolicy", image=image)


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
    """Run the OFFICIAL longhealth_train.py from cartridges repo."""
    import subprocess
    import os
    import sys
    
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["WANDB_MODE"] = "disabled"  # Don't need wandb for this test
    
    # Run their official training script
    cmd = [
        sys.executable,
        "/opt/cartridges/examples/benchmarks/longhealth/longhealth_train.py",
    ]
    
    print("=" * 70)
    print("RUNNING OFFICIAL CARTRIDGES longhealth_train.py")
    print("=" * 70)
    print("Config from their code:")
    print("  - LR = 0.02")
    print("  - global_batch_size = 32")
    print("  - epochs = 2")
    print("  - HF datasets: hazyresearch/m07d11_longhealth_synthesize_llama-3.2-3b_p10_n65536-*")
    print("  - NUM_TOKENS = 512 (our override)")
    print("=" * 70)
    
    result = subprocess.run(cmd, env=env, stdout=sys.stdout, stderr=sys.stderr)
    
    results_volume.commit()
    return result.returncode


@app.local_entrypoint()
def main():
    print("Launching OFFICIAL cartridges training...")
    exit_code = train.remote()
    print(f"Training finished with exit code: {exit_code}")
