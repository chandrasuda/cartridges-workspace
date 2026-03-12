"""
Off-policy cartridge training on Modal — Uses PRE-COMPUTED teacher logprobs.

The HazyResearch team already synthesized ~196K QA conversations with teacher logprobs.
This script just loads that data and trains - NO synthesis needed!

Data source: hazyresearch/m07d11_longhealth_synthesize_llama-3.2-3b_p10_n65536-{0,1,2}

Usage:
    modal run --detach scripts/modal_offpolicy_compare.py
"""

import modal

WORKSPACE_VERSION = "v61-persistent-logs"
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
        "CARTRIDGES_DIR": "/opt/workspace/cartridges",
        "CARTRIDGES_OUTPUT_DIR": "/results",
    })
    .pip_install("torch==2.6.0", "packaging", "numpy")
    .run_commands(
        "pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl",
    )
    .run_commands(
        f"echo '{WORKSPACE_VERSION}' && "
        "git clone --depth 1 https://github.com/chandrasuda/cartridges-workspace.git /opt/workspace"
    )
    .run_commands(
        "git clone --depth 1 https://github.com/HazyResearch/cartridges.git /opt/workspace/cartridges "
        "&& git clone --depth 1 -b geoff/cartridges https://github.com/ScalingIntelligence/tokasaurus.git /opt/workspace/tokasaurus"
    )
    .run_commands(
        "pip install -e /opt/workspace/cartridges "
        "&& pip install -e /opt/workspace/tokasaurus"
    )
    .pip_install(
        "requests", "transformers==4.53.0", "pandas", "pyarrow", "huggingface_hub",
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
def train(total_steps: int = 1000, batch_size: int = 4, lr: float = 0.02, eval_every: int = 50, save_every: int = 50):
    """Run off-policy training using PRE-COMPUTED teacher logprobs from HF."""
    import subprocess, os, sys
    from huggingface_hub import snapshot_download

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    # Download HF shard with pre-computed logprobs
    print("Downloading HF shard with pre-computed teacher logprobs...")
    hf_shard_dir = snapshot_download(
        repo_id="hazyresearch/m07d11_longhealth_synthesize_llama-3.2-3b_p10_n65536-0",
        repo_type="dataset",
        local_dir="/opt/hf_shard",
    )
    print(f"Downloaded to: {hf_shard_dir}")
    
    # The parquets are in data/ subdirectory
    data_dir = os.path.join(hf_shard_dir, "data")
    if not os.path.exists(data_dir):
        data_dir = hf_shard_dir  # Fallback if no data/ subdir
    print(f"Using data dir: {data_dir}")

    cmd = [
        sys.executable, "/opt/workspace/scripts/local_offpolicy_train.py",
        "--model", "meta-llama/Llama-3.2-3B-Instruct",
        "--hf-shard-dir", data_dir,
        "--num-tokens", "512",
        "--lr", str(lr),
        "--total-steps", str(total_steps),
        "--batch-size", str(batch_size),
        "--eval-every", str(eval_every),
        "--save-every", str(save_every),
        "--save-dir", "/results/offpolicy",
    ]  # No --max-eval-samples means full eval

    print(f"Running off-policy training (pre-computed logprobs): {' '.join(cmd)}")
    print(f"  Using PRE-COMPUTED teacher logprobs from HF shard")
    print(f"  Train for exactly 1 epoch (NO DATA REUSE)")
    result = subprocess.run(cmd, env=env, stdout=sys.stdout, stderr=sys.stderr)
    
    results_volume.commit()
    return result.returncode


@app.local_entrypoint()
def main(
    total_steps: int = 1000,
    batch_size: int = 4,
    lr: float = 0.02,
    eval_every: int = 50,
    save_every: int = 50,
):
    print("=" * 70)
    print("OFF-POLICY TRAINING (PRE-COMPUTED LOGPROBS)")
    print("=" * 70)
    print(f"  total_steps: {total_steps}")
    print(f"  batch_size: {batch_size}")
    print(f"  samples to load: {total_steps * batch_size}")
    print(f"  data: HF shard with pre-computed teacher logprobs")
    print("=" * 70)
    
    exit_code = train.remote(
        total_steps=total_steps,
        batch_size=batch_size,
        lr=lr,
        eval_every=eval_every,
        save_every=save_every,
    )
    print(f"Training finished with exit code: {exit_code}")
