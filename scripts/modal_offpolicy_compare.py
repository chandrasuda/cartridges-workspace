"""
Off-policy cartridge training on Modal — PAPER-MATCHING VERSION.

Key differences from on-policy:
1. SYNTHESIS PHASE: Generate (total_steps × batch_size) samples with teacher logprobs
2. TRAINING PHASE: Train for exactly 1 epoch (NO DATA REUSE)
3. Teacher model only needed during synthesis, not training

This matches the paper: "No synthetically generated data is reused (i.e. training
proceeds for one epoch)." - Section 5, Figure 3 caption.

Usage:
    modal run --detach scripts/modal_offpolicy_compare.py
"""

import modal

WORKSPACE_VERSION = "v53-overnight"
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
        "requests", "transformers==4.53.0", "pandas", "pyarrow",
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
def train(total_steps: int = 500, batch_size: int = 4, lr: float = 0.02, eval_every: int = 50, save_every: int = 50):
    """Run off-policy training (paper-matching: 1 epoch, no data reuse)."""
    import subprocess, os, sys

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    train_parquet = "/opt/workspace/data/on_policy/train.parquet"
    assert os.path.exists(train_parquet), f"Missing {train_parquet}"

    # Paper-matching: auto-calculates num_samples = total_steps × batch_size
    # NO --num-pregenerated argument needed!
    cmd = [
        sys.executable, "/opt/workspace/scripts/local_offpolicy_train.py",
        "--model", "meta-llama/Llama-3.2-3B-Instruct",
        "--train-parquet", train_parquet,
        "--num-tokens", "512",
        "--lr", str(lr),
        "--total-steps", str(total_steps),
        "--batch-size", str(batch_size),
        "--eval-every", str(eval_every),
        "--save-every", str(save_every),
        "--save-dir", "/results/offpolicy",
    ]  # No --max-eval-samples means full eval

    print(f"Running off-policy training (paper-matching): {' '.join(cmd)}")
    print(f"  Will synthesize {total_steps * batch_size} samples with teacher logprobs")
    print(f"  Train for exactly 1 epoch (NO DATA REUSE)")
    result = subprocess.run(cmd, env=env, stdout=sys.stdout, stderr=sys.stderr)
    
    results_volume.commit()
    return result.returncode


@app.local_entrypoint()
def main(
    total_steps: int = 500,
    batch_size: int = 4,
    lr: float = 0.02,
    eval_every: int = 50,
    save_every: int = 50,
):
    print("=" * 70)
    print("OFF-POLICY TRAINING (PAPER-MATCHING)")
    print("=" * 70)
    print(f"  total_steps: {total_steps}")
    print(f"  batch_size: {batch_size}")
    print(f"  samples to synthesize: {total_steps * batch_size} (NO REUSE)")
    print("=" * 70)
    
    exit_code = train.remote(
        total_steps=total_steps,
        batch_size=batch_size,
        lr=lr,
        eval_every=eval_every,
        save_every=save_every,
    )
    print(f"Training finished with exit code: {exit_code}")
