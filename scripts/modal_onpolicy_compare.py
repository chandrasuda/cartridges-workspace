"""
On-policy cartridge training on Modal for comparison experiment.

Uses local_cartridge_train.py which has proper token tracking for fair comparison.

Usage:
    modal run --detach scripts/modal_onpolicy_compare.py
"""

import modal

WORKSPACE_VERSION = "v87-fix-response-length"
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
app = modal.App("onpolicy-compare", image=image)


@app.function(
    gpu=GPU,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=TIMEOUT_HOURS * 3600,
    min_containers=0,
    max_containers=1,
    scaledown_window=600,
    volumes={"/results": results_volume},
)
def train(total_steps: int = 1000, batch_size: int = 32, lr: float = 0.02, eval_every: int = 50, save_every: int = 50):
    """Run on-policy training overnight with full eval every N steps."""
    import subprocess, os, sys

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    train_parquet = "/opt/workspace/data/on_policy/train.parquet"
    assert os.path.exists(train_parquet), f"Missing {train_parquet}"

    cmd = [
        sys.executable, "/opt/workspace/scripts/local_cartridge_train.py",
        "--model", "meta-llama/Llama-3.2-3B-Instruct",
        "--train-parquet", train_parquet,
        "--num-tokens", "512",
        "--lr", str(lr),
        "--total-steps", str(total_steps),
        "--batch-size", str(batch_size),
        "--eval-every", str(eval_every),
        "--save-every", str(save_every),
        "--save-dir", "/results/onpolicy",
    ]  # No --max-eval-samples means full eval

    print(f"Running on-policy training: {' '.join(cmd)}")
    result = subprocess.run(cmd, env=env, stdout=sys.stdout, stderr=sys.stderr)
    
    results_volume.commit()
    return result.returncode


@app.local_entrypoint()
def main(
    total_steps: int = 1000,
    batch_size: int = 32,
    lr: float = 0.02,
    eval_every: int = 50,
    save_every: int = 50,
):
    exit_code = train.remote(
        total_steps=total_steps,
        batch_size=batch_size,
        lr=lr,
        eval_every=eval_every,
        save_every=save_every,
    )
    print(f"Training finished with exit code: {exit_code}")
