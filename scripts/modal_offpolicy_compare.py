"""
Off-policy cartridge training on Modal for comparison experiment.

Uses local_offpolicy_train.py which has proper token tracking for fair comparison.
Key difference from on-policy: teacher generates responses (not student).

Usage:
    modal run --detach scripts/modal_offpolicy_compare.py
"""

import modal

WORKSPACE_VERSION = "v50-offpolicy-compare"
GPU = "A100-80GB"

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
        "git clone --recurse-submodules --depth 1 "
        "https://github.com/chandrasuda/cartridges-workspace.git /opt/workspace"
    )
    .run_commands(
        "pip install -e /opt/workspace/cartridges "
        "&& pip install -e /opt/workspace/verl "
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
    timeout=86400,
    min_containers=0,
    max_containers=1,
    scaledown_window=600,
    volumes={"/results": results_volume},
)
def train(total_steps: int = 5, batch_size: int = 4, lr: float = 0.02, max_eval_samples: int = 50, num_pregenerated: int = 20):
    """Run off-policy training with token tracking."""
    import subprocess, os, sys

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    train_parquet = "/opt/workspace/data/on_policy/train.parquet"
    assert os.path.exists(train_parquet), f"Missing {train_parquet}"

    cmd = [
        sys.executable, "/opt/workspace/scripts/local_offpolicy_train.py",
        "--model", "meta-llama/Llama-3.2-3B-Instruct",
        "--train-parquet", train_parquet,
        "--num-tokens", "512",
        "--lr", str(lr),
        "--total-steps", str(total_steps),
        "--batch-size", str(batch_size),
        "--max-eval-samples", str(max_eval_samples),
        "--num-pregenerated", str(num_pregenerated),
        "--save-dir", "/results/offpolicy",
    ]

    print(f"Running off-policy training: {' '.join(cmd)}")
    result = subprocess.run(cmd, env=env, stdout=sys.stdout, stderr=sys.stderr)
    
    results_volume.commit()
    return result.returncode


@app.local_entrypoint()
def main(
    total_steps: int = 5,
    batch_size: int = 4,
    lr: float = 0.02,
    max_eval_samples: int = 50,
    num_pregenerated: int = 20,
):
    exit_code = train.remote(
        total_steps=total_steps,
        batch_size=batch_size,
        lr=lr,
        max_eval_samples=max_eval_samples,
        num_pregenerated=num_pregenerated,
    )
    print(f"Training finished with exit code: {exit_code}")
