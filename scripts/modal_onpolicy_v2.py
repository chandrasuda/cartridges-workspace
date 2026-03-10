"""
Optimized on-policy cartridge training on Modal.
No Ray, no PPO, no FSDP — direct training loop with Tokasaurus generation.

Usage:
    modal run --detach scripts/modal_onpolicy_v2.py
"""

import modal

TOKASAURUS_URL = "https://kiran1234c--tokasaurus-cartridge-server-serve.modal.run"
GPU = "A100-80GB"
WORKSPACE_VERSION = "v32-full-optimized-run"

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04",
        add_python="3.12",
    )
    .apt_install("git", "patch")
    .env({
        "CUDA_HOME": "/usr/local/cuda",
        "CARTRIDGES_DIR": "/opt/workspace/cartridges",
        "CARTRIDGES_OUTPUT_DIR": "/tmp/cartridge_output",
    })
    .pip_install("torch==2.6.0", "packaging", "numpy")
    .run_commands(
        "pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl",
        "pip install flashinfer-python==0.2.0.post2 --extra-index-url https://flashinfer.ai/whl/cu124/torch2.6/",
    )
    .run_commands(f"echo '{WORKSPACE_VERSION}'")
    .run_commands(
        "git clone --recurse-submodules --depth 1 "
        "https://github.com/chandrasuda/cartridges-workspace.git /opt/workspace"
    )
    .run_commands(
        "pip install -e /opt/workspace/cartridges "
        "&& pip install -e /opt/workspace/verl "
        "&& pip install -e /opt/workspace/tokasaurus"
    )
    .pip_install(
        "requests", "aiohttp", "transformers==4.53.0", "pandas", "pyarrow",
        "omegaconf", "hydra-core", "cachetools",
    )
)

results_volume = modal.Volume.from_name("onpolicy-v2-results", create_if_missing=True)
app = modal.App("onpolicy-v2-optimized", image=image)


@app.function(
    gpu=GPU,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=86400,
    min_containers=0,
    max_containers=1,
    scaledown_window=600,
    volumes={"/results": results_volume},
)
def train():
    import subprocess, os, sys

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    train_parquet = "/opt/workspace/data/on_policy/train.parquet"
    assert os.path.exists(train_parquet), f"Missing {train_parquet}"

    cmd = [
        sys.executable, "/opt/workspace/scripts/online_cartridge_train.py",
        "--model", "meta-llama/Llama-3.2-3B-Instruct",
        "--tokasaurus-url", TOKASAURUS_URL,
        "--train-parquet", train_parquet,
        "--num-tokens", "512",
        "--lr", "0.02",
        "--total-steps", "500",
        "--batch-size", "256",
        "--eval-every", "50",
        "--save-dir", "/results/onpolicy",
    ]

    print(f"Running optimized on-policy trainer...")
    result = subprocess.run(cmd, env=env, stdout=sys.stdout, stderr=sys.stderr)
    results_volume.commit()
    return result.returncode


@app.local_entrypoint()
def main():
    exit_code = train.remote()
    print(f"Training finished with exit code: {exit_code}")
