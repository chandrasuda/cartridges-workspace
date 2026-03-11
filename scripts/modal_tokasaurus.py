"""
Modal deployment for Tokasaurus with Cartridge support.

Launches a Tokasaurus inference server on a Modal GPU that supports
the /custom/cartridge/completions endpoint for KV tensor injection.

Usage:
    # One-time setup:
    pip install modal
    modal setup
    modal secret create huggingface-secret HF_TOKEN=hf_YOUR_TOKEN

    # Deploy (keeps running, gives you a URL):
    modal deploy scripts/modal_tokasaurus.py

    # Test:
    curl https://<your-modal-url>/ping
    curl -X POST https://<your-modal-url>/custom/cartridge/completions \\
        -H 'Content-Type: application/json' \\
        -d '{"model":"default","prompt":"Hello world","max_tokens":64}'
"""

import modal

MODEL = "meta-llama/Llama-3.2-3B-Instruct"
GPU = "A100-40GB"  # A100 for fast batched generation during on-policy training
PORT = 10210
HF_CACHE = "/root/.cache/huggingface"


# ---------------------------------------------------------------------------
# Pre-download model weights during image build (no GPU needed)
# ---------------------------------------------------------------------------
def download_model():
    """Runs during `modal deploy` — downloads weights into the image."""
    import os
    os.environ["HF_HOME"] = HF_CACHE
    from huggingface_hub import snapshot_download
    snapshot_download(MODEL, ignore_patterns=["*.gguf", "original/*"])
    print(f"✓ Downloaded {MODEL}")


# ---------------------------------------------------------------------------
# Image: install Tokasaurus + bake in model weights
# ---------------------------------------------------------------------------
image = (
    # nvidia/cuda devel image includes nvcc + headers needed by flashinfer JIT
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04",
        add_python="3.12",
    )
    .apt_install("git")
    .env({
        "CUDA_HOME": "/usr/local/cuda",
        "HF_HOME": HF_CACHE,
    })
    .pip_install(
        "torch==2.6.0",
        "flashinfer-python==0.2.0.post2",
        extra_index_url="https://flashinfer.ai/whl/cu124/torch2.6/",
    )
    .pip_install(
        "transformers==4.53.0",
        "huggingface-hub",
        "pydra-config>=0.0.13",
        "accelerate",
        "art",
        "statsd",
        "fastapi",
        "ninja",
        "tabulate",
        "uvicorn",
        "typer",
        "openai",
        "loguru",
        "python-multipart",
        "tqdm",
        "wandb",
        "boto3",
        "requests",  # for warmup request during server startup
    )
    .run_commands("echo 'toka-v3-warmup-on-startup'")
    .run_commands(
        "pip install git+https://github.com/chandrasuda/tokasaurus.git@geoff/cartridges"
    )
    # Download model weights at build time (baked into image, no runtime download)
    .run_function(
        download_model,
        secrets=[modal.Secret.from_name("huggingface-secret")],
    )
)

app = modal.App("tokasaurus-cartridge-server", image=image)

# Shared volume with training container so cartridge syncs work via local path
cartridge_volume = modal.Volume.from_name("onpolicy-v2-results", create_if_missing=True)


@app.function(
    gpu=GPU,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=86400,
    min_containers=1,
    max_containers=1,
    scaledown_window=3600,
    volumes={"/results": cartridge_volume},  # same volume as training container
)
@modal.web_server(port=PORT, startup_timeout=900)  # 15 min for compile
def serve():
    """Start Tokasaurus server and warm up torch.compile before accepting requests."""
    import subprocess
    import time
    import requests as http_requests

    # Start Tokasaurus in background
    subprocess.Popen(
        [
            "toka",
            f"model={MODEL}",
            f"port={PORT}",
            # 32K tokens ≈ 3.5 GB KV cache.  Model weights ≈ 6 GB → ~10 GB total.
            "kv_cache_num_tokens=32768",
            # torch_compile + CUDA graphs = fastest per-request (1.33s vs 2.93s)
            "torch_compile=True",
            "log_level=INFO",
        ]
    )

    # Wait for server to be ready
    print("Waiting for Tokasaurus to start...")
    for attempt in range(60):  # 5 min max
        try:
            r = http_requests.get(f"http://localhost:{PORT}/ping", timeout=5)
            if r.status_code == 200:
                print(f"Tokasaurus responding to ping (attempt {attempt+1})")
                break
        except Exception:
            pass
        time.sleep(5)

    # Warm up torch.compile with a real inference request
    print("Warming up torch.compile with inference request...")
    for attempt in range(10):  # 5 min max for compile
        try:
            r = http_requests.post(
                f"http://localhost:{PORT}/custom/cartridge/completions",
                json={"model": "default", "prompt": [128000, 9906], "max_tokens": 5},
                timeout=180,  # compile can take 90s+
            )
            if r.status_code == 200:
                print(f"✓ Tokasaurus warmed up and ready! (took {attempt+1} attempts)")
                break
        except Exception as e:
            print(f"  Warmup attempt {attempt+1}/10: {e}")
        time.sleep(30)

    print("Tokasaurus server fully ready for requests")
