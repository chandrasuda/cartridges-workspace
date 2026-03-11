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
    # Install Tokasaurus from the workspace repo (same source as training image)
    # so that fixes to tokasaurus/ in the workspace are picked up by both images.
    .run_commands("echo 'toka-v4-workspace-install-failfast-local'")
    .run_commands(
        "git clone --recurse-submodules --depth 1 "
        "https://github.com/chandrasuda/cartridges-workspace.git /opt/workspace"
    )
    .run_commands(
        "pip install -e /opt/workspace/tokasaurus"
    )
    # Download model weights at build time (baked into image, no runtime download)
    .run_function(
        download_model,
        secrets=[modal.Secret.from_name("huggingface-secret")],
    )
)

# Shared volume for cartridge checkpoints (same as training job)
results_volume = modal.Volume.from_name("onpolicy-v2-results", create_if_missing=True)

app = modal.App("tokasaurus-cartridge-server", image=image)


@app.function(
    gpu=GPU,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=86400,
    min_containers=1,
    max_containers=1,
    scaledown_window=3600,
    volumes={"/results": results_volume},  # Shared with training job for cartridge checkpoints
)
@modal.web_server(port=PORT, startup_timeout=600)
def serve():
    """Start Tokasaurus server. Model weights are already in the image."""
    import subprocess
    import time
    import requests
    import threading
    import os
    import signal
    import sys

    # -------------------------------------------------------------------------
    # GPU + Process monitoring thread
    # -------------------------------------------------------------------------
    def monitor_loop(proc):
        """Background thread: logs GPU memory + checks if toka process died."""
        import time
        while True:
            time.sleep(30)  # Log every 30 seconds
            
            # Check if toka process died
            ret = proc.poll()
            if ret is not None:
                print(f"!!! TOKA PROCESS DIED with exit code {ret} !!!", flush=True)
                # Try to get any stderr output
                try:
                    stdout, stderr = proc.communicate(timeout=1)
                    if stdout:
                        print(f"TOKA STDOUT: {stdout.decode()[-2000:]}", flush=True)
                    if stderr:
                        print(f"TOKA STDERR: {stderr.decode()[-2000:]}", flush=True)
                except:
                    pass
                print("Exiting monitor - toka is dead", flush=True)
                return
            
            # Log GPU memory usage
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    parts = result.stdout.strip().split(", ")
                    if len(parts) >= 3:
                        mem_used, mem_total, gpu_util = parts[0], parts[1], parts[2]
                        print(f"[MONITOR] GPU: {mem_used}/{mem_total} MB ({gpu_util}% util) | toka pid={proc.pid} running", flush=True)
            except Exception as e:
                print(f"[MONITOR] nvidia-smi failed: {e}", flush=True)

    # -------------------------------------------------------------------------
    # Start toka process
    # -------------------------------------------------------------------------
    print(f"Starting Tokasaurus with model={MODEL}, port={PORT}", flush=True)
    # Set cartridge_dir to the shared volume path where training saves checkpoints
    proc = subprocess.Popen(
        ["toka", f"model={MODEL}", f"port={PORT}",
         "kv_cache_num_tokens=32768", "torch_compile=False", "log_level=INFO",
         "cartridge_dir=/results/onpolicy/cartridge_checkpoints"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # Merge stderr into stdout
    )
    print(f"Toka process started with PID {proc.pid}", flush=True)

    # Start monitoring thread
    monitor_thread = threading.Thread(target=monitor_loop, args=(proc,), daemon=True)
    monitor_thread.start()

    # -------------------------------------------------------------------------
    # Stream toka output to Modal logs in background
    # -------------------------------------------------------------------------
    def stream_output(proc):
        """Stream toka stdout/stderr to Modal logs."""
        try:
            for line in iter(proc.stdout.readline, b''):
                if line:
                    print(f"[TOKA] {line.decode().rstrip()}", flush=True)
        except Exception as e:
            print(f"[TOKA OUTPUT ERROR] {e}", flush=True)
    
    output_thread = threading.Thread(target=stream_output, args=(proc,), daemon=True)
    output_thread.start()

    # -------------------------------------------------------------------------
    # Volume refresh thread: reload every few seconds so Tokasaurus sees new
    # cartridge checkpoint files written by the training container.
    # Modal volumes require explicit reload() to pick up changes from other
    # containers — the mounted view is otherwise stale for the lifetime of
    # this container.
    # -------------------------------------------------------------------------
    def volume_refresh_loop():
        """Periodically reload the shared volume so newly-written cartridges are visible."""
        reload_count = 0
        last_log_time = 0
        while True:
            time.sleep(5)  # Refresh every 5 seconds
            try:
                results_volume.reload()
                reload_count += 1
                now = time.time()
                if now - last_log_time > 60:  # Log once per minute to avoid spam
                    print(f"[VOLUME] Reloaded {reload_count} times (cartridge dir refreshed)", flush=True)
                    last_log_time = now
            except Exception as e:
                print(f"[VOLUME] Reload error: {e}", flush=True)

    volume_thread = threading.Thread(target=volume_refresh_loop, daemon=True)
    volume_thread.start()
    print("Started volume refresh thread (5s interval) — new cartridges will be visible within 5s", flush=True)

    # -------------------------------------------------------------------------
    # Wait for server to be ready
    # -------------------------------------------------------------------------
    print("Waiting for tokasaurus to be ready...", flush=True)
    max_wait = 540  # 9 minutes max
    start = time.time()
    
    while time.time() - start < max_wait:
        # Check if process died during startup
        if proc.poll() is not None:
            print(f"!!! TOKA DIED DURING STARTUP with exit code {proc.returncode} !!!", flush=True)
            return
        
        try:
            resp = requests.get(f"http://localhost:{PORT}/ping", timeout=5)
            if resp.status_code == 200 and resp.json().get("message") == "pong":
                print(f"Ping OK after {time.time() - start:.1f}s, doing warmup inference...", flush=True)
                # Do one warmup inference
                warmup_resp = requests.post(
                    f"http://localhost:{PORT}/v1/chat/completions",
                    json={
                        "model": MODEL,
                        "messages": [{"role": "user", "content": "Hi"}],
                        "max_tokens": 5,
                    },
                    timeout=300,
                )
                if warmup_resp.status_code == 200:
                    print(f"✓ Tokasaurus ready after {time.time() - start:.1f}s", flush=True)
                    return  # Return after ready - don't block!
                else:
                    print(f"Warmup failed: {warmup_resp.status_code} {warmup_resp.text[:200]}", flush=True)
        except requests.exceptions.RequestException as e:
            if time.time() - start > 60:  # Only log after 60s to avoid spam
                print(f"Still waiting for toka... ({time.time() - start:.0f}s) - {type(e).__name__}", flush=True)
        time.sleep(1)
    
    print("WARNING: Tokasaurus may not be fully ready after 9 minutes!", flush=True)
