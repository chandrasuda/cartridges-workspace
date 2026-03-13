"""
Off-policy cartridge training on Modal — Uses official cartridges components.

Same as official longhealth_train.py but with:
- NUM_TOKENS=512 (cache size)
- Eval every 50 steps (not 128)
- Step 0 eval included

Usage:
    modal run --detach scripts/modal_offpolicy_compare.py
"""

import modal

WORKSPACE_VERSION = "v79-patient-doc-init"
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
        "WANDB_MODE": "disabled",
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
    Off-policy training using official cartridges components.
    
    Same as their longhealth_train.py but with:
    - NUM_TOKENS=512 (cache size)
    - generate_eval_every_n_steps=50 (not 128)
    - Step 0 eval included
    """
    import os
    import json
    import time
    
    # Use their exact imports
    from cartridges.initialization import KVFromText
    from cartridges.train import GenerationEvalConfig, TrainConfig
    from cartridges.models.config import HFModelConfig
    from cartridges.datasets import TrainDataset, DataSource
    from cartridges.data.longhealth.evals import LongHealthMultipleChoiceGenerateDataset
    from cartridges.data.longhealth.resources import LongHealthResource
    from cartridges.utils.wandb import WandBConfig
    from cartridges.models.llama.modeling_llama import FlexLlamaForCausalLM
    import pydrantic
    import tempfile
    
    NUM_TOKENS = 512  # Our cache size
    NUM_PATIENTS = 10
    patient_ids = [f"patient_{idx:02d}" for idx in range(1, NUM_PATIENTS + 1)]
    
    # Write patient docs to temp file for cache init (same as on-policy)
    print("Building patient document text for cache initialization...")
    resource = LongHealthResource(config=LongHealthResource.Config(patient_ids=patient_ids))
    patient_text = resource.to_string()
    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
    tmp.write(patient_text)
    tmp.close()
    patient_text_path = tmp.name
    print(f"Patient text written to {patient_text_path} ({len(patient_text):,} chars)")
    
    # Their exact data sources
    data_sources = [
        "hazyresearch/m07d11_longhealth_synthesize_llama-3.2-3b_p10_n65536-0",
        "hazyresearch/m07d11_longhealth_synthesize_llama-3.2-3b_p10_n65536-1",
        "hazyresearch/m07d11_longhealth_synthesize_llama-3.2-3b_p10_n65536-2"
    ]
    
    print("=" * 70)
    print("OFF-POLICY TRAINING (official components)")
    print("=" * 70)
    print(f"  NUM_TOKENS (cache size): {NUM_TOKENS}")
    print(f"  lr: 0.02")
    print(f"  epochs: 2")
    print(f"  global_batch_size: 32")
    print(f"  generate_eval_every_n_steps: 50")
    print("=" * 70)
    
    # Build config - same as theirs but with eval every 50 steps
    config = TrainConfig(
        model=HFModelConfig(
            pretrained_model_name_or_path="meta-llama/Llama-3.2-3B-Instruct",
            model_cls=FlexLlamaForCausalLM,
        ),
        kv_cache_initializer=KVFromText.Config(
            max_tokens=NUM_TOKENS,
            text_source=patient_text_path,  # init from patient docs, same as on-policy
        ),
        
        lr=2e-2,
        epochs=2,
        global_batch_size=32,
        
        dataset=TrainDataset.Config(
            data_sources=[DataSource(path=source, type="hf") for source in data_sources],
            top_k_logits=20,
            packed_seq_length=2048,
            packing_mode="truncate",
        ),
        
        save_every_n_steps=50,
        generate_eval_every_n_steps=50,  # also evals at step 0 (generate_before_training defaults True)
        generate_evals=[
            GenerationEvalConfig(
                dataset=LongHealthMultipleChoiceGenerateDataset.Config(
                    patient_ids=patient_ids,
                ),
                name_for_wandb=f"longhealth_p{NUM_PATIENTS}",
                generate_max_new_tokens=512,
                batch_size=32,
                temperature=0.3,
            )
        ],
        distributed_backend="gloo",
        wandb=WandBConfig(tags=["train", "longhealth", "offpolicy"]),
        output_dir="/results/offpolicy",
        name="offpolicy_compare",
    )
    
    # Capture evals via logging handler (logger goes to stdout but we need to intercept)
    import re
    import sys
    import logging

    EVALS_PATH = "/results/offpolicy/evals.json"
    os.makedirs("/results/offpolicy", exist_ok=True)
    evals = []
    current_step = [0]  # Use list to allow mutation in nested function

    class EvalCaptureHandler(logging.Handler):
        """Logging handler that captures eval scores."""
        def emit(self, record):
            try:
                msg = self.format(record)
                # Capture score: {'generate_longhealth_p10/score': np.float64(0.28)}
                if 'score' in msg and 'float' in msg:
                    m = re.search(r'float\w*\(([\d.]+)\)', msg)
                    if m:
                        score = float(m.group(1))
                        step = current_step[0]
                        entry = {"step": step, "accuracy": round(score * 100, 1)}
                        if not any(e["step"] == step for e in evals):
                            evals.append(entry)
                            with open(EVALS_PATH, 'w') as f:
                                json.dump({"evals": evals}, f, indent=2)
                            results_volume.commit()
                            print(f"\n✓ EVAL SAVED — step={step}, accuracy={score*100:.1f}%\n", flush=True)
            except Exception as e:
                print(f"EvalCaptureHandler error: {e}", flush=True)

    # Add handler to cartridges.train logger
    eval_handler = EvalCaptureHandler()
    eval_handler.setLevel(logging.INFO)
    logging.getLogger("cartridges.train").addHandler(eval_handler)
    logging.getLogger().addHandler(eval_handler)  # root logger too

    # Also wrap stdout/stderr to track step from tqdm progress bars
    class StepTracker:
        def __init__(self, wrapped):
            self.wrapped = wrapped
        def write(self, text):
            self.wrapped.write(text)
            # Track step from tqdm: "Generating [step=0]" or "optimizer_step=50"
            m = re.search(r'step[=\]](\d+)', text)
            if m:
                current_step[0] = int(m.group(1))
        def flush(self):
            self.wrapped.flush()
        def fileno(self):
            return self.wrapped.fileno()

    sys.stdout = StepTracker(sys.stdout)
    sys.stderr = StepTracker(sys.stderr)

    # Run training (step 0 eval happens automatically since 0 % 50 == 0)
    print("Running training with pydrantic...", flush=True)
    pydrantic.main(config)

    results_volume.commit()
    print(f"Training complete! Evals saved to {EVALS_PATH}")
    return 0


@app.local_entrypoint()
def main():
    print("=" * 70)
    print("OFF-POLICY TRAINING")
    print("Using official cartridges components with eval every 50 steps")
    print("=" * 70)
    
    exit_code = train.remote()
    print(f"Training finished with exit code: {exit_code}")
