"""
Modal training instance for on-policy cartridge distillation.

Runs veRL training on an A100-80GB GPU with the cartridge patches applied.
Tokasaurus inference runs on a separate Modal deployment (already running).

Usage:
    modal run scripts/modal_train.py
"""

import modal

TOKASAURUS_URL = "https://kiran1234c--tokasaurus-cartridge-server-serve.modal.run"
GPU = "A100-80GB"

# Build image: clone cartridges-workspace with all submodules (verl, cartridges, tokasaurus)
# Cache bust: bump to force re-clone when any repo changes
WORKSPACE_VERSION = "v26-fix-eval-cuda-visibility"

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
        # Pre-built flash-attn wheel — avoids compiling from source
        "pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl",
        "pip install flashinfer-python==0.2.0.post2 --extra-index-url https://flashinfer.ai/whl/cu124/torch2.6/",
    )
    # Clone workspace + all submodules in one shot
    .run_commands(f"echo '{WORKSPACE_VERSION}'")
    .run_commands(
        "git clone --recurse-submodules --depth 1 "
        "https://github.com/chandrasuda/cartridges-workspace.git /opt/workspace"
    )
    # Install all three packages in editable mode
    .run_commands(
        "pip install -e /opt/workspace/cartridges "
        "&& pip install -e /opt/workspace/verl "
        "&& pip install -e /opt/workspace/tokasaurus"
    )
    .pip_install("requests")
    .pip_install(
        "transformers==4.53.0",
        "ray[default]",
        "omegaconf",
        "hydra-core",
        "pandas",
        "pyarrow",
        "aiohttp",
        "codetiming",
        "torchdata",
        "peft",
        "cachetools",
    )
)

results_volume = modal.Volume.from_name("onpolicy-results", create_if_missing=True)
app = modal.App("onpolicy-training", image=image)


@app.function(
    gpu=GPU,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=86400,  # 24 hours
    min_containers=0,
    max_containers=1,
    scaledown_window=600,
    volumes={"/results": results_volume},
)
def train():
    """Run cartridge distillation training."""
    import subprocess
    import os
    import sys
    import shutil

    # Clean old checkpoints from previous runs to avoid eval pollution
    ckpt_dir = "/results/onpolicy/cartridge_checkpoints"
    try:
        if os.path.exists(ckpt_dir):
            old_files = [f for f in os.listdir(ckpt_dir) if f.endswith(".pt")]
            if old_files:
                print(f"⚠ Cleaning {len(old_files)} old checkpoints from previous run")
                for f in old_files:
                    os.remove(os.path.join(ckpt_dir, f))
                results_volume.commit()
                print(f"✓ Cleaned checkpoint directory")
    except Exception as e:
        print(f"⚠ Cleanup failed (non-fatal): {e}")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Verify GPU
    try:
        import torch
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    except Exception as e:
        print(f"GPU check failed: {e}")

    # Verify our fork is installed correctly
    from verl.workers.config.actor import CartridgeConfig
    print(f"✓ veRL fork installed with CartridgeConfig")

    # Data shipped with the workspace repo
    train_parquet = "/opt/workspace/data/on_policy/train.parquet"
    val_parquet   = "/opt/workspace/data/on_policy/val.parquet"
    assert os.path.exists(train_parquet), f"Missing {train_parquet}"
    assert os.path.exists(val_parquet), f"Missing {val_parquet}"
    import pandas as pd
    train_df = pd.read_parquet(train_parquet)
    print(f"✓ Training data: {len(train_df):,} rows, {train_df.patient_id.nunique()} patients — {train_parquet}")
    print(f"  Patient counts: {dict(train_df.patient_id.value_counts())}")
    print(f"✓ Validation data: {val_parquet}")

    # Pre-download LongHealth data for the teacher
    import requests as req
    data = req.get('https://raw.githubusercontent.com/kbressem/LongHealth/refs/heads/main/data/benchmark_v5.json').json()
    import json
    with open('/tmp/longhealth_data.json', 'w') as f:
        json.dump(data, f)
    print(f'Saved LongHealth data ({len(data)} patients) to /tmp/longhealth_data.json')

    # Create dummy reward function (cartridge uses KL loss, not rewards)
    os.makedirs("/tmp/reward", exist_ok=True)
    with open("/tmp/reward/dummy_reward.py", "w") as f:
        f.write("def compute_score(data_source, solution_str, ground_truth, extra_info=None):\n    return 0.0\n")

    # Run training — unbuffered so every log line streams immediately
    env = os.environ.copy()
    env["TOKASAURUS_URL"] = TOKASAURUS_URL
    env["PYTHONUNBUFFERED"] = "1"

    cmd = [
        sys.executable, "-m", "verl.trainer.main_ppo",
        "algorithm.adv_estimator=grpo",
        #
        f"data.train_files={train_parquet}",
        f"data.val_files={val_parquet}",
        "data.train_batch_size=32",
        "data.max_prompt_length=512",
        "data.max_response_length=512",
        "data.filter_overlong_prompts=True",
        "data.truncation=right",
        "data.shuffle=False",  # keep patient-sorted order for patient-grouped batches
        #
        "actor_rollout_ref.model.path=meta-llama/Llama-3.2-3B-Instruct",
        "actor_rollout_ref.model.external_lib=cartridges",
        "actor_rollout_ref.model.enable_gradient_checkpointing=True",
        #
        "actor_rollout_ref.actor.strategy=fsdp",
        "actor_rollout_ref.actor.ppo_mini_batch_size=32",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8",
        "actor_rollout_ref.actor.use_kl_loss=True",
        "actor_rollout_ref.actor.kl_loss_coef=1.0",
        "actor_rollout_ref.actor.kl_loss_type=low_var_kl",
        "actor_rollout_ref.actor.ppo_epochs=1",
        "actor_rollout_ref.actor.fsdp_config.param_offload=False",
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload=False",
        "actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16",
        "actor_rollout_ref.actor.fsdp_config.use_orig_params=True",
        "actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16",
        "+actor_rollout_ref.actor.cartridge.enabled=True",
        # No checkpoint_path → KVFromText init (same as off-policy baseline)
        "+actor_rollout_ref.actor.cartridge.num_tokens=512",
        "+actor_rollout_ref.actor.cartridge.num_frozen_tokens=1",
        "+actor_rollout_ref.actor.cartridge.lr=0.02",
        #
        "actor_rollout_ref.rollout.name=tokasaurus",
        "actor_rollout_ref.rollout.temperature=0.7",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4",
        "actor_rollout_ref.rollout.agent.num_workers=32",  # all 32 requests concurrent → Tokasaurus batches them
        f"+actor_rollout_ref.rollout.custom.tokasaurus_url={TOKASAURUS_URL}",
        # Cartridge synced from actor after each step (no pre-loaded cartridge)
        "+actor_rollout_ref.rollout.custom.cartridges=[]",
        #
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4",
        "actor_rollout_ref.ref.fsdp_config.param_offload=True",
        #
        "algorithm.use_kl_in_reward=False",
        # Dummy reward (cartridge distillation uses KL loss, not reward-based RL)
        "reward.custom_reward_function.path=/tmp/reward/dummy_reward.py",
        "reward.custom_reward_function.name=compute_score",
        #
        "trainer.critic_warmup=0",
        'trainer.logger=["console"]',
        "trainer.project_name=cartridge_distill",
        "trainer.experiment_name=longhealth_llama3b_onpolicy",
        "trainer.n_gpus_per_node=1",
        "trainer.nnodes=1",
        "trainer.save_freq=-1",  # Disable full-model checkpointing (CacheAndModel missing .config)
        "trainer.test_freq=-1",  # Disable reward-based test (dummy reward = useless)
        "+trainer.cartridge_save_freq=50",  # Save cache .pt every 50 steps for eval
        "trainer.default_local_dir=/results/onpolicy",
        "trainer.total_epochs=100",  # High ceiling
        "trainer.total_training_steps=500",  # ~500 steps ≈ 37 hours ≈ $140 on A100
        "trainer.val_before_train=False",
    ]

    print(f"\n{'='*60}")
    print("ON-POLICY CARTRIDGE DISTILLATION")
    print(f"  Tokasaurus URL : {TOKASAURUS_URL}")
    print(f"  GPU            : {GPU}")
    print(f"  Cartridge size : 512 tokens")
    print(f"  Batch size     : 32 (patient-grouped, same patient per batch)")
    print(f"  Loss           : top-k CE (k=20), teacher prefix KV opt")
    print(f"  Rollout        : 8 concurrent workers")
    print(f"  Data           : {len(train_df):,} prompts, {train_df.patient_id.nunique()} patients")
    print(f"  Save every     : 10 steps  |  Eval every: 10 steps")
    print(f"{'='*60}\n")

    import time as _time
    train_start = _time.time()
    print(f"[{_time.strftime('%H:%M:%S')}] Training subprocess starting...")
    result = subprocess.run(cmd, env=env, stdout=sys.stdout, stderr=sys.stderr)
    
    train_elapsed = _time.time() - train_start
    print(f"\n[{_time.strftime('%H:%M:%S')}] Training took {train_elapsed/60:.1f} min ({train_elapsed:.0f}s)")

    if result.returncode != 0:
        print(f"Training failed with exit code {result.returncode}")
        return result.returncode

    # Save the trained cartridge to persistent volume
    print("\n" + "=" * 60)
    print("Training complete. Saving results to volume...")
    print("=" * 60)

    # Copy all cartridge checkpoints to volume
    import shutil, glob, re as _re
    ckpt_dir = "/results/onpolicy/cartridge_checkpoints"
    if os.path.exists(ckpt_dir):
        ckpts = glob.glob(os.path.join(ckpt_dir, "cache-step*.pt"))
        # Sort numerically by step number, not alphabetically
        ckpts.sort(key=lambda p: int(_re.search(r"step(\d+)", p).group(1)))
        print(f"✓ Found {len(ckpts)} cartridge checkpoints")
        for c in ckpts:
            print(f"  {os.path.basename(c)}")
    else:
        print(f"⚠ No checkpoint dir at {ckpt_dir}")

    # Also copy the latest synced cartridge
    src = "/tmp/verl_cartridge_latest.pt"
    dst = "/results/onpolicy/on_policy_cartridge_final.pt"
    if os.path.exists(src):
        shutil.copy2(src, dst)
        print(f"✓ Saved final cartridge: {dst}")

    results_volume.commit()

    # ------------------------------------------------------------------
    # 5. Evaluate all saved checkpoints on LongHealth
    # ------------------------------------------------------------------
    _eval_all_checkpoints(ckpt_dir)
    results_volume.commit()

    return result.returncode


def _eval_all_checkpoints(ckpt_dir: str):
    """Evaluate all cartridge checkpoints on LongHealth after training.

    Matches the working pattern from quick_eval.py:
    - Baseline uses standard AutoModelForCausalLM (FlexLlama doesn't support .generate())
    - Cartridge eval uses FlexLlamaForCausalLM + flex_generate with 1D input_ids, seq_ids, position_ids
    - flex_generate returns dict {seq_id: [token_ids]}, not a tensor
    - Manual cache reconstruction to work around from_pretrained bug with num_frozen_tokens
    """
    import glob
    import json
    import os
    import re
    import time
    import requests as http_req

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from cartridges.models.llama.modeling_llama import FlexLlamaForCausalLM
    from cartridges.cache import TrainableCache, AttnConfig
    from cartridges.generation import flex_generate

    NUM_EVAL = 40
    MODEL = "meta-llama/Llama-3.2-3B-Instruct"

    ckpts = glob.glob(os.path.join(ckpt_dir, "cache-step*.pt"))
    # Sort numerically by step number (cache-step2 before cache-step10)
    ckpts.sort(key=lambda p: int(re.search(r"step(\d+)", p).group(1)))
    if not ckpts:
        print("[eval] No checkpoints found, skipping eval")
        return

    # Log checkpoint sizes to confirm they're clean (~224 MB each)
    print(f"\n{'='*60}")
    print(f"EVALUATING {len(ckpts)} CHECKPOINTS ON LONGHEALTH")
    print(f"{'='*60}")
    for c in ckpts:
        sz_mb = os.path.getsize(c) / 1e6
        print(f"  {os.path.basename(c):>30s}  {sz_mb:.1f} MB")
    print()

    # Load LongHealth questions
    data = http_req.get(
        "https://raw.githubusercontent.com/kbressem/LongHealth/refs/heads/main/data/benchmark_v5.json"
    ).json()

    questions = []
    for pid, patient in data.items():
        if int(pid.split("_")[1]) > 8:
            continue
        for q in patient["questions"]:
            options = "\n".join(f"{L}) {q[f'answer_{L.lower()}']}" for L in "ABCDE")
            prompt = (
                f"You are answering a multiple choice question about patient {patient['name']}.\n\n"
                f"Question: {q['question']}\n\nOptions:\n{options}\n\n"
                f"Answer with ONLY the letter (A, B, C, D, or E):"
            )
            answer_map = {q[f"answer_{L.lower()}"]: L for L in "ABCDE"}
            questions.append({
                "prompt": prompt,
                "correct": answer_map.get(q["correct"], "?"),
            })
    eval_qs = questions[:NUM_EVAL]
    print(f"Eval questions: {len(eval_qs)}")

    def extract_answer(text):
        m = re.search(r'\b([A-E])\b', text.strip()[:20])
        return m.group(1) if m else "?"

    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    # ---- Baseline eval (no cartridge) — uses standard Llama, NOT FlexLlama ----
    print(f"\n--- Baseline (no cartridge) ---")
    print(f"  Loading standard AutoModelForCausalLM...")
    base_model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).cuda().eval()
    correct_bl = 0
    for i, q in enumerate(eval_qs):
        ids = tokenizer.encode(q["prompt"], return_tensors="pt").cuda()
        with torch.no_grad():
            out = base_model.generate(ids, max_new_tokens=10, do_sample=False)
        gen_text = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        pred = extract_answer(gen_text)
        if i < 3:
            print(f"    Q{i}: gen='{gen_text[:60]}' pred={pred} correct={q['correct']}")
        if pred == q["correct"]:
            correct_bl += 1
    baseline_acc = correct_bl / len(eval_qs) * 100
    print(f"  Baseline: {correct_bl}/{len(eval_qs)} ({baseline_acc:.1f}%)")
    del base_model
    torch.cuda.empty_cache()

    # ---- Load FlexLlama for cartridge eval ----
    print(f"\n  Loading FlexLlamaForCausalLM for cartridge eval...")
    flex_model = FlexLlamaForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).cuda().eval()
    print(f"  ✓ FlexLlama loaded")

    # Format matches off_policy.json so plot_comparison.py can read both
    results = {
        "method": "on_policy",
        "config": {"cartridge_tokens": 512, "model": "Llama-3.2-3B", "loss": "topk_ce_k20"},
        "baseline": {"correct": correct_bl, "total": len(eval_qs), "accuracy": baseline_acc},
        "evals": [],
    }

    for ckpt_path in ckpts:
        m_step = re.search(r"cache-step(\d+)\.pt", ckpt_path)
        if not m_step:
            continue
        step = int(m_step.group(1))

        print(f"\n--- Step {step} ---")
        try:
            # Fix legacy key names
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            if "fixed_keys" in ckpt and "frozen_keys" not in ckpt:
                ckpt["frozen_keys"] = ckpt.pop("fixed_keys")
                ckpt["frozen_values"] = ckpt.pop("fixed_values")
                torch.save(ckpt, ckpt_path)
                ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

            # Manual cache reconstruction (works around from_pretrained bug with num_frozen_tokens)
            trainable_keys = ckpt["trainable_keys"]
            frozen_keys = ckpt["frozen_keys"]
            n_layers = len(trainable_keys)
            n_heads = trainable_keys[0].size(1)
            head_dim = trainable_keys[0].size(3)
            num_frozen = frozen_keys[0].size(2) if frozen_keys else 0
            num_trainable = trainable_keys[0].size(2)
            print(f"  Cache: {n_layers} layers, {n_heads} heads, {num_trainable} trainable + {num_frozen} frozen tokens")

            init_keys = [
                torch.cat([frozen_keys[i], trainable_keys[i]], dim=2).contiguous()
                if num_frozen > 0 else trainable_keys[i]
                for i in range(n_layers)
            ]
            init_values = [
                torch.cat([ckpt["frozen_values"][i], ckpt["trainable_values"][i]], dim=2).contiguous()
                if num_frozen > 0 else ckpt["trainable_values"][i]
                for i in range(n_layers)
            ]
            cache = TrainableCache(
                config=AttnConfig(n_layers=n_layers, n_heads=n_heads, head_dim=head_dim),
                init_keys=init_keys, init_values=init_values,
                num_frozen_tokens=num_frozen,
            ).cuda()
            del ckpt

            correct = 0
            for qi, q in enumerate(eval_qs):
                ids = tokenizer.encode(q["prompt"])
                input_ids = torch.tensor(ids, dtype=torch.long, device="cuda")  # 1D
                seq_ids = torch.zeros_like(input_ids)
                position_ids = torch.arange(len(ids), dtype=torch.long, device="cuda")
                cache.clear()
                gen_output = flex_generate(
                    model=flex_model, tokenizer=tokenizer, cache=cache,
                    input_ids=input_ids, seq_ids=seq_ids, position_ids=position_ids,
                    max_new_tokens=10, temperature=0.0,
                )
                gen_tokens = gen_output.get(0, [])
                gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
                pred = extract_answer(gen_text)
                if qi < 3:
                    print(f"    Q{qi}: gen='{gen_text[:80]}' pred={pred} correct={q['correct']}")
                if pred == q["correct"]:
                    correct += 1

            acc = correct / len(eval_qs) * 100
            # total_tokens: step × batch_size × avg_response_len (~350 tokens)
            total_tokens = step * 32 * 350
            print(f"  Step {step} ({total_tokens:,} tokens): {correct}/{len(eval_qs)} ({acc:.1f}%)")
            results["evals"].append({
                "optimizer_step": step,
                "total_tokens": total_tokens,
                "scores": {"score": round(acc / 100, 4)},
                "num_eval_questions": len(eval_qs),
                "correct": correct,
            })
            # Save incrementally after each checkpoint so results persist if interrupted
            import time as _t
            with open("/results/onpolicy_eval.json", "w") as _f:
                json.dump(results, _f, indent=2)
            print(f"  ✓ Saved incrementally to /results/onpolicy_eval.json")

            del cache
            torch.cuda.empty_cache()
        except Exception as e:
            import traceback
            print(f"  ✗ Step {step} failed: {e}")
            traceback.print_exc()

    del flex_model
    torch.cuda.empty_cache()

    # Save results
    eval_path = "/results/onpolicy_eval.json"
    with open(eval_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Saved eval results to {eval_path}")

    # Print summary
    print(f"\n{'='*60}")
    print("EVAL SUMMARY")
    print(f"{'='*60}")
    print(f"  Baseline:  {baseline_acc:.1f}%")
    for e in sorted(results["evals"], key=lambda x: x["optimizer_step"]):
        step = e["optimizer_step"]
        acc = e["correct"] / e["num_eval_questions"] * 100
        print(f"  Step {step:>4}:  {acc:.1f}%  ({e['correct']}/{e['num_eval_questions']})")
    print(f"{'='*60}")


@app.local_entrypoint()
def main():
    exit_code = train.remote()
    print(f"\nTraining finished with exit code: {exit_code}")
