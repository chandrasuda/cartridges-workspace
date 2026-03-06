# Cartridges Workspace

On-policy cartridge training: **veRL** + **Tokasaurus** + **Cartridges**.

## Repos

| Directory | Repo | Branch | What |
|-----------|------|--------|------|
| `verl/` | [chandrasuda/verl-cartridge](https://github.com/chandrasuda/verl-cartridge) | main | veRL fork with cartridge training support |
| `cartridges/` | [HazyResearch/cartridges](https://github.com/HazyResearch/cartridges) | main | Trainable KV cache library |
| `tokasaurus/` | [chandrasuda/tokasaurus](https://github.com/chandrasuda/tokasaurus) | geoff/cartridges | Inference server with cartridge injection |

## Quick Start

```bash
# Clone with submodules
git clone --recurse-submodules https://github.com/chandrasuda/cartridges-workspace.git
cd cartridges-workspace

# Install
pip install -e verl/ -e cartridges/

# Deploy Tokasaurus on Modal
pip install modal && modal setup
modal secret create huggingface-secret HF_TOKEN=hf_YOUR_TOKEN
modal deploy verl/verl/workers/rollout/tokasaurus_rollout/modal_tokasaurus.py

# Test the rollout
python verl/verl/workers/rollout/tokasaurus_rollout/example_query.py

# Prepare data + train (on Modal A100-80GB)
modal run verl/examples/cartridge_distill/modal_train.py

# Evaluate
python verl/examples/cartridge_distill/eval_longhealth.py
```

## What was changed in veRL

| File | Change |
|------|--------|
| `verl/workers/config/actor.py` | Added `CartridgeConfig` (enabled, checkpoint_path, lr, etc.) |
| `verl/workers/fsdp_workers.py` | Load `FlexLlamaForCausalLM` + `TrainableCache`, freeze model, optimizer on cache only, teacher with documents |
| `verl/workers/actor/dp_actor.py` | Cartridge forward pass (`seq_ids` instead of `attention_mask`) |
| `verl/workers/rollout/replica.py` | Register `"tokasaurus"` in `RolloutReplicaRegistry` |
| `verl/workers/rollout/base.py` | Register in `_ROLLOUT_REGISTRY` |
| `verl/workers/rollout/tokasaurus_rollout/` | Created `TokasaurusHttpServer`, `TokasaurusReplica`, `ServerAdapter`, Modal deploy |
| `verl/trainer/ppo/ray_trainer.py` | Cartridge sync after each step |
| `verl/experimental/agent_loop/agent_loop.py` | Prompt truncation fix |
| `verl/examples/cartridge_distill/` | Created training config, data prep, eval, Modal train script |

## Results

| Metric | Value |
|--------|-------|
| KL loss (start → end) | 5.46 → 3.24 (41% reduction) |
| Training time | 53 min (40 steps, A100-80GB) |
| Trainable params | 2048 tokens × 28 layers × 2 (keys + values) × 8 KV heads × 128 head_dim × bfloat16 (117 million parameters) / 3.33B (3.5%) |

## On-Policy Training: Issues & Fixes Log

Detailed log of roadblocks encountered during on-policy training setup and how each was solved.

### 1. Data Preparation — Off-Policy vs On-Policy Format

**Problem:** The raw HuggingFace shards (`data/hf_shards/shard_*/`) contain the full off-policy format: `messages` (user + assistant turns with pre-baked `token_ids` and `top_logprobs`), `system_prompt`, `metadata`, `type`. veRL's on-policy loop needs a completely different format — just the user question as a prompt, since the student generates its own answers.

**Why:** Off-policy training uses pre-computed teacher logprobs baked into the data. On-policy training runs the teacher live on the student's own rollouts, so the assistant responses and teacher logprobs in the raw data are useless.

**Fix:** Created `prepare_data_local.py` that reads from local parquet files (no HF download), extracts only the user question, maps patient names → IDs, and outputs veRL-format parquets to `processed_data/`. Output schema:
```json
{"prompt": [{"role": "user", "content": "..."}], "data_source": "longhealth_synthesized", "patient_id": "patient_09", "extra_info": {"patient_id": "patient_09"}, "reward_model": {"ground_truth": "", "style": "rule"}}
```

### 2. Patient ID Matching — 43% Failure Rate

**Problem:** Initial `prepare_data.py` matched patient names by searching only the user question text. But ~43% of questions are generic ("What are the treatment options for hepatocellular carcinoma...") and don't mention the patient by name — the name only appears in `system_prompt`.

**Fix:** Fall back to `system_prompt` when user question match fails. Result: 0% unknown, all 196,608 rows matched.

### 3. Patient ID Dropped by veRL Pipeline

**Problem:** Even with `patient_id` in the parquet, the teacher saw it for only 23/32 samples. Root cause in `ray_trainer.py` line 478:
```python
reward_keys = set({"data_source", "reward_model", "extra_info", "uid"}) & batch.non_tensor_batch.keys()
```
`patient_id` is not in the preserved set, so `_get_gen_batch()` strips it from the batch before it reaches the teacher.

**Fix:** Stash `patient_id` inside `extra_info` dict (which veRL preserves through the entire pipeline), and update the teacher to check three locations: `non_tensor_batch["patient_id"]` → `non_tensor_batch["extra_info"]["patient_id"]` → decoded text name matching fallback.

### 4. Tokasaurus Scaling Down Between Steps

**Problem:** Step 1 took ~7 minutes. Tokasaurus Modal deployment had `scaledown_window=300` (5 min), so it scaled to zero mid-step during the teacher forward pass. Step 2 rollout requests then hit a cold server with repeated timeouts.

**Fix:** Increased `scaledown_window` to 1800 (30 min) in `modal_tokasaurus.py`.

### 5. Teacher Forward Pass — Sequential Bottleneck

**Problem:** `_compute_ref_with_documents()` ran 32 separate forward passes in a sequential for-loop, each with a different-length document (~7-9K tokens) prepended. This took ~3-4 minutes per step.

**Fix:** Batched teacher forward with micro-batches of 4 samples. Left-pad sequences to max length within each micro-batch, run a single forward pass, then extract per-sample logprobs from the response region. Also pre-tokenize all patient documents once at startup instead of re-tokenizing every step.

**Result:** Teacher now runs in ~18s (down from ~3-4 min), ~10x speedup.

### 6. Teacher Batched Forward — OOM at Micro-Batch 8

**Problem:** Initial `TEACHER_MICRO_BATCH=8` tried to allocate 23.67 GiB but only 18.38 GiB was free (model + actor + ref model already using 61GB of the A100-80GB).

**Fix:** Reduced to `TEACHER_MICRO_BATCH=4`. 8 sequences × ~9K tokens was too much; 4 fits comfortably.

### 7. Triton AutoTune Overhead on Step 1

**Problem:** Every fresh run spends ~5-8 minutes on step 1 for Triton kernel autotuning. FlexAttention encounters different sequence lengths and compiles optimized kernels for each. This is unavoidable on first run but does not repeat for subsequent steps.

**Status:** Accepted cost. Step 1 = ~6-7 min, step 2+ should be significantly faster.

### 8. Not Even One Epoch

**Clarification:** 196,608 prompts ÷ 32 batch size = 6,144 steps per epoch. We run 300 steps = ~5% of one epoch. `total_epochs=100` is set high as a no-op ceiling; `total_training_steps=300` is the real limit.

### Current Run Configuration

```
Model:           meta-llama/Llama-3.2-3B-Instruct
GPU:             A100-80GB (Modal)
Trainable:       117M params (cartridge KV cache only, 3.5% of model)
Batch size:      32
Prompt length:   512 tokens max
Response length: 512 tokens max
Steps:           300
Cartridge save:  every 10 steps
Teacher:         batched (micro-batch=4), patient_id from extra_info
Rollout:         Tokasaurus (A100-40GB, Modal, 30min scaledown)
```

### Version History (Modal cache busts)

| Version | Change |
|---------|--------|
| v5 | Initial working on-policy training |
| v6 | Fix teacher patient matching (use patient_id from batch) |
| v7 | Batched teacher forward (8x micro-batch) |
| v8 | Reduce teacher micro-batch to 4 (OOM fix) |
| v9 | patient_id in extra_info, 3-strategy teacher lookup, pre-tokenize docs |
| v10 | Add raw_prompt matching strategy, decode all tokens fallback, debug logging |

### Parallel Eval Pipeline

Training and eval run as separate Modal apps in parallel:

```bash
# Training (runs on A100-80GB, saves checkpoints every 10 steps)
modal run verl/examples/cartridge_distill/modal_train.py

# Eval (runs on A10G, polls for new checkpoints every 5 min)
modal run verl/examples/cartridge_distill/eval_checkpoints.py --poll
```

Eval loads FlexLlama + each cartridge checkpoint directly (no Tokasaurus needed), runs 40 LongHealth questions, and saves results to `/results/onpolicy_eval.json` on the shared volume.

## Detailed write-up

See [on-policy-cartridge-training/WRITEUP.md](https://github.com/chandrasuda/on-policy-cartridge-training/blob/main/WRITEUP.md) for the full technical narrative (33 issues encountered and solved).
