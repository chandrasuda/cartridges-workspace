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
# Clone with submodules (verl, cartridges, tokasaurus + data all in one)
git clone --recurse-submodules https://github.com/chandrasuda/cartridges-workspace.git
cd cartridges-workspace

# Install all packages
pip install -e verl/ -e cartridges/ -e tokasaurus/

# One-time Modal setup
pip install modal && modal setup
modal secret create huggingface-secret HF_TOKEN=hf_YOUR_TOKEN

# Deploy Tokasaurus inference server (keeps running, scales to zero when idle)
modal deploy scripts/modal_tokasaurus.py

# Run off-policy baseline (paper replication, ~9 hours, ~$30)
modal run --detach scripts/modal_offpolicy_train.py

# Run on-policy training (with Tokasaurus rollout, ~24 hours, ~$100)
modal run --detach scripts/modal_train.py

# After runs finish — download results and plot
modal volume get offpolicy-results eval_scores.json results/off_policy.json
modal volume get onpolicy-results onpolicy_eval.json results/on_policy.json
python scripts/plot_comparison.py --off results/off_policy.json --on results/on_policy.json --xaxis tokens
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

---

### Session 2 — Fair Comparison & Efficiency (March 2026)

The core scientific requirement: the off-policy and on-policy runs must differ **only in the algorithm** (who generates the training trajectories). All hyperparameters, initialization, data distribution, and loss formulation must match.

### 9. Loss Function Mismatch — Full-Vocab KL vs Top-k CE

**Problem:** The paper's off-policy training uses `top_k_logits=20` — it only stores/matches the top-20 teacher logprobs per token position. Our on-policy was computing full-vocabulary KL divergence (128K logits per position). This is a confounding variable: off-policy sees a weaker but cheaper signal (20 values), on-policy sees a richer but more expensive signal (128K values).

**Fix (`dp_actor.py`, `fsdp_workers.py`):**
- Teacher now returns `ref_topk_logprobs` (bs, resp_len, 20) and `ref_topk_ids` (bs, resp_len, 20) alongside the old single-token logprob
- Student forward extracts logprobs at teacher's top-20 positions via `log_softmax(logits).gather(-1, ref_topk_ids)`
- Loss: `CE = -Σ p_teacher(x) * log p_student(x)` over top-20 — exactly the paper's formula
- Falls back to single-token KL if top-k data not available (backward compat)

### 10. Cartridge Initialization Mismatch — Pre-trained vs Fresh

**Problem:** On-policy was loading `hazyresearch/cartridge-wauoq23f` (a pre-trained cartridge) while off-policy starts from `KVFromText(gradient.txt)`. This gave on-policy a massive head start — comparing an untrained cartridge vs a pre-trained one is not a fair algorithm comparison.

**Fix (`fsdp_workers.py`, `modal_train.py`):** Removed `checkpoint_path` from the on-policy config. Also fixed a bug in the KVFromText initialization path — it was constructing the initializer incorrectly (`KVFromText(max_tokens=N)` instead of `KVFromText(KVFromText.Config(max_tokens=N))`) and not moving the model to GPU before the forward pass needed to initialize the KV cache.

### 11. Training Prompt Distribution Mismatch

**Problem:** On-policy was training on the 92 LongHealth eval questions, while off-policy trains on 196K diverse synthesized QA conversations (structuring, summarization, creative, use_case, etc.). Training on the eval set is data contamination; training on a different distribution is a confound.

**Fix (`prepare_data_local.py`):** Rewrote data prep to extract only the user-turn prompts from the paper's 196K HF conversations (`hazyresearch/m07d11_longhealth_synthesize_llama-3.2-3b_p10_n65536-{0,1,2}`). Both methods now train on the same 110K unique questions, extracted from the same 3 HF shards. Only 112K of 196K were matched (43% were generic questions without patient names in the user turn; filtered to keep only patient-identified prompts).

### 12. Data Shuffling Breaks Patient Grouping

**Problem:** veRL's default is `data.shuffle=True` (RandomSampler). This randomizes row order regardless of how the parquet is sorted — our patient-sorted data would be shuffled into mixed-patient batches, breaking the patient-grouping optimization.

**Fix:** Added `data.shuffle=False` to the training config → uses `SequentialSampler` → reads rows in parquet order → patient-sorted data means consecutive 32-row windows are guaranteed same-patient.

**Verification:** 3441/3441 (100%) of 32-row windows are same-patient after sorting + truncating each patient to nearest multiple of 32 (dropped 210 rows = 0.2% of data).

### 13. Batch Size Mismatch

**Problem:** Off-policy uses `global_batch_size=32` (32 packed sequences per optimizer step). On-policy was using `train_batch_size=8`. This means off-policy computes gradients over 4× more samples per step — the gradient estimates are smoother and training is more stable.

**Fix:** Changed on-policy to `data.train_batch_size=32` and `ppo_mini_batch_size=32`. This is equivalent because veRL handles gradient accumulation internally.

### 14. Serial Teacher Forward — O(batch × doc_len) Tokens

**Problem:** With 32 samples per step and ~12K document tokens per patient, the teacher was running 32 independent forward passes of `[12K docs + ~500 tokens]` each = 400K tokens total. This dominated per-step wall time.

**Root cause:** Each sample needed different-length sequences so they couldn't be trivially batched (well, they could be padded, which the code now does). But the deeper insight: since all 32 samples are from the same patient (patient-grouped batches), the 12K document tokens are **identical across all samples**.

**Fix (`fsdp_workers.py` — prefix KV optimization):**
1. Detect shared patient: check if all 32 samples share the same `patient_id`
2. If yes, run one doc-only forward → `past_key_values` (the doc KV cache, shape: `[1, n_heads, 12K, head_dim]`)
3. Expand the doc KV to micro-batch size (`DynamicCache.expand()`) — read-only, no copy needed with `use_cache=False`
4. For each micro-batch of 4: forward only `[prompt + response]` (~500 tokens) with `past_key_values=doc_kv_expanded`
5. Position IDs start at `doc_len` to correctly account for the prefix

**Token count:** 12K (doc, once) + 32 × 500 (prompts+responses) = 28K total vs 32 × 12.5K = 400K. **~14× reduction.**

### 15. Serial Tokasaurus Rollout — One Request at a Time

**Problem:** `agent.num_workers=1` sent rollout requests to Tokasaurus sequentially: prompt 1 → wait → prompt 2 → wait → ... 32 serial HTTP calls × ~600ms each = ~20 seconds just for rollout dispatch.

**Fix:** Set `actor_rollout_ref.rollout.agent.num_workers=8`. veRL's `AsyncLLMServerManager` sends 8 concurrent requests via asyncio. Tokasaurus's continuous batching engine picks them up and processes them together on the GPU → effectively the same throughput as batched generation.

### 16. Serial Student Forward — Per-Sample Loop in Actor

**Problem:** `_forward_micro_batch_cartridge()` ran a separate forward pass for each sample in the micro-batch (loop of 32). Step 1 timings showed `timing_s/old_log_prob: 148s` and `timing_s/update_actor: 169s` — the actor was the dominant bottleneck, not the teacher.

**Root cause:** FlexLlamaForCausalLM uses `seq_ids` instead of `attention_mask`. The original code assigned `seq_id=0` to every sample and called `cache.clear()` between each — inherently serial.

**Key insight:** FlexAttention's block-diagonal attention mask already supports packing natively. Cache tokens use `seq_id=-1` (global, attended by all sequences). Real tokens only attend to tokens with the same `seq_id`. So assigning each sample a different `seq_id` (0, 1, 2, ..., 31) and concatenating into one packed sequence gives correct attention without padding.

**Fix (`dp_actor.py`):**
1. Extract valid tokens from each sample (remove padding)
2. Concatenate all samples into a single 1D packed sequence
3. Create `seq_ids = [0,0,...,0, 1,1,...,1, ..., 31,31,...,31]`
4. Position IDs restart from 0 for each sample
5. One `cache.clear()` + one forward pass through the entire packed sequence
6. Slice the output logits by sample offsets to extract per-sample log-probs

**Result:** 32 serial forward passes → 1 packed forward pass. Applies to both `compute_log_prob` (old_log_prob) and `update_policy` (training forward).

### 17. Git Large File Rejection

**Problem:** The raw HF shard parquets (`data/hf_shards/*/`) — each 92-113 MB — were accidentally staged and committed to git. GitHub rejected the push with `GH001: Large files detected (>100MB)`.

**Fix:** `git rm -r --cached data/hf_shards/` to unstage, then `git filter-branch` to rewrite the entire git history removing the large files from all previous commits, then `git push --force`. Added `data/hf_shards/` to `.gitignore`.

**Lesson:** Add large binary file patterns to `.gitignore` before creating them.

### 18. Workspace Clone Strategy — Three Separate Clones → One

**Problem:** Modal image build was cloning `HazyResearch/cartridges`, `chandrasuda/verl-cartridge`, and `chandrasuda/cartridges-workspace` separately. Three clone operations, three `pip install -e` steps, potential version drift between what's locally tested vs what Modal runs.

**Fix:** Since `cartridges-workspace` already defines `verl`, `cartridges`, and `tokasaurus` as git submodules, a single clone with `--recurse-submodules` gets everything in the exact same state as the local workspace:
```bash
git clone --recurse-submodules --depth 1 \
  https://github.com/chandrasuda/cartridges-workspace.git /opt/workspace
pip install -e /opt/workspace/cartridges \
         -e /opt/workspace/verl \
         -e /opt/workspace/tokasaurus
```
Data files (`data/on_policy/*.parquet`) come along for free.

### 19. Old Checkpoints Pollute Eval — 0/40 on Every Step

**Problem:** The results volume persists between Modal runs. Old 2048-token checkpoints from previous (broken) runs sat alongside new 512-token checkpoints. The eval evaluated all of them. All old checkpoints generated degenerate output (`" " " " " " " " " "`) → 0/40 accuracy on every step except the current run's step 2.

**Root cause:** (a) No cleanup of the checkpoint directory between runs. (b) `sorted(glob)` sorted alphabetically — "cache-step10" before "cache-step2" — so the eval order was scrambled. (c) The eval summary used wrong dict keys (`e["step"]` instead of `e["optimizer_step"]`).

**Fix (`modal_train.py`):**
1. Clean `/results/onpolicy/cartridge_checkpoints/` at the start of each run
2. Sort checkpoints numerically: `ckpts.sort(key=lambda p: int(re.search(r"step(\d+)", p).group(1)))`
3. Fix eval summary to use correct keys from the results dict

### 20. Wasted old_log_prob Forward Pass — 14s/step for Nothing

**Problem:** veRL's PPO loop calls `compute_log_prob()` to get `old_log_probs` before `update_policy()`. This is a full student forward pass (~14s). But with `ppo_epochs=1` and `ppo_mini_batch_size=batch_size`, `update_policy` detects `on_policy=True` and replaces old_log_probs with `log_prob.detach()` from the training forward pass. The separately computed old_log_probs are thrown away.

**Why it exists:** PPO with multiple epochs (`ppo_epochs>1`) needs the "old" policy's log probs as a stable reference for the importance sampling ratio `π_new(a|s) / π_old(a|s)`. With one epoch, old = new, so the ratio is always 1.0 and the clipping never activates.

**Fix (`ray_trainer.py`):** Detect cartridge distillation mode (`cartridge.enabled=True` + `ppo_epochs=1`). When true, inject zeros for `old_log_probs` and skip the forward pass entirely. `update_policy` overwrites them anyway.

---

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
