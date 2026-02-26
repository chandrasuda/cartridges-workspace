# Cartridges Workspace

On-policy cartridge distillation: **veRL** + **Tokasaurus** + **Cartridges**.

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
| `verl/workers/rollout/tokasaurus_rollout/` | NEW: `TokasaurusHttpServer`, `TokasaurusReplica`, `ServerAdapter`, Modal deploy |
| `verl/trainer/ppo/ray_trainer.py` | Cartridge sync after each step |
| `verl/experimental/agent_loop/agent_loop.py` | Prompt truncation fix |
| `verl/examples/cartridge_distill/` | Created training config, data prep, eval, Modal train script |

## Results

| Metric | Value |
|--------|-------|
| KL loss (start → end) | 5.46 → 3.24 (41% reduction) |
| Baseline accuracy | 30.0% |
| Offline cartridge accuracy | 47.5% |
| Training time | 53 min (40 steps, A100-80GB) |
| Trainable params | 117M / 3.33B (3.5%) |
