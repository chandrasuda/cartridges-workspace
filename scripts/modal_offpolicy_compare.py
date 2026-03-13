"""
Off-policy cartridge training on Modal — Uses OFFICIAL cartridges TrainDataset.

Uses the exact same code as HazyResearch paper:
- TrainDataset with packing_mode="truncate" and packed_seq_length=2048
- HF datasets with pre-computed teacher logprobs
- Proper sequence truncation to avoid Triton recompiles

Usage:
    modal run --detach scripts/modal_offpolicy_compare.py
"""

import modal

WORKSPACE_VERSION = "v72-use-official-traindataset"
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
    .run_commands(
        "git clone --depth 1 https://github.com/chandrasuda/cartridges-workspace.git /opt/workspace"
    )
    .pip_install(
        "requests", "transformers==4.53.0", "pandas", "pyarrow", "huggingface_hub",
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
def train(total_steps: int = 1000, batch_size: int = 4, lr: float = 0.02, eval_every: int = 50, save_every: int = 50):
    """
    Off-policy training using OFFICIAL cartridges TrainDataset.
    
    This uses the exact same infrastructure as the paper:
    - TrainDataset with packing_mode="truncate" 
    - packed_seq_length=2048 (all sequences same length, no recompiles!)
    - HF datasets with pre-computed teacher logprobs
    """
    import os
    import sys
    import json
    import time
    import gc
    import tempfile
    import re
    
    import numpy as np
    import requests
    import torch
    import torch.nn.functional as F
    import torch.optim as optim
    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader
    
    # Import cartridges modules
    sys.path.insert(0, "/opt/cartridges")
    from cartridges.cache import AttnConfig
    from cartridges.initialization import KVFromText
    from cartridges.models.llama.modeling_llama import FlexLlamaForCausalLM
    from cartridges.train import CacheAndModel
    from cartridges.generation import flex_generate
    from cartridges.datasets import TrainDataset, DataSource
    
    print("=" * 70)
    print("OFF-POLICY TRAINING (OFFICIAL TrainDataset)")
    print("=" * 70)
    print(f"  LR: {lr}")
    print(f"  batch_size: {batch_size}")
    print(f"  total_steps: {total_steps}")
    print(f"  packed_seq_length: 2048 (TRUNCATE mode)")
    print("=" * 70)
    
    device = "cuda"
    dtype = torch.bfloat16
    save_dir = "/results/offpolicy"
    os.makedirs(save_dir, exist_ok=True)
    
    # Load model
    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    flex_model = FlexLlamaForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(device).eval()
    for p in flex_model.parameters():
        p.requires_grad = False
    
    attn_config = AttnConfig(
        n_layers=flex_model.config.num_hidden_layers,
        n_heads=flex_model.config.num_key_value_heads,
        head_dim=flex_model.config.hidden_size // flex_model.config.num_attention_heads,
    )
    
    # Load LongHealth for cache init and eval
    print("Loading LongHealth data...")
    lh_data = requests.get(
        "https://raw.githubusercontent.com/kbressem/LongHealth/refs/heads/main/data/benchmark_v5.json"
    ).json()
    
    patient_doc_texts = {}
    for pid, patient in lh_data.items():
        doc_text = "\n\n".join(f"--- {did} ---\n{txt}" for did, txt in patient["texts"].items())
        patient_doc_texts[pid] = doc_text
    
    # Initialize cache from concatenated patient corpus (matches paper)
    TRAIN_PATIENT_IDS = [f"patient_{i:02d}" for i in range(1, 11)]
    full_corpus = "\n\n".join([patient_doc_texts[pid] for pid in sorted(TRAIN_PATIENT_IDS)])
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(full_corpus)
        init_file = f.name
    
    initializer = KVFromText.Config(max_tokens=512, text_source=init_file).instantiate()
    cache = initializer.initialize_kv_cache(tokenizer=tokenizer, model=flex_model, attn_config=attn_config).to(device)
    os.unlink(init_file)
    print(f"Cache initialized: {cache.num_cartridge_tokens()} tokens")
    
    wrapped_model = CacheAndModel(cache, flex_model)
    optimizer = optim.Adam(cache.parameters(), lr=lr)
    
    # =========================================================================
    # USE OFFICIAL TrainDataset WITH TRUNCATION!
    # This is the key fix - uses packing_mode="truncate" and packed_seq_length=2048
    # =========================================================================
    print("Loading TrainDataset with OFFICIAL cartridges infrastructure...")
    print("  - packing_mode: truncate")
    print("  - packed_seq_length: 2048")
    print("  - This ensures ALL sequences are exactly 2048 tokens (no recompiles!)")
    
    dataset_config = TrainDataset.Config(
        data_sources=[
            DataSource(path="hazyresearch/m07d11_longhealth_synthesize_llama-3.2-3b_p10_n65536-0", type="hf"),
        ],
        top_k_logits=20,
        packed_seq_length=2048,
        packing_mode="truncate",  # KEY: truncate long sequences!
    )
    dataset = dataset_config.instantiate(tokenizer=tokenizer, seed=42)
    print(f"Dataset loaded: {len(dataset)} batches")
    
    dataloader = DataLoader(
        dataset,
        batch_size=1,  # Dataset already handles batching
        collate_fn=lambda x: x[0],
        shuffle=True,
    )
    
    # Evaluation function
    def extract_answer(text):
        m = re.search(r"\b([A-E])\b", text.strip()[:20])
        return m.group(1) if m else "?"
    
    def load_eval_questions():
        questions = []
        EVAL_PATIENT_IDS = {f"patient_{i:02d}" for i in range(1, 11)}
        for pid, patient in lh_data.items():
            if pid not in EVAL_PATIENT_IDS:
                continue
            for q in patient["questions"]:
                options = "\n".join(L + ") " + q["answer_" + L.lower()] for L in "ABCDE")
                prompt = (
                    f"You are answering a multiple choice question about patient {patient['name']}.\n\n"
                    f"Question: {q['question']}\n\nOptions:\n{options}\n\n"
                    f"Answer with ONLY the letter (A, B, C, D, or E):"
                )
                answer_map = {q["answer_" + L.lower()]: L for L in "ABCDE"}
                questions.append({"prompt": prompt, "correct": answer_map.get(q["correct"], "?")})
        np.random.seed(42)
        np.random.shuffle(questions)
        return questions
    
    @torch.no_grad()
    def evaluate_cache(step):
        print(f"  [EVAL] Evaluating at step {step}...")
        eval_start = time.time()
        questions = load_eval_questions()
        correct = 0
        
        for q in questions:
            ids = tokenizer.encode(q["prompt"])
            input_ids = torch.tensor(ids, dtype=torch.long, device=device)
            seq_ids = torch.zeros_like(input_ids)
            position_ids = torch.arange(len(ids), dtype=torch.long, device=device)
            cache.clear()
            
            try:
                gen_output = flex_generate(
                    model=flex_model, tokenizer=tokenizer, cache=cache,
                    input_ids=input_ids, seq_ids=seq_ids, position_ids=position_ids,
                    max_new_tokens=10, temperature=0.0,
                )
                gen_text = tokenizer.decode(gen_output.get(0, []), skip_special_tokens=True)
                if extract_answer(gen_text) == q["correct"]:
                    correct += 1
            except:
                pass
        
        accuracy = 100 * correct / len(questions)
        print(f"  [EVAL] Step {step}: {correct}/{len(questions)} = {accuracy:.1f}% ({time.time()-eval_start:.1f}s)")
        return {"step": step, "accuracy": round(accuracy, 2), "correct": correct, "total": len(questions)}
    
    # Training loop
    eval_results = []
    token_history = []
    total_tokens = 0
    
    # Step 0 eval
    eval_result = evaluate_cache(0)
    eval_results.append(eval_result)
    token_history.append({"step": 0, "tokens": 0, "accuracy": eval_result["accuracy"]})
    
    # Save initial results
    results_path = os.path.join(save_dir, "offpolicy_results.json")
    with open(results_path, "w") as f:
        json.dump({"eval_results": eval_results, "token_history": token_history,
                   "config": {"mode": "off-policy", "lr": lr, "batch_size": batch_size}}, f, indent=2)
    
    print("=" * 70)
    print("TRAINING with official TrainDataset (truncate mode)")
    print("=" * 70)
    
    data_iter = iter(dataloader)
    for step in range(1, total_steps + 1):
        step_t0 = time.time()
        
        # Get batch from official dataset
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)
        
        # Count tokens (fixed 2048 per batch due to truncation)
        total_tokens += 2048 + 4000  # seq + doc context (fair comparison)
        
        cache.clear()
        outputs = wrapped_model(
            input_ids=batch.input_ids.to(device),
            seq_ids=batch.element_ids.to(device),
            position_ids=batch.position_ids.to(device),
        )
        
        # Loss using pre-computed teacher logprobs
        topk_pred_logprobs = F.log_softmax(outputs.logits.float(), dim=-1)[
            0,
            batch.topk_token_idxs.to(device) - 1,
            batch.topk_token_ids.to(device),
        ]
        ce_by_token = -batch.topk_logprobs.to(device).float().exp() * topk_pred_logprobs
        loss = ce_by_token.mean()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        elapsed = time.time() - step_t0
        print(f"Step {step}: loss={loss.item():.4f} | time={elapsed:.1f}s | tokens={total_tokens:,}")
        
        # Evaluate
        if step % eval_every == 0 or step == total_steps:
            eval_result = evaluate_cache(step)
            eval_results.append(eval_result)
            token_history.append({"step": step, "tokens": total_tokens, "accuracy": eval_result["accuracy"]})
            
            with open(results_path, "w") as f:
                json.dump({"eval_results": eval_results, "token_history": token_history,
                           "config": {"mode": "off-policy", "lr": lr, "batch_size": batch_size}}, f, indent=2)
        
        # Save checkpoint
        if step % save_every == 0 or step == total_steps:
            ckpt_path = os.path.join(save_dir, f"step-{step}", "cartridge.pt")
            os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
            cache.save(ckpt_path)
        
        gc.collect()
        torch.cuda.empty_cache()
    
    print("=" * 70)
    print("OFF-POLICY TRAINING COMPLETE")
    print(f"Results saved to: {results_path}")
    print("=" * 70)
    
    results_volume.commit()
    return 0


@app.local_entrypoint()
def main(
    total_steps: int = 1000,
    batch_size: int = 4,
    lr: float = 0.02,
    eval_every: int = 50,
    save_every: int = 50,
):
    print("=" * 70)
    print("OFF-POLICY TRAINING (OFFICIAL TrainDataset)")
    print("=" * 70)
    print(f"  lr: {lr}")
    print(f"  total_steps: {total_steps}")
    print(f"  Using cartridges TrainDataset with packing_mode='truncate'")
    print("=" * 70)
    
    exit_code = train.remote(
        total_steps=total_steps,
        batch_size=batch_size,
        lr=lr,
        eval_every=eval_every,
        save_every=save_every,
    )
    print(f"Training finished with exit code: {exit_code}")
