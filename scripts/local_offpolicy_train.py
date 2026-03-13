#!/usr/bin/env python3
"""
Off-policy cartridge training — Uses PRE-COMPUTED teacher logprobs from HF shards.

The HazyResearch team already synthesized ~196K QA conversations with teacher logprobs.
This script just loads that data and trains - NO synthesis needed!

Data source: hazyresearch/m07d11_longhealth_synthesize_llama-3.2-3b_p10_n65536-{0,1,2}
"""

import argparse
import gc
import json
import os
import re
import tempfile
import time
import logging
from pathlib import Path

# =============================================================================
# LOGGING SETUP
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("local_offpolicy_train")
logger.setLevel(logging.DEBUG)

# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================
SCRIPT_DIR = Path(__file__).parent.resolve()
WORKSPACE_DIR = SCRIPT_DIR.parent
CARTRIDGES_DIR = WORKSPACE_DIR / "cartridges"

os.environ["CARTRIDGES_DIR"] = str(CARTRIDGES_DIR)
os.environ["CARTRIDGES_OUTPUT_DIR"] = str(WORKSPACE_DIR / "local_checkpoints")

logger.info("=" * 70)
logger.info("OFF-POLICY CARTRIDGE TRAINING (PRE-COMPUTED LOGPROBS)")
logger.info("=" * 70)

import numpy as np
import pandas as pd
import requests
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

import sys
sys.path.insert(0, str(CARTRIDGES_DIR))

from cartridges.cache import AttnConfig, TrainableCache
from cartridges.initialization import KVFromText
from cartridges.models.llama.modeling_llama import FlexLlamaForCausalLM
from cartridges.train import CacheAndModel
from cartridges.generation import flex_generate
from cartridges.structs import Conversation, read_conversations
from cartridges.datasets import TrainDataset, DataSource

logger.info("All modules imported successfully!")


def cleanup_memory(device: str):
    """Force garbage collection and clear device memory."""
    gc.collect()
    if device == "mps":
        torch.mps.synchronize()
        torch.mps.empty_cache()
    elif device.startswith("cuda"):
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def extract_answer(text: str) -> str:
    """Extract answer letter (A-E) from generated text."""
    m = re.search(r"\b([A-E])\b", text.strip()[:20])
    return m.group(1) if m else "?"


def load_eval_questions(lh_data: dict, eval_patient_ids: set) -> list:
    """Load evaluation questions for specified patient IDs (MATCHES on-policy exactly)."""
    questions = []
    for pid, patient in lh_data.items():
        if pid not in eval_patient_ids:
            continue
        for q in patient["questions"]:
            options = "\n".join(L + ") " + q["answer_" + L.lower()] for L in "ABCDE")
            prompt = (
                f"You are answering a multiple choice question about patient {patient['name']}.\n\n"
                f"Question: {q['question']}\n\nOptions:\n{options}\n\n"
                f"Answer with ONLY the letter (A, B, C, D, or E):"
            )
            answer_map = {q["answer_" + L.lower()]: L for L in "ABCDE"}
            questions.append({
                "prompt": prompt,
                "correct": answer_map.get(q["correct"], "?"),
                "patient_id": pid,
            })
    return questions


def evaluate_cache(flex_model, tokenizer, cache, eval_questions, device, step, max_eval_samples=None):
    """Evaluate cache on LongHealth questions (MATCHES on-policy evaluation exactly)."""
    logger.info(f"  [EVAL] Evaluating at step {step}...")
    eval_start = time.time()
    
    samples = eval_questions[:max_eval_samples] if max_eval_samples else eval_questions
    correct = 0
    total = len(samples)
    
    with torch.no_grad():
        for i, q in enumerate(samples):
            # Use SAME format as on-policy: raw encode, not chat template
            ids = tokenizer.encode(q["prompt"])
            input_ids = torch.tensor(ids, dtype=torch.long, device=device)
            seq_ids = torch.zeros_like(input_ids)
            position_ids = torch.arange(len(ids), dtype=torch.long, device=device)
            
            cache.clear()
            
            try:
                gen_output = flex_generate(
                    model=flex_model,
                    tokenizer=tokenizer,
                    cache=cache,
                    input_ids=input_ids,
                    seq_ids=seq_ids,
                    position_ids=position_ids,
                    max_new_tokens=10,
                    temperature=0.0,
                )
                gen_tokens = gen_output.get(0, [])
                gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
                predicted = extract_answer(gen_text)
            except Exception as e:
                logger.debug(f"    Eval generation failed for Q{i}: {e}")
                predicted = "?"
            
            if predicted == q["correct"]:
                correct += 1
    
    accuracy = 100 * correct / total
    elapsed = time.time() - eval_start
    logger.info(f"  [EVAL] Step {step}: {correct}/{total} = {accuracy:.1f}% ({elapsed:.1f}s)")
    
    return {"step": step, "accuracy": round(accuracy, 2), "correct": correct, "total": total}


def synthesize_with_teacher(teacher_model, tokenizer, doc_ids, prompt_ids_list, max_tokens, temperature, device, top_k=20):
    """
    SYNTHESIS: Generate responses AND compute teacher logprobs IN ONE PASS.
    
    This matches the paper's approach:
    - Generate response tokens using teacher with full document
    - Compute top-k logprobs for each generated token
    - Return both response AND logprobs (no recomputation needed later)
    
    Returns list of dicts with: prompt_ids, response_ids, topk_token_ids, topk_logprobs
    """
    results = []
    
    # Pre-compute document KV cache once
    doc_t = torch.tensor([doc_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        doc_out = teacher_model(doc_t, use_cache=True)
        doc_kv = doc_out.past_key_values
    doc_len = len(doc_ids)
    
    with torch.no_grad():
        for prompt_ids in prompt_ids_list:
            # Clone doc KV for this sample
            from transformers.cache_utils import DynamicCache
            sample_kv = DynamicCache()
            try:
                kv_pairs = list(zip(doc_kv.key_cache, doc_kv.value_cache))
            except AttributeError:
                kv_pairs = [(layer.keys, layer.values) for layer in doc_kv.layers]
            for k, v in kv_pairs:
                try:
                    sample_kv.key_cache.append(k.clone())
                    sample_kv.value_cache.append(v.clone())
                except AttributeError:
                    layer_idx = len(sample_kv)
                    sample_kv.update(k.clone(), v.clone(), layer_idx)
            
            input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
            position_ids = torch.arange(doc_len, doc_len + len(prompt_ids), dtype=torch.long, device=device).unsqueeze(0)
            
            # Generate AND collect logprobs in one pass
            generated_ids = input_ids.clone()
            all_topk_ids = []
            all_topk_lps = []
            
            for _ in range(max_tokens):
                with torch.autocast(device_type="cuda" if device.startswith("cuda") else device, 
                                   dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
                                   enabled=device.startswith("cuda")):
                    outputs = teacher_model(
                        input_ids=generated_ids[:, -1:] if generated_ids.shape[1] > len(prompt_ids) else generated_ids,
                        position_ids=position_ids[:, -1:] if position_ids.shape[1] > len(prompt_ids) else position_ids,
                        past_key_values=sample_kv,
                        use_cache=True,
                    )
                sample_kv = outputs.past_key_values
                
                logits = outputs.logits[:, -1, :]
                
                # Compute top-k logprobs for this position
                log_probs = F.log_softmax(logits.float(), dim=-1)
                topk_lps, topk_ids = torch.topk(log_probs, k=top_k, dim=-1)
                all_topk_ids.append(topk_ids[0].cpu())
                all_topk_lps.append(topk_lps[0].cpu())
                
                # Sample next token
                if temperature > 0:
                    probs = F.softmax(logits / temperature, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = logits.argmax(dim=-1, keepdim=True)
                
                generated_ids = torch.cat([generated_ids, next_token], dim=1)
                position_ids = torch.cat([position_ids, position_ids[:, -1:] + 1], dim=1)
                
                # Stop on EOS
                if next_token.item() == tokenizer.eos_token_id:
                    break
            
            # Extract response (excluding prompt)
            response_ids = generated_ids[0, len(prompt_ids):].tolist()
            
            if response_ids:  # Only add if we generated something
                results.append({
                    "prompt_ids": prompt_ids,
                    "response_ids": response_ids,
                    "topk_token_ids": torch.stack(all_topk_ids),  # [resp_len, top_k]
                    "topk_logprobs": torch.stack(all_topk_lps),   # [resp_len, top_k]
                })
            
            del sample_kv
    
    del doc_kv, doc_out, doc_t
    cleanup_memory(device)
    
    return results


def compute_teacher_topk(model, tokenizer, doc_ids, prompts, responses, device, top_k=20, micro_batch=4):
    """Compute teacher model's top-k logprobs for responses."""
    results = []
    
    # Compute document KV cache once
    doc_t = torch.tensor([doc_ids], device=device)
    with torch.no_grad():
        doc_out = model(doc_t, use_cache=True)
        doc_kv = doc_out.past_key_values
    
    doc_len = len(doc_ids)
    
    for start in range(0, len(prompts), micro_batch):
        mb_prompts = prompts[start:start + micro_batch]
        mb_responses = responses[start:start + micro_batch]
        mb_size = len(mb_prompts)
        
        sequences = [p + r for p, r in zip(mb_prompts, mb_responses)]
        max_len = max(len(s) for s in sequences)
        
        padded_ids = []
        padded_mask = []
        padded_pos = []
        
        for seq in sequences:
            pad_len = max_len - len(seq)
            ids = [tokenizer.pad_token_id or 0] * pad_len + seq
            mask = [0] * pad_len + [1] * len(seq)
            pos = list(range(doc_len, doc_len + max_len))
            
            padded_ids.append(torch.tensor(ids))
            padded_mask.append(torch.tensor(mask))
            padded_pos.append(torch.tensor(pos))
        
        batch_ids = torch.stack(padded_ids).to(device)
        batch_mask = torch.stack(padded_mask)
        batch_pos = torch.stack(padded_pos).to(device)
        
        from transformers.cache_utils import DynamicCache
        expanded_kv = DynamicCache()
        try:
            kv_pairs = list(zip(doc_kv.key_cache, doc_kv.value_cache))
        except AttributeError:
            kv_pairs = [(layer.keys, layer.values) for layer in doc_kv.layers]
        for k, v in kv_pairs:
            ek = k.expand(mb_size, -1, -1, -1).contiguous()
            ev = v.expand(mb_size, -1, -1, -1).contiguous()
            try:
                expanded_kv.key_cache.append(ek)
                expanded_kv.value_cache.append(ev)
            except AttributeError:
                layer_idx = len(expanded_kv)
                expanded_kv.update(ek, ev, layer_idx)
        
        doc_prefix_mask = torch.ones(mb_size, doc_len, dtype=torch.long, device=device)
        full_mask = torch.cat([doc_prefix_mask, batch_mask.to(device)], dim=1)
        
        with torch.no_grad():
            output = model(
                input_ids=batch_ids,
                attention_mask=full_mask,
                position_ids=batch_pos,
                past_key_values=expanded_kv,
                use_cache=False,
            )
        
        logits = output.logits
        
        for j in range(mb_size):
            seq = sequences[j]
            pad_offset = max_len - len(seq)
            prompt_len = len(mb_prompts[j])
            response_len = len(mb_responses[j])
            
            resp_start = pad_offset + prompt_len
            resp_logits = logits[j, resp_start - 1 : resp_start + response_len - 1]
            
            log_probs = F.log_softmax(resp_logits.float(), dim=-1)
            topk_lps, topk_ids = torch.topk(log_probs, k=top_k, dim=-1)
            
            results.append({
                "topk_token_ids": topk_ids.cpu(),
                "topk_logprobs": topk_lps.cpu(),
                "response_ids": mb_responses[j],
                "prompt_ids": mb_prompts[j],
            })
        
        del expanded_kv, output, logits
        cleanup_memory(device)
    
    del doc_kv, doc_out, doc_t
    cleanup_memory(device)
    
    return results


def build_packed_batch(samples, packed_seq_length=2048):
    """Pack samples into a single sequence (for compute_teacher_topk output)."""
    all_input_ids, all_element_ids, all_position_ids = [], [], []
    all_topk_ids, all_topk_lps, all_topk_idxs = [], [], []
    curr_offset = 0
    
    for elem_id, s in enumerate(samples):
        seq = s["prompt_ids"] + s["response_ids"]
        seq_len = len(seq)
        prompt_len = len(s["prompt_ids"])
        response_len = len(s["response_ids"])
        
        if curr_offset + seq_len > packed_seq_length:
            break
        
        all_input_ids.append(torch.tensor(seq, dtype=torch.long))
        all_element_ids.append(torch.full((seq_len,), elem_id, dtype=torch.long))
        all_position_ids.append(torch.arange(seq_len, dtype=torch.long))
        
        resp_idxs = torch.arange(prompt_len, prompt_len + response_len, dtype=torch.long) + curr_offset
        all_topk_idxs.append(resp_idxs.unsqueeze(1).expand(-1, s["topk_token_ids"].shape[1]))
        all_topk_ids.append(s["topk_token_ids"])
        all_topk_lps.append(s["topk_logprobs"])
        
        curr_offset += seq_len
    
    input_ids = torch.cat(all_input_ids)
    element_ids = torch.cat(all_element_ids)
    position_ids = torch.cat(all_position_ids)
    
    if len(input_ids) < packed_seq_length:
        pad_len = packed_seq_length - len(input_ids)
        input_ids = torch.cat([input_ids, torch.zeros(pad_len, dtype=torch.long)])
        element_ids = torch.cat([element_ids, torch.zeros(pad_len, dtype=torch.long)])
        position_ids = torch.cat([position_ids, torch.zeros(pad_len, dtype=torch.long)])
    
    return {
        "input_ids": input_ids,
        "element_ids": element_ids,
        "position_ids": position_ids,
        "topk_token_ids": torch.cat(all_topk_ids).reshape(-1),
        "topk_logprobs": torch.cat(all_topk_lps).reshape(-1),
        "topk_token_idxs": torch.cat(all_topk_idxs).reshape(-1),
    }


def build_packed_batch_precomputed(samples, packed_seq_length=2048):
    """
    Pack samples with PRE-COMPUTED logprobs into a single sequence.
    
    This is for the paper-matching approach where logprobs are computed
    during synthesis, not during training.
    """
    all_input_ids, all_element_ids, all_position_ids = [], [], []
    all_topk_ids, all_topk_lps, all_topk_idxs = [], [], []
    curr_offset = 0
    
    for elem_id, s in enumerate(samples):
        seq = s["prompt_ids"] + s["response_ids"]
        seq_len = len(seq)
        prompt_len = len(s["prompt_ids"])
        response_len = len(s["response_ids"])
        
        if curr_offset + seq_len > packed_seq_length:
            break
        
        all_input_ids.append(torch.tensor(seq, dtype=torch.long))
        all_element_ids.append(torch.full((seq_len,), elem_id, dtype=torch.long))
        all_position_ids.append(torch.arange(seq_len, dtype=torch.long))
        
        # Pre-computed logprobs are [response_len, top_k]
        topk_ids = s["topk_token_ids"]  # Already a tensor
        topk_lps = s["topk_logprobs"]   # Already a tensor
        
        # Build position indices for response tokens
        resp_idxs = torch.arange(prompt_len, prompt_len + response_len, dtype=torch.long) + curr_offset
        all_topk_idxs.append(resp_idxs.unsqueeze(1).expand(-1, topk_ids.shape[1]))
        all_topk_ids.append(topk_ids)
        all_topk_lps.append(topk_lps)
        
        curr_offset += seq_len
    
    if not all_input_ids:
        raise ValueError("No samples fit in packed sequence")
    
    input_ids = torch.cat(all_input_ids)
    element_ids = torch.cat(all_element_ids)
    position_ids = torch.cat(all_position_ids)
    
    if len(input_ids) < packed_seq_length:
        pad_len = packed_seq_length - len(input_ids)
        input_ids = torch.cat([input_ids, torch.zeros(pad_len, dtype=torch.long)])
        element_ids = torch.cat([element_ids, torch.zeros(pad_len, dtype=torch.long)])
        position_ids = torch.cat([position_ids, torch.zeros(pad_len, dtype=torch.long)])
    
    return {
        "input_ids": input_ids,
        "element_ids": element_ids,
        "position_ids": position_ids,
        "topk_token_ids": torch.cat(all_topk_ids).reshape(-1),
        "topk_logprobs": torch.cat(all_topk_lps).reshape(-1),
        "topk_token_idxs": torch.cat(all_topk_idxs).reshape(-1),
    }


def load_hf_shard_conversations(shard_dir: str, limit: int = None) -> list:
    """Load conversations with pre-computed logprobs from local HF shard."""
    import pyarrow.parquet as pq
    
    conversations = []
    parquet_files = sorted(Path(shard_dir).glob("*.parquet"))
    
    for pf in parquet_files:
        df = pd.read_parquet(pf)
        for _, row in df.iterrows():
            if limit and len(conversations) >= limit:
                break
            conversations.append(Conversation.from_dict(row.to_dict()))
        if limit and len(conversations) >= limit:
            break
    
    return conversations[:limit] if limit else conversations


def train_offpolicy(
    model_path: str,
    hf_shard_dir: str,
    num_tokens: int = 512,
    lr: float = 0.001,
    total_steps: int = 100,
    batch_size: int = 8,
    eval_every: int = 50,
    save_every: int = 50,
    save_dir: str = "./local_checkpoints",
    max_eval_samples: int = None,
):
    """
    Off-policy training using PRE-COMPUTED teacher logprobs from HF shards.
    
    NO synthesis needed - data already has logprobs!
    Train for 1 epoch (no data reuse) matching the paper.
    """
    num_samples_needed = total_steps * batch_size
    
    logger.info("=" * 70)
    logger.info("OFF-POLICY TRAINING (PRE-COMPUTED LOGPROBS)")
    logger.info("=" * 70)
    logger.info(f"  Total steps: {total_steps}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Samples needed: {num_samples_needed}")
    logger.info(f"  Data source: {hf_shard_dir}")
    logger.info("=" * 70)
    
    # Device selection
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.bfloat16
    elif torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float32
    else:
        device = "cpu"
        dtype = torch.float32
    
    logger.info(f"Device: {device}, dtype: {dtype}")
    
    # Load model (NO teacher model needed - logprobs are pre-computed!)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model_lower = model_path.lower()
    if "llama-3.2-3b-instruct" in model_lower:
        tokenizer.name_or_path = "meta-llama/Llama-3.2-3B-Instruct"
    
    flex_model = FlexLlamaForCausalLM.from_pretrained(model_path, torch_dtype=dtype).to(device).eval()
    for p in flex_model.parameters():
        p.requires_grad = False
    
    attn_config = AttnConfig(
        n_layers=flex_model.config.num_hidden_layers,
        n_heads=flex_model.config.num_key_value_heads,
        head_dim=flex_model.config.hidden_size // flex_model.config.num_attention_heads,
    )
    
    # ==========================================================================
    # LOAD PRE-COMPUTED DATA (NO SYNTHESIS!)
    # ==========================================================================
    logger.info("=" * 70)
    logger.info("LOADING PRE-COMPUTED DATA WITH TEACHER LOGPROBS")
    logger.info(f"  Source: {hf_shard_dir}")
    logger.info(f"  Loading {num_samples_needed} samples...")
    logger.info("=" * 70)
    
    load_start = time.time()
    conversations = load_hf_shard_conversations(hf_shard_dir, limit=num_samples_needed)
    load_elapsed = time.time() - load_start
    logger.info(f"Loaded {len(conversations)} conversations in {load_elapsed:.1f}s")
    
    # Convert to training format using cartridges infrastructure
    from cartridges.datasets import llama3_messages_to_element
    
    train_elements = []
    precomputed_gen_tokens = 0  # Track tokens that "cost" to generate the pre-computed data
    for convo in conversations:
        elem = llama3_messages_to_element(convo.messages, retokenize=False, tokenizer=tokenizer)
        train_elements.append(elem)
        # Count the full sequence length - this is what it would cost to generate
        precomputed_gen_tokens += len(elem.input_ids)
    logger.info(f"Converted {len(train_elements)} elements for training")
    logger.info(f"Pre-computed generation tokens (fair cost): {precomputed_gen_tokens:,}")
    
    # Shuffle for training
    np.random.seed(42)
    np.random.shuffle(train_elements)
    
    # Load LongHealth for evaluation
    logger.info("Downloading LongHealth benchmark data...")
    lh_data = requests.get(
        "https://raw.githubusercontent.com/kbressem/LongHealth/refs/heads/main/data/benchmark_v5.json"
    ).json()
    
    # Build patient documents for cache initialization
    patient_doc_texts = {}
    for pid, patient in lh_data.items():
        doc_text = "\n\n".join(f"--- {did} ---\n{txt}" for did, txt in patient["texts"].items())
        patient_doc_texts[pid] = doc_text
    
    # Load evaluation questions
    EVAL_PATIENT_IDS = {f"patient_{i:02d}" for i in range(1, 11)}
    eval_questions = load_eval_questions(lh_data, EVAL_PATIENT_IDS)
    np.random.shuffle(eval_questions)
    logger.info(f"Loaded {len(eval_questions)} eval questions")
    
    # Initialize cache from patient documents
    training_patients = list(patient_doc_texts.keys())[:10]  # Use first 10 patients
    tokens_per_patient = num_tokens // len(training_patients)
    init_text_parts = []
    for pid in training_patients:
        doc_text = patient_doc_texts[pid]
        char_limit = tokens_per_patient * 4
        init_text_parts.append(doc_text[:char_limit])
    init_text = "\n\n".join(init_text_parts)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(init_text)
        init_file = f.name
    
    initializer = KVFromText.Config(
        max_tokens=num_tokens,
        text_source=init_file,
    ).instantiate()
    cache = initializer.initialize_kv_cache(
        tokenizer=tokenizer, model=flex_model, attn_config=attn_config,
    ).to(device)
    os.unlink(init_file)
    
    wrapped_model = CacheAndModel(cache, flex_model)
    optimizer = optim.Adam(cache.parameters(), lr=lr)
    
    # Setup
    os.makedirs(save_dir, exist_ok=True)
    
    # Add file handler for persistent logging
    log_file_path = os.path.join(save_dir, "training.log")
    file_handler = logging.FileHandler(log_file_path, mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s'))
    logger.addHandler(file_handler)
    logger.info(f"  - Log file: {log_file_path}")
    
    eval_results = []
    # Start with pre-computed generation tokens for FAIR comparison with on-policy
    # On-policy counts generation tokens, so off-policy must too
    total_tokens = precomputed_gen_tokens
    logger.info(f"Starting token count at {total_tokens:,} (pre-computed generation cost)")
    token_history = []
    
    # Step 0 evaluation
    eval_result = evaluate_cache(
        flex_model=flex_model,
        tokenizer=tokenizer,
        cache=cache,
        eval_questions=eval_questions,
        device=device,
        step=0,
        max_eval_samples=max_eval_samples,
    )
    eval_results.append(eval_result)
    token_history.append({"step": 0, "tokens": total_tokens, "accuracy": eval_result["accuracy"]})
    
    # Save initial results (crash protection)
    results_path = os.path.join(save_dir, "offpolicy_results.json")
    with open(results_path, "w") as f:
        json.dump({
            "eval_results": eval_results,
            "token_history": token_history,
            "config": {"mode": "off-policy", "total_steps": total_steps, "batch_size": batch_size, "lr": lr}
        }, f, indent=2)
    logger.info(f"  - Results saved: {results_path}")
    
    # Save step-0 checkpoint
    ckpt_path = os.path.join(save_dir, "step-0", "cartridge.pt")
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    cache.save(ckpt_path)
    
    # ==========================================================================
    # TRAINING LOOP - 1 EPOCH, NO REUSE (matches paper)
    # Uses PRE-COMPUTED logprobs from HF shards!
    # ==========================================================================
    logger.info("=" * 70)
    logger.info("TRAINING: 1 epoch on pre-computed data (NO REUSE)")
    logger.info("  Using PRE-COMPUTED teacher logprobs from HF shards")
    logger.info("  NO teacher model needed!")
    logger.info("=" * 70)
    
    for step in range(1, total_steps + 1):
        cleanup_memory(device)
        step_t0 = time.time()
        logger.info(f"--- STEP {step}/{total_steps} ---")
        
        # Get batch - each sample used EXACTLY ONCE (no cycling!)
        batch_start = (step - 1) * batch_size
        batch_end = min(batch_start + batch_size, len(train_elements))
        batch_elements = train_elements[batch_start:batch_end]
        
        if not batch_elements:
            logger.warning(f"No more data at step {step}, stopping early")
            break
        
        # Train on batch using cartridges collate (handles tensor shapes correctly)
        train_t0 = time.time()
        
        accum_loss = 0.0
        optimizer.zero_grad()
        
        # Use cartridges collate function for correct tensor packing
        from cartridges.datasets import TrainDataset
        
        # Create a minimal config for collate
        class MinimalConfig:
            packed_seq_length = 2048
            packing_mode = "pad"
        
        # Manually collate the batch elements
        input_ids_list, element_ids_list, position_ids_list = [], [], []
        topk_ids_list, topk_lps_list, topk_idxs_list = [], [], []
        curr_offset = 0
        
        for elem_id, elem in enumerate(batch_elements):
            seq_len = len(elem.input_ids)
            input_ids_list.append(elem.input_ids)
            element_ids_list.append(torch.full((seq_len,), elem_id, dtype=torch.long))
            position_ids_list.append(torch.arange(seq_len, dtype=torch.long))
            
            if elem.topk_token_ids is not None and len(elem.topk_token_ids) > 0:
                topk_ids_list.append(elem.topk_token_ids)
                topk_lps_list.append(elem.topk_logprobs)
                topk_idxs_list.append(elem.topk_token_idxs + curr_offset)
            
            curr_offset += seq_len
        
        input_ids = torch.cat(input_ids_list)
        element_ids = torch.cat(element_ids_list)
        position_ids = torch.cat(position_ids_list)
        
        # Pad to fixed length to avoid recompiles
        packed_seq_length = 2048
        if len(input_ids) < packed_seq_length:
            pad_len = packed_seq_length - len(input_ids)
            input_ids = torch.cat([input_ids, torch.zeros(pad_len, dtype=torch.long)])
            element_ids = torch.cat([element_ids, torch.zeros(pad_len, dtype=torch.long)])
            position_ids = torch.cat([position_ids, torch.zeros(pad_len, dtype=torch.long)])
        
        # Track training tokens
        total_tokens += curr_offset  # actual tokens, not padded
        
        if topk_ids_list:
            # Flatten to 1D for indexing (these are already 1D from cartridges)
            topk_token_ids = torch.cat(topk_ids_list)
            topk_logprobs = torch.cat(topk_lps_list)
            topk_token_idxs = torch.cat(topk_idxs_list)
            
            cache.clear()
            outputs = wrapped_model(
                input_ids=input_ids.to(device),
                seq_ids=element_ids.to(device),
                position_ids=position_ids.to(device),
            )
            
            # Compute loss using pre-computed teacher logprobs
            # outputs.logits is [1, seq_len, vocab_size] - need batch index 0
            topk_pred_logprobs = F.log_softmax(outputs.logits.float(), dim=-1)[
                0,
                topk_token_idxs.long().to(device) - 1,
                topk_token_ids.long().to(device),
            ]
            
            ce_by_token = -topk_logprobs.to(device).float().exp() * topk_pred_logprobs
            loss = ce_by_token.mean()
            
            loss.backward()
            accum_loss = loss.detach().item()
        
        optimizer.step()
        optimizer.zero_grad()
        train_elapsed = time.time() - train_t0
        step_elapsed = time.time() - step_t0
        
        logger.info(f"Step {step}: loss={accum_loss:.4f} | train={train_elapsed:.1f}s | total={step_elapsed:.1f}s | tokens={total_tokens:,}")
        
        # Evaluate (every eval_every steps or final step)
        if step % eval_every == 0 or step == total_steps:
            eval_result = evaluate_cache(
                flex_model=flex_model,
                tokenizer=tokenizer,
                cache=cache,
                eval_questions=eval_questions,
                device=device,
                step=step,
                max_eval_samples=max_eval_samples,
            )
            eval_results.append(eval_result)
            token_history.append({"step": step, "tokens": total_tokens, "accuracy": eval_result["accuracy"]})
            
            # Save intermediate results (crash protection)
            results_path = os.path.join(save_dir, "offpolicy_results.json")
            with open(results_path, "w") as f:
                json.dump({
                    "eval_results": eval_results,
                    "token_history": token_history,
                    "config": {"mode": "off-policy", "total_steps": total_steps, "batch_size": batch_size, "lr": lr}
                }, f, indent=2)
            logger.info(f"  - Results saved: {results_path}")
        
        # Save checkpoint (every save_every steps or final step)
        if step % save_every == 0 or step == total_steps:
            ckpt_path = os.path.join(save_dir, f"step-{step}", "cartridge.pt")
            os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
            cache.save(ckpt_path)
            logger.info(f"  - Checkpoint saved: {ckpt_path}")
        
        cleanup_memory(device)
    
    # Save results
    results_path = os.path.join(save_dir, "offpolicy_results.json")
    with open(results_path, "w") as f:
        json.dump({
            "eval_results": eval_results,
            "token_history": token_history,
            "config": {
                "mode": "off-policy (pre-computed logprobs)",
                "data_source": hf_shard_dir,
                "num_samples": len(train_elements),
                "total_steps": total_steps,
                "batch_size": batch_size,
                "lr": lr,
                "data_reuse": "NONE (1 epoch)",
            }
        }, f, indent=2)
    
    logger.info("=" * 70)
    logger.info("OFF-POLICY TRAINING COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Total tokens: {total_tokens:,}")
    logger.info(f"Results saved to: {results_path}")
    
    return eval_results, token_history


def main():
    parser = argparse.ArgumentParser(description="Off-policy cartridge training (pre-computed logprobs)")
    parser.add_argument("--model", required=True, help="Path to model")
    parser.add_argument("--hf-shard-dir", required=True, help="Directory with HF shard parquets (has pre-computed logprobs)")
    parser.add_argument("--num-tokens", type=int, default=512, help="Cache size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--total-steps", type=int, default=500, help="Training steps")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--eval-every", type=int, default=50, help="Evaluate every N steps")
    parser.add_argument("--save-every", type=int, default=50, help="Save checkpoint every N steps")
    parser.add_argument("--save-dir", type=str, default="./local_checkpoints/off_policy_run")
    parser.add_argument("--max-eval-samples", type=int, default=None, help="Limit eval samples (default: all)")
    
    args = parser.parse_args()
    
    logger.info("Arguments:")
    for k, v in vars(args).items():
        logger.info(f"  --{k}: {v}")
    logger.info(f"  Samples needed: {args.total_steps * args.batch_size} (total_steps × batch_size)")
    
    train_offpolicy(
        model_path=args.model,
        hf_shard_dir=args.hf_shard_dir,
        num_tokens=args.num_tokens,
        lr=args.lr,
        total_steps=args.total_steps,
        batch_size=args.batch_size,
        eval_every=args.eval_every,
        save_every=args.save_every,
        save_dir=args.save_dir,
        max_eval_samples=args.max_eval_samples,
    )


if __name__ == "__main__":
    main()
