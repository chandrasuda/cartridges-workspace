#!/usr/bin/env python3
"""
Off-policy cartridge training — pre-generates responses once, then trains repeatedly.

This is the off-policy baseline: responses are generated once at step 0 and 
reused for all training steps. Contrast with on-policy (local_cartridge_train.py)
which regenerates responses each step using the current cartridge.
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
logger.info("OFF-POLICY CARTRIDGE TRAINING")
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


def generate_with_teacher(teacher_model, tokenizer, doc_ids, prompt_ids_list, max_tokens, temperature, device):
    """
    Generate responses using TEACHER model with FULL DOCUMENT context.
    
    This is the key difference from on-policy:
    - On-policy: Student (cartridge) generates responses
    - Off-policy: Teacher (full doc) generates responses
    """
    responses = []
    
    # Pre-compute document KV cache once
    doc_t = torch.tensor([doc_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        doc_out = teacher_model(doc_t, use_cache=True)
        doc_kv = doc_out.past_key_values
    doc_len = len(doc_ids)
    
    with torch.no_grad():
        for prompt_ids in prompt_ids_list:
            # Expand doc KV for this sample
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
            
            # Generate with teacher using full document as KV prefix
            generated_ids = input_ids.clone()
            
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
            responses.append(response_ids)
            
            del sample_kv
    
    del doc_kv, doc_out, doc_t
    cleanup_memory(device)
    
    return responses


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
    """Pack samples into a single sequence."""
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


def train_offpolicy(
    model_path: str,
    train_parquet: str,
    num_tokens: int = 512,
    lr: float = 0.001,
    total_steps: int = 100,
    batch_size: int = 8,
    max_response_length: int = 256,
    temperature: float = 0.7,
    save_dir: str = "./local_checkpoints",
    max_eval_samples: int = None,
    max_doc_tokens: int = 4096,
    num_pregenerated: int = 100,  # Number of responses to pre-generate
):
    """Off-policy training: pre-generate responses once, then train on them repeatedly."""
    
    logger.info("=" * 70)
    logger.info("OFF-POLICY TRAINING")
    logger.info("=" * 70)
    logger.info(f"  Pre-generating {num_pregenerated} responses at step 0")
    logger.info(f"  Then training {total_steps} steps on fixed data")
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
    
    # Load models
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model_lower = model_path.lower()
    if "llama-3.2-3b-instruct" in model_lower:
        tokenizer.name_or_path = "meta-llama/Llama-3.2-3B-Instruct"
    
    teacher_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype).to(device).eval()
    flex_model = FlexLlamaForCausalLM.from_pretrained(model_path, torch_dtype=dtype).to(device).eval()
    for p in flex_model.parameters():
        p.requires_grad = False
    
    attn_config = AttnConfig(
        n_layers=flex_model.config.num_hidden_layers,
        n_heads=flex_model.config.num_key_value_heads,
        head_dim=flex_model.config.hidden_size // flex_model.config.num_attention_heads,
    )
    
    # Load training data
    train_df = pd.read_parquet(train_parquet)
    # CRITICAL: Shuffle to mix patients - data is sorted by patient_id!
    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
    logger.info(f"Loaded {len(train_df)} training prompts (SHUFFLED)")
    
    # Load LongHealth data directly from GitHub (MATCHES on-policy exactly)
    logger.info("Downloading LongHealth benchmark data...")
    lh_data = requests.get(
        "https://raw.githubusercontent.com/kbressem/LongHealth/refs/heads/main/data/benchmark_v5.json"
    ).json()
    
    # Build patient documents (MATCHES on-policy exactly)
    patient_doc_ids = {}
    patient_doc_texts = {}
    total_doc_tokens = 0
    for pid, patient in lh_data.items():
        doc_text = "\n\n".join(f"--- {did} ---\n{txt}" for did, txt in patient["texts"].items())
        patient_doc_texts[pid] = doc_text
        patient_doc_ids[pid] = tokenizer.encode(doc_text, add_special_tokens=False)
        total_doc_tokens += len(patient_doc_ids[pid])
    logger.info(f"Loaded {len(patient_doc_ids)} patient documents, {total_doc_tokens:,} total tokens")
    
    # Load evaluation questions (MATCHES on-policy exactly)
    EVAL_PATIENT_IDS = {f"patient_{i:02d}" for i in range(1, 11)}
    eval_questions = load_eval_questions(lh_data, EVAL_PATIENT_IDS)
    # Shuffle for balanced coverage when using max_eval_samples
    np.random.seed(42)
    np.random.shuffle(eval_questions)
    logger.info(f"Loaded {len(eval_questions)} eval questions (SHUFFLED for balanced coverage)")
    
    # Initialize cache from patient documents (MATCHES on-policy exactly)
    training_patients = sorted(train_df["patient_id"].unique())
    tokens_per_patient = num_tokens // len(training_patients)
    init_text_parts = []
    for pid in training_patients:
        if pid in patient_doc_texts:
            # Get the first portion of this patient's document
            doc_text = patient_doc_texts[pid]
            # Truncate to approximately tokens_per_patient worth of text
            char_limit = tokens_per_patient * 4
            init_text_parts.append(doc_text[:char_limit])
    init_text = "\n\n".join(init_text_parts)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(init_text)
        init_file = f.name
    
    # Initialize cache using the Config pattern (matches on-policy script)
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
    
    # Setup save directory
    os.makedirs(save_dir, exist_ok=True)
    eval_results = []
    
    # Token tracking
    total_tokens = 0
    token_history = []
    
    # ==========================================================================
    # STEP 0: Pre-generate ALL responses using TEACHER with FULL DOCUMENT
    # This is the key difference from on-policy:
    # - Off-policy: Teacher (full doc) generates responses
    # - On-policy: Student (cartridge) generates responses
    # ==========================================================================
    logger.info("=" * 70)
    logger.info(f"PRE-GENERATING {num_pregenerated} RESPONSES WITH TEACHER (OFF-POLICY)")
    logger.info("  Using TEACHER model with FULL DOCUMENT context for generation")
    logger.info("=" * 70)
    
    # Collect prompts grouped by patient for efficient generation
    prompts_by_patient = {}
    for i in range(num_pregenerated):
        row = train_df.iloc[i % len(train_df)]
        messages = row["prompt"]
        if isinstance(messages, str):
            messages = json.loads(messages)
        patient_id = row["patient_id"]
        prompt_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
        prompts_by_patient.setdefault(patient_id, []).append({
            "prompt_ids": prompt_ids,
            "patient_id": patient_id,
        })
    
    pregenerated_data = []
    gen_start = time.time()
    gen_count = 0
    
    for patient_id, patient_prompts in prompts_by_patient.items():
        # Get document for this patient
        doc_ids = patient_doc_ids.get(patient_id, [])
        if len(doc_ids) > max_doc_tokens:
            doc_ids = doc_ids[:max_doc_tokens]
        
        # Generate all responses for this patient using teacher with full document
        prompt_ids_list = [p["prompt_ids"] for p in patient_prompts]
        
        # Track generation tokens (doc + prompt + response for each)
        for prompt_ids in prompt_ids_list:
            total_tokens += len(doc_ids) + len(prompt_ids)  # Doc processed once per prompt
        
        response_ids_list = generate_with_teacher(
            teacher_model, tokenizer, doc_ids, prompt_ids_list, 
            max_response_length, temperature, device
        )
        
        # Track response tokens
        for resp_ids in response_ids_list:
            total_tokens += len(resp_ids)
        
        for prompt_data, resp_ids in zip(patient_prompts, response_ids_list):
            if resp_ids:
                pregenerated_data.append({
                    "prompt_ids": prompt_data["prompt_ids"],
                    "response_ids": resp_ids,
                    "patient_id": patient_id,
                })
        
        gen_count += len(patient_prompts)
        logger.info(f"  Pre-generated {gen_count}/{num_pregenerated} (patient {patient_id})")
        cleanup_memory(device)
    
    gen_elapsed = time.time() - gen_start
    logger.info(f"Pre-generation complete: {len(pregenerated_data)} samples in {gen_elapsed:.1f}s")
    logger.info(f"Generation tokens (teacher with doc): {total_tokens:,}")
    
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
    
    # Save step-0 checkpoint
    ckpt_path = os.path.join(save_dir, "step-0", "cartridge.pt")
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    cache.save(ckpt_path)
    
    # ==========================================================================
    # TRAINING LOOP (reusing pre-generated data)
    # ==========================================================================
    logger.info("=" * 70)
    logger.info("TRAINING ON PRE-GENERATED DATA (OFF-POLICY)")
    logger.info("=" * 70)
    
    for step in range(1, total_steps + 1):
        cleanup_memory(device)
        step_t0 = time.time()
        logger.info(f"--- STEP {step}/{total_steps} ---")
        
        # Sample from pre-generated data (cyclic)
        batch_indices = [(step * batch_size + i) % len(pregenerated_data) for i in range(batch_size)]
        batch_data = [pregenerated_data[idx] for idx in batch_indices]
        
        # Group by patient for teacher computation
        by_patient = {}
        for item in batch_data:
            pid = item["patient_id"]
            by_patient.setdefault(pid, []).append((item["prompt_ids"], item["response_ids"]))
        
        # Compute teacher top-k (NO generation needed - using pre-generated responses)
        teacher_t0 = time.time()
        all_samples = []
        for pid, items in by_patient.items():
            doc_ids = patient_doc_ids.get(pid, [])
            if len(doc_ids) > max_doc_tokens:
                doc_ids = doc_ids[:max_doc_tokens]
            p_list = [x[0] for x in items]
            r_list = [x[1] for x in items]
            
            # Track teacher tokens
            for p, r in zip(p_list, r_list):
                total_tokens += len(doc_ids) + len(p) + len(r)
            
            samples = compute_teacher_topk(teacher_model, tokenizer, doc_ids, p_list, r_list, device)
            all_samples.extend(samples)
        teacher_elapsed = time.time() - teacher_t0
        
        # Train
        train_t0 = time.time()
        np.random.shuffle(all_samples)
        
        accum_loss = 0.0
        n_batches = 0
        optimizer.zero_grad()
        
        i = 0
        while i < len(all_samples):
            batch = build_packed_batch(all_samples[i:])
            packed_count = batch["element_ids"].max().item() + 1
            i += max(packed_count, 1)
            
            # Track training tokens
            total_tokens += len(batch["input_ids"])
            
            cache.clear()
            outputs = wrapped_model(
                input_ids=batch["input_ids"].to(device),
                seq_ids=batch["element_ids"].to(device),
                position_ids=batch["position_ids"].to(device),
            )
            
            topk_pred_logprobs = F.log_softmax(outputs.logits, dim=-1)[
                0,
                batch["topk_token_idxs"].to(device) - 1,
                batch["topk_token_ids"].to(device),
            ]
            
            ce_by_token = -batch["topk_logprobs"].to(device).exp() * topk_pred_logprobs
            loss = ce_by_token.mean()
            
            loss.backward()
            accum_loss += loss.detach().item()
            n_batches += 1
        
        optimizer.step()
        optimizer.zero_grad()
        train_elapsed = time.time() - train_t0
        step_elapsed = time.time() - step_t0
        
        avg_loss = accum_loss / max(n_batches, 1)
        logger.info(f"Step {step}: loss={avg_loss:.4f} | teacher={teacher_elapsed:.1f}s | train={train_elapsed:.1f}s | total={step_elapsed:.1f}s | tokens={total_tokens:,}")
        
        # Evaluate
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
        
        # Save checkpoint
        ckpt_path = os.path.join(save_dir, f"step-{step}", "cartridge.pt")
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        cache.save(ckpt_path)
        
        cleanup_memory(device)
    
    # Save results
    results_path = os.path.join(save_dir, "offpolicy_results.json")
    with open(results_path, "w") as f:
        json.dump({
            "eval_results": eval_results,
            "token_history": token_history,
            "config": {
                "mode": "off-policy",
                "num_pregenerated": num_pregenerated,
                "total_steps": total_steps,
                "batch_size": batch_size,
                "lr": lr,
            }
        }, f, indent=2)
    
    logger.info("=" * 70)
    logger.info("OFF-POLICY TRAINING COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Total tokens: {total_tokens:,}")
    logger.info(f"Results saved to: {results_path}")
    
    return eval_results, token_history


def main():
    parser = argparse.ArgumentParser(description="Off-policy cartridge training")
    parser.add_argument("--model", required=True, help="Path to model")
    parser.add_argument("--train-parquet", required=True, help="Training data parquet")
    parser.add_argument("--num-tokens", type=int, default=512, help="Cache size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--total-steps", type=int, default=30, help="Training steps")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--save-dir", type=str, default="./local_checkpoints/off_policy_run")
    parser.add_argument("--max-eval-samples", type=int, default=50)
    parser.add_argument("--max-doc-tokens", type=int, default=4096)
    parser.add_argument("--num-pregenerated", type=int, default=100, help="Number of responses to pre-generate")
    
    args = parser.parse_args()
    
    logger.info("Arguments:")
    for k, v in vars(args).items():
        logger.info(f"  --{k}: {v}")
    
    train_offpolicy(
        model_path=args.model,
        train_parquet=args.train_parquet,
        num_tokens=args.num_tokens,
        lr=args.lr,
        total_steps=args.total_steps,
        batch_size=args.batch_size,
        save_dir=args.save_dir,
        max_eval_samples=args.max_eval_samples,
        max_doc_tokens=args.max_doc_tokens,
        num_pregenerated=args.num_pregenerated,
    )


if __name__ == "__main__":
    main()
