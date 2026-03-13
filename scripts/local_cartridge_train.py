#!/usr/bin/env python3
"""
Local cartridge training — no Tokasaurus, no Modal, pure MPS/CUDA.

Uses flex_generate directly for on-policy generation.
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
# LOGGING SETUP - Configure before anything else
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("local_cartridge_train")
logger.setLevel(logging.DEBUG)

# =============================================================================
# ENVIRONMENT SETUP - Must be set BEFORE importing cartridges
# =============================================================================
SCRIPT_DIR = Path(__file__).parent.resolve()
WORKSPACE_DIR = SCRIPT_DIR.parent
CARTRIDGES_DIR = WORKSPACE_DIR / "cartridges"

# Set required environment variables for cartridges module
os.environ["CARTRIDGES_DIR"] = str(CARTRIDGES_DIR)
os.environ["CARTRIDGES_OUTPUT_DIR"] = str(WORKSPACE_DIR / "local_checkpoints")

logger.info("=" * 70)
logger.info("LOCAL CARTRIDGE TRAINING SCRIPT")
logger.info("=" * 70)
logger.info(f"SCRIPT_DIR: {SCRIPT_DIR}")
logger.info(f"WORKSPACE_DIR: {WORKSPACE_DIR}")
logger.info(f"CARTRIDGES_DIR: {os.environ['CARTRIDGES_DIR']}")
logger.info(f"CARTRIDGES_OUTPUT_DIR: {os.environ['CARTRIDGES_OUTPUT_DIR']}")
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

logger.info("Importing cartridges modules...")
from cartridges.cache import AttnConfig, TrainableCache
logger.info("  - Imported cache module (AttnConfig, TrainableCache)")
from cartridges.initialization import KVFromText
logger.info("  - Imported initialization module (KVFromText)")
from cartridges.models.llama.modeling_llama import FlexLlamaForCausalLM
logger.info("  - Imported FlexLlamaForCausalLM")
from cartridges.train import CacheAndModel
logger.info("  - Imported train module (CacheAndModel)")
from cartridges.generation import flex_generate
logger.info("  - Imported generation module (flex_generate)")
logger.info("All cartridges modules imported successfully!")


# =============================================================================
# MEMORY MANAGEMENT
# =============================================================================

def cleanup_memory(device: str):
    """Force garbage collection and clear device memory."""
    gc.collect()
    if device == "mps":
        # Synchronize MPS to ensure all operations complete before clearing
        torch.mps.synchronize()
        torch.mps.empty_cache()
    elif device.startswith("cuda"):
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


# =============================================================================
# EVALUATION FUNCTIONS
# =============================================================================

def extract_answer(text: str) -> str:
    """Extract answer letter (A-E) from generated text."""
    m = re.search(r"\b([A-E])\b", text.strip()[:20])
    return m.group(1) if m else "?"


def load_eval_questions(lh_data: dict, eval_patient_ids: set) -> list:
    """Load evaluation questions for specified patient IDs."""
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


@torch.no_grad()
def evaluate_cache(
    flex_model,
    tokenizer,
    cache: TrainableCache,
    eval_questions: list,
    device: str,
    step: int,
    max_eval_samples: int = None,
) -> dict:
    """
    Evaluate the current cache on the test set.
    
    Args:
        flex_model: The FlexLlama model
        tokenizer: The tokenizer
        cache: The trainable cache to evaluate
        eval_questions: List of evaluation questions
        device: Device to run on
        step: Current training step (for logging)
        max_eval_samples: Maximum number of samples to evaluate (None = all)
    
    Returns:
        dict with evaluation results
    """
    logger.info(f"  [EVAL] Evaluating at step {step}...")
    
    questions_to_eval = eval_questions
    if max_eval_samples is not None and max_eval_samples < len(eval_questions):
        questions_to_eval = eval_questions[:max_eval_samples]
    
    correct_count = 0
    total_count = len(questions_to_eval)
    eval_t0 = time.time()
    
    for qi, q in enumerate(questions_to_eval):
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
            pred = extract_answer(gen_text)
        except Exception as e:
            logger.debug(f"    Eval generation failed for Q{qi}: {e}")
            pred = "?"
        
        if pred == q["correct"]:
            correct_count += 1
        
        # Log all predictions (to persistent log file)
        logger.debug(f"    Q{qi}: pred='{pred}' correct='{q['correct']}' | gen='{gen_text[:50]}...' ")
        
        # Progress every 20 questions
        if (qi + 1) % 20 == 0:
            running_acc = correct_count / (qi + 1) * 100
            logger.debug(f"    [{qi+1}/{total_count}] running acc={running_acc:.1f}%")
    
    eval_elapsed = time.time() - eval_t0
    accuracy = correct_count / total_count * 100
    
    logger.info(f"  [EVAL] Step {step}: {correct_count}/{total_count} = {accuracy:.1f}% ({eval_elapsed:.1f}s)")
    
    # Clear cache after eval
    cleanup_memory(device)
    
    return {
        "step": step,
        "correct": correct_count,
        "total": total_count,
        "accuracy": round(accuracy, 2),
        "eval_time": round(eval_elapsed, 2),
    }


def generate_batch_local(model, tokenizer, cache, prompt_ids_list, max_tokens, temperature, device):
    """Generate responses locally using flex_generate."""
    logger.debug(f"generate_batch_local: Starting generation for {len(prompt_ids_list)} prompts")
    logger.debug(f"  - max_tokens: {max_tokens}, temperature: {temperature}, device: {device}")
    
    responses = []
    
    for idx, prompt_ids in enumerate(prompt_ids_list):
        logger.debug(f"  - Generating response {idx+1}/{len(prompt_ids_list)}, prompt length: {len(prompt_ids)}")
        
        input_ids = torch.tensor(prompt_ids, dtype=torch.long, device=device)
        seq_ids = torch.zeros_like(input_ids)
        position_ids = torch.arange(len(input_ids), dtype=torch.long, device=device)
        
        cache.clear()
        
        try:
            output = flex_generate(
                model=model,
                tokenizer=tokenizer,
                cache=cache,
                input_ids=input_ids,
                seq_ids=seq_ids,
                position_ids=position_ids,
                max_new_tokens=max_tokens,
                temperature=temperature,
            )
            
            response_ids = output.get(0, [])
            responses.append(list(response_ids))
            logger.debug(f"    - Generated {len(response_ids)} tokens")
        except Exception as e:
            logger.error(f"    - Generation failed: {e}")
            responses.append([])
    
    logger.debug(f"generate_batch_local: Completed. {sum(1 for r in responses if r)} successful responses")
    return responses


@torch.no_grad()
def compute_teacher_topk(model, tokenizer, doc_ids, prompt_ids_list, response_ids_list, device, top_k=20, micro_batch=4):
    """Compute teacher top-k logprobs with full document as KV prefix."""
    logger.debug(f"compute_teacher_topk: Processing {len(prompt_ids_list)} prompts")
    logger.debug(f"  - doc_len: {len(doc_ids)}, top_k: {top_k}, micro_batch: {micro_batch}")
    
    # Compute doc KV cache once
    logger.debug("  - Computing document KV cache...")
    doc_t = torch.tensor(doc_ids, dtype=torch.long, device=device).unsqueeze(0)
    with torch.autocast(device_type="cuda" if device.startswith("cuda") else device, dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32, enabled=device.startswith("cuda")):
        doc_out = model(doc_t, use_cache=True)
    doc_kv = doc_out.past_key_values
    doc_len = len(doc_ids)
    logger.debug(f"  - Document KV cache computed, doc_len: {doc_len}")

    results = []

    for mb_start in range(0, len(prompt_ids_list), micro_batch):
        mb_prompts = prompt_ids_list[mb_start:mb_start + micro_batch]
        mb_responses = response_ids_list[mb_start:mb_start + micro_batch]
        mb_size = len(mb_prompts)

        sequences = [p + r for p, r in zip(mb_prompts, mb_responses)]
        max_len = max(len(s) for s in sequences)

        # Left-pad
        padded_ids, padded_mask, padded_pos = [], [], []
        for seq in sequences:
            pad_len = max_len - len(seq)
            if pad_len > 0:
                padded_ids.append(torch.cat([
                    torch.zeros(pad_len, dtype=torch.long, device=device),
                    torch.tensor(seq, dtype=torch.long, device=device),
                ]))
                padded_mask.append(torch.cat([
                    torch.zeros(pad_len, dtype=torch.long, device=device),
                    torch.ones(len(seq), dtype=torch.long, device=device),
                ]))
                padded_pos.append(torch.cat([
                    torch.zeros(pad_len, dtype=torch.long, device=device),
                    torch.arange(doc_len, doc_len + len(seq), dtype=torch.long, device=device),
                ]))
            else:
                padded_ids.append(torch.tensor(seq, dtype=torch.long, device=device))
                padded_mask.append(torch.ones(len(seq), dtype=torch.long, device=device))
                padded_pos.append(torch.arange(doc_len, doc_len + len(seq), dtype=torch.long, device=device))

        batch_ids = torch.stack(padded_ids)
        batch_mask = torch.stack(padded_mask)
        batch_pos = torch.stack(padded_pos)

        # Expand doc KV
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
        full_mask = torch.cat([doc_prefix_mask, batch_mask], dim=1)

        with torch.autocast(device_type="cuda" if device.startswith("cuda") else device, dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32, enabled=device.startswith("cuda")):
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
        
        # Clean up expanded KV cache after each microbatch
        del expanded_kv, output, logits
        cleanup_memory(device)

    # Clean up document KV cache
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


def train(
    model_path: str,
    train_parquet: str,
    num_tokens: int = 512,
    lr: float = 0.001,
    total_steps: int = 100,
    batch_size: int = 8,  # Smaller for local
    max_response_length: int = 256,
    temperature: float = 0.7,
    eval_every: int = 50,
    save_every: int = 50,
    save_dir: str = "./local_checkpoints",
    max_eval_samples: int = None,
    max_doc_tokens: int = 4096,
):
    logger.info("=" * 70)
    logger.info("STARTING TRAINING")
    logger.info("=" * 70)
    logger.info("Training Configuration:")
    logger.info(f"  - model_path: {model_path}")
    logger.info(f"  - train_parquet: {train_parquet}")
    logger.info(f"  - num_tokens (cache size): {num_tokens}")
    logger.info(f"  - learning_rate: {lr}")
    logger.info(f"  - total_steps: {total_steps}")
    logger.info(f"  - batch_size: {batch_size}")
    logger.info(f"  - max_response_length: {max_response_length}")
    logger.info(f"  - temperature: {temperature}")
    logger.info(f"  - eval_every: {eval_every}")
    logger.info(f"  - save_dir: {save_dir}")
    logger.info(f"  - max_eval_samples: {max_eval_samples if max_eval_samples else 'all'}")
    logger.info(f"  - max_doc_tokens: {max_doc_tokens}")
    
    # Device selection
    logger.info("Detecting compute device...")
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.bfloat16
        logger.info(f"  - CUDA available: {torch.cuda.get_device_name(0)}")
        logger.info(f"  - CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    elif torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float32
        logger.info("  - Apple MPS available")
    else:
        device = "cpu"
        dtype = torch.float32
        logger.info("  - Using CPU (no GPU detected)")
    
    logger.info(f"Selected device: {device}, dtype: {dtype}")

    # Load models
    logger.info("-" * 50)
    logger.info(f"Loading tokenizer from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    logger.info(f"  - Tokenizer loaded, vocab_size: {tokenizer.vocab_size}")
    
    # Fix tokenizer name_or_path for local models (cartridges expects HF names)
    model_lower = model_path.lower()
    original_name = tokenizer.name_or_path
    if "llama-3.2-3b-instruct" in model_lower:
        tokenizer.name_or_path = "meta-llama/Llama-3.2-3B-Instruct"
    elif "llama-3.2-1b-instruct" in model_lower:
        tokenizer.name_or_path = "meta-llama/Llama-3.2-1B-Instruct"
    elif "llama-3.1-8b-instruct" in model_lower:
        tokenizer.name_or_path = "meta-llama/Llama-3.1-8B-Instruct"
    logger.info(f"  - Tokenizer name_or_path: {original_name} -> {tokenizer.name_or_path}")
    
    # Teacher model (standard HF)
    logger.info("-" * 50)
    logger.info("Loading teacher model (standard HuggingFace)...")
    teacher_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype).to(device).eval()
    logger.info(f"  - Teacher model loaded successfully")
    logger.info(f"  - Model config: {teacher_model.config.num_hidden_layers} layers, {teacher_model.config.hidden_size} hidden_size")
    
    # Student model (FlexLlama for cache-augmented forward)
    logger.info("Loading flex model (FlexLlamaForCausalLM)...")
    flex_model = FlexLlamaForCausalLM.from_pretrained(model_path, torch_dtype=dtype).to(device).eval()
    for p in flex_model.parameters():
        p.requires_grad = False
    logger.info(f"  - FlexLlama model loaded, all parameters frozen")

    attn_config = AttnConfig(
        n_layers=flex_model.config.num_hidden_layers,
        n_heads=flex_model.config.num_key_value_heads,
        head_dim=flex_model.config.hidden_size // flex_model.config.num_attention_heads,
    )
    logger.info(f"  - AttnConfig: {attn_config}")

    # Load training data FIRST (needed to know which patients we're training on)
    logger.info("-" * 50)
    logger.info(f"Loading training data from: {train_parquet}")
    if not os.path.exists(train_parquet):
        logger.error(f"Training parquet file not found: {train_parquet}")
        raise FileNotFoundError(f"Training parquet file not found: {train_parquet}")
    train_df = pd.read_parquet(train_parquet)
    # CRITICAL: Shuffle to mix patients - data is sorted by patient_id!
    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
    logger.info(f"  - Loaded {len(train_df)} training prompts (SHUFFLED)")
    logger.info(f"  - Columns: {list(train_df.columns)}")
    logger.info(f"  - Patient distribution in first 100: {train_df['patient_id'].head(100).value_counts().to_dict()}")

    # Load patient documents BEFORE cache initialization (following Cartridges paper)
    logger.info("-" * 50)
    logger.info("Downloading LongHealth benchmark data...")
    lh_data = requests.get(
        "https://raw.githubusercontent.com/kbressem/LongHealth/refs/heads/main/data/benchmark_v5.json"
    ).json()
    patient_doc_ids = {}
    patient_doc_texts = {}
    total_doc_tokens = 0
    for pid, patient in lh_data.items():
        doc_text = "\n\n".join(f"--- {did} ---\n{txt}" for did, txt in patient["texts"].items())
        patient_doc_texts[pid] = doc_text
        patient_doc_ids[pid] = tokenizer.encode(doc_text, add_special_tokens=False)
        total_doc_tokens += len(patient_doc_ids[pid])
    logger.info(f"  - Loaded {len(patient_doc_ids)} patient documents")
    logger.info(f"  - Total document tokens: {total_doc_tokens:,}")
    logger.info(f"  - Average tokens per document: {total_doc_tokens // len(patient_doc_ids):,}")

    # ==========================================================================
    # CACHE INITIALIZATION (following Cartridges paper arXiv:2506.06266)
    # Initialize from ACTUAL patient documents, not random Wikipedia text!
    # The paper initializes from the first p tokens of the target corpus.
    # For multi-patient training, we concatenate samples from each patient.
    # ==========================================================================
    logger.info("-" * 50)
    logger.info(f"Initializing trainable cache with {num_tokens} tokens...")
    logger.info("  - Using FIRST P TOKENS initialization (per Cartridges paper Section 3)")
    
    # PAPER METHOD: Concatenate all patient documents, then take FIRST p tokens
    # "We initialize the trainable cache from the first p tokens of the target corpus"
    training_patients = sorted(train_df["patient_id"].unique())
    logger.info(f"  - Training patients: {training_patients}")
    
    # Concatenate ALL patient documents into one corpus (like paper does)
    corpus_parts = []
    for pid in training_patients:
        if pid in patient_doc_texts:
            corpus_parts.append(patient_doc_texts[pid])
    
    full_corpus = "\n\n".join(corpus_parts)
    logger.info(f"  - Full corpus: {len(full_corpus):,} chars from {len(corpus_parts)} patients")
    
    # Take just the FIRST portion (will be truncated to num_tokens by KVFromText)
    # No need to pre-truncate - KVFromText handles max_tokens
    init_text = full_corpus
    logger.info(f"  - Init will use first {num_tokens} tokens of concatenated corpus")
    
    # Write to temp file for KVFromText initializer
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(init_text)
        init_text_path = f.name
    logger.info(f"  - Wrote initialization text to: {init_text_path}")
    
    # Initialize cache from patient documents (NOT random Wikipedia!)
    initializer = KVFromText.Config(
        max_tokens=num_tokens,
        text_source=init_text_path,
    ).instantiate()
    cache = initializer.initialize_kv_cache(
        tokenizer=tokenizer, model=flex_model, attn_config=attn_config,
    ).to(device)
    
    # Clean up temp file
    os.unlink(init_text_path)
    
    logger.info(f"  - Cache initialized: {cache.num_cartridge_tokens()} cartridge tokens")
    logger.info(f"  - Trainable params in cache: {sum(p.numel() for p in cache.parameters() if p.requires_grad):,}")
    logger.info("  - ✓ Cache initialized from PATIENT DOCUMENTS (not random text)")

    wrapped_model = CacheAndModel(cache, flex_model)
    optimizer = optim.Adam(cache.parameters(), lr=lr)
    logger.info(f"  - Optimizer: Adam(lr={lr})")

    # Load evaluation questions (patients 1-10 for test set)
    logger.info("-" * 50)
    logger.info("Loading evaluation questions...")
    EVAL_PATIENT_IDS = {f"patient_{i:02d}" for i in range(1, 11)}
    eval_questions = load_eval_questions(lh_data, EVAL_PATIENT_IDS)
    # CRITICAL: Shuffle eval questions to get balanced patient coverage when using max_eval_samples
    np.random.seed(42)
    np.random.shuffle(eval_questions)
    logger.info(f"  - Loaded {len(eval_questions)} evaluation questions (SHUFFLED for balanced coverage)")
    logger.info(f"  - Evaluation patients: {sorted(EVAL_PATIENT_IDS)}")

    os.makedirs(save_dir, exist_ok=True)
    logger.info(f"  - Save directory: {save_dir}")
    
    # Add file handler for persistent logging
    log_file_path = os.path.join(save_dir, "training.log")
    file_handler = logging.FileHandler(log_file_path, mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s'))
    logger.addHandler(file_handler)
    logger.info(f"  - Log file: {log_file_path}")
    
    # Track evaluation results
    eval_results = []

    logger.info("=" * 70)
    logger.info("TRAINING INITIALIZATION COMPLETE")
    logger.info("=" * 70)
    logger.info(f"  Model: {model_path}")
    logger.info(f"  Cache: {num_tokens} tokens, LR={lr}")
    logger.info(f"  Batch: {batch_size} samples")
    logger.info(f"  Steps: {total_steps}")
    logger.info("=" * 70)

    prompt_idx = 0
    training_start_time = time.time()
    
    # Token tracking for on-policy vs off-policy comparison
    total_tokens = 0
    token_history = []

    # ==========================================================================
    # STEP 0 EVALUATION (before any training)
    # ==========================================================================
    logger.info("=" * 70)
    logger.info("STEP 0 EVALUATION (before training)")
    logger.info("=" * 70)
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
    logger.info(f"  Step 0 baseline accuracy: {eval_result['accuracy']:.1f}%")
    
    # Save initial results (crash protection)
    results_path = os.path.join(save_dir, "onpolicy_results.json")
    with open(results_path, "w") as f:
        json.dump({
            "eval_results": eval_results,
            "token_history": token_history,
            "config": {"mode": "on-policy", "total_steps": total_steps, "batch_size": batch_size, "lr": lr}
        }, f, indent=2)
    logger.info(f"  - Results saved: {results_path}")
    
    # Save step-0 checkpoint
    ckpt_path = os.path.join(save_dir, "step-0", "cartridge.pt")
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    cache.save(ckpt_path)
    logger.info(f"  - Checkpoint saved: {ckpt_path}")
    
    # Memory cleanup after step 0
    cleanup_memory(device)

    logger.info("=" * 70)
    logger.info("Starting training loop...")
    logger.info("=" * 70)
    
    for step in range(1, total_steps + 1):
        # Pre-step memory cleanup (important for MPS)
        cleanup_memory(device)
        
        step_t0 = time.time()
        logger.info(f"--- STEP {step}/{total_steps} ---")

        # 1. Sample prompts
        logger.debug(f"Step {step}: Sampling {batch_size} prompts...")
        prompts_data = []
        for _ in range(batch_size):
            row = train_df.iloc[prompt_idx % len(train_df)]
            prompt_idx += 1
            messages = row["prompt"]
            if isinstance(messages, str):
                messages = json.loads(messages)
            prompts_data.append({"messages": messages, "patient_id": row["patient_id"]})

        prompt_ids_list = [
            tokenizer.apply_chat_template(p["messages"], add_generation_prompt=True, tokenize=True)
            for p in prompts_data
        ]
        avg_prompt_len = sum(len(p) for p in prompt_ids_list) / len(prompt_ids_list)
        logger.debug(f"  - Sampled {len(prompts_data)} prompts, avg length: {avg_prompt_len:.0f} tokens")

        # 2. Generate responses locally
        logger.debug(f"Step {step}: Generating responses locally...")
        gen_t0 = time.time()
        response_ids_list = generate_batch_local(
            flex_model, tokenizer, cache, prompt_ids_list, max_response_length, temperature, device
        )
        gen_elapsed = time.time() - gen_t0
        logger.debug(f"  - Generation complete in {gen_elapsed:.2f}s")

        # Filter empty and track generation tokens
        valid = [(i, p, r) for i, (p, r) in enumerate(zip(prompt_ids_list, response_ids_list)) if r]
        if not valid:
            logger.warning(f"Step {step}: all generations failed, skipping")
            continue
        
        # Track generation tokens (on-policy specific cost)
        gen_tokens = sum(len(p) + len(r) for _, p, r in valid)
        total_tokens += gen_tokens
        logger.debug(f"  - {len(valid)}/{len(response_ids_list)} valid responses, gen_tokens={gen_tokens:,}")

        # 3. Compute teacher top-k
        logger.debug(f"Step {step}: Computing teacher top-k logprobs...")
        teacher_t0 = time.time()
        by_patient = {}
        for idx, prompt_ids, resp_ids in valid:
            pid = prompts_data[idx]["patient_id"]
            by_patient.setdefault(pid, []).append((prompt_ids, resp_ids))
        logger.debug(f"  - Grouped by {len(by_patient)} patients")

        all_samples = []
        teacher_tokens = 0
        for pid, items in by_patient.items():
            doc_ids = patient_doc_ids.get(pid, [])
            # Truncate document to max_doc_tokens to avoid OOM on MPS
            if len(doc_ids) > max_doc_tokens:
                doc_ids = doc_ids[:max_doc_tokens]
                logger.debug(f"  - Truncated {pid} doc from {len(patient_doc_ids.get(pid, []))} to {max_doc_tokens} tokens")
            p_list = [x[0] for x in items]
            r_list = [x[1] for x in items]
            
            # Track teacher tokens (doc + prompt + response for each sample)
            for p, r in zip(p_list, r_list):
                teacher_tokens += len(doc_ids) + len(p) + len(r)
            
            samples = compute_teacher_topk(teacher_model, tokenizer, doc_ids, p_list, r_list, device)
            all_samples.extend(samples)
        
        total_tokens += teacher_tokens
        teacher_elapsed = time.time() - teacher_t0
        logger.debug(f"  - Teacher top-k computed for {len(all_samples)} samples in {teacher_elapsed:.2f}s, teacher_tokens={teacher_tokens:,}")

        # 4. Train
        logger.debug(f"Step {step}: Training on samples...")
        train_t0 = time.time()
        np.random.shuffle(all_samples)

        accum_loss = 0.0
        n_batches = 0
        train_tokens = 0
        optimizer.zero_grad()

        i = 0
        while i < len(all_samples):
            batch = build_packed_batch(all_samples[i:])
            packed_count = batch["element_ids"].max().item() + 1
            i += max(packed_count, 1)
            
            # Track training tokens
            train_tokens += len(batch["input_ids"])

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

        total_tokens += train_tokens
        optimizer.step()
        optimizer.zero_grad()
        train_elapsed = time.time() - train_t0
        step_elapsed = time.time() - step_t0

        avg_loss = accum_loss / max(n_batches, 1)
        logger.info(f"Step {step}: loss={avg_loss:.4f} | gen={gen_elapsed:.1f}s | teacher={teacher_elapsed:.1f}s | train={train_elapsed:.1f}s | total={step_elapsed:.1f}s | tokens={total_tokens:,}")

        # 5. Evaluate on test set (every eval_every steps or final step)
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
            
            # Save intermediate results after each eval (crash protection)
            results_path = os.path.join(save_dir, "onpolicy_results.json")
            with open(results_path, "w") as f:
                json.dump({
                    "eval_results": eval_results,
                    "token_history": token_history,
                    "config": {
                        "mode": "on-policy",
                        "total_steps": total_steps,
                        "batch_size": batch_size,
                        "lr": lr,
                    }
                }, f, indent=2)
            logger.info(f"  - Results saved: {results_path}")
        
        # 6. Save checkpoint (every save_every steps or final step)
        if step % save_every == 0 or step == total_steps:
            ckpt_path = os.path.join(save_dir, f"step-{step}", "cartridge.pt")
            os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
            cache.save(ckpt_path)
            logger.info(f"  - Checkpoint saved: {ckpt_path}")
        
        # 7. Aggressive memory cleanup after step
        del all_samples, by_patient
        cleanup_memory(device)
        logger.debug(f"  - Memory cleaned up after step {step}")

    total_training_time = time.time() - training_start_time
    
    # ==========================================================================
    # TRAINING SUMMARY
    # ==========================================================================
    logger.info("=" * 70)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 70)
    logger.info(f"  Total steps: {total_steps}")
    logger.info(f"  Total time: {total_training_time:.1f}s ({total_training_time/60:.1f} minutes)")
    logger.info(f"  Avg time per step: {total_training_time/total_steps:.1f}s")
    
    # Print evaluation summary
    logger.info("-" * 50)
    logger.info("EVALUATION SUMMARY:")
    logger.info("-" * 50)
    for r in eval_results:
        bar = "█" * int(r["accuracy"] / 5)  # Scale bar to fit
        logger.info(f"  Step {r['step']:3d}: {r['accuracy']:5.1f}% {bar}")
    
    # Find best step
    best_result = max(eval_results, key=lambda x: x["accuracy"])
    logger.info("-" * 50)
    logger.info(f"  BEST: Step {best_result['step']} with {best_result['accuracy']:.1f}% accuracy")
    
    # Save evaluation results and token history to JSON
    results_path = os.path.join(save_dir, "onpolicy_results.json")
    with open(results_path, "w") as f:
        json.dump({
            "eval_results": eval_results,
            "token_history": token_history,
            "config": {
                "mode": "on-policy",
                "total_steps": total_steps,
                "batch_size": batch_size,
                "lr": lr,
            }
        }, f, indent=2)
    logger.info(f"  - Results saved: {results_path}")
    logger.info(f"  Total tokens processed: {total_tokens:,}")
    
    logger.info("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Local cartridge training - no Tokasaurus, no Modal, pure MPS/CUDA",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--model", default="/Users/csuda/cartridges-workspace/model/Llama-3.2-3B-Instruct",
                        help="Path to the pretrained model")
    parser.add_argument("--train-parquet", default="data/on_policy/train.parquet",
                        help="Path to the training parquet file")
    parser.add_argument("--num-tokens", type=int, default=512,
                        help="Number of tokens in the trainable cache")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate for the cache optimizer (default was 0.02, now 0.001)")
    parser.add_argument("--total-steps", type=int, default=100,
                        help="Total number of training steps")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size (number of prompts per step)")
    parser.add_argument("--eval-every", type=int, default=50,
                        help="Evaluate every N steps")
    parser.add_argument("--save-every", type=int, default=50,
                        help="Save checkpoint every N steps")
    parser.add_argument("--save-dir", default="./local_checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug logging")
    parser.add_argument("--max-eval-samples", type=int, default=None,
                        help="Limit evaluation to N samples (default: all 200)")
    parser.add_argument("--max-doc-tokens", type=int, default=4096,
                        help="Max document tokens for teacher (default: 4096, reduce for MPS memory)")
    args = parser.parse_args()

    # Adjust logging level based on --debug flag
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
        logger.info("Debug logging enabled")
    else:
        # Set debug loggers to INFO to reduce verbosity
        logging.getLogger().setLevel(logging.INFO)
    
    logger.info("Command line arguments:")
    for arg, value in vars(args).items():
        logger.info(f"  --{arg.replace('_', '-')}: {value}")

    try:
        train(
            model_path=args.model,
            train_parquet=args.train_parquet,
            num_tokens=args.num_tokens,
            lr=args.lr,
            total_steps=args.total_steps,
            batch_size=args.batch_size,
            eval_every=args.eval_every,
            save_every=args.save_every,
            save_dir=args.save_dir,
            max_eval_samples=args.max_eval_samples,
            max_doc_tokens=args.max_doc_tokens,
        )
    except Exception as e:
        logger.exception(f"Training failed with error: {e}")
        raise
