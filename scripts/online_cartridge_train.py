#!/usr/bin/env python3
"""Optimized on-policy cartridge training — no Ray, no PPO, no FSDP.

Reuses the cartridges training loop directly with online data generation.
Each step:
  1. Sample prompts from training data
  2. Generate responses via Tokasaurus HTTP
  3. Compute teacher top-k logprobs (model + full patient docs as KV prefix)
  4. Forward through CacheAndModel → top-k CE loss → backward → optimizer step
  5. Save cache checkpoint for Tokasaurus (next step generates with updated cache)

This matches off-policy's training computation exactly, adding only the
irreducible costs of online generation and teacher inference.
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

import aiohttp
import asyncio
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from cartridges.cache import AttnConfig, TrainableCache
from cartridges.initialization import KVFromText
from cartridges.models.llama.modeling_llama import FlexLlamaForCausalLM
from cartridges.train import CacheAndModel


# ---------------------------------------------------------------------------
# Tokasaurus HTTP client (async, batched)
# ---------------------------------------------------------------------------

async def _generate_one(session, url, prompt_ids, max_tokens, temperature, cartridges):
    """Single Tokasaurus request."""
    payload = {
        "model": "default",
        "prompt": prompt_ids,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "logprobs_in_fingerprint": True,
    }
    if cartridges:
        payload["cartridges"] = cartridges

    for attempt in range(5):
        try:
            async with session.post(
                f"{url}/custom/cartridge/completions",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=120),
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    # Token IDs are in system_fingerprint.completion_ids[0]
                    import json as _json
                    fp = data.get("system_fingerprint", "{}")
                    if isinstance(fp, str):
                        fp = _json.loads(fp)
                    completion_ids = fp.get("completion_ids", [[]])[0]
                    tokens = [int(t) for t in completion_ids]
                    return tokens
                else:
                    text = await resp.text()
                    print(f"  [toka] HTTP {resp.status} (attempt {attempt+1}/5): {text[:100]}")
        except Exception as e:
            print(f"  [toka] Request failed (attempt {attempt+1}/5): {e}")
        await asyncio.sleep(2 ** attempt)
    return []


async def generate_batch(url, prompt_ids_list, max_tokens, temperature, cartridges, max_concurrent=32):
    """Generate responses for a batch of prompts, up to max_concurrent at once."""
    connector = aiohttp.TCPConnector(limit=max_concurrent)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [
            _generate_one(session, url, pids, max_tokens, temperature, cartridges)
            for pids in prompt_ids_list
        ]
        return await asyncio.gather(*tasks)


# ---------------------------------------------------------------------------
# Teacher: compute top-k logprobs with patient document prefix
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_teacher_topk(
    model, tokenizer, doc_ids, prompt_ids_list, response_ids_list,
    top_k=20, micro_batch=4,
):
    """Compute teacher top-k logprobs for responses given full document context.

    Uses prefix KV cache optimization: compute doc KV once, reuse for all samples.

    Returns per-sample lists of (topk_token_ids, topk_logprobs, topk_token_idxs)
    in the same format as off-policy training data.
    """
    device = next(model.parameters()).device
    device_str = str(device) if isinstance(device, torch.device) else device

    # Compute doc KV cache once
    doc_t = torch.tensor(doc_ids, dtype=torch.long, device=device).unsqueeze(0)
    use_autocast = device_str.startswith("cuda")
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_autocast):
        doc_out = model(doc_t, use_cache=True)
    doc_kv = doc_out.past_key_values
    doc_len = len(doc_ids)

    results = []  # list of (topk_ids, topk_lps, topk_idxs) per sample

    for mb_start in range(0, len(prompt_ids_list), micro_batch):
        mb_prompts = prompt_ids_list[mb_start:mb_start + micro_batch]
        mb_responses = response_ids_list[mb_start:mb_start + micro_batch]
        mb_size = len(mb_prompts)

        # Build sequences: prompt + response (doc is in KV cache)
        sequences = [p + r for p, r in zip(mb_prompts, mb_responses)]
        max_len = max(len(s) for s in sequences)

        # Left-pad to max_len
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

        # Expand doc KV for this micro-batch
        from transformers.cache_utils import DynamicCache
        expanded_kv = DynamicCache()
        # Handle both old (.key_cache list) and new (.layers[i].keys) cache APIs
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
                # New transformers: use update() method
                layer_idx = len(expanded_kv)
                expanded_kv.update(ek, ev, layer_idx)

        doc_prefix_mask = torch.ones(mb_size, doc_len, dtype=torch.long, device=device)
        full_mask = torch.cat([doc_prefix_mask, batch_mask], dim=1)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_autocast):
            output = model(
                input_ids=batch_ids,
                attention_mask=full_mask,
                position_ids=batch_pos,
                past_key_values=expanded_kv,
                use_cache=False,
            )

        logits = output.logits  # (mb, max_len, vocab)

        # Extract top-k logprobs for response tokens
        for j in range(mb_size):
            seq = sequences[j]
            pad_offset = max_len - len(seq)
            prompt_len = len(mb_prompts[j])
            response_len = len(mb_responses[j])

            # Response positions in the padded tensor
            resp_start = pad_offset + prompt_len
            resp_logits = logits[j, resp_start - 1 : resp_start + response_len - 1]  # shifted by 1

            log_probs = F.log_softmax(resp_logits.float(), dim=-1)
            topk_lps, topk_ids = torch.topk(log_probs, k=top_k, dim=-1)

            results.append({
                "topk_token_ids": topk_ids.cpu(),      # (response_len, top_k)
                "topk_logprobs": topk_lps.cpu(),        # (response_len, top_k)
                "response_ids": mb_responses[j],
                "prompt_ids": mb_prompts[j],
            })

    return results


# ---------------------------------------------------------------------------
# Build packed batch from on-policy data (same format as off-policy)
# ---------------------------------------------------------------------------

def build_packed_batch(samples, packed_seq_length=2048):
    """Pack multiple (prompt, response, teacher_topk) samples into a single
    sequence, matching the off-policy DatasetBatch format.

    Returns (input_ids, element_ids, position_ids, topk_token_ids, topk_logprobs, topk_token_idxs)
    all as 1D tensors ready for CacheAndModel.
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
            break  # packed enough

        all_input_ids.append(torch.tensor(seq, dtype=torch.long))
        all_element_ids.append(torch.full((seq_len,), elem_id, dtype=torch.long))
        all_position_ids.append(torch.arange(seq_len, dtype=torch.long))

        # Top-k indices: positions of response tokens in the packed sequence
        # Response starts at curr_offset + prompt_len, teacher logprobs
        # predict token at position t from logits at position t-1
        resp_idxs = torch.arange(prompt_len, prompt_len + response_len, dtype=torch.long) + curr_offset
        all_topk_idxs.append(resp_idxs.unsqueeze(1).expand(-1, s["topk_token_ids"].shape[1]))
        all_topk_ids.append(s["topk_token_ids"])
        all_topk_lps.append(s["topk_logprobs"])

        curr_offset += seq_len

    # Pad to packed_seq_length
    input_ids = torch.cat(all_input_ids)
    element_ids = torch.cat(all_element_ids)
    position_ids = torch.cat(all_position_ids)

    if len(input_ids) < packed_seq_length:
        pad_len = packed_seq_length - len(input_ids)
        input_ids = torch.cat([input_ids, torch.zeros(pad_len, dtype=torch.long)])
        element_ids = torch.cat([element_ids, torch.zeros(pad_len, dtype=torch.long)])
        position_ids = torch.cat([position_ids, torch.zeros(pad_len, dtype=torch.long)])

    topk_token_ids = torch.cat(all_topk_ids).reshape(-1)    # flatten
    topk_logprobs = torch.cat(all_topk_lps).reshape(-1)     # flatten
    topk_token_idxs = torch.cat(all_topk_idxs).reshape(-1)  # flatten

    return {
        "input_ids": input_ids,
        "element_ids": element_ids,
        "position_ids": position_ids,
        "topk_token_ids": topk_token_ids,
        "topk_logprobs": topk_logprobs,
        "topk_token_idxs": topk_token_idxs,
    }


# ---------------------------------------------------------------------------
# Inline eval — reuses the student model, no extra memory
# ---------------------------------------------------------------------------

@torch.no_grad()
def _run_inline_eval(flex_model, cache, tokenizer, ckpt_path, step, device, eval_json_path, patient_doc_ids):
    """Evaluate a cartridge checkpoint on LongHealth using the already-loaded model."""
    from cartridges.generation import flex_generate
    from cartridges.cache import AttnConfig, TrainableCache

    EVAL_PATIENT_IDS = {f"patient_{i:02d}" for i in range(1, 11)}

    # Load questions (cached after first call)
    if not hasattr(_run_inline_eval, "_questions"):
        data = requests.get(
            "https://raw.githubusercontent.com/kbressem/LongHealth/refs/heads/main/data/benchmark_v5.json"
        ).json()
        _run_inline_eval._questions = []
        for pid, patient in data.items():
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
                _run_inline_eval._questions.append({
                    "prompt": prompt,
                    "correct": answer_map.get(q["correct"], "?"),
                })
        print(f"  [eval] Loaded {len(_run_inline_eval._questions)} questions")

    questions = _run_inline_eval._questions

    # Load cache from checkpoint (to eval the saved state, not the live state)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "fixed_keys" in ckpt and "frozen_keys" not in ckpt:
        ckpt["frozen_keys"] = ckpt.pop("fixed_keys")
        ckpt["frozen_values"] = ckpt.pop("fixed_values")
    tk = ckpt["trainable_keys"]
    fk = ckpt["frozen_keys"]
    nl, nh, hd = len(tk), tk[0].size(1), tk[0].size(3)
    nf = fk[0].size(2) if fk else 0
    ik = [torch.cat([fk[i], tk[i]], dim=2).contiguous() if nf > 0 else tk[i] for i in range(nl)]
    iv = [torch.cat([ckpt["frozen_values"][i], ckpt["trainable_values"][i]], dim=2).contiguous() if nf > 0 else ckpt["trainable_values"][i] for i in range(nl)]
    eval_cache = TrainableCache(
        config=AttnConfig(n_layers=nl, n_heads=nh, head_dim=hd),
        init_keys=ik, init_values=iv, num_frozen_tokens=nf,
    ).to(device)
    del ckpt

    def extract_answer(text):
        m = re.search(r"\b([A-E])\b", text.strip()[:20])
        return m.group(1) if m else "?"

    t0 = time.time()
    correct = 0
    for qi, q in enumerate(questions):
        ids = tokenizer.encode(q["prompt"])
        input_ids = torch.tensor(ids, dtype=torch.long, device=device)
        seq_ids = torch.zeros_like(input_ids)
        position_ids = torch.arange(len(ids), dtype=torch.long, device=device)
        eval_cache.clear()
        gen_output = flex_generate(
            model=flex_model, tokenizer=tokenizer, cache=eval_cache,
            input_ids=input_ids, seq_ids=seq_ids, position_ids=position_ids,
            max_new_tokens=10, temperature=0.0,
        )
        gen_tokens = gen_output.get(0, [])
        gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
        pred = extract_answer(gen_text)
        if qi < 5:
            print(f"  Q{qi}: gen='{gen_text[:80]}' pred={pred} correct={q['correct']}")
        if pred == q["correct"]:
            correct += 1

    total = len(questions)
    acc = correct / total * 100
    elapsed = time.time() - t0
    print(f"\n  {'='*56}")
    print(f"  EVAL @ step {step}: {correct}/{total} ({acc:.1f}%) [{elapsed:.0f}s]")
    print(f"  {'='*56}\n")

    # Save results
    eval_results_path = eval_json_path
    if os.path.exists(eval_results_path):
        with open(eval_results_path) as f:
            results = json.load(f)
    else:
        results = {"method": "on_policy", "evals": []}
    results["evals"].append({
        "optimizer_step": step,
        "total_tokens": step * 256 * 250,
        "scores": {"score": round(acc / 100, 4)},
        "num_eval_questions": total,
        "correct": correct,
    })
    os.makedirs(os.path.dirname(eval_results_path), exist_ok=True)
    with open(eval_results_path, "w") as f:
        json.dump(results, f, indent=2)

    del eval_cache
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(
    model_name: str,
    tokasaurus_url: str,
    train_parquet: str,
    num_tokens: int = 512,
    num_frozen_tokens: int = 1,
    lr: float = 0.02,
    total_steps: int = 500,
    batch_size: int = 256,
    max_response_length: int = 512,
    temperature: float = 0.7,
    packed_seq_length: int = 2048,
    top_k: int = 20,
    eval_every: int = 50,
    save_dir: str = "/results/onpolicy",
    eval_json_path: str = None,  # defaults to save_dir/eval_scores.json
):
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.bfloat16
    elif torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float32  # MPS doesn't support bfloat16 well
    else:
        device = "cpu"
        dtype = torch.float32
    print(f"Device: {device}, dtype: {dtype}")

    if eval_json_path is None:
        eval_json_path = os.path.join(save_dir, "eval_scores.json")

    # ---- Load model + cache ----
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # If loading from local path, set name_or_path so cartridges lookup tables work
    if not tokenizer.name_or_path.startswith("meta-llama") and not tokenizer.name_or_path.startswith("Qwen"):
        # Map local paths to canonical HF names
        model_lower = model_name.lower()
        if "llama-3.2-3b-instruct" in model_lower:
            tokenizer.name_or_path = "meta-llama/Llama-3.2-3B-Instruct"
        elif "llama-3.2-1b-instruct" in model_lower:
            tokenizer.name_or_path = "meta-llama/Llama-3.2-1B-Instruct"
        elif "llama-3.1-8b-instruct" in model_lower:
            tokenizer.name_or_path = "meta-llama/Llama-3.1-8B-Instruct"
        print(f"  Set tokenizer.name_or_path = {tokenizer.name_or_path}")

    # Teacher model (standard HF, for prefix KV computation)
    from transformers import AutoModelForCausalLM
    teacher_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype
    ).to(device).eval()

    # Student model (FlexLlama for cache-augmented forward)
    flex_model = FlexLlamaForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype
    ).to(device).eval()

    # Freeze student model, only cache is trainable
    for p in flex_model.parameters():
        p.requires_grad = False

    attn_config = AttnConfig(
        n_layers=flex_model.config.num_hidden_layers,
        n_heads=flex_model.config.num_key_value_heads,
        head_dim=flex_model.config.hidden_size // flex_model.config.num_attention_heads,
    )

    # Initialize cache via KVFromText (same as off-policy)
    initializer = KVFromText.Config(max_tokens=num_tokens).instantiate()
    cache = initializer.initialize_kv_cache(
        tokenizer=tokenizer, model=flex_model, attn_config=attn_config,
    ).to(device)
    print(f"Cache: {num_tokens} tokens ({cache._num_trainable_tokens} trainable, {num_frozen_tokens} frozen)")

    wrapped_model = CacheAndModel(cache, flex_model)
    optimizer = optim.Adam(cache.parameters(), lr=lr)

    # ---- Load training data ----
    train_df = pd.read_parquet(train_parquet)
    print(f"Training data: {len(train_df)} prompts, {train_df.patient_id.nunique()} patients")

    # ---- Load patient documents for teacher ----
    lh_data = requests.get(
        "https://raw.githubusercontent.com/kbressem/LongHealth/refs/heads/main/data/benchmark_v5.json"
    ).json()
    patient_doc_ids = {}
    for pid, patient in lh_data.items():
        doc_text = "\n\n".join(f"--- {did} ---\n{txt}" for did, txt in patient["texts"].items())
        patient_doc_ids[pid] = tokenizer.encode(doc_text, add_special_tokens=False)
    print(f"Loaded {len(patient_doc_ids)} patient documents for teacher")

    # ---- Training loop ----
    ckpt_dir = os.path.join(save_dir, "cartridge_checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    cartridges = []  # start with no cartridge (KVFromText init)

    print(f"\n{'='*60}")
    print(f"OPTIMIZED ON-POLICY TRAINING (no Ray, no PPO, no FSDP)")
    print(f"  Model      : {model_name}")
    print(f"  Cache      : {num_tokens} tokens, LR={lr}")
    print(f"  Batch      : {batch_size} samples")
    print(f"  Steps      : {total_steps}")
    print(f"  Eval every : {eval_every} steps")
    print(f"  Tokasaurus : {tokasaurus_url}")
    print(f"{'='*60}\n")

    prompt_idx = 0  # rotating index into training data

    for step in range(total_steps):
        step_t0 = time.time()

        # ---- 1. Sample prompts ----
        prompts_data = []
        for _ in range(batch_size):
            row = train_df.iloc[prompt_idx % len(train_df)]
            prompt_idx += 1
            # prompt column is a list of chat messages [{"role": "user", "content": "..."}]
            messages = row["prompt"]
            if isinstance(messages, str):
                messages = json.loads(messages)
            prompts_data.append({
                "messages": messages,
                "patient_id": row["patient_id"],
            })

        prompt_ids_list = [
            tokenizer.apply_chat_template(
                p["messages"], add_generation_prompt=True, tokenize=True,
            )
            for p in prompts_data
        ]

        # ---- 2. Generate responses via Tokasaurus ----
        gen_t0 = time.time()
        response_ids_list = asyncio.run(generate_batch(
            tokasaurus_url, prompt_ids_list, max_response_length, temperature, cartridges,
        ))
        gen_elapsed = time.time() - gen_t0

        # Filter out empty responses
        valid = [(i, p, r) for i, (p, r) in enumerate(zip(prompt_ids_list, response_ids_list)) if r]
        if not valid:
            print(f"Step {step}: all generations failed, skipping")
            continue

        # ---- 3. Compute teacher top-k logprobs ----
        teacher_t0 = time.time()
        # Group by patient for prefix KV reuse
        by_patient = {}
        for idx, prompt_ids, resp_ids in valid:
            pid = prompts_data[idx]["patient_id"]
            by_patient.setdefault(pid, []).append((prompt_ids, resp_ids))

        all_samples = []
        for pid, items in by_patient.items():
            doc_ids = patient_doc_ids.get(pid, [])
            p_list = [x[0] for x in items]
            r_list = [x[1] for x in items]
            samples = compute_teacher_topk(
                teacher_model, tokenizer, doc_ids, p_list, r_list, top_k=top_k,
            )
            all_samples.extend(samples)
        teacher_elapsed = time.time() - teacher_t0

        # ---- 4. Pack into batches and train ----
        train_t0 = time.time()
        np.random.shuffle(all_samples)

        # Process in packed batches of packed_seq_length
        accum_loss = 0.0
        n_batches = 0
        optimizer.zero_grad()

        i = 0
        while i < len(all_samples):
            batch = build_packed_batch(all_samples[i:], packed_seq_length=packed_seq_length)
            # Count how many samples fit in this packed batch
            packed_count = batch["element_ids"].max().item() + 1
            i += max(packed_count, 1)

            with (torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16) if device == "cuda" else torch.enable_grad()):
                cache.clear()
                # FlexLlamaModel expects 1D inputs (adds batch dim internally)
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

                ce_by_token = (
                    -batch["topk_logprobs"].to(device).exp()
                    * topk_pred_logprobs
                )
                loss = ce_by_token.mean()

            loss.backward()
            accum_loss += loss.detach().item()
            n_batches += 1

        optimizer.step()
        optimizer.zero_grad()
        train_elapsed = time.time() - train_t0
        step_elapsed = time.time() - step_t0

        avg_loss = accum_loss / max(n_batches, 1)
        print(
            f"Step {step}: loss={avg_loss:.4f} "
            f"[gen={gen_elapsed:.1f}s teacher={teacher_elapsed:.1f}s train={train_elapsed:.1f}s total={step_elapsed:.1f}s] "
            f"({len(valid)}/{batch_size} valid, {n_batches} packed batches)"
        )

        # ---- 5. Save checkpoint + sync ----
        ckpt_path = os.path.join(ckpt_dir, f"cache-step{step}.pt")
        cache.save(ckpt_path)

        # Note: cartridge sync to remote Tokasaurus requires shared storage.
        # For local training, we generate without cartridge (base model).
        # On Modal (same container), local paths would work.
        # cartridges = [{"id": ckpt_path, "source": "local", "force_redownload": True}]

        # ---- 6. Eval (inline — reuses student model, no extra memory) ----
        if eval_every > 0 and step > 0 and step % eval_every == 0:
            # Free teacher to make room for eval on memory-constrained devices
            teacher_model.cpu()
            if device == "mps":
                torch.mps.empty_cache()
            _run_inline_eval(
                flex_model, cache, tokenizer, ckpt_path, step,
                device, eval_json_path, patient_doc_ids,
            )
            teacher_model.to(device)  # reload teacher for next step

    print(f"\nTraining complete. {total_steps} steps.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--tokasaurus-url", required=True)
    parser.add_argument("--train-parquet", required=True)
    parser.add_argument("--num-tokens", type=int, default=512)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--total-steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--save-dir", default="/results/onpolicy")
    args = parser.parse_args()

    train(
        model_name=args.model,
        tokasaurus_url=args.tokasaurus_url,
        train_parquet=args.train_parquet,
        num_tokens=args.num_tokens,
        lr=args.lr,
        total_steps=args.total_steps,
        batch_size=args.batch_size,
        eval_every=args.eval_every,
        save_dir=args.save_dir,
    )