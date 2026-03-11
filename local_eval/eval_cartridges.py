#!/usr/bin/env python3
"""
Local eval of step-0, step-1, step-2 cartridges on LongHealth test set.

Usage:
    cd /Users/csuda/cartridges-workspace
    pip install -e cartridges
    python local_eval/eval_cartridges.py
"""

import re
import sys
import time
import json
from pathlib import Path

import requests
import torch
from transformers import AutoTokenizer

# ── paths ─────────────────────────────────────────────────────────────────────
WORKSPACE = Path(__file__).parent.parent
MODEL_PATH = str(WORKSPACE / "model" / "Llama-3.2-3B-Instruct")
CARTRIDGE_DIR = WORKSPACE / "local_eval" / "cartridges"
STEPS = [15]

# ── device ────────────────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    device = "mps"
    dtype  = torch.float32
elif torch.cuda.is_available():
    device = "cuda"
    dtype  = torch.bfloat16
else:
    device = "cpu"
    dtype  = torch.float32
print(f"Device: {device}, dtype: {dtype}")

# ── load model ────────────────────────────────────────────────────────────────
sys.path.insert(0, str(WORKSPACE / "cartridges"))
from cartridges.cache import AttnConfig, TrainableCache
from cartridges.models.llama.modeling_llama import FlexLlamaForCausalLM
from cartridges.generation import flex_generate

print(f"Loading model from {MODEL_PATH} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.name_or_path = "meta-llama/Llama-3.2-3B-Instruct"

flex_model = FlexLlamaForCausalLM.from_pretrained(
    MODEL_PATH, torch_dtype=dtype
).to(device).eval()
for p in flex_model.parameters():
    p.requires_grad = False
print("Model loaded.")

# ── load LongHealth questions (patients 1-10) ─────────────────────────────────
EVAL_PATIENT_IDS = {f"patient_{i:02d}" for i in range(1, 11)}

print("Fetching LongHealth benchmark...")
data = requests.get(
    "https://raw.githubusercontent.com/kbressem/LongHealth/refs/heads/main/data/benchmark_v5.json"
).json()

questions = []
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
        questions.append({
            "prompt": prompt,
            "correct": answer_map.get(q["correct"], "?"),
        })
print(f"Loaded {len(questions)} questions across {len(EVAL_PATIENT_IDS)} patients.")

# ── helper: load cartridge from .pt ──────────────────────────────────────────
def load_cartridge(ckpt_path: str, device: str) -> TrainableCache:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    # normalise key names (frozen_keys / fixed_keys both accepted)
    if "fixed_keys" in ckpt and "frozen_keys" not in ckpt:
        ckpt["frozen_keys"]   = ckpt.pop("fixed_keys")
        ckpt["frozen_values"] = ckpt.pop("fixed_values")
    tk = ckpt["trainable_keys"]
    fk = ckpt["frozen_keys"]
    nl  = len(tk)
    nh  = tk[0].size(1)
    hd  = tk[0].size(3)
    nf  = fk[0].size(2) if fk and len(fk) > 0 else 0
    if nf > 0:
        ik = [torch.cat([fk[i], tk[i]], dim=2).contiguous() for i in range(nl)]
        iv = [torch.cat([ckpt["frozen_values"][i], ckpt["trainable_values"][i]], dim=2).contiguous() for i in range(nl)]
    else:
        ik = [tk[i] for i in range(nl)]
        iv = [ckpt["trainable_values"][i] for i in range(nl)]
    cache = TrainableCache(
        config=AttnConfig(n_layers=nl, n_heads=nh, head_dim=hd),
        init_keys=ik, init_values=iv, num_frozen_tokens=nf,
    ).to(device)
    del ckpt
    return cache

# ── helper: extract answer letter ────────────────────────────────────────────
def extract_answer(text: str) -> str:
    m = re.search(r"\b([A-E])\b", text.strip()[:20])
    return m.group(1) if m else "?"

# ── eval one checkpoint ───────────────────────────────────────────────────────
def eval_step(step: int) -> dict:
    ckpt_path = str(CARTRIDGE_DIR / f"step-{step}" / "cartridge.pt")
    print(f"\n{'='*60}")
    print(f"  Evaluating step-{step}  ({ckpt_path})")
    print(f"{'='*60}")

    cache = load_cartridge(ckpt_path, device)

    correct_count = 0
    t0 = time.time()
    for qi, q in enumerate(questions):
        ids = tokenizer.encode(q["prompt"])
        input_ids   = torch.tensor(ids, dtype=torch.long, device=device)
        seq_ids     = torch.zeros_like(input_ids)
        position_ids = torch.arange(len(ids), dtype=torch.long, device=device)

        cache.clear()
        with torch.no_grad():
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
        gen_text   = tokenizer.decode(gen_tokens, skip_special_tokens=True)
        pred       = extract_answer(gen_text)

        if qi < 3:
            print(f"  Q{qi}: gen='{gen_text[:60]}' pred={pred} correct={q['correct']}")
        if pred == q["correct"]:
            correct_count += 1

        # progress every 20 questions
        if (qi + 1) % 20 == 0:
            print(f"  [{qi+1}/{len(questions)}] running acc={correct_count/(qi+1)*100:.1f}%")

    elapsed = time.time() - t0
    acc = correct_count / len(questions) * 100
    print(f"\n  RESULT step-{step}: {correct_count}/{len(questions)} = {acc:.1f}%  [{elapsed:.0f}s]")
    del cache
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()
    return {"step": step, "correct": correct_count, "total": len(questions), "acc": round(acc, 2)}

# ── run ───────────────────────────────────────────────────────────────────────
results = []
for step in STEPS:
    r = eval_step(step)
    results.append(r)

print("\n" + "="*60)
print("  SUMMARY")
print("="*60)
for r in results:
    bar = "█" * int(r["acc"] / 2)
    print(f"  step-{r['step']:3d}  {r['acc']:5.1f}%  {bar}")

out_path = WORKSPACE / "local_eval" / "results.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {out_path}")
