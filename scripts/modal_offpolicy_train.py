"""
Off-policy baseline replication on Modal.
Runs the paper's context-distillation training on a single GPU.
No WandB — eval scores saved to /results/eval_scores.json on a Modal Volume.

Usage:
    modal run scripts/modal_offpolicy_train.py
"""

import modal

# ---------------------------------------------------------------------------
# Image
# ---------------------------------------------------------------------------
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.12"
    )
    .apt_install("git")
    .env({
        "CUDA_HOME": "/usr/local/cuda",
        "WANDB_MODE": "disabled",            # wandb imported but neutered
        "CARTRIDGES_DIR": "/opt/cartridges",  # repo root (not package dir)
        "CARTRIDGES_OUTPUT_DIR": "/results",
    })
    .pip_install("torch==2.6.0", "packaging", "numpy")
    .run_commands(
        # Pre-built flash-attn wheel (not strictly required, but some codepaths import it)
        "pip install https://github.com/Dao-AILab/flash-attention/releases/download/"
        "v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl",
    )
    .run_commands(
        "git clone https://github.com/HazyResearch/cartridges.git /opt/cartridges "
        "&& pip install -e /opt/cartridges"
    )
)

volume = modal.Volume.from_name("offpolicy-results", create_if_missing=True)
app = modal.App("offpolicy-baseline", image=image)


# ---------------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------------
@app.function(
    gpu="A100-80GB",
    timeout=43200,       # 12 hours
    volumes={"/results": volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    min_containers=0,
    max_containers=1,
)
def train():
    """Run off-policy cartridge training (paper baseline) and save eval curve."""
    import os, sys, json, time

    os.environ.setdefault("CARTRIDGES_DIR", "/opt/cartridges")
    os.environ.setdefault("CARTRIDGES_OUTPUT_DIR", "/results")
    os.environ["WANDB_MODE"] = "disabled"
    sys.path.insert(0, "/opt/cartridges")

    # ------------------------------------------------------------------
    # 1. Monkey-patch evaluate_generations to save scores to JSON
    # ------------------------------------------------------------------
    import cartridges.train as train_mod

    _orig_eval_gen = train_mod.evaluate_generations
    _eval_log: list[dict] = []
    _log_path = "/results/eval_scores.json"

    GLOBAL_BATCH_SIZE = 32
    PACKED_SEQ_LEN = 2048

    def _saving_eval_gen(*args, **kwargs):
        """Wrapper: runs eval, then appends scores to JSON file."""
        # Force log_to_wandb=False so no wandb calls happen
        kwargs["log_to_wandb"] = False
        optimizer_step = kwargs.get(
            "optimizer_step", args[4] if len(args) > 4 else 0
        )

        results = _orig_eval_gen(*args, **kwargs)

        if results:
            import pandas as pd

            df = pd.DataFrame(results)
            score_cols = [c for c in df.columns if c.endswith("score")]
            scores = {c: round(float(df[c].mean()), 4) for c in score_cols}

            entry = {
                "optimizer_step": int(optimizer_step),
                "total_tokens": int(optimizer_step) * GLOBAL_BATCH_SIZE * PACKED_SEQ_LEN,
                "total_packed_seqs": int(optimizer_step) * GLOBAL_BATCH_SIZE,
                "scores": scores,
                "num_eval_questions": len(results),
                "timestamp": time.time(),
            }
            _eval_log.append(entry)

            with open(_log_path, "w") as f:
                json.dump(
                    {"method": "off_policy", "evals": _eval_log}, f, indent=2
                )

            print(
                f"\n{'='*60}\n"
                f"  EVAL @ step {optimizer_step}  "
                f"({entry['total_tokens']:,} tokens)\n"
                f"  {scores}\n"
                f"{'='*60}\n"
            )

        return results

    train_mod.evaluate_generations = _saving_eval_gen

    # ------------------------------------------------------------------
    # 2. Build the training config (matches longhealth_train.py exactly,
    #    except wandb=None)
    # ------------------------------------------------------------------
    import pydrantic
    from pydrantic.variables import FormatStringVariable

    from cartridges.initialization import KVFromText
    from cartridges.train import GenerationEvalConfig, TrainConfig
    from cartridges.models.config import HFModelConfig
    from cartridges.datasets import TrainDataset, DataSource
    from cartridges.data.longhealth.evals import (
        LongHealthMultipleChoiceGenerateDataset,
    )
    from cartridges.models.llama.modeling_llama import FlexLlamaForCausalLM

    NUM_TOKENS = 512
    NUM_PATIENTS = 10
    patient_ids = [f"patient_{idx:02d}" for idx in range(1, NUM_PATIENTS + 1)]

    config = TrainConfig(
        model=HFModelConfig(
            pretrained_model_name_or_path="meta-llama/Llama-3.2-3B-Instruct",
            model_cls=FlexLlamaForCausalLM,
        ),
        kv_cache_initializer=KVFromText.Config(max_tokens=NUM_TOKENS),
        lr=2e-2,
        epochs=2,
        global_batch_size=GLOBAL_BATCH_SIZE,
        dataset=TrainDataset.Config(
            data_sources=[
                DataSource(path=s, type="hf")
                for s in [
                    "hazyresearch/m07d11_longhealth_synthesize_llama-3.2-3b_p10_n65536-0",
                    "hazyresearch/m07d11_longhealth_synthesize_llama-3.2-3b_p10_n65536-1",
                    "hazyresearch/m07d11_longhealth_synthesize_llama-3.2-3b_p10_n65536-2",
                ]
            ],
            top_k_logits=20,
            packed_seq_length=PACKED_SEQ_LEN,
            packing_mode="truncate",
        ),
        save_every_n_steps=512,
        generate_eval_every_n_steps=128,
        generate_evals=[
            GenerationEvalConfig(
                dataset=LongHealthMultipleChoiceGenerateDataset.Config(
                    patient_ids=patient_ids,
                ),
                name_for_wandb=f"longhealth_p{NUM_PATIENTS}",
                generate_max_new_tokens=512,
                batch_size=32,
                temperature=0.3,
            )
        ],
        distributed_backend="gloo",
        wandb=None,  # ← no wandb
        output_dir="/results",
        name=FormatStringVariable(
            "longhealth_offpolicy_lr{lr}_toks{kv_cache_initializer.max_tokens}"
        ),
    )

    # ------------------------------------------------------------------
    # 3. Run
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Off-policy baseline — context distillation")
    print(f"  Model : meta-llama/Llama-3.2-3B-Instruct")
    print(f"  Tokens: {NUM_TOKENS}  LR: 0.02  Batch: {GLOBAL_BATCH_SIZE}")
    print(f"  Eval every {config.generate_eval_every_n_steps} steps")
    print(f"  Save every {config.save_every_n_steps} steps")
    print("=" * 60)

    sys.argv = ["offpolicy_train"]  # minimal argv for pydrantic
    pydrantic.main(config)

    # ------------------------------------------------------------------
    # 4. Summary
    # ------------------------------------------------------------------
    if _eval_log:
        best = max(_eval_log, key=lambda e: max(e["scores"].values()))
        print("\n" + "=" * 60)
        print("Training complete!")
        print(f"  Best eval: step {best['optimizer_step']}  {best['scores']}")
        print(f"  Eval log : {_log_path}")
        print(f"  Results  : /results/")
        print("=" * 60)


@app.local_entrypoint()
def main():
    train.remote()
