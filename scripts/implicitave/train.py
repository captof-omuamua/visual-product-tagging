#!/usr/bin/env python3
"""Fine-tune Qwen3.5-4B (LoRA) on ImplicitAVE's training set, then merge it for vLLM.

Each example is the exact test-time question (implicitave.prompt: photo, title, attribute,
allowed values) and the answer as JSON. Training labels are the dataset's own (from MAVE,
machine-made, as used by the paper's fine-tuned baselines). At most --cap examples per
(domain, attribute, value), so common values don't drown out rare ones. Vision layers frozen.

Run: .venv-train/bin/python scripts/implicitave/train.py   (~1.5-2 h)
Out: outputs/iave-qwen35-4b-merged (16-bit, for vLLM), reports/implicitave/train.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "stage1"))

from train_smoke import accepted  # noqa: E402  (drops arguments a library version lacks)

from catalog_cascade import implicitave as ia  # noqa: E402


def sample(cap: int) -> list[dict]:
    """At most `cap` training rows per (domain, attribute, value), fixed seed."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in ia.load("train"):
        groups[(r["domain"], r["attribute"], r["value"])].append(r)
    rng = random.Random(0)
    rows = [r for g in groups.values() for r in rng.sample(g, min(cap, len(g)))]
    rng.shuffle(rows)
    return rows


def to_example(row: dict) -> dict:
    from PIL import Image

    image = Image.open(row["image"]).convert("RGB")
    return {"messages": [
        {"role": "user", "content": [{"type": "image", "image": image},
                                     {"type": "text", "text": ia.prompt(row)}]},
        {"role": "assistant", "content": [{"type": "text",
                                           "text": json.dumps({"value": row["value"]})}]},
    ]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--cap", type=int, default=150)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)  # 16 examples per update
    ap.add_argument("--name", default="iave-qwen35-4b")
    args = ap.parse_args()

    # Unsloth patches transformers and TRL, so it must be imported before them.
    from unsloth import FastVisionModel, is_bf16_supported  # isort: skip
    from unsloth.trainer import UnslothVisionDataCollator  # isort: skip
    import torch  # isort: skip
    from trl import SFTConfig, SFTTrainer  # isort: skip

    dataset = [to_example(r) for r in sample(args.cap)]
    print(f"{len(dataset)} training examples", flush=True)
    model, processor = FastVisionModel.from_pretrained(
        args.model, load_in_4bit=False, use_gradient_checkpointing="unsloth")
    model = FastVisionModel.get_peft_model(
        model, finetune_vision_layers=False, finetune_language_layers=True,
        finetune_attention_modules=True, finetune_mlp_modules=True,
        r=args.rank, lora_alpha=args.rank, lora_dropout=0, bias="none", random_state=3407)
    FastVisionModel.for_training(model)

    bf16 = bool(is_bf16_supported())
    config = SFTConfig(**accepted(SFTConfig, {
        "output_dir": str(ROOT / "outputs" / f"{args.name}-train"),
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "num_train_epochs": args.epochs,
        "warmup_ratio": 0.03,
        "learning_rate": 1e-4,
        "lr_scheduler_type": "cosine",
        "weight_decay": 0.01,
        "optim": "adamw_torch",
        "logging_steps": 20,
        "seed": 3407,
        "bf16": bf16,
        "fp16": not bf16,
        "report_to": "none",
        "save_strategy": "steps",
        "save_steps": 250,
        "save_total_limit": 2,
        "remove_unused_columns": False,
        "dataset_text_field": "",
        "dataset_kwargs": {"skip_prepare_dataset": True},
        "max_length": 2048,
        "max_seq_length": 2048,
    }))
    common = {"model": model, "data_collator": UnslothVisionDataCollator(model, processor),
              "train_dataset": dataset, "args": config}
    try:
        trainer = SFTTrainer(processing_class=processor, **common)
    except TypeError:  # older TRL named it `tokenizer`
        trainer = SFTTrainer(tokenizer=processor, **common)

    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    trainer.train()
    losses = [round(float(h["loss"]), 4) for h in trainer.state.log_history if "loss" in h]
    merged_dir = ROOT / "outputs" / f"{args.name}-merged"
    model.save_pretrained(str(ROOT / "outputs" / f"{args.name}-lora"))
    model.save_pretrained_merged(str(merged_dir), processor, save_method="merged_16bit")
    report = {"examples": len(dataset), "cap": args.cap, "epochs": args.epochs,
              "rank": args.rank, "train_min": round((time.time() - start) / 60, 1),
              "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
              "loss_first": losses[0] if losses else None,
              "loss_last": losses[-1] if losses else None, "losses": losses}
    out = ROOT / "reports" / "implicitave" / "train.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"Done in {report['train_min']} min, loss {report['loss_first']} -> "
          f"{report['loss_last']}; merged in {merged_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
