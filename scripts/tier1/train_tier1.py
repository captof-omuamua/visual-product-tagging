#!/usr/bin/env python3
"""Tier 1 v0: LoRA fine-tune Qwen3.5-4B on Gemma 4's labels for the train split, then merge.

Each example is the exact serving question (labelling.prompt_for, 1-3 photos at 512 px)
followed by Gemma's JSON answer. Vision layers stay frozen; LoRA on the language layers.
The setup matches the stage 1 smoke test that passed (reports/train_smoke.json).

Run: .venv-train/bin/python scripts/tier1/train_tier1.py   (~1.5-2 h; log in reports/)
Out: outputs/tier1-v0-lora (adapter), outputs/tier1-v0-merged (16-bit, for vLLM)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "stage1"))

from train_smoke import accepted  # noqa: E402  (drops arguments a library version lacks)

from catalog_cascade.labelling import prompt_for  # noqa: E402
from catalog_cascade.schema import fields_for  # noqa: E402


def load_examples(image_size: int) -> list[dict]:
    """Train-split products that Gemma answered, as chat conversations."""
    from PIL import Image

    splits = json.loads((ROOT / "data" / "splits.json").read_text())
    gemma = {r["item_id"]: r["answer"]
             for r in map(json.loads, (ROOT / "data" / "labels" / "gemma4.jsonl").open())
             if r.get("answer")}
    examples = []
    for p in map(json.loads, (ROOT / "data" / "products.jsonl").open()):
        if splits.get(p["item_id"]) != "train" or p["item_id"] not in gemma:
            continue
        answer = {f: gemma[p["item_id"]][f] for f in fields_for(p["product_type"])}
        images = []
        for i in p["images"]:
            im = Image.open(ROOT / "data" / "images" / f"{i}.jpg").convert("RGB")
            im.thumbnail((image_size, image_size))
            images.append(im)
        user = [{"type": "image", "image": im} for im in images]
        user.append({"type": "text", "text": prompt_for(p)})
        examples.append({"messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": [{"type": "text", "text": json.dumps(answer)}]},
        ]})
    return examples


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=4)  # 8 examples per update
    ap.add_argument("--image-size", type=int, default=512)
    ap.add_argument("--name", default="tier1-v0")
    args = ap.parse_args()

    # Unsloth patches transformers and TRL, so it must be imported before them.
    from unsloth import FastVisionModel, is_bf16_supported  # isort: skip
    from unsloth.trainer import UnslothVisionDataCollator  # isort: skip
    import torch  # isort: skip
    from trl import SFTConfig, SFTTrainer  # isort: skip

    dataset = load_examples(args.image_size)
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
        "logging_steps": 10,
        "seed": 3407,
        "bf16": bf16,
        "fp16": not bf16,
        "report_to": "none",
        "save_strategy": "steps",
        "save_steps": 200,
        "save_total_limit": 2,
        "remove_unused_columns": False,
        "dataset_text_field": "",
        "dataset_kwargs": {"skip_prepare_dataset": True},
        "max_length": 4096,
        "max_seq_length": 4096,
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
    adapter_dir = ROOT / "outputs" / f"{args.name}-lora"
    merged_dir = ROOT / "outputs" / f"{args.name}-merged"
    model.save_pretrained(str(adapter_dir))
    processor.save_pretrained(str(adapter_dir))
    model.save_pretrained_merged(str(merged_dir), processor, save_method="merged_16bit")
    report = {"examples": len(dataset), "epochs": args.epochs, "rank": args.rank,
              "train_min": round((time.time() - start) / 60, 1),
              "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
              "loss_first": losses[0] if losses else None,
              "loss_last": losses[-1] if losses else None, "losses": losses,
              "merged_dir": str(merged_dir)}
    (ROOT / "reports" / f"train_{args.name}.json").write_text(json.dumps(report, indent=2))
    print(f"Done in {report['train_min']} min, loss {report['loss_first']} -> "
          f"{report['loss_last']}, peak {report['peak_vram_gb']} GB; merged in {merged_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
