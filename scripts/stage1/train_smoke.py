#!/usr/bin/env python3
"""Stage 1 training smoke test: a short 16-bit LoRA fine-tune of Qwen3.5-4B with Unsloth.

Run in the training environment:  .venv-train/bin/python scripts/stage1/train_smoke.py

What it proves, for the exact Tier 1 setup in the design doc:
- Unsloth, torch and Qwen3.5-4B load and train on this GPU (Blackwell, compute 12.0).
- 16-bit LoRA on the language layers, vision layers frozen, with 3 images of 512 px per
  item, fits in memory: the peak is recorded (the design's limit is about 22 GB).
- The chat template's thinking switch: Qwen3.5 thinks unless `enable_thinking=False`, so
  the report records whether a <think> block appears in training and inference text.
- The adapter saves (outputs/smoke-lora) and merges into 16-bit weights
  (outputs/smoke-merged), which is how vLLM serves Tier 1.

It is not a model: 96 synthetic items, 30 steps. It writes reports/train_smoke.json even
when a step fails, recording which step and why.
"""

from __future__ import annotations

import argparse
import gc
import importlib.metadata as md
import inspect
import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))  # works even before `pip install -e .`

from catalog_cascade.synthetic import (  # noqa: E402
    EVAL_SEED,
    PROMPT,
    TRAIN_SEED,
    Item,
    make_items,
    save_items,
    views,
)

DEFAULT_MODEL = "Qwen/Qwen3.5-4B"  # Apache-2.0; Tier 1 in the design doc
MEMORY_LIMIT_GB = 22.0
PACKAGES = (
    "torch",
    "torchvision",
    "unsloth",
    "unsloth_zoo",
    "transformers",
    "trl",
    "peft",
    "accelerate",
    "bitsandbytes",
    "datasets",
)


def versions() -> dict[str, str | None]:
    found: dict[str, str | None] = {}
    for name in PACKAGES:
        try:
            found[name] = md.version(name)
        except md.PackageNotFoundError:
            found[name] = None
    return found


def accepted(fn, kwargs: dict) -> dict:
    """Keep only the keyword arguments `fn` accepts: these library APIs drift between versions."""
    params = inspect.signature(fn).parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return dict(kwargs)
    return {k: v for k, v in kwargs.items() if k in params}


def user_turn(images: list) -> dict:
    content = [{"type": "image", "image": im} for im in images]
    content.append({"type": "text", "text": PROMPT})
    return {"role": "user", "content": content}


def to_conversation(item: Item, images: list) -> dict:
    """The chat format Unsloth's vision collator expects: images and question, then the answer."""
    return {
        "messages": [
            user_turn(images),
            {"role": "assistant", "content": [{"type": "text", "text": item.target}]},
        ]
    }


def render(processor, messages: list[dict], **kwargs) -> str:
    """The chat template as text, with images replaced by placeholders."""
    plain = [
        {
            "role": m["role"],
            "content": [
                {"type": "image"} if c["type"] == "image" else c for c in m["content"]
            ],
        }
        for m in messages
    ]
    return processor.apply_chat_template(plain, tokenize=False, **kwargs)


def thinking_check(processor, item: Item, images: list) -> dict:
    """Does a <think> block appear with and without enable_thinking=False?"""
    conversation = to_conversation(item, images)["messages"]
    prompt_only = [conversation[0]]
    out: dict = {}
    for name, kwargs in (
        ("inference_default", {"add_generation_prompt": True}),
        ("inference_thinking_off", {"add_generation_prompt": True, "enable_thinking": False}),
        ("training_default", {}),
        ("training_thinking_off", {"enable_thinking": False}),
    ):
        messages = prompt_only if name.startswith("inference") else conversation
        try:
            text = render(processor, messages, **kwargs)
            out[name] = {"has_think": "<think>" in text, "tail": text[-160:]}
        except Exception as exc:  # informational
            out[name] = {"error": f"{type(exc).__name__}: {exc}"}
    default = out.get("training_default", {}).get("tail")
    off = out.get("training_thinking_off", {}).get("tail")
    # If these differ, the collator's default rendering is not the thinking-off format the
    # server will use, and stage 5 must render training text with enable_thinking=False.
    out["training_matches_thinking_off"] = default is not None and default == off
    return out


def generate(fast_vision_model, model, processor, items: list[Item], k: int, size: int):
    """Greedy answers from the trained model with thinking off, to eyeball before serving."""
    fast_vision_model.for_inference(model)
    rows = []
    for item in items:
        images = views(item, k, EVAL_SEED, size)
        prompt = render(processor, [user_turn(images)], add_generation_prompt=True,
                        enable_thinking=False)
        inputs = processor(
            images=images, text=prompt, add_special_tokens=False, return_tensors="pt"
        ).to(model.device)
        output = model.generate(**inputs, max_new_tokens=32, do_sample=False, use_cache=True)
        new_tokens = output[:, inputs["input_ids"].shape[1] :]
        text = processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
        rows.append({"target": item.target, "output": text, "exact": text == item.target})
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Stage 1 training smoke test (Unsloth LoRA)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--image-size", type=int, default=512)
    ap.add_argument("--images-per-item", type=int, default=3)
    ap.add_argument("--n-per-combo", type=int, default=4, help="6 colours x 4 shapes x n items")
    ap.add_argument("--load-in-4bit", action="store_true",
                    help="QLoRA, only for the Qwen3-VL-8B fallback; not advised for Qwen3.5")
    ap.add_argument("--skip-merge", action="store_true")
    ap.add_argument("--adapter-dir", default=str(ROOT / "outputs" / "smoke-lora"))
    ap.add_argument("--merged-dir", default=str(ROOT / "outputs" / "smoke-merged"))
    ap.add_argument("--samples-dir", default=str(ROOT / "outputs" / "smoke-data"))
    ap.add_argument("--out", default=str(ROOT / "reports" / "train_smoke.json"))
    args = ap.parse_args(argv)

    report: dict = {
        "ok": False,
        "model": args.model,
        "steps": args.steps,
        "rank": args.rank,
        "batch_size": args.batch_size,
        "image_size": args.image_size,
        "images_per_item": args.images_per_item,
        "load_in_4bit": args.load_in_4bit,
        "versions": versions(),
    }
    stage = "import"
    try:
        # isort: off
        # Unsloth patches transformers and TRL, so it must be imported before them.
        from unsloth import FastVisionModel, is_bf16_supported
        from unsloth.trainer import UnslothVisionDataCollator
        import torch
        from trl import SFTConfig, SFTTrainer

        # isort: on

        stage = "gpu"
        report["gpu"] = {
            "name": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
            "torch_cuda": torch.version.cuda,
            "arch_list": torch.cuda.get_arch_list(),
        }

        stage = "data"
        train_items = make_items(args.n_per_combo, TRAIN_SEED, args.image_size)
        save_items(train_items[:8], Path(args.samples_dir))  # samples to look at
        dataset = [
            to_conversation(it, views(it, args.images_per_item, TRAIN_SEED, args.image_size))
            for it in train_items
        ]
        report["train_examples"] = len(dataset)

        stage = "load_model"
        start = time.time()
        model, processor = FastVisionModel.from_pretrained(
            args.model, load_in_4bit=args.load_in_4bit, use_gradient_checkpointing="unsloth"
        )
        report["load_s"] = round(time.time() - start, 1)

        stage = "thinking_check"
        first = train_items[0]
        report["thinking"] = thinking_check(
            processor, first, views(first, args.images_per_item, TRAIN_SEED, args.image_size)
        )

        stage = "lora"
        model = FastVisionModel.get_peft_model(
            model,
            finetune_vision_layers=False,  # vision frozen, as the design doc says
            finetune_language_layers=True,
            finetune_attention_modules=True,
            finetune_mlp_modules=True,
            r=args.rank,
            lora_alpha=args.rank,
            lora_dropout=0,
            bias="none",
            random_state=3407,
        )
        FastVisionModel.for_training(model)

        stage = "trainer"
        bf16 = bool(is_bf16_supported())
        config_kwargs = {
            "output_dir": str(ROOT / "outputs" / "smoke-train"),
            "per_device_train_batch_size": args.batch_size,
            "gradient_accumulation_steps": 1,
            "warmup_steps": 3,
            "max_steps": args.steps,
            "learning_rate": 2e-4,
            "lr_scheduler_type": "linear",
            "weight_decay": 0.01,
            "optim": "adamw_torch",  # keeps bitsandbytes out of this step
            "logging_steps": 1,
            "seed": 3407,
            "bf16": bf16,
            "fp16": not bf16,
            "report_to": "none",
            "save_strategy": "no",
            "remove_unused_columns": False,  # the collator needs the image column
            "dataset_text_field": "",
            "dataset_kwargs": {"skip_prepare_dataset": True},  # the vision collator tokenises
            "max_length": 4096,  # newer TRL; 3 images at 512 px need room
            "max_seq_length": 4096,  # older TRL; `accepted` drops whichever is unknown
        }
        config = SFTConfig(**accepted(SFTConfig, config_kwargs))
        common = {
            "model": model,
            "data_collator": UnslothVisionDataCollator(model, processor),
            "train_dataset": dataset,
            "args": config,
        }
        try:
            trainer = SFTTrainer(processing_class=processor, **common)
        except TypeError:  # older TRL named it `tokenizer`
            trainer = SFTTrainer(tokenizer=processor, **common)

        stage = "train"
        torch.cuda.reset_peak_memory_stats()
        start = time.time()
        trainer.train()
        report["train_s"] = round(time.time() - start, 1)
        report["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1024**3, 2)
        report["memory_ok"] = report["peak_vram_gb"] <= MEMORY_LIMIT_GB
        losses = [round(float(h["loss"]), 4) for h in trainer.state.log_history if "loss" in h]
        report["losses"] = losses
        report["loss_first"] = losses[0] if losses else None
        report["loss_last"] = losses[-1] if losses else None
        report["loss_fell"] = bool(losses) and losses[-1] < losses[0]

        stage = "save"
        adapter_dir = Path(args.adapter_dir)
        model.save_pretrained(str(adapter_dir))
        processor.save_pretrained(str(adapter_dir))
        adapter_config = json.loads((adapter_dir / "adapter_config.json").read_text())
        report["adapter_dir"] = str(adapter_dir)
        report["adapter_config"] = {
            k: adapter_config.get(k)
            for k in ("r", "lora_alpha", "target_modules", "base_model_name_or_path")
        }

        stage = "generate"
        eval_items = make_items(1, EVAL_SEED, args.image_size)[:4]  # items it never saw
        try:
            report["generation"] = generate(
                FastVisionModel, model, processor, eval_items, args.images_per_item,
                args.image_size,
            )
        except Exception as exc:  # informational: serving is the real test
            report["generation_error"] = f"{type(exc).__name__}: {exc}"

        if not args.skip_merge:
            stage = "merge"
            merged_dir = Path(args.merged_dir)
            start = time.time()
            # Unsloth's documented path for Qwen3.5: merge the adapter into 16-bit weights.
            model.save_pretrained_merged(str(merged_dir), processor, save_method="merged_16bit")
            report["merge_s"] = round(time.time() - start, 1)
            report["merged_dir"] = str(merged_dir)
            merged_bytes = sum(p.stat().st_size for p in merged_dir.glob("*.safetensors"))
            report["merged_gb"] = round(merged_bytes / 1024**3, 2)
            report["merged_ok"] = merged_bytes > 0 and (merged_dir / "config.json").exists()

        report["ok"] = report["loss_fell"] and (args.skip_merge or report.get("merged_ok", False))
        if not report["loss_fell"]:
            report["failed_stage"] = "train"
            report["error"] = "the loss did not fall over the run"
        elif not report["ok"]:
            report["failed_stage"] = "merge"
            report["error"] = "the merged model has no weights"
        del trainer, model
        gc.collect()
    except Exception as exc:
        report["ok"] = False
        report["failed_stage"] = stage
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc().splitlines()[-30:]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    if report["ok"]:
        print(
            f"PASS: trained {args.steps} steps in {report['train_s']} s, loss "
            f"{report['loss_first']} -> {report['loss_last']}, peak {report['peak_vram_gb']} GB "
            f"(limit {MEMORY_LIMIT_GB}); merged weights in {report.get('merged_dir')}"
        )
    else:
        print(f"FAIL at '{report.get('failed_stage')}': {report.get('error')}")
    thinking = report.get("thinking") or {}
    if thinking and not thinking.get("training_matches_thinking_off"):
        print("NOTE: training text differs from the thinking-off format; "
              "see reports/train_smoke.json")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
