"""Dry run of scripts/stage1/train_smoke.py with fake unsloth, trl and torch modules.

The real script needs a GPU. This test exercises its control flow on CPU: argument
handling, the API-drift shims, the report it writes, and how it records a failure.
"""

import json
import sys
import types
from dataclasses import dataclass, field

import numpy as np
import pytest
import train_smoke


class FakeBatch(dict):
    def to(self, _device):
        return self


class FakeProcessor:
    answer = '{"color": "red", "shape": "circle"}'

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False,
                            enable_thinking=True):
        # Like Qwen3.5: a <think> block unless thinking is switched off explicitly.
        assert tokenize is False
        return "prompt" + ("" if enable_thinking is False else "<think>\n")

    def __call__(self, images, text, add_special_tokens, return_tensors):
        return FakeBatch(input_ids=np.zeros((1, 5)))

    def batch_decode(self, tokens, skip_special_tokens):
        assert tokens.shape == (1, 4)  # only the newly generated tokens
        return [self.answer]

    def save_pretrained(self, path):
        pass


class FakeModel:
    device = "cpu"

    def generate(self, **kwargs):
        return np.zeros((1, 9))

    def save_pretrained(self, path):
        from pathlib import Path

        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        config = {
            "r": 16,
            "lora_alpha": 16,
            "target_modules": "regex",
            "base_model_name_or_path": "m",
        }
        (out / "adapter_config.json").write_text(json.dumps(config))
        (out / "adapter_model.safetensors").write_bytes(b"0" * 2048)

    def save_pretrained_merged(self, path, processor, save_method):
        from pathlib import Path

        assert save_method == "merged_16bit"
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        (out / "model.safetensors").write_bytes(b"0" * 4096)
        (out / "config.json").write_text("{}")


@dataclass
class FakeSFTConfig:  # knows max_length but not max_seq_length, like newer TRL
    output_dir: str = ""
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    warmup_steps: int = 0
    max_steps: int = 1
    learning_rate: float = 1e-4
    lr_scheduler_type: str = "linear"
    weight_decay: float = 0.0
    optim: str = "adamw_torch"
    logging_steps: int = 1
    seed: int = 0
    bf16: bool = False
    fp16: bool = False
    report_to: str = "none"
    save_strategy: str = "no"
    remove_unused_columns: bool = True
    dataset_text_field: str = "text"
    dataset_kwargs: dict = field(default_factory=dict)
    max_length: int = 512


class FakeTrainer:
    def __init__(self, model, data_collator, train_dataset, args, processing_class=None):
        assert processing_class is not None and args.remove_unused_columns is False
        self.args = args
        self.state = types.SimpleNamespace(log_history=[])

    def train(self):
        losses = np.linspace(2.0, 0.4, self.args.max_steps)
        self.state.log_history = [{"loss": float(x), "step": i} for i, x in enumerate(losses)]


def install_fakes(monkeypatch, fail_load=False):
    fvm = types.SimpleNamespace()

    def from_pretrained(model_id, load_in_4bit, **kwargs):
        if fail_load:
            raise RuntimeError("CUDA error: no kernel image is available for execution")
        return FakeModel(), FakeProcessor()

    fvm.from_pretrained = from_pretrained
    fvm.get_peft_model = lambda model, **kwargs: model
    fvm.for_training = lambda model: None
    fvm.for_inference = lambda model: None

    unsloth = types.ModuleType("unsloth")
    unsloth.FastVisionModel = fvm
    unsloth.is_bf16_supported = lambda: True
    unsloth_trainer = types.ModuleType("unsloth.trainer")
    unsloth_trainer.UnslothVisionDataCollator = lambda model, processor: object()
    trl = types.ModuleType("trl")
    trl.SFTConfig = FakeSFTConfig
    trl.SFTTrainer = FakeTrainer

    torch = types.ModuleType("torch")
    torch.version = types.SimpleNamespace(cuda="13.0")
    torch.cuda = types.SimpleNamespace(
        get_device_name=lambda i: "Fake RTX 5090",
        get_device_capability=lambda i: (12, 0),
        get_arch_list=lambda: ["sm_120"],
        reset_peak_memory_stats=lambda: None,
        max_memory_allocated=lambda: 12 * 1024**3,
        empty_cache=lambda: None,
    )
    for name, module in {
        "unsloth": unsloth,
        "unsloth.trainer": unsloth_trainer,
        "trl": trl,
        "torch": torch,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)


@pytest.fixture
def paths(tmp_path):
    return [
        "--adapter-dir", str(tmp_path / "adapter"),
        "--merged-dir", str(tmp_path / "merged"),
        "--image-size", "64",
        "--samples-dir", str(tmp_path / "samples"),
        "--out", str(tmp_path / "train.json"),
    ]


def test_dry_run_passes_and_writes_a_full_report(monkeypatch, tmp_path, paths):
    install_fakes(monkeypatch)
    assert train_smoke.main(["--steps", "5", "--n-per-combo", "1", *paths]) == 0
    report = json.loads((tmp_path / "train.json").read_text())
    assert report["ok"] is True and report["loss_fell"] is True
    assert report["gpu"]["capability"] == [12, 0]
    assert report["peak_vram_gb"] == 12.0
    assert report["adapter_config"]["r"] == 16
    assert report["memory_ok"] is True and report["merged_ok"] is True
    assert len(report["generation"]) == 4
    thinking = report["thinking"]
    assert thinking["inference_default"]["has_think"] is True
    assert thinking["inference_thinking_off"]["has_think"] is False
    assert thinking["training_matches_thinking_off"] is False
    assert len(list((tmp_path / "samples").glob("*.png"))) == 8


def test_dry_run_records_which_step_failed(monkeypatch, tmp_path, paths):
    install_fakes(monkeypatch, fail_load=True)
    assert train_smoke.main(["--steps", "5", "--n-per-combo", "1", *paths]) == 1
    report = json.loads((tmp_path / "train.json").read_text())
    assert report["ok"] is False and report["failed_stage"] == "load_model"
    assert "no kernel image" in report["error"]


def test_accepted_drops_unknown_keywords():
    kept = train_smoke.accepted(FakeSFTConfig, {"max_length": 1, "max_seq_length": 2})
    assert kept == {"max_length": 1}
