"""Tests for the standard-library scripts in scripts/stage1 (on the pytest path)."""

import json

import check_env
import report


def test_parse_cuda_version_from_banner():
    banner = "| NVIDIA-SMI 580.95.05   Driver Version: 580.95.05   CUDA Version: 13.0     |"
    assert check_env.parse_cuda_version(banner) == (13, 0)
    assert check_env.parse_cuda_version("no banner") is None
    assert check_env.parse_cuda_version(None) is None


def test_parse_gpu_query():
    line = "NVIDIA GeForce RTX 5090 Laptop GPU, 580.95.05, 24463 MiB, 12.0\n"
    rows = check_env.parse_gpu_query(line)
    assert rows == [
        {
            "name": "NVIDIA GeForce RTX 5090 Laptop GPU",
            "driver": "580.95.05",
            "vram_gb": 23.9,
            "compute_capability": "12.0",
        }
    ]


GPU = [{"name": "RTX 5090", "driver": "580", "vram_gb": 23.9, "compute_capability": "12.0"}]


def test_evaluate_passes_a_ready_machine():
    blockers, warnings = check_env.evaluate("Linux", GPU, (13, 0), True, 120.0)
    assert blockers == [] and warnings == []


def test_evaluate_names_each_blocker():
    blockers, _ = check_env.evaluate("Windows", GPU, (12, 8), False, 10.0)
    text = " ".join(blockers)
    assert "Ubuntu" in text and "CUDA 12.8" in text and "uv" in text and "10.0 GB" in text
    assert len(blockers) == 4


def test_evaluate_warns_on_small_gpu_and_tight_disk():
    small = [{**GPU[0], "vram_gb": 16.0}]
    blockers, warnings = check_env.evaluate("Linux", small, (13, 1), True, 90.0)
    assert blockers == [] and len(warnings) == 2


def _write(tmp_path, name, data):
    (tmp_path / name).write_text(json.dumps(data))


SERVE_OK = {"server_ready": True, "burst_all_ok": True, "healthy_after_load": True}
BURST = {"requests": 200, "ok": 200, "latency_s": {"p50": 4.1, "p95": 9.8}}


def _passing_reports(tmp_path):
    _write(tmp_path, "env_check.json", {"ok": True, "gpus": GPU, "driver_cuda": "13.0"})
    _write(tmp_path, "train_smoke.json", {
        "ok": True, "steps": 30, "loss_first": 2.1, "loss_last": 0.3, "peak_vram_gb": 14.2,
        "memory_ok": True, "merged_gb": 8.7,
        "thinking": {"training_matches_thinking_off": True}})
    _write(tmp_path, "serve_runtime.json", {"checks": SERVE_OK, "burst": BURST})
    _write(tmp_path, "serve_merged.json", {
        "checks": {**SERVE_OK, "grammar_enforced": True}, "burst": BURST,
        "merged_vs_runtime_match": 0.98,
        "grammar": {"forced_enforced_rate": 1.0, "enum_valid_rate": 1.0,
                    "structure_tokens_near_zero": True,
                    "structure_logprob": {"median": 0.0}, "value_logprob": {"median": -0.2}}})
    _write(tmp_path, "tier2_qwen.json", {
        "model": "Qwen/Qwen3.5-27B-GPTQ-Int4", "ok": True, "load": {"requests_per_s": 0.8},
        "overnight_hours_for_8600_items": 3.0, "grammar": {"forced_enforced_rate": 1.0},
        "log_flags": {}})
    _write(tmp_path, "abo_check.json", {
        "failing_types": [], "types": [{"product_type": "CHAIR", "products": 2000,
                                         "checks": {"size": True}}],
        "certificate_reach": {"accepted_per_field": 1505,
                              "needed": {"2%": {"needed": 322, "reachable": True}}}})
    _write(tmp_path, "gemini_pilot.json", {"calls": 20, "mean_thinking_tokens": 410,
                                           "logprobs_for_images": False, "spent_usd": 0.19})


def test_report_passes_when_every_step_passes(tmp_path):
    _passing_reports(tmp_path)
    assert report.main(str(tmp_path)) == 0
    markdown = (tmp_path / "stage1_report.md").read_text()
    assert markdown.startswith("# Stage 1 report: PASS")
    assert "forced schema obeyed 1.0" in markdown
    assert "~3.0 h for 8,600 items" in markdown
    assert "## Warnings" not in markdown


def test_report_fails_and_names_the_failed_step(tmp_path):
    _passing_reports(tmp_path)
    _write(tmp_path, "train_smoke.json",
           {"ok": False, "failed_stage": "load_model", "error": "CUDA error: no kernel image"})
    assert report.main(str(tmp_path)) == 1
    summary = json.loads((tmp_path / "stage1_report.json").read_text())
    assert summary["status"] == "FAIL"
    assert any("load_model" in f for f in summary["failures"])


def test_report_fails_when_grammar_is_not_enforced(tmp_path):
    _passing_reports(tmp_path)
    _write(tmp_path, "serve_merged.json",
           {"checks": {**SERVE_OK, "grammar_enforced": False}, "burst": BURST})
    assert report.main(str(tmp_path)) == 1


def test_report_one_large_model_failing_is_a_warning(tmp_path):
    _passing_reports(tmp_path)
    _write(tmp_path, "tier2_gemma.json", {"model": "gemma", "ok": False, "log_flags":
                                          {"triton_attention_fallback": True}})
    assert report.main(str(tmp_path)) == 0
    summary = json.loads((tmp_path / "stage1_report.json").read_text())
    assert any("gemma failed" in w for w in summary["warnings"])


def test_report_fails_when_a_step_never_ran(tmp_path):
    _write(tmp_path, "env_check.json", {"ok": True})
    assert report.main(str(tmp_path)) == 1
    summary = json.loads((tmp_path / "stage1_report.json").read_text())
    assert "The training smoke test did not run." in summary["failures"]
