import json

from catalog_cascade import smoke
from catalog_cascade.synthetic import EVAL_SEED, make_items


def test_parse_attributes_handles_fences_and_british_spelling():
    text = 'Sure!\n```json\n{"Colour": "Red", "Shape": "circle"}\n```'
    assert smoke.parse_attributes(text) == {"colour": "red", "shape": "circle"}
    assert smoke.parse_attributes("no json here") is None
    assert smoke.parse_attributes(None) is None
    assert smoke.parse_attributes('{broken} {"color": "blue"}') == {"color": "blue"}


def test_score_separates_exact_format_from_field_accuracy():
    items = make_items(1, EVAL_SEED)[:2]
    outputs = [items[0].target, f'```json\n{{"colour": "{items[1].color}"}}\n```']
    result = smoke.score(outputs, items)
    assert result["exact_format_rate"] == 0.5
    assert result["json_rate"] == 1.0
    assert result["color_accuracy"] == 1.0
    assert result["shape_accuracy"] == 0.5


def test_wait_ready_gives_up(closed_port_url):
    ready, _, why = smoke.wait_ready(closed_port_url, timeout_s=0.2, poll_s=0.05)
    assert not ready and why == "startup_timeout"


def test_end_to_end_against_mock_server(mock_server, tmp_path):
    out = tmp_path / "serve.json"
    code = smoke.main(
        [
            "--base-url", mock_server,
            "--model", "base",
            "--adapter", "smoke-lora",
            "--burst", "40",
            "--sustained", "20",
            "--sustained-concurrency", "10",
            "--eval-n", "6",
            "--image-size", "64",
            "--images-per-item", "2",
            "--answers-out", str(tmp_path / "answers.json"),
            "--out", str(out),
        ]
    )
    report = json.loads(out.read_text())
    assert code == 0 and report["ok"] is True
    assert len(json.loads((tmp_path / "answers.json").read_text())) == 6
    assert report["vllm_version"] == "0.0-mock"
    assert report["burst"]["ok"] == 40 and report["sustained"]["ok"] == 20
    assert report["checks"]["healthy_after_load"] is True
    # The mock answers identically for base and adapter, so no LoRA effect is visible.
    assert report["lora_effect_visible"] is False


def test_missing_adapter_fails_the_listing_check(mock_server, tmp_path):
    out = tmp_path / "serve.json"
    code = smoke.main(
        [
            "--base-url", mock_server,
            "--model", "base",
            "--adapter", "not-served",
            "--burst", "5",
            "--eval-n", "2",
            "--image-size", "64",
            "--out", str(out),
        ]
    )
    report = json.loads(out.read_text())
    assert code == 1 and report["checks"]["adapter_listed"] is False


def test_merged_phase_runs_grammar_and_compares(mock_server, tmp_path):
    previous = tmp_path / "runtime.json"
    previous.write_text(json.dumps(['{"color": "red", "shape": "circle"}'] * 4))
    out = tmp_path / "merged.json"
    code = smoke.main(
        [
            "--base-url", mock_server,
            "--model", "base",
            "--burst", "8",
            "--eval-n", "4",
            "--image-size", "64",
            "--grammar",
            "--compare-with", str(previous),
            "--out", str(out),
        ]
    )
    report = json.loads(out.read_text())
    assert code == 0 and report["checks"]["grammar_enforced"] is True
    assert report["merged_vs_runtime_match"] == 1.0
