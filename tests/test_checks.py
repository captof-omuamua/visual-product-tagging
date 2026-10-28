"""Grammar, log-prob and comparison checks, against the mock server in conftest.py."""

from __future__ import annotations

import json

from catalog_cascade import checks, tier2


def test_split_logprobs_separates_values_from_structure():
    text = '{"color": "red", "shape": "circle"}'
    tokens = [(ch, -0.3 if 11 <= i < 14 or 27 <= i < 33 else 0.0) for i, ch in enumerate(text)]
    parts = checks.split_logprobs(tokens, ["red", "circle"])
    assert set(parts["value"]) == {-0.3} and len(parts["value"]) == 9
    assert set(parts["structure"]) == {0.0}


def test_payload_carries_every_image_and_the_schema():
    body = checks.payload("m", "q", ["u1", "u2", "u3"], checks.schema(["a"], ["b"]),
                          logprobs=True)
    images = [c for c in body["messages"][0]["content"] if c["type"] == "image_url"]
    assert len(images) == 3
    assert body["response_format"]["json_schema"]["schema"]["properties"]["color"] == {
        "enum": ["a"]}
    assert body["logprobs"] is True and body["top_logprobs"] == 5


def test_grammar_check_passes_when_enforced(mock_server):
    result = checks.grammar_check(mock_server, "base", "q", [["u"]] * 4)
    assert result["ok"] is True and result["forced_enforced_rate"] == 1.0
    assert result["structure_tokens_near_zero"] is True
    assert result["value_logprob"]["median"] == -0.3


def test_grammar_check_catches_a_bypass(mock_server):
    result = checks.grammar_check(mock_server, "nogrammar", "q", [["u"]] * 4)
    assert result["ok"] is False and result["forced_enforced_rate"] == 0.0


def test_match_rate_ignores_failed_answers():
    assert checks.match_rate(["a", "b", None], ["a", "c", "x"]) == 0.5
    assert checks.match_rate([None], ["x"]) is None


def test_scan_log_flags():
    flags = tier2.scan_log(["Using TRITON_ATTN backend", "all good"])
    assert flags["triton_attention_fallback"] and not flags["out_of_memory"]


def test_overnight_hours():
    assert tier2.overnight_hours(1.0) == 2.4
    assert tier2.overnight_hours(None) is None


def test_tier2_end_to_end(mock_server, tmp_path):
    out = tmp_path / "tier2.json"
    code = tier2.main(["--base-url", mock_server, "--model", "base", "--n", "6",
                       "--image-size", "64", "--images-per-item", "2", "--out", str(out)])
    report = json.loads(out.read_text())
    assert code == 0 and report["ok"] is True
    assert report["load"]["ok"] == 6 and report["grammar"]["forced_enforced_rate"] == 1.0
