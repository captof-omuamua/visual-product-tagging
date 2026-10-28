import requests

from catalog_cascade.loadtest import chat_payload, percentile, run_load, send_one, summarize


def payloads(model: str, n: int) -> list[dict]:
    return [chat_payload(model, "hi", "data:image/png;base64,AAAA") for _ in range(n)]


def test_payload_puts_image_before_text():
    body = chat_payload("m", "question", "data:image/png;base64,AAAA", max_tokens=8)
    content = body["messages"][0]["content"]
    assert [part["type"] for part in content] == ["image_url", "text"]
    assert body["max_tokens"] == 8 and body["temperature"] == 0.0


def test_all_requests_succeed_and_are_summarised(mock_server):
    results, wall = run_load(mock_server, payloads("base", 60), concurrency=30, timeout_s=10)
    summary = summarize(results, wall)
    assert summary["ok"] == 60 and summary["failed"] == 0 and summary["errors"] == {}
    assert summary["latency_s"]["p50"] >= 0.05
    assert summary["completion_tokens_per_s"] > 0
    assert all(r.completion_tokens == 12 for r in results)


def test_concurrency_is_bounded_and_real(mock_server):
    requests.post(f"{mock_server}/reset", timeout=5)
    run_load(mock_server, payloads("base", 40), concurrency=8, timeout_s=10)
    assert requests.get(f"{mock_server}/stats", timeout=5).json()["max_in_flight"] <= 8

    requests.post(f"{mock_server}/reset", timeout=5)
    run_load(mock_server, payloads("base", 120), concurrency=60, timeout_s=10)
    assert requests.get(f"{mock_server}/stats", timeout=5).json()["max_in_flight"] > 8


def test_http_errors_are_counted_not_raised(mock_server):
    results, wall = run_load(mock_server, payloads("fail", 5), concurrency=5, timeout_s=10)
    assert summarize(results, wall)["errors"] == {"http_500": 5}


def test_malformed_body_is_a_bad_response(mock_server):
    result = send_one(mock_server, payloads("garbage", 1)[0], timeout_s=10)
    assert not result.ok and result.error == "bad_response"


def test_timeouts_are_reported(mock_server):
    result = send_one(mock_server, payloads("slow", 1)[0], timeout_s=0.3)
    assert not result.ok and result.error == "timeout"


def test_connection_errors_are_reported(closed_port_url):
    result = send_one(closed_port_url, payloads("base", 1)[0], timeout_s=1)
    assert not result.ok and result.error == "connection_error"


def test_percentile_interpolates():
    assert percentile([], 50) is None
    assert percentile([3.0], 95) == 3.0
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert percentile(list(range(101)), 95) == 95
