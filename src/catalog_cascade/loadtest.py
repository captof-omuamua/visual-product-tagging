"""Concurrent load test for an OpenAI-compatible chat endpoint (vLLM, SGLang, ...).

Why threads, not asyncio: the client only waits on sockets, and CPython releases
the GIL while a thread waits, so a thread pool reaches the same concurrency with
code short enough to explain line by line. A few hundred threads is fine.

Stage 1 uses it for the 200-request burst; stage 7 reuses it for latency and
throughput at concurrency 1-64.
"""

from __future__ import annotations

import json
import statistics
import threading
import time
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import requests

CONNECT_TIMEOUT_S = 10


@dataclass
class Result:
    ok: bool
    latency_s: float
    status: int | None = None
    error: str | None = None
    text: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    raw: dict | None = None  # the parsed choice, kept only when asked (log-prob checks)


def chat_payload(
    model: str,
    prompt: str,
    image_url: str | None = None,
    max_tokens: int = 48,
    temperature: float = 0.0,
) -> dict:
    """One chat-completions request body, with an optional image before the text."""
    content: list[dict] = []
    if image_url:
        content.append({"type": "image_url", "image_url": {"url": image_url}})
    content.append({"type": "text", "text": prompt})
    return {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }


_local = threading.local()


def _session() -> requests.Session:
    """One HTTP session per thread: sessions reuse connections but are not thread-safe."""
    session = getattr(_local, "session", None)
    if session is None:
        session = requests.Session()
        _local.session = session
    return session


def send_one(
    base_url: str,
    payload: dict,
    timeout_s: float,
    api_key: str | None = None,
    keep_raw: bool = False,
) -> Result:
    """Send one request and classify the outcome; never raises."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    start = time.perf_counter()
    try:
        response = _session().post(
            url, data=json.dumps(payload), headers=headers, timeout=(CONNECT_TIMEOUT_S, timeout_s)
        )
        elapsed = time.perf_counter() - start
        if response.status_code != 200:
            return Result(False, elapsed, response.status_code, f"http_{response.status_code}")
        body = response.json()
        choice = body["choices"][0]
        text = choice["message"].get("content")
        usage = body.get("usage") or {}
        return Result(
            True,
            elapsed,
            200,
            None,
            text,
            int(usage.get("prompt_tokens") or 0),
            int(usage.get("completion_tokens") or 0),
            choice if keep_raw else None,
        )
    except requests.Timeout:
        return Result(False, time.perf_counter() - start, None, "timeout")
    except requests.ConnectionError:
        return Result(False, time.perf_counter() - start, None, "connection_error")
    except requests.RequestException:
        return Result(False, time.perf_counter() - start, None, "request_error")
    except (ValueError, KeyError, IndexError, TypeError):
        return Result(False, time.perf_counter() - start, 200, "bad_response")


def run_load(
    base_url: str,
    payloads: Sequence[dict],
    concurrency: int,
    timeout_s: float,
    api_key: str | None = None,
    keep_raw: bool = False,
) -> tuple[list[Result], float]:
    """Send every payload with at most `concurrency` in flight; return results in order."""
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        results = list(
            pool.map(lambda p: send_one(base_url, p, timeout_s, api_key, keep_raw), payloads)
        )
    return results, time.perf_counter() - start


def percentile(values: Sequence[float], q: float) -> float | None:
    """Linear-interpolated percentile, q in [0, 100]."""
    if not values:
        return None
    xs = sorted(values)
    rank = (len(xs) - 1) * q / 100
    lo = int(rank)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (rank - lo)


def _round(x: float | None, digits: int = 3) -> float | None:
    return None if x is None else round(x, digits)


def summarize(results: Sequence[Result], wall_s: float) -> dict:
    """Counts, error types, latency percentiles over successful requests, throughput."""
    ok = [r for r in results if r.ok]
    latencies = [r.latency_s for r in ok]
    completion_tokens = sum(r.completion_tokens for r in ok)
    return {
        "requests": len(results),
        "ok": len(ok),
        "failed": len(results) - len(ok),
        "errors": dict(Counter(r.error for r in results if not r.ok)),
        "wall_s": _round(wall_s),
        "latency_s": {
            "p50": _round(percentile(latencies, 50)),
            "p95": _round(percentile(latencies, 95)),
            "p99": _round(percentile(latencies, 99)),
            "max": _round(max(latencies) if latencies else None),
            "mean": _round(statistics.fmean(latencies) if latencies else None),
        },
        "requests_per_s": _round(len(ok) / wall_s if wall_s > 0 else None),
        "completion_tokens_per_s": _round(completion_tokens / wall_s if wall_s > 0 else None, 1),
    }
