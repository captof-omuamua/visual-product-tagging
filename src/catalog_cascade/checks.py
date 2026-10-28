"""Stage 1 checks that the design depends on, run against a live vLLM server.

1. Grammar enforcement. A schema whose only allowed values are nonsense words the model
   would never write unprompted. If vLLM enforces the schema, every answer must contain
   them. (Gemma 4 once silently skipped enforcement: vLLM issue #39130.)
2. Log-probs under a grammar. On a normal enum schema, the log-probs of structural
   tokens (braces, keys, quotes) should be about 0, because vLLM masks disallowed tokens
   before computing log-probs. Only the value tokens carry a confidence signal.
3. Merged against run-time adapter. The same prompts at temperature 0 through both
   servings; the share of identical answers.
"""

from __future__ import annotations

import json
import statistics
from collections.abc import Sequence

from .loadtest import Result, run_load

FORCED = {"color": "xq-zebra", "shape": "qq-moon"}
COLORS = ("red", "green", "blue", "yellow", "purple", "orange", "none")
SHAPES = ("circle", "square", "triangle", "diamond", "none")


def schema(colors: Sequence[str], shapes: Sequence[str]) -> dict:
    return {
        "type": "object",
        "properties": {"color": {"enum": list(colors)}, "shape": {"enum": list(shapes)}},
        "required": ["color", "shape"],
        "additionalProperties": False,
    }


def payload(
    model: str,
    prompt: str,
    image_urls: Sequence[str],
    json_schema: dict | None = None,
    logprobs: bool = False,
    max_tokens: int = 48,
) -> dict:
    """Chat request with several images before the text, optionally schema-constrained."""
    content: list[dict] = [{"type": "image_url", "image_url": {"url": u}} for u in image_urls]
    content.append({"type": "text", "text": prompt})
    body: dict = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    if json_schema is not None:
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "attributes", "schema": json_schema},
        }
    if logprobs:
        body["logprobs"] = True
        body["top_logprobs"] = 5
    return body


def parse(text: str | None) -> dict | None:
    try:
        obj = json.loads(text or "")
    except (ValueError, TypeError):
        return None
    return obj if isinstance(obj, dict) else None


def split_logprobs(tokens: Sequence[tuple[str, float]], values: Sequence[str]) -> dict:
    """Separate the log-probs of value tokens from structural ones.

    `tokens` are (token text, log-prob) in output order; a token counts as a value token
    if its characters overlap any occurrence of a chosen value in the output text.
    """
    text = "".join(t for t, _ in tokens)
    spans = []
    for value in values:
        start = text.find(f'"{value}"')
        if start >= 0:
            spans.append((start + 1, start + 1 + len(value)))
    value_lp, structure_lp = [], []
    pos = 0
    for token, lp in tokens:
        end = pos + len(token)
        overlaps = any(pos < b and end > a for a, b in spans)
        (value_lp if overlaps else structure_lp).append(lp)
        pos = end
    return {"value": value_lp, "structure": structure_lp}


def _tokens(raw: dict | None) -> list[tuple[str, float]]:
    content = ((raw or {}).get("logprobs") or {}).get("content") or []
    return [(c.get("token", ""), float(c.get("logprob", 0.0))) for c in content]


def grammar_check(
    base_url: str, model: str, prompt: str, image_urls: Sequence[Sequence[str]],
    timeout_s: float = 300,
) -> dict:
    """Forced-schema enforcement rate, and log-probs on a normal enum schema."""
    forced = [payload(model, prompt, urls, schema([FORCED["color"]], [FORCED["shape"]]))
              for urls in image_urls]
    results, _ = run_load(base_url, forced, concurrency=8, timeout_s=timeout_s)
    answers = [parse(r.text) if r.ok else None for r in results]
    enforced = sum(1 for a in answers if a == FORCED)

    normal = [payload(model, prompt, urls, schema(COLORS, SHAPES), logprobs=True)
              for urls in image_urls]
    results_lp, _ = run_load(base_url, normal, concurrency=8, timeout_s=timeout_s,
                             keep_raw=True)
    valid = 0
    value_lp: list[float] = []
    structure_lp: list[float] = []
    for r in results_lp:
        ans = parse(r.text) if r.ok else None
        if ans and ans.get("color") in COLORS and ans.get("shape") in SHAPES:
            valid += 1
            parts = split_logprobs(_tokens(r.raw), [ans["color"], ans["shape"]])
            value_lp += parts["value"]
            structure_lp += parts["structure"]
    n = len(image_urls)

    def summary(xs: list[float]) -> dict:
        if not xs:
            return {"n": 0}
        return {"n": len(xs), "median": round(statistics.median(xs), 4),
                "min": round(min(xs), 4)}

    structure_ok = bool(structure_lp) and statistics.median(structure_lp) > -0.01
    return {
        "requests": n,
        "forced_enforced_rate": round(enforced / n, 3) if n else None,
        "enum_valid_rate": round(valid / n, 3) if n else None,
        "logprobs_returned": bool(structure_lp or value_lp),
        "structure_logprob": summary(structure_lp),
        "value_logprob": summary(value_lp),
        "structure_tokens_near_zero": structure_ok,
        "ok": n > 0 and enforced == n and valid == n,
    }


def answers(base_url: str, model: str, prompt: str, image_urls: Sequence[Sequence[str]],
            timeout_s: float = 300) -> list[str | None]:
    """Greedy, unconstrained answers, for comparing two servings of the same weights."""
    bodies = [payload(model, prompt, urls) for urls in image_urls]
    results: list[Result]
    results, _ = run_load(base_url, bodies, concurrency=8, timeout_s=timeout_s)
    return [r.text if r.ok else None for r in results]


def match_rate(a: Sequence[str | None], b: Sequence[str | None]) -> float | None:
    pairs = [(x, y) for x, y in zip(a, b, strict=False) if x is not None and y is not None]
    if not pairs:
        return None
    return round(sum(1 for x, y in pairs if x.strip() == y.strip()) / len(pairs), 3)
