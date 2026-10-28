"""One labelling request against a vLLM server: the prompt, the photos, and per-field confidence.

The same prompt is used by the teachers, by Tier 1 in training and by Tier 1 in serving, so
the fine-tuned model sees exactly the question it was trained on.
"""

from __future__ import annotations

import base64
import io
import math
import time
from pathlib import Path

import requests
from PIL import Image

from . import schema
from .checks import _tokens, parse, payload

PROMPT = (
    "You label product attributes for an online catalogue. Using the photos and the title, "
    "give one allowed value per field. Use \"other\" if no allowed value fits and \"none\" "
    "if the field does not apply or cannot be seen. For shoes, style is the kind of shoe; "
    "for home products, it is the design style.\n"
    "Product type: {ptype}\nTitle: {title}\nAllowed values:\n{allowed}\nAnswer in JSON."
)


def prompt_for(product: dict) -> str:
    fields = schema.fields_for(product["product_type"])
    allowed = "\n".join(f"- {f}: {', '.join(v)}" for f, v in fields.items())
    return PROMPT.format(ptype=product["product_type"], title=product["title"], allowed=allowed)


def image_url(path: Path, size: int) -> str:
    """Shrink so the longer side is at most `size` px, and inline it as a JPEG data URL."""
    image = Image.open(path).convert("RGB")
    image.thumbnail((size, size))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def field_confidence(tokens: list[tuple[str, float]], field: str) -> float | None:
    """exp(sum of log-probs) of the tokens inside this field's value.

    The value is found right after its own key ("color": "...") rather than by searching
    for the word, because two fields often share a value such as "none" or "other".
    """
    text = "".join(t for t, _ in tokens)
    key = text.find(f'"{field}"')
    if key < 0:
        return None
    a = text.find('"', text.find(":", key)) + 1  # first character of the value
    b = text.find('"', a)                       # closing quote
    lp, pos = [], 0
    for token, logprob in tokens:
        if pos < b and pos + len(token) > a:   # token overlaps the value's characters
            lp.append(logprob)
        pos += len(token)
    return round(math.exp(sum(lp)), 4) if lp else None


def label(base_url: str, model: str, product: dict, image_dir: Path, size: int) -> dict:
    """One request; the answer, per-field confidence and latency (or the error)."""
    urls = [image_url(image_dir / f"{i}.jpg", size) for i in product["images"]]
    body = payload(model, prompt_for(product), urls,
                   schema.json_schema(product["product_type"]), logprobs=True, max_tokens=128)
    start = time.perf_counter()
    try:
        raw = requests.post(f"{base_url}/v1/chat/completions", json=body, timeout=600).json()
        choice = raw["choices"][0]
    except (requests.RequestException, ValueError, KeyError, IndexError) as exc:
        return {"item_id": product["item_id"], "error": f"{type(exc).__name__}: {exc}"}
    answer = parse(choice["message"]["content"])
    tokens = _tokens(choice)
    return {
        "item_id": product["item_id"],
        "answer": answer,
        "confidence": {f: field_confidence(tokens, f) for f in (answer or {})},
        "latency_s": round(time.perf_counter() - start, 2),
    }
