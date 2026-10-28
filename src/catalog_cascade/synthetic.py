"""Synthetic smoke-test images: one coloured shape per image, with its exact label.

Stage 1 needs data that (a) needs no download, (b) has one right answer, and
(c) gives a LoRA adapter something visible to learn in a few steps: an exact
JSON format. Real catalogue data (Amazon Berkeley Objects) arrives in stage 2.
"""

from __future__ import annotations

import base64
import io
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw

COLORS: dict[str, tuple[int, int, int]] = {
    "red": (220, 40, 40),
    "green": (40, 160, 70),
    "blue": (40, 90, 220),
    "yellow": (240, 200, 30),
    "purple": (140, 60, 180),
    "orange": (245, 130, 30),
}
SHAPES: tuple[str, ...] = ("circle", "square", "triangle", "diamond")

# 448 px is a multiple of both Qwen2.5-VL's 28 px tokens (256 visual tokens)
# and Qwen3-VL's 32 px tokens (196 visual tokens), so neither model resizes it.
IMAGE_SIZE = 448

# Deliberately loose wording: the base model answers in its own format, while the
# adapter is trained to answer in one exact format. That gap is how the serving
# test can tell whether vLLM really applied the adapter.
PROMPT = "What colour and shape is the object in this product photo? Answer in JSON."

TRAIN_SEED = 7
EVAL_SEED = 1234


@dataclass(frozen=True)
class Item:
    image: Image.Image
    color: str
    shape: str

    @property
    def target(self) -> str:
        """The exact string the adapter is trained to produce."""
        return json.dumps({"color": self.color, "shape": self.shape})


def draw_item(color: str, shape: str, rng: random.Random, size: int = IMAGE_SIZE) -> Item:
    """Draw one shape at a random position and size on a light background."""
    background = tuple(rng.randint(225, 250) for _ in range(3))
    image = Image.new("RGB", (size, size), background)
    draw = ImageDraw.Draw(image)
    radius = rng.randint(size // 6, size // 3)
    cx = rng.randint(radius + 8, size - radius - 8)
    cy = rng.randint(radius + 8, size - radius - 8)
    fill = COLORS[color]
    if shape == "circle":
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=fill)
    elif shape == "square":
        draw.rectangle((cx - radius, cy - radius, cx + radius, cy + radius), fill=fill)
    elif shape == "triangle":
        angles = (-math.pi / 2, math.pi / 6, 5 * math.pi / 6)  # an upright equilateral triangle
        points = [(cx + radius * math.cos(a), cy + radius * math.sin(a)) for a in angles]
        draw.polygon(points, fill=fill)
    elif shape == "diamond":
        points = [(cx, cy - radius), (cx + radius, cy), (cx, cy + radius), (cx - radius, cy)]
        draw.polygon(points, fill=fill)
    else:
        raise ValueError(f"unknown shape: {shape}")
    return Item(image, color, shape)


def make_items(n_per_combo: int, seed: int, size: int = IMAGE_SIZE) -> list[Item]:
    """Every colour x shape combination n_per_combo times, shuffled reproducibly."""
    rng = random.Random(seed)
    items = [
        draw_item(color, shape, rng, size)
        for color in COLORS
        for shape in SHAPES
        for _ in range(n_per_combo)
    ]
    rng.shuffle(items)
    return items


def views(item: Item, k: int, seed: int, size: int = IMAGE_SIZE) -> list[Image.Image]:
    """k photos of the same product: the item's image plus k-1 redraws of the same object.

    Real listings have 1-3 images; stage 1 uses this to measure memory at that load.
    """
    rng = random.Random(f"{seed}-{item.color}-{item.shape}")
    extra = [draw_item(item.color, item.shape, rng, size).image for _ in range(max(0, k - 1))]
    return [item.image, *extra]


def png_base64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def data_url(image: Image.Image) -> str:
    """Inline the image in the request, so the server needs no file access."""
    return "data:image/png;base64," + png_base64(image)


def save_items(items: list[Item], out_dir: Path) -> Path:
    """Write the PNGs plus a JSONL manifest, for inspecting what the model saw."""
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / "items.jsonl"
    with manifest.open("w", encoding="utf-8") as f:
        for i, item in enumerate(items):
            name = f"{i:04d}_{item.color}_{item.shape}.png"
            item.image.save(out_dir / name)
            row = {"image": name, "color": item.color, "shape": item.shape, "target": item.target}
            f.write(json.dumps(row) + "\n")
    return manifest
