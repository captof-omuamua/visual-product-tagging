"""Count what Amazon Berkeley Objects (ABO) listings contain for the chosen product types.

Stage 1 runs this before the product types are fixed (design doc, Data section). It
answers: per type, how many products survive de-duplication, how often each target field
has an English value, how many brand families there are, and how long the titles are
(Marqo's text encoder reads only the first 64 tokens).

Listing format, from the ABO page: one JSON object per line; text fields are lists of
{"language_tag": "en_US", "value": ...}; colour entries may also carry
"standardized_values"; product_type is a list of {"value": "SOFA"}.

Standard library only, so it runs before any environment is set up.
"""

from __future__ import annotations

import gzip
import json
import random
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

from .ltt import accepted_needed

# Option A from the design doc. LIGHT_FIXTURE is counted separately so the merge with
# LAMP is a visible choice, not a silent one.
TYPES = ("SHOES", "CHAIR", "SOFA", "TABLE", "RUG", "LAMP", "LIGHT_FIXTURE")
CAPS = {"SHOES": 3000}  # shoes have ~13k listings; capped so they don't swamp the rest
FIELDS = ("color", "material", "pattern", "style", "fabric_type", "finish_type", "item_shape")
CORE_FIELDS = ("color", "material", "pattern", "style")

# Working rules from the design doc (not published standards).
MIN_ENGLISH_FILL = 0.60
MIN_FIELDS_FILLED = 3
MIN_FAMILIES = 5
MIN_PRODUCTS = 500
MARQO_MAX_TOKENS = 64
CALIBRATION_SHARE = 0.25
ASSUMED_ACCEPT_RATE = 0.70


def iter_listings(paths: Iterable[Path]) -> Iterator[dict]:
    """Every listing in the given .json.gz (or .json) files, one dict per line."""
    for path in paths:
        opener = gzip.open if str(path).endswith(".gz") else open
        with opener(path, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)


def english(entries: object) -> str | None:
    """The first English value in a language-tagged list, or None."""
    if not isinstance(entries, list):
        return None
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        tag = str(entry.get("language_tag") or "")
        value = entry.get("value")
        if tag.lower().startswith("en") and isinstance(value, str) and value.strip():
            return value.strip()
    return None


def product_type(listing: dict) -> str | None:
    entries = listing.get("product_type")
    if isinstance(entries, list) and entries and isinstance(entries[0], dict):
        value = entries[0].get("value")
        return str(value) if value else None
    return None


def brand(listing: dict) -> str | None:
    value = english(listing.get("brand"))
    return value.lower() if value else None


def standardized_colours(listing: dict) -> list[str]:
    out: list[str] = []
    for entry in listing.get("color") or []:
        if isinstance(entry, dict) and str(entry.get("language_tag", "")).lower().startswith("en"):
            out.extend(str(v).lower() for v in entry.get("standardized_values") or [])
    return out


def dedupe(listings: Iterable[dict]) -> list[dict]:
    """Keep the first listing per item_id, then the first per main image.

    The same item can appear under several country sites, and copies of one product can
    share a photo; either kind of duplicate would leak across splits.
    """
    seen_items: set[str] = set()
    seen_images: set[str] = set()
    kept: list[dict] = []
    for listing in listings:
        item_id = listing.get("item_id")
        image = listing.get("main_image_id")
        if not item_id or item_id in seen_items:
            continue
        seen_items.add(item_id)
        if image:
            if image in seen_images:
                continue
            seen_images.add(image)
        kept.append(listing)
    return kept


@dataclass
class TypeStats:
    product_type: str
    listings: int = 0
    products: int = 0  # after de-duplication
    with_main_image: int = 0
    with_english_title: int = 0
    india_share: float = 0.0
    english_fill: dict[str, float] = field(default_factory=dict)
    families: int = 0
    top_family_share: float = 0.0
    title_tokens_p50: float | None = None
    title_tokens_p95: float | None = None
    titles_over_limit: float | None = None
    token_counts_exact: bool = False
    top_values: dict[str, list[tuple[str, int]]] = field(default_factory=dict)

    def checks(self) -> dict[str, bool]:
        filled = sum(1 for f in FIELDS if self.english_fill.get(f, 0) >= MIN_ENGLISH_FILL)
        return {
            "english_fill": filled >= MIN_FIELDS_FILLED,
            "families": self.families >= MIN_FAMILIES,
            "size": self.products >= MIN_PRODUCTS,
        }

    @property
    def ok(self) -> bool:
        return all(self.checks().values())


def _percentile(values: list[int], q: float) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    rank = (len(xs) - 1) * q / 100
    lo = int(rank)
    hi = min(lo + 1, len(xs) - 1)
    return round(xs[lo] + (xs[hi] - xs[lo]) * (rank - lo), 1)


def approx_tokens(text: str) -> int:
    """Rough SentencePiece-style count when the real tokenizer isn't installed (inf)."""
    return max(1, round(len(text.split()) * 1.4))


def type_stats(
    name: str,
    listings: list[dict],
    count_tokens: Callable[[str], int] | None = None,
) -> TypeStats:
    products = dedupe(listings)
    stats = TypeStats(name, listings=len(listings), products=len(products))
    if not products:
        return stats
    n = len(products)
    stats.with_main_image = sum(1 for p in products if p.get("main_image_id"))
    titles = [english(p.get("item_name")) for p in products]
    stats.with_english_title = sum(1 for t in titles if t)
    stats.india_share = round(
        sum(1 for p in products if str(p.get("country", "")).upper() == "IN") / n, 3
    )
    for f in FIELDS:
        values = [english(p.get(f)) for p in products]
        stats.english_fill[f] = round(sum(1 for v in values if v) / n, 3)
        counts = Counter(v.lower() for v in values if v)
        stats.top_values[f] = counts.most_common(20)
    if any(standardized_colours(p) for p in products):
        counts = Counter(c for p in products for c in standardized_colours(p))
        stats.top_values["color_standardized"] = counts.most_common(20)
    brands = Counter(b for b in (brand(p) for p in products) if b)
    stats.families = len(brands)
    stats.top_family_share = round(brands.most_common(1)[0][1] / n, 3) if brands else 0.0
    counter = count_tokens or approx_tokens
    stats.token_counts_exact = count_tokens is not None
    lengths = [counter(t) for t in titles if t]
    stats.title_tokens_p50 = _percentile(lengths, 50)
    stats.title_tokens_p95 = _percentile(lengths, 95)
    if lengths:
        stats.titles_over_limit = round(
            sum(1 for x in lengths if x > MARQO_MAX_TOKENS) / len(lengths), 3
        )
    return stats


def group_by_type(listings: Iterable[dict], types: Iterable[str] = TYPES) -> dict[str, list[dict]]:
    wanted = set(types)
    groups: dict[str, list[dict]] = defaultdict(list)
    for listing in listings:
        t = product_type(listing)
        if t in wanted:
            groups[t].append(listing)
    return groups


def certificate_reach(
    total_products: int,
    n_fields: int,
    calibration_share: float = CALIBRATION_SHARE,
    accept_rate: float = ASSUMED_ACCEPT_RATE,
) -> dict:
    """First estimate of whether each field can be certified (design doc, hand-off rule).

    Accepted answers per field = products x calibration share x accept rate. Compared with
    what Learn-then-Test needs at 2% and 3% observed error, with delta split across fields.
    The accept rate is an assumption until stage 6 measures it.
    """
    accepted = round(total_products * calibration_share * accept_rate)
    out = {
        "calibration_products": round(total_products * calibration_share),
        "assumed_accept_rate": accept_rate,
        "accepted_per_field": accepted,
        "n_fields": n_fields,
        "needed": {},
    }
    for rate in (0.02, 0.03, 0.04):
        _, safe = accepted_needed(rate, n_fields, limit=4000)
        out["needed"][f"{rate:.0%}"] = {"needed": safe, "reachable": safe is not None and
                                        accepted >= safe}
    return out


def pick_pilot(
    groups: dict[str, list[dict]], n: int = 20, seed: int = 20261027
) -> list[dict]:
    """A small, mixed sample for the API mini-pilot: English title, an image, 2+ fields."""
    rng = random.Random(seed)
    usable: dict[str, list[dict]] = {}
    for name, listings in groups.items():
        rows = [
            p
            for p in dedupe(listings)
            if p.get("main_image_id")
            and english(p.get("item_name"))
            and sum(1 for f in CORE_FIELDS if english(p.get(f))) >= 2
        ]
        rng.shuffle(rows)
        usable[name] = rows
    picked: list[dict] = []
    names = [t for t in TYPES if usable.get(t)]
    while len(picked) < n and any(usable[t] for t in names):
        for t in names:
            if usable[t] and len(picked) < n:
                picked.append(usable[t].pop())
    return picked
