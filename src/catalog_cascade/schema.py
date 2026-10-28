"""Value lists and synonym maps per field (v0, built from ABO's most common seller values).

Every field's answer is one value from its closed list, or "other" (doesn't fit the list)
or "none" (doesn't apply, or can't be seen). `normalize` turns a seller's free text into a
list value, or None when no list word or synonym appears in it.
"""

from __future__ import annotations

import re

OTHER, NONE = "other", "none"
FOOTWEAR = {"SHOES"}

COLORS = ["black", "white", "grey", "brown", "beige", "blue", "green", "red", "pink",
          "yellow", "orange", "purple", "gold", "silver", "bronze", "clear", "multicolor"]
MATERIALS = ["wood", "metal", "glass", "plastic", "stone", "ceramic", "fabric", "wool",
             "cotton", "leather", "faux leather", "suede", "mesh", "rubber", "natural fiber",
             "wicker"]
PATTERNS = ["solid", "striped", "checked", "floral", "geometric", "abstract", "medallion",
            "textured", "color block"]
HOME_STYLES = ["modern", "contemporary", "mid-century", "transitional", "traditional",
               "rustic", "industrial", "glam", "bohemian", "coastal", "scandinavian"]
SHOE_STYLES = ["athletic", "sneaker", "formal", "loafer", "flat", "sandal", "boot", "slipper",
               "heel"]
SHAPES = ["rectangular", "round", "square", "oval", "runner"]
FINISHES = ["brushed", "matte", "polished", "satin", "oil-rubbed", "antique", "glossy"]

# Seller words -> list value. List values also match themselves.
SYNONYMS = {
    "color": {"gray": "grey", "charcoal": "grey", "slate": "grey", "graphite": "grey",
              "navy": "blue", "denim": "blue", "teal": "blue", "aqua": "blue",
              "tan": "brown", "cognac": "brown", "saddle": "brown", "espresso": "brown",
              "walnut": "brown", "chocolate": "brown", "chestnut": "brown", "cherry": "brown",
              "cream": "beige", "ivory": "beige", "off-white": "beige", "taupe": "beige",
              "sand": "beige", "khaki": "beige", "nude": "beige", "natural": "beige",
              "olive": "green", "sage": "green", "burgundy": "red", "maroon": "red",
              "blush": "pink", "mustard": "yellow", "rust": "orange", "lavender": "purple",
              "nickel": "silver", "chrome": "silver", "pewter": "silver", "steel": "silver",
              "brass": "gold", "multi": "multicolor", "multi-colored": "multicolor",
              "multicolour": "multicolor", "transparent": "clear"},
    "material": {"polyester": "fabric", "textile": "fabric", "canvas": "fabric",
                 "linen": "fabric", "velvet": "fabric", "nylon": "fabric", "knit": "fabric",
                 "denim": "fabric", "microfiber": "fabric", "chenille": "fabric",
                 "polyurethane": "faux leather", "pu": "faux leather",
                 "leatherette": "faux leather", "vegan leather": "faux leather",
                 "engineered wood": "wood", "mdf": "wood", "pine": "wood", "oak": "wood",
                 "hardwood": "wood", "beech": "wood", "teak": "wood", "walnut": "wood",
                 "mango wood": "wood", "plywood": "wood", "aluminum": "metal",
                 "aluminium": "metal", "steel": "metal", "iron": "metal", "brass": "metal",
                 "polypropylene": "plastic", "acrylic": "plastic", "resin": "plastic",
                 "jute": "natural fiber", "sisal": "natural fiber", "seagrass": "natural fiber",
                 "bamboo": "natural fiber", "rattan": "wicker", "marble": "stone",
                 "granite": "stone", "concrete": "stone", "porcelain": "ceramic"},
    "pattern": {"plain": "solid", "plain-coloured": "solid", "stripe": "striped",
                "stripes": "striped", "plaid": "checked", "checkered": "checked",
                "trellis": "geometric", "moroccan": "geometric", "chevron": "geometric",
                "ikat": "geometric", "geo": "geometric", "oriental": "medallion",
                "persian": "medallion"},
    "home_style": {"mid-century modern": "mid-century", "farmhouse": "rustic",
                   "farm_house": "rustic", "classic": "traditional", "boho": "bohemian",
                   "glam": "glam", "nautical": "coastal"},
    "shoe_style": {"running": "athletic", "walking": "athletic", "training": "athletic",
                   "sneakers": "sneaker", "low-top": "sneaker", "derby": "formal",
                   "oxford": "formal", "brogue": "formal", "monk": "formal",
                   "loafers": "loafer", "moccasin": "loafer", "ballet": "flat",
                   "flats": "flat", "ankle-strap": "sandal", "flip-flops": "sandal",
                   "thong": "sandal", "sandals": "sandal", "floaters": "sandal",
                   "boots": "boot", "slippers": "slipper", "pumps": "heel",
                   "stiletto": "heel"},
    "shape": {"rectangle": "rectangular", "rectanglular": "rectangular",
              "circle": "round", "circular": "round"},
    "finish": {"oil rubbed": "oil-rubbed", "high-gloss": "glossy", "gloss": "glossy",
               "lacquered": "glossy", "distressed": "antique"},
}


def fields_for(ptype: str) -> dict[str, list[str]]:
    """The fields asked for this product type, each with its closed list."""
    fields = {"color": COLORS, "material": MATERIALS, "pattern": PATTERNS,
              "style": SHOE_STYLES if ptype in FOOTWEAR else HOME_STYLES}
    if ptype in ("RUG", "TABLE"):
        fields["shape"] = SHAPES
    if ptype in ("LAMP", "LIGHT_FIXTURE"):
        fields["finish"] = FINISHES
    return fields


def json_schema(ptype: str) -> dict:
    """JSON schema whose values must come from each field's list, "other" or "none"."""
    props = {f: {"enum": [*v, OTHER, NONE]} for f, v in fields_for(ptype).items()}
    return {"type": "object", "properties": props, "required": list(props),
            "additionalProperties": False}


def _synonym_table(field: str, ptype: str) -> dict[str, str]:
    if field == "style":
        key = "shoe_style" if ptype in FOOTWEAR else "home_style"
    else:
        key = field
    values = fields_for(ptype).get(field, [])
    return {**{v: v for v in values}, **SYNONYMS.get(key, {})}


def normalize(field: str, ptype: str, text: str | None) -> str | None:
    """The list value named earliest in the seller's text; the longer phrase wins a tie.

    "Grey Linen" -> grey; "velvet, hardwood frame, metal legs" -> fabric (velvet comes
    first); "Synthetic" -> None (no list word, so it gives no vote).
    """
    if not text:
        return None
    text = text.lower()
    best: tuple[int, int, str] | None = None
    for word, value in _synonym_table(field, ptype).items():
        match = re.search(rf"(?<![a-z]){re.escape(word)}(?![a-z])", text)
        if match:
            rank = (match.start(), -len(word), value)
            if best is None or rank < best:
                best = rank
    return best[2] if best else None
