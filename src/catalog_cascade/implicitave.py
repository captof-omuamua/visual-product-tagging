"""ImplicitAVE (Zou et al., ACL Findings 2024): loading, the question, and scoring.

Each row asks for one attribute of one product (e.g. Footwear / "Boot Style" -> "Rain Boot")
from its photo and title. The value is never written in the title, so it must be inferred.
The answer is one value from the attribute's fixed list; the test set's 1,610 answers were
re-checked by hand by the dataset's authors.

Data lives in data/raw/implicitave/ (repo/ = the authors' GitHub repo; train_images/ = their
Google Drive images). We evaluate on it and cite it; we don't republish it.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "implicitave"
DOMAINS = {  # tsv name -> (paper's column name, options file, training image folder)
    "Clothing": ("Clothing", "Clothing_Shoes_and_Jewelry", "Clothing_Shoes_and_Jewelry"),
    "Footwear": ("Footwear", "Clothing_Shoes_and_Jewelry", "Clothing_Shoes_and_Jewelry"),
    "Jewlery_and_General_Apparel": ("Jewelry&GA", "Clothing_Shoes_and_Jewelry",
                                    "Clothing_Shoes_and_Jewelry"),
    "Food": ("Food", "Food", "Grocery_and_Gourmet_Food"),
    "Home": ("Home", "Home", "Home_and_Kitchen"),
}
PROMPT = ("Product category: {category}\nTitle: {title}\n"
          "Question: Looking at the photo and the title, what is the product's {attribute}?\n"
          "Options: {options}\nAnswer in JSON.")


def options() -> dict[str, dict[str, list[str]]]:
    """Allowed values per options file and attribute."""
    out = {}
    for name in {v[1] for v in DOMAINS.values()}:
        path = RAW / "repo" / "data" / "evaluation_data" / f"options_{name}.json"
        out[name] = json.loads(path.read_text())
    return out


def load(split: str) -> list[dict]:
    """Rows of "test" (the 1,610 hand-checked items), "train" or "val", with allowed values.

    Training rows whose value isn't in the attribute's list, or whose photo is missing, are
    dropped (they can't be asked as a closed question).
    """
    opts = options()
    rows = []
    for tsv, (domain, opt_file, image_dir) in DOMAINS.items():
        if split == "test":
            evaluation = RAW / "repo" / "data" / "evaluation_data"
            path = evaluation / "texts" / f"{tsv}_annotated_final.tsv"
            images = evaluation / "images"
        else:  # "train" or "val": the authors' splits of the machine-labelled data
            path = RAW / "repo" / "data" / "training_data" / "texts" / tsv / f"{split}.tsv"
            images = RAW / "train_images" / image_dir
        with path.open(encoding="utf-8") as f:
            for r in csv.DictReader(f, delimiter="\t"):
                allowed = opts[opt_file].get(r["attribute_names"])
                image = images / r["images"]
                if not allowed or r["attribute_values"] not in allowed or not image.exists():
                    continue
                rows.append({
                    "id": f"{r['id']}|{r['attribute_names']}",
                    "domain": domain,
                    "category": r["category"],
                    "title": r["texts"].removeprefix("[Title]").strip(),
                    "attribute": r["attribute_names"],
                    "value": r["attribute_values"],
                    "options": allowed,
                    "image": str(image),
                })
    return rows


def prompt(row: dict) -> str:
    return PROMPT.format(category=row["category"], title=row["title"],
                         attribute=row["attribute"], options=", ".join(row["options"]))


def schema(row: dict) -> dict:
    """The answer must be one of the row's allowed values."""
    return {"type": "object", "properties": {"value": {"enum": row["options"]}},
            "required": ["value"], "additionalProperties": False}
