#!/usr/bin/env python3
"""Stage 1 data check: download ABO's listings and decide whether Option A's types hold up.

Run:  python3 scripts/stage1/abo_check.py            (standard library only)
      .venv-serve/bin/python scripts/stage1/abo_check.py   (exact Marqo token counts,
                                                          once open_clip is installed)

Downloads (about 90 MB, once, into data/raw/abo/ with SHA-256 checksums):
  archives/abo-listings.tar   listings, 16 JSON-lines files
  images/metadata/images.csv.gz   image id -> path, used to fetch pilot images
Then fetches the small (256 px) images for 20 pilot products for the Gemini mini-pilot.

Writes reports/abo_check.json and reports/abo_check.md. Exit code 0 if every type passes.
ABO is CC BY 4.0: credit Amazon.com for the data and cite the CVPR 2022 paper.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import sys
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from catalog_cascade import abo  # noqa: E402

BUCKET = "https://amazon-berkeley-objects.s3.amazonaws.com"
LISTINGS_URL = f"{BUCKET}/archives/abo-listings.tar"
IMAGES_META_URL = f"{BUCKET}/images/metadata/images.csv.gz"


def download(url: str, dest: Path) -> dict:
    """Fetch once (skip if already complete) and record size and SHA-256."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        tmp = dest.with_suffix(dest.suffix + ".part")
        print(f"Downloading {url} ...", flush=True)
        with urllib.request.urlopen(url, timeout=60) as response, tmp.open("wb") as out:
            while chunk := response.read(1 << 20):
                out.write(chunk)
        tmp.rename(dest)
    digest = hashlib.sha256(dest.read_bytes()).hexdigest()
    return {"url": url, "path": str(dest.relative_to(ROOT)), "bytes": dest.stat().st_size,
            "sha256": digest}


def listing_files(tar_path: Path, out_dir: Path) -> list[Path]:
    """Extract only listings/metadata/*.json.gz, refusing any path that escapes out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    with tarfile.open(tar_path) as tar:
        for member in tar.getmembers():
            name = member.name
            if not member.isfile() or not name.endswith(".json.gz") or "metadata" not in name:
                continue
            target = (out_dir / Path(name).name).resolve()
            if out_dir.resolve() not in target.parents:
                continue
            if not target.exists():
                with tar.extractfile(member) as src:  # type: ignore[union-attr]
                    target.write_bytes(src.read())
            paths.append(target)
    return sorted(paths)


def marqo_token_counter():
    """Marqo's own tokenizer via open_clip, if installed; else None (approximate counts)."""
    try:
        import open_clip  # type: ignore[import-not-found]

        tokenizer = open_clip.get_tokenizer("hf-hub:Marqo/marqo-ecommerce-embeddings-L")
        inner = getattr(tokenizer, "tokenizer", None)
        if inner is None:
            return None
        return lambda text: len(inner.encode(text, add_special_tokens=False))
    except Exception:  # optional: fall back quietly to the approximation
        return None


def fetch_pilot_images(picked: list[dict], meta_path: Path, out_dir: Path) -> list[dict]:
    """Small images (main plus up to one more) for each pilot product."""
    wanted = set()
    for p in picked:
        wanted.add(p["main_image_id"])
        wanted.update((p.get("other_image_id") or [])[:1])
    paths: dict[str, str] = {}
    with gzip.open(meta_path, "rt", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["image_id"] in wanted:
                paths[row["image_id"]] = row["path"]
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for p in picked:
        ids = [p["main_image_id"], *(p.get("other_image_id") or [])[:1]]
        files = []
        for image_id in ids:
            if image_id not in paths:
                continue
            dest = out_dir / f"{image_id}.jpg"
            if not dest.exists():
                with urllib.request.urlopen(f"{BUCKET}/images/small/{paths[image_id]}",
                                            timeout=60) as r:
                    dest.write_bytes(r.read())
            files.append(dest.name)
        truth = {f: abo.english(p.get(f)) for f in abo.FIELDS}
        rows.append({
            "item_id": p["item_id"],
            "product_type": abo.product_type(p),
            "title": abo.english(p.get("item_name")),
            "images": files,
            "seller_values": {k: v for k, v in truth.items() if v},
        })
    with (out_dir / "items.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return rows


def render(report: dict) -> str:
    lines = [f"# ABO data check: {'PASS' if report['ok'] else 'NEEDS A DECISION'}", "",
             "Counts are after de-duplication by item id and main image. English fill is the",
             "share of products with an English value for the field (any `en_*` tag).", "",
             "| Type | Listings | Products | Families (top share) | India share | "
             "Title tokens p50 / p95 | Titles over 64 | Checks |",
             "|---|---|---|---|---|---|---|---|"]
    for t in report["types"]:
        checks = ", ".join(f"{k} {'ok' if v else 'FAIL'}" for k, v in t["checks"].items())
        lines.append(
            f"| {t['product_type']} | {t['listings']} | {t['products']} | {t['families']} "
            f"({t['top_family_share']:.0%}) | {t['india_share']:.0%} | {t['title_tokens_p50']} / "
            f"{t['title_tokens_p95']} | {t['titles_over_limit']} | {checks} |")
    lines += ["", "**English fill per field**", "",
              "| Type | " + " | ".join(abo.FIELDS) + " |",
              "|---|" + "---|" * len(abo.FIELDS)]
    for t in report["types"]:
        lines.append(f"| {t['product_type']} | " +
                     " | ".join(f"{t['english_fill'][f]:.0%}" for f in abo.FIELDS) + " |")
    reach = report["certificate_reach"]
    lines += ["", "**Can each field be certified?** (first estimate)", "",
              f"{reach['calibration_products']} calibration products x an assumed "
              f"{reach['assumed_accept_rate']:.0%} accept rate = {reach['accepted_per_field']} "
              f"accepted answers per field, with the 5% risk split across "
              f"{reach['n_fields']} fields.", ""]
    for rate, row in reach["needed"].items():
        lines.append(f"- At {rate} observed error: needs {row['needed']} -> "
                     f"{'reachable' if row['reachable'] else 'NOT reachable'}")
    note = ("exact (Marqo tokenizer)" if report["token_counts_exact"]
            else "approximate (words x 1.4); rerun with .venv-serve/bin/python for exact counts")
    lines += ["", f"Title token counts are {note}.",
              f"Pilot products picked: {report['pilot_items']} (data/raw/pilot/items.jsonl).", ""]
    if report["failing_types"]:
        lines += ["**Types that fail a check:** " + ", ".join(report["failing_types"]) +
                  ". Options B and C in the design doc are the swaps.", ""]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw-dir", default=str(ROOT / "data" / "raw" / "abo"))
    ap.add_argument("--pilot-n", type=int, default=20)
    ap.add_argument("--n-fields", type=int, default=6, help="fields you expect to certify")
    ap.add_argument("--skip-pilot-images", action="store_true")
    args = ap.parse_args(argv)
    raw = Path(args.raw_dir)

    downloads = [download(LISTINGS_URL, raw / "abo-listings.tar")]
    files = listing_files(raw / "abo-listings.tar", raw / "listings")
    print(f"Reading {len(files)} listing files ...", flush=True)
    groups = abo.group_by_type(abo.iter_listings(files))

    counter = marqo_token_counter()
    stats = [abo.type_stats(t, groups.get(t, []), counter) for t in abo.TYPES]
    capped = sum(min(s.products, abo.CAPS.get(s.product_type, s.products)) for s in stats
                 if s.product_type != "LIGHT_FIXTURE")
    failing = [s.product_type for s in stats if not s.ok and s.product_type != "LIGHT_FIXTURE"]

    pilot_rows: list[dict] = []
    if not args.skip_pilot_images:
        downloads.append(download(IMAGES_META_URL, raw / "images.csv.gz"))
        picked = abo.pick_pilot(groups, n=args.pilot_n)
        pilot_rows = fetch_pilot_images(picked, raw / "images.csv.gz", ROOT / "data" / "raw" /
                                        "pilot")

    report = {
        "ok": not failing,
        "downloads": downloads,
        "types": [{**s.__dict__, "checks": s.checks()} for s in stats],
        "products_option_a_capped": capped,
        "failing_types": failing,
        "certificate_reach": abo.certificate_reach(capped, args.n_fields),
        "token_counts_exact": counter is not None,
        "pilot_items": len(pilot_rows),
    }
    out = ROOT / "reports"
    out.mkdir(exist_ok=True)
    (out / "abo_check.json").write_text(json.dumps(report, indent=2, default=list),
                                        encoding="utf-8")
    markdown = render(report)
    (out / "abo_check.md").write_text(markdown, encoding="utf-8")
    sys.stdout.write(markdown if sys.stdout.encoding.lower().startswith("utf") else
                     markdown.encode("ascii", "replace").decode())
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
