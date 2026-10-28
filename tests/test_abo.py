"""ABO counting logic, on hand-made listings in ABO's format."""

from __future__ import annotations

import gzip
import json

from catalog_cascade import abo


def listing(item_id, ptype="CHAIR", image=None, brand="acme", country="US", **fields):
    row = {
        "item_id": item_id,
        "product_type": [{"value": ptype}],
        "brand": [{"language_tag": "en_US", "value": brand}],
        "item_name": [{"language_tag": "en_US", "value": f"Acme chair {item_id}"}],
        "main_image_id": image if image is not None else f"img{item_id}",
        "country": country,
    }
    for name, value in fields.items():
        row[name] = [{"language_tag": "en_IN", "value": value}]
    return row


def test_english_picks_first_english_value():
    entries = [{"language_tag": "de_DE", "value": "rot"}, {"language_tag": "en_GB", "value": "Red"}]
    assert abo.english(entries) == "Red"
    assert abo.english([{"language_tag": "hi_IN", "value": "x"}]) is None
    assert abo.english(None) is None


def test_dedupe_drops_repeat_items_and_shared_images():
    rows = [listing("a"), listing("a"), listing("b", image="imga"), listing("c")]
    kept = abo.dedupe(rows)
    assert [r["item_id"] for r in kept] == ["a", "c"]


def test_type_stats_and_checks():
    rows = [
        listing(str(i), brand=f"b{i % 6}", country="IN" if i % 2 else "US",
                color="Red", material="Wood", pattern="Solid")
        for i in range(600)
    ]
    stats = abo.type_stats("CHAIR", rows)
    assert stats.products == 600
    assert stats.english_fill["color"] == 1.0
    assert stats.english_fill["style"] == 0.0
    assert stats.families == 6
    assert stats.india_share == 0.5
    assert stats.checks() == {"english_fill": True, "families": True, "size": True}
    assert stats.top_values["material"][0] == ("wood", 600)


def test_type_stats_flags_small_type():
    stats = abo.type_stats("RUG", [listing(str(i), ptype="RUG") for i in range(10)])
    assert stats.checks()["size"] is False
    assert stats.ok is False


def test_title_tokens_with_custom_counter():
    rows = [listing("1")]
    stats = abo.type_stats("CHAIR", rows, count_tokens=lambda t: 100)
    assert stats.token_counts_exact is True
    assert stats.titles_over_limit == 1.0


def test_iter_listings_reads_gzip(tmp_path):
    path = tmp_path / "listings_0.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as f:
        f.write(json.dumps(listing("x")) + "\n\n" + json.dumps(listing("y", ptype="SOFA")) + "\n")
    rows = list(abo.iter_listings([path]))
    groups = abo.group_by_type(rows)
    assert set(groups) == {"CHAIR", "SOFA"}


def test_certificate_reach():
    reach = abo.certificate_reach(8600, n_fields=6)
    assert reach["accepted_per_field"] == round(8600 * 0.25 * 0.70)
    assert reach["needed"]["2%"] == {"needed": 322, "reachable": True}
    assert reach["needed"]["4%"]["reachable"] is False


def test_pick_pilot_spreads_across_types():
    groups = {
        "CHAIR": [listing(f"c{i}", color="Red", material="Wood") for i in range(30)],
        "SOFA": [listing(f"s{i}", ptype="SOFA", color="Grey", pattern="Solid") for i in range(30)],
        "RUG": [listing(f"r{i}", ptype="RUG") for i in range(30)],  # no fields: unusable
    }
    picked = abo.pick_pilot(groups, n=10)
    kinds = {abo.product_type(p) for p in picked}
    assert len(picked) == 10 and kinds == {"CHAIR", "SOFA"}
