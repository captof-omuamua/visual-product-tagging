#!/usr/bin/env python3
"""Results on ImplicitAVE's 1,610 hand-checked test questions: models, cascade, cost, chart.

The cascade: the fine-tuned small model answers every question; when its confidence is
below a threshold, the question goes to the big model (Gemma 4 26B) and its answer is used.
Each threshold gives one (accuracy, cost) point; the best ones form the frontier.

Score = share of questions answered with exactly the hand-checked value. For one answer per
question this equals the paper's micro-F1 (Zou et al. 2024, Table 4), so the rows compare.

Cost per 1,000 questions = measured GPU time on this machine (RTX 5090 laptop, batches of
32) x an assumed cloud rental price for one comparable GPU (PRICE_USD_PER_HOUR, below).

Run: .venv-api/bin/python scripts/implicitave/analyze.py
Out: reports/implicitave/results.md, results.json, scores.png, review.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from catalog_cascade import implicitave as ia  # noqa: E402

OUT = ROOT / "reports" / "implicitave"
DOMAINS = ["Clothing", "Footwear", "Jewelry&GA", "Food", "Home"]
PRICE_USD_PER_HOUR = 0.60  # assumption: rented single 24-32 GB GPU; state it, change it here
USD_TO_INR = 95.46
SMALL, BIG = "qwen35_4b_finetuned", "gemma4_26b_zeroshot"
RUNS = {  # file name -> label
    "qwen35_4b_zeroshot": "Qwen3.5-4B, zero-shot (ours)",
    "qwen3vl_8b_zeroshot": "Qwen3-VL-8B, zero-shot (ours)",
    "gemma4_26b_zeroshot": "Gemma 4 26B-A4B, zero-shot (ours)",
    "qwen35_4b_finetuned": "Qwen3.5-4B, fine-tuned (ours)",
}
PAPER = {  # Zou et al. 2024, Table 4 (micro-F1): Clothing, Footwear, Jewelry&GA, Food, Home, All
    "GPT-4V (paper)": [77.43, 81.39, 90.45, 90.77, 89.93, 86.77],
    "LLaVA-1.5-13B (paper)": [49.12, 63.72, 81.81, 76.15, 80.31, 71.86],
    "Qwen-VL-7B (paper)": [59.73, 57.72, 84.09, 76.92, 73.96, 70.86],
    "LAVIN, fine-tuned (paper's previous best)": [65.93, 75.39, 78.64, 60.77, 64.33, 67.83],
}


def scores(correct: list[bool], rows: list[dict]) -> list[float]:
    """Per-domain and overall accuracy, in %."""
    out = []
    for d in DOMAINS:
        hits = [c for c, r in zip(correct, rows, strict=True) if r["domain"] == d]
        out.append(round(100 * sum(hits) / len(hits), 2))
    out.append(round(100 * sum(correct) / len(correct), 2))
    return out


THRESHOLDS = [i / 100 for i in range(0, 100)] + [0.995, 0.999]


def cascade(small: list[dict], big: list[dict], rows: list[dict], t: float) -> dict:
    """Score when the small model's answer is kept if its confidence >= t, else the big one's."""
    correct, sent = [], 0
    for s, b, r in zip(small, big, rows, strict=True):
        if (s.get("conf") or 0) >= t:
            correct.append(s.get("pred") == r["value"])
        else:
            sent += 1
            correct.append(b.get("pred") == r["value"])
    return {"threshold": t, "acc": round(100 * sum(correct) / len(rows), 2),
            "to_big": round(100 * sent / len(rows), 1), "scores": scores(correct, rows)}


def accepted(preds: list[dict], rows: list[dict], t: float) -> tuple[float, float]:
    """(share accepted, accuracy among accepted), in %, when answers with conf >= t are kept."""
    keep = [p.get("pred") == r["value"]
            for p, r in zip(preds, rows, strict=True) if (p.get("conf") or 0) >= t]
    if not keep:
        return 0.0, 0.0
    return 100 * len(keep) / len(rows), 100 * sum(keep) / len(keep)


def auto_accept(test_preds, test_rows, val_preds, val_rows, targets=(90, 95, 98)) -> list:
    """Per target: the lowest cut-off reaching it on validation, then its result on test."""
    out = []
    for target in targets:
        t = next((t for t in THRESHOLDS if accepted(val_preds, val_rows, t)[1] >= target),
                 THRESHOLDS[-1])
        coverage, accuracy = accepted(test_preds, test_rows, t)
        out.append({"target": target, "threshold": t, "coverage": round(coverage, 1),
                    "accuracy": round(accuracy, 1)})
    return out


def inr_per_1000(seconds_per_item: float) -> float:
    return round(seconds_per_item * 1000 / 3600 * PRICE_USD_PER_HOUR * USD_TO_INR, 2)


def main() -> int:
    rows = ia.load("test")
    runs = {}
    for name in RUNS:
        path = OUT / f"{name}.json"
        if path.exists():
            data = json.loads(path.read_text())
            by_id = {p["id"]: p for p in data["preds"]}
            runs[name] = {"preds": [by_id.get(r["id"], {}) for r in rows],
                          "sec_per_item": data["seconds"] / data["items"]}
    table = []
    for name, run in runs.items():
        correct = [p.get("pred") == r["value"] for p, r in zip(run["preds"], rows, strict=True)]
        table.append({"setup": RUNS[name], "scores": scores(correct, rows),
                      "to_big": 0.0 if name != BIG else 100.0,
                      "inr_per_1000": inr_per_1000(run["sec_per_item"])})

    frontier, best = [], None
    if SMALL in runs and BIG in runs:
        # The threshold is chosen on validation questions (machine-labelled, never on the
        # test set): the highest validation score, fewest hand-offs on a tie.
        val_rows = ia.load("val")
        val = {n: {p["id"]: p for p in json.loads((OUT / f"val_{n}.json").read_text())["preds"]}
               for n in (SMALL, BIG)}
        val_rows = [r for r in val_rows if r["id"] in val[SMALL] and r["id"] in val[BIG]]
        val_curve = [cascade([val[SMALL][r["id"]] for r in val_rows],
                             [val[BIG][r["id"]] for r in val_rows], val_rows, t)
                     for t in THRESHOLDS]
        chosen = max(val_curve, key=lambda f: (f["acc"], -f["to_big"]))["threshold"]

        small, big = runs[SMALL], runs[BIG]
        for t in THRESHOLDS:
            f = cascade(small["preds"], big["preds"], rows, t)
            f["inr_per_1000"] = inr_per_1000(small["sec_per_item"]
                                             + f["to_big"] / 100 * big["sec_per_item"])
            frontier.append(f)
        best = next(f for f in frontier if f["threshold"] == chosen)
        table.append({"setup": f"Cascade: fine-tuned 4B -> Gemma when confidence < "
                               f"{chosen:.2f}, threshold set on validation (ours)",
                      "scores": best["scores"], "to_big": best["to_big"],
                      "inr_per_1000": best["inr_per_1000"]})

    lines = ["# ImplicitAVE results (1,610 hand-checked test questions)", "",
             "Score = % answered with the hand-checked value (= the paper's micro-F1). "
             "Cost = measured GPU time x an assumed rental price of "
             f"${PRICE_USD_PER_HOUR}/h for one GPU (₹{USD_TO_INR}/$). Paper rows from "
             "Zou et al., ACL Findings 2024, Table 4 (no cost reported).", "",
             "| Setup | " + " | ".join(DOMAINS) + " | **All** | To big model | ₹ / 1,000 |",
             "|---|" + "---|" * (len(DOMAINS) + 3)]
    for row in table:
        s = row["scores"]
        lines.append(f"| {row['setup']} | " + " | ".join(f"{v:.1f}" for v in s[:-1])
                     + f" | **{s[-1]:.1f}** | {row['to_big']:.0f}% | {row['inr_per_1000']:.2f} |")
    for name, s in PAPER.items():
        lines.append(f"| {name} | " + " | ".join(f"{v:.1f}" for v in s[:-1])
                     + f" | **{s[-1]:.1f}** | - | - |")
    if frontier:
        lines += ["", "## Cascade: accuracy vs share sent to the big model", "",
                  "| Confidence threshold | Score (All) | Sent to Gemma | ₹ / 1,000 |",
                  "|---|---|---|---|"]
        for t in (0.5, 0.8, 0.9, 0.95, 0.99, 0.999):
            f = min(frontier, key=lambda x: abs(x["threshold"] - t))
            lines.append(f"| {f['threshold']} | {f['acc']:.1f} | {f['to_big']:.0f}% | "
                         f"{f['inr_per_1000']:.2f} |")
    review = []
    if SMALL in runs:
        review = auto_accept(runs[SMALL]["preds"], rows,
                             [val[SMALL][r["id"]] for r in val_rows], val_rows)
        lines += ["", "## Auto-accept or flag for review (fine-tuned 4B alone)", "",
                  "Keep the answers whose confidence clears a cut-off chosen on validation "
                  "questions to hit the target accuracy; flag the rest for a person.", "",
                  "| Target accuracy | Cut-off | Test: auto-accepted | Test: accuracy of "
                  "accepted |", "|---|---|---|---|"]
        for x in review:
            lines.append(f"| {x['target']}% | {x['threshold']} | {x['coverage']:.0f}% | "
                         f"{x['accuracy']:.1f}% |")
    (OUT / "results.md").write_text("\n".join(lines) + "\n")
    (OUT / "results.json").write_text(json.dumps({"table": table, "frontier": frontier,
                                                  "auto_accept": review, "paper": PAPER},
                                                 indent=2))
    print("\n".join(lines))
    if frontier:
        charts(table, runs[SMALL]["preds"], rows)
    return 0


INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
OURS, PAPER_GREY = "#2a78d6", "#b9b8b2"  # categorical slot 1 for ours; neutral for the paper's


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)


def charts(table: list[dict], test_preds: list[dict], rows: list[dict]) -> None:
    """scores.png: every setup's score (ours vs the paper's); review.png: auto-accept curve."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 1. Score by setup, best at the top. Ours in blue with cost; the paper's in grey.
    bars = [(r["setup"].replace(" (ours)", ""), r["scores"][-1], r["inr_per_1000"], True)
            for r in table if not r["setup"].startswith("Cascade")]
    bars += [(name, s[-1], None, False) for name, s in PAPER.items()]
    bars.sort(key=lambda b: b[1])
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    y = range(len(bars))
    ax.barh(y, [b[1] for b in bars], height=0.62, color=[OURS if b[3] else PAPER_GREY
                                                        for b in bars])
    for i, (_, score, cost, ours) in enumerate(bars):
        note = f"{score:.1f}" + (f"   ₹{cost:.2f} / 1,000" if ours else "")
        ax.text(score + 0.6, i, note, va="center", fontsize=8, color=INK)
    ax.set_yticks(list(y), [b[0] for b in bars], fontsize=8, color=INK)
    ax.set_xlim(0, 118)  # room for the labels; ticks stop at 100
    ax.set_xticks(range(0, 101, 20))
    ax.set_xlabel("Score on 1,610 hand-checked questions (%)", color=INK, fontsize=9)
    ax.set_title("ImplicitAVE: our local models (blue) vs published results (grey)",
                 color=INK, loc="left", fontsize=10)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT / "scores.png")
    plt.close(fig)

    # 2. Auto-accept: share of questions kept vs accuracy of the kept ones, over all cut-offs.
    pts = [accepted(test_preds, rows, t) for t in THRESHOLDS]
    fig, ax = plt.subplots(figsize=(6.4, 3.8), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    ax.plot([p[0] for p in pts], [p[1] for p in pts], color=OURS, lw=2)
    ax.axhline(95, color=MUTED, lw=1, ls=(0, (4, 3)))
    ax.text(2, 95.3, "95% accuracy", fontsize=8, color=MUTED)
    ax.set_xlabel("Questions auto-accepted (%)  —  the rest go to a person", color=INK,
                  fontsize=9)
    ax.set_ylabel("Accuracy of accepted (%)", color=INK, fontsize=9)
    ax.set_title("Fine-tuned Qwen3.5-4B: auto-accept vs accuracy (test)", color=INK,
                 loc="left", fontsize=10)
    ax.set_xlim(0, 101)
    ax.grid(color=GRID, lw=0.8)
    fig.tight_layout()
    fig.savefig(OUT / "review.png")
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
