# ABO data check: NEEDS A DECISION

Counts are after de-duplication by item id and main image. English fill is the
share of products with an English value for the field (any `en_*` tag).

| Type | Listings | Products | Families (top share) | India share | Title tokens p50 / p95 | Titles over 64 | Checks |
|---|---|---|---|---|---|---|---|
| SHOES | 12965 | 5749 | 47 (22%) | 59% | 25.0 / 33.0 | 0.0 | english_fill FAIL, families ok, size ok |
| CHAIR | 2100 | 1934 | 28 (26%) | 3% | 21.0 / 31.0 | 0.0 | english_fill FAIL, families ok, size ok |
| SOFA | 1199 | 1121 | 11 (32%) | 5% | 22.0 / 32.0 | 0.0 | english_fill FAIL, families ok, size ok |
| TABLE | 936 | 905 | 26 (17%) | 4% | 21.0 / 35.0 | 0.001 | english_fill FAIL, families ok, size ok |
| RUG | 817 | 763 | 6 (41%) | 0% | 24.0 / 33.0 | 0.0 | english_fill ok, families ok, size ok |
| LAMP | 544 | 544 | 10 (27%) | 7% | 28.0 / 37.0 | 0.0 | english_fill FAIL, families ok, size ok |
| LIGHT_FIXTURE | 544 | 515 | 8 (35%) | 3% | 32.0 / 43.0 | 0.0 | english_fill ok, families ok, size ok |

**English fill per field**

| Type | color | material | pattern | style | fabric_type | finish_type | item_shape |
|---|---|---|---|---|---|---|---|
| SHOES | 71% | 36% | 15% | 73% | 9% | 0% | 0% |
| CHAIR | 73% | 55% | 1% | 38% | 11% | 2% | 4% |
| SOFA | 78% | 64% | 6% | 49% | 20% | 0% | 5% |
| TABLE | 64% | 62% | 1% | 52% | 1% | 6% | 33% |
| RUG | 81% | 88% | 52% | 82% | 14% | 0% | 85% |
| LAMP | 82% | 51% | 0% | 46% | 0% | 49% | 36% |
| LIGHT_FIXTURE | 86% | 61% | 0% | 61% | 0% | 50% | 7% |

**Can each field be certified?** (first estimate)

2067 calibration products x an assumed 70% accept rate = 1447 accepted answers per field, with the 5% risk split across 6 fields.

- At 2% observed error: needs 322 -> reachable
- At 3% observed error: needs 739 -> reachable
- At 4% observed error: needs 2853 -> NOT reachable

Title token counts are exact (Marqo tokenizer).
Pilot products picked: 20 (data/raw/pilot/items.jsonl).

**Types that fail a check:** SHOES, CHAIR, SOFA, TABLE, LAMP. Options B and C in the design doc are the swaps.

