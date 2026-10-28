"""The certificate arithmetic must reproduce the design doc's table exactly."""

from __future__ import annotations

import pytest

from catalog_cascade.ltt import accepted_needed, binom_cdf, certify, clopper_pearson, p_value

# (observed error, fields) -> (smallest passing count, count from which all larger pass)
DOC_TABLE = {
    (0.00, 1): (59, 59),
    (0.00, 8): (99, 99),
    (0.01, 1): (93, 124),
    (0.02, 1): (181, 208),
    (0.02, 4): (281, 309),
    (0.02, 6): (293, 322),
    (0.02, 8): (331, 359),
    (0.03, 1): (361, 410),
    (0.03, 8): (727, 802),
}


@pytest.mark.parametrize(("key", "expected"), sorted(DOC_TABLE.items()))
def test_accepted_needed_matches_doc(key, expected):
    rate, fields = key
    assert accepted_needed(rate, fields, limit=1500) == expected


def test_binom_cdf_edges():
    assert binom_cdf(-1, 10, 0.3) == 0.0
    assert binom_cdf(10, 10, 0.3) == 1.0
    assert binom_cdf(0, 3, 0.5) == pytest.approx(0.125)


def test_p_value_zero_accepted_is_one():
    assert p_value(0, 0, 0.05) == 1.0


def test_clopper_pearson_matches_doc():
    lo, hi = clopper_pearson(9, 300)
    assert round(lo * 100, 1) == 1.4
    assert round(hi * 100, 1) == 5.6
    lo, hi = clopper_pearson(3, 100)
    assert round(lo * 100, 1) == 0.6
    assert round(hi * 100, 1) == 8.5


def test_certify_picks_loosest_passing_threshold():
    # 300 answers at confidence 0.9 with no errors, then 200 at 0.5 that are all wrong.
    conf = [0.9] * 300 + [0.5] * 200
    correct = [True] * 300 + [False] * 200
    cert = certify(conf, correct, thresholds=[0.95, 0.9, 0.5], delta=0.05)
    assert cert.threshold == 0.9  # 0.95 accepts nobody (skipped); 0.5 fails
    assert cert.accepted == 300 and cert.errors == 0
    assert cert.tested == 2


def test_certify_returns_none_when_nothing_passes():
    conf = [0.9] * 100
    correct = [i % 5 != 0 for i in range(100)]  # 20% wrong
    assert certify(conf, correct, thresholds=[0.9]).threshold is None


def test_certify_length_mismatch():
    with pytest.raises(ValueError):
        certify([0.5], [True, False], thresholds=[0.5])
