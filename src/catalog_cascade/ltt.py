"""Learn-then-Test arithmetic for the per-field error certificate (standard library only).

The promise: among the answers a field accepts, at most `alpha` are wrong, and this holds
with probability at least 1 - delta over the draw of the calibration set. Learn-then-Test
(Angelopoulos et al., arXiv 2110.01052) treats each candidate threshold as a hypothesis
"the true error is above alpha" and rejects it with an exact binomial p-value.

Stage 1 uses `accepted_needed` to check whether a field can be certified at all. Stage 6
uses `certify` on the real calibration split.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


def binom_cdf(k: int, n: int, p: float) -> float:
    """P(X <= k) for X ~ Binomial(n, p), summed in log space so large n stays accurate."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    log_terms = [
        math.lgamma(n + 1)
        - math.lgamma(i + 1)
        - math.lgamma(n - i + 1)
        + i * math.log(p)
        + (n - i) * math.log1p(-p)
        for i in range(k + 1)
    ]
    top = max(log_terms)
    return min(1.0, math.exp(top) * sum(math.exp(t - top) for t in log_terms))


def p_value(errors: int, accepted: int, alpha: float) -> float:
    """Exact binomial p-value for "true error rate > alpha", given the observed errors.

    Small means: seeing this few errors would be unlikely if the error rate were above alpha.
    """
    if accepted == 0:
        return 1.0
    return binom_cdf(errors, accepted, alpha)


def passes(accepted: int, error_rate: float, alpha: float, delta: float) -> bool:
    """Whether `accepted` answers at `error_rate` observed error pass the test at level delta.

    Errors are counted as ceil(rate x n), the convention behind the design doc's table.
    """
    errors = math.ceil(error_rate * accepted - 1e-12)
    return p_value(errors, accepted, alpha) <= delta


def accepted_needed(
    error_rate: float,
    n_fields: int = 1,
    alpha: float = 0.05,
    delta: float = 0.05,
    limit: int = 20_000,
) -> tuple[int | None, int | None]:
    """(smallest passing count, count from which every larger count passes).

    delta is split equally across `n_fields` (Bonferroni). Because errors come in whole
    numbers, a few counts just above the minimum can fail again, hence the second number.
    Returns (None, None) if nothing up to `limit` passes.
    """
    level = delta / n_fields
    first = next((n for n in range(1, limit) if passes(n, error_rate, alpha, level)), None)
    if first is None:
        return None, None
    last_fail = None
    for n in range(first, limit):
        if not passes(n, error_rate, alpha, level):
            last_fail = n
    return first, (last_fail + 1 if last_fail is not None else first)


@dataclass(frozen=True)
class Certificate:
    threshold: float | None  # None: no threshold could be certified
    accepted: int
    errors: int
    p_value: float | None
    tested: int  # thresholds tested before stopping


def certify(
    confidences: Sequence[float],
    correct: Sequence[bool],
    thresholds: Sequence[float],
    alpha: float = 0.05,
    delta: float = 0.05,
    min_accepted: int = 25,
) -> Certificate:
    """Fixed-sequence Learn-then-Test over thresholds, from strictest to loosest.

    An answer is accepted when its confidence is at least the threshold. Thresholds that
    accept fewer than `min_accepted` answers are skipped (LTT's own example used 25). Testing
    stops at the first threshold that fails; the last one that passed is returned.
    `delta` here is this field's share of the overall budget.
    """
    if len(confidences) != len(correct):
        raise ValueError("confidences and correct must be the same length")
    best: Certificate = Certificate(None, 0, 0, None, 0)
    tested = 0
    for t in sorted(set(thresholds), reverse=True):
        accepted = [ok for c, ok in zip(confidences, correct, strict=True) if c >= t]
        if len(accepted) < min_accepted:
            continue
        tested += 1
        errors = sum(1 for ok in accepted if not ok)
        p = p_value(errors, len(accepted), alpha)
        if p > delta:
            break
        best = Certificate(t, len(accepted), errors, p, tested)
    return Certificate(best.threshold, best.accepted, best.errors, best.p_value, tested)


def clopper_pearson(errors: int, n: int, level: float = 0.95) -> tuple[float, float]:
    """Exact two-sided interval for an error rate, e.g. how wrong a label set is on gold."""
    if n <= 0:
        raise ValueError("n must be positive")
    tail = (1 - level) / 2

    def bisect(too_low) -> float:
        """The p where `too_low(p)` flips from True to False."""
        lo, hi = 0.0, 1.0
        for _ in range(100):
            mid = (lo + hi) / 2
            if too_low(mid):
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    # Lower end: the p at which seeing `errors` or more becomes as likely as the tail.
    lower = 0.0 if errors == 0 else bisect(lambda p: 1 - binom_cdf(errors - 1, n, p) < tail)
    # Upper end: the p at which seeing `errors` or fewer becomes as unlikely as the tail.
    upper = 1.0 if errors == n else bisect(lambda p: binom_cdf(errors, n, p) > tail)
    return lower, upper
