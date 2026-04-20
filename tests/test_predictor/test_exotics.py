"""Tests for Harville exotic-probability math.

All identities here come directly from the Harville (1973) conditional
model — they must hold for *any* valid probability distribution.
"""

import pytest

from ganyan.predictor.exotics import (
    cumulative_coverage,
    dortlu_probabilities,
    ganyan_probabilities,
    ikili_probabilities,
    plase_probabilities,
    sirali_ikili_probabilities,
    uclu_probabilities,
)


# A 4-horse race, easy to reason about by hand.
WIN_PROBS = {1: 0.50, 2: 0.25, 3: 0.15, 4: 0.10}


def _approx(x: float, target: float, tol: float = 1e-9) -> bool:
    return abs(x - target) < tol


# --- Ganyan ---------------------------------------------------------------


def test_ganyan_sorted_descending():
    combos = ganyan_probabilities(WIN_PROBS)
    probs = [c.probability for c in combos]
    assert probs == sorted(probs, reverse=True)
    assert combos[0].horses == (1,)
    assert _approx(combos[0].probability, 0.50)


def test_ganyan_normalises_to_one():
    unnorm = {1: 5.0, 2: 2.5, 3: 1.5, 4: 1.0}  # sums to 10
    combos = ganyan_probabilities(unnorm)
    total = sum(c.probability for c in combos)
    assert _approx(total, 1.0)


# --- Sıralı İkili (ordered pairs) -----------------------------------------


def test_sirali_ikili_sum_equals_one():
    """All ordered-pair probabilities must sum to 1 (partition of top-2 space)."""
    combos = sirali_ikili_probabilities(WIN_PROBS)
    assert _approx(sum(c.probability for c in combos), 1.0)


def test_sirali_ikili_hand_check():
    """P(1st=1, 2nd=2) = 0.50 * 0.25 / (1 - 0.50) = 0.25"""
    combos = sirali_ikili_probabilities(WIN_PROBS)
    top = next(c for c in combos if c.horses == (1, 2))
    assert _approx(top.probability, 0.25)


def test_sirali_ikili_excludes_same_horse():
    combos = sirali_ikili_probabilities(WIN_PROBS)
    for c in combos:
        assert c.horses[0] != c.horses[1]


# --- İkili (unordered pairs) ----------------------------------------------


def test_ikili_equals_sum_of_both_orders():
    """P({i,j} top 2) = P(i,j) + P(j,i)"""
    ordered = {c.horses: c.probability for c in sirali_ikili_probabilities(WIN_PROBS)}
    for c in ikili_probabilities(WIN_PROBS):
        i, j = c.horses
        expected = ordered.get((i, j), 0) + ordered.get((j, i), 0)
        assert _approx(c.probability, expected)


def test_ikili_sum_equals_one():
    """All unordered pairs form a partition of top-2 outcomes."""
    combos = ikili_probabilities(WIN_PROBS)
    assert _approx(sum(c.probability for c in combos), 1.0)


def test_ikili_pairs_are_unordered():
    combos = ikili_probabilities(WIN_PROBS)
    seen = set()
    for c in combos:
        # unordered: first < second when treated as sorted
        key = tuple(sorted(c.horses))
        assert key not in seen
        seen.add(key)


# --- Plase (top-k per horse) ----------------------------------------------


def test_plase_top2_sums_to_two():
    """Sum of individual top-2 probabilities == 2 (two horses in top 2)."""
    combos = plase_probabilities(WIN_PROBS, top_k=2)
    assert _approx(sum(c.probability for c in combos), 2.0)


def test_plase_top3_sums_to_three():
    combos = plase_probabilities(WIN_PROBS, top_k=3)
    assert _approx(sum(c.probability for c in combos), 3.0)


def test_plase_sorted_descending():
    combos = plase_probabilities(WIN_PROBS, top_k=2)
    probs = [c.probability for c in combos]
    assert probs == sorted(probs, reverse=True)


def test_plase_rejects_invalid_k():
    with pytest.raises(ValueError):
        plase_probabilities(WIN_PROBS, top_k=0)


# --- Üçlü (ordered triples) -----------------------------------------------


def test_uclu_sum_equals_one():
    combos = uclu_probabilities(WIN_PROBS)
    assert _approx(sum(c.probability for c in combos), 1.0)


def test_uclu_hand_check():
    """P(1=1, 2=2, 3=3) = 0.50 * (0.25/0.50) * (0.15/0.25) = 0.15"""
    combos = uclu_probabilities(WIN_PROBS)
    top = next(c for c in combos if c.horses == (1, 2, 3))
    assert _approx(top.probability, 0.15)


def test_uclu_needs_three_horses():
    """With only 2 horses there is no Üçlü space."""
    combos = uclu_probabilities({1: 0.7, 2: 0.3})
    assert combos == []


# --- Dörtlü ---------------------------------------------------------------


def test_dortlu_sum_equals_one():
    combos = dortlu_probabilities(WIN_PROBS)
    assert _approx(sum(c.probability for c in combos), 1.0)


def test_dortlu_needs_four_horses():
    combos = dortlu_probabilities({1: 0.5, 2: 0.3, 3: 0.2})
    assert combos == []


# --- Coverage helper ------------------------------------------------------


def test_cumulative_coverage_monotone():
    combos = sirali_ikili_probabilities(WIN_PROBS)
    cum = cumulative_coverage(combos)
    assert cum == sorted(cum)  # non-decreasing
    assert _approx(cum[-1], 1.0)  # completes the probability mass
