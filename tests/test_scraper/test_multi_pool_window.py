"""Unit tests for multi-race pool race-window extraction.

Covers _parse_multi_race_pool_window and the new start_race_no /
end_race_no fields returned by _parse_multi_race_pools.
"""

from __future__ import annotations

import pytest

from ganyan.scraper.tjk_api import (
    _parse_multi_race_pool_window,
    _parse_multi_race_pools,
)


# ---------------------------------------------------------------------------
# _parse_multi_race_pool_window — unit tests
# ---------------------------------------------------------------------------


class TestParseMultiRacePoolWindow:
    """Direct tests for the window-extraction helper."""

    def test_standard_format(self):
        """'4. Koşu - 9. Koşu' returns (4, 9)."""
        block = "... 4. Koşu - 9. Koşu 6'LI GANYAN 3/1/4/2/5/7 45.230,75 ..."
        start, end = _parse_multi_race_pool_window(block, near_pos=30)
        assert start == 4
        assert end == 9

    def test_uppercase_kosu(self):
        """'4. KOŞU - 9. KOŞU' (uppercase) returns (4, 9)."""
        block = "4. KOŞU - 9. KOŞU 6'LI GANYAN 3/1/4/2/5/7 12.500,00"
        start, end = _parse_multi_race_pool_window(block, near_pos=20)
        assert start == 4
        assert end == 9

    def test_ascii_kosu_fallback(self):
        """'4. Kosu - 9. Kosu' (ASCII s) returns (4, 9)."""
        block = "4. Kosu - 9. Kosu 6'LI GANYAN 1/2/3/4/5/6 8.000,00"
        start, end = _parse_multi_race_pool_window(block, near_pos=20)
        assert start == 4
        assert end == 9

    def test_single_digit_races(self):
        """'1. Koşu - 6. Koşu' returns (1, 6)."""
        block = "1. Koşu - 6. Koşu 6'LI GANYAN 1/2/3/4/5/6 1.234,50"
        start, end = _parse_multi_race_pool_window(block, near_pos=20)
        assert start == 1
        assert end == 6

    def test_leading_whitespace_variants(self):
        """Extra spaces inside the pattern are tolerated."""
        block = "3.  Koşu  -  8.  Koşu 5'LI GANYAN 1/2/3/4/5 900,00"
        start, end = _parse_multi_race_pool_window(block, near_pos=25)
        assert start == 3
        assert end == 8

    def test_seven_leg_pool(self):
        """'2. Koşu - 8. Koşu' maps to (2, 8) for a 7'lı pool."""
        block = "2. Koşu - 8. Koşu 7'LI GANYAN 1/2/3/4/5/6/7 250.000,00"
        start, end = _parse_multi_race_pool_window(block, near_pos=20)
        assert start == 2
        assert end == 8

    def test_window_before_match_pos(self):
        """Window text appearing 150 chars before the combo is still found."""
        prefix = "4. Koşu - 9. Koşu" + " " * 130
        combo = "6'LI GANYAN 1/2/3/4/5/6 50.000,00"
        block = prefix + combo
        match_pos = len(prefix)
        start, end = _parse_multi_race_pool_window(block, near_pos=match_pos)
        assert start == 4
        assert end == 9

    def test_no_window_text_returns_none(self):
        """Block without a race-range marker returns (None, None)."""
        block = "GANYAN 3 2,40 İKİLİ 3/1 4,10 ÜÇLÜ BAHİS 3/1/5 18,30"
        start, end = _parse_multi_race_pool_window(block, near_pos=0)
        assert start is None
        assert end is None

    def test_empty_block_returns_none(self):
        """Empty string returns (None, None) without raising."""
        start, end = _parse_multi_race_pool_window("", near_pos=0)
        assert start is None
        assert end is None

    def test_inverted_range_rejected(self):
        """start >= end fails the sanity check; returns (None, None)."""
        block = "9. Koşu - 4. Koşu 6'LI GANYAN 1/2/3/4/5/6 0,00"
        start, end = _parse_multi_race_pool_window(block, near_pos=20)
        assert start is None
        assert end is None

    def test_race_zero_rejected(self):
        """Race number 0 fails the sanity check."""
        block = "0. Koşu - 5. Koşu 5'LI GANYAN 1/2/3/4/5 0,00"
        start, end = _parse_multi_race_pool_window(block, near_pos=20)
        assert start is None
        assert end is None


# ---------------------------------------------------------------------------
# _parse_multi_race_pools — integration tests for start/end fields
# ---------------------------------------------------------------------------


class TestParseMultiRacePoolsWindowFields:
    """Verify that _parse_multi_race_pools propagates window fields."""

    def test_6li_with_window(self):
        """6'lı with '4. Koşu - 9. Koşu' captures window on the pool entry."""
        block = (
            "GANYAN 5 3,20 4. Koşu - 9. Koşu "
            "6'LI GANYAN 3/1/4/2/5,7/6 45.230,75"
        )
        pools = _parse_multi_race_pools(block)
        assert len(pools) == 1
        p = pools[0]
        assert p["pool_type"] == "6li"
        assert p["start_race_no"] == 4
        assert p["end_race_no"] == 9

    def test_7li_with_window(self):
        """7'lı with '1. Koşu - 7. Koşu' captures window correctly."""
        block = "1. Koşu - 7. Koşu 7'LI GANYAN 2/1/3/4/5/6/7 250.000,00"
        pools = _parse_multi_race_pools(block)
        assert len(pools) == 1
        p = pools[0]
        assert p["pool_type"] == "7li"
        assert p["start_race_no"] == 1
        assert p["end_race_no"] == 7

    def test_5li_with_window(self):
        """5'lı with '3. Koşu - 7. Koşu' window is captured."""
        block = "3. Koşu - 7. Koşu 5'LI GANYAN 1/2/3/4/5 3.000,00"
        pools = _parse_multi_race_pools(block)
        assert len(pools) == 1
        p = pools[0]
        assert p["pool_type"] == "5li"
        assert p["start_race_no"] == 3
        assert p["end_race_no"] == 7

    def test_no_window_yields_none(self):
        """Pool without a window in the block yields (None, None)."""
        block = "6'LI GANYAN 1/2/3/4/5/6 12.000,00"
        pools = _parse_multi_race_pools(block)
        assert len(pools) == 1
        assert pools[0]["start_race_no"] is None
        assert pools[0]["end_race_no"] is None

    def test_empty_block_yields_empty_list(self):
        """Empty block returns [] without raising."""
        assert _parse_multi_race_pools("") == []

    def test_two_6li_distinct_windows(self):
        """Two 6'lı pools on same program capture separate windows."""
        block = (
            "1. Koşu - 6. Koşu 6'LI GANYAN 1/2/3/4/5/6 8.000,00 "
            "4. Koşu - 9. Koşu 6'LI GANYAN 3/1/4/2/5/7 45.230,75"
        )
        pools = _parse_multi_race_pools(block)
        assert len(pools) == 2

        first = pools[0]
        assert first["pool_index"] == 1
        assert first["start_race_no"] == 1
        assert first["end_race_no"] == 6

        second = pools[1]
        assert second["pool_index"] == 2
        assert second["start_race_no"] == 4
        assert second["end_race_no"] == 9

    def test_mixed_5li_6li_7li(self):
        """Block with all three pool types each get their own window."""
        block = (
            "1. Koşu - 5. Koşu 5'LI GANYAN 1/2/3/4/5 2.000,00 "
            "2. Koşu - 7. Koşu 6'LI GANYAN 1/2/3/4/5/6 5.000,00 "
            "1. Koşu - 7. Koşu 7'LI GANYAN 1/2/3/4/5/6/7 100.000,00"
        )
        pools = _parse_multi_race_pools(block)
        assert len(pools) == 3
        types = {p["pool_type"]: p for p in pools}
        assert types["5li"]["start_race_no"] == 1
        assert types["5li"]["end_race_no"] == 5
        assert types["6li"]["start_race_no"] == 2
        assert types["6li"]["end_race_no"] == 7
        assert types["7li"]["start_race_no"] == 1
        assert types["7li"]["end_race_no"] == 7

    def test_winning_combo_still_parsed(self):
        """Adding window extraction does not break combo / payout parsing."""
        block = "4. Koşu - 9. Koşu 6'LI GANYAN 3/1,12/4/2/5/7 45.230,75"
        pools = _parse_multi_race_pools(block)
        assert len(pools) == 1
        p = pools[0]
        assert p["winning_combo"] == "3/1,12/4/2/5/7"
        assert p["payout_tl"] == pytest.approx(45230.75)
