from datetime import date

from ganyan.scraper.parser import (
    parse_eid_to_seconds,
    parse_last_six,
    normalize_track_name,
    RawRaceCard,
    RawHorseEntry,
    parse_race_card,
)


def test_parse_eid_to_seconds_standard():
    assert parse_eid_to_seconds("1.30.45") == 90.45


def test_parse_eid_to_seconds_short():
    assert parse_eid_to_seconds("58.20") == 58.20


def test_parse_eid_to_seconds_empty():
    assert parse_eid_to_seconds("") is None
    assert parse_eid_to_seconds(None) is None


def test_parse_last_six():
    assert parse_last_six("2 4 4 5 2 7") == [2, 4, 4, 5, 2, 7]


def test_parse_last_six_with_missing():
    assert parse_last_six("1 3 - 2 - 4") == [1, 3, None, 2, None, 4]


def test_parse_last_six_empty():
    assert parse_last_six("") == []
    assert parse_last_six(None) == []


def test_parse_last_six_production_format_concatenated():
    """TJK production format: one char per finish, no spaces."""
    assert parse_last_six("212464") == [2, 1, 2, 4, 6, 4]
    assert parse_last_six("043335") == [10, 4, 3, 3, 3, 5]
    assert parse_last_six("8225") == [8, 2, 2, 5]


def test_parse_last_six_production_format_with_separator():
    """Dash in the concatenated string is a gap/season separator."""
    assert parse_last_six("13-2315") == [1, 3, None, 2, 3, 1, 5]
    assert parse_last_six("8-34124") == [8, None, 3, 4, 1, 2, 4]
    assert parse_last_six("7-0") == [7, None, 10]


def test_parse_last_six_direction_is_oldest_to_newest():
    """Pin the direction convention — last element is MOST RECENT.

    Verified against production DB: a horse whose previous race
    finish_position is ``N`` always has ``N`` as the last character of
    its next race's ``last_six`` string.  Flipping this interpretation
    would silently invert compute_form_cycle.
    """
    # Horse finished 5th in their previous race → last slot is 5.
    assert parse_last_six("212465")[-1] == 5
    # Horse finished 10+ in their previous race → last slot is 10.
    assert parse_last_six("402770")[-1] == 10


def test_normalize_track_name():
    assert normalize_track_name("İstanbul") == "İstanbul"
    assert normalize_track_name("istanbul") == "İstanbul"
    assert normalize_track_name("ISTANBUL") == "İstanbul"
    assert normalize_track_name(" İstanbul ") == "İstanbul"


def test_parse_race_card():
    raw = RawRaceCard(
        track_name="İstanbul",
        date=date(2026, 4, 5),
        race_number=3,
        distance_meters=1400,
        surface="Çim",
        race_type="Handikap",
        horse_type="İngiliz",
        weight_rule="Handikap",
        horses=[
            RawHorseEntry(
                name="Karayel",
                age=4,
                origin="TR",
                owner="Ali Kaya",
                trainer="Mehmet Demir",
                gate_number=3,
                jockey="Ahmet Çelik",
                weight_kg=57.0,
                hp=85.5,
                kgs=21,
                s20=12.50,
                eid="1.30.45",
                gny=8.30,
                agf=5.20,
                last_six="1 3 2 4 1 2",
            ),
        ],
    )
    parsed = parse_race_card(raw)
    assert parsed.track_name == "İstanbul"
    assert parsed.race_number == 3
    assert len(parsed.horses) == 1
    assert parsed.horses[0].name == "Karayel"
    assert parsed.horses[0].eid_seconds == 90.45
    assert parsed.horses[0].last_six_parsed == [1, 3, 2, 4, 1, 2]
