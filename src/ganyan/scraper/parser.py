"""TJK race card parser — pure data transformation, no I/O."""

from dataclasses import dataclass, field
from datetime import date


# --- Raw data structures (from scraper) ---


@dataclass
class RawHorseEntry:
    name: str
    age: int | None = None
    origin: str | None = None
    owner: str | None = None
    trainer: str | None = None
    gate_number: int | None = None  # betting program number
    start_gate: int | None = None  # physical draw
    jockey: str | None = None
    weight_kg: float | None = None
    hp: float | None = None
    kgs: int | None = None
    s20: float | None = None
    eid: str | None = None
    gny: float | None = None
    agf: float | None = None
    last_six: str | None = None
    finish_position: int | None = None
    finish_time: str | None = None
    # Plase pool payout (TL per 1 TL bilet) for this horse, when it
    # placed.  TJK publishes one row per placed horse:
    # ``PLASE <program_no> <amount>``.  None for non-placed horses or
    # races where the plase pool didn't form (rare on small fields).
    plase_payout_tl: float | None = None
    # TJK's stable internal id for this horse.  Extracted from the
    # horse-name <a href> on the results/program pages so we can hit
    # /Query/ConnectedPage/AtKosuBilgileri?QueryParameter_AtId=X for
    # pedigree details.
    tjk_at_id: int | None = None
    # Equipment (takı) codes attached to the horse for this race, e.g.
    # "KG DB SK".  Space-separated string of 1-3-letter codes lifted
    # from the <sup> tags on the horse name cell.  Domain handicappers
    # consider first-time equipment a major form-change signal.
    equipment: str | None = None
    # True when TJK marks the horse as "(Koşmaz)" pre-race — i.e.
    # withdrawn from the field. Predictor must exclude scratched
    # horses and renormalise win probabilities across the runners.
    scratched: bool = False


@dataclass
class RawRaceCard:
    track_name: str
    date: date
    race_number: int
    post_time: str | None = None  # HH:MM; parsed from race-no header
    distance_meters: int | None = None
    surface: str | None = None
    race_type: str | None = None
    horse_type: str | None = None
    weight_rule: str | None = None
    # Raw "Son 800" strings from the results page:
    # first horse's last-800m time, second horse's last-800m time.
    # Format "M.SS.HH" (e.g. "0.58.40" = 58.40s).  Only present on
    # the results endpoint, never on the pre-race program.
    pace_l800_leader: str | None = None
    pace_l800_runner_up: str | None = None
    # Exotic-pool payouts in TL per 1 TL bet.  Only present on the
    # results page, and only for combinations that actually had winning
    # tickets (some races only offer a subset of exotic pools).
    ganyan_payout_tl: float | None = None
    ikili_payout_tl: float | None = None
    sirali_ikili_payout_tl: float | None = None
    uclu_payout_tl: float | None = None
    dortlu_payout_tl: float | None = None
    # Multi-race pool payouts (5'lı / 6'lı / 7'lı) appearing on this
    # race's bahisSonucCard. Each entry is
    # ``{pool_type, pool_index, winning_combo, payout_tl}``. TJK shows
    # the multi-race pool payout on the LAST leg's card; earlier legs
    # may carry the same data redundantly. Storage layer dedupes by
    # (date, track, pool_type, pool_index).
    multi_race_pools: list[dict] = field(default_factory=list)
    horses: list[RawHorseEntry] = field(default_factory=list)


# --- Parsed data structures (validated, enriched) ---


@dataclass
class ParsedHorseEntry:
    name: str
    age: int | None = None
    origin: str | None = None
    owner: str | None = None
    trainer: str | None = None
    gate_number: int | None = None  # betting program number
    start_gate: int | None = None  # physical draw
    jockey: str | None = None
    weight_kg: float | None = None
    hp: float | None = None
    kgs: int | None = None
    s20: float | None = None
    eid: str | None = None
    eid_seconds: float | None = None
    gny: float | None = None
    agf: float | None = None
    last_six: str | None = None
    last_six_parsed: list[int | None] = field(default_factory=list)
    finish_position: int | None = None
    finish_time: str | None = None
    plase_payout_tl: float | None = None
    tjk_at_id: int | None = None
    equipment: str | None = None
    scratched: bool = False


@dataclass
class ParsedRaceCard:
    track_name: str
    date: date
    race_number: int
    post_time: str | None = None
    distance_meters: int | None = None
    surface: str | None = None
    race_type: str | None = None
    horse_type: str | None = None
    weight_rule: str | None = None
    # Last-800m times in seconds (parsed to float).  Either may be
    # None when TJK only publishes one value (e.g. wire-to-wire wins).
    pace_l800_leader_s: float | None = None
    pace_l800_runner_up_s: float | None = None
    # Exotic-pool payouts (TL per 1 TL bet).
    ganyan_payout_tl: float | None = None
    ikili_payout_tl: float | None = None
    sirali_ikili_payout_tl: float | None = None
    uclu_payout_tl: float | None = None
    dortlu_payout_tl: float | None = None
    multi_race_pools: list[dict] = field(default_factory=list)
    horses: list[ParsedHorseEntry] = field(default_factory=list)


# --- Parsing functions ---

TRACK_NAMES = {
    "istanbul": "İstanbul",
    "İstanbul": "İstanbul",
    "ankara": "Ankara",
    "izmir": "İzmir",
    "İzmir": "İzmir",
    "bursa": "Bursa",
    "adana": "Adana",
    "antalya": "Antalya",
    "elazığ": "Elazığ",
    "Elazığ": "Elazığ",
    "diyarbakır": "Diyarbakır",
    "Diyarbakır": "Diyarbakır",
    "kocaeli": "Kocaeli",
    "şanlıurfa": "Şanlıurfa",
    "Şanlıurfa": "Şanlıurfa",
}


def parse_eid_to_seconds(eid: str | None) -> float | None:
    """Convert TJK EID / finish-time string to seconds.

    Formats:
        "1.30.45" -> 90.45  (minutes.seconds.hundredths)
        "58.20"   -> 58.20  (seconds.hundredths)

    Production data contains a non-trivial fraction of malformed strings
    — empty components (``"1..45"``), stray whitespace, non-numeric
    tokens where a horse DNF'd.  Return ``None`` on any parse error
    instead of raising; callers treat missing finish time as
    "can't use this row".
    """
    if not eid or not eid.strip():
        return None
    parts = eid.strip().split(".")
    try:
        if len(parts) == 3:
            minutes = int(parts[0]) if parts[0] else 0
            seconds = int(parts[1]) if parts[1] else 0
            hundredths = int(parts[2]) if parts[2] else 0
            return minutes * 60 + seconds + hundredths / 100
        if len(parts) == 2:
            seconds = int(parts[0]) if parts[0] else 0
            hundredths = int(parts[1]) if parts[1] else 0
            return seconds + hundredths / 100
    except ValueError:
        return None
    return None


def parse_last_six(last_six: str | None) -> list[int | None]:
    """Parse last-six-races finish positions.

    Left-to-right is OLDEST → NEWEST (verified against the production DB:
    the rightmost token always matches the horse's previous ``finish_position``).
    So the returned list has ``list[-1]`` = most recent race.

    TJK publishes the field as a compact single-char-per-finish string,
    with ``0`` meaning "position 10 or worse" and ``-`` as a season/gap
    separator we preserve as ``None``:

        "212464" -> [2, 1, 2, 4, 6, 4]
        "043335" -> [10, 4, 3, 3, 3, 5]     # leading 0 = 10+
        "13-2315" -> [1, 3, None, 2, 3, 1, 5]
        "8225" -> [8, 2, 2, 5]                # horse has <6 recorded races

    Legacy space-separated format (still produced by some test fixtures
    and older scraper outputs) is tolerated for back-compat:

        "2 4 4 5 2 7" -> [2, 4, 4, 5, 2, 7]
        "1 3 - 2 - 4" -> [1, 3, None, 2, None, 4]
    """
    if not last_six:
        return []
    s = last_six.strip()
    if not s:
        return []

    # Legacy whitespace-tokenised format used by fixtures + old scrapes.
    if " " in s:
        result: list[int | None] = []
        for part in s.split():
            if part == "-":
                result.append(None)
            else:
                try:
                    result.append(int(part))
                except ValueError:
                    result.append(None)
        return result

    # Production format: one character per finish position.  Digits
    # 1-9 map to their face-value finish; ``0`` means "10 or worse"
    # (TJK has no two-digit column in this field); ``-`` preserves a
    # gap so positional alignment with "most recent = last" is
    # retained.
    result = []
    for ch in s:
        if ch == "-":
            result.append(None)
        elif ch.isdigit():
            n = int(ch)
            result.append(10 if n == 0 else n)
        else:
            result.append(None)
    return result


def normalize_track_name(name: str) -> str:
    """Normalize Turkish track names with correct casing and İ/ı handling."""
    stripped = name.strip()
    if stripped in TRACK_NAMES:
        return TRACK_NAMES[stripped]
    lower = stripped.lower()
    if lower in TRACK_NAMES:
        return TRACK_NAMES[lower]
    return stripped.title()


def parse_race_card(raw: RawRaceCard) -> ParsedRaceCard:
    """Transform a RawRaceCard into a ParsedRaceCard with enriched fields."""
    horses = []
    for h in raw.horses:
        horses.append(ParsedHorseEntry(
            name=h.name.strip(),
            age=h.age,
            origin=h.origin,
            owner=h.owner,
            trainer=h.trainer,
            gate_number=h.gate_number,
            start_gate=h.start_gate,
            jockey=h.jockey,
            weight_kg=h.weight_kg,
            hp=h.hp,
            kgs=h.kgs,
            s20=h.s20,
            eid=h.eid,
            eid_seconds=parse_eid_to_seconds(h.eid),
            gny=h.gny,
            agf=h.agf,
            last_six=h.last_six,
            last_six_parsed=parse_last_six(h.last_six),
            finish_position=h.finish_position,
            finish_time=h.finish_time,
            plase_payout_tl=h.plase_payout_tl,
            tjk_at_id=h.tjk_at_id,
            equipment=h.equipment,
            scratched=getattr(h, "scratched", False),
        ))

    return ParsedRaceCard(
        track_name=normalize_track_name(raw.track_name),
        date=raw.date,
        race_number=raw.race_number,
        post_time=raw.post_time,
        distance_meters=raw.distance_meters,
        surface=raw.surface.lower() if raw.surface else None,
        race_type=raw.race_type,
        horse_type=raw.horse_type,
        weight_rule=raw.weight_rule,
        pace_l800_leader_s=parse_eid_to_seconds(raw.pace_l800_leader),
        pace_l800_runner_up_s=parse_eid_to_seconds(raw.pace_l800_runner_up),
        ganyan_payout_tl=raw.ganyan_payout_tl,
        ikili_payout_tl=raw.ikili_payout_tl,
        sirali_ikili_payout_tl=raw.sirali_ikili_payout_tl,
        uclu_payout_tl=raw.uclu_payout_tl,
        dortlu_payout_tl=raw.dortlu_payout_tl,
        multi_race_pools=list(raw.multi_race_pools or []),
        horses=horses,
    )
