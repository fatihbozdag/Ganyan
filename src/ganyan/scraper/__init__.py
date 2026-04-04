"""TJK scraper package — fetch and parse Turkish horse racing data."""

from ganyan.scraper.parser import (
    ParsedHorseEntry,
    ParsedRaceCard,
    RawHorseEntry,
    RawRaceCard,
    parse_race_card,
)
from ganyan.scraper.tjk_api import TJKClient

__all__ = [
    "ParsedHorseEntry",
    "ParsedRaceCard",
    "RawHorseEntry",
    "RawRaceCard",
    "TJKClient",
    "parse_race_card",
]
