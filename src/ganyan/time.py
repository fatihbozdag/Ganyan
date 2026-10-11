"""Race clocks: timezone-aware Istanbul operations, naive UTC database times."""
from datetime import datetime, time, timezone
from zoneinfo import ZoneInfo

ISTANBUL = ZoneInfo("Europe/Istanbul")


def local_now():
    return datetime.now(ISTANBUL)


def today():
    return local_now().date()


def utcnow():
    return datetime.now(timezone.utc).replace(tzinfo=None)


def race_cutoff(race):
    """Unknown post time uses start of race day, never an invented late cutoff."""
    try:
        post = time.fromisoformat(race.post_time) if race.post_time else time.min
    except ValueError:
        post = time.min
    return datetime.combine(race.date, post, ISTANBUL).astimezone(timezone.utc).replace(tzinfo=None)


def prediction_cutoff(race, as_of=None):
    now = as_of or utcnow()
    if now.tzinfo is not None:
        now = now.astimezone(timezone.utc).replace(tzinfo=None)
    return min(now, race_cutoff(race))


def is_upcoming(race, now=None, margin_minutes=0):
    from datetime import timedelta
    from ganyan.db.models import RaceStatus
    return (race.status == RaceStatus.scheduled and bool(race.post_time)
            and race_cutoff(race) > (now or utcnow()) + timedelta(minutes=margin_minutes))
