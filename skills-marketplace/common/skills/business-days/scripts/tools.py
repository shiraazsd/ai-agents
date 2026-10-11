"""Business-day arithmetic."""

from datetime import date, timedelta

from skillkit import run


def business_days(start: str, end: str) -> dict:
    """Count Mon-Fri days from start (inclusive) to end (exclusive). Dates are YYYY-MM-DD."""
    d, stop, n = date.fromisoformat(start), date.fromisoformat(end), 0
    while d < stop:
        n += d.weekday() < 5
        d += timedelta(days=1)
    return {"start": start, "end": end, "business_days": n}


def add_business_days(start: str, days: int) -> dict:
    """Return the date that is `days` business days after start (YYYY-MM-DD)."""
    d, left = date.fromisoformat(start), days
    while left:
        d += timedelta(days=1)
        left -= d.weekday() < 5
    return {"start": start, "days": days, "date": d.isoformat()}


if __name__ == "__main__":
    run()
