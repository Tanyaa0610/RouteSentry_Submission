"""Small, dependency-free text helpers: tokenization for TF-IDF and a
conservative "explicit date range" extractor for note applicability
windows. See config.DEFAULT_NOTE_WINDOW_DAYS for the fallback policy.
"""
from __future__ import annotations

import re
from datetime import date
from typing import Tuple

from routesentry.dates import monday_of, sunday_of

_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}

_RANGE_RE = re.compile(
    r"from\s+([A-Za-z]{3,9})\.?\s+(\d{1,2})\s+to\s+([A-Za-z]{3,9})\.?\s+(\d{1,2})",
    re.IGNORECASE,
)

_STOPWORDS = {
    "a", "an", "the", "on", "in", "to", "of", "for", "and", "or", "this",
    "was", "were", "is", "are", "by", "with", "at", "from", "about",
    "that", "but", "not", "no", "be", "as", "due", "it", "its",
}

_WORD_RE = re.compile(r"[a-z]+")


def parse_note_window(text: str, note_date: date) -> Tuple[date, date, str]:
    """Extract an explicit "from X to Y" date range from note text.

    Falls back to the note's own ISO week (config.DEFAULT_NOTE_WINDOW_DAYS)
    when no explicit range is present -- see config.py for why we refuse to
    extrapolate an undated duration.
    """
    match = _RANGE_RE.search(text)
    if match:
        m1, d1, m2, d2 = match.groups()
        month1 = _MONTHS.get(m1[:3].lower())
        month2 = _MONTHS.get(m2[:3].lower())
        if month1 and month2:
            year1 = note_date.year
            year2 = year1 if month2 >= month1 else year1 + 1
            try:
                start = date(year1, month1, int(d1))
                end = date(year2, month2, int(d2))
                if end >= start:
                    return start, end, "explicit"
            except ValueError:
                pass  # malformed date in text -- fall through to default window

    return monday_of(note_date), sunday_of(note_date), "default"


def tokenize(text: str) -> list[str]:
    words = _WORD_RE.findall(text.lower())
    return [w for w in words if w not in _STOPWORDS and len(w) > 1]
