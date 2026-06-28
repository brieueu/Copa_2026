from __future__ import annotations

import re
import unicodedata
from typing import Iterable

CANONICAL_ALIASES = {
    "usa": "United States",
    "u s a": "United States",
    "united states of america": "United States",
    "united states": "United States",
    "korea republic": "South Korea",
    "republic of korea": "South Korea",
    "south korea": "South Korea",
    "cape verde": "Cabo Verde",
    "cabo verde": "Cabo Verde",
    "ivory coast": "Côte d'Ivoire",
    "cote d ivoire": "Côte d'Ivoire",
    "côte d ivoire": "Côte d'Ivoire",
    "côte d'ivoire": "Côte d'Ivoire",
    "bosnia herzegovina": "Bosnia and Herzegovina",
    "bosnia and herzegovina": "Bosnia and Herzegovina",
    "czechia": "Czech Republic",
    "czech republic": "Czech Republic",
    "turkiye": "Turkey",
    "türkiye": "Turkey",
    "turkey": "Turkey",
    "dr congo": "Congo DR",
    "congo dr": "Congo DR",
    "democratic republic of congo": "Congo DR",
}


def _key(value: object) -> str:
    if value is None:
        return ""
    s = str(value).strip().lower().replace("&", " and ")
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def canonical_team_name(value: object) -> str:
    key = _key(value)
    if not key:
        return ""
    if key in CANONICAL_ALIASES:
        return CANONICAL_ALIASES[key]
    return " ".join(part.capitalize() for part in key.split())


def normalized_team_key(value: object) -> str:
    return _key(canonical_team_name(value))


def normalize_team_series(series):
    return series.map(canonical_team_name)


def missing_from_reference(values: Iterable[object], reference: Iterable[object]) -> set[str]:
    ref = {normalized_team_key(v) for v in reference}
    return {canonical_team_name(v) for v in values if normalized_team_key(v) not in ref}
