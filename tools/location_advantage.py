from __future__ import annotations

try:
    from team_name_normalization import canonical_team_name
except ImportError:
    from tools.team_name_normalization import canonical_team_name

HOST_COUNTRIES = {"United States", "Mexico", "Canada"}
REGIONAL_CONFEDS = {
    "United States": "CONCACAF",
    "Mexico": "CONCACAF",
    "Canada": "CONCACAF",
}
TEAM_CONFED_HINTS = {
    "United States": "CONCACAF", "Mexico": "CONCACAF", "Canada": "CONCACAF", "Costa Rica": "CONCACAF", "Panama": "CONCACAF", "Jamaica": "CONCACAF",
    "Brazil": "CONMEBOL", "Argentina": "CONMEBOL", "Uruguay": "CONMEBOL", "Colombia": "CONMEBOL", "Ecuador": "CONMEBOL", "Paraguay": "CONMEBOL",
}


def location_advantage_points(team: str, host_country: str | None = None, team_confederation: str | None = None) -> float:
    team = canonical_team_name(team)
    host = canonical_team_name(host_country) if host_country else ""
    if host and team == host:
        return 100.0
    confed = team_confederation or TEAM_CONFED_HINTS.get(team)
    if host in HOST_COUNTRIES and confed == REGIONAL_CONFEDS.get(host):
        return 35.0
    return 0.0


def match_location_adjustment(home_team: str, away_team: str, host_country: str | None = None, home_confederation: str | None = None, away_confederation: str | None = None) -> tuple[float, float]:
    return (
        location_advantage_points(home_team, host_country, home_confederation),
        location_advantage_points(away_team, host_country, away_confederation),
    )
