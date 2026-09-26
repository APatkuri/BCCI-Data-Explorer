"""Client for iplt20.com's backend-for-frontend -- the IPL discovery step.

The bcci.tv BFF lists international and domestic competitions but never the
IPL, so the season -> competition mapping has to come from iplt20.com itself.
Same response shape as the bcci.tv one: ``data.competitions.data[].externalGid``
is what ``stats.bcci.tv`` expects as ``comp_gid``.

Each IPL season is its own competition with its own gid -- the 2009 edition is
``291e6199-...``, 2025 is ``f2d3061a-...`` -- so every season needs one lookup.
The HTML match page is no help here: it always server-renders the default
season's gid whatever ``?season=`` says, and switches client-side.
"""

import requests

from shared import get_json

BFF_URL = "https://www.iplt20.com/api/bff/cms/matches"

HEADERS = {
    "accept": "application/json",
    "accept-language": "en-GB,en;q=0.9",
    "origin": "https://www.iplt20.com",
    "referer": "https://www.iplt20.com/",
    "user-agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36"
    ),
}


def make_session():
    session = requests.Session()
    session.headers.update(HEADERS)
    return session


def fetch_filter_config(session, season=None):
    """Competitions + filter vocabulary for one season (default: the site's current)."""
    params = {"status": "results"}
    if season:
        params["season"] = season
    payload = get_json(session, BFF_URL, params=params)
    if not payload or payload.get("status") != 200:
        return None
    return payload.get("data")


def competition_gids(config):
    return [c["externalGid"] for c in (config.get("competitions") or {}).get("data", [])]


def available_seasons(config):
    return (config.get("filters") or {}).get("seasonYears", [])


def default_season(config):
    return (config.get("filters") or {}).get("defaultSeasonYear")
