"""Thin HTTP layer for the rebuilt BCCI endpoints.

Two hosts are involved:

* ``www.bcci.tv/api/bff/cms`` -- the site's backend-for-frontend. It serves the
  *filter config* for the match listing page: which competitions exist for a
  given class/season, plus the team and format vocabularies. This is the
  discovery step; the competition ``externalGid`` values it returns are what the
  stats host expects as ``comp_gid``.
* ``stats.bcci.tv`` -- the actual match data. Filtered by ``comp_gid``; returns
  fixtures and results with identical schemas apart from a handful of
  result-only fields.
"""

import time

import requests

BFF_URL = "https://www.bcci.tv/api/bff/cms/matches"
STATS_URL = "https://stats.bcci.tv/match/{status}/"

# stats.bcci.tv honours arbitrarily large page sizes, so a season fits in one
# request. Kept below the observed 5000-row truncation point of the *unfiltered*
# endpoint, which we never rely on anyway.
PAGE_SIZE = 1000

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36"
)

HEADERS = {
    "accept": "application/json",
    "accept-language": "en-GB,en;q=0.9",
    "origin": "https://www.bcci.tv",
    "referer": "https://www.bcci.tv/",
    "user-agent": USER_AGENT,
}


def make_session():
    session = requests.Session()
    session.headers.update(HEADERS)
    return session


def get_json(session, url, params=None, timeout=60, retries=3):
    """GET with linear backoff. Returns parsed JSON, or None once retries run out."""
    for attempt in range(retries):
        try:
            response = session.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as exc:
            if attempt == retries - 1:
                print(f"  ! give up on {url}: {exc}")
                return None
            time.sleep(2 * (attempt + 1))
    return None


def fetch_filter_config(session, match_class="international", season="2026"):
    """Competition/team vocabulary for one class+season, straight from the BFF.

    ``status`` is deliberately not a parameter: the BFF returns the same
    competition list for fixtures and results, so asking twice is wasted work.
    """
    payload = get_json(
        session,
        BFF_URL,
        params={"class": match_class, "season": season, "status": "fixtures"},
    )
    if not payload or payload.get("status") != 200:
        return None
    return payload.get("data")


def competition_gids(config):
    return [c["externalGid"] for c in (config.get("competitions") or {}).get("data", [])]


def available_seasons(config):
    return (config.get("filters") or {}).get("seasonYears", [])


def fetch_matches(session, status, comp_gids):
    """All fixtures or results for the given competitions, following pagination.

    Only ``comp_gid`` actually constrains the query -- ``team_gid`` is accepted
    but ignored server-side, so it is not sent.
    """
    if not comp_gids:
        return []

    params = [("comp_gid", gid) for gid in comp_gids]
    matches, page = [], 1

    while True:
        payload = get_json(
            session,
            STATS_URL.format(status=status),
            params=params + [("page", page), ("size", PAGE_SIZE)],
        )
        if not payload:
            break

        batch = payload.get("match") or []
        matches.extend(batch)

        page_info = payload.get("page") or {}
        next_page = page_info.get("next_page")
        if not next_page or not batch:
            break
        page = next_page

    return matches
