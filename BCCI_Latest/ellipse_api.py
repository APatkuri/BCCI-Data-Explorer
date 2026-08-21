"""Client for epr.ellipsedata.com -- the per-match detail provider.

This is the replacement for the old polls.iplt20.com Hawkeye feed, and unlike
that one it is plain JSON: no base64, no XOR, no token dance. A single static
``edak`` header is all it wants.

Endpoints, all under ``/redirect/1/match/{gid}/``:

======================  ====================================================
endpoint                payload
======================  ====================================================
commentary?type=        ball-by-ball; deliveries nested in ``bbb[].balls[]``
  enhanced              grouped by over. The most complete ball list.
pitchmap                per-ball bounce position + length zone
beehive                 per-ball position at the stumps + grid cell
wagon                   ``spider_data`` (shot placement, scoring shots only)
                        and ``catch_map`` (dismissal placements)
manhattan               per-over runs/wickets, keyed by innings
scorecard               innings totals, batting/bowling cards, fall of wickets
summary                 match header, result, per-innings totals
======================  ====================================================

Ball identity is ``(innings_number, overs_unique)`` throughout, where
``overs_unique`` is ``"{over_index}.{ball:02d}"`` -- e.g. ``"77.04"`` is the 4th
delivery of the 78th over. ``wagon`` is the one exception: it carries
``over_number``/``ball_number`` instead, which :func:`spider_key` converts.
"""

import time

import requests

BASE_URL = "https://epr.ellipsedata.com/redirect/1/match/{gid}/{endpoint}"

# Static key lifted from the bcci.tv frontend. If responses start coming back
# 401/403, this is the first thing to re-check against the site.
EDAK = "dREj-f+6etraCroX6"

HEADERS = {
    "accept": "application/json",
    "edak": EDAK,
    "origin": "https://www.bcci.tv",
    "referer": "https://www.bcci.tv/",
    "user-agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36"
    ),
}

ENDPOINTS = (
    "commentary?type=enhanced",
    "pitchmap",
    "beehive",
    "wagon",
    "manhattan",
    "scorecard",
    "summary",
)


def make_session():
    session = requests.Session()
    session.headers.update(HEADERS)
    return session


def fetch_endpoint(session, gid, endpoint, timeout=90, retries=3):
    """One endpoint for one match. Returns parsed JSON or None."""
    url = BASE_URL.format(gid=gid, endpoint=endpoint)
    for attempt in range(retries):
        try:
            response = session.get(url, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as exc:
            if attempt == retries - 1:
                print(f"    ! {endpoint}: {exc}")
                return None
            time.sleep(2 * (attempt + 1))
    return None


def fetch_match(session, gid, endpoints=ENDPOINTS, pause=0.3):
    """All requested endpoints for one match, keyed by a filesystem-safe name."""
    out = {}
    for endpoint in endpoints:
        name = endpoint.split("?")[0]
        out[name] = fetch_endpoint(session, gid, endpoint)
        time.sleep(pause)
    return out


def spider_key(record):
    """wagon records use over/ball numbers; convert to an ``overs_unique`` string.

    ``over_number`` is 1-based while ``overs_unique`` counts completed overs, so
    the over index is one less. ``ball_number`` can exceed 6 when an over
    contains extras, which the 2-digit padding handles.
    """
    return f"{record['over_number'] - 1}.{record['ball_number']:02d}"
