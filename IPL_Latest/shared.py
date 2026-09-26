"""Bridge to the BCCI_Latest modules this pipeline reuses.

IPL matches live on the same stack as everything else BCCI publishes --
``stats.bcci.tv`` for the index, ``epr.ellipsedata.com`` for per-match detail --
so the HTTP clients, the endpoint joiner and the commentary parser are imported
from ``../BCCI_Latest`` rather than copied. A parser fix there lands here on the
next ``parse_balls.py`` run.

Only discovery differs: IPL competitions are not listed by the bcci.tv BFF, so
``ipl_api.py`` asks iplt20.com's own BFF instead.
"""

import os
import sys

# IPL modules are named fetch_ipl_* / parse_ipl_* precisely so they never shadow
# BCCI_Latest's fetch_matches / fetch_balls / parse_balls, which import each
# other by bare name.
BCCI_LATEST = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "BCCI_Latest")
if BCCI_LATEST not in sys.path:
    sys.path.append(BCCI_LATEST)

from bcci_api import fetch_matches, get_json  # noqa: E402
from bcci_api import make_session as make_stats_session  # noqa: E402
from commentary_parser import enrich  # noqa: E402
from ellipse_api import fetch_match  # noqa: E402
from ellipse_api import make_session as make_ellipse_session  # noqa: E402
from fetch_balls import build_match_table  # noqa: E402
from fetch_matches import innings_rows, merge_existing, normalise_match  # noqa: E402
from parse_balls import yield_report  # noqa: E402

__all__ = [
    "build_match_table", "enrich", "fetch_match", "fetch_matches", "get_json",
    "innings_rows", "make_ellipse_session", "make_stats_session",
    "merge_existing", "normalise_match", "yield_report",
]
