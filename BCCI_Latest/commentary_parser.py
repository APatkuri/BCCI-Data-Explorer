"""Extract structured ball attributes from ellipsedata commentary text.

The commentary is not one format but four, and they differ enormously in how
much they give up. Vocabularies below were mined from the corpus rather than
invented, so they are closed sets observed in real data:

``over/round-the-wicket``
    ``Over the wicket. Length ball, outside off, front foot, forward defensive
    along the ground to Rishabh Pant at short leg. No run.``
    Comma-delimited slots -- length, line, footwork, then shot/trajectory/fielder.

``structured``
    ``Manav Suthar to Pasindu Sooriyabandara. Left-arm orthodox stock length
    ball, off stump on the front foot driving to extra cover for no run,
    fielded by Prasidh Krishna``
    Bowler/batter header, then a bowl descriptor carrying type + variation +
    length, then line, footwork, shot, fielder.

``minimal``
    ``Sethmika Seneviratne to Lakshya Raichandani. No run``
    Nothing beyond what the tabular columns already hold.

``prose``
    LLM-written description. No fixed grammar, but it draws on the same
    vocabulary, so a keyword scan recovers most attributes at lower confidence.

Every parse records ``parse_method`` so downstream work can trust template rows
over keyword rows, or drop the latter entirely.
"""

import re

# --- vocabularies (longest-first; order matters for prefix/suffix matching) ---

LENGTHS = (
    "back of a length", "half tracker", "full toss", "length ball",
    "yorker", "short", "full", "long hop", "beamer",
)

LINES = (
    "wide outside off", "wide outside leg", "outside off", "outside leg",
    "middle and leg", "middle and off", "middle stump", "off stump",
    "leg stump", "down the leg side", "down leg",
)

FOOTWORK = (
    "front foot", "back foot", "down the track", "no movement",
    "moves infront", "swayed away", "backs away", "ducked",
)

# Surface form -> normalised shot name.
SHOTS = {
    "forward defensive": "forward defensive",
    "backward defensive": "backward defensive",
    "shoulders arms": "shoulders arms",
    "reverse sweep": "reverse sweep",
    "slog-sweep": "slog sweep",
    "slog sweep": "slog sweep",
    "late cutted": "late cut",
    "late cut": "late cut",
    "driving": "drive", "driven": "drive", "drive": "drive",
    "pulled": "pull", "pull": "pull",
    "slogged": "slog", "slog": "slog",
    "swept": "sweep", "sweep": "sweep",
    "flicked": "flick", "flick": "flick",
    "glanced": "glance", "glance": "glance",
    "steered": "steer", "steer": "steer",
    "cutted": "cut", "cut": "cut",
    "worked": "work", "work": "work",
    "pushed": "push", "push": "push",
    "padded": "pad", "pad": "pad",
    "dropped": "drop",
    "leaving left alone": "leave", "leave": "leave",
    "blocked": "block", "block": "block",
}

TRAJECTORIES = ("uncontrolled in air", "along the ground", "in air")
CONTROLS = ("uncontrolled", "controlled", "well timed", "mistimed")

DISMISSALS = (
    "caught and bowled", "caught & bowled", "run out", "hit wicket",
    "stumped", "caught", "bowled", "lbw", "retired hurt", "retired",
    "obstructing the field", "handled the ball", "timed out",
)

# Observed in the templates, plus the deep/long variants prose tends to use.
# Longest-first so 'deep backward square leg' wins over 'square leg'.
FIELD_POSITIONS = tuple(sorted((
    "deep backward square leg", "deep backward point", "deep extra cover",
    "short extra cover", "backward square leg", "short third man",
    "short mid wicket", "deep mid wicket", "deep square leg", "silly mid off",
    "silly mid on", "backward point", "short fine leg", "deep fine leg",
    "deep third man", "deep cover", "deep point", "extra cover", "silly point",
    "first slip", "second slip", "third slip", "fourth slip", "leg slip",
    "cow corner", "long leg", "square leg", "short leg", "fine leg",
    "third man", "mid wicket", "midwicket", "long off", "long on", "mid off",
    "mid on", "wicketkeeper", "keeper", "bowler", "gully", "cover", "point",
    "slip",
), key=len, reverse=True))

# Prose says 'caught by the fielder at deep square leg', 'given out leg before
# wicket', 'hits the stumps clean' -- none of which the template regexes see.
PROSE_DISMISSALS = (
    (r"\bcaught\s+(?:and|&)\s+bowled\b", "caught and bowled"),
    (r"\brun\s+out\b", "run out"),
    (r"\bhit\s+wicket\b", "hit wicket"),
    (r"\bstump(?:ed|s\s+him)\b", "stumped"),
    (r"\bcaught\b|\bcatch\s+(?:is\s+)?(?:taken|held)\b|\btakes?\s+the\s+catch\b", "caught"),
    (r"\bleg\s+before\s+wicket\b|\blbw\b|\btrap(?:s|ped)\s+.{0,20}\blbw\b", "lbw"),
    (r"\bbowled\b|\bhits?\s+the\s+stumps\b|\bcastled\b|\btimber\b|\bknocks?\s+"
     r"(?:back|over)\s+the\s+(?:stumps|bails)\b", "bowled"),
)

_PROSE_POSITION = re.compile(
    r"\b(?:to|at|through|towards|past|over|into|behind|down\s+to)\s+"
    r"(?:the\s+)?(" + "|".join(re.escape(p) for p in FIELD_POSITIONS) + r")\b",
    re.I,
)

# Scoring codes carry extras as unicode superscripts. Longest first: the bye
# marker is a substring of the leg-bye and no-ball markers.
EXTRAS_MARKERS = (
    ("ʷᵈ", "wide"),      # wd
    ("ⁿᵇ", "no ball"),   # nb
    ("ˡᵇ", "leg bye"),   # lb
    ("ᵇ", "bye"),             # b
    ("ᵖ", "penalty"),         # p
)

BOWL_VARIATIONS = ("stock", "slower", "quicker")

_ARM = re.compile(r"^(Over|Round) the wicket\.\s*(.*)$", re.I)
_HEADER = re.compile(
    r"^(?:(?P<flag>OUT|FOUR|SIX|No ball|Wide)[!.]\s*)?"
    r"(?P<bowler>[\w'\-. ]+?) to (?P<batter>[\w'\-. ]+?)\.\s+(?P<body>.+)$"
)
# 'on the' is followed by the footwork phrase and then the shot segment. The
# boundary between them is not punctuated, so the footwork is matched against
# the known vocabulary rather than guessed at by a non-greedy group -- which
# would stop at the first word and yield 'front' instead of 'front foot'.
_STRUCT_BODY = re.compile(
    r"^(?P<desc>.+?),\s*(?P<line>[\w ]+?) on the (?P<tail>.+)$"
)
_FOOT_PREFIX = re.compile(
    r"^(" + "|".join(sorted(FOOTWORK, key=len, reverse=True)) + r")\b\s*(.*)$", re.I
)
_FIELDER_AT = re.compile(r"\bto ([\w'\-. ]+?) at ([\w' ]+?)(?=[.,]|$)")
_CAUGHT_BY = re.compile(r"\b(?:caught|fielded) by ([\w'\-. ]+?)(?=[.,]|$)", re.I)
_DISMISS_TAIL = re.compile(
    r"\b([\w'\-. ]+?)\s+(" + "|".join(DISMISSALS) + r")\s+for\s+(\d+)", re.I
)
_DISMISS_DASH = re.compile(
    r"\b([\w'\-. ]+?)\s+(" + "|".join(DISMISSALS) + r")\s*-\s*(\d+)\((\d+)\)", re.I
)

EMPTY = {
    "bowler_arm": None, "delivery_type": None, "delivery_variation": None,
    "length": None, "line": None, "footwork": None, "shot_type": None,
    "shot_trajectory": None, "shot_control": None, "field_position": None,
    "fielder_name": None, "dismissal_type": None, "dismissed_batter": None,
    "dismissal_score": None, "parse_method": "none",
}


def _find_first(text, options):
    """Longest-first scan; returns the matched option or None."""
    low = text.lower()
    for option in options:
        if option in low:
            return option
    return None


def _split_bowl_descriptor(desc):
    """'Left-arm orthodox stock length ball' -> (type, variation, length)."""
    text = desc.strip()
    low = text.lower()

    length = None
    for candidate in LENGTHS:
        if low.endswith(candidate):
            length = candidate
            text = text[: len(text) - len(candidate)].strip()
            break

    variation = None
    for candidate in BOWL_VARIATIONS:
        pattern = re.compile(rf"\b{candidate}\b\s*$", re.I)
        if pattern.search(text):
            variation = candidate
            text = pattern.sub("", text).strip()
            break

    bowl_type = text.strip(" ,.") or None
    return (bowl_type.lower() if bowl_type else None), variation, length


def _parse_shot_segment(segment, out):
    """Pull shot, trajectory, control, fielder and position from the tail."""
    text = segment

    match = _FIELDER_AT.search(text)
    if match:
        out["fielder_name"] = match.group(1).strip()
        out["field_position"] = match.group(2).strip()
        text = text[: match.start()]
    else:
        match = re.search(r"\bto ([\w' ]+?)\s+for\b", text)
        if match:
            out["field_position"] = match.group(1).strip()
            text = text[: match.start()]

    caught = _CAUGHT_BY.search(segment)
    if caught and not out.get("fielder_name"):
        out["fielder_name"] = caught.group(1).strip()

    out["shot_trajectory"] = _find_first(text, TRAJECTORIES)
    out["shot_control"] = _find_first(text, CONTROLS)

    shot = _find_first(text, sorted(SHOTS, key=len, reverse=True))
    if shot:
        out["shot_type"] = SHOTS[shot]


def _parse_dismissal(text, out):
    """Dismissal mode, and the batter/score when the text spells them out.

    Three shapes occur: a '<batter> Caught for 33.' tail, a '<batter> caught -
    26(63)' tail, and prose that only describes what happened. The first two
    give a batter and score; the third gives mode alone.
    """
    match = _DISMISS_DASH.search(text) or _DISMISS_TAIL.search(text)
    if match:
        out["dismissed_batter"] = match.group(1).strip()
        out["dismissal_type"] = match.group(2).lower().strip()
        out["dismissal_score"] = match.group(3)
        return

    for pattern, mode in PROSE_DISMISSALS:
        if re.search(pattern, text, re.I):
            out["dismissal_type"] = mode
            return


def parse_scoring(code):
    """Decode a ``scoring`` cell: runs, extras type and the usual boolean flags."""
    out = {
        "runs_scored": None, "extras_type": None, "is_wicket": False,
        "is_wide": False, "is_no_ball": False, "is_bye": False,
        "is_leg_bye": False, "is_penalty": False, "is_boundary_four": False,
        "is_boundary_six": False, "is_dot": False,
    }
    if code is None:
        return out

    text = str(code).strip()
    out["is_wicket"] = "W" in text

    for marker, name in EXTRAS_MARKERS:
        if marker in text:
            out["extras_type"] = name
            out[{
                "wide": "is_wide", "no ball": "is_no_ball", "bye": "is_bye",
                "leg bye": "is_leg_bye", "penalty": "is_penalty",
            }[name]] = True
            text = text.replace(marker, "")
            break

    digits = re.search(r"\d+", text)
    if digits:
        runs = int(digits.group())
        out["runs_scored"] = runs
        out["is_boundary_four"] = runs == 4 and not out["extras_type"]
        out["is_boundary_six"] = runs == 6 and not out["extras_type"]
        out["is_dot"] = runs == 0 and not out["extras_type"] and not out["is_wicket"]
    elif text.replace(",", "").strip() in ("W", ""):
        out["runs_scored"] = 0

    return out


def parse_commentary(text, is_wicket=None):
    """Best-effort structured read of one commentary string.

    ``is_wicket`` comes from the ``scoring`` column and gates dismissal
    extraction. Without it a bare word like "caught" in ordinary fielding
    commentary would be read as a dismissal, so when the flag says the ball was
    not a wicket the dismissal fields are left empty. Passing None falls back to
    looking for an explicit "OUT!" marker.
    """
    out = dict(EMPTY)
    if not text or not isinstance(text, str):
        return out

    if is_wicket is None:
        is_wicket = "OUT!" in text

    arm = _ARM.match(text)
    if arm:
        out["bowler_arm"] = f"{arm.group(1).lower()} the wicket"
        parts = [p.strip() for p in arm.group(2).split(",")]
        if len(parts) >= 3:
            out["length"] = _find_first(parts[0], LENGTHS)
            out["line"] = _find_first(parts[1], LINES)
            out["footwork"] = _find_first(parts[2], FOOTWORK)
            tail = ",".join(parts[3:]) if len(parts) > 3 else ""
            tail = re.sub(r"\.\s*(No run|\d+ \w+).*$", "", tail).strip()
            _parse_shot_segment(tail, out)
            out["parse_method"] = "template_arm"
            if is_wicket:
                _parse_dismissal(text, out)
            return out

    header = _HEADER.match(text)
    if header:
        body = header.group("body")
        struct = _STRUCT_BODY.match(body)
        if struct:
            bowl_type, variation, length = _split_bowl_descriptor(struct.group("desc"))
            out["delivery_type"] = bowl_type
            out["delivery_variation"] = variation
            out["length"] = length
            out["line"] = (
                _find_first(struct.group("line"), LINES)
                or struct.group("line").strip().lower()
            )

            tail = struct.group("tail")
            foot = _FOOT_PREFIX.match(tail)
            if foot:
                out["footwork"] = foot.group(1).lower()
                tail = foot.group(2)
            _parse_shot_segment(tail, out)
            out["parse_method"] = "template_structured"
            if is_wicket:
                _parse_dismissal(text, out)
            return out

        # Header present but no descriptor -- the minimal form.
        if re.match(r"^\s*(No run|\d+\s+\w+)\.?\s*$", body, re.I):
            out["parse_method"] = "minimal"
            if is_wicket:
                _parse_dismissal(text, out)
            return out

    # Prose: same vocabulary, no grammar.
    out["length"] = _find_first(text, LENGTHS)
    out["line"] = _find_first(text, LINES)
    out["footwork"] = _find_first(text, FOOTWORK)
    out["shot_trajectory"] = _find_first(text, TRAJECTORIES)
    out["shot_control"] = _find_first(text, CONTROLS)
    shot = _find_first(text, sorted(SHOTS, key=len, reverse=True))
    if shot:
        out["shot_type"] = SHOTS[shot]
    caught = _CAUGHT_BY.search(text)
    if caught:
        out["fielder_name"] = caught.group(1).strip()

    position = _PROSE_POSITION.search(text)
    if position:
        out["field_position"] = position.group(1).lower()

    if is_wicket:
        _parse_dismissal(text, out)

    if any(out[k] for k in ("length", "line", "shot_type", "footwork")):
        out["parse_method"] = "keyword"
    return out


# --- frame-level enrichment -------------------------------------------------

PARSED_COLS = [
    "bowler_arm", "delivery_type", "delivery_variation", "length", "line",
    "footwork", "shot_type", "shot_trajectory", "shot_control",
    "field_position", "fielder_name", "dismissal_type", "dismissed_batter",
    "dismissal_score", "parse_method",
]
SCORING_COLS = [
    "runs_scored", "extras_type", "is_wicket", "is_wide", "is_no_ball",
    "is_bye", "is_leg_bye", "is_penalty", "is_boundary_four",
    "is_boundary_six", "is_dot",
]
DERIVED_COLS = PARSED_COLS + SCORING_COLS + ["dismissal_source"]


def enrich(df):
    """Add the parsed attribute and decoded-scoring columns to a ball table.

    Idempotent: re-running on an already-enriched frame replaces the derived
    columns rather than duplicating them, so ball tables can be re-parsed in
    place when the vocabularies improve.
    """
    import pandas as pd

    base = df.drop(columns=[c for c in DERIVED_COLS if c in df.columns])

    # Scoring is decoded first: its wicket flag gates dismissal extraction, so
    # ordinary fielding commentary can't be misread as a dismissal.
    scored = pd.DataFrame([parse_scoring(s) for s in base["scoring"]], index=base.index)
    parsed = pd.DataFrame(
        [
            parse_commentary(text, is_wicket=wicket)
            for text, wicket in zip(base["commentary"], scored["is_wicket"])
        ],
        index=base.index,
    )
    out = pd.concat([base, parsed[PARSED_COLS], scored[SCORING_COLS]], axis=1)

    # The wagon endpoint's catch_map is an independent record of catches. Where
    # the text was too thin to name a mode, it settles the question. Only fills
    # blanks -- an explicit mode in the text always wins.
    out["dismissal_source"] = None
    out.loc[out["dismissal_type"].notna(), "dismissal_source"] = "commentary"

    if "in_catch_map" in out.columns:
        fill = (
            out["in_catch_map"].eq(True)
            & out["is_wicket"].eq(True)
            & out["dismissal_type"].isna()
        )
        out.loc[fill, ["dismissal_type", "dismissal_source"]] = ["caught", "catch_map"]

    return out
