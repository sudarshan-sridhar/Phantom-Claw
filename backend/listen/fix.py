"""A small, careful fixer for the words that matter, run on every transcript.

It only ever writes one of these: the six cast names, "Phantom Claw", and the five item names.
- Names are looked for in the first one or two words (after a "hey" or an "okay"), and right
  after "Phantom Claw start". Long names (Escanor, Elizabeth, Gowther, Merlin) may be fuzzy
  matched by edit distance, also across two words ("s kanor", "go there"). Ban and King are
  short enough that fuzzy matching would eat ordinary words, so they only come from a list.
- "Phantom Claw" is fixed anywhere when both halves are close ("fan tom claw", "phantom clock").
- Items are fixed anywhere, as whole words, from a fixed list of known misses ("sharp e").
- Right after a name at the start, "pointed (at) the" and "picked up the" go back to "point at
  the" and "pick up the" (ElevenLabs hears "Ban, point at the" as "Ban pointed the").
Everything else is left alone.
"""
from __future__ import annotations

import difflib
import re

NAME_CANON = {"escanor": "Escanor", "elizabeth": "Elizabeth", "ban": "Ban", "king": "King", "gowther": "Gowther", "merlin": "Merlin"}
LONG_NAMES = ("escanor", "elizabeth", "gowther", "merlin")

# Known misses, matched exactly (after lower-casing). Two-word misses are written with a space.
NAME_ALIASES = {
    "ban": ["ban", "bann", "bahn", "band", "bun", "baan", "bon", "van", "ben", "bans"],
    "king": ["king", "keng", "kink", "kin", "kings"],
    "escanor": ["s kanor", "es kanor", "is kanor", "s canor", "es canor", "escano", "scanner", "scanor", "ask an or", "s connor", "escanar"],
    "gowther": ["go there", "go ther", "gau ther", "gow ther", "goth er", "gotha", "gauther", "gouther", "gother", "gaither", "gauthier", "go thar", "gowda", "gotter"],
    "merlin": ["marlin", "merlyn", "merlen", "mer lin", "mar lin", "marlene"],
    "elizabeth": ["elisabeth", "elizabet", "alizabeth", "eliza beth", "elisa beth", "elizebeth"],
}
_ALIAS = {a: n for n, al in NAME_ALIASES.items() for a in al}

# Real words that sit close to a long name and must never be rewritten by the fuzzy match.
PROTECT = {
    "gather", "gathered", "mother", "other", "others", "together", "either", "neither", "father", "further",
    "rather", "whether", "weather", "brother", "bother", "go", "got", "goat", "escape", "escaped", "scan",
    "merely", "marlon", "marine", "berlin", "melting", "learning", "elisa", "isabel", "elastic",
}

# Two-word misses that are also ordinary phrases. These only count when a pause or the end
# follows, so "go there and pick it up" is left alone.
NEEDS_PAUSE = {"go there", "go ther", "go thar", "ask an or", "mar lin"}

# Short words that may sit in front of a name and are skipped when looking for it.
LEAD = {"hey", "hi", "hello", "ok", "okay", "um", "uh", "oh", "so", "and", "yo", "please", "now", "well"}

ITEM_ALIASES = {
    "Sharpie": ["sharp e", "sharpy", "sharpey", "sharpee", "sharp pea", "sharp pee", "shar pee", "sharpe", "sherpa"],
    "glue stick": ["glue stik", "glu stick", "blue stick", "clue stick", "gluestick", "glue steak", "glue stake"],
    "earbuds": ["ear buds", "air buds", "airbuds", "earbud", "ear bud", "ear pods", "here buds", "beer buds", "ear bugs"],
    "scissors": ["scissor", "sizzors", "scizzors", "sissors", "scisors", "sizzers"],
    "bottle": ["bottel", "botel", "bottal", "botal", "bottl"],
}

_PHANTOM = ["phantom", "fantom", "phantum", "fanton", "phanton", "fandom", "phantoms", "fantum", "phantam", "panthom", "pantom"]
_CLAW = ["claw", "claws", "clock", "cloud", "clause", "claude", "clog", "klaw", "clawed", "call", "crawl", "clo", "clow", "claro", "cloth", "clah", "glow", "law", "close", "clough"]

_FAN = {"fan", "phan", "pan", "fun", "fen", "phun", "fann"}
_TOM = {"tom", "tum", "dom", "thom", "tam", "ton", "tomb"}
_NOT_PHANTOM = {"random", "tandem", "fathom", "phantasm"}

_TOKEN = re.compile(r"[A-Za-z']+")


def _skel(w: str) -> str:
    """A rough sound-alike spelling: ph=f, th=t, c=k, au/ow/ou=o, no doubled letters."""
    w = w.lower().replace("'", "")
    for a, b in (("ph", "f"), ("th", "t"), ("ck", "k"), ("c", "k"), ("z", "s"), ("au", "o"), ("ow", "o"), ("ou", "o"), ("oo", "o"), ("ee", "i"), ("y", "i")):
        w = w.replace(a, b)
    return re.sub(r"(.)\1+", r"\1", w)


def _ratio(a: str, b: str) -> float:
    return max(difflib.SequenceMatcher(None, a, b).ratio(), difflib.SequenceMatcher(None, _skel(a), _skel(b)).ratio())


def _like(w: str, choices: list[str], cutoff: float) -> bool:
    w = w.lower()
    return w in choices or any(_ratio(w, c) >= cutoff for c in choices)


def match_name(words: list[str]) -> str | None:
    """The cast name these one or two words are a near miss for, or None."""
    joined = " ".join(w.lower() for w in words)
    if joined in _ALIAS:
        return _ALIAS[joined]
    glued = "".join(w.lower() for w in words)
    if len(words) == 1 and glued in NAME_CANON:
        return glued
    if glued in PROTECT or (len(words) == 1 and len(glued) < 5):
        return None
    best, score = None, 0.0
    for n in LONG_NAMES:
        r = _ratio(glued, n)
        if r > score:
            best, score = n, r
    # One word needs to be close; two words glued together need to be closer still.
    need = 0.78 if len(words) == 1 else 0.84
    return best if score >= need else None


def _tokens(text: str) -> list[tuple[int, int, str]]:
    return [(m.start(), m.end(), m.group(0)) for m in _TOKEN.finditer(text)]


def _replace(text: str, spans: list[tuple[int, int, str]]) -> str:
    for a, b, new in sorted(spans, reverse=True):
        text = text[:a] + new + text[b:]
    return text


def _fix_phantom(text: str, changes: list) -> str:
    toks = _tokens(text)
    spans = []
    i = 0
    while i < len(toks):
        w = toks[i][2].lower()
        hit = None
        if (i + 1 < len(toks) and w not in _NOT_PHANTOM and _like(w, _PHANTOM, 0.75)
                and _like(toks[i + 1][2], _CLAW, 0.75)):
            hit = (i, i + 1)
        elif i + 2 < len(toks) and w in _FAN and toks[i + 1][2].lower() in _TOM and _like(toks[i + 2][2], _CLAW, 0.75):
            hit = (i, i + 2)  # "fan tom claw"
        elif w.startswith(("phantom", "fantom")) and len(w) > 8 and _like(w[w.index("tom") + 3:], _CLAW, 0.7):
            hit = (i, i)       # "phantomclaw"
        if hit:
            a, b = toks[hit[0]][0], toks[hit[1]][1]
            old = text[a:b]
            if old != "Phantom Claw":
                spans.append((a, b, "Phantom Claw"))
                changes.append((old, "Phantom Claw"))
            i = hit[1] + 1
            continue
        i += 1
    return _replace(text, spans)


def _fix_names(text: str, changes: list) -> str:
    toks = _tokens(text)
    starts = []
    i = 0
    while i < len(toks) and toks[i][2].lower() in LEAD:
        i += 1
    starts.append(i)
    # Right after "Phantom Claw start" too ("Phantom Claw start, Ban ...").
    for j in range(len(toks) - 1):
        if toks[j][2] == "Phantom" and toks[j + 1][2] == "Claw":
            k = j + 2
            if k < len(toks) and toks[k][2].lower() in ("start", "wake", "begin"):
                k += 1
            while k < len(toks) and toks[k][2].lower() in LEAD:
                k += 1
            starts.append(k)
    # And a name at the very end after a comma ("bring me the Sharpie, Elizabeth").
    comma = text.rfind(",")
    if comma > 0:
        tail = [n for n, t in enumerate(toks) if t[0] > comma]
        if 1 <= len(tail) <= 2:
            starts.append(tail[0])
    spans = []
    for s in starts:
        if s >= len(toks):
            continue
        # Prefer the two-word reading only when the pair is followed by a pause or the end,
        # so "go there and pick it up" stays as it is.
        for n in (2, 1):
            if s + n > len(toks):
                continue
            if n == 2:
                end = toks[s + 1][1]
                after = text[end:end + 1]
                paused = after in (",", ".", "!", "?", ":", ";") or end >= len(text.rstrip())
                pair = " ".join(t[2].lower() for t in toks[s:s + 2])
                if not paused and (pair in NEEDS_PAUSE or pair not in _ALIAS):
                    continue
            name = match_name([t[2] for t in toks[s:s + n]])
            if name:
                a, b = toks[s][0], toks[s + n - 1][1]
                old, new = text[a:b], NAME_CANON[name]
                if old != new:
                    spans.append((a, b, new))
                    changes.append((old, new))
                break
    return _replace(text, spans)


def _fix_items(text: str, changes: list) -> str:
    for canon, aliases in ITEM_ALIASES.items():
        for al in sorted(aliases, key=len, reverse=True):
            pat = re.compile(r"\b" + r"\s+".join(map(re.escape, al.split())) + r"\b", re.IGNORECASE)
            def sub(m, canon=canon):
                if m.group(0).lower() == canon.lower():
                    return m.group(0)
                changes.append((m.group(0), canon))
                return canon
            text = pat.sub(sub, text)
    return text


# Right after a name at the start, a verb heard in the wrong tense ("Ban pointed the Sharpie")
# is put back to the order it almost certainly was. Nothing else about verbs is touched.
_VERB_AFTER_NAME = re.compile(
    r"^(?P<lead>(?:(?:hey|ok|okay|so)\W+)?)(?P<name>Escanor|Elizabeth|Ban|King|Gowther|Merlin)\W+"
    r"(?P<verb>pointed(?:\s+at)?|points(?:\s+at)?|picked\s+up|picks\s+up)\s+(?P<obj>the|it|that|this)\b",
    re.IGNORECASE,
)
_VERB_FIX = {"pointed": "point at", "points": "point at", "picked": "pick up", "picks": "pick up"}


def _fix_verbs(text: str, changes: list) -> str:
    m = _VERB_AFTER_NAME.match(text)
    if not m:
        return text
    new_verb = _VERB_FIX[m.group("verb").split()[0].lower()]
    new = f"{m.group('lead')}{m.group('name')}, {new_verb} {m.group('obj')}"
    changes.append((m.group(0), new))
    return new + text[m.end():]


def repair(text: str) -> tuple[str, list[tuple[str, str]]]:
    """Return the fixed text and a list of (old, new) changes."""
    if not text:
        return text, []
    changes: list[tuple[str, str]] = []
    out = _fix_phantom(text, changes)
    out = _fix_names(out, changes)
    out = _fix_items(out, changes)
    out = _fix_verbs(out, changes)
    return out, changes
