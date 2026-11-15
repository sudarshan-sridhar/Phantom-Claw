"""Reading names and commands out of what the speech model heard.

Character names go through Merlin's tolerant matcher first. On top of that we accept the
mistakes speech recognition tends to make with these names ("band" for Ban, "gother" for
Gowther, "scanner" for Escanor and so on).
"""
from __future__ import annotations

import difflib
import re

from backend.agents.cast import CAST
from backend.agents.merlin import find_named_agent

# Extra spellings the speech model comes up with. Matched as whole words.
SPEECH_ALIASES = {
    "ban": ["ban", "band", "bann", "bahn", "van", "bans", "ben"],
    "gowther": ["gowther", "gother", "gauther", "gouther", "gowter", "gowthar", "gotha", "gauthier", "gaither", "goether"],
    "escanor": ["escanor", "escanoor", "escaner", "eskanor", "scanner", "scanor", "escanar", "escano"],
    "merlin": ["merlin", "merlyn", "merlins", "marlin", "merlen"],
    "elizabeth": ["elizabeth", "elisabeth", "elizabeths", "elisabet"],
    "king": ["king", "kings"],
}
_ALIAS_WORD = {w: agent for agent, words in SPEECH_ALIASES.items() for w in words}

# Little words people put around a name that do not make it a request.
FILLER = {
    "hey", "hi", "hello", "ok", "okay", "yo", "um", "uh", "oh", "so", "and", "please", "the",
    "you", "there", "a", "ah", "hmm", "mm", "well", "now",
}

_PHANTOM = ["phantom", "fantom", "phantum", "fanton", "phanton", "fandom", "phantoms", "fantum"]
_CLAW = ["claw", "claws", "clock", "cloud", "clause", "claude", "clog", "clo", "klaw", "cla", "clawed", "law", "close"]
_START = {"start", "started", "starts", "starting", "wake", "begin", "go", "on", "up"}
_STOP = {"stop", "stopped", "stops", "stopping", "sleep", "sleeping", "slip", "off", "quiet", "pause", "rest"}


def words_of(text: str) -> list[str]:
    t = (text or "").lower().replace("'s", "").replace("’s", "")
    return re.findall(r"[a-z]+", t)


def _like(word: str, choices: list[str], cutoff: float) -> bool:
    if word in choices:
        return True
    return bool(difflib.get_close_matches(word, choices, n=1, cutoff=cutoff))


def find_wake(text: str) -> tuple[str | None, int, int]:
    """Look for "Phantom Claw" and what follows it.

    Returns (command, start, end) where command is "start", "stop" or None when the phrase
    is not there, and start..end is the span of words it used.
    """
    ws = words_of(text)
    for i, w in enumerate(ws):
        claw_at = None
        if w.startswith(("phantom", "fantom")) and len(w) > 8 and _like(w[w.index("tom") + 3:], _CLAW, 0.7):
            claw_at = i  # "phantomclaw" written as one word
        elif _like(w, _PHANTOM, 0.72) and i + 1 < len(ws) and _like(ws[i + 1], _CLAW, 0.7):
            claw_at = i + 1
        if claw_at is None:
            continue
        # Only the word right after the phrase counts (skipping a "please" or "now"), so
        # "Phantom Claw start, Ban stop that" is still a start.
        j = claw_at + 1
        while j < len(ws) and ws[j] in ("please", "now", "and", "so"):
            j += 1
        if ws[j : j + 3] == ["go", "to", "sleep"]:
            return "stop", i, j + 3
        if j < len(ws) and ws[j] in _STOP:
            return "stop", i, j + 1
        if j < len(ws) and ws[j] in _START:
            return "start", i, j + 1
        return "start", i, claw_at + 1
    return None, -1, -1


def find_agent(text: str) -> tuple[str | None, int]:
    """The first character named in the text and the index of that word, or (None, -1)."""
    for i, w in enumerate(words_of(text)):
        agent = _ALIAS_WORD.get(w) or find_named_agent(w)
        if agent:
            return agent, i
    return None, -1


def name_alone(text: str, agent: str) -> bool:
    """True when the utterance is just the name, maybe with a "hey" or an "okay" around it."""
    rest = []
    for w in words_of(text):
        if w in FILLER:
            continue
        if (_ALIAS_WORD.get(w) or find_named_agent(w)) == agent:
            continue
        rest.append(w)
    return not rest


def called_text(agent: str) -> str:
    """What we send when someone only says a name."""
    return f"{CAST[agent]['name']}, are you there?"
