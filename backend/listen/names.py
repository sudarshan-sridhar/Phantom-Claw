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

_PHANTOM = ["phantom", "fantom", "phantum", "fanton", "phanton", "fandom", "phantoms", "fantum", "panthom",
            "pantom", "phantam", "fanthom", "fantam", "phandom", "fantham", "phantem"]
_CLAW = ["claw", "claws", "clock", "cloud", "clause", "claude", "clog", "clo", "klaw", "cla", "clawed", "law", "close",
         "call", "crawl", "clow", "claro", "cloth", "clah", "glow", "klo", "clough", "claws", "clore", "clos"]
# "start" and "stop" as an Indian English accent and the speech models tend to give them back.
_START = {"start", "starts", "started", "starting", "stat", "star", "stars", "stark", "wake", "begin", "go", "on", "up"}
_STOP = {"stop", "stopped", "stops", "stopping", "stap", "stab", "top", "halt", "sleep", "sleeping", "slip", "off",
         "end", "quit", "quiet", "pause", "rest", "freeze"}
_FAN = {"fan", "phan", "pan", "fun", "fen", "phun", "fann"}
_TOM = {"tom", "tum", "dom", "thom", "tam", "ton", "tomb"}
_NOT_PHANTOM = {"random", "tandem", "fathom", "phantasm"}
_SKIP = {"please", "now", "and", "so", "um", "uh", "the", "a"}

# Said on its own while awake, these stop the arm too: "stop", "stop it", "arm stop", "halt".
_SHORT_STOP = {"stop", "stopped", "stops", "stap", "halt", "freeze"}
_SHORT_FILL = {"please", "now", "it", "arm", "robot", "the", "everything", "hey", "ok", "okay", "just", "all",
               "right", "phantom", "claw", "oh", "no", "wait", "that", "everyone", "everybody"}


def words_of(text: str) -> list[str]:
    t = (text or "").lower().replace("'s", "").replace("’s", "")
    return re.findall(r"[a-z]+", t)


def _sk(w: str) -> str:
    for a, b in (("ph", "f"), ("th", "t"), ("ck", "k"), ("c", "k"), ("w", ""), ("h", "")):
        w = w.replace(a, b)
    return re.sub(r"(.)\1+", r"\1", w)


def _like(word: str, choices: list[str], cutoff: float) -> bool:
    if word in choices:
        return True
    if difflib.get_close_matches(word, choices, n=1, cutoff=cutoff):
        return True
    sk = _sk(word)
    return len(sk) >= 3 and bool(difflib.get_close_matches(sk, [_sk(c) for c in choices], n=1, cutoff=cutoff + 0.05))


def _phantom_at(ws: list[str], i: int) -> int | None:
    """If a "phantom" starts at word i, the index of its last word (it may be split: "fan tom")."""
    w = ws[i]
    if len(w) >= 5 and w not in _NOT_PHANTOM and _like(w, _PHANTOM, 0.72):
        return i
    if i + 1 < len(ws) and w in _FAN and ws[i + 1] in _TOM:
        return i + 1
    return None


def find_wake(text: str) -> tuple[str | None, int, int]:
    """Look for "Phantom Claw" and a start or stop word within two words after it.

    Returns (command, start, end) where command is "start", "stop" or None when the phrase
    is not there, and start..end is the span of words it used. "Phantom Claw" alone is a start.
    A "phantom" whose "claw" got lost still counts when a start or stop word follows it.
    """
    ws = words_of(text)
    for i in range(len(ws)):
        w = ws[i]
        claw_at = None
        if w.startswith(("phantom", "fantom")) and len(w) > 8 and _like(w[w.index("tom") + 3:], _CLAW, 0.7):
            p_end = claw_at = i  # "phantomclaw" written as one word
        else:
            p_end = _phantom_at(ws, i)
            if p_end is None:
                continue
            if p_end + 1 < len(ws) and _like(ws[p_end + 1], _CLAW, 0.72):
                claw_at = p_end + 1
        after = claw_at if claw_at is not None else p_end
        # The command is one of the next two words, not counting a "please" or a "now".
        j, seen = after + 1, 0
        while j < len(ws) and seen < 2:
            if ws[j] in _SKIP:
                j += 1
                continue
            if ws[j : j + 3] == ["go", "to", "sleep"]:
                return "stop", i, j + 3
            if ws[j] in _STOP:
                return "stop", i, j + 1
            if ws[j] in _START:
                if ws[j] == "wake" and j + 1 < len(ws) and ws[j + 1] == "up":
                    return "start", i, j + 2
                return "start", i, j + 1
            seen += 1
            j += 1
        if claw_at is not None:
            return "start", i, claw_at + 1
    return None, -1, -1


def find_short_stop(text: str) -> bool:
    """True for a bare stop: "stop", "stop it", "stop now", "arm stop", "freeze", "Escanor, stop"."""
    ws = words_of(text)
    if not ws or len(ws) > 5:
        return False
    rest = [w for w in ws if w not in _SHORT_FILL and not (_ALIAS_WORD.get(w) or find_named_agent(w))]
    return bool(rest) and all(w in _SHORT_STOP for w in rest)


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
