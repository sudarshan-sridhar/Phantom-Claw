"""Merlin the Conductor: name matching, choosing by role, the short argument, and the plan check."""
from __future__ import annotations

import difflib
import os
import random
import re
from typing import Any

from . import cast as cast_mod
from .cast import AGENT_IDS, ARM_MAP, CAST, SAFE_LINES

# Skills an LLM plan may use. `gesture` exists on the arm but only the server starts it.
PLAN_SKILLS = {"grip_bottle", "point", "half_reach", "wave", "rest"}
TOOLS = {"note", "todo", "timer", "open_url", "lookup"}
SAY_MAX = 140

# ---------------------------------------------------------------- name matching

_ALIASES = {
    "escanor": ["escanor"],
    "elizabeth": ["elizabeth", "liz", "ellie"],
    "ban": ["ban"],
    "king": ["king"],
    "gowther": ["gowther"],
    "merlin": ["merlin"],
}


def _skeleton(word: str) -> str:
    s = re.sub(r"[aeiouwhy]", "", word)
    return re.sub(r"(.)\1+", r"\1", s)


def _word_matches(word: str, name: str) -> bool:
    if word == name:
        return True
    if len(name) <= 4 or len(word) < 4:
        return False  # short names (ban, king, liz) only match exactly
    if word[0] != name[0]:
        return False
    ratio = difflib.SequenceMatcher(None, word, name).ratio()
    if ratio >= 0.75:
        return True
    return ratio >= 0.6 and _skeleton(word) == _skeleton(name)


def find_named_agent(text: str) -> str | None:
    """Return the first agent named in the text, tolerating near misses. Merlin is returned as 'merlin'."""
    for word in re.findall(r"[a-z]+", text.lower()):
        for agent, names in _ALIASES.items():
            if any(_word_matches(word, n) for n in names):
                return agent
    return None


# ---------------------------------------------------------------- choosing by role

_ROLE_RULES = [
    ("gowther", re.compile(r"\b(what|who|why|how|when|where|look up|lookup|explain|define|tell me about)\b|\?", re.I)),
    ("elizabeth", re.compile(r"\b(note|notes|todo|to do|to-do|remind|remember|write down|list)\b", re.I)),
    ("king", re.compile(r"\b(timer|minutes?|seconds?|alarm|countdown)\b", re.I)),
    ("escanor", re.compile(r"\b(wave|show off|impress|strong|hello|hi there|greet)\b", re.I)),
    ("ban", re.compile(r"\b(quick|quickly|fast|hurry|now|asap|right away)\b", re.I)),
]

_last_chosen: list[str] = []


def choose_by_role(text: str, rng: random.Random | None = None) -> str:
    rng = rng or random
    again = cast_mod.pending_repeat_agent(text)
    if again:
        return again  # whoever dodged it last time gets it back
    for agent, rx in _ROLE_RULES:
        if rx.search(text):
            return agent
    pool = [a for a in AGENT_IDS if not _last_chosen or a != _last_chosen[-1]]
    return rng.choice(pool)


def note_chosen(agent: str) -> None:
    _last_chosen.append(agent)
    del _last_chosen[:-5]


# ---------------------------------------------------------------- Merlin's lines and the argument

_HANDOFF = [
    "{name}, this one's yours.",
    "{name}, you take it.",
    "Let's give this to {name}.",
    "{name}, you're up.",
    "All right, {name}. Go on.",
]

_WANT = {
    "escanor": ["Leave it. This one needs me.", "Step aside, I've got this."],
    "elizabeth": ["I can do it, if that's okay?", "Oh, let me. I'll be careful."],
    "ban": ["Mine. Hand it over.", "Me, me. I'm way faster."],
    "king": ["Fine. If nobody else will.", "I guess I could. Barely."],
    "gowther": ["I'm free. I'll do it.", "I can do that one."],
}
_DODGE = {
    "escanor": ["That's beneath me. Ask someone else.", "I don't fetch."],
    "elizabeth": ["Maybe someone else wants this one?", "I don't mind if someone else goes first."],
    "ban": ["Boring. Pass.", "Nah, not that one. Give me something fun."],
    "king": ["Ugh, not me.", "Way too far. Pass."],
    "gowther": ["Makes no difference to me.", "Someone else can have it."],
}
# How keen each one is, used to pick want or dodge.
_KEEN = {"escanor": 0.3, "elizabeth": 0.7, "ban": 0.95, "king": 0.1, "gowther": 0.5}

_SETTLE_WANTER = ["{name} wants it, so {name} gets it.", "Fine. {name}, go."]
_SETTLE_DODGER = ["Nice try, {name}. It's yours.", "{name}, you're doing it anyway."]


def merlin_handoff(chosen: str, rng: random.Random | None = None) -> str:
    rng = rng or random
    return rng.choice(_HANDOFF).format(name=CAST[chosen]["name"])


def argue(rng: random.Random | None = None) -> tuple[list[dict], str, str]:
    """Two agents each say one line wanting or dodging. Returns (argued, chosen, merlin_line)."""
    rng = rng or random
    a, b = rng.sample(AGENT_IDS, 2)
    argued = []
    for agent in (a, b):
        wants = rng.random() < _KEEN[agent]
        line = rng.choice(_WANT[agent] if wants else _DODGE[agent])
        argued.append({"agent": agent, "say": line, "wants": wants})
    wanters = [x for x in argued if x["wants"]]
    dodgers = [x for x in argued if not x["wants"]]
    if dodgers and (not wanters or rng.random() < 0.5):
        pick = rng.choice(dodgers)
        line = rng.choice(_SETTLE_DODGER)
    else:
        pick = rng.choice(wanters)
        line = rng.choice(_SETTLE_WANTER)
    chosen = pick["agent"]
    for x in argued:
        x.pop("wants")
    return argued, chosen, line.format(name=CAST[chosen]["name"])


def should_argue(rng: random.Random | None = None) -> bool:
    """About one request in four. PHANTOM_ARGUE=0..1 changes the chance (for rehearsals)."""
    try:
        chance = float(os.environ.get("PHANTOM_ARGUE", "0.25"))
    except ValueError:
        chance = 0.25
    return (rng or random).random() < chance


# ---------------------------------------------------------------- the plan check


def _clamp(v: Any, lo: float, hi: float, default: float) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    if f != f:  # NaN
        return default
    return round(min(hi, max(lo, f)), 2)


def _clean_say(text: Any) -> str:
    s = str(text or "").strip()
    s = s.replace(chr(0x2014), ", ").replace(chr(0x2013), ", ")  # long and short dashes
    s = re.sub(r"\s+", " ", s)
    if len(s) > SAY_MAX:
        cut = s[:SAY_MAX]
        sp = cut.rfind(" ")
        s = (cut[:sp] if sp > 80 else cut).rstrip(" ,;:") + "."
        s = s[:SAY_MAX]
    return s


def safe_plan(agent: str) -> dict:
    return {"agent": agent, "say": SAFE_LINES.get(agent, "I cannot do that."), "arm": None, "laptop": []}


def _check_arm(arm: Any, agent: str, issues: list[str]) -> tuple[dict | None, bool]:
    """Return (arm, ok). ok False means the whole plan must be replaced."""
    if arm is None:
        return None, True
    if not isinstance(arm, dict):
        issues.append("arm is not an object")
        return None, False
    skill = arm.get("skill")
    if skill not in PLAN_SKILLS:
        issues.append(f"unknown or forbidden skill {skill!r}")
        return None, False
    if skill == "grip_bottle":
        base = ARM_MAP[agent]["agree"]
        out = {
            "skill": skill,
            "pace": _clamp(arm.get("pace", base["pace"]), 0.0, 1.0, base["pace"]),
            "hold_s": _clamp(arm.get("hold_s", base["hold_s"]), 1, 8, base["hold_s"]),
            "flourish": bool(arm.get("flourish", False)),
            "look_first": bool(arm.get("look_first", False)),
        }
    elif skill in ("point", "half_reach"):
        out = {"skill": skill, "pace": _clamp(arm.get("pace", 0.5), 0.0, 1.0, 0.5)}
    else:
        out = {"skill": skill}
    for k, v in arm.items():
        if k != "skill" and k not in out:
            issues.append(f"dropped arm param {k}")
        elif k in ("pace", "hold_s") and isinstance(v, (int, float)) and out.get(k) != round(float(v), 2):
            issues.append(f"clamped {k} {v} -> {out[k]}")
    return out, True


def _check_tool(call: Any, issues: list[str]) -> dict | None:
    if not isinstance(call, dict) or call.get("tool") not in TOOLS:
        issues.append(f"dropped unknown tool {call!r}"[:120])
        return None
    t = call["tool"]
    if t in ("note", "todo"):
        text = str(call.get("text") or "").strip()[:500]
        return {"tool": t, "text": text} if text else None
    if t == "timer":
        secs = int(_clamp(call.get("seconds"), 1, 3600, 60))
        return {"tool": t, "seconds": secs, "label": str(call.get("label") or "timer")[:80]}
    if t == "open_url":
        url = str(call.get("url") or "").strip()
        if not re.match(r"^https?://[^\s]+$", url, re.I):
            issues.append("dropped open_url that was not http or https")
            return None
        return {"tool": t, "url": url[:2000]}
    if t == "lookup":
        q = str(call.get("query") or "").strip()[:300]
        return {"tool": t, "query": q} if q else None
    return None


def check_plan(raw: Any, agent: str, decision: dict | None = None) -> tuple[dict, list[str], bool]:
    """Validate and clamp a plan. Returns (plan, issues, replaced).

    A plan with a bad arm part, or no usable line, is replaced by a safe one.
    For bottle jobs the agent's ego mapping (from cast.decide) is enforced.
    """
    issues: list[str] = []
    if agent not in AGENT_IDS:
        raise ValueError(f"not an agent: {agent}")
    if not isinstance(raw, dict):
        return safe_plan(agent), ["plan is not an object"], True

    if raw.get("agent") != agent:
        if raw.get("agent") is not None:
            issues.append(f"agent {raw.get('agent')!r} replaced by {agent!r}")

    say = _clean_say(raw.get("say"))
    if not say:
        return safe_plan(agent), issues + ["empty say"], True

    arm, ok = _check_arm(raw.get("arm"), agent, issues)
    if not ok:
        return safe_plan(agent), issues, True

    if decision and decision.get("bottle_job"):
        wanted = dict(decision["arm"])
        if arm is None or arm.get("skill") in ("grip_bottle", "point", "half_reach"):
            if arm != wanted:
                issues.append(f"ego mapping applied: {wanted['skill']}")
            arm = wanted
    elif arm and arm["skill"] == "grip_bottle":
        arm = dict(ARM_MAP[agent]["agree"])

    laptop_raw = raw.get("laptop") or []
    if not isinstance(laptop_raw, list):
        issues.append("laptop is not a list")
        laptop_raw = []
    laptop = [c for c in (_check_tool(x, issues) for x in laptop_raw[:5]) if c]

    return {"agent": agent, "say": say, "arm": arm, "laptop": laptop}, issues, False
