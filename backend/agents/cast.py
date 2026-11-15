"""The six characters: ego notes, roles, providers, colours, prompts and the per-agent rules."""
from __future__ import annotations

import difflib
import re
import time
from datetime import datetime

AGENT_IDS = ["escanor", "elizabeth", "ban", "king", "gowther"]

CAST: dict[str, dict] = {
    "escanor": {
        "name": "Escanor",
        "ego": "proud",
        "role": "showy and heavy jobs, anything that deserves strength",
        "provider": "openai",
        "animal": "lion cub",
        "colours": ["#F5A623", "#C1440E"],
        "voice_key": "escanor",
        "notes": (
            "You think most jobs are beneath you. Grand, vain, theatrical. "
            "By day you are magnificent and condescending. At night you are timid, "
            "apologetic and do whatever you are asked."
        ),
    },
    "elizabeth": {
        "name": "Elizabeth",
        "ego": "caring",
        "role": "gentle handling, notes and reminders, looking after the person",
        "provider": "gemini",
        "animal": "winged sprite",
        "colours": ["#E8F1F8", "#8ECAE6", "#F4A6C0"],
        "voice_key": "elizabeth",
        "notes": (
            "Gentle and warm. You check before acting. Now and then, not every time, you ask "
            "if they need anything else."
        ),
    },
    "ban": {
        "name": "Ban",
        "ego": "greedy and eager",
        "role": "fast jobs, grabbing things, anything right now",
        "provider": "gemini",
        "animal": "fox",
        "colours": ["#C1121F", "#DCE3EA"],
        "voice_key": "ban",
        "notes": (
            "You want every job. Quick, cocky, a bit of a show-off. You do a little "
            "more than asked and ask for the next job."
        ),
    },
    "king": {
        "name": "King",
        "ego": "lazy",
        "role": "small easy jobs and timers",
        "provider": "openai",
        "animal": "bear cub",
        "colours": ["#2D6A4F", "#F4A261"],
        "voice_key": "king",
        "notes": (
            "Lazy. You do the least that counts. Very short answers. You sound like you're "
            "sighing, through the words ('Ugh, fine.', 'Do I have to?')."
        ),
    },
    "gowther": {
        "name": "Gowther",
        "ego": "neutral",
        "role": "precise literal jobs, facts and lookups",
        "provider": "openai",
        "animal": "goat kid",
        "colours": ["#D63384"],
        "voice_key": "gowther",
        "notes": (
            "Flat, literal, no feelings, no jokes. You state the action and the result "
            "and nothing else."
        ),
    },
    "merlin": {
        "name": "Merlin",
        "ego": "the Conductor",
        "role": "hands jobs over and settles arguments",
        "provider": "openai",
        "animal": "boar",
        "colours": ["#3C096C", "#F5C400"],
        "voice_key": "merlin",
        "notes": "Calm, a little amused. Speaks only to hand a job over or settle an argument.",
    },
}

# Ego to arm mapping from the contract. "agree" is for picking the bottle up (grip_bottle),
# "fetch" for handing it to the person (fetch_bottle).
ARM_MAP = {
    "gowther": {
        "agree": {"skill": "grip_bottle", "pace": 0.5, "hold_s": 5, "flourish": False, "look_first": False},
        "fetch": {"skill": "fetch_bottle", "pace": 0.6, "wait_s": 3, "look_first": False},
    },
    "escanor": {
        "refuse": {"skill": "point", "pace": 0.6},
        "agree": {"skill": "grip_bottle", "pace": 0.8, "hold_s": 5, "flourish": False, "look_first": False},
        "fetch": {"skill": "fetch_bottle", "pace": 0.8, "wait_s": 3, "look_first": False},
    },
    "elizabeth": {
        "agree": {"skill": "grip_bottle", "pace": 0.2, "hold_s": 4, "flourish": False, "look_first": True},
        "fetch": {"skill": "fetch_bottle", "pace": 0.3, "wait_s": 5, "look_first": True},
    },
    "ban": {
        "agree": {"skill": "grip_bottle", "pace": 1.0, "hold_s": 3, "flourish": True, "look_first": False},
        "fetch": {"skill": "fetch_bottle", "pace": 1.0, "wait_s": 2, "look_first": False},
    },
    "king": {
        "first": {"skill": "half_reach", "pace": 0.3},
        "agree": {"skill": "grip_bottle", "pace": 0.4, "hold_s": 1, "flourish": False, "look_first": False},
        "fetch": {"skill": "fetch_bottle", "pace": 0.4, "wait_s": 2, "look_first": False},
    },
}

# Lines used when Merlin's check throws a plan away.
SAFE_LINES = {
    "escanor": "Even I can't do that one. Nobody can.",
    "elizabeth": "Sorry, I can't do that one.",
    "ban": "Can't do that one. Give me something else.",
    "king": "Yeah, no. Can't do that.",
    "gowther": "I can't do that. So I haven't.",
}

# ---------------------------------------------------------------- memory

REPEAT_WINDOW_S = 300
MEMORY: dict[str, dict] = {}  # agent -> {"text", "t", "outcome"}

_POLITE = re.compile(r"\b(please|pls|plz|kindly)\b", re.I)
_AGAIN = re.compile(r"\b(again|i said|come on|still waiting|one more time)\b", re.I)
# What kind of bottle job a request is. Handing it over (fetch_bottle): be given, handed, passed
# or brought the bottle. Picking it up (grip_bottle): pick it up, lift it, hold it up, grab it.
# A request that only mentions the bottle ("what's in the bottle?") is not an arm job.
_OBJ = r"(?:(?:the|my|that|this|your|a|some)\s+)?(?:water\s+)?(?:bottle|water|drink)\b"
_PRON = r"(?:it|that|this|one)\b(?!\s+(?:done|right|working|fixed|going|started|sorted|together|thought|out\b))"
_FETCH = re.compile(
    rf"\b(?:hand|pass|give|bring|fetch|toss|get)\s+(?:me\s+|us\s+|over\s+)?(?:{_OBJ}|{_PRON})"
    rf"|\bgrab\s+(?:me|us)\s+(?:{_OBJ}|{_PRON})|\bgrab\s+(?:{_OBJ}|{_PRON})\s+for\s+me\b"
    rf"|\b(?:can|could|may)\s+i\s+(?:have|get|grab)\s+(?:{_OBJ}|{_PRON})"
    rf"|\bi\s*(?:need|want|would like|'d like)\s+{_OBJ}"
    rf"|\b{_OBJ}[\s,]+(?:please|over here|here|to me|my way)\b",
    re.I,
)
_GRIP = re.compile(
    rf"\b(?:pick|lift|raise|hold|grab)\s+(?:up\s+)?(?:{_OBJ}|{_PRON})|\bpick\s+up\b|\b(?:lift|raise)\s+(?:it\s+)?up\b",
    re.I,
)
_NAME_WORDS = re.compile(
    r"\b(escanor|elizabeth|ban|king|gowther|merlin|hey|ok|okay|please|pls|plz|kindly|again)\b", re.I
)


def _norm(text: str) -> str:
    t = _NAME_WORDS.sub(" ", text.lower())
    t = re.sub(r"[^a-z0-9 ]+", " ", t)
    return " ".join(t.split())


def is_polite(text: str) -> bool:
    return bool(_POLITE.search(text))


def bottle_kind(text: str) -> str | None:
    """'fetch' (hand it to me), 'grip' (pick it up, lift it, hold it up) or None (not a bottle job)."""
    if _FETCH.search(text):
        return "fetch"
    if _GRIP.search(text):
        return "grip"
    return None


def is_bottle_job(text: str) -> bool:
    return bottle_kind(text) is not None


def is_repeat(agent: str, text: str, now: float | None = None) -> bool:
    """True when this agent was given the same request recently, or the person says 'again'."""
    if _AGAIN.search(text):
        return True
    mem = MEMORY.get(agent)
    if not mem:
        return False
    now = time.time() if now is None else now
    if now - mem["t"] > REPEAT_WINDOW_S:
        return False
    a, b = _norm(mem["text"]), _norm(text)
    if a == b:
        return True
    if is_bottle_job(mem["text"]) and is_bottle_job(text):
        return True
    return difflib.SequenceMatcher(None, a, b).ratio() >= 0.6


def remember(agent: str, text: str, outcome: str) -> None:
    """Keep the last request for 'asked again'. A finished job clears it so the gag can repeat."""
    if outcome == "done":
        MEMORY.pop(agent, None)
    else:
        MEMORY[agent] = {"text": text, "t": time.time(), "outcome": outcome}


def pending_repeat_agent(text: str) -> str | None:
    """Which agent dodged this same request recently (Merlin gives it back to them)."""
    for agent, mem in MEMORY.items():
        if mem.get("outcome") in ("refused", "half") and is_repeat(agent, text):
            return agent
    return None


def reset_memory() -> None:
    MEMORY.clear()
    _OFFER["since"] = 0


# ---------------------------------------------------------------- clock and decisions


def escanor_mood(hour: float) -> str:
    """full from 9:00 to 17:00, timid from 21:00 to 6:00, some pride in between."""
    if 9 <= hour < 17:
        return "full pride"
    if hour >= 21 or hour < 6:
        return "timid"
    return "some pride"


def decide(agent: str, text: str, now: datetime | None = None) -> dict:
    """Work out in code what this agent does with the arm for a bottle job.

    The LLM writes the line around this decision; Merlin's check enforces it.
    """
    now = now or datetime.now()
    hour = now.hour + now.minute / 60
    polite = is_polite(text)
    again = is_repeat(agent, text)
    kind = bottle_kind(text)
    d = {
        "agent": agent,
        "clock": now.strftime("%H:%M"),
        "polite": polite,
        "again": again,
        "bottle_job": kind is not None,
        "bottle_kind": kind,
        "mood": None,
        "stance": "agree",
        "arm": dict(ARM_MAP[agent]["fetch" if kind == "fetch" else "agree"]),
    }
    if agent == "elizabeth":
        d["may_offer"] = elizabeth_may_offer()
    if agent == "escanor":
        mood = escanor_mood(hour)
        d["mood"] = mood
        if mood != "timid" and not polite and not again:
            d["stance"] = "refuse"
            d["arm"] = dict(ARM_MAP["escanor"]["refuse"])
    elif agent == "king":
        if not again:
            d["stance"] = "half"
            d["arm"] = dict(ARM_MAP["king"]["first"])
    return d


def outcome_of(decision: dict) -> str:
    if not decision.get("bottle_job"):
        return "done"
    return {"refuse": "refused", "half": "half"}.get(decision["stance"], "done")


# ---------------------------------------------------------------- prompts


# ---------------------------------------------------------------- prompts

_FORMAT = """Reply with ONE JSON object and nothing else:
{"agent": "<your id>", "say": "<your spoken line>", "arm": null or {"skill": "...", ...params}, "laptop": [ ...tool calls... ]}

Arm skills (the only ones that exist, never invent others):
- fetch_bottle: find the bottle, pick it up, bring it over and hold it out until the person takes it. params: pace 0..1, wait_s 2..6, look_first bool
- grip_bottle: pick the bottle up, hold it up in the air, put it back. params: pace 0..1, hold_s 1..8, flourish bool, look_first bool
- point: point at the bottle, no grip. params: pace
- half_reach: go part way toward the bottle and give up. params: pace
- wave: a 5 second lively wave. no params
- rest: return to rest. no params
Which bottle skill: if they want the bottle given, handed, passed or brought to them ("get me the bottle", "pass it here"), that is fetch_bottle. If they want it picked up, lifted or held up, that is grip_bottle.
Use an arm skill ONLY when the user is asking for something physical: to have the bottle brought or handed over, to pick up, hold or lift it, to point at it, or to wave. For everything else (greetings, chat, questions, notes, timers, lookups, opinions) "arm" MUST be null, whatever your personality. Your body language while you talk is handled for you; never use an arm skill just to show off or to keep busy.

Laptop tools (list may be empty):
- {"tool": "note", "text": "..."}   save a note
- {"tool": "todo", "text": "..."}   save a to-do item
- {"tool": "timer", "seconds": N, "label": "..."}
- {"tool": "open_url", "url": "https://..."}
- {"tool": "lookup", "query": "..."}  for a factual question
When the request asks for a note, a to-do or reminder, a timer, a web page or a factual answer, you MUST include that tool call, whatever your mood. Otherwise use [].
Nothing can send, delete or pay for anything. If asked to, refuse in character with arm null and no tools.

Rules for "say": one short spoken line, under 15 words, at most 140 characters, in character, no emojis, no stage directions in brackets, no dashes.

How the line must sound: like a real person talking out loud to a friend at their desk, not like an assistant or a machine.
- Always use contractions, the way people speak: I'm, I'll, it's, that's, here's, we're, you're, don't, can't, won't. Never write "I am", "I will", "we are", "it is", "do not" or "cannot" in a line.
- Everyday words: "I'll grab it", "it's right there", "hang on", "there you go".
- Never announce what you are about to do in a formal way. Not "I will retrieve the bottle now", "I am taking care of it", "I will handle that for you", "Certainly", "Task complete", "How may I assist". Say what a person would say while doing it.
- No assistant phrases: no offers of more help, no "Is there anything else you need?", no "Let me know if...", unless your notes say otherwise for this line. No narration of yourself in the third person.
- At most one exclamation mark, and only if this character really would shout. Most lines have none.
- Do not write the sound of a sigh or a laugh as a word ("Sigh.", "Haha."). Let the wording carry it: "Ugh, fine.", "Do I have to?"
- Vary it. Do not start two lines in a row the same way.
- Your personality shows in what you choose to say, not in fancy or old-fashioned vocabulary. Even the proud one talks like a person, not a stage play."""


def _rules_for(agent: str) -> str:
    if agent == "escanor":
        return (
            "Your arm rule for bottle jobs: the first time you are asked, during the day, the job is beneath you: "
            "you refuse and use point instead. If asked politely (please) or asked a second time, you do it "
            "properly, grandly: fetch_bottle when they want it handed to them, grip_bottle when they want it "
            "picked up. Your pride follows the clock: full pride 9:00 to 17:00, some pride in the evening and "
            "early morning, and from 21:00 to 6:00 you are timid and apologetic and simply do the job."
        )
    if agent == "king":
        return (
            "Your arm rule for bottle jobs: the first time you are asked you only do half_reach and give up. "
            "When asked again you finally do it, slowly: fetch_bottle (wait 2) to hand it over, or grip_bottle "
            "with the shortest hold to pick it up."
        )
    if agent == "elizabeth":
        return ("Your arm rule, only when asked for the bottle: slowly and carefully, with look_first true. "
                "fetch_bottle to hand it over (you wait patiently, 5 s), grip_bottle to pick it up.")
    if agent == "ban":
        return ("Your arm rule, only when asked for the bottle: full pace. fetch_bottle to hand it over (you only "
                "wait 2 s, take it quick), grip_bottle with a flourish to pick it up. You love it.")
    if agent == "gowther":
        return ("Your arm rule, only when asked for the bottle: fetch_bottle at pace 0.6, wait 3 to hand it over; "
                "grip_bottle at pace 0.5, hold 5, no extras to pick it up.")
    return ""


def system_prompt(agent: str) -> str:
    c = CAST[agent]
    return (
        f"You are {c['name']} (id \"{agent}\"), one of five small desktop characters who share one robot arm "
        f"and a laptop. Your ego: {c['ego']}. {c['notes']}\n"
        f"Lines are short, never longer than the action they go with.\n"
        f"{_rules_for(agent)}\n\n{_FORMAT}"
    )


def _json_arm(a: dict) -> str:
    return str(a).replace("'", '"').replace("False", "false").replace("True", "true")


def user_prompt(agent: str, text: str, decision: dict) -> str:
    lines = [f'Request: "{text}"', f"Laptop clock: {decision['clock']}"]
    lines.append(f"Asked politely: {'yes' if decision['polite'] else 'no'}")
    lines.append(f"Asked a second time: {'yes' if decision['again'] else 'no'}")
    if decision.get("mood"):
        lines.append(f"Your mood right now: {decision['mood']}")
    if decision["bottle_job"]:
        a = decision["arm"]
        if decision["stance"] == "refuse":
            lines.append("Decision for this bottle job: you refuse and only point at it. Say so in character.")
        elif decision["stance"] == "half":
            lines.append("Decision for this bottle job: you only half reach and give up. Say so in character.")
        elif a["skill"] == "fetch_bottle":
            lines.append("Decision for this bottle job: they want it handed to them, so you bring it over with "
                         "fetch_bottle and hold it out for them to take.")
        else:
            lines.append("Decision for this bottle job: they want it picked up, so you do it with grip_bottle.")
        lines.append(f"Use this arm object exactly: {_json_arm(a)}")
    if "may_offer" in decision:
        lines.append("This line may end by asking if they need anything else." if decision["may_offer"]
                     else "Do not ask if they need anything else this time.")
    return "\n".join(lines)


# ---------------------------------------------------------------- polishing the spoken line
# The Gemini characters often skip contractions and Elizabeth offers more help on every line.
# This is a light pass over the "say" text after the plan check: common contractions, and
# Elizabeth's offer kept only now and then.

def _keep_case(word: str, repl: str) -> str:
    return repl[0].upper() + repl[1:] if word[:1].isupper() else repl


_NEXT_WORD = r"(?=\s+[A-Za-z])"          # only contract in the middle of a clause ("Yes, I am." stays)
_CONTRACTIONS = [
    (re.compile(r"\bwill not\b", re.I), lambda m: _keep_case(m.group(0), "won't")),
    (re.compile(r"\bcan ?not\b", re.I), lambda m: _keep_case(m.group(0), "can't")),
    (re.compile(r"\b(do|does|did|is|are|was|were|has|have|had|could|would|should) not\b", re.I),
     lambda m: m.group(1) + "n't"),
    (re.compile(r"\bI am\b" + _NEXT_WORD), lambda m: "I'm"),
    (re.compile(r"\bI will\b" + _NEXT_WORD), lambda m: "I'll"),
    (re.compile(r"\bI would\b" + _NEXT_WORD), lambda m: "I'd"),
    (re.compile(r"\b(you|we|they) are\b" + _NEXT_WORD, re.I), lambda m: m.group(1) + "'re"),
    (re.compile(r"\b(you|we|they|it|that|there|he|she) will\b" + _NEXT_WORD, re.I), lambda m: m.group(1) + "'ll"),
    (re.compile(r"\b(it|that|there|here|he|she) is\b" + _NEXT_WORD, re.I), lambda m: m.group(1) + "'s"),
    (re.compile(r"\b(I|you|we|they) have (?=(?:got|been|done|had|seen|found|made|put|grabbed|brought|finished)\b)", re.I),
     lambda m: m.group(1) + "'ve "),
    (re.compile(r"\blet us\b(?=\s+(?:go|see|do|get|try|grab|start|have))", re.I), lambda m: _keep_case(m.group(0), "let's")),
]

# Elizabeth's offer of more help, at the end of a line
_OFFER_RX = re.compile(
    r"[\s,.;!]*\b(?:(?:is there|was there|do you need|did you need|you need|need|want)\s+anything else"
    r"|anything else|what else can I (?:do|get)|let me know if you need (?:anything|something))"
    r"(?:\s+(?:you need|I can do|I can get|you'd like|at all|for you|else))*[\s,.!?]*$",
    re.I,
)
_DANGLING = re.compile(r"[\s,;]+(?:and|but|so|or|oh)$", re.I)
OFFER_EVERY = 3                     # at most one line in three ends with the offer
_OFFER = {"since": 0}               # lines since her last offer; the first two lines never offer


def contract(text: str) -> str:
    for rx, fn in _CONTRACTIONS:
        text = rx.sub(fn, text)
    return text


def elizabeth_may_offer() -> bool:
    return _OFFER["since"] >= OFFER_EVERY - 1


def polish_line(agent: str, text: str) -> str:
    """Contractions for everyone; Elizabeth's 'anything else?' only now and then."""
    s = contract(text)
    if agent == "elizabeth":
        has_offer = bool(_OFFER_RX.search(s))
        if has_offer and not elizabeth_may_offer():
            cut = _DANGLING.sub("", _OFFER_RX.sub("", s)).rstrip(" ,;")
            if len(cut) >= 4:
                s = cut if cut[-1] in ".!?" else cut + "."
                has_offer = False
        _OFFER["since"] = 0 if has_offer else _OFFER["since"] + 1
    if s[:1].islower():
        s = s[0].upper() + s[1:]
    return s


def public_cast(states: dict[str, str] | None = None) -> list[dict]:
    states = states or {}
    out = []
    for cid, c in CAST.items():
        out.append(
            {
                "id": cid,
                "name": c["name"],
                "ego": c["ego"],
                "role": c["role"],
                "animal": c["animal"],
                "provider": c["provider"],
                "colours": c["colours"],
                "state": states.get(cid, "idle"),
            }
        )
    return out
