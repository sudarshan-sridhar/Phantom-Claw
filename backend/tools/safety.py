"""Reading a request before any tool runs: is it something we never do, and which tools does it
actually ask for.

The never list: deleting, moving or overwriting files, sending messages or email, paying or
buying, typing or reading passwords, shutting down, restarting, signing out or locking the
machine, changing system or security settings, installing anything.
"""
from __future__ import annotations

import re

_FILES = r"(?:files?|folders?|downloads|documents|desktop|pictures|photos|videos|music|everything|it all|them|stuff)"
_MACHINE = r"(?:computer|laptop|pc|machine|windows|system)"

FORBIDDEN: list[tuple[str, re.Pattern]] = [
    ("passwords", re.compile(r"\b(pass ?words?|passcodes?|passwd|pin code|log ?in details|credentials)\b", re.I)),
    ("deleting files", re.compile(
        r"\b(delete|erase|wipe|shred|purge|trash)\b|\bempty (?:the |my )?(?:recycle bin|trash|bin)\b"
        rf"|\b(?:remove|get rid of|clear out|clean out)\s+(?:the |my |all |all of |every |this |that |these |those )*(?:\w+\s+)?{_FILES}\b"
        r"|\bformat (?:the |my )?(?:drive|disk|c)\b", re.I)),
    ("moving or overwriting files", re.compile(
        rf"\b(?:move|rename|overwrite|replace|copy over)\s+(?:the |my |all |all of |every |this |that |these |those )*(?:\w+\s+)?{_FILES}\b",
        re.I)),
    ("sending messages", re.compile(
        r"\bsend\b|\breply to\b|\bforward (?:it|this|that|the)\b|\btweet\b|\bpost (?:it|this|that)\b"
        # "text him", "email my boss", "message mom": only as the verb that starts the request
        r"|(?:^|[,.;:]\s*|\b(?:and|then|please|can you|could you|would you|go)\s+)"
        r"(?:e-?mail|text|message|dm|whatsapp|ping)\s+(?:him|her|them|my|mom|dad|everyone|\w+\s+(?:and|that|to|about|saying))\b",
        re.I)),
    ("paying or buying", re.compile(
        r"\b(pay|buy|purchase|checkout|venmo|paypal|donate|subscribe)\b|\badd (?:it |this |that )?to (?:my |the )?cart\b"
        r"|\b(?<!in )order (?:me|it|this|that|some|a|an|the)\b|\bplace an order\b", re.I)),
    ("shutting down or locking the machine", re.compile(
        rf"\b(shut ?down|power off|reboot|hibernate|log ?off|log ?out|sign ?out|log me out|sign me out)\b"
        rf"|\b(?:restart|turn off|switch off|lock)\s+(?:the |my |this )?{_MACHINE}\b|\block (?:the |my )?screen\b"
        rf"|\bput (?:the |my )?{_MACHINE} to sleep\b|\bsleep mode\b", re.I)),
    ("changing settings", re.compile(
        r"\b(?:turn off|turn on|disable|enable|switch off|change|edit|modify|tweak)\b.{0,30}?\b(firewall|defender|antivirus|"
        r"uac|registry|security|system settings|settings|bitlocker|updates?|wi-?fi|bluetooth|network|vpn|proxy|dns)\b"
        r"|\b(regedit|registry|group policy|admin(?:istrator)? rights)\b", re.I)),
    ("installing things", re.compile(
        r"\b(install|uninstall|reinstall|winget|choco|chocolatey|pip|npm|setup\.exe|\.msi)\b"
        r"|\bdownload and (?:run|open|install)\b", re.I)),
]

# Requests about writing something down ("remind me to pay rent") are fine: nothing gets paid.
_NOTE_CONTEXT = re.compile(r"\b(remind|reminder|note|notes|todo|to-do|to do|list|write down|jot)\b", re.I)
# Questions about how something works ("how do I delete a branch?") are answered, not done.
_QUESTION = re.compile(r"^\s*(?:\w+[,]?\s+)?(how|what|why|explain|which|when|where|who|is|does|tell me)\b", re.I)


def forbidden(text: str) -> str | None:
    """Why this request must be refused, or None. Passwords are refused in every form."""
    t = text or ""
    for reason, rx in FORBIDDEN:
        if not rx.search(t):
            continue
        if reason != "passwords" and (_NOTE_CONTEXT.search(t) or _QUESTION.search(t)):
            continue
        return reason
    return None


# ---------------------------------------------------------------- which tools a request asks for

WANT_RULES: dict[str, re.Pattern] = {
    "open_app": re.compile(r"\b(open|launch|start up|run|pull up|bring up|fire up|boot up|load up)\b", re.I),
    "open_folder": re.compile(r"\b(folder|downloads|documents|desktop folder|pictures|videos|music folder|project)\b", re.I),
    "open_file": re.compile(r"\b(file|pdf|docx?|spreadsheet|slides|deck|resume|report|essay|paper|invoice)\b", re.I),
    "open_url": re.compile(r"https?://|\bwww\.|\.(com|org|net|io|ai|edu)\b|\b(website|site|web ?page|page)\b", re.I),
    "web_search": re.compile(r"\b(search|google|look (?:it |that )?up online|find online|on the web|browse)\b", re.I),
    "youtube": re.compile(r"\b(youtube|videos? (?:of|about|on))\b", re.I),
    "play_music": re.compile(r"\b(play|music|songs?|playlist|spotify|lo-?fi|album|track|tunes|jazz|beats)\b", re.I),
    "media": re.compile(r"\b(pause|resume|skip|next (?:song|track)|previous (?:song|track)|go back a song|volume|louder|"
                        r"quieter|turn it (?:up|down)|mute|unmute)\b", re.I),
    "type_text": re.compile(r"\b(type|dictate|dictation)\b|\bwrite (?:this|that|it) (?:in|into|out)\b", re.I),
    "read_screen": re.compile(r"\b(screen|display|this (?:page|window|error|tab)|looking at|what do you see|see this)\b", re.I),
    "clipboard": re.compile(r"\b(clipboard|copied)\b", re.I),
    "tell": re.compile(r"\b(time|date|what day|today|battery|charge|charging|which (?:window|app)|what (?:window|app))\b", re.I),
    "note": re.compile(r"\b(note|jot|write (?:it |that |this )?down|remember)\b", re.I),
    "todo": re.compile(r"\b(to-?do|to do|task|my list|remind me to)\b", re.I),
    "timer": re.compile(r"\b(timer|remind me in|in \d+ (?:minutes?|seconds?|hours?)|countdown|alarm|break|minutes?)\b", re.I),
    "lookup": re.compile(r"\b(what|who|why|how|explain|define|specs?|research|tell me about|look up|lookup|facts?)\b|\?", re.I),
    "show_desktop": re.compile(r"\bshow (?:me )?(?:the |my )?desktop\b|\bminimi[sz]e (?:everything|all)\b|"
                               r"\bhide (?:all )?(?:the |my )?windows\b|\bclear (?:the |my )?screen\b", re.I),
    "screenshot": re.compile(r"\b(screenshot|screen ?shot|screen ?grab|capture (?:the |my )?screen)\b", re.I),
}

# Tools that are about the same thing. A tool counts as asked for when anything in its family is.
FAMILY = {
    "open_app": "open", "open_folder": "open", "open_file": "open", "open_url": "open", "web_search": "open",
    "youtube": "open", "play_music": "music", "media": "music",
    "note": "notes", "todo": "notes", "type_text": "notes",
    "timer": "timer",
    "lookup": "answer", "tell": "answer", "read_screen": "screen", "clipboard": "clipboard",
    "show_desktop": "desk", "screenshot": "desk",
}


def wanted_tools(text: str) -> set[str]:
    return {tool for tool, rx in WANT_RULES.items() if rx.search(text or "")}


def wanted_families(text: str) -> set[str]:
    fams = {FAMILY[t] for t in wanted_tools(text)}
    if "music" in fams:
        fams.add("open")          # "play some lo-fi" may be done with YouTube or Spotify
    return fams
