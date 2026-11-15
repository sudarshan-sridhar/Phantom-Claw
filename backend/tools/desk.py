"""Small Windows things done with ctypes: media keys, typing text, Win+D, the clipboard, the
battery, the window in front, the clock and a saved screenshot.

None of it changes a setting, sends anything, or touches a password field on purpose.
"""
from __future__ import annotations

import asyncio
import ctypes
import re
import time
from ctypes import wintypes
from datetime import datetime
from pathlib import Path

from .common import fail, is_dry, ok, would

KEYUP = 0x0002
EXTENDED = 0x0001
UNICODE = 0x0004

MEDIA_KEYS = {
    "play_pause": 0xB3,
    "next": 0xB0,
    "previous": 0xB1,
    "volume_up": 0xAF,
    "volume_down": 0xAE,
    "mute": 0xAD,
}
VOLUME_STEPS = 5          # each press is about 2 percent, so one request moves it about 10
TYPE_MAX = 600


def _user32():
    return ctypes.windll.user32


def _tap(vk: int, times: int = 1) -> None:
    u = _user32()
    for _ in range(times):
        u.keybd_event(vk, 0, EXTENDED, 0)
        u.keybd_event(vk, 0, EXTENDED | KEYUP, 0)
        time.sleep(0.02)


async def media(agent: str, action: str) -> dict:
    """Press a media key: play_pause, next, previous, volume_up, volume_down or mute."""
    vk = MEDIA_KEYS.get(action)
    if vk is None:
        return fail("media", f"unknown media action {action!r}")
    times = VOLUME_STEPS if action.startswith("volume") else 1
    if is_dry():
        return would("media", f"press the {action} key" + (f" {times} times" if times > 1 else ""), action=action)
    await asyncio.to_thread(_tap, vk, times)
    return ok("media", action=action)


async def show_desktop(agent: str) -> dict:
    """Minimise everything (Win+D)."""
    if is_dry():
        return would("show_desktop", "press Win+D")

    def _go():
        u = _user32()
        u.keybd_event(0x5B, 0, 0, 0)       # left Windows key down
        u.keybd_event(0x44, 0, 0, 0)       # D
        u.keybd_event(0x44, 0, KEYUP, 0)
        u.keybd_event(0x5B, 0, KEYUP, 0)

    await asyncio.to_thread(_go)
    return ok("show_desktop")


# ---------------------------------------------------------------- typing (dictation)

ULONG_PTR = ctypes.c_size_t


class _KEYBDINPUT(ctypes.Structure):
    _fields_ = [("wVk", wintypes.WORD), ("wScan", wintypes.WORD), ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD), ("dwExtraInfo", ULONG_PTR)]


class _MOUSEINPUT(ctypes.Structure):
    _fields_ = [("dx", wintypes.LONG), ("dy", wintypes.LONG), ("mouseData", wintypes.DWORD),
                ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD), ("dwExtraInfo", ULONG_PTR)]


class _INPUTUNION(ctypes.Union):
    _fields_ = [("ki", _KEYBDINPUT), ("mi", _MOUSEINPUT)]


class _INPUT(ctypes.Structure):
    _fields_ = [("type", wintypes.DWORD), ("u", _INPUTUNION)]


_PASSWORD = re.compile(r"\b(pass ?words?|passcodes?|passwd|pin code)\b", re.I)


def clean_typing(text: str) -> str:
    """Plain text only. Line breaks and tabs become spaces so nothing gets submitted or sent,
    other control characters are dropped, and it stops at 600 characters."""
    s = re.sub(r"[\r\n\t]+", " ", str(text or ""))
    s = "".join(ch for ch in s if ch.isprintable())
    return s.strip()[:TYPE_MAX]


def _send_text(text: str) -> None:
    data = text.encode("utf-16-le")
    units = [int.from_bytes(data[i:i + 2], "little") for i in range(0, len(data), 2)]
    for unit in units:
        down = _INPUT(type=1, u=_INPUTUNION(ki=_KEYBDINPUT(0, unit, UNICODE, 0, 0)))
        up = _INPUT(type=1, u=_INPUTUNION(ki=_KEYBDINPUT(0, unit, UNICODE | KEYUP, 0, 0)))
        arr = (_INPUT * 2)(down, up)
        _user32().SendInput(2, arr, ctypes.sizeof(_INPUT))
        time.sleep(0.004)


async def type_text(agent: str, text: str) -> dict:
    """Type the text into whichever window is in front, like dictation. Text only: no Enter,
    no shortcuts, at most 600 characters, and never anything that looks like a password."""
    s = clean_typing(text)
    if not s:
        return fail("type_text", "nothing to type")
    if _PASSWORD.search(s):
        return fail("type_text", "won't type passwords", refused=True)
    if is_dry():
        return would("type_text", f"type {len(s)} characters into the window in front", chars=len(s))
    await asyncio.to_thread(_send_text, s)
    return ok("type_text", chars=len(s))


# ---------------------------------------------------------------- reading the machine


def clipboard_text() -> str:
    """The text on the clipboard, or an empty string."""
    u, k = _user32(), ctypes.windll.kernel32
    u.GetClipboardData.restype = ctypes.c_void_p
    k.GlobalLock.argtypes = [ctypes.c_void_p]
    k.GlobalLock.restype = ctypes.c_void_p
    k.GlobalUnlock.argtypes = [ctypes.c_void_p]
    for _ in range(5):                    # another program may hold it for a moment
        if u.OpenClipboard(None):
            break
        time.sleep(0.05)
    else:
        return ""
    try:
        h = u.GetClipboardData(13)        # CF_UNICODETEXT
        if not h:
            return ""
        p = k.GlobalLock(h)
        if not p:
            return ""
        try:
            return ctypes.wstring_at(p)
        finally:
            k.GlobalUnlock(h)
    finally:
        u.CloseClipboard()


class _POWER(ctypes.Structure):
    _fields_ = [("ac", ctypes.c_ubyte), ("flag", ctypes.c_ubyte), ("percent", ctypes.c_ubyte),
                ("saver", ctypes.c_ubyte), ("life", wintypes.DWORD), ("full", wintypes.DWORD)]


def battery() -> dict:
    st = _POWER()
    if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(st)):
        return {"known": False}
    if st.flag == 128 or st.percent == 255:
        return {"known": False, "plugged": st.ac == 1}
    return {"known": True, "percent": int(st.percent), "plugged": st.ac == 1, "charging": bool(st.flag & 8)}


def front_window() -> str:
    u = _user32()
    hwnd = u.GetForegroundWindow()
    n = u.GetWindowTextLengthW(hwnd)
    buf = ctypes.create_unicode_buffer(n + 1)
    u.GetWindowTextW(hwnd, buf, n + 1)
    return buf.value.strip()


TELL_WORDS = {
    "time": "time", "clock": "time", "hour": "time",
    "date": "date", "day": "date", "today": "date",
    "battery": "battery", "power": "battery", "charge": "battery",
    "window": "window", "app": "window", "front": "window", "focus": "window",
}


def tell_kind(what: str) -> str | None:
    for w in re.findall(r"[a-z]+", str(what or "").lower()):
        if w in TELL_WORDS:
            return TELL_WORDS[w]
    return None


def _fact(kind: str, now: datetime | None = None) -> str:
    now = now or datetime.now()
    if kind == "time":
        h = now.hour % 12 or 12
        return f"{h}:{now.minute:02d} {'AM' if now.hour < 12 else 'PM'}"
    if kind == "date":
        return f"{now.strftime('%A')}, {now.strftime('%B')} {now.day}"
    if kind == "battery":
        b = battery()
        if not b.get("known"):
            return "plugged in, no battery reading" if b.get("plugged") else "unknown"
        state = "charging" if b["charging"] else ("plugged in" if b["plugged"] else "on battery")
        return f"{b['percent']} percent, {state}"
    if kind == "window":
        t = front_window()
        return t[:80] if t else "nothing in particular"
    return ""


async def tell(agent: str, what: str) -> dict:
    """Say the time, the date, the battery level, or which window is in front. Read only."""
    kind = tell_kind(what)
    if kind is None:
        return fail("tell", f"can't tell {what!r}")
    value = await asyncio.to_thread(_fact, kind)
    return ok("tell", what=kind, value=value)


# ---------------------------------------------------------------- screenshots


def grab_screen():
    from PIL import ImageGrab
    return ImageGrab.grab()


def _new_screenshot_path(folder: Path) -> Path:
    stamp = datetime.now().strftime("%Y-%m-%d %H%M%S")
    p = folder / f"Phantom Claw {stamp}.png"
    n = 2
    while p.exists():                     # never overwrite an existing file
        p = folder / f"Phantom Claw {stamp} ({n}).png"
        n += 1
    return p


async def screenshot(agent: str) -> dict:
    """Save a screenshot as a new file in Pictures\\Screenshots and say where."""
    from .apps import folder_path

    folder = folder_path("screenshots")
    if is_dry():
        return would("screenshot", f"save a screenshot in {folder}", folder=str(folder))

    def _go() -> Path:
        folder.mkdir(parents=True, exist_ok=True)
        p = _new_screenshot_path(folder)
        grab_screen().save(p, "PNG")
        return p

    p = await asyncio.to_thread(_go)
    return ok("screenshot", path=str(p), where="Pictures, Screenshots")
