"""Opening things: programs, folders, files, web searches, YouTube and Spotify.

Everything here only opens. Nothing is deleted, moved, renamed or written.
Programs are found by their Start menu shortcut (fuzzy match on the spoken name) or through a
small table of common names. If nothing matches we say so; we never make up an exe path.
"""
from __future__ import annotations

import asyncio
import ctypes
import difflib
import os
import re
import shutil
import time
import uuid
from pathlib import Path
from urllib.parse import quote, quote_plus

from backend.agents.config import PROJECT_DIR

from .basic import open_address
from .common import fail, is_dry, ok, would

# ---------------------------------------------------------------- the Start menu

SHORTCUT_EXT = {".lnk", ".url", ".appref-ms"}
# Start menu entries we never launch, even if the name matches.
_SKIP_WORDS = re.compile(
    r"\b(uninstall\w*|install\w*|setup|repair|remove|reset|updater|update|recovery|readme|license|"
    r"release notes|documentation|safe mode)\b",
    re.I,
)
_FILLER = {"the", "my", "app", "application", "program", "please", "open", "up", "launch", "start", "for", "me",
           "a", "an", "can", "you"}

# Common spoken names. Each one lists where to look, in order:
#   ("menu", name)  a Start menu shortcut with that name
#   ("exe", name)   a program Windows already knows by that name (on PATH or in App Paths)
#   ("uri", addr)   a registered address like ms-settings:
#   ("url", addr)   a web page, for things that only live in the browser
_CHROME = [("menu", "Google Chrome"), ("exe", "chrome.exe")]
_CODE = [("menu", "Visual Studio Code"), ("exe", "code.exe")]
_EXPLORER = [("exe", "explorer.exe")]
_TERMINAL = [("exe", "wt.exe"), ("menu", "Windows Terminal"), ("menu", "Terminal")]
COMMON: dict[str, list[tuple[str, str]]] = {
    "chrome": _CHROME, "google chrome": _CHROME,
    "edge": [("menu", "Microsoft Edge"), ("exe", "msedge.exe"), ("uri", "microsoft-edge:")],
    "microsoft edge": [("menu", "Microsoft Edge"), ("exe", "msedge.exe")],
    "notepad": [("exe", "notepad.exe")],
    "calculator": [("exe", "calc.exe"), ("uri", "ms-calculator:")], "calc": [("exe", "calc.exe")],
    "file explorer": _EXPLORER, "explorer": _EXPLORER, "files": _EXPLORER, "windows explorer": _EXPLORER,
    "vs code": _CODE, "vscode": _CODE, "visual studio code": _CODE, "code": _CODE,
    "spotify": [("menu", "Spotify"), ("uri", "spotify:"), ("url", "https://open.spotify.com")],
    "discord": [("menu", "Discord"), ("url", "https://discord.com/app")],
    "terminal": _TERMINAL, "windows terminal": _TERMINAL,
    "command prompt": [("menu", "Command Prompt"), ("exe", "cmd.exe")], "cmd": [("exe", "cmd.exe")],
    "powershell": [("menu", "Windows PowerShell"), ("exe", "powershell.exe")],
    "settings": [("uri", "ms-settings:")], "windows settings": [("uri", "ms-settings:")],
    "word": [("menu", "Word"), ("exe", "winword.exe")], "microsoft word": [("menu", "Word"), ("exe", "winword.exe")],
    "excel": [("menu", "Excel"), ("exe", "excel.exe")], "microsoft excel": [("menu", "Excel"), ("exe", "excel.exe")],
    "powerpoint": [("menu", "PowerPoint"), ("exe", "powerpnt.exe")],
    "power point": [("menu", "PowerPoint"), ("exe", "powerpnt.exe")],
    "paint": [("exe", "mspaint.exe"), ("menu", "Paint")], "ms paint": [("exe", "mspaint.exe")],
    "camera": [("uri", "microsoft.windows.camera:")],
    "snipping tool": [("exe", "SnippingTool.exe"), ("uri", "ms-screenclip:")],
    "snip": [("exe", "SnippingTool.exe"), ("uri", "ms-screenclip:")],
    "youtube": [("url", "https://www.youtube.com")],
    "gmail": [("url", "https://mail.google.com")],
    "google": [("url", "https://www.google.com")],
    "github": [("url", "https://github.com")],
}


def start_menu_dirs() -> list[Path]:
    out = []
    for var in ("APPDATA", "PROGRAMDATA"):
        base = os.environ.get(var)
        if base:
            out.append(Path(base) / "Microsoft" / "Windows" / "Start Menu" / "Programs")
    return out


_CACHE: dict = {"key": None, "t": 0.0, "items": []}


def _norm(name: str) -> str:
    s = name.lower()
    s = re.sub(r"\((?:x64|x86|64[ -]?bit|32[ -]?bit|user|preview)\)", " ", s)
    s = re.sub(r"[^a-z0-9+#]+", " ", s)
    return " ".join(s.split())


def spoken_name(name: str) -> str:
    words = [w for w in _norm(name).split() if w not in _FILLER]
    return " ".join(words)


def list_shortcuts(dirs: list[Path] | None = None) -> list[tuple[str, Path]]:
    """Every Start menu shortcut as (normalised name, path), skipping uninstallers and the like.
    Cached for a minute because the folders rarely change."""
    dirs = start_menu_dirs() if dirs is None else dirs
    key = tuple(str(d) for d in dirs)
    if _CACHE["key"] == key and time.monotonic() - _CACHE["t"] < 60:
        return _CACHE["items"]
    items: list[tuple[str, Path]] = []
    for d in dirs:
        if not d.is_dir():
            continue
        for root, _subdirs, files in os.walk(d):
            for f in files:
                p = Path(root) / f
                if p.suffix.lower() not in SHORTCUT_EXT or _SKIP_WORDS.search(p.stem):
                    continue
                items.append((_norm(p.stem), p))
    _CACHE.update(key=key, t=time.monotonic(), items=items)
    return items


def clear_cache() -> None:
    _CACHE.update(key=None, t=0.0, items=[])


def _score(spoken: str, name: str) -> float:
    if not spoken or not name:
        return 0.0
    if spoken == name:
        return 1.0
    sw, nw = spoken.split(), name.split()
    if nw[: len(sw)] == sw:
        return 0.92                       # "visual studio" -> "visual studio code"
    if f" {spoken} " in f" {name} ":
        return 0.88                       # "chrome" inside "google chrome"
    if f" {name} " in f" {spoken} ":
        return 0.85                       # "google chrome browser" -> "google chrome"
    return difflib.SequenceMatcher(None, spoken, name).ratio()


def find_shortcut(spoken: str, dirs: list[Path] | None = None, cutoff: float = 0.78) -> Path | None:
    best: tuple[float, int, Path] | None = None
    for name, path in list_shortcuts(dirs):
        s = _score(spoken, name)
        if s >= cutoff and (best is None or (s, -len(name)) > (best[0], best[1])):
            best = (s, -len(name), path)
    return best[2] if best else None


def _app_paths(exe: str) -> str | None:
    """Look a program up in the registry's App Paths, where installers say where they put it."""
    try:
        import winreg
    except ImportError:
        return None
    for hive in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
        try:
            with winreg.OpenKey(hive, rf"Software\Microsoft\Windows\CurrentVersion\App Paths\{exe}") as k:
                val, _ = winreg.QueryValueEx(k, "")
                val = str(val).strip('"')
                if val and Path(val).exists():
                    return val
        except OSError:
            continue
    return None


def find_exe(exe: str) -> str | None:
    return shutil.which(exe) or _app_paths(exe)


def uri_registered(addr: str) -> bool:
    scheme = addr.split(":", 1)[0]
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_CLASSES_ROOT, scheme):
            return True
    except (ImportError, OSError):
        return False


def resolve_app(name: str, dirs: list[Path] | None = None) -> dict | None:
    """Work out what to launch for a spoken program name.

    Returns {"target", "how", "label"} or None. how is "shortcut", "exe", "uri" or "url".
    """
    spoken = spoken_name(name)
    if not spoken:
        return None
    for how, value in COMMON.get(spoken, []):
        if how == "menu":
            p = find_shortcut(_norm(value), dirs, cutoff=0.9)
            if p:
                return {"target": str(p), "how": "shortcut", "label": p.stem}
        elif how == "exe":
            found = find_exe(value)
            if found:
                return {"target": found, "how": "exe", "label": value}
        elif how == "uri":
            if uri_registered(value):
                return {"target": value, "how": "uri", "label": value}
        elif how == "url":
            return {"target": value, "how": "url", "label": value}
    p = find_shortcut(spoken, dirs)
    if p:
        return {"target": str(p), "how": "shortcut", "label": p.stem}
    if dirs is None:
        app = find_start_app(spoken)
        if app:
            return {"target": "shell:AppsFolder\\" + app[1], "how": "app", "label": app[0]}
    return None


_START_APPS: list[tuple[str, str]] | None = None


def list_start_apps() -> list[tuple[str, str]]:
    """Every app Windows lists in Start, as (name, app id). This is how Store apps such as
    WhatsApp are found: they have no shortcut file and no exe on the path."""
    global _START_APPS
    if _START_APPS is None:
        import json
        import subprocess

        apps: list[tuple[str, str]] = []
        try:
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command", "Get-StartApps | ConvertTo-Json -Compress"],
                capture_output=True, text=True, timeout=20, creationflags=0x08000000).stdout
            rows = json.loads(out or "[]")
            if isinstance(rows, dict):
                rows = [rows]
            apps = [(str(r.get("Name") or ""), str(r.get("AppID") or "")) for r in rows if r.get("AppID")]
        except Exception:  # noqa: BLE001 - no list just means no match
            apps = []
        _START_APPS = apps
    return _START_APPS


def find_start_app(spoken: str, cutoff: float = 0.78) -> tuple[str, str] | None:
    best, best_score = None, cutoff
    for name, app_id in list_start_apps():
        if _SKIP_WORDS.search(name):
            continue
        s = _score(spoken, _norm(name))
        if s > best_score:
            best, best_score = (name, app_id), s
    return best


def _windows() -> dict[int, str]:
    """Every visible top-level window with a title: {handle: title}."""
    import ctypes
    from ctypes import wintypes

    user = ctypes.windll.user32
    found: dict[int, str] = {}

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def each(hwnd, _):
        if user.IsWindowVisible(hwnd):
            n = user.GetWindowTextLengthW(hwnd)
            if n:
                buf = ctypes.create_unicode_buffer(n + 1)
                user.GetWindowTextW(hwnd, buf, n + 1)
                found[int(hwnd)] = buf.value
        return True

    user.EnumWindows(each, 0)
    return found


def _to_front(hwnd: int) -> None:
    """Put a window on top and give it the keyboard. Windows only lets a background program do
    this right after a key press, so one tap of Alt is sent first."""
    import ctypes

    user = ctypes.windll.user32
    if user.IsIconic(hwnd):
        user.ShowWindow(hwnd, 9)        # restore from the taskbar
    user.keybd_event(0x12, 0, 0, 0)     # Alt down
    user.keybd_event(0x12, 0, 2, 0)     # Alt up
    user.SetForegroundWindow(hwnd)
    user.BringWindowToTop(hwnd)


def show_window(before: dict[int, str], label: str, wait_s: float = 8.0) -> bool:
    """After opening something, bring its window to the front: the new window if one appeared,
    else an open window whose title carries the name (the program was already running)."""
    import time

    words = [w for w in re.split(r"[^a-z0-9]+", label.lower().replace(".exe", "")) if len(w) > 2]
    end = time.monotonic() + wait_s
    while time.monotonic() < end:
        now = _windows()
        fresh = [h for h in now if h not in before]
        named = [h for h, title in now.items() if words and any(w in title.lower() for w in words)]
        pick = next((h for h in fresh if h in named), None) or (fresh[0] if fresh else None)
        if pick is None and named and time.monotonic() > end - wait_s + 1.5:
            pick = named[0]             # nothing new after a moment: it was already open
        if pick is not None:
            _to_front(pick)
            return True
        time.sleep(0.25)
    return False


def _launch(tool: str, target: str, label: str, **extra) -> dict:
    if is_dry():
        return would(tool, f"open {target}", label=label, **extra)
    before = _windows()
    os.startfile(target)  # noqa: S606 - opening is the whole point; the target came from the Start menu or a fixed table
    shown = show_window(before, label)
    return ok(tool, label=label, shown=shown, **extra)


async def open_app(agent: str, name: str, dirs: list[Path] | None = None) -> dict:
    """Open an installed program by its spoken name ("chrome", "vs code", "spotify")."""
    found = await asyncio.to_thread(resolve_app, name, dirs)
    if not found:
        return fail("open_app", f"no program called {name!r}", not_found=name)
    if found["how"] == "url":
        r = await asyncio.to_thread(open_address, "open_app", found["target"])
        return {**r, "label": name}
    return await asyncio.to_thread(_launch, "open_app", found["target"], found["label"], how=found["how"])


# ---------------------------------------------------------------- folders

_KNOWN = {
    "downloads": "374DE290-123F-4565-9164-39C4925E467B",
    "documents": "FDD39AD0-238F-46AF-ADB4-6C85480369C7",
    "desktop": "B4BFCC3A-DB2C-424C-B029-7FE99A87C641",
    "pictures": "33E28130-4E1E-4676-835A-98395C3BC3BB",
    "videos": "18989B1D-99B5-455B-841C-AB7C74E4DDFC",
    "music": "4BD8D571-6D19-48D3-BE97-422220080E43",
}
FOLDER_WORDS = {
    "downloads": "downloads", "download": "downloads",
    "documents": "documents", "document": "documents", "docs": "documents",
    "desktop": "desktop",
    "pictures": "pictures", "picture": "pictures", "photos": "pictures", "images": "pictures",
    "videos": "videos", "video": "videos", "movies": "videos",
    "music": "music", "songs": "music",
    "screenshots": "screenshots", "screenshot": "screenshots",
    "project": "project", "phantom claw": "project", "third hand": "project", "code": "project",
}


class _GUID(ctypes.Structure):
    _fields_ = [("d1", ctypes.c_uint32), ("d2", ctypes.c_uint16), ("d3", ctypes.c_uint16), ("d4", ctypes.c_ubyte * 8)]


def _known_folder(guid: str) -> Path | None:
    """Ask Windows where a user folder really is (Documents may live in OneDrive, for example)."""
    try:
        u = uuid.UUID(guid)
        g = _GUID(u.time_low, u.time_mid, u.time_hi_version, (ctypes.c_ubyte * 8)(*u.bytes[8:]))
        ptr = ctypes.c_wchar_p()
        shell32 = ctypes.windll.shell32
        if shell32.SHGetKnownFolderPath(ctypes.byref(g), 0, None, ctypes.byref(ptr)) != 0:
            return None
        path = ptr.value
        ctypes.windll.ole32.CoTaskMemFree(ptr)
        return Path(path) if path else None
    except (AttributeError, OSError, ValueError):
        return None


def folder_key(name: str) -> str | None:
    s = " ".join(w for w in _norm(name).split() if w not in _FILLER and w not in ("folder", "directory"))
    if s in FOLDER_WORDS:
        return FOLDER_WORDS[s]
    for word, key in FOLDER_WORDS.items():
        if f" {word} " in f" {s} ":
            return key
    return None


def folder_path(key: str) -> Path:
    if key == "project":
        return PROJECT_DIR
    if key == "screenshots":
        return folder_path("pictures") / "Screenshots"
    return _known_folder(_KNOWN[key]) or (Path.home() / key.capitalize())


async def open_folder(agent: str, name: str) -> dict:
    """Open one of the user folders (Downloads, Documents, Desktop, Pictures, Videos, Music,
    Screenshots) or the project folder in File Explorer."""
    key = folder_key(name)
    if key is None:
        return fail("open_folder", f"no folder called {name!r}", not_found=name)
    path = folder_path(key)
    if not path.is_dir():
        return fail("open_folder", f"{path} is not there", not_found=name)
    return await asyncio.to_thread(_launch, "open_folder", str(path), key, path=str(path))


# ---------------------------------------------------------------- files

# Never "open" anything that would run a program or install something.
BLOCKED_EXT = {
    ".exe", ".com", ".bat", ".cmd", ".ps1", ".psm1", ".vbs", ".vbe", ".js", ".jse", ".wsf", ".wsh", ".msc",
    ".msi", ".msix", ".msixbundle", ".msp", ".appx", ".appxbundle", ".appinstaller", ".reg", ".lnk", ".url",
    ".scr", ".pif", ".cpl", ".inf", ".hta", ".jar", ".py", ".pyw", ".sh", ".application", ".iso", ".img",
    ".vhd", ".vhdx", ".sys", ".dll", ".drv", ".appref-ms", ".settingcontent-ms", ".library-ms", ".gadget",
}
_SKIP_DIRS = {"node_modules", ".git", "__pycache__", ".venv", "venv", "site-packages", "$recycle.bin", "appdata"}
SEARCH_DEPTH = 3
SEARCH_MAX_FILES = 40000
_FILE_FILLER = _FILLER | {"file", "called", "named", "document", "that", "this", "latest", "last", "recent", "newest"}


def search_roots() -> list[Path]:
    return [folder_path(k) for k in ("downloads", "documents", "desktop")]


def find_file(query: str, roots: list[Path] | None = None) -> Path | None:
    """The most recently changed file whose name has every word of the query in it."""
    words = [w for w in _norm(query).split() if w not in _FILE_FILLER]
    if not words:
        return None
    roots = search_roots() if roots is None else roots
    best: tuple[float, Path] | None = None
    seen = 0
    for root in roots:
        if not root.is_dir():
            continue
        base_depth = len(root.parts)
        for dirpath, dirnames, files in os.walk(root):
            depth = len(Path(dirpath).parts) - base_depth
            dirnames[:] = [] if depth >= SEARCH_DEPTH else [
                d for d in dirnames if not d.startswith(".") and d.lower() not in _SKIP_DIRS]
            for f in files:
                seen += 1
                if seen > SEARCH_MAX_FILES:
                    return best[1] if best else None  # a huge tree; settle for what we have
                p = Path(dirpath) / f
                if p.suffix.lower() in BLOCKED_EXT or f.startswith(("~$", ".")) or re.search(r"env|secret|password|credential|token|\bkeys?\b", f.lower()):
                    continue
                name = f" {_norm(f)} "
                # each word starts a word in the name ("so101" also matches "so 101")
                if all(f" {w}" in name or (len(w) >= 4 and w in name.replace(" ", "")) for w in words):
                    try:
                        mt = p.stat().st_mtime
                    except OSError:
                        continue
                    if best is None or mt > best[0]:
                        best = (mt, p)
    return best[1] if best else None


async def open_file(agent: str, query: str, roots: list[Path] | None = None) -> dict:
    """Find a file by part of its name in Downloads, Documents and Desktop and open it with its
    usual program. The newest match wins. Programs and scripts are never opened."""
    p = await asyncio.to_thread(find_file, query, roots)
    if p is None:
        return fail("open_file", f"no file matching {query!r}", not_found=query)
    return await asyncio.to_thread(_launch, "open_file", str(p), p.name, path=str(p))


# ---------------------------------------------------------------- web, YouTube, Spotify


async def web_search(agent: str, query: str) -> dict:
    """Open the default browser on a web search."""
    url = "https://www.google.com/search?q=" + quote_plus(query)
    return await asyncio.to_thread(open_address, "web_search", url)


async def youtube(agent: str, query: str) -> dict:
    """Open a YouTube search in the browser."""
    url = "https://www.youtube.com/results?search_query=" + quote_plus(query)
    return await asyncio.to_thread(open_address, "youtube", url)


def _spotify(query: str) -> dict:
    if uri_registered("spotify:"):
        target = "spotify:search:" + quote(query)
        return _launch("play_music", target, "Spotify", via="app")
    return {**open_address("play_music", "https://open.spotify.com/search/" + quote(query)), "via": "web"}


async def play_music(agent: str, query: str) -> dict:
    """Open Spotify on a search for the music (the web player if the app isn't installed)."""
    return await asyncio.to_thread(_spotify, query)
