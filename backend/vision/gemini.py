"""Ask Gemini about one picture and get JSON back. Also the box parser that copes with the odd
shapes Gemini sometimes answers in. Taken from setup/tools/fetch_bottle.py.

The key is read from the environment (loaded from third-hand/.env) and sent in the
x-goog-api-key header. It is never printed or put in a URL.
"""
import base64
import json
import os
import urllib.request
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
# the script tried these in turn; the middle one is a fallback name for the same small model
MODELS = ("gemini-3.5-flash-lite", "gemini-flash-lite-latest", "gemini-3.5-flash-lite")


class GeminiError(RuntimeError):
    pass


def _key():
    k = (os.environ.get("GEMINI_API_KEY") or "").strip()
    if k:
        return k
    try:
        from dotenv import load_dotenv
        load_dotenv(PROJECT / ".env", override=False)
    except Exception:
        pass
    k = (os.environ.get("GEMINI_API_KEY") or "").strip()
    if not k:
        raise GeminiError("GEMINI_API_KEY missing")
    return k


def encode_jpg(img, size, quality=85):
    import cv2
    ok, jpg = cv2.imencode(".jpg", cv2.resize(img, size), [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise GeminiError("could not encode the picture")
    return base64.b64encode(jpg.tobytes()).decode()


CLAUDE_VISION = "claude-sonnet-5-5"


def ask_claude(img, prompt, size=(960, 540), quality=85, timeout=14.0):
    """The same question put to Claude, for when Gemini is busy. Returns the parsed JSON answer."""
    import re

    import anthropic
    try:
        from dotenv import load_dotenv
        load_dotenv(PROJECT / ".env", override=False)
    except Exception:
        pass
    k = (os.environ.get("ANTHROPIC_API_KEY") or "").strip()
    if not k:
        raise GeminiError("ANTHROPIC_API_KEY missing")
    client = anthropic.Anthropic(api_key=k, timeout=timeout, max_retries=0)
    msg = client.messages.create(
        model=CLAUDE_VISION, max_tokens=1500,
        output_config={"effort": "low"},
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                         "data": encode_jpg(img, size, quality)}},
            {"type": "text", "text": prompt + "\nReply with the JSON object only."}]}])
    text = "".join(b.text for b in msg.content if b.type == "text")
    # take the first complete JSON object in the reply, whatever is written around it
    i = text.find("{")
    if i < 0:
        raise GeminiError("Claude gave no JSON")
    obj, _end = json.JSONDecoder().raw_decode(text[i:])
    return obj


def ask_json(img, prompt, size=(960, 540), timeout=25, models=MODELS, quality=85):
    """Send one picture and a prompt; return the parsed JSON answer. Tries each model in turn."""
    body = {"contents": [{"parts": [{"inline_data": {"mime_type": "image/jpeg", "data": encode_jpg(img, size, quality)}},
                                    {"text": prompt}]}],
            "generationConfig": {"responseMimeType": "application/json"}}
    data = json.dumps(body).encode()
    key = _key()
    last = None
    # A long look (the desk scene): Gemini places things on the desk more accurately than Claude
    # in our checks, so it goes first, twice, each with a short limit so a busy Gemini cannot
    # hang the job. Claude is the last resort. Short looks keep their own timeout.
    long_look = timeout >= 10 and len(models) > 1
    for n, model in enumerate(models[:2] if long_look else models):
        try:
            req = urllib.request.Request(URL.format(model=model), data=data,
                                         headers={"x-goog-api-key": key, "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=(7 if long_look else timeout)) as r:
                return json.loads(json.loads(r.read())["candidates"][0]["content"]["parts"][0]["text"])
        except Exception as e:
            last = e
    if long_look:
        try:
            return ask_claude(img, prompt, size, quality)
        except Exception as e:  # noqa: BLE001
            last = e    # the error text from urllib never holds the key (it is only in a header)
    raise GeminiError(f"Gemini did not answer: {type(last).__name__}: {str(last)[:160]}")


def parse_box(v):
    """Gemini sometimes wraps the box in another list or a dict. Return four numbers or None."""
    while isinstance(v, dict):
        v = v.get("box_2d") or v.get("box") or next(iter(v.values()), None)
    while isinstance(v, list) and len(v) == 1 and isinstance(v[0], (list, dict)):
        v = v[0]
        while isinstance(v, dict):
            v = v.get("box_2d") or v.get("box") or next(iter(v.values()), None)
    if isinstance(v, list) and len(v) == 4 and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in v):
        return [float(x) for x in v]
    return None


def parse_boxes(raw, names=("bottle", "person")):
    """{"bottle": box or None, ...} from an answer that may be a dict or a one-item list."""
    if isinstance(raw, list) and raw and isinstance(raw[0], dict):
        raw = raw[0]
    if not isinstance(raw, dict):
        return {n: None for n in names}
    return {n: parse_box(raw.get(n)) for n in names}
